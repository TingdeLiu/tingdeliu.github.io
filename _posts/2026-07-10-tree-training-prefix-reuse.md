---
layout: post
title: "Tree Training 技术详解：Robostral Navigate 如何用树状注意力复用共享前缀"
date:   2026-07-10
tags: [VLA, VLN, LLM Training, Prefix Caching, Tree Attention, Robostral]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "从 Robostral Navigate 宣称的 22× 训练 token 压缩出发，系统推导训练期共享前缀复用、DFS 序列化、树状注意力掩码、加权损失、可微分分区边界及其在 VLA/VLN 长轨迹监督训练中的实现方式。"
---


# 1. Robostral Navigate 的 22× 到底是什么？

Mistral AI 在 Robostral Navigate 的技术介绍中提出：把一个完整导航 episode 压缩成单条 sequence，通过 **tree-based attention masking** 在一次 forward 中训练全部时间步，同时避免时间步之间的信息泄漏。官方给出的核心数字是：

- 相比“每个时间步构造一个独立样本”，**训练 token 数减少 22×**；
- 所有时间步的监督信号仍被保留；
- 原本可能持续数月的训练可缩短到数天。

这里最重要的限定词是 **training tokens**。22× 首先描述重复 token 的减少，不能直接写成“墙钟时间严格加速 22×”。真实速度还受 attention kernel、视觉编码器、通信、数据加载、序列长度和显存分区影响。

本文以本地论文 **Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse**（arXiv:2511.00413v5，2026-04-23）作为公开数学与工程参照。该论文来自快手团队，并非 Mistral 的技术报告；二者思想相关，但不能据此断言 Robostral 的内部实现完全相同。

<div align="center"><img src="/images/llm-training/tree-training/01-baseline-vs-tree.png" width="96%" alt="传统逐时间步训练与 Tree Training 对比" /><figcaption>图 1：传统训练重复处理共享历史；Tree Training 将唯一 token 打包后一次训练。</figcaption></div>

<!-- more -->

## 1.1 一句话结论

这项技术不是简单地“把所有帧拼起来”，而是同时完成：

1. 将共享前缀的样本组织成前缀树；
2. 用 DFS 序列化，使每个唯一 token 只出现一次；
3. 用树状 attention mask 阻断兄弟分支；
4. 用路径计数设置 loss 权重，保持梯度等价；
5. 整树放不进显存时，用可微分 gateway 传递 KV、状态和梯度。

## 1.2 对 Gemini 解读的事实核查

| 说法 | 判断 | 更准确的解释 |
|---|---:|---|
| 直接拼接会让早期动作看到未来帧 | 部分正确 | causal mask 能阻断右侧未来 token；DFS packing 的关键问题是阻断已经位于左侧、但属于兄弟分支的 token。 |
| 历史图像 KV 在显存中只存一份 | 方向正确但过度简化 | 唯一 token 的 K/V 和激活只计算一次，但训练还要保留反向传播所需张量，不能等同推理期只读 KV cache。 |
| 不同 episode 因场景相同而共享视觉前缀 | 缺乏来源支持 | Mistral 原文明确的是压缩一个 episode；Tree Training 论文也强调一个 global batch 内的单 rollout tree。 |
| 22× 是端到端训练加速 | 不准确 | Mistral 说的是训练 token 减少 22×；Tree Training 真实轨迹实验报告约 6.2×-6.3× 端到端加速。 |
| 显存严格从 O(T²) 降为 O(T) | 不够严谨 | 去重激活随唯一 token 数增长；attention 复杂度还取决于路径深度和稀疏 kernel。 |

# 2. 为什么逐时间步 VLA 训练会重复计算？

设导航 episode 有 $T$ 个决策时间步，第 $t$ 步监督样本为：

$$
x_t=[P,H_t,O_t], \qquad y_t=A_t
$$

- $P$：系统提示、导航指令、机器人能力描述等固定前缀；
- $H_t$：过去观测、动作、位姿、摘要或记忆；
- $O_t$：当前视觉、深度或状态观测；
- $A_t$：当前动作标签。

传统数据管线将每个时间步当成独立样本：

$$
[P,O_1]\rightarrow A_1
$$

$$
[P,O_1,A_1,O_2]\rightarrow A_2
$$

$$
[P,O_1,A_1,O_2,A_2,O_3]\rightarrow A_3
$$

假设固定前缀长度为 $p$，每步新增 $\Delta$ 个 token，则逐时间步展开的总 token 数为：

$$
N_{flat}=\sum_{t=1}^{T}(p+t\Delta)=Tp+\frac{\Delta T(T+1)}{2}
$$

唯一 token 数为：

$$
N_{unique}=p+T\Delta
$$

token 压缩比为：

$$
R_{token}=\frac{N_{flat}}{N_{unique}}
=\frac{Tp+\Delta T(T+1)/2}{p+T\Delta}
$$

当固定前缀较短且每步增量接近时，压缩比约为 $(T+1)/2$。因此，纯前缀链中 22× 大致对应四十多个时间步；真实 VLA 还包含不等长视觉 token、历史裁剪和提示模板，不能据此反推准确 episode 长度。

# 3. 从线性样本恢复为前缀树

## 3.1 为什么不是普通长序列？

如果每个上下文永远只是上一时刻追加新 token，那么可以在普通 causal sequence 的多个位置计算动作 loss。但真实 agent/VLA 数据常出现：

- 同一父状态产生多个候选动作或采样分支；
- think-mode 删除、替换或隐藏中间推理；
- 历史被截断、摘要或重新组织；
- 不同时间步共享指令和历史块，却附加独立的当前观测与 action query；
- 并行工具调用或 retokenization drift 改变后续上下文。

更一般的数据结构是前缀树。每个节点 $n$ 保存 token segment $$T_n$$，每条根到叶路径是一条完整训练序列。

<div align="center"><img src="/images/llm-training/tree-training/02-dfs-serialization.png" width="92%" alt="前缀树与 DFS 序列化" /><figcaption>图 2：前缀树按 DFS 序列化为 n0,n1,n3,n4,n2,n5，每个节点只出现一次。</figcaption></div>

## 3.2 基线与 DFS 序列化

设树有 $K$ 条根到叶路径。基线展开会为每个子树重复父节点：

$$
X_{base}(i)=\operatorname{Concat}[token(i),X_{base}(c_1),token(i),X_{base}(c_2),\ldots]
$$

Tree Training 改用 DFS：

$$
X_{DFS}(i)=\operatorname{Concat}[token(i),X_{DFS}(c_1),X_{DFS}(c_2),\ldots]
$$

每个节点只出现一次，但兄弟分支被放入同一物理 sequence，因此必须修正 attention、position id，以及混合 SSM 模型中的 recurrent state。

# 4. 核心一：Tree-based Attention Mask

## 4.1 标准 causal mask 为什么不够？

标准 causal mask 只检查物理位置：$i$ 可以读取所有 $j\le i$。在 DFS 顺序

$$
[n_0,n_1,n_3,n_4,n_2,n_5]
$$

中，$n_2$ 位于 $n_4$ 之后。普通 causal mask 会允许 $n_2$ 读取 $n_1,n_3,n_4$，但这些节点属于另一个兄弟分支。这是 DFS packing 中真正的信息泄漏。

## 4.2 可见性规则

令 $node(i)$ 表示 token $i$ 所属节点，$$Anc(n)$$ 表示节点及其祖先集合：

$$
M_{ij}=\begin{cases}
0,& j\le i\ \land\ node(j)\in Anc(node(i))\\
-\infty,& \text{otherwise}
\end{cases}
$$

$$
Attention(Q,K,V)=Softmax\left(\frac{QK^\top}{\sqrt d}+M\right)V
$$

每个 token 的可见上下文因此与独立根到叶路径运行完全一致。

<div align="center"><img src="/images/llm-training/tree-training/03-tree-attention-mask.png" width="72%" alt="Tree Attention Mask 矩阵" /><figcaption>图 3：蓝色单元表示可见；位于 DFS 左侧的兄弟分支也会被显式阻断。</figcaption></div>

## 4.3 生产实现不能构造巨大 dense mask

概念上 $M$ 是 $N\times N$ 矩阵，但长序列不能显式保存完整 mask。常见实现方式是：

- 在 FlashAttention kernel 内按节点祖先关系生成 block 可见性；
- 使用 block-sparse attention，只计算祖先路径对应 block；
- 预计算每个节点的 DFS 区间、parent、depth 和 ancestor block table；
- 小模型先用 PyTorch FlexAttention 的 `mask_mod` 验证，再换定制 kernel。

论文实现基于 FlashAttention V3 的节点级共享前缀 mask。若先做完整 dense attention 再把非法位置乘零，语义正确但很可能没有预期加速。

# 5. 核心二：位置编码按树深度，而非 DFS 下标

DFS 物理下标不等于 token 在独立路径中的逻辑位置。直接使用 packed sequence 下标会让后访问的兄弟分支获得过大的 RoPE position id。

若 token $t$ 位于节点 $n$ 内部偏移 $j$，正确位置为：

$$
pos(t)=\sum_{n'\in Anc(n)\setminus\{n\}}|T_{n'}|+j
$$

即位置由根到当前节点的累计长度决定。同深度、相同祖先长度的兄弟节点可以复用相同 position range，保证 RoPE 与逐路径基线一致。

# 6. 核心三：加权损失保证梯度等价

## 6.1 逐路径基线

设根到叶路径集合为 $$\mathcal P=\{p_1,\ldots,p_K\}$$：

$$
L_{sep-avg}=\frac{1}{K}\sum_{k=1}^{K}\sum_{t\in p_k}\ell_t(\theta)
$$

SFT 中：

$$
\ell_t(\theta)=-\log p_\theta(y_t\mid x_{\le t})
$$

policy-gradient RL 中也可令：

$$
\ell_t(\theta)=-A_t\log p_\theta(y_t\mid x_{\le t})
$$

## 6.2 交换求和顺序

令 $g_t$ 是经过 token $t$ 的根到叶路径数：

$$
\sum_{k=1}^{K}\sum_{t\in p_k}\ell_t
=\sum_t g_t\ell_t
$$

于是：

$$
L_{tree}=\sum_t\frac{g_t}{K}\ell_t(\theta)=L_{sep-avg}
$$

由微分线性性：

$$
\frac{\partial L_{tree}}{\partial\theta}
=\frac{\partial L_{sep-avg}}{\partial\theta}
$$

共享前缀只 forward 一次，但子分支的梯度会在共享祖先处自动相加。

<div align="center"><img src="/images/llm-training/tree-training/04-loss-equivalence.png" width="92%" alt="加权损失等价性" /><figcaption>图 4：n0 和 n1 按 3/3、2/3 加权，得到与三条路径独立训练后平均相同的目标。</figcaption></div>

## 6.3 VLA action loss 的细节

机器人模型通常只对 action token、waypoint token、坐标 token 或 stop token 计算 loss；指令、环境回传和视觉 token 的 label 设为 ignore index。此时 $$g_t/K$$ 只乘到有效监督位置。上下文 token 虽无直接 loss，仍通过后续 action loss 接收梯度。

# 7. 训练期 Prefix Cache 不等于推理期 KV Cache

推理时，过去 token 已确定，缓存 K/V 后只计算新 token，通常不需要对前缀反向传播。

训练时，共享前缀必须接收全部后续分支的梯度。如果直接把前缀 K/V 永久 `detach`，共享前缀会丢失学习信号。训练期复用的本质是：

- 同一次 packed forward 内，共享 token 的 K/V 和激活只创建一次；
- 同一 autograd graph 内，多个后代对共享表示的梯度自动相加；
- 跨显存分区时，可以临时 detach 成 gateway leaf，但随后必须把 leaf gradient 显式中继回父图。

因此，“训练期 prefix caching”更准确的名字是 **可微分的共享前缀计算复用**。

# 8. 整棵树放不进 GPU：Redundancy-Free Tree Partitioning

## 8.1 朴素切分仍会重算祖先

若唯一 token 数 $$N_{tree}$$ 超过容量 $C$，树必须切成多个 connected subtree。若每个子分区重新拼接祖先，边界会再次产生重复。

论文示例：

- 展平基线：164k token；
- 树唯一 token：83k；
- 普通树分区：102k；
- 可微分边界：仍为 83k。

## 8.2 Gateway forward 与 backward

在切分节点 $$n_c$$，父分区输出：

- 各层 ancestor K/V；
- ancestor-aware attention bias；
- 基于树深度的 position offset；
- 混合 SSM 模型所需 recurrent state 和 causal-conv context。

这些张量 `detach()` 后重新设置 `requires_grad=True`，作为子分区 gateway leaf。子分区直接读取，不重算祖先。子分区 backward 后，系统将 gateway leaf gradient 显式传回父分区原始张量，再沿父图继续反向。

多个子分区共享同一 cut node 时，论文使用 float32 accumulator hook 汇总梯度，减少多次 bfloat16 累加的舍入误差。

<div align="center"><img src="/images/llm-training/tree-training/05-differentiable-partition.png" width="92%" alt="可微分树分区边界" /><figcaption>图 5：父分区只前向一次；子分区读取 gateway，梯度再中继回父图。</figcaption></div>

## 8.3 为什么沿节点边界切分？

每个 partition 必须是连通子树，使 partition dependency graph 仍是一棵树，每个子分区只有一个父分区。这样反向峰值激活可约束在一条根到叶路径规模。随意混合无关子树可能让一个 partition 依赖多个父图，迫使多组祖先计算图同时驻留。

# 9. 混合 Attention + SSM 模型的额外修复

对 full-attention 与 Gated Delta Net 等 SSM 层交错的模型，仅有 attention mask 不够，因为 SSM 会按物理顺序传递 recurrent state。

## 9.1 State routing

DFS 为 $n_0,n_1,n_3,n_4,n_2,n_5$ 时，朴素顺序会把 $n_4$ 状态传给 $n_2$。正确做法是让 chunk $c$ 从父节点 $$\pi(c)$$ 读取初始状态：

$$
h_c^{(0)}=h_{\pi(c)}^{(end)}
$$

## 9.2 Causal convolution context

若 SSM 含 kernel size 为 $$K_{conv}$$ 的 causal conv1d，子节点应读取父节点保存的最后 $$K_{conv}-1$$ 个有效 token，而不是 DFS 地址上相邻的兄弟 token：

$$
y_c=Conv([ctx_{\pi(c)};x_c])[K_{conv}-1:]
$$

纯 Transformer VLA 不需要此修改；包含 Mamba、GDN 等 recurrent 模块时，这是保证 forward equivalence 的必要条件。

# 10. 正确理解 22× 与 POR

Tree Training 定义 Potential Overlap Ratio：

$$
POR=1-\frac{N_{tree}}{N_{flat}}
$$

完全消除重复时，理论 token/计算复用上限为：

$$
S_{upper}=\frac{N_{flat}}{N_{tree}}=\frac{1}{1-POR}
$$

22× token reduction 对应：

$$
POR=1-\frac{1}{22}\approx95.45\%
$$

即传统样本中约 95.45% 的 token 是共享前缀重复，不代表 GPU 所有模块同步提速 22×。

<div align="center"><img src="/images/llm-training/tree-training/06-speedup-interpretation.png" width="96%" alt="POR 与加速数字解释" /><figcaption>图 6：Mistral 的 22× token 减少与论文的端到端 speedup 是不同指标。</figcaption></div>

## 10.1 公开论文的实验数字

Tree Training 在 64 张 NVIDIA Hopper GPU、Megatron-Core 和 sequence packing 基线上测试：

| 场景 | 模型/数据 | 结果 |
|---|---|---:|
| 真实 agentic rollout | Qwen3-32B Dense | 平均 6.3× 端到端加速 |
| 真实 agentic rollout | Qwen3-30B MoE | 平均 6.2× 端到端加速 |
| 合成高 POR，整树可放显存 | Qwen3-8B/32B | 最高 8.7× |
| 额外 Tree Training 张量 | Qwen3-32B | 约 1.2 MB |
| Terminal Bench 2.0 RL | Qwen3-32B | full tree avg@4 28.8，最长路径基线 20.9 |

真实 rollout 的理论上限约 6.5×，实现达到 6.2×-6.3×，捕获超过 95% 的理论潜力。但这不能替代 Robostral 自身尚未公开的完整 profiling。

# 11. 映射到 Robostral Navigate：合理推断与未知项

Mistral 没有公开完整 tensor layout、mask kernel 和 loss weighting。根据“entire episode into a single sequence”“tree-based attention-masking”“all time steps in a single forward pass”，一个合理结构是：

## 11.1 Episode tree

根节点可以包含：

- system prompt 与机器人能力；
- 自然语言导航指令；
- 坐标系和 episode 固定元数据。

各时间步节点包含：

- 当前保留的历史表示；
- RGB、多视角、深度、位姿 token；
- action query 模板；
- 需要监督的动作 token。

上下文继承时合并共同 token segment；发生历史裁剪、摘要或模板变化时，从最后一个相同前缀处分叉。

## 11.2 视觉 token 的三层复用机会

1. 同一帧重复出现在历史中时，图像 encoder 输出可只计算一次；
2. multimodal projector 后的视觉 token 可成为共享树节点；
3. LLM 各层 K/V 与激活通过 tree attention 只创建一次。

官方页面明确的是训练序列和 attention 层面的压缩；是否进一步缓存视觉 backbone 特征，需要代码或技术报告确认。

## 11.3 必须阻止的泄漏

- 时间步 $t$ 的 action query 读取 $t+1$ 观测；
- 一个候选动作分支读取其他候选分支；
- teacher-forced action token 泄漏到其预测位置；
- DFS 左侧兄弟分支的视觉 token 被当前分支读取；
- history summary 读取部署时不可见的原始 hidden reasoning。

# 12. 可实现的数据结构与伪代码

```python
class TreeNode:
    token_ids: Tensor
    labels: Tensor             # 非监督位置为 -100
    parent: int
    children: list[int]
    depth_token_offset: int
    path_count: int            # g_n
```

DFS packing：

```python
packed_ids = concat(node.token_ids for node in dfs_nodes)

position_ids = concat(
    arange(node.depth_token_offset,
           node.depth_token_offset + len(node.token_ids))
    for node in dfs_nodes
)

loss_weights = concat(
    full(len(node.token_ids), node.path_count / num_paths)
    for node in dfs_nodes
)
```

mask 语义原型：

```python
def tree_mask(query_token, key_token):
    query_node = token_to_node[query_token]
    key_node = token_to_node[key_token]
    return key_token <= query_token and is_ancestor(key_node, query_node)
```

loss：

```python
token_loss = cross_entropy(
    logits[:, :-1], labels[:, 1:],
    reduction="none", ignore_index=-100,
)
loss = (token_loss * loss_weights[:, 1:] * valid_mask).sum()
```

生产实现必须避免逐 token `is_ancestor`，将节点级 block table 融入 attention kernel。

# 13. 复杂度与显存：不要只看 token 数

设 $D(t)$ 为 token $t$ 的祖先路径长度。线性层、MLP、归一化等计算主要从 $N_{flat}$ 降至 $N_{tree}$；tree attention 的有效 pair 数更接近：

$$
N_{pairs}=\sum_{t\in tree}D(t)
$$

而不是简单的 $$N_{tree}^2$$。真正的 block-sparse/tree-aware kernel 可以跳过兄弟分支 pair；dense attention 加 mask 则可能无法获得预期速度。

训练峰值显存仍包括 hidden activation、Q/K/V、MLP 中间量、vision encoder activation、optimizer state、通信 buffer 和 gateway 张量。Tree Training 消除共享 token 的重复激活，但不等于长上下文训练没有成本；仍需 activation checkpointing、context parallel、ZeRO/FSDP 与树分区配合。

# 14. 正确性验证清单

## 14.1 最小等价性测试

对一棵小树分别运行：

1. 每条根到叶路径独立 forward，loss 求平均；
2. DFS packed forward + tree mask + depth position + weighted loss。

比较每个监督 token 的 logits、总 loss、关键参数 gradient 和 optimizer step 后参数。float32 小模型应接近机器精度；bf16 大模型允许因 GEMM 形状和浮点加法顺序产生小误差。

## 14.2 泄漏测试

- 修改兄弟分支 token，当前分支 logits 不应变化；
- 修改祖先 token，所有后代 logits 应变化；
- 修改未来子节点，祖先和其他分支不应变化；
- 交换兄弟 DFS 顺序，逻辑 position 与路径 logits 应不变；
- action label 只影响 loss，不得错误进入同位置输入。

## 14.3 性能 profiling

记录 $N_{flat}$、$N_{tree}$、POR、forward/backward 时间、attention/MLP/vision encoder/通信占比、HBM 峰值、有效 attention block 比例，以及不同树深和分支因子的收益。

# 15. 何时最值得使用？

适合：

- episode 长且历史持续增长；
- 所有时间步都保留监督；
- 固定指令和早期视觉历史占比高；
- 同一 rollout 有候选分支或上下文重写；
- POR 高且 kernel 能利用结构化稀疏；
- 主要成本位于 LLM 主干。

收益较小：

- 每步上下文几乎完全不同；
- 模型只看当前帧；
- episode 很短或 POR 低；
- 使用 dense mask，非法 attention pair 仍完整计算；
- vision backbone 占主要时间且视觉特征未复用。

# 16. 总结

Robostral Navigate 的高效监督训练可以概括为：

> 把“许多共享前缀的时间步样本”恢复成原本的树结构，使每个唯一 token 只参与一次前向计算，同时通过树状可见性、逻辑位置、加权损失和可微分状态中继，保留逐样本训练的因果语义与梯度。

真正关键的不是“KV cache”这个词，而是训练计算图中的 **可微分共享**：

- DFS 解决唯一 token 的紧凑存储；
- tree attention mask 解决跨分支泄漏；
- depth-based position 解决 RoPE 等价性；
- path-count loss weight 解决目标与梯度等价性；
- differentiable gateway 解决超长树显存约束；
- SSM state routing 解决混合架构的状态污染。

Mistral 的 22× 表明导航数据具有极高前缀重复率。Tree Training 论文进一步证明，当这种复用落实到 forward、backward 和显存分区中时，理论 token 节省能够高比例转化为真实训练加速。

# 参考资料

1. Mistral AI. **Robostral Navigate**. `https://mistral.ai/news/robostral-navigate/`
2. Jinghui Wang et al. **Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse**. arXiv:2511.00413v5, 2026-04-23. `https://arxiv.org/abs/2511.00413`
3. 本地论文文件：`new_paper/2511.00413v5.pdf`
4. Kwon et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention**. SOSP 2023.
5. Shah et al. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**. 2024.
