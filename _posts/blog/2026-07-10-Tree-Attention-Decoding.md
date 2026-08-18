---
layout: post
title: "树状注意力训练：Robostral Navigate 如何将 VLN 训练 Token 压缩 22×"
date:   2026-07-17
last_modified_at: 2026-07-29
tags: [VLA, VLN, Robostral, Prefix Caching, Tree Attention, LLM Training]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "聚焦 Robostral Navigate 的前缀树监督训练：为什么观测构成共享主干、动作必须成为叶节点，以及 tree attention 如何在阻止 ground-truth action 泄漏的同时将训练 Token 压缩 22×。"
---

# 1. 问题：导航历史为何让训练 Token 数二次增长？

Robostral Navigate 是 Mistral AI 提出的 8B 单目 RGB 导航 VLM。它根据自然语言指令与历史图像预测下一导航 waypoint；为了覆盖长轨迹中的全部监督信号，第 $t$ 步必须读取从 episode 开始到当前时刻的观测历史。

训练数据完全由仿真生成，约包含 35 万个场景和 240 万条轨迹。每个样本由自然语言指令、RGB 观测序列与导航动作组成，其中还包括跨楼层的长轨迹。数据规模越大、episode 越长，逐时间步展开造成的历史图像重复就越严重。

如果把每个时间步都构造成独立样本，相同的指令和早期图像会被反复编码：长度为 $T$ 的轨迹需要处理 $O(T^2)$ 个 Token。Robostral 的关键改动，是把整段 episode 恢复为一棵**以观测历史为主干、以监督动作为叶子**的前缀树，在一次 forward 中计算所有时间步的 loss。

论文报告，这种训练表示在不丢弃 action target 的前提下将训练 Token 数减少 **22×**，把原本以月计的训练缩短到数天。本文只聚焦这一训练机制；机器人迁移、在线强化学习和导航榜单不在讨论范围内。

> 本文依据 [Robostral Navigate（arXiv:2607.20785v2）](https://arxiv.org/abs/2607.20785v2) 更新。论文明确给出了前缀树结构与注意力语义，但没有公开完整训练代码、position ID 构造和 mask kernel；下文会区分论文事实与复现时的工程要求。

<!-- more -->

## 1.1 核心结论速览

| 问题 | Robostral 的处理 |
|---|---|
| 重复计算 | 指令和历史观测只在共享主干中编码一次 |
| 保留监督 | 每个时间步的 action target 都作为独立叶子保留 |
| 标签泄漏 | 后续观测不能读取先前的 ground-truth action |
| 训练 Token 规模 | 从逐时间步展开的 $O(T^2)$ 降到整 episode 的 $O(T)$ |
| 论文结果 | 训练 Token 数减少 22×；训练周期从月缩短到天 |

# 2. 前缀树监督训练：22× Token 压缩是怎么来的？

## 2.1 逐时间步训练的 Token 数为什么是 $O(T^2)$？

设指令为 $I$，第 $t$ 步观测为 $O_t$，监督动作是 $a_t$。部署时不存在 expert action history，因此第 $t$ 步应在下面的上下文中预测动作：

$$
[I,O_0,\ldots,O_t]\rightarrow a_t
$$

传统训练会构造 $T$ 个样本：

$$
[I,O_0]\rightarrow a_0
$$

$$
[I,O_0,O_1]\rightarrow a_1
$$

$$
\cdots
$$

$$
[I,O_0,\ldots,O_{T-1}]\rightarrow a_{T-1}
$$

总 Token 数近似为：

$$
N_{\text{naive}}
=\sum_{t=0}^{T-1}
\left(
|I|+\sum_{k=0}^{t}|O_k|+|a_t|
\right)
$$

早期观测 $O_0$ 会被编码 $T$ 次，$O_1$ 会被编码 $T-1$ 次。若每帧视觉 Token 数量近似固定，主导项随 $T^2$ 增长。

## 2.2 Robostral 的树：观测是主干，动作是叶子

<div align="center">
  <img src="/images/vln/Robostral-Navigate-prefix-tree.webp" width="96%" alt="Robostral Navigate 的逐时间步训练、前缀缓存与树状注意力掩码" />
  <figcaption>图 1：指令与观测构成共享主干，每个动作是独立叶节点。动作叶子只用于自身监督，后续观测不会再次读取它。</figcaption>
</div>

Robostral 将整个 episode 物理打包成：

$$
I\mid O_0\mid a_0\mid O_1\mid a_1\mid\cdots\mid O_{T-1}\mid a_{T-1}
$$

但它的**逻辑结构不是一条普通因果链**，而是：

$$
I\rightarrow O_0\rightarrow O_1\rightarrow\cdots\rightarrow O_{T-1}
$$

并在每个 $O_t$ 下挂一个动作叶子 $a_t$。于是唯一 Token 数变为：

$$
N_{\text{tree}}
=|I|+\sum_{t=0}^{T-1}\left(|O_t|+|a_t|\right)
=O(T)
$$

这也是论文所说的 prefix caching：相同的指令和历史观测只编码一次，却能在一次 forward 中保留所有时间步的动作监督。

这里的 **prefix caching 是训练期共享前缀计算，不是推理期 KV cache**。共享主干仍处于同一个 autograd graph 中，需要参与反向传播，并接收所有动作叶子汇总而来的梯度；如果将前缀 K/V 直接 `detach`，就会丢失这些训练信号。

## 2.3 为什么不能直接使用 causal mask？

在物理序列

$$
I,O_0,a_0,O_1,a_1,O_2,a_2
$$

中，普通 causal mask 会允许 $O_1$ 读取左侧的 ground-truth action $a_0$，也会让 $O_2$ 读取 $a_0$ 和 $a_1$。训练时这些动作高度提示下一个 waypoint，但部署时模型拿不到 expert action，因此会产生严重的 train–test mismatch。

Tree attention 改用祖先关系定义可见性。令 $node(i)$ 表示 Token $i$ 所属树节点，$Anc(n)$ 表示节点 $n$ 及其祖先，则可写为：

$$
M_{ij}=
\begin{cases}
0,& node(j)\in Anc(node(i))\ \text{且满足节点内因果顺序}\\
-\infty,& \text{otherwise}
\end{cases}
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}
\left(
\frac{QK^\top}{\sqrt d}+M
\right)V
$$

对应到 Robostral 的 episode tree：

- 查询 $O_t$ 只能读取 $I,O_0,\ldots,O_t$ 中位于它之前的 Token；
- 查询动作叶子 $a_t$ 可以读取 $I,O_0,\ldots,O_t$；
- 若 $a_t$ 由多个 Token 组成，它们可在同一叶子内部自回归；
- $O_{t+1}$ **不能**读取 $a_t$；
- $a_i$ 与 $a_j$ 位于不同叶子，彼此不可见。

因此，图中的动作是“leaves — never re-attended”：它们提供监督，但不会进入后续决策历史。

## 2.4 为什么一次 forward 仍与逐时间步监督等价？

每个动作叶子 $a_t$ 看到的上下文与独立样本

$$
[I,O_0,\ldots,O_t]\rightarrow a_t
$$

完全相同。所有 action target 都只出现一次，指令与观测主干则由各动作 loss 共同反向传播：

$$
L=\sum_{t=0}^{T-1}
L_{\text{action}}
\left(
a_t\mid I,O_{\le t}
\right)
$$

共享主干只执行一次 forward，但各叶子的梯度会在 autograd 图中自然汇总到共享表示。

这也澄清了一个重要区别：Robostral 论文**没有**引入通用 Tree Training 中的路径计数 loss weight、DFS 树分区或可微分 gateway。对于这里的“单条观测主干 + 每步一个动作叶子”，每个 action target 本来就只计算一次，不需要额外的路径计数加权。把其他树训练系统的组件直接当作 Robostral 已公开实现，会超出论文证据。

## 2.5 22× 该如何解读？

论文报告，相比逐时间步样本，这种表示在训练数据上将 Token 数减少 **22×**，同时不丢弃任何动作预测目标，并把原本以月计的训练缩短到数天。RxR-CE 的指令与轨迹尤其长，因此收益更明显。

22× 的准确含义是：

$$
\frac{N_{\text{naive}}}{N_{\text{tree}}}\approx22
$$

它不是“端到端训练吞吐严格提高 22×”。实际 wall-clock speedup 还取决于：

- 视觉编码器是否复用图像特征；
- attention mask 使用 dense、block-sparse 还是定制 kernel；
- packed sequence 长度与显存上限；
- activation checkpointing、通信与数据加载；
- VLM、视觉编码器和其他模块各自的耗时占比。

论文没有披露精确训练时长、GPU 数量、硬件利用率、position ID 构造或 mask kernel。因此可以确认的是 **需要处理的训练 Token 数从 $O(T^2)$ 降到 $O(T)$、实测 Token 减少 22×、训练周期从月缩短到天**，不能据此补写一个未经报告的端到端加速倍数。

# 3. 复现时最关键的正确性检查

论文给出了训练表示和 mask 语义，但没有公开完整代码与 kernel 细节。实现时至少应检查以下事项。

## 3.1 数据与标签

- 指令和 $O_0,\ldots,O_{T-1}$ 只在主干中出现一次；
- 每个 $a_t$ 是从 $O_t$ 分出的独立叶子；
- instruction 与 observation Token 的 label 设为 ignore；
- action Token 只在自己的叶子内计算 loss；
- 后续上下文中不能重新拼入 teacher-forced ground-truth action。

## 3.2 Mask 与逻辑位置

- 修改 $a_t$，$O_{t+1}$ 与其他动作叶子的 logits 不应变化；
- 修改 $O_t$，所有后续观测与对应动作 logits 应发生变化；
- 同一动作叶子内部保持 causal autoregression；
- attention 实现不能先计算所有非法 pair 再指望仅靠置零获得相同效率；
- 对使用 RoPE 的模型，逻辑 position 应与独立样本一致，不能让被 mask 的动作叶子平白推高后续主干位置。

最后一点是由“训练信号严格等价”推导出的工程要求；论文没有披露具体 position ID 实现。

## 3.3 最小等价性测试

在一段很短的 episode 上运行两种版本：

1. 每个时间步构造独立样本并分别 forward；
2. 整个 episode 一次 forward，使用 tree attention mask。

应逐项比较：

- 每个 action target 的 logits；
- 总 action loss；
- 关键参数 gradient；
- 一次 optimizer step 后的参数。

float32 小模型应接近数值精度；bf16 下可允许由 kernel 形状和累加顺序产生的小误差。性能测试则要分别报告 $N_{\text{naive}}/N_{\text{tree}}$、wall-clock throughput 和显存峰值，不能只用 Token 压缩比代替真实加速。

# 4. 总结

Robostral 的 Tree Attention Training 可以概括为四步：

1. 把逐时间步样本恢复为一个 episode 级前缀树；
2. 让指令与观测历史形成共享主干，每个 ground-truth action 成为独立叶子；
3. 用树状注意力阻断动作叶子与后续观测、其他动作叶子之间的信息流；
4. 在一次 forward 中计算全部 action loss，使需要处理的训练 Token 数从 $O(T^2)$ 降到 $O(T)$。

最容易被误解的是 22×：它表示训练 Token 的去重比例，而不是一个不受硬件、kernel 和模型结构影响的 GPU 加速倍数。真正有启发性的地方，是 Robostral 把历史前缀从“重复输入”恢复为“可微分共享结构”，既保留所有时间步的监督，又阻止部署时不可获得的 expert action 泄漏。

# 参考资料

1. Mistral AI. [**Robostral Navigate**](https://arxiv.org/abs/2607.20785v2). arXiv:2607.20785v2, 2026-07-24.
2. 本地论文文件：[2607.20785v2.pdf](/new_paper/2607.20785v2.pdf)。
3. Mistral AI. [**Robostral Navigate 官方介绍**](https://mistral.ai/news/robostral-navigate/), 2026.
