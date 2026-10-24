---
layout: post
title: "Mixture-of-Transformers (MoT) 架构详解"
date: 2026-10-24
tags: [LLM Training, Multi-Modal, MoT, MoE, Sparse Transformer, Deep Learning, NVIDIA Cosmos]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "Mixture-of-Transformers（MoT）把 FFN、注意力投影和 LayerNorm 按模态拆开，只在全局自注意力处让各模态交汇。本文拆解它的计算流程，核对论文里 55.8%、37.2%、47.2% 这些数字各自指什么，并说明 NVIDIA Cosmos 3 的双塔结构借用了 MoT 的哪一部分、又改了哪一部分。"
---

把文本、图像、语音 token 放进同一个 Dense Transformer，是统一多模态模型最直接的做法：所有 token 共用同一套 $$W_Q, W_K, W_V$$、FFN 和 LayerNorm。它的代价在于，分布差异很大的模态要在同一组参数上争夺容量，训练也随之变慢。

Meta 与斯坦福的 Liang 等人在 *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*（arXiv:2411.04996，TMLR 2025）中给出的方案很朴素：**按模态复制所有非嵌入参数，只保留全局自注意力作为模态之间唯一的交汇点**。由于每个 token 属于哪个模态是已知的，路由不需要学习。

本文先拆解 MoT 的计算流程，再逐条核对论文实验数字的含义，最后看 NVIDIA Cosmos 3 的双塔结构与原始 MoT 的异同。

<!-- more -->

## 1. 问题：模态互相拖累

在 Chameleon、Transfusion 这类早期融合（early-fusion）的多模态模型中，不同模态的 token 化方式不同（文本用 BPE，图像用 VQ 离散码或连续 latent），但进入 Transformer 之后，它们经过的是**同一套**投影矩阵、FFN 与 LayerNorm。

直觉上的担忧有两点：

- **分布差异**：文本 token 离散、语义密集；图像 patch 冗余度高、空间相关性强。用同一组权重拟合两种统计结构，容量可能被互相挤占。
- **梯度干扰**：不同模态的损失在同一组参数上产生的更新方向未必一致，一个模态的进步可能以另一个模态为代价。

这两点是 MoT 的动机，论文并没有直接测量梯度冲突。它给出的间接证据是第 4 节的留一（leave-one-out）实验：只把某一个模态单独拆出、其余模态继续共享参数时，各模态受益的程度并不对称。作者称之为**非对称的模态竞争**（non-reciprocal modality competition），比如单独拆出语音时，语音的收益保住了，但图像与文本的收益随之消失。

## 2. MoT 做了什么

### 2.1 与 MoE 的区别

稀疏化最常见的做法是 Mixture-of-Experts（MoE）。MoT 与 MoE 有三点不同：

1. **拆分范围**：MoE（如 Mixtral）通常只把 FFN 换成多个专家，注意力投影与 LayerNorm 仍然共享；MoT 把 FFN、$$W_Q, W_K, W_V, W_O$$ 与 LayerNorm **全部**按模态复制。
2. **路由方式**：MoE 用可学习的 router 给 token 分配专家，需要负载均衡损失等手段防止专家闲置；MoT 直接按模态标签分流，没有 router，也就没有路由塌缩和负载均衡损失的问题。
3. **专家数量**：MoE 的专家数 $E$ 可以很大；MoT 的“专家”数等于模态数 $K$，通常只有 2–3 个。

### 2.2 计算量不变，参数量变多

这是理解后文实验数字的关键：**MoT 每个 token 的计算量与同宽度的 Dense 模型完全相同**。每个 token 只经过自己模态那一套参数，论文称之为与 Dense 模型 “exact same computational structure and FLOP count”。变多的是总参数量，约为 Dense 的 $K$ 倍（嵌入层除外）。

因此，MoT 节省的不是单步计算，而是**达到同等损失所需的训练步数**。由于每步 FLOPs 相同，论文把“步数占比”直接写成“FLOPs 占比”。

<div align="center">
  <img src="/images/llm-training/mixture-of-transformers/mot_architecture.webp" width="90%" loading="lazy" decoding="async" style="aspect-ratio:1024/1024" />
<figcaption>图 1：Dense、MoE 与 MoT 的结构对比（示意）。MoT 拆分了全部非嵌入参数，只共享注意力计算本身</figcaption>
</div>

## 3. 一个 MoT 层的计算流程

### 3.1 符号

设输入序列 $X = [x_1, \dots, x_N] \in \mathbb{R}^{N \times d}$，每个 token $x_i$ 带一个模态标签 $m_i \in \mathcal{M}$，例如 $\mathcal{M} = \{\text{text}, \text{image}, \text{speech}\}$。

对每个模态 $m$，MoT 维护一套专属参数：

* 两个 LayerNorm：$$\text{LN}_m$$ 与 $$\text{LN}'_m$$
* 注意力投影：$$\mathbf{W}_{Q,m}, \mathbf{W}_{K,m}, \mathbf{W}_{V,m}, \mathbf{W}_{O,m} \in \mathbb{R}^{d \times d}$$
* 前馈网络：$$\text{FFN}_m$$

### 3.2 四个步骤

论文 Algorithm 1 的流程是：按模态分组 → 各自投影 → 恢复原序列顺序 → 全局注意力 → 再按模态分组做输出投影、LayerNorm 和 FFN。写成公式如下。

**第一步：按模态投影出 Q、K、V。** 每个 token 用自己模态的 LayerNorm 与投影矩阵：

$$
\mathbf{q}_i = \text{LN}_{m_i}(x_i)\,\mathbf{W}_{Q,m_i},\quad
\mathbf{k}_i = \text{LN}_{m_i}(x_i)\,\mathbf{W}_{K,m_i},\quad
\mathbf{v}_i = \text{LN}_{m_i}(x_i)\,\mathbf{W}_{V,m_i}
$$

**第二步：全局自注意力。** 把所有 token 的 $$\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i$$ 按原序列顺序拼回 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$，做一次标准（多头）注意力，因果掩码等与 Dense 模型一致：

$$
\mathbf{H} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

这是各模态唯一交换信息的地方：投影空间是各自的，但文本 token 能直接注意到图像 token，反之亦然。

**第三步：按模态做输出投影与残差。**

$$
y_i = x_i + \mathbf{h}_i\,\mathbf{W}_{O,m_i}
$$

**第四步：按模态做 FFN。**

$$
z_i = y_i + \text{FFN}_{m_i}\big(\text{LN}'_{m_i}(y_i)\big)
$$

$Z = [z_1, \dots, z_N]$ 即该层输出。可以看出，如果把所有 $m_i$ 都设成同一个模态，这一层就退化回普通的 Pre-LN Transformer 层，这也是 MoT 每 token FLOPs 与 Dense 相同的原因。

## 4. Dense、MoE 与 MoT 对比

| 维度 | Dense | MoE | MoT |
| :--- | :--- | :--- | :--- |
| 注意力投影（QKV/O） | 共享 | 共享 | **按模态拆分** |
| LayerNorm | 共享 | 共享 | **按模态拆分** |
| FFN | 共享 | 稀疏专家 | **按模态拆分** |
| 路由 | 无 | 可学习 router | **按模态标签，确定性** |
| 每 token 计算量 | 基准 | 取决于激活专家数（top-1 时接近 Dense） | **与 Dense 相同** |
| 总参数（非嵌入） | 基准 | 约 $E$ 倍（FFN 部分） | 约 $K$ 倍（全部） |
| 额外训练机制 | 无 | 负载均衡损失等 | 无 |

论文还从通信角度比较了两者：每层新增参数，MoT 约为 $7(K-1)D^2$，MoE 约为 $3(E-1)D^2$ 以上。由于 $K$ 通常远小于 $E$，MoT 的“参数/FLOPs”比更低，分布式训练时的通信压力也更小。

## 5. 实验：每个数字各指什么

论文在三种设置下评测：Chameleon（文本与图像都做自回归）、Chameleon + 语音，以及 Transfusion（文本自回归、图像扩散）。下面逐条说明主要数字的含义。

### 5.1 达到同等损失的 FLOPs

* **Chameleon，7B，文本 + 图像**：MoT 用 Dense 模型 **55.8%** 的训练步数（等价于训练 FLOPs）就达到了 Dense 训练结束时的验证损失。
* **Chameleon + 语音，7B**：在语音模态上，MoT 用 **37.2%** 的 FLOPs 达到 Dense 的语音性能。语音实验覆盖 443M、880M、1.5B 和 7B 四个规模，37.2% 是 7B 的结果。
* **Transfusion，7B**：在图像指标上，MoT 用约 **三分之一** 的 FLOPs 达到 Dense 的水平；此外，760M 的 MoT 在关键图像生成指标上超过了 1.4B 的 Dense 模型。

需要注意的是：这些都是**训练效率**的数字，不是推理时“只激活了 55.8% 的参数”。另外，Transfusion 设置下文本模态的收益很小甚至没有，作者列出了几种可能的原因，但没有定论。

### 5.2 墙钟时间

FLOPs 节省不一定能换成实际的训练时间节省。论文在 AWS p4de.24xlarge（A100）上用 256 张卡做了 Chameleon 设置的测速：

* 图像：MoT 用 Dense **47.2%** 的 GPU 时间达到 Dense 的图像训练损失；
* 文本：用 **75.6%** 的时间达到 Dense 的文本损失；
* 对照组 MoE-4x 在文本上没有加速，在图像上反而**慢了 1.7 倍**。

实现方式就是第 3 节的“按模态分组 → 投影 → 还原顺序”，配合 FSDP 与 PyTorch 2 编译，没有用到专门的融合算子。作者提到的主要开销来自两处：按模态分组带来的 CPU-GPU 同步，以及模态在 batch 和 GPU 之间分布不均造成的负载不平衡。这些开销在他们的设置下不在关键路径上；grouped GEMM、块稀疏矩阵乘等优化留作后续工作。

### 5.3 拆哪些参数最重要

论文的消融（Section 3.5）显示：

* 只拆 FFN 就能带来大部分收益，图像上尤其明显；
* 再拆 Q/K/V，还有显著的额外收益：在 Obelisc 数据上，相比只拆 FFN，图像节省约 33.3% 的 FLOPs，文本约 10%；
* 拆 LayerNorm 的影响可以忽略。

### 5.4 与 MoE-4x 的比较

MoE-4x 的稀疏参数比 MoT 更多，但整体表现不如 MoT，在图像和语音上差距更明显：其图像优势在 7B 规模消失，语音验证损失甚至不如 Dense。作者也提醒，评测中 Expert Choice 路由可能泄露未来 token 的信息，MoE-4x 的结果因此可能被高估，换到分布外数据时又可能被低估。

论文第 5 节还尝试在 MoT 的文本分支里再套一层 MoE-4x，文本有初步提升且不损害图像，作者明确称之为早期概念验证。

## 6. Cosmos 3：按角色拆分

NVIDIA 在 2026 年 6 月发布的 Cosmos 3（arXiv:2606.02800）用一个模型同时覆盖理解、生成与动作，其骨干是**双塔 MoT**。但它拆分参数的依据和原始 MoT 不同，这一点容易被忽略。

<div align="center">
  <img src="/images/wm/Cosmos3-Fig5-MoTArchitecture.webp" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1553/701" />
<figcaption>图：Cosmos 3 的 MoT 架构。同一条序列由 AR 子序列（语言 + ViT 视觉 token，以 EOS/BOG 收尾）与 DM 子序列（VAE 视觉、音频、动作 token，训练时加噪）拼接而成；层内 AR 与 DM token 各用独立 LayerNorm 与 MLP（均由预训练 VLM 共同初始化），仅在共享自注意力处交汇。右图为注意力掩码：AR 为因果三角、DM 为全注意力。（图源：Cosmos 3）</figcaption>
</div>

### 6.1 两座塔分别是什么

序列被切成两段：

*   **AR 子序列（Reasoner 塔）**：语言 token 与 ViT 视觉 token，用因果注意力做下一个 token 预测，承担理解与推理；
*   **DM 子序列（Generator 塔）**：VAE 视觉 latent、音频和动作 token，训练时加噪，用 Flow Matching 去噪，承担生成。

每个解码层并行持有两套参数，二者都从预训练 VLM（Nano 与 Super 分别来自 Qwen3-VL 8B 与 32B）初始化。两段在注意力上的规则是：

*   AR token 只看自身之前的 AR token；
*   DM token 以 AR 与 DM token 的并集作为 Key/Value，双向注意；
*   AR token 永远不会被 DM token 更新，生成过程不会反过来改写理解通路。

### 6.2 与原始 MoT 的异同

**相同**的是核心机制：参数按 token 类别拆开、分流规则固定、所有 token 在共享的自注意力里交汇。

**不同**的有两点：

1. **分流依据**：原始 MoT 按**模态**分流，一个模态一套参数；Cosmos 3 按**角色**分流，同为视觉信息，ViT token 进 Reasoner、VAE latent 进 Generator。
2. **注意力掩码**：原始 MoT 的掩码与 Dense 模型一致；Cosmos 3 的 AR 段用因果掩码、DM 段用全注意力，并且是单向的：DM 能读 AR，AR 不能读 DM。

### 6.3 规模

双塔意味着总参数约为同级 Dense 骨干的 2 倍：

| 变体 | 总参 / Dense 骨干 | 状态 |
| :--- | :--- | :--- |
| Cosmos3-Edge | 4B / 2B | 后续发布 |
| Cosmos3-Nano | 16B / 8B | 已发布 |
| Cosmos3-Super | 64B / 32B | 已发布 |

位置编码、动作表示、三种动作模式（前向动力学、逆动力学、策略）和各项评测结果，详见 [世界模型综述 §7.3 Cosmos 3](/World-Models-Survey/#cosmos-3)。

## 7. 互动沙盘

下面的沙盘演示 token 在 Dense、MoE、MoT 中分别经过哪些参数。它只是示意：MoE 一栏为了简化画成“每个模态一个 FFN 专家”，真实 MoE 的专家并不对应模态，而是由 router 动态分配。

<div id="mot-playground" style="margin: 20px 0; padding: 20px; background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 100%); border-radius: 12px; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);">
  <h3 style="margin-top: 0; color: #38bdf8; text-align: center; font-size: 1.4rem; font-weight: bold;">多模态 Token 分流示意</h3>
  <p style="font-size: 0.9rem; color: #94a3b8; text-align: center; margin-bottom: 20px;">
    切换架构模式，点击“开始分流”观察 Token 在各层权重（LN、QKV、Attention、FFN）中的流动轨迹。
  </p>

  <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 20px;">
    <button onclick="setPlaygroundMode('dense')" id="btn-dense" style="padding: 8px 16px; border: 1px solid #334155; background: #1e293b; color: #94a3b8; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.3s;">Dense (密集共享)</button>
    <button onclick="setPlaygroundMode('moe')" id="btn-moe" style="padding: 8px 16px; border: 1px solid #334155; background: #1e293b; color: #94a3b8; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.3s;">MoE (仅 FFN 专家)</button>
    <button onclick="setPlaygroundMode('mot')" id="btn-mot" style="padding: 8px 16px; border: 1px solid #334155; background: #1e293b; color: #94a3b8; border-radius: 6px; cursor: pointer; font-weight: bold; transition: all 0.3s;">MoT (全模块解耦)</button>
  </div>

  <div style="position: relative; border: 1px solid #334155; background: #020617; border-radius: 8px; height: 350px; overflow: hidden; margin-bottom: 20px;">
    <div id="layer-container" style="display: flex; justify-content: space-around; align-items: center; height: 100%; width: 100%; padding: 0 10px; box-sizing: border-box; position: absolute; z-index: 1;">
      <div class="sim-phase" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <span style="font-size: 0.75rem; color: #64748b; font-weight: bold;">输入 Tokens</span>
        <div id="input-zone" style="width: 80px; height: 200px; border: 1px dashed #334155; border-radius: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 8px; background: rgba(15, 23, 42, 0.5);"></div>
      </div>
      <div class="sim-phase" style="display: flex; flex-direction: column; align-items: center; gap: 10px; position: relative;">
        <span style="font-size: 0.75rem; color: #64748b; font-weight: bold;">LayerNorm & QKV</span>
        <div id="qkv-zone" style="width: 100px; height: 220px; display: flex; flex-direction: column; justify-content: space-around; align-items: center; position: relative;"></div>
      </div>
      <div class="sim-phase" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <span style="font-size: 0.75rem; color: #64748b; font-weight: bold;">全局自注意力</span>
        <div id="attn-zone" style="width: 100px; height: 200px; border: 2px solid #334155; border-radius: 8px; display: flex; justify-content: center; align-items: center; background: rgba(30, 27, 75, 0.4); font-size: 0.8rem; font-weight: bold; color: #c084fc; transition: all 0.3s; text-align: center;">Global Attention</div>
      </div>
      <div class="sim-phase" style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
        <span style="font-size: 0.75rem; color: #64748b; font-weight: bold;">FFN (前馈专家)</span>
        <div id="ffn-zone" style="width: 100px; height: 220px; display: flex; flex-direction: column; justify-content: space-around; align-items: center; position: relative;"></div>
      </div>
    </div>
    <div id="token-canvas" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 2; pointer-events: none;"></div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; background: rgba(15, 23, 42, 0.6); padding: 15px; border-radius: 8px; border: 1px solid #334155;">
    <div style="text-align: center;">
      <div style="font-size: 0.8rem; color: #94a3b8;">每 token 计算量</div>
      <div id="stat-flops" style="font-size: 1.3rem; font-weight: bold; color: #ef4444; margin-top: 5px;">基准</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 0.8rem; color: #94a3b8;">共享的参数</div>
      <div id="stat-conflict" style="font-size: 1.3rem; font-weight: bold; color: #ef4444; margin-top: 5px;">全部</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 0.8rem; color: #94a3b8;">分流路由机制</div>
      <div id="stat-routing" style="font-size: 1rem; font-weight: bold; color: #38bdf8; margin-top: 8px;">无 (全共享)</div>
    </div>
  </div>

  <div style="display: flex; justify-content: center; gap: 15px; margin-top: 20px;">
    <button onclick="startPlaygroundSimulation()" id="btn-start" style="padding: 10px 24px; background: #38bdf8; color: #0f172a; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 1rem; transition: background 0.3s; box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2);">开始分流</button>
    <button onclick="resetPlayground()" style="padding: 10px 20px; background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 0.95rem; transition: all 0.3s;">重置</button>
  </div>
</div>

<script>
  (function() {
    let mode = 'dense';
    let isSimulating = false;
    let animFrames = [];

    const tokenTypes = [
      { name: '文本 (Text)', color: '#38bdf8', glow: 'rgba(56, 189, 248, 0.4)', type: 'text' },
      { name: '图像 (Vision)', color: '#c084fc', glow: 'rgba(192, 132, 252, 0.4)', type: 'vision' },
      { name: '语音 (Speech)', color: '#fbbf24', glow: 'rgba(251, 191, 36, 0.4)', type: 'speech' }
    ];

    window.setPlaygroundMode = function(newMode) {
      if (isSimulating) return;
      mode = newMode;
      
      ['dense', 'moe', 'mot'].forEach(m => {
        const btn = document.getElementById('btn-' + m);
        if (m === mode) {
          btn.style.background = '#38bdf8';
          btn.style.color = '#0f172a';
          btn.style.borderColor = '#38bdf8';
        } else {
          btn.style.background = '#1e293b';
          btn.style.color = '#94a3b8';
          btn.style.borderColor = '#334155';
        }
      });

      const flops = document.getElementById('stat-flops');
      const conflict = document.getElementById('stat-conflict');
      const routing = document.getElementById('stat-routing');

      if (mode === 'dense') {
        flops.innerText = '基准';
        flops.style.color = '#ef4444';
        conflict.innerText = '全部';
        conflict.style.color = '#ef4444';
        routing.innerText = '无 (全局共享)';
      } else if (mode === 'moe') {
        flops.innerText = '≈ Dense（top-1）';
        flops.style.color = '#fbbf24';
        conflict.innerText = 'LN + 注意力投影';
        conflict.style.color = '#fbbf24';
        routing.innerText = '可学习动态路由';
      } else if (mode === 'mot') {
        flops.innerText = '= Dense';
        flops.style.color = '#10b981';
        conflict.innerText = '仅注意力计算';
        conflict.style.color = '#10b981';
        routing.innerText = '确定性模态路由';
      }

      resetPlayground();
    };

    window.resetPlayground = function() {
      animFrames.forEach(cancelAnimationFrame);
      animFrames = [];
      isSimulating = false;
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-start').style.opacity = '1';

      document.getElementById('token-canvas').innerHTML = '';
      const inputZone = document.getElementById('input-zone');
      const qkvZone = document.getElementById('qkv-zone');
      const ffnZone = document.getElementById('ffn-zone');

      inputZone.innerHTML = '';
      qkvZone.innerHTML = '';
      ffnZone.innerHTML = '';

      const activeTokens = [
        tokenTypes[0], // text
        tokenTypes[1], // vision
        tokenTypes[0], // text
        tokenTypes[2]  // speech
      ];

      activeTokens.forEach((t, index) => {
        const pill = document.createElement('div');
        pill.className = 'sim-token';
        pill.style.cssText = `
          width: 60px; padding: 4px; border-radius: 4px; text-align: center;
          font-size: 0.7rem; font-weight: bold; background: ${t.color}; color: #0f172a;
          box-shadow: 0 0 10px ${t.glow}; transition: all 0.5s; opacity: 0.9;
        `;
        pill.innerText = t.type.toUpperCase();
        pill.setAttribute('data-type', t.type);
        inputZone.appendChild(pill);
      });

      if (mode === 'dense') {
        qkvZone.innerHTML = `<div style="width: 90px; height: 160px; border: 2px solid #475569; border-radius: 6px; background: rgba(51, 65, 85, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; text-align: center; font-weight: bold; color: #94a3b8; transition: all 0.3s;">共享 LN+QKV</div>`;
        ffnZone.innerHTML = `<div style="width: 90px; height: 160px; border: 2px solid #475569; border-radius: 6px; background: rgba(51, 65, 85, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; text-align: center; font-weight: bold; color: #94a3b8; transition: all 0.3s;">共享 FFN</div>`;
      } else if (mode === 'moe') {
        qkvZone.innerHTML = `<div style="width: 90px; height: 160px; border: 2px solid #475569; border-radius: 6px; background: rgba(51, 65, 85, 0.3); display: flex; align-items: center; justify-content: center; font-size: 0.75rem; text-align: center; font-weight: bold; color: #94a3b8; transition: all 0.3s;">共享 LN+QKV</div>`;
        ffnZone.innerHTML = `
          <div style="width: 90px; height: 50px; border: 2px solid #475569; border-radius: 6px; background: rgba(56, 189, 248, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #38bdf8; transition: all 0.3s;">FFN 文本专家</div>
          <div style="width: 90px; height: 50px; border: 2px solid #475569; border-radius: 6px; background: rgba(192, 132, 252, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #c084fc; transition: all 0.3s;">FFN 视觉专家</div>
          <div style="width: 90px; height: 50px; border: 2px solid #475569; border-radius: 6px; background: rgba(251, 191, 36, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #fbbf24; transition: all 0.3s;">FFN 语音专家</div>
        `;
      } else if (mode === 'mot') {
        qkvZone.innerHTML = `
          <div style="width: 90px; height: 50px; border: 2px dashed #38bdf8; border-radius: 6px; background: rgba(56, 189, 248, 0.15); display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: bold; color: #38bdf8; transition: all 0.3s;">LN/QKV<br>文本专家</div>
          <div style="width: 90px; height: 50px; border: 2px dashed #c084fc; border-radius: 6px; background: rgba(192, 132, 252, 0.15); display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: bold; color: #c084fc; transition: all 0.3s;">LN/QKV<br>视觉专家</div>
          <div style="width: 90px; height: 50px; border: 2px dashed #fbbf24; border-radius: 6px; background: rgba(251, 191, 36, 0.15); display: flex; flex-direction: column; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: bold; color: #fbbf24; transition: all 0.3s;">LN/QKV<br>语音专家</div>
        `;
        ffnZone.innerHTML = `
          <div style="width: 90px; height: 50px; border: 2px solid #38bdf8; border-radius: 6px; background: rgba(56, 189, 248, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #38bdf8; transition: all 0.3s;">FFN 文本专家</div>
          <div style="width: 90px; height: 50px; border: 2px solid #c084fc; border-radius: 6px; background: rgba(192, 132, 252, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #c084fc; transition: all 0.3s;">FFN 视觉专家</div>
          <div style="width: 90px; height: 50px; border: 2px solid #fbbf24; border-radius: 6px; background: rgba(251, 191, 36, 0.15); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; color: #fbbf24; transition: all 0.3s;">FFN 语音专家</div>
        `;
      }
    };

    window.startPlaygroundSimulation = function() {
      if (isSimulating) return;
      isSimulating = true;
      document.getElementById('btn-start').disabled = true;
      document.getElementById('btn-start').style.opacity = '0.5';

      const canvas = document.getElementById('token-canvas');
      const inputZone = document.getElementById('input-zone');
      const qkvZone = document.getElementById('qkv-zone');
      const attnZone = document.getElementById('attn-zone');
      const ffnZone = document.getElementById('ffn-zone');

      const canvasRect = canvas.getBoundingClientRect();
      const tokens = Array.from(inputZone.getElementsByClassName('sim-token'));
      const qkvExperts = Array.from(qkvZone.children);
      const ffnExperts = Array.from(ffnZone.children);

      const entities = tokens.map((token, index) => {
        const type = token.getAttribute('data-type');
        const color = type === 'text' ? '#38bdf8' : type === 'vision' ? '#c084fc' : '#fbbf24';
        const glow = type === 'text' ? 'rgba(56, 189, 248, 0.4)' : type === 'vision' ? 'rgba(192, 132, 252, 0.4)' : 'rgba(251, 191, 36, 0.4)';
        
        const dot = document.createElement('div');
        dot.style.cssText = `
          position: absolute; width: 14px; height: 14px; border-radius: 50%;
          background: ${color}; box-shadow: 0 0 12px 3px ${color}, 0 0 4px ${glow};
          transform: translate(-50%, -50%); transition: opacity 0.3s;
        `;
        canvas.appendChild(dot);

        token.style.opacity = '0.1';

        const startRect = token.getBoundingClientRect();
        
        let qkvTargetY = 0;
        if (mode === 'dense' || mode === 'moe') {
          const targetRect = qkvExperts[0].getBoundingClientRect();
          qkvTargetY = targetRect.top + targetRect.height / 2 - (index - 1.5) * 20;
        } else {
          const expertIdx = type === 'text' ? 0 : type === 'vision' ? 1 : 2;
          const targetRect = qkvExperts[expertIdx].getBoundingClientRect();
          qkvTargetY = targetRect.top + targetRect.height / 2 + (index % 2 - 0.5) * 8;
        }
        const qkvTargetRect = qkvZone.getBoundingClientRect();
        const qkvTargetX = qkvTargetRect.left + qkvTargetRect.width / 2;

        const attnRect = attnZone.getBoundingClientRect();
        const attnTargetX = attnRect.left + attnRect.width / 2;
        const attnTargetY = attnRect.top + attnRect.height / 2 - (index - 1.5) * 20;

        let ffnTargetY = 0;
        if (mode === 'dense') {
          const targetRect = ffnExperts[0].getBoundingClientRect();
          ffnTargetY = targetRect.top + targetRect.height / 2 - (index - 1.5) * 20;
        } else {
          const expertIdx = type === 'text' ? 0 : type === 'vision' ? 1 : 2;
          const targetRect = ffnExperts[expertIdx].getBoundingClientRect();
          ffnTargetY = targetRect.top + targetRect.height / 2;
        }
        const ffnTargetRect = ffnZone.getBoundingClientRect();
        const ffnTargetX = ffnTargetRect.left + ffnTargetRect.width / 2;

        const outTargetX = canvasRect.right - 20;
        const outTargetY = startRect.top + startRect.height / 2;

        return {
          dom: dot,
          type,
          color,
          path: [
            { x: startRect.left + startRect.width / 2, y: startRect.top + startRect.height / 2 },
            { x: qkvTargetX, y: qkvTargetY },
            { x: attnTargetX, y: attnTargetY },
            { x: ffnTargetX, y: ffnTargetY },
            { x: outTargetX, y: outTargetY }
          ]
        };
      });

      let start = null;
      function step(timestamp) {
        if (!start) start = timestamp;
        const progress = (timestamp - start) / 4500;
        
        entities.forEach(ent => {
          const segmentCount = ent.path.length - 1;
          const totalT = Math.min(Math.max(progress, 0), 0.999) * segmentCount;
          const segmentIdx = Math.floor(totalT);
          const segmentT = totalT - segmentIdx;

          const p0 = ent.path[segmentIdx];
          const p1 = ent.path[segmentIdx + 1];

          const currentX = p0.x + (p1.x - p0.x) * segmentT - canvasRect.left;
          const currentY = p0.y + (p1.y - p0.y) * segmentT - canvasRect.top;

          ent.dom.style.left = currentX + 'px';
          ent.dom.style.top = currentY + 'px';

          if (segmentIdx === 1) {
            if (mode !== 'mot') {
              qkvZone.firstElementChild.style.background = 'rgba(239, 68, 68, 0.2)';
              qkvZone.firstElementChild.style.borderColor = '#ef4444';
              qkvZone.firstElementChild.style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.4)';
            } else {
              const expertIdx = ent.type === 'text' ? 0 : ent.type === 'vision' ? 1 : 2;
              qkvExperts[expertIdx].style.background = ent.color + '33';
              qkvExperts[expertIdx].style.boxShadow = '0 0 15px ' + ent.color + '66';
            }
          }

          if (segmentIdx === 2) {
            attnZone.style.boxShadow = '0 0 20px rgba(192, 132, 252, 0.6)';
            attnZone.style.borderColor = '#c084fc';
            attnZone.style.background = 'rgba(192, 132, 252, 0.2)';
            
            if (mode !== 'mot') {
              qkvZone.firstElementChild.style.background = 'rgba(51, 65, 85, 0.3)';
              qkvZone.firstElementChild.style.borderColor = '#475569';
              qkvZone.firstElementChild.style.boxShadow = 'none';
            } else {
              qkvExperts.forEach(ex => {
                ex.style.background = 'transparent';
                ex.style.boxShadow = 'none';
              });
            }
          }

          if (segmentIdx === 3) {
            attnZone.style.boxShadow = 'none';
            attnZone.style.borderColor = '#334155';
            attnZone.style.background = 'rgba(30, 27, 75, 0.4)';

            if (mode === 'dense') {
              ffnZone.firstElementChild.style.background = 'rgba(239, 68, 68, 0.2)';
              ffnZone.firstElementChild.style.borderColor = '#ef4444';
              ffnZone.firstElementChild.style.boxShadow = '0 0 15px rgba(239, 68, 68, 0.4)';
            } else {
              const expertIdx = ent.type === 'text' ? 0 : ent.type === 'vision' ? 1 : 2;
              ffnExperts[expertIdx].style.background = ent.color + '33';
              ffnExperts[expertIdx].style.boxShadow = '0 0 15px ' + ent.color + '66';
            }
          }
        });

        if (progress < 1) {
          animFrames.push(requestAnimationFrame(step));
        } else {
          entities.forEach(ent => {
            ent.dom.style.opacity = '0';
          });
          if (mode === 'dense') {
            ffnZone.firstElementChild.style.background = 'rgba(51, 65, 85, 0.3)';
            ffnZone.firstElementChild.style.borderColor = '#475569';
            ffnZone.firstElementChild.style.boxShadow = 'none';
          } else {
            ffnExperts.forEach(ex => {
              ex.style.background = 'transparent';
              ex.style.boxShadow = 'none';
            });
          }
          tokens.forEach(t => {
            t.style.opacity = '0.9';
          });
          isSimulating = false;
          document.getElementById('btn-start').disabled = false;
          document.getElementById('btn-start').style.opacity = '1';
        }
      }

      animFrames.push(requestAnimationFrame(step));
    };

    document.addEventListener("DOMContentLoaded", () => {
      setPlaygroundMode('dense');
    });
    // 如果 DOM 已加载，直接初始化
    if (document.readyState === "complete" || document.readyState === "interactive") {
      setTimeout(() => setPlaygroundMode('dense'), 100);
    }
  })();
</script>

## 8. 什么时候适合用 MoT

**适合的场景**：模态边界清晰、各模态数据量都不小、训练算力是主要瓶颈。MoT 不改变每 token 的计算量，也不需要调 router，从 Dense 改过来的工程量较小，论文在三种设置下都观察到了训练加速。

**需要权衡的地方**：

1. **显存与存储**：总参数约为 Dense 的 $K$ 倍，训练时的优化器状态和推理时的权重占用都会相应增加。
2. **推理批处理**：一个 batch 里各模态 token 的比例不固定，每个分支的矩阵乘可能吃不满。作者建议用动态或连续批处理，或者把较小的模态 padding 到与较大的模态等长后合并计算，代价是部分无效计算或更高的延迟。
3. **结论的适用范围**：墙钟时间的结果来自一套特定的 AWS 集群，作者明确表示换到其他硬件需要重新验证；Transfusion 设置下文本收益有限；7B Transfusion 模型只训练了 0.5T token，尚未饱和。
4. **模态定义**：MoT 依赖“每个 token 属于哪一类”事先已知。Cosmos 3 的做法说明，这个类别不一定是模态，也可以是理解与生成这样的角色。

一句话总结：MoT 用 $K$ 倍的参数换取更快的训练收敛，每 token 计算量不变；拆注意力投影和 FFN 有用，拆 LayerNorm 作用不大；全局自注意力是唯一不拆的部分。

---
**参考文献：**
1. Liang, W., Yu, L., Luo, L., et al. *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*. TMLR, 2025. [arXiv:2411.04996](https://arxiv.org/abs/2411.04996)
2. NVIDIA. *Cosmos 3: Omnimodal World Models for Physical AI*. 2026. [arXiv:2606.02800](https://arxiv.org/abs/2606.02800)
3. Chameleon Team. *Chameleon: Mixed-Modal Early-Fusion Foundation Models*. 2024. [arXiv:2405.09818](https://arxiv.org/abs/2405.09818)
4. Zhou, C., Yu, L., Babu, A., et al. *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model*. 2024. [arXiv:2408.11039](https://arxiv.org/abs/2408.11039)
