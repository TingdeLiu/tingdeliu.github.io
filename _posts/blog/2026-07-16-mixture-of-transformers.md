---
layout: post
title: "Mixture-of-Transformers (MoT) 架构详解：多模态基础模型的模态解耦与稀疏化演进"
date:   2026-07-16
tags: [LLM Training, Multi-Modal, MoT, MoE, Sparse Transformer, Deep Learning, NVIDIA Cosmos]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "深入剖析 Meta AI、NVIDIA 等机构提出的 Mixture-of-Transformers (MoT) 稀疏架构。从密集模型的模态竞争难题出发，详解 MoT 核心的「模态特定参数解耦 + 全局自注意力」机制、确定性模态路由，并重点剖析 NVIDIA Cosmos 3 物理世界模型在此架构上的前沿工程落地与巨大能效比。"
---

* 目录
{:toc}

# Mixture-of-Transformers (MoT) 架构详解：多模态基础模型的模态解耦与稀疏化演进

在多模态基础模型（Multimodal Foundation Models）的开发与训练中，我们常常面临一个两难的困境：为了让模型拥有强大的跨模态理解与融合能力，我们倾向于将文本、图像、视频、音频等不同模态的 token 混合输入到同一个 Dense（密集型）Transformer 模型中。然而，这种“一刀切”的设计在实际训练中会带来极大的计算浪费与参数竞争冲突。

2024 年底，来自 Meta AI 和斯坦福大学等机构的研究者在论文 *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models* (arXiv:2411.04996) 中提出了一种全新的稀疏架构——**Mixture-of-Transformers (MoT)**。此后，NVIDIA 更是将其发扬光大，在最新推出的 **Cosmos 3** 物理世界模型（Physical AI World Model）中全线采用了 MoT 架构，将物理推理、视频生成与动作规划统一在一个高效的计算框架中。

本文将从密集模型的模态竞争难题出发，深入拆解 MoT 的核心机制，分析 NVIDIA Cosmos 3 的落地实现，并详细对比其与传统 Dense 及 Mixture-of-Experts (MoE) 的异同。

<!-- more -->

## 1. 多模态训练的痛点：模态竞争与梯度冲突

在传统的密集型多模态 Transformer（如 Chameleon、Transfusion 或 Unified-IO）中，虽然各种模态的输入被转化为不同的 token（例如文本的 BPE token，图像的 Vector Quantized patch token 等），但它们进入 Transformer Block 后，都会通过**完全相同**的一套参数进行投影与变换（共享相同的 $$W_Q, W_K, W_V$$ 投影矩阵，共享相同的 FFN 模块以及 LayerNorm 层）。

这种设计看似简单优雅，但在优化动力学上存在显著的痛点：**模态干扰（Modality Interference）**与**模态竞争（Modality Competition）**。

### 1.1 不同的数据分布与特征结构
文本和图像/视频在信息密度和统计分布上截然不同。文本是高度离散、符号化且语义密集的；而图像/视频补丁通常是连续、高冗余且空间高度相关的。强行使用同一套权重矩阵来处理这两种性质大相径庭的信号，无异于强迫一个通用工具同时处理粗活与细活。

### 1.2 梯度冲突（Gradient Conflict）
在联合训练过程中，文本损失函数产生的梯度和图像损失函数产生的梯度，在更新同一组参数时常常会发生反向拉扯。更新参数以拟合文本，可能会损害图像的表征质量，反之亦然。这种负面的干扰导致了多模态模型在扩大规模（Scaling Up）时，收敛效率极低，甚至出现表现停滞（Performance Plateau）。

<div align="center"><img src="/images/llm-training/mixture-of-transformers/modality_interference.jpg" width="90%" /><figcaption>图 1：传统密集模型中的模态梯度冲突 vs. MoT 中的模态隔离</figcaption></div>

为了解决这一冲突，最直接的想法是引入稀疏性，让不同的参数处理不同的信号。这就自然引出了 MoE 与 MoT 的对比。

---

## 2. 从 MoE 走向 MoT：为什么我们需要“全模块”的模态解耦？

在单模态大模型中，**Mixture of Experts (MoE)** 已经是公认的降低计算成本（FLOPs）的利器。然而，将 MoE 直接套用到多模态场景时，仍有诸多痛点。

### 2.1 传统 MoE 在多模态下的局限
1. **局部稀疏性**：传统的 MoE（如 Mixtral）通常**仅将前馈网络（FFN）替换为稀疏专家**，而自注意力机制（Self-Attention）中的投影矩阵以及层归一化（LayerNorm）依然是共享的密集参数。在多模态模型中，自注意力层的参数量巨大，且是捕捉模态特征的关键，这部分的参数竞争依然无法解决。
2. **可学习路由（Learnable Routing）的痛点**：MoE 依赖一个轻量级的门控网络（Router）来计算 token 到专家的亲和力分数。在多模态场景下，可学习路由容易面临**表征塌陷（Representation Collapse）**和**负载不均（Routing Imbalance）**。为了防止某些专家“饿死”，必须引入辅助损失（Auxiliary Loss），但这会干扰主任务的优化，且路由决策在训练中极不稳定。

### 2.2 MoT 的解决方案：完全模态解耦 + 确定性路由
为了克服上述局限，Mixture-of-Transformers (MoT) 进行了彻底的架构重构：

* **非嵌入参数完全解耦**：在每个 Transformer 层中，不仅 FFN，连同 LayerNorm、自注意力投影矩阵（$$W_Q, W_K, W_V$$）以及输出投影矩阵（$$W_O$$）都**按模态进行独立复制**。每个模态都拥有专属于自己的“Transformer 专家分支”。
* **确定性路由（Deterministic Routing）**：由于每个 token 属于什么模态在数据输入时是已知且固定的（例如，文本 token 还是图像 token），MoT 抛弃了可学习的门控 Router，直接使用预定义的模态掩码（`modality_masks`）进行**静态分流**。这消除了路由计算开销，彻底避免了表征塌陷和负载不均，极大地稳定了训练。
* **全局融合的桥梁：全局自注意力**：虽然投影矩阵是模态特定的，但投影出来的 $$Q, K, V$$ 向量会被重新拼回全局序列，进行统一 of 的自注意力计算。这保证了模型依然具备全序列的跨模态交互能力。

<div align="center"><img src="/images/llm-training/mixture-of-transformers/mot_architecture.jpg" width="90%" /><figcaption>图 2：Dense、MoE 与 MoT 的架构对比（MoT 实现了非嵌入参数的全面解耦）</figcaption></div>

---

## 3. MoT 架构数学拆解与工作流程

为了精确理解 MoT 在一个 Transformer Block 内是如何运转的，我们来拆解其数学形式。

### 3.1 符号定义
设输入序列为 $X = [x_1, x_2, \dots, x_N] \in \mathbb{R}^{N \times d}$，其中 $N$ 为序列长度，$d$ 为隐藏层维度。
每个 token $x_i$ 都有一个与之绑定的模态标签 $m_i \in \mathcal{M} = \{\text{text}, \text{image}, \text{speech}\}$。

对于任意模态 $m \in \mathcal{M}$，MoT 维护了一套该模态专属的参数：
* 专属的层归一化参数：$$\text{LN}_m$$
* 专属的注意力投影矩阵：$$\mathbf{W}_{Q,m}, \mathbf{W}_{K,m}, \mathbf{W}_{V,m} \in \mathbb{R}^{d \times d}$$
* 专属的输出投影矩阵：$$\mathbf{W}_{O,m} \in \mathbb{R}^{d \times d}$$
* 专属的前馈网络：$$\text{FFN}_m$$

### 3.2 详细计算步骤

#### 第一步：模态分流与专属投影（Router & Projection）
对输入序列中的每个 token $x_i$，根据其对应的模态标签 $m_i$，使用对应的专属 LayerNorm 和投影矩阵进行变换，得到查询（Query）、键（Key）和值（Value）：

$$
\mathbf{q}_i = \text{LN}_{m_i}(x_i) \mathbf{W}_{Q,m_i}
$$

$$
\mathbf{k}_i = \text{LN}_{m_i}(x_i) \mathbf{W}_{K,m_i}
$$

$$
\mathbf{v}_i = \text{LN}_{m_i}(x_i) \mathbf{W}_{V,m_i}
$$

通过这种方式，文本 token 使用模态特定的投影参数，图像 token 使用其专属参数，各行其道，参数之间不再干扰。

#### 第二步：全局注意力机制（Global Attention）
将各模态分别计算得到的 $\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i$ 按原始的序列索引重新排列，拼接为全局的矩阵：

$$
\mathbf{Q} = [\mathbf{q}_1; \mathbf{q}_2; \dots; \mathbf{q}_N], \quad \mathbf{K} = [\mathbf{k}_1; \mathbf{k}_2; \dots; \mathbf{k}_N], \quad \mathbf{V} = [\mathbf{v}_1; \mathbf{v}_2; \dots; \mathbf{v}_N]
$$

然后在全局上计算标准的多头注意力（Multi-Head Attention）：

$$
\mathbf{H} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

> **💡 关键设计哲学**：
> 这一步非常关键。虽然 $$Q, K, V$$ 的**生成阶段**是模态解耦的（使得不同模态可以使用最适合自身的映射空间），但是**计算注意力阶段**是全局的。这使得文本能够关注到图像的内容，图像也能融合上下文的文本语义，确保了跨模态表征的“融会贯通”。

#### 第三步：模态特定输出投影与残差连接（Output Projection & Residual）
注意力机制计算得出的全局表征向量 $\mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N]$ 会再次根据每个位置 of 的模态进行分流，并通过模态特定的输出投影矩阵与残差连接进行处理：

$$
y_i = x_i + \mathbf{h}_i \mathbf{W}_{O,m_i}
$$

#### 第四步：模态特定的 FFN 变换（Modality-Specific FFN）
最后，对每个 token $y_i$，应用其模态特定的 FFN 模块：

$$
z_i = y_i + \text{FFN}_{m_i}(\text{LN}'_{m_i}(y_i))
$$

其中，$$\text{LN}'_{m_i}$$ 是第二层模态特定的 LayerNorm。输出的 $Z = [z_1, z_2, \dots, z_N]$ 即为当前 Transformer Block 的输出。

---

## 4. 架构对比：Dense vs. MoE vs. MoT

为了帮助大家理清思路，我们可以将这三种主要的架构设计进行横向对比：

| 维度 | Dense (密集型) | MoE (混合专家) | MoT (混合 Transformer) |
| :--- | :--- | :--- | :--- |
| **自注意力机制投影 (QKV)** | 全局共享（密集参数） | 全局共享（密集参数） | **模态解耦（专属参数）** |
| **层归一化 (LayerNorm)** | 全局共享（密集参数） | 全局共享（密集参数） | **模态解耦（专属参数）** |
| **前馈网络 (FFN)** | 全局共享（密集参数） | 稀疏专家（多通道分布） | **模态解耦（专属参数）** |
| **路由/分流机制 (Routing)** | 无需路由 | 可学习门控网络 (Learnable Router) | **确定性路由 (Deterministic Mask)** |
| **训练稳定性** | 高，但存在模态干扰 | 较低，受路由坍塌和负载均衡影响 | **高，免去路由优化难题** |
| **推理时激活参数** | 100% | 依 Top-K 决定（通常占 10-30%） | **由输入序列的模态比例决定** |

---

## 5. 工业界前沿落地：NVIDIA Cosmos 3 中的 MoT 物理世界模型

如果说 Meta AI 2024 年的论文为 Mixture-of-Transformers 奠定了理论与小型实验的基石，那么 **NVIDIA Cosmos 3** 的发布则证明了这一稀疏架构在大型工业界物理 AI（Physical AI）上的巨大统治力。

### 5.1 双塔 MoT 结构：终结“范式割裂”
在自动驾驶、具身智能（Robotics）等物理世界模拟中，传统的方案通常是拼凑型的（Duct-taped）：由一个视觉语言模型（VLM）处理感知，一个扩散模型（Diffusion）负责视频预测，再由另外的策略网络（Policy）决定动作。这不仅导致极大的推理延迟，更阻碍了跨模态物理规律的深度融合。

NVIDIA Cosmos 3 彻底打破了这一界限。它构建了一个**“双塔（Dual-Tower）”架构**，但底座完全基于统一的 **Mixture-of-Transformers (MoT)**。它将输入序列划分为自回归（AR）与扩散（DM）子序列，在每一层解码器内部，并行持有两套独立参数（推理塔 + 生成塔），二者由预训练权重共同初始化：

*   **推理塔（Reasoner Tower，自回归）**：作为模型的“大脑”。这是一个因果自注意力（Causal Self-Attention）结构的 VLM，主要负责多模态感知输入、高层级任务规划、三维时空推理以及物理世界的意图理解。
*   **生成塔（Generator Tower，扩散 Transformer）**：作为“执行器和物理模拟器”。这是一个扩散基础的 Transformer 架构，使用全双向注意力（Bidirectional Attention），条件化于推理塔输出的上下文，通过 Flow Matching 迭代去噪预测未来视频帧、音频和机器人执行器的连续动作轨迹（Actions）。

这两个塔内部通过**双流联合注意力（Dual-Stream Joint Attention）**相耦合：
*   **AR 子序列（推理塔）**仅能关注自身的历史，以维持自回归的因果完整性；
*   **DM 子序列（生成塔）**则以 AR 和 DM 标记的并集为 Key/Value，使得每个扩散生成 token 都能自由“读取”推理塔提供的上下文语义；
*   **关键约束**：AR 标记永远不会被 DM 标记污染或更新，保证了推理通路的因果隔离。

这种设计的精妙之处在于：**推理为生成提供高层语义指导，而生成过程不污染推理逻辑**，二者在同一张注意力图里协作，却互不破坏各自的归纳偏置。

### 5.2 Cosmos 3 中 MoT 架构的核心设计
1.  **3D 多维旋转位置编码（mRoPE）**：为了让 MoT 能够无缝协调图像、文本、视频帧与动作 token，Cosmos 3 引入了 **mRoPE (3D Multi-dimensional Rotary Position Embedding)**。mRoPE 将空间（高度、宽度）和时间轴融合在一个多维旋转编码空间中，使不同模态投影参数在进入全局注意力时，具备了物理世界的相对位置坐标系。
2.  **模态灵活部署与微调**：得益于 MoT 的参数隔离性质，开发者在使用 NVIDIA Cosmos 3 时，甚至可以独立于生成塔去优化或微调推理塔（Reasoner Tower），或者在 Reasoner 的制约下单独微调 Generator 产生特定的动作序列。
3.  **模型规模与参数**：NVIDIA 基于 MoT 架构开源了不同尺度的变体，其总参数量由于双塔设计（推理塔 + 生成塔各持一套参数）约为同级密集模型的两倍：
    *   **Cosmos 3 Nano (16B / 密集骨干 8B)**：对端侧及工作站算力极其友好，可在单张 RTX PRO 6000 级显卡上实现实时具身导航与推理。
    *   **Cosmos 3 Super (64B / 密集骨干 32B)**：适用于大型数据中心 Hopper 和 Blackwell GPU，能生成大规模、高保真度的合成物理世界轨迹。

> 💡 **系统级关联**：关于 Cosmos 3 平台的完整技术演进、数据策展 Video Curator 以及三种动作模式（前向动力学、逆动力学与策略模型）的详细拆解，可以参考我之前撰写的 [世界模型综述：Cosmos 3 全模态统一世界模型](/World-Models-Survey/#46-cosmos-3%E5%85%A8%E6%A8%A1%E6%80%81%E7%BB%9F%E4%B8%80%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B2026)。

---

## 6. 实验结果与系统效率提升

Meta AI 团队对 MoT 在多种多模态基础模型设置中进行了广泛的评测。主要评估了两种典型的多模态框架：以自回归方式统一处理图文的 **Chameleon** 框架，以及结合扩散模型和自回归的 **Transfusion** 框架。

### 6.1 极佳的 FLOPs 压缩效率
* **Chameleon 图文生成 (7B参数量)**：MoT 架构仅激活了 **55.8% 的 FLOPs**，就达到了与 Dense 7B Baseline **完全等价**的生成质量与下游任务表现。
* **三模态扩展 (Text+Image+Speech, 443M参数量)**：当引入第三种模态（语音）时，MoT 的稀疏化优势更加巨大，仅需密集模型 **37.2% 的 FLOPs** 即可达到同等性能。
* **Transfusion 图像生成 (7B参数量)**：在处理扩散模型目标时，MoT 架构对图像模态的计算代价缩减到了原来的 **1/3**，而图像质量没有受到任何损失。

### 6.2 硬件运行时间与吞吐（Wall-Clock Speedup）
理论上的 FLOPs 压缩并不总是能等价转化为实际的速度提升（由于稀疏算子的通信开销），但 MoT 凭借**确定性分流**和**算子融合（Kernel Fusion）**，在 A100 GPU 上的实际测速非常亮眼：
* 在处理图像模态时，MoT 达到了 Dense 47% 的硬件运行时间（即 **2.13× 的实际加速**）。
* 在处理文本模态时，也仅需 Dense 的 75.6% 的时间。

### 6.3 理论支撑：子任务凸化（Subtask Convexification）
除了工程和性能优势，理论分析也为 MoT 的快速收敛提供了强力解释：
多模态联合优化由于存在梯度冲突，往往呈现高度非凸的性质。MoT 通过确定性掩码将参数分配给对应的专家，实质上将复杂的非凸混合优化问题拆解为了多个更平滑、接近强凸（Strongly Convex）的子问题。研究表明，在这种状态下，模型的收敛速度会呈现**对数级（Logarithmic）的加速**，优化过程更加平稳。

---

## 7. 互动沙盘：直观感受 Token 分流与参数冲突

为了更直观地理解 Dense、MoE 以及 MoT 对 Token 的分流控制和参数冲突情况，你可以在下方的**互动沙盘**中切换不同架构，点击“开始分流”观察其动态执行流向和性能指标：

<div id="mot-playground" style="margin: 20px 0; padding: 20px; background: radial-gradient(circle at top, #1e1b4b 0%, #0f172a 100%); border-radius: 12px; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);">
  <h3 style="margin-top: 0; color: #38bdf8; text-align: center; font-size: 1.4rem; font-weight: bold;">多模态 Token 分流物理模拟器</h3>
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
      <div style="font-size: 0.8rem; color: #94a3b8;">激活参数占比 (FLOPs)</div>
      <div id="stat-flops" style="font-size: 1.3rem; font-weight: bold; color: #ef4444; margin-top: 5px;">100%</div>
    </div>
    <div style="text-align: center;">
      <div style="font-size: 0.8rem; color: #94a3b8;">模态梯度冲突率</div>
      <div id="stat-conflict" style="font-size: 1.3rem; font-weight: bold; color: #ef4444; margin-top: 5px;">极高 (100%)</div>
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
        flops.innerText = '100%';
        flops.style.color = '#ef4444';
        conflict.innerText = '极高 (100%)';
        conflict.style.color = '#ef4444';
        routing.innerText = '无 (全局共享)';
      } else if (mode === 'moe') {
        flops.innerText = '激活约 55%';
        flops.style.color = '#fbbf24';
        conflict.innerText = '中 (共享QKV/LN)';
        conflict.style.color = '#fbbf24';
        routing.innerText = '可学习动态路由';
      } else if (mode === 'mot') {
        flops.innerText = '激活约 45%';
        flops.style.color = '#10b981';
        conflict.innerText = '极低 (0% 解耦)';
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

---

## 8. 总结与未来展望

Mixture-of-Transformers (MoT) 的提出，为我们思考多模态模型的参数排布带来了极具启发性的视角。过去，我们一味追求“同一个模型、同一个权重”去理解世界；而 MoT 和 NVIDIA Cosmos 3 的成功经验告诉我们，**“物理层面的独立参数解耦”与“逻辑层面的全局注意力融合”相结合，才是多模态世界模型更高效的发展方向**。


### 💡 MoT 的几点核心启示：
1. **多模态不等于参数大锅饭**：对于结构和密度差异巨大的模态，前期的 QKV 投影和后期的 FFN 进行参数隔离，能够极大地释放各自的潜力，避免“互相妥协”。
2. **路由不一定非要“可学习”**：在天然拥有模态标签的多模态场景中，利用先验知识的确定性路由（Static Routing）比可学习路由（Dynamic Routing）更加稳定、简单且高效。
3. **全局自注意力依然是护城河**：保留跨模态的全局注意力是确保模型进行深度推理和模态对齐的根本，绝不能为了稀疏化而将注意力也完全割裂。

随着未来基础模型走向“声音、视频、三维场景、物理动作”等更广阔的 Physical AI 世界，MoT 这样兼具高性能与低能耗的稀疏解耦架构，无疑将成为支撑下一代多模态智能的核心基石。

---
**参考文献：**
1. Liang, et al. *"Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models"*, arXiv:2411.04996, 2024. (Published in TMLR 2025)
2. NVIDIA Cosmos Team. *"NVIDIA Cosmos: World Foundation Models for Physical AI"*, Technical Report, 2025.
3. Chameleon Team, Meta AI. *"Chameleon: Mixed-Modal Early-Fusion Foundation Models"*, arXiv:2405.09818, 2024.
---
