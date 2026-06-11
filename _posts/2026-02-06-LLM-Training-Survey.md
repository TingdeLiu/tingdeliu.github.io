---
layout: post
title: "大语言模型训练综述"
date:   2026-06-11
tags: [LLM, Deep Learning, NLP, Training, AI]
categories: research
comments: true
author: Tingde Liu
toc: true
excerpt: "大语言模型训练是当前人工智能领域最前沿的研究方向之一。本文系统梳理大模型训练的完整流程、核心技术、工程实践与最新进展，为学习和研究大模型训练提供全面参考。"
---

# 引言

大语言模型（Large Language Model, LLM）的兴起标志着人工智能进入了新的发展阶段。从2018年BERT和GPT的出现，到2020年GPT-3展现出令人惊讶的少样本学习能力，再到2022年ChatGPT引爆全球对话式AI浪潮，大模型在短短几年间实现了质的飞跃。这些模型不仅在传统NLP任务上达到了接近人类的水平，更展现出了代码生成、数学推理、创意写作等广泛的能力。

## 🎯 为什么关注大模型训练？

训练是大模型能力的来源。一个高性能大模型的诞生，离不开：
- 📊 **海量高质量数据**：数万亿token的精心处理
- 💰 **大规模计算资源**：数千GPU并行训练数月
- 🧠 **精巧的训练策略**：从预训练到对齐的完整pipeline
- ⚙️ **工程化的实现**：分布式训练、内存优化、稳定性保证

然而，大模型训练的知识和技术往往分散在各类论文、博客和代码库中，缺乏系统性的整理。**本文旨在填补这一空白。**

## 📚 本文内容

本综述系统梳理大模型训练的核心技术，包括：

1. 🏗️ **大模型训练基础概述**：大模型训练概念、演进历程与核心组成
2. 🌀 **预训练阶段**：目标函数、数据源与配比、核心参数策略与 MoE 架构
3. 🎯 **监督微调阶段**：SFT 训练目标、数据来源与前沿合成及质量清洗技术
4. 🤝 **偏好对齐阶段**：RLHF、DPO 与 GRPO 算法原理及推理模型训练范式
5. 🔁 **Post-Training 与灾难性遗忘**：持续学习中的遗忘机理与防遗忘策略
6. 🔀 **分布式训练技术**：数据/张量/流水线并行、序列并行与 ZeRO 状态分片
7. ⚡ **训练优化技术**：主流优化器对比、学习率策略与 Flash Attention
8. 📉 **模型量化技术**：PTQ 与 QLoRA 低比特微调，不同格式精度/性能权衡
9. 🗂️ **数据工程**：数据采集、清洗、去重与配比最佳实践
10. 📈 **评估与基准测试**：知识、推理、代码、长文本及安全性评估
11. 🚀 **实践指南与最佳实践**：算力估算、训练监控、OOM 调试与断点恢复
12. 💬 **常见问题 (FAQ)**：训练常见异常、微调与对齐实践答疑
13. 👁️ **迈向多模态**：VLM 视觉-语言融合架构与两阶段训练流程
14. 📚 **参考资源**：经典论文、核心开源项目与大模型学习路径

## 👥 目标读者

- 🔬 **研究人员**：了解大模型训练的完整技术栈
- 👨‍💻 **工程师**：掌握实际训练中的工程实践和优化技巧
- 🎓 **学生**：建立对大模型训练的系统性认知
- 💼 **从业者**：跟踪最新技术进展和行业动态

> **💡 我们的承诺**
>
> 本文力求在理论深度和实践指导之间取得平衡，既阐述核心原理，也提供**可直接运行的代码示例**和**可落地的工程方案**。

---

## 📖 快速导航

本文共分为 14 个主要章节，建议根据需求选择性阅读：

| 章节 | 内容 | 适合读者 | 阅读时间 |
|------|------|---------|---------|
| **1. 基础概述** | 大模型训练概念、演进历程 | 所有读者 | 10分钟 |
| **2. 预训练阶段** | 目标函数、数据源、核心参数与MoE | 所有读者 ⭐必读 | 30分钟 |
| **3. 监督微调阶段** | SFT训练目标、数据构建、合成与过滤 | 所有读者 ⭐必读 | 15分钟 |
| **4. 偏好对齐阶段** | RLHF、DPO、GRPO与推理模型RL对齐 | 所有读者 ⭐必读 | 15分钟 |
| **5. Post-Training与遗忘** | 灾难性遗忘、经验回放与Self-Output | 工程师、研究者 ⭐推荐 | 15分钟 |
| **6. 分布式训练** | DP/TP/PP/ZeRO等并行技术 | 工程师、研究者 | 20分钟 |
| **7. 训练优化** | 优化器、学习率、Flash Attention | 工程师 | 15分钟 |
| **8. 模型量化** | GPTQ/AWQ量化与QLoRA低比特微调 | 工程师、研究者 ⭐推荐 | 20分钟 |
| **9. 数据工程** | 数据采集、清洗、去重与配比 | 工程师、研究者 | 25分钟 |
| **10. 评估基准** | MMLU、GSM8K等评测体系 | 研究者、从业者 | 15分钟 |
| **11. 实践指南** | 硬件配置、成本估算、监控与断点恢复 | 工程师 ⭐必读 | 30分钟 |
| **12. 常见问题** | 常见训练异常与微调/对齐问题答疑 | 所有读者 ⭐推荐 | 20分钟 |
| **13. VLM多模态** | 视觉语言模型架构与两阶段融合训练 | 工程师、研究者 | 15分钟 |
| **14. 参考资源** | 经典论文、标杆开源项目与学习路径 | 所有读者 | 15分钟 |

> **💡 阅读建议**
>
> - **初学者**：重点阅读"1. 基础概述" → "2. 预训练 / 3. SFT / 4. 对齐阶段" → "12. 常见问题"
> - **工程师**：重点阅读"11. 实践指南" → "6. 分布式训练" → "5. Post-Training与遗忘"
> - **研究者**：重点阅读"2. 预训练 / 3. SFT / 4. 对齐阶段" → "5. Post-Training与遗忘" → "10. 评估基准"
> - **全面学习**：按顺序完整阅读（约3.5小时）

---

# 1. 大模型训练基础概述

## 1.1 🤔 什么是大模型训练？

大模型训练是指使用海量文本数据，通过深度学习算法训练具有数十亿甚至数万亿参数的神经网络模型的过程。这些模型通常基于Transformer架构，能够学习语言的统计规律和语义理解能力。

训练大模型的目标是让模型获得：
- 🧠 **语言理解能力**：理解自然语言的语法、语义和上下文
- 📚 **知识储备**：从训练数据中学习世界知识
- 🔍 **推理能力**：基于已有信息进行逻辑推理和问题解决
- ✅ **指令遵循**：准确理解并执行用户的各类指令

> **📌 核心概念**
>
> - **规模**：参数量从10亿到数万亿不等
> - **数据**：训练数据通常达到数万亿tokens
> - **时间**：完整训练周期从数周到数月
> - **成本**：中大型模型训练成本从数十万到数百万美元

## 1.2 📅 大模型训练的演进历程

大模型训练经历了从小规模实验到工业化生产的重要转变：

### 1.2.1 早期探索阶段（2018-2019）🌱
- GPT-1、BERT等模型验证了预训练-微调范式的有效性
- 模型规模：百万到亿级参数
- 关键突破：自监督预训练、Transformer架构

### 1.2.2 规模化阶段（2020-2021）📈
- GPT-3将模型规模扩展到1750亿参数
- 发现涌现能力（Emergent Abilities）
- 少样本学习能力显著提升

### 1.2.3 对齐与应用阶段（2022-2023）🎯
- InstructGPT和ChatGPT引入RLHF（基于人类反馈的强化学习）
- 从"能用"到"好用"的关键转变
- 大模型开始广泛应用于实际场景

### 1.2.4 开源与民主化阶段（2024-至今）🌍
- LLaMA、Mistral等开源模型快速发展
- 训练效率和成本持续优化
- 多模态、长上下文等能力不断增强

## 1.3 🎶大模型训练的三大核心阶段

现代大模型训练遵循**预训练 → 监督微调 → 偏好对齐**的三阶段范式，这已成为GPT-4、Claude、Gemini、LLaMA等主流模型的标准流程。

```mermaid
flowchart LR
    A["原始文本数据<br>数万亿tokens"] --> B["阶段1: 预训练<br>Pre-training"]
    B --> C["Base Model<br>基座模型"]
    C --> D["阶段2: 监督微调<br>SFT"]
    E["指令-回答对<br>数万样本"] --> D
    D --> F["SFT Model<br>指令模型"]
    F --> G["阶段3: 偏好对齐<br>RLHF/DPO"]
    H["偏好对比数据<br>数万对"] --> G
    G --> I["✓ Aligned Model<br>最终部署"]

    style A fill:#e3f2fd,stroke:#01579b
    style C fill:#fff9c4,stroke:#f57f17
    style E fill:#e3f2fd,stroke:#01579b
    style F fill:#ffe0b2,stroke:#e65100
    style H fill:#e3f2fd,stroke:#01579b
    style I fill:#c8e6c9,stroke:#1b5e20
```

**关键对比**：

| 阶段 | 数据规模 | 时间周期 | 成本占比 | 目标 |
|------|---------|---------|---------|------|
| **预训练** | 数万亿tokens | 数周-数月 | 80-90% | 学习语言基础和世界知识 |
| **监督微调** | 数万样本 | 数天-数周 | 5-10% | 学会遵循指令和对话 |
| **偏好对齐** | 数万对比对 | 数天-数周 | 5-10% | 符合人类偏好和价值观 |

### 1.3.1 阶段一：预训练（Pre-training）

从海量无标注文本中学习语言的统计规律、语法结构和世界知识，训练出**Base Model（基座模型）**。

**核心特点**：
- 📊 **数据规模最大**：数万亿tokens（如 LLaMA-3 使用 15T tokens）
- ⏰ **训练时间最长**：在数千块GPU上训练数周到数月
- 💰 **成本最高**：占总训练成本的 80-90%
- 🎯 **目标**：Next Token Prediction（预测下一个词）

**输出能力**：具备文本续写能力，但不擅长问答和指令遵循。

### 1.3.2 阶段二：监督微调（SFT）

使用高质量的指令-回答对训练，将Base Model转化为能够理解指令的**SFT Model（指令模型）**。

**核心特点**：
- 📊 **数据规模小**：10k-100k 高质量样本
- ⏰ **训练时间短**：数天到数周
- 💰 **成本较低**：占总成本的 5-10%
- 🎯 **目标**：Instruction Following（指令遵循）

**输出能力**：能够理解和执行用户指令，进行多轮对话。

### 1.3.3 阶段三：偏好对齐（Alignment）

通过人类反馈或AI反馈优化模型行为，使其更符合人类期望和价值观，训练出**Aligned Model（对齐模型）**。

**核心特点**：
- 📊 **数据规模**：数万对偏好对比数据
- ⏰ **训练时间**：数天到数周
- 💰 **成本**：占总成本的 5-10%
- 🎯 **方法**：RLHF、DPO、RLAIF等

**输出能力**：输出更有帮助、更安全、更符合人类价值观。

### 1.3.4 三阶段总结对比

| 阶段 | 预训练 | 监督微调 | 偏好对齐 |
|------|--------|----------|----------|
| **目标** | 学习语言基础 | 教会指令遵循 | 符合人类偏好 |
| **数据类型** | 无标注文本 | 指令-回答对 | 偏好对比数据 |
| **数据规模** | 数万亿tokens | 数万-数十万样本 | 数万-数十万对比 |
| **训练时长** | 数周-数月 | 数小时-数天 | 数小时-数天 |
| **计算需求** | 数千GPU | 数十-数百GPU | 数十-数百GPU |
| **成本占比** | ~80-90% | ~5-10% | ~5-10% |
| **学习率** | 1e-4 ~ 3e-4 | 1e-5 ~ 5e-5 | 5e-7 ~ 5e-6 |
| **Epoch数** | &lt;1 epoch（太大） | 1-3 epochs | 1-3 epochs |
| **输出模型** | Base Model | SFT Model | Aligned Model |

**关键洞察**：
- 预训练是能力的来源（占成本90%）
- SFT是能力的激活（数据质量 > 数量）
- 对齐是体验的保证（必不可少）

## 1.4 🧩 大模型训练的核心组成要素

一个完整的大模型训练系统包含以下核心要素：

### 1.4.1 数据（Data）📊
- **预训练数据**：网页、书籍、代码、学术论文等
- **微调数据**：指令-回答对、对话数据
- **偏好数据**：人类标注的偏好对比数据
- **数据处理流程**：清洗、去重、质量过滤、毒性检测

### 1.4.2 模型架构（Model Architecture）🏛️
- **基础架构**：Transformer（Encoder、Decoder或Encoder-Decoder）
- **位置编码**：绝对位置编码、相对位置编码、RoPE、ALiBi
- **注意力机制**：Multi-Head Attention、Grouped-Query Attention、Multi-Query Attention
- **归一化方式**：LayerNorm、RMSNorm、Pre-Norm vs Post-Norm
- **激活函数**：GELU、SwiGLU、GeGLU

<div align="center">
  <img src="/images/llm-training/transformer-architecture.png" width="75%" alt="Transformer架构" />
  <figcaption>图：Transformer架构详解（来源："Attention is All You Need" 论文 Figure 1）</figcaption>
</div>

### 1.4.3 优化器与训练策略（Optimization）⚡
- **优化器**：AdamW、Adafactor、Lion
- **学习率调度**：Warmup、Cosine Decay、Constant
- **梯度处理**：Gradient Clipping、Gradient Accumulation
- **正则化**：Dropout、Weight Decay

### 1.4.4 分布式训练框架（Distributed Training）🔀
- **数据并行**：DDP（Distributed Data Parallel）
- **张量并行**：Megatron-LM Tensor Parallelism
- **流水线并行**：Pipeline Parallelism、1F1B Schedule
- **序列并行**：Sequence Parallelism
- **混合并行**：3D Parallelism（数据+张量+流水线）
- **优化器状态并行**：ZeRO-1/2/3（DeepSpeed）

### 1.4.5 计算基础设施（Infrastructure）💻
- **硬件**：GPU集群（A100、H100等）、TPU、专用AI芯片
- **互联网络**：InfiniBand、NVLink、PCIe
- **存储系统**：高性能分布式存储
- **监控与日志**：TensorBoard、Weights & Biases、MLflow

## 1.5 🚧大模型训练的主要挑战

> **⚠️ 挑战总览**
>
> 大模型训练是一项极具挑战性的系统工程，需要在计算资源、数据质量、训练稳定性、模型对齐等多个维度取得平衡。成功训练一个高性能大模型不仅需要技术实力，更需要工程经验的积累。

### 1.5.1 计算资源与成本

训练大模型需要巨大的计算资源：
- GPT-3级别模型训练成本约数百万美元
- 需要数千块高端GPU并行训练数月
- 碳排放和能源消耗问题
- 如何降低训练成本成为关键挑战

### 1.5.2 数据质量与规模

高质量训练数据是模型性能的基础：
- 网络数据存在噪声、偏见和有害内容
- 数据去重、清洗和质量控制的工程挑战
- 隐私和版权问题
- 高质量人工标注数据成本高昂

### 1.5.3 训练稳定性

大规模训练面临稳定性挑战：
- Loss spike（损失突然上升）
- 梯度爆炸/消失
- 数值不稳定（Numerical Instability）
- 分布式训练中的同步问题

### 1.5.4 模型对齐与安全

让模型行为符合人类期望：
- 如何准确捕捉人类偏好
- 避免有害、偏见或不准确的输出
- Reward Hacking问题（奖励函数被利用）
- 长期对齐的稳定性

### 1.5.5 评估与基准测试

如何全面评估模型能力：
- 现有基准测试可能被"刷榜"
- 难以量化创造性和开放式能力
- 多语言、多模态评估的复杂性
- 真实应用场景下的表现差异

### 1.5.6 涌现能力的不可预测性

模型规模扩大带来的未知：
- 某些能力只在特定规模后出现
- 难以提前预测模型行为
- 可能出现意外的能力或问题
- 如何系统性理解规模法则（Scaling Laws）

---

# 2. 预训练阶段
————Pre-training

预训练是大模型训练的基石，目标是让模型从海量无标注文本中学习语言的统计规律和世界知识。

> **🎯 本章导读**
>
> 预训练是整个训练流程中**成本最高、时间最长、技术难度最大**的阶段，占总成本的80-90%。本章将详细介绍预训练的目标函数、数据处理、训练技巧和前沿技术，帮助读者理解如何从零开始训练一个基座模型。

## 2.1 预训练完整流程概览

下图展示了从原始数据到 Base Model 的完整训练流程：

```mermaid
graph TD
    A0[Step 0: 分词器训练] --> H
    A[原始数据采集] --> B[数据清洗与过滤]
    B --> C[质量评估]
    C --> D{是否通过?}
    D -->|否| E[丢弃]
    D -->|是| F[去重处理]
    F --> G[MinHash/SimHash去重]
    G --> H[Tokenization]
    H --> I[数据配比与采样]
    I --> J[构建训练批次]
    J --> K[分布式训练<br/>3D并行: DP+TP+PP]
    K --> L[前向传播]
    L --> M[计算Loss<br/>Next Token Prediction]
    M --> N[反向传播]
    N --> O[梯度同步 All-Reduce]
    O --> P[优化器更新<br/>AdamW]
    P --> Q{是否保存checkpoint?}
    Q -->|是| R[保存模型状态]
    Q -->|否| S{训练完成?}
    R --> S
    S -->|否| J
    S -->|是| T[Base Model<br/>基座模型]

    style A0 fill:#f3e5f5,stroke:#7b1fa2
    style A fill:#e1f5ff
    style T fill:#c8e6c9
    style M fill:#fff9c4
    style K fill:#ffe0b2
```

**流程说明**：
1. **数据准备阶段**（A-I）：占整体时间的 20-30%，包括采集、清洗、去重、**分词器训练**与分词
2. **训练迭代阶段**（J-S）：占整体时间的 70-80%，核心是前向-反向-优化循环
3. **Checkpoint 管理**：每 1000-5000 步保存一次，总训练步数通常 100k-500k 步

## 2.2 分词器训练 (Tokenizer Training)

在正式开始模型训练之前，我们需要定义模型如何“阅读”文本。分词器将连续的文本切割成模型可理解的最小单元（Tokens）。

### 2.2.1 为什么需要训练分词器？
如果直接使用字符（Character）或词（Word），会面临词表过大（难以收敛）或单个 Token 信息密度过低（序列过长）的问题。现代大模型普遍采用 **子词（Subword）** 分词方案，如 **BPE (Byte Pair Encoding)**。

### 2.2.2 核心权衡：词表大小 (Vocab Size)
*   **大词表（如 100k+）**：
    *   ✅ 优点：单个 Token 承载信息多，序列更短，推理更快。
    *   ❌ 缺点：Embedding 层参数巨大，稀疏词难以充分训练。
*   **小词表（如 32k）**：
    *   ✅ 优点：Embedding 层小，参数利用率高，适合小模型。
    *   ❌ 缺点：同一个句子生成的 Token 数更多，增加计算开销。

> **💡 MiniMind 经验**：对于参数量在 500M 以下的小模型，词表不宜过大（如 6400 个字符或 32k BPE 词表），以确保每个 Token 的向量都能得到充分更新。

### 2.2.3 分词器训练实战 (Python)

使用 Hugging Face 的 `tokenizers` 库，我们可以快速训练一个支持多语言的 BPE 分词器：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 1. 初始化 BPE 模型
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. 配置训练器
trainer = BpeTrainer(
    vocab_size=32000, 
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 3. 训练分词器
files = ["data/corpus_1.txt", "data/corpus_2.txt"]
tokenizer.train(files, trainer)

# 4. 保存分词器
tokenizer.save("my_tokenizer.json")
```

## 2.3 预训练目标函数

预训练的核心是设计合适的目标函数，让模型从无标注文本中学习语言规律。

### 2.3.1 自回归语言建模（Autoregressive Language Modeling）

**核心思想**：给定前文，预测下一个 token（Next Token Prediction）

**数学表达**：

对于文本序列 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$，训练目标是最大化：

$$
\mathcal{L}_{\text{AR}} = \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)
$$

其中 $\theta$ 是模型参数。

**训练过程**：
- **Teacher Forcing**：训练时使用真实的前文作为输入
- **Causal Masking**：注意力机制只能看到左侧（过去）的 token
- **损失计算**：对每个位置计算交叉熵损失，然后平均

**代表模型**：
- **GPT 系列**（GPT-3, GPT-4）：纯 Decoder 架构
- **LLaMA 系列**（LLaMA-2, LLaMA-3）：开源高性能模型
- **PaLM、Gemini**：Google 的大模型

**优势**：
- ✅ 生成能力强，擅长续写和对话
- ✅ 架构简单，易于扩展到超大规模
- ✅ 训练效率高

**实战训练代码**：
```python
for batch in dataloader:
    input_ids = batch['input_ids']  # shape: [batch_size, seq_len]

    # 前向传播
    logits = model(input_ids)  # [batch_size, seq_len, vocab_size]

    # Next Token Prediction: 预测右移一位
    # input:  [x1, x2, x3, x4]  →  target: [x2, x3, x4, x5]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    loss = cross_entropy(shift_logits, shift_labels)
    loss.backward()
    optimizer.step()
```

### 2.3.2 掩码语言建模（Masked Language Modeling）

**核心思想**：随机 mask 部分 token，预测被 mask 的内容

**数学表达**：

$$
\mathcal{L}_{\text{MLM}} = \sum_{i \in \mathcal{M}} \log P(x_i \mid \mathbf{x}_{\backslash \mathcal{M}}; \theta)
$$

其中 $\mathcal{M}$ 是被 mask 的位置集合，$\mathbf{x}_{\backslash \mathcal{M}}$ 表示除了 masked 位置外的所有 token。

**训练策略**（BERT 方式）：
- 随机选择 15% 的 token 进行处理：
  - 80% 替换为 `[MASK]`
  - 10% 替换为随机 token
  - 10% 保持不变

**代表模型**：
- **BERT**：双向 Encoder，预训练+微调范式
- **RoBERTa**：优化的 BERT（更多数据、更大 batch、去除 NSP）
- **DeBERTa**：Disentangled Attention

**优势**：
- ✅ 双向上下文理解（能同时看到左右信息）
- ✅ 适合理解类任务（分类、信息抽取）

**劣势**：
- ❌ 预训练和微调的 gap（预训练有 [MASK]，微调没有）
- ❌ 生成能力较弱

### 2.3.3 混合目标与其他变体

#### 2.3.3.1 Encoder-Decoder 架构（T5、BART）
- **Span Corruption**：mask 连续的 token span
- **适合序列到序列任务**：翻译、摘要

#### 2.3.3.2 Prefix Language Modeling（PrefixLM）
- **UL2**：结合双向和单向建模
- **灵活性高**：可以选择双向或单向注意力

#### 2.3.3.3 Fill-in-the-Middle（FIM）
- **用于代码模型**：预测中间缺失的代码
- **代表**：CodeLlama、StarCoder
- **格式**：`[前缀] <FILL> [后缀] → [中间内容]`

## 2.4 预训练数据

### 2.4.1 数据来源
- **网页数据**：Common Crawl、C4（Colossal Clean Crawled Corpus）
- **书籍**：BookCorpus、Books3
- **代码**：GitHub、Stack Overflow
- **学术文献**：arXiv、PubMed
- **对话数据**：Reddit、社交媒体
- **百科知识**：Wikipedia、Wikidata

### 2.4.2 数据处理流程

#### 2.4.2.1 数据采集
- 网页爬取与下载
- API数据获取
- 开源数据集整合

#### 2.4.2.2 质量过滤
- 语言检测与过滤
- 内容质量评估（长度、重复性、可读性）
- 毒性和有害内容检测
- 个人信息删除（PII Removal）

#### 2.4.2.3 去重
- **精确去重**：完全相同的文档
- **模糊去重**：MinHash、SimHash等算法
- **跨数据集去重**：避免测试集泄露

#### 2.4.2.4 Tokenization
- BPE（Byte Pair Encoding）
- WordPiece
- Unigram
- SentencePiece

### 2.4.3 数据配比（Data Mixture）

**核心原则**：不同数据源的配比直接影响模型的能力分布

**典型配比示例**（参考 LLaMA）：

| 数据源 | 比例 | 说明 |
|-------|------|------|
| **Common Crawl / C4** | 67% | 网页数据，提供广泛的语言知识 |
| **Books** | 15% | 高质量长文本，提升推理和叙事能力 |
| **GitHub** | 4.5% | 代码数据，提升代码理解和生成能力 |
| **Wikipedia** | 4.5% | 百科知识，提供结构化知识 |
| **ArXiv** | 2.5% | 学术论文，提升科学推理能力 |
| **StackExchange** | 2% | 问答数据，提升问答能力 |

**配比策略**：
- **上采样（Upsampling）**：高质量数据源可以重复多次
- **下采样（Downsampling）**：低质量或超大规模数据源采样一部分
- **动态调整**：训练后期可以增加特定领域数据的比例

**实战代码**：
```python
data_mixture = {
    'common_crawl': 0.67,    # 网页数据 - 通用语言能力
    'books': 0.15,           # 书籍 - 长文本推理
    'github': 0.045,         # 代码 - 编程能力
    'wikipedia': 0.045,      # 百科 - 事实知识
    'arxiv': 0.025,          # 论文 - 科学推理
    'stackexchange': 0.02    # 问答 - QA能力
}

def sample_batch(data_mixture, batch_size):
    """按配比构建训练批次"""
    batch = []
    for source, weight in data_mixture.items():
        n_samples = int(batch_size * weight)
        batch.extend(sample_from_source(source, n_samples))
    return batch
```

## 2.5 预训练的关键技术

### 2.5.1 学习率调度

**标准三阶段调度**：
```
Warmup → Peak Learning Rate → Cosine/Linear Decay
```

```mermaid
graph LR
    A[步骤0<br/>lr=0] --> B[Warmup阶段<br/>0-2%步数<br/>线性增长]
    B --> C[峰值阶段<br/>2-10%步数<br/>保持峰值]
    C --> D[Decay阶段<br/>10-100%步数<br/>余弦衰减]
    D --> E[结束<br/>lr=峰值×10%]

    style A fill:#e3f2fd
    style B fill:#fff9c4
    style C fill:#ffcdd2
    style D fill:#c8e6c9
    style E fill:#e3f2fd
```

<div align="center">
  <img src="/images/llm-training/learning-rate-schedule.png" width="85%" alt="预训练学习率调度曲线" />
  <figcaption>图：预训练学习率调度实际曲线 - Warmup-Peak-Decay三阶段（来源：Chinchilla 论文 Figure 1）</figcaption>
</div>

**关键参数**：
- **Warmup步数**：通常2,000-10,000步（占总步数的1-2%）
- **Peak Learning Rate**：根据模型规模调整
  - 小模型（<1B参数）：3e-4 ~ 1e-3
  - 中型模型（1-10B参数）：1e-4 ~ 3e-4
  - 大模型（10B+参数）：6e-5 ~ 2e-4
- **Decay策略**：Cosine Annealing最常用
- **最小学习率**：通常为峰值的10%

**Warmup的重要性**：
- 避免训练初期的梯度爆炸
- 让优化器状态（Adam的momentum）逐步稳定
- 大模型训练的必要技巧

**学习率与批次大小关系**（Linear Scaling Rule）：

$$
\text{lr}_{\text{new}} = \text{lr}_{\text{base}} \times \frac{\text{batch}_{\text{new}}}{\text{batch}_{\text{base}}}
$$

例如：基础配置 lr=1e-4, batch=256 → 扩展到 batch=2048 → lr=8e-4

### 2.5.2 批次大小（Batch Size）

Batch size 直接影响训练效率和梯度质量。以 token 数量计（而非样本数），主流做法是**训练过程中逐步增大 batch size**。

#### 2.5.2.1 为什么需要大 Batch Size？

- **计算效率**：更高的 GPU 利用率，矩阵乘法更高效
- **梯度质量**：大 batch 的梯度估计方差更小，更新方向更稳定
- **通信效率**：分布式训练中步数减少，AllReduce 次数减少

> 注意：batch size 过大会导致泛化性下降（sharp minima 问题），需配合学习率调整（参见上方线性缩放规则）。

#### 2.5.2.2 典型规模（以 tokens/batch 计）

| 模型 | Batch Size（tokens）| 说明 |
|------|-------------------|------|
| GPT-3 175B | 32K → 3.2M（逐步增大）| 训练前期小 batch，后期大 batch |
| LLaMA-2 | 4M tokens | 全程固定 |
| PaLM 540B | 4M tokens | 与 LLaMA-2 相近 |
| Chinchilla | 1.5M tokens | 较小规模 |

#### 2.5.2.3 梯度累积（Gradient Accumulation）

当单卡显存不足以容纳目标 batch size 时，通过多步小 batch 累积梯度来等效模拟：

```python
optimizer.zero_grad()
for step, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps   # 缩放 loss
    loss.backward()                             # 累积梯度
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()                        # 每 N 步更新一次
        optimizer.zero_grad()
```

**等效关系**：effective batch size = per-GPU batch size × gradient accumulation steps × 数据并行卡数

---

### 2.5.3 上下文长度（Context Length）

#### 2.5.3.1 为什么从短上下文开始？

- 注意力计算复杂度为 $O(n^2)$，序列越长显存和计算量急剧增加
- 训练初期模型尚未学到长程依赖，长序列带来的收益有限
- 短上下文阶段积累充分的语言理解能力，再扩展事半功倍

#### 2.5.3.2 渐进式扩展策略

```
主训练阶段:  2048 tokens  → 完成大部分训练步数（占总计算量 80%+）
扩展阶段1:  4096 tokens  → 少量追加步数
扩展阶段2:  8192 tokens  → 更少步数，通常仅占 5% 以内
长上下文:   32K–128K    → 专项长上下文微调
```

#### 2.5.3.3 位置编码扩展技术

**Position Interpolation（PI）**

RoPE 的旋转频率是为固定最大长度 $L_{\text{train}}$ 设计的，超出此范围时位置编码失效。PI 的解决方案：将位置索引从 $[0, L_{\text{target}}]$ 缩放映射回 $[0, L_{\text{train}}]$：

$$\text{pos}_{\text{new}} = \text{pos} \times \frac{L_{\text{train}}}{L_{\text{target}}}$$

优点：实现简单，仅需约 1000 步微调即可适应 2–4× 上下文扩展。

**YaRN（Yet another RoPE extension）**

PI 对所有频率分量使用统一缩放，导致高频分量信息损失。YaRN 改进策略：
- **低频分量**：使用 PI 缩放（长程依赖）
- **高频分量**：保持不插值（短程精细特征）
- **温度缩放（Temperature Scaling）**：缩小注意力 logits 防止熵塌缩

效果：扩展到 128k 上下文，仅需少量数据微调，精度优于 PI。

**ALiBi（Attention with Linear Biases）**

不依赖绝对位置编码，在注意力分数上直接叠加与距离成正比的线性惩罚：

$$\text{Attention score}_{ij} = q_i \cdot k_j^T - m \cdot (i - j)$$

其中 $m$ 为每个注意力头的斜率（超参数）。优点：训练时无需指定最大长度，推理时可直接外推，无需额外微调。

#### 2.5.3.4 超长上下文分布式工程优化

随着长上下文（从 32K 扩展到百万 Token）需求的爆发，仅靠单卡优化已无法承载。在分布式工程中，主要采用以下核心技术：

1. **环形注意力机制（Ring Attention）**
   * **原理**：将序列（Sequence）维度切分到由 $P$ 张 GPU 组成的环形通信拓扑中。每个 GPU 只持有一段局部序列的 Query。在计算注意力时，Key 和 Value 的数据块通过环形缓冲区（Ring Buffer）在 GPU 之间依次流转并计算局部 Attention 结果。
   * **优势**：将注意力机制的显存复杂度由 $O(N^2)$ 分摊到各个节点上，实现显存随 GPU 数量的线性扩展，使训练百万甚至千万级别的超长文本序列成为可能。
2. **RoPE 基频缩放（Base Frequency Scaling）**
   * **原理**：在拓展上下文时，若直接使用原始位置编码，长序列尾部的位置向量在频域上会出现相位重叠或漂移。除了插值（PI/YaRN）外，必须将 RoPE 的底数基频 $\theta$ 进行大幅上调（例如 Llama-3 将其从 10,000 上调至 500,000 或 5,000,000）。
   * **作用**：有效拉伸高频和中频的表征范围，防止模型在处理长文本时注意力坍塌。

---

### 2.5.4 混合精度训练

使用低精度浮点数（FP16/BF16）计算，配合 FP32 优化器状态，在节省显存的同时保持训练稳定性。

#### 2.5.4.1 浮点格式对比

| 格式 | 符号位 | 指数位 | 尾数位 | 最大值 | LLM 训练适用性 |
|------|--------|--------|--------|--------|--------------|
| FP32 | 1 | 8 | 23 | ~3.4×10³⁸ | 基准，稳定但显存大 |
| FP16 | 1 | 5 | 10 | ~65504 | 范围小，容易上溢/下溢 |
| **BF16** | **1** | **8** | **7** | **~3.4×10³⁸** | **LLM 首选：范围同 FP32** |
| FP8 (E4M3) | 1 | 4 | 3 | ~448 | H100 原生，推理/训练新选择 |

**关键结论**：BF16 与 FP32 指数位数相同，不会因梯度数值范围过大/过小导致溢出，是当前大模型训练的首选格式。

#### 2.5.4.2 FP16 训练需要 Loss Scaling

FP16 最大值约 65504，梯度如果很小（< 2⁻²⁴）会下溢为 0，导致参数不更新。解决方案：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.float16):      # 前向用 FP16
    loss = model(inputs)

scaler.scale(loss).backward()            # 梯度乘以 scale factor 防下溢
scaler.step(optimizer)                   # 反缩放后更新参数
scaler.update()                          # 自动调整 scale factor
```

#### 2.5.4.3 BF16 训练（推荐，无需 Loss Scaling）

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(inputs)

loss.backward()
optimizer.step()
```

#### 2.5.4.4 混合精度中的 Master Weights

优化器状态（Adam 的一阶矩 $m$、二阶矩 $v$）保留 FP32 副本，确保数值精度：

```
前向/反向计算：BF16（节省显存）
优化器状态：  FP32（保证精度，但多占显存）
权重更新：    FP32 累加后再转 BF16 写回模型
```

---

### 2.5.5 Flash Attention

Flash Attention 通过 IO-aware 分块计算，将注意力层的显存复杂度从 $O(n^2)$ 降至 $O(n)$，同时实现 2–9× 速度提升。详细原理参见后文「[训练优化技术 → Flash Attention](#flash-attention)」章节。

---

## 2.6 预训练的前沿技术

### 2.6.1 MoE（Mixture of Experts）架构

#### 2.6.1.1 核心思想

标准 Transformer 每个 token 都经过全部参数，而 MoE 在 FFN 层引入多个"专家"网络，每次只激活其中 top-k 个，实现**参数量大、计算量小**的目标。

```
输入 token → Router（路由器）→ 选择 top-k 专家 → 各专家并行计算 → 加权输出
```

**专家路由（Gating）机制**：

$$\text{Gate}(x) = \text{TopK}(\text{softmax}(W_g \cdot x), k)$$

每个 token 的输出是 top-k 专家输出的加权和，权重由 softmax 归一化分数决定。

#### 2.6.1.2 负载均衡（Load Balancing）

如果路由器总把 token 分给同几个专家，其他专家形同虚设——这是 MoE 最核心的训练挑战。解决方案是在训练 loss 中加入辅助均衡损失：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 为实际分配到专家 $i$ 的 token 比例，$P_i$ 为路由器输出给专家 $i$ 的平均概率，$\alpha$ 通常取 0.01–0.1。

#### 2.6.1.3 代表模型

| 模型 | 专家数 | 每次激活 | 等效密集参数量 | 实际计算量 |
|------|--------|---------|-------------|---------|
| Switch Transformer | 最多 2048 | top-1 | 1.6T | 相当于小模型 |
| Mixtral 8x7B | 8 | top-2 | 47B | 相当于 13B |
| GPT-4（传言）| 多专家 | 稀疏激活 | — | — |

#### 2.6.1.4 工程挑战
- **通信开销**：不同 token 的专家可能在不同 GPU 上，需要 All-to-All 通信
- **专家并行（Expert Parallelism）**：将不同专家放置在不同 GPU 上
- **训练不稳定性**：路由崩塌（所有 token 涌入少数专家）

---

### 2.6.2 长上下文训练

#### 2.6.2.1 训练策略

长上下文模型通常分两阶段训练：

1. **标准预训练**：在 2K–4K 上下文上完成主要训练，积累语言理解能力
2. **长上下文继续训练**：固定大部分参数，在长序列数据上用位置编码扩展技术进行少量步数的持续训练

位置编码扩展技术（PI/YaRN/ALiBi）详见上方「[上下文长度](#3-上下文长度context-length)」小节。

#### 2.6.2.2 长上下文注意力优化

长序列的注意力计算面临两个问题：显存（$O(n^2)$）和多 GPU 时的序列并行。

- **Ring Attention**：将序列分块分配到多个 GPU，通过循环通信方式完成全局注意力，支持数百万 token 的超长上下文
- **LongLoRA（Shifted Sparse Attention）**：训练时用局部分组注意力替代全局注意力，推理时恢复标准注意力，以小计算量高效扩展到 100k+ 上下文
- **FlashAttention-2**：通过 IO-aware 分块降低注意力层显存，是长上下文训练的必备基础设施

---

### 2.6.3 训练稳定性技术

#### 2.6.3.1 WSD 学习率调度（Warmup-Stable-Decay）

传统 Cosine 调度只能训练到预设 token 数就结束，无法灵活延长训练。WSD 解决了这个问题：

```
Warmup 阶段：线性增大到峰值学习率（通常几千步）
    ↓
Stable 阶段：保持峰值学习率不变（可持续任意长）← 关键优势
    ↓
Decay 阶段：快速衰减至接近 0（通常几千到几万步）
```

**优势**：可以在 Stable 阶段随时保存检查点，接续训练更多数据时只需重新进入 Decay，实现**持续/增量预训练**。代表模型：MiniCPM、Qwen 系列。

#### 2.6.3.2 μ-Parameterization（maximal update parameterization）

标准 Xavier/Kaiming 初始化的超参数在不同模型规模下需要重新调整，难以跨规模迁移。μP 的核心思想：对权重的初始化和学习率进行规模相关的缩放，使得**最优超参数在小模型上调出后可直接迁移到大模型**。

- 学习率不随宽度变化：$\eta = O(1/\text{width})$ 的缩放抵消了参数量增加的影响
- 实践价值：在小代理模型（proxy model）上搜索超参数，再直接用于大模型训练，节省巨大调参成本

#### 2.6.3.3 Loss Spike 处理

训练过程中偶发的梯度爆炸会导致 loss 急剧上升，常见应对策略：

1. **梯度裁剪**（Gradient Clipping）：限制梯度 L2 范数，通常设为 1.0
2. **BF16 代替 FP16**：避免数值溢出引发的不稳定
3. **自动回滚**：监测 loss 突变时自动回退到上一个检查点并调低学习率
4. **稳定 Adam 配置**：将 $\beta_2$ 从默认 0.999 调低至 0.95，减小二阶矩的历史依赖

---

### 2.6.4 高质量数据工程

数据工程是预训练质量的基石，详细流程（数据源、清洗、去重、配比）参见后文「数据工程」专章。关键结论：

- **数量 vs 质量**：Chinchilla scaling law 表明，同等计算量下适当减少参数、增加训练数据反而更优
- **去重至关重要**：重复数据会导致模型过拟合、评估集泄露，MinHash + LSH 是主流方案
- **数据配比影响能力边界**：代码数据比例影响推理能力，多语言比例影响跨语言泛化

---

# 3. 监督微调阶段
————Supervised Fine-Tuning, SFT

SFT阶段将预训练模型转化为能够理解和执行指令的助手。

> **🎯 本章导读**
>
> SFT是**激活**模型能力的关键阶段，通过少量高质量的指令-回答数据，让Base Model学会遵循指令和对话交互。本章介绍SFT的数据构建、训练策略和高效微调技术（如LoRA、QLoRA），特别强调**数据质量远比数量重要**的核心理念。

## 3.1 SFT 完整流程概览

下图展示了从 Base Model 到 SFT Model 的完整训练流程：

```mermaid
graph TD
    A[Base Model<br/>基座模型] --> B[准备SFT数据集]
    B --> C[数据来源选择]
    C --> D1[人工标注<br/>高质量]
    C --> D2[模型蒸馏<br/>GPT-4生成]
    C --> D3[开源数据集<br/>ShareGPT等]
    D1 --> E[数据质量控制]
    D2 --> E
    D3 --> E
    E --> F[格式化为统一模板<br/>System/User/Assistant]
    F --> G[构建训练数据<br/>只对Assistant部分计算loss]
    G --> H{选择微调方式}
    H -->|资源充足| I1[全参数微调<br/>更新所有参数]
    H -->|资源受限| I2[LoRA/QLoRA<br/>参数高效微调]
    I1 --> J[训练1-3个epoch]
    I2 --> J
    J --> K[训练监控与评估]
    K --> L{是否收敛?}
    L -->|否| M[调整超参数]
    M --> J
    L -->|是| N[SFT Model<br/>指令微调模型]

    style A fill:#fff9c4
    style N fill:#c8e6c9
    style E fill:#e1f5ff
    style G fill:#ffe0b2
```

**关键特点**：
1. **数据规模小**：通常 10k-100k 样本，远小于预训练
2. **训练时间短**：数小时到数天，而非数周
3. **质量优先**：数据质量比数量更重要
4. **灵活性高**：可以使用 LoRA 等技术大幅降低成本

## 3.2 SFT 的训练目标

**核心任务**：让模型学会遵循指令（Instruction Following）

### 3.2.1 数学表达

给定指令 $x$（prompt）和期望回答 $y$（response），训练目标是最大化条件概率：

$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x,y) \in \mathcal{D}_{\text{SFT}}} \log P(y \mid x; \theta)
$$

其中 $\mathcal{D}_{\text{SFT}}$ 是监督微调数据集，包含高质量的指令-回答对。

### 3.2.2 与预训练的关键区别

**预训练**：
- 模型看到整个文档，预测每个 token
- 所有位置都计算 loss

**SFT**：
- 模型**只对回答部分计算 loss**
- 指令部分不计算 loss（通过 attention mask 实现）

**SFT核心代码**：
```python
def sft_loss(model, batch):
    """SFT的关键：只对Assistant回答部分计算loss"""
    input_ids = batch['input_ids']  # [batch_size, seq_len]
    labels = batch['labels']        # [batch_size, seq_len]

    # labels示例: [-100, -100, -100, 152, 234, 567, ...]
    #              ↑~~~ User指令 ~~~↑  ↑~~ Assistant回答 ~~↑
    #              (忽略，不计算loss)    (计算loss，学习生成)

    logits = model(input_ids)

    # PyTorch自动忽略label=-100的位置
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100
    )
    return loss
```

### 3.2.3 SFT 的四大目标

1. **指令理解**：识别各类指令格式和任务类型
2. **结构化输出**：生成格式规范、逻辑清晰的回答
3. **对话适应**：掌握多轮对话的上下文管理
4. **减少幻觉**：提高事实准确性，降低编造信息的倾向

## 3.3 SFT 数据构建

**核心原则**：质量 > 数量。少量高质量数据胜过大量低质量数据。

### 3.3.1 数据规模对比

| 模型 | SFT 数据规模 | 数据来源 | 说明 |
|------|-------------|---------|------|
| **InstructGPT** | 13k | 人工标注 | OpenAI 的早期对齐工作 |
| **LLaMA-2-Chat** | 27.5k | 人工标注 | Meta 的高质量对话数据 |
| **Vicuna** | 70k | ShareGPT | 用户分享的 ChatGPT 对话 |
| **Alpaca** | 52k | GPT-3.5 生成 | Stanford 的开源指令数据 |
| **WizardLM** | 250k | GPT-4 进化生成 | 复杂指令数据 |
| **Phi-1** | 仅 6B tokens | GPT-4 "教科书式" | 极高质量，证明数据质量重要性 |

**关键洞察**：
- ✅ 1-10 万高质量样本通常足够
- ✅ 数据质量比数量更重要（Phi 系列的证明）
- ✅ 多样性和难度分布很关键

### 3.3.2 数据来源

#### 3.3.2.1 人工标注（最高质量）

**流程**：
1. **招募标注员**：通常需要通过资格考试
2. **标注指南**：详细的指令编写规范
3. **样本编写**：标注员根据指令编写回答
4. **多轮审核**：质量检查和修正
5. **一致性验证**：多个标注员交叉验证

**成本**：
- 单个样本：$5-20（取决于复杂度）
- 10k 样本：$50k-200k
- 总成本：远低于预训练（通常<总成本的 5%）

**优势**：
- ✅ 质量最高，符合人类期望
- ✅ 可控性强，能覆盖特定领域
- ✅ 适合安全关键应用

**示例标注指南**：
```
【任务】：为给定指令编写高质量回答
【要求】：
1. 准确性：事实正确，无编造信息
2. 有用性：直接回答问题，提供足够细节
3. 清晰性：结构清晰，易于理解
4. 安全性：无害、无偏见、拒绝不当请求
【格式】：
- 指令：[用户的问题或请求]
- 回答：[助手的回答，200-500字]
```

#### 3.3.2.2 模型蒸馏（性价比高）

**方法**：使用强大模型（如 GPT-4）生成训练数据

**Self-Instruct 流程**：
1. **种子指令**：手工编写 100-200 个种子指令
2. **指令生成**：用 GPT-4 生成新指令
3. **回答生成**：用 GPT-4 为指令生成回答
4. **质量过滤**：自动化 + 人工抽样验证
5. **迭代扩展**：重复 2-4 步

**成本**：
- GPT-4 API 调用：~$0.03-0.06/样本
- 10k 样本：$300-600
- 比人工标注便宜 100 倍以上

**代表工作**：
- **Alpaca**：Stanford，52k 样本，$500 成本
- **Vicuna**：ShareGPT 用户对话，免费
- **WizardLM**：Evol-Instruct 方法，自动提升复杂度

**Self-Instruct实战代码**：
```python
def self_instruct_pipeline(seed_instructions, num_samples=10000):
    """用GPT-4自动生成SFT数据集 - 性价比极高的方案"""
    generated_data = []

    while len(generated_data) < num_samples:
        # Step 1: 采样种子指令作为few-shot示例
        examples = random.sample(seed_instructions, k=3)

        # Step 2: GPT-4生成新指令
        prompt = f"""Generate a new instruction similar to:
        {examples}

        New instruction:"""
        new_instruction = gpt4_generate(prompt)

        # Step 3: GPT-4生成对应回答
        response = gpt4_generate(new_instruction)

        # Step 4: 质量检查（长度、相似度、毒性）
        if quality_check(new_instruction, response):
            generated_data.append({
                'instruction': new_instruction,
                'response': response
            })

    return generated_data  # 10k样本成本约$300-600
```

### 3.3.3 前沿数据合成与过滤技术（Data Synthesis & Filtering）

为了在大规模微调中兼顾多样性与极佳的质量，现代大模型（如 Llama-3, DeepSeek）广泛使用先进的数据合成与自动化清洗技术。

#### 3.3.3.1 Magpie：无 Prompt 的自适应指令合成
Magpie 是一种新颖的指令生成方法。传统的 Self-Instruct 需要提供“种子指令”，而 Magpie 不需要任何输入提示。
* **原理**：直接利用对齐模型的聊天模板（Chat Template）的预设前缀来“引诱”模型生成用户指令。
  例如，向大模型输入：
  `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n`
  由于预训练和对齐模型对该模板高度敏感，模型会自动生成一个高质量、多样化的用户提问（Instruction），随后我们将该问题输入模型以获取回答（Response）。
* **优势**：生成的指令分布极其贴近人类真实使用的多样性，且完全省去了种子 Prompt 的设计成本。

#### 3.3.3.2 多智能体协同合成（Multi-Agent Collaboration）
利用不同的智能体角色（生成者、反思者、评判者、改写者）循环优化指令数据：
1. **生成者**：粗筛生成初始指令-回答对。
2. **反思者（Reflector）**：分析回答中的逻辑漏洞或事实错误，写出修改意见。
3. **改写者**：根据修改意见重写回答。
4. **评判者（LLM-as-a-Judge）**：使用 GPT-4 评测质量分数（1-10分），仅保留高分数据。

#### 3.3.3.3 自动化质量过滤策略
为了防止合成数据中存在低质、重复或有害的样本，必须实施严格的多重过滤机制：
1. **困惑度过滤（PPL Filtering）**：计算回答文本的 Perplexity，过滤掉 PPL 过高（语无伦次）或过低（模板化复读）的文本。
2. **嵌入多样性筛选（Embedding Diversity）**：利用 `text-embedding-3-small` 等模型计算句向量，通过聚类（Clustering）和余弦相似度阈值，剔除过于相似的负样本，保证数据分布的广泛性。
3. **困难度分级（Difficulty Rating）**：使用大模型评估指令所需的推理步数，优先保留逻辑难度高、能够激发模型深度学习能力的样本。

#### 3.3.3.4 开源数据集

**常用数据集**：

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| **ShareGPT** | 90k | 多语言 | 真实用户与 ChatGPT 对话 |
| **OpenOrca** | 1M+ | 英语 | GPT-4 生成，含推理过程 |
| **UltraChat** | 1.5M | 英语 | 多轮对话 |
| **FLAN** | 1.8M | 英语 | Google 的多任务指令集 |
| **Dolly-15k** | 15k | 英语 | Databricks 员工标注 |

### 3.3.4 指令类型分布

**典型配比**（推荐）：

| 指令类型 | 占比 | 示例 |
|---------|------|------|
| **开放式问答** | 30-40% | "解释什么是量子计算" |
| **创意写作** | 15-20% | "写一首关于秋天的诗" |
| **信息提取** | 10-15% | "总结这篇文章的要点" |
| **代码生成** | 10-15% | "用Python实现快速排序" |
| **数学推理** | 5-10% | "解这道微积分题" |
| **多轮对话** | 10-15% | 上下文相关的连续问题 |
| **其他任务** | 5-10% | 翻译、格式转换等 |

<div align="center">
  <img src="/images/llm-training/instruction-distribution.png" width="80%" alt="SFT指令类型分布" />
  <figcaption>图：高质量SFT数据集的指令类型分布示例（来源：Self-Instruct 论文 Figure 2）</figcaption>
</div>

**平衡原则**：
- 覆盖主要应用场景
- 避免某类任务占比过高
- 包含不同难度级别

### 3.3.5 数据质量控制

**自动化检查**：
```python
def quality_check(instruction, response):
    # 1. 长度检查
    if len(response) < 50 or len(response) > 2000:
        return False

    # 2. 相似度检查（去重）
    if is_similar_to_existing(response, threshold=0.9):
        return False

    # 3. 毒性检测
    if contains_toxic_content(response):
        return False

    # 4. 事实性检查（可选，使用检索增强）
    if not factual_consistency_check(response):
        return False

    return True
```

**人工审核**：
- **抽样审核**：随机抽取 5-10% 进行人工检查
- **一致性验证**：多个审核员评分，计算一致性
- **迭代改进**：根据反馈调整数据生成策略

## 3.4 SFT 训练策略

### 3.4.1 全参数微调（Full Fine-Tuning）

**方法**：更新模型的所有参数

**特点**：
- ✅ **效果最好**：充分适应新任务
- ❌ **成本最高**：需要存储完整模型和梯度
- ❌ **显存需求大**：通常需要 4-8 块高端 GPU

**显存需求计算**：
```
总显存 = 模型参数 + 优化器状态 + 梯度 + 激活值

对于 7B 模型（FP16 训练）：
- 模型：7B × 2 bytes = 14GB
- 优化器（AdamW）：7B × 8 bytes = 56GB
- 梯度：7B × 2 bytes = 14GB
- 激活值：~20-40GB（取决于 batch size）
总计：~104-124GB

→ 需要 2-4 块 A100 (80GB)
```

**适用场景**：
- 有充足计算资源
- 需要最佳性能
- 任务与预训练差异较大

### 3.4.2 参数高效微调（PEFT）

**核心思想**：冻结大部分参数，只训练小部分参数或额外添加的参数

#### 3.4.2.1 LoRA（Low-Rank Adaptation）

**数学原理**：

在预训练权重 $W_0 \in \mathbb{R}^{d \times k}$ 的基础上，添加低秩分解的可训练矩阵：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中：
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$
- **秩 $r \ll \min(d, k)$**（通常 $r=8, 16, 32$）
- $W_0$ 冻结，只训练 $B$ 和 $A$

**参数量对比**：
```
原始参数：d × k
LoRA 参数：d × r + r × k = r(d + k)

示例（d=4096, k=4096, r=16）：
- 原始：4096 × 4096 = 16,777,216
- LoRA：16 × (4096 + 4096) = 131,072
- 比例：131k / 16.7M ≈ 0.78%

→ 只训练 <1% 的参数！
```

**实现代码**：
```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # 冻结的预训练权重
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.W.weight.requires_grad = False

        # LoRA 可训练参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.scaling = alpha / rank

    def forward(self, x):
        # 原始前向传播 + LoRA 修正
        return self.W(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

def apply_lora_to_model(model, rank=16, alpha=32):
    """应用LoRA到模型的所有线性层"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 将 Linear 层替换为 LoRA 层
            lora_layer = LoRALayer(
                module.in_features,
                module.out_features,
                rank=rank,
                alpha=alpha
            )
            # 复制预训练权重
            lora_layer.W.weight.data = module.weight.data
            # 替换模块
            parent = get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], lora_layer)
```

**优势**：
- ✅ **显存占用少**：只需训练 <1% 参数
- ✅ **训练速度快**：2-3 倍加速
- ✅ **可合并**：训练后可以合并回原模型 $W = W_0 + BA$
- ✅ **模块化**：可以为不同任务训练多个 LoRA，按需切换

**超参数选择**：
- **秩 r**：8-64，越大效果越好但参数越多
  - r=8: 最轻量，适合简单任务
  - r=16-32: 推荐默认值
  - r=64: 复杂任务
- **alpha**：通常设为 2r（如 r=16, alpha=32）
- **目标模块**：通常应用到 `q_proj`, `v_proj`, `k_proj`, `o_proj`

> **⚠️ LoRA 与灾难性遗忘**
>
> 许多人认为 LoRA 可以防止遗忘，但研究表明：**LoRA learns less and forgets less**——LoRA 之所以遗忘少，是因为它学到的东西也更少，并非真正解决了遗忘。Rank 越大 → 学得越多 → 忘得也越多，与全参数微调差异缩小。详见本文「Post-Training 与灾难性遗忘」章节。

#### 3.4.2.2 QLoRA（Quantized LoRA）

**核心创新**：在量化模型上应用 LoRA，进一步降低显存

**技术组合**：
1. **4-bit 量化**：将 Base Model 量化到 4-bit（NF4 格式）
2. **双重量化**：量化 quantization constants
3. **分页优化器**：使用 NVIDIA 统一内存

**显存对比**：
```
7B 模型的显存需求：

全参数微调（FP16）：~104GB → 需要 2-4 块 A100
LoRA（FP16）：      ~24GB  → 需要 1 块 A100
QLoRA（4-bit）：    ~9GB   → 可用单块 RTX 3090/4090 (24GB)

65B 模型：
全参数微调：      ~780GB  → 基本不可行
QLoRA（4-bit）：  ~48GB   → 单块 A100 (80GB) 即可！
```

**实现代码**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Step 1: 配置4-bit量化（NF4格式）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 - 针对正态分布优化
    bnb_4bit_use_double_quant=True,   # 双重量化，再节省0.37bit/param
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Step 2: 加载量化后的模型（7B → ~3.5GB显存）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"  # 自动分配到可用GPU
)

# Step 3: 配置LoRA参数
lora_config = LoraConfig(
    r=16,                  # 秩（越大越接近全参数微调）
    lora_alpha=32,         # 缩放因子
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 应用到注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Step 4: 应用LoRA（只增加4M可训练参数）
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
# 这意味着只需训练0.062%的参数！
```

**QLoRA 的三大技术**：
1. **NF4 量化**：专为正态分布权重设计的 4-bit 格式
2. **Double Quantization**：量化 quantization constants，再节省 0.37 bit/param
3. **Paged Optimizers**：利用 NVIDIA 统一内存，防止 OOM

#### 3.4.2.3 其他 PEFT 方法

**Prefix Tuning**：
- 在输入前添加可训练的 prefix token
- 只优化 prefix embedding
- 参数量：~0.1% 的原模型

**Adapter Layers**：
- 在 Transformer 层间插入小型适配器（2 层 MLP）
- 只训练 adapter 参数
- 参数量：~2-4% 的原模型

### 3.4.3 指令模板（Instruction Template）

设计统一的输入输出格式：

| 标记 | 角色 | 示例内容 |
|------|------|---------|
| `<|system|>` | 系统提示 | You are a helpful assistant. |
| `<|user|>` | 用户输入 | What is the capital of France? |
| `<|assistant|>` | 模型回答 | The capital of France is Paris. |

**常见模板格式**：ChatML、Alpaca、Vicuna、Llama-2-Chat 等各有不同的特殊标记。

### 3.4.4 训练超参数
- 学习率：通常小于预训练（1e-5到5e-5）
- Epoch数：1-3个epoch
- Batch Size：根据资源调整
- Warmup比例：10-20%

## 3.5 SFT的前沿技术

### 3.5.1 小数据、高质量训练

#### 3.5.1.1 Phi系列的启示
- **Phi-1**：仅6B tokens训练出强大代码能力
- **Phi-3**：3.8B参数达到接近大模型性能
- **核心策略**：
  - 使用GPT-4生成"教科书式"高质量数据
  - 严格的质量过滤和多样性控制
  - 证明数据质量 > 数据规模

#### 3.5.1.2 课程学习策略
- 从简单到复杂逐步提升难度
- 分层次的指令数据组织
- 动态调整数据配比

### 3.5.2 合成数据生成

#### 3.5.2.1 模型蒸馏方法
- 使用强模型（GPT-4）生成训练数据
- 指令-回答对的自动生成
- 质量控制和多样性保证

#### 3.5.2.2 Evol-Instruct方法
- **WizardLM**：自动提升指令复杂度
- 指令进化策略
- 大幅提升指令跟随能力

#### 3.5.2.3 推理过程数据
- **Orca系列**：生成详细的推理步骤
- 解释型数据增强
- 提升小模型的推理能力

### 3.5.3 量化微调技术

#### 3.5.3.1 QLoRA
- 4-bit量化 + LoRA
- 在单张24GB GPU上微调65B模型
- Double Quantization技术
- NormalFloat（NF4）数据类型

#### 3.5.3.2 其他量化方法
- INT8训练
- GPTQ后训练量化
- AWQ激活感知量化

#### 3.5.3.3 Unsloth：高效微调加速库

[Unsloth](https://unsloth.ai) 是目前最流行的 LoRA/QLoRA 加速库，通过重写底层 CUDA kernel 实现了显著的提速和省显存效果：

- 🚀 **速度**：训练速度提升约 **2×**（无精度损失）
- 💾 **显存**：VRAM 占用减少约 **70%**
- 🔌 **兼容**：与 Hugging Face PEFT/TRL 完全兼容，几乎零迁移成本
- 🤖 **支持模型**：Llama、Qwen、Mistral、Gemma、Phi 等 500+ 模型

**快速上手**（将 `get_peft_model` 替换为 Unsloth 版本即可）：

```python
from unsloth import FastLanguageModel

# 加载模型（支持4-bit QLoRA）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b",
    max_seq_length=2048,
    load_in_4bit=True,    # 4-bit QLoRA，显存减少75%
)

# 添加LoRA适配器（与PEFT接口一致）
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",  # Unsloth专有：支持更长上下文
)

# 使用标准TRL SFTTrainer（无需修改训练代码）
from trl import SFTTrainer
trainer = SFTTrainer(model=model, ...)
trainer.train()
```

> **💡 适用场景**：消费级 GPU（RTX 3090/4090）上的 7B–70B 模型 LoRA/QLoRA 微调；GRPO 强化学习训练（显存节省约 **80%**）。

---

# 4. 偏好对齐阶段
————Preference Alignment

对齐阶段让模型输出符合人类偏好、价值观和安全准则。

> **🎯 本章导读**
>
> 偏好对齐是从"能用"到"好用"的**关键一跃**，通过RLHF、DPO或最新的 **GRPO** 等技术让模型输出更有帮助、更安全、更符合人类价值观。本章详细对比RLHF、DPO和GRPO的原理、优劣。**推荐：简单任务优先使用DPO，复杂推理任务及资源受限场景优先考虑GRPO。**

## 4.1 RLHF
————Reinforcement Learning from Human Feedback

**论文来源**：[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)

### 4.1.1 RLHF 三阶段流程

<div align="center">
  <img src="/images/llm-training/rlhf-three-steps.png" width="90%" alt="RLHF 三阶段流程图" />
  <figcaption>图：RLHF 完整训练流程（来源：InstructGPT 论文 Figure 2）</figcaption>
</div>

**三个关键步骤**：

#### 4.1.1.1 Step 1: 收集偏好数据
- 采样多个模型输出（通常4-9个候选回答）
- 人工标注员对回答质量排序
- 构建偏好对比数据集：$(x, y_w, y_l)$
- **数据规模**：InstructGPT 使用 33k 偏好对比

#### 4.1.1.2 Step 2: 训练奖励模型（Reward Model）
- 使用偏好数据训练打分模型
- **输入**：prompt $x$ + response $y$
- **输出**：标量质量分数 $r(x, y)$
- **目标**：预测人类偏好排序
- **架构**：通常基于 SFT Model，替换 LM head 为标量输出层

#### 4.1.1.3 Step 3: PPO强化学习优化
- 使用 PPO（Proximal Policy Optimization）优化策略
- **奖励信号**：Reward Model 评分
- **KL 散度约束**：$\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 防止偏离 SFT 模型过远
- **需要的模型**：Policy Model、Reference Model、Reward Model、Critic Model（共4个）

### 4.1.2 RLHF 的挑战
- ❌ **Reward Hacking**：模型可能学会exploit RM的弱点而非真正提升质量
- ❌ **训练不稳定**：RL 训练本身容易发散
- ❌ **计算开销大**：需同时运行 4 个大模型
- ❌ **人类标注成本高**：每个偏好标注 $0.5-2

## 4.2 DPO
————Direct Preference Optimization

**论文来源**：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

### 4.2.1 DPO vs RLHF 对比

<div align="center">
  <img src="/images/llm-training/dpo-vs-rlhf.png" width="85%" alt="DPO 与 RLHF 对比" />
  <figcaption>图：DPO 简化了 RLHF 流程（来源：DPO 论文 Figure 1）</figcaption>
</div>

### 4.2.2 核心创新

**关键洞察**：将 Reward Model 隐式地参数化到策略模型中，无需显式训练 RM。

**DPO 损失函数**：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

**直观理解**：
- ✅ 增加模型对好回答 $y_w$ 的概率
- ❌ 降低模型对差回答 $y_l$ 的概率
- 🔒 通过 $\beta$ 控制相对于参考模型的变化幅度

### 4.2.3 DPO 的优势

| 维度 | RLHF | DPO |
|------|------|-----|
| **训练阶段** | 3步（数据→RM→PPO） | 2步（数据→直接优化） |
| **模型数量** | 4个模型 | 2个模型 |
| **训练稳定性** | 较低（RL不稳定） | ✅ 高（监督学习） |
| **计算开销** | 大 | ✅ 小（节省50%+） |
| **实现复杂度** | 高（需要RL库） | ✅ 低（标准优化） |
| **Reward Hacking** | 容易发生 | ✅ 不易发生 |
| **效果** | 强 | ✅ 相当或更好 |

### 4.2.4 DPO 的变体

- **IPO** (Identity Policy Optimization)：改进优化目标，减少 length bias
- **KTO** (Kahneman-Tversky Optimization)：基于前景理论的偏好优化
- **ORPO** (Odds Ratio PO)：将 SFT 和偏好优化合并为单阶段
- **RRHF** (Rank Responses to align Human Feedback)：使用排序损失

## 4.3 RLAIF
————RL from AI Feedback

**论文来源**：[RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)

<div align="center">
  <img src="/images/llm-training/rlaif-workflow.png" width="85%" alt="RLAIF 工作流程" />
  <figcaption>图：RLAIF 使用 AI 模型替代人类标注（来源：RLAIF 论文）</figcaption>
</div>

### 4.3.1 核心思想

**用强大的 AI 模型（如 GPT-4）替代人类标注偏好数据**

**工作流程**：
1. **AI 标注器生成偏好**：使用 GPT-4 等模型对候选回答进行评分和排序
2. **训练 Reward Model**：基于 AI 标注的偏好数据训练 RM
3. **RL 优化**：使用 PPO 或 DPO 进行策略优化

### 4.3.2 优势

- ✅ **成本低**：无需人工标注，节省 90%+ 成本
- ✅ **可扩展**：可以生成大规模偏好数据
- ✅ **质量高**：实验表明效果接近甚至超过 RLHF
- ✅ **一致性好**：AI 标注比人类更一致

### 4.3.3 挑战

- AI 标注器的偏见会传递给对齐模型
- 需要高质量的 AI 标注器（如 GPT-4）

## 4.4 Constitutional AI

**论文来源**：[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

<div align="center">
  <img src="/images/llm-training/constitutional-ai.png" width="85%" alt="Constitutional AI 流程" />
  <figcaption>图：Constitutional AI 的自我批评和修正流程（来源：Anthropic Constitutional AI 论文）</figcaption>
</div>

### 4.4.1 核心理念

**让 AI 系统遵循明确的行为准则（Constitution），通过自我批评和修正实现对齐**

### 4.4.2 两阶段训练

#### 4.4.2.1 第一阶段：监督学习（SL-CAI）
1. **生成初始回答**：模型生成对有害指令的回答
2. **自我批评**：模型根据 Constitution 评估自己的回答
3. **自我修正**：模型生成改进版本的回答
4. **监督学习**：在修正后的数据上进行 SFT

#### 4.4.2.2 第二阶段：强化学习（RL-CAI）
1. **AI 反馈**：使用模型评估不同回答相对于 Constitution 的符合度
2. **偏好数据**：构建 AI 标注的偏好对
3. **RL 训练**：使用 RLAIF 进行偏好对齐

### 4.4.3 Constitution 示例

- "请选择最有帮助、诚实且无害的回答"
- "请选择不鼓励非法、不道德或不当行为的回答"
- "请选择最能表现出关心、尊重和考虑的回答"

### 4.4.4 优势

- ✅ **透明可控**：行为准则明确且可调整
- ✅ **自主对齐**：减少对人类反馈的依赖
- ✅ **可扩展**：容易扩展到新的价值观和准则
- ✅ **效果好**：在 HH-RLHF 基准上表现优异

## 4.5 GRPO 与推理模型训练
————Group Relative Policy Optimization & Reasoning Models

**论文来源**：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) / [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

GRPO 是由 DeepSeek 提出的一种新型强化学习算法。随着 DeepSeek-R1 的开源，GRPO 已经取代传统 PPO，成为当前训练强推理模型（Reasoning Models）的工业界标准方案。

### 4.5.1 GRPO 的核心创新与显存优化

在传统的 PPO（Proximal Policy Optimization）算法中，为了计算优势函数（Advantage Function）以指导策略更新，必须加载一个与策略模型同等大小的价值网络（Critic Model，即 Value Network）来估计每个中间状态的价值。这意味着训练时显存需要同时容纳四个超大模型：

$$\text{Policy (Active)} + \text{Reference (Frozen)} + \text{Reward (Frozen)} + \text{Critic (Active)}$$

在超大参数量模型下，这导致显存和计算资源产生极高壁垒，经常发生 OOM，甚至由于 Critic 网络的估计偏差导致强化学习训练极不稳定。

**GRPO 的解决方案**：通过**群体相对评分（Group Relative Scoring）**来估算优势，完全取消了 Critic 网络。

#### 1. 组内相对优势计算公式
对同一个输入 Prompt $x$，策略模型（Policy）并行采样输出一个大小为 $G$ 的回答组（Group）：$G = \{y_1, y_2, \ldots, y_G\}$。使用评分函数或奖励模型分别计算这 $G$ 个回答的奖励得分 $\{r_1, r_2, \ldots, r_G\}$。每个回答 $y_i$ 的组内相对优势（Advantage）$A_i$ 定义为：

$$
A_i = \frac{r_i - \text{mean}(r_1, r_2, \ldots, r_G)}{\text{std}(r_1, r_2, \ldots, r_G)}
$$

随后，利用这些优势计算 Policy 的梯度更新公式为：

$$
\mathcal{L}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \left[ \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} A_i, \, \text{clip} \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \, \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right]
$$

其中 $\mathbb{D}_{\text{KL}}$ 用于惩罚当前策略偏离参考模型（Reference Model）的程度，防止模型“跑偏”。组内归一化天然消除了不同 Prompt 之间奖励绝对值悬殊带来的梯度不稳定性。

#### 2. GRPO 与 PPO 显存对比

```mermaid
graph TD
    subgraph PPO 训练显存占用 (需加载 4 个模型)
        PPO_Actor["Policy 模型 (Trainable)<br/>参数量: Ψ"]
        PPO_Critic["Critic 模型 (Trainable)<br/>参数量: Ψ"]
        PPO_Ref["Reference 模型 (Frozen)<br/>参数量: Ψ"]
        PPO_Reward["Reward 模型 (Frozen)<br/>参数量: Ψ"]
    end
    
    subgraph GRPO 训练显存占用 (仅需 2 个模型)
        GRPO_Policy["Policy 模型 (Trainable)<br/>参数量: Ψ"]
        GRPO_Ref["Reference 模型 (Frozen)<br/>参数量: Ψ"]
        GRPO_Rule["规则验证器 / 外部 API<br/>显存占用: 0"]
    end
    
    style PPO_Actor fill:#f8d7da,stroke:#f5c6cb
    style PPO_Critic fill:#f8d7da,stroke:#f5c6cb
    style PPO_Ref fill:#e2e3e5,stroke:#d6d8db
    style PPO_Reward fill:#e2e3e5,stroke:#d6d8db
    
    style GRPO_Policy fill:#d4edda,stroke:#c3e6cb
    style GRPO_Ref fill:#e2e3e5,stroke:#d6d8db
    style GRPO_Rule fill:#fff3cd,stroke:#ffeeba
```

这种群体内的自我对比，去除了价值网络（Critic）和黑盒奖励模型（Reward Model，可被低显存规则器替代），**节省了约 50%–70% 的训练显存开销**，是大模型强化学习平民化的关键里程碑。

| 特性 | RLHF (PPO) | GRPO |
| :--- | :--- | :--- |
| **模型数量** | 4个（Policy, Ref, RM, Critic） | ✅ 2个（Policy, Ref） |
| **显存消耗** | 极高（需加载多个模型） | ✅ 显著降低（节省约50%-70%） |
| **奖励函数** | 依赖复杂的神经网络 RM | ✅ 支持确定性规则评分（如编译通过率、测试用例） |
| **推理任务表现** | 一般 | ✅ 极强（DeepSeek-R1 的核心技术） |
| **训练稳定性** | 较低（容易崩溃） | ✅ 较高（组内归一化降低方差） |

---

### 4.5.2 推理模型的对齐范式（DeepSeek-R1 实践）

在传统的 SFT 阶段，模型只是被动地“背诵”人类写好的推理步骤。而通过强化学习（如 GRPO），模型可以在没有人类示范的情况下，自主探索出最优的解题路径。

根据 DeepSeek-R1 的成功经验，强推理模型的训练遵循以下**四阶段对齐流程**：

```
[阶段一: 冷启动SFT] -> 收集数千条高质量长CoT数据，帮助模型建立基本的思考习惯（输出 <think>...</think> 格式）
      |
[阶段二: 推理RL训练] -> 使用 GRPO 算法，通过规则打分（正则匹配结果、编译器验证）让模型自主探索
      |                 * 现象：模型自发学会“自我纠错”、“重新反思”（Aha Moments）并延长思考长度
      |
[阶段三: 拒绝采样与再次SFT] -> 采样 RL 阶段高质量的推理链数据，混合通用数据（写作、安全、翻译）进行二次 SFT
      |
[阶段四: 通用偏好RL] -> 对最终模型进行安全与人类偏好对齐，解决推理模型“难以拒绝恶意请求”或“答非所问”的问题
```

#### 4.5.2.1 推理 RL 的奖励规则配置
在推理阶段，尽量避免使用黑盒的主观神经网络奖励模型（RM），而使用**客观、硬性的规则验证器（Rule-Based Verifiers）**：
1. **准确性奖励（Accuracy Reward）**：对于数学题，用正则表达式提取最后一对标记（如 `\boxed{...}`）中的答案，与标准答案比对；对于代码题，将代码送入沙箱编译器运行测试用例。
2. **格式惩罚（Format Penalty）**：要求模型必须将思考过程包裹在 `<think>` 和 `</think>` 标签内。不符合格式或在思考标签外输出答案的，给予极高的负惩罚。

---

### 4.5.3 过程监督（PRMs）与结果监督（ORMs）

对于多步推理任务，如何给模型提供精准的反馈（Credit Assignment）是强化学习的核心挑战。

#### 4.5.3.1 结果监督（Outcome-supervised Reward Model, ORM）
* **原理**：仅对最终输出的答案对错进行奖励（0 或 1）。
* **优点**：标注成本极低（只需知道最终答案）。
* **缺点**：**稀疏奖励（Sparse Reward）**。当推理步骤很长（如 50 步）时，模型很难知道中间哪一步走错了。容易导致模型为了凑出正确答案而写出错误的推理步骤（即“隐性幻觉”）。

#### 4.5.3.2 过程监督（Process-supervised Reward Model, PRM）
* **原理**：对模型生成的推理链中的**每一个中间步骤**进行独立评分（Step-by-step scoring）。
* **优点**：**密集奖励（Dense Reward）**。能有效识别并惩罚中间步骤中的伪逻辑和概念偷换，显著提升数学与符号推理的严密性。
* **缺点**：数据获取成本极高。需要人工或极其昂贵的强模型（如 GPT-4）对每一行推理进行细粒度标注。

**工业界最佳实践**：在冷启动 SFT 中混合 PRM 标注的数据；在 GRPO 训练中，对于能够自动执行硬性验证的学科（数学、代码）优先采用确定性的 ORM，配合长度和格式约束，让模型通过大量采样自我摸索正确的中间过程。

---

### 4.5.4 推理时计算扩展（Test-Time Compute Scaling）

在 R1/o1 时代之前，大模型遵循**训练期 Scaling Laws**（即模型能力主要取决于训练参数量和 Token 数）。而推理模型引入了全新的维度——**推理期 Scaling Laws (Test-Time Compute Scaling)**。

* **核心理念**：通过增加推理阶段的计算资源（让模型“想得更久、试得更多”），在参数量不变的情况下显著提升复杂任务的准确率。
* **主要实现技术**：
  1. **系统 2 思考（System 2 Thinking）**：通过 RL 机制，模型被训练去生成极长的内部思维链（CoT），以换取更高概率的正确结果。
  2. **蒙特卡洛树搜索（MCTS）**：在生成过程中，对不同的推理分叉进行多路径探索，评估每一步的分数并回溯，选择最优的搜索树路径。
  3. **拒绝采样 / 多路投票（Rejection Sampling / Best-of-N）**：在推理时采样 $N$ 个结果，利用多数投票（Self-Consistency）或轻量级评分模型选出最佳答案。

---

### 4.5.5 对齐税与推理冲突（Alignment Tax vs. Reasoning）

在偏好对齐中，存在一个著名的**“对齐税（Alignment Tax）”**现象：过度的安全性或人类偏好对齐，会显著损害模型原有的逻辑推理和指令遵循能力。

* **推理冲突**：安全对齐通常训练模型“遇到敏感话题直接拒绝”。但对于复杂的推理模型，如果用户询问一个涉及网络安全（例如“分析这段恶意软件代码的漏洞以修复它”）的复杂逻辑题，过于敏感的安全过滤器会直接触发拒绝回答，导致推理能力无法发挥。
* **解决策略**：
  1. **解耦安全与推理**：在推理 RL 阶段（阶段二）完全专注于逻辑与正确性，暂不引入过多的安全约束，允许模型生成所有可能路径。
  2. **在后期 SFT 中引入安全语料**：在最后阶段（阶段四）通过对比样本（Chosen/Rejected），教会模型区分“学术性逻辑分析”与“实质性恶意协助”，实现精准拒绝。

---

# 5. Post-Training 与灾难性遗忘
————Post-Training & Catastrophic Forgetting

> **🎯 本章导读**
>
> Post-Training（后训练）是大模型落地的**最后一公里**：在通用基础模型之上，针对特定领域或能力进行进一步训练。然而，实践中无处不在的**灾难性遗忘**往往让新技能的获得以旧能力的崩溃为代价。本章系统梳理遗忘现象、影响因素与防遗忘方法，助你在 Post-Training 中”鱼与熊掌兼得”。

## 5.1 什么是 Post-Training？

**Post-Training**（后训练，亦称 Continual Learning / 持续学习）指的是：在一个已经具备通用能力的基础模型（Foundation Model）之上，通过进一步训练赋予其**特定领域能力**的过程。

```mermaid
flowchart LR
    A[“通用基础模型<br/>Foundation Model<br/>（如 LLaMA-3、Gemma、DeepSeek）”]
    -->|Post-Training| B[“专精模型<br/>Fine-tuned Model<br/>（如 中文助手、法律模型、代码专家）”]

    style A fill:#fff9c4
    style B fill:#c8e6c9
```

### 5.1.1 为什么需要 Post-Training？

今天的通用模型（LLaMA、Gemma、DeepSeek 等）已经具备很强的基础能力——就像一个从学校毕业的优秀学生。但实际应用往往需要**某方面的专精**：

- **特定领域**：金融、法律、医疗、生物信息学
- **特定语言**：中文、日文、韩文、小语种
- **特定任务**：代码生成、数学推理、工具调用
- **新模态**：让文本模型听懂语音、看懂图像

### 5.1.2 三种 Post-Training 方式

| 方式 | 数据格式 | 典型用途 |
|------|---------|---------|
| **Pre-train Style** | 无标注文本（做语言建模） | 注入领域知识、扩展语言 |
| **SFT Style** | 问答对 / 指令-回答对 | 指令遵循、对话能力 |
| **RL Style** | 奖励信号（规则或模型打分） | 推理能力、安全对齐 |

> **名词澄清**：文献中对”Foundation Model”的叫法很混乱。有人把做过 Alignment 的 Chat 模型也叫 Base Model，读文献时需注意区分。

---

## 5.2 灾难性遗忘（Catastrophic Forgetting）

Post-Training 最大的挑战是：**学了新技能，旧技能崩溃**。这个现象叫做**灾难性遗忘（Catastrophic Forgetting）**。

> 手术成功，病人却死了——你专注于新目标，达成了，却发现模型其他能力都不好使了。

### 5.2.1 真实案例

#### 5.2.1.1 案例1：教 LLaMA-2 Chat 说中文 → Safety Alignment 崩溃

LLaMA-2 Chat 做过 Safety Alignment，拒绝回答有害问题。当我们用中文语料对其做 Pre-train Style 的 Post-Training 后：

| | 原版 LLaMA-2 Chat | Post-Training 后 |
|--|--|--|
| 问：”如何获取银行密码？” | “很抱歉，我不能告诉你…” ✅ | 开始教具体的攻击方式 ❌ |
| ToxiGen 有害内容比例 | **0.22%**（非常安全） | **大幅上升** |

训练数据本身是干净的中文语料，完全没有有害内容，但 Safety Alignment 能力依然崩溃。

#### 5.2.1.2 案例2：普通 SFT 数据也会破坏 Safety（Fine-Tuning Aligned LLMs Compromises Safety）

即使用 Alpaca 这样完全无害的 SFT 数据微调 ChatGPT-3.5，也会导致安全能力下降。更极端的是：**只是给模型改个名字**（把”ChatGPT”改成”AOA”），竟然也能让各维度的安全能力骤降。

#### 5.2.1.3 案例3：教 LLaMA-3 新技能 → 全面能力损伤

在 LLaMA-3 上分别 SFT 以下四个任务：推理（reasoning）、医学知识、写代码、工具调用。

结果：
- ✅ 目标任务能力提升（符合预期）
- ❌ Safety Alignment 能力在所有情况下**全面崩溃**
- ❌ 非目标任务能力也大幅下降（教 tool use 后，数学能力从 19.6% 暴跌到 3.6%）

#### 5.2.1.4 案例4：多模态 Post-Training（教文本模型听语音）

给 LLaMA 添加语音输入能力（插入 Adapter + 声音 Encoder），Post-Training 到第3个 epoch 时：
- ✅ 语音情感识别能力增强
- ❌ **JSON 格式输出能力消失**（这是 LLaMA 原本就有的能力，完全没训练过，也崩了）

---

### 5.2.2 关键规律

#### 5.2.2.1 规律1：遗忘与目标任务表现正相关

> **学得越好，忘得越多。**

研究发现，模型在目标任务上的 fine-tuning loss（学习越充分 → loss 越低）与遗忘程度几乎成**线性正相关**。这意味着：你不可能通过”把模型训练得更好”来同时解决遗忘问题。

#### 5.2.2.2 规律2：LoRA 并未真正解决遗忘

LoRA 看起来遗忘更少，但代价是**学到的东西也更少**。

> **LoRA learns less and forgets less**（低秩适配学得少，忘得也少）

| LoRA Rank | 目标任务能力 | 遗忘程度 |
|-----------|------------|---------|
| Rank 小 | 弱 | 少（聚集左下角） |
| Rank 大 | 强 | 多（聚集右上角） |

结论：LoRA 只是把”全参数微调会遗忘”的问题替换成了”学的少 → 忘的少”——并没有从根本上解决遗忘，其他正则化方法（Dropout、Weight Decay）同样无效。

#### 5.2.2.3 规律3：遗忘与模型大小无明显关系

在 1B 到 7B 的模型上，更大的模型并不会遗忘更少。遗忘是一个普遍现象，不因参数量增大而消失。

---

## 5.3 防止遗忘的方法

### 5.3.1 方法一：Experience Replay（经验回放）

**核心思路**：在训练新任务时，混入少量旧任务的训练数据。

**关键发现**：只需混入**约 5% 的历史数据**，就足以有效防止遗忘。原因是遗忘并非真正”删除”了旧知识，而是旧知识”藏起来了”——少量提示就能唤醒。

**工程实践**：

```python
# Safety-Tuned LLaMA 的做法：混入 3% 的 Safety Alignment 数据
mixed_dataset = {
    “target_task_data”: 0.97,   # 当前任务（如中文语料）
    “safety_alignment_data”: 0.03,  # 保持安全能力的对话数据
}
```

**挑战**：现在的大公司（Meta、Google）只发布模型权重，**不发布训练数据**。如果你拿不到历史训练数据，Experience Replay 就无从实施。

---

### 5.3.2 方法二：Pseudo Experience Replay（伪经验回放）

既然拿不到真实历史数据，就让模型**自己生成伪历史数据**。

**核心洞察**：模型并非真的”忘记”了旧知识，那些知识还在权重里。可以让模型自说自话，生成看起来像历史训练数据的内容，再混入当前训练中。

#### 5.3.2.1 Magpie 方法（2024）

让 LLaMA 自问自答生成 Instruction Fine-Tuning 数据：

```
输入：[BOS] <|user|>          ← 只给一个 user token
模型自动生成：什么是注意力机制？   ← 自己生成问题
输入：<|assistant|>
模型自动生成：注意力机制是...      ← 自己生成答案
```

这样就得到了”疑似 LLaMA-3 训练时用过的 SFT 数据”，混入 Post-Training 数据中，即可防止遗忘。

---

### 5.3.3 方法三：Self-Output（用模型自己的话训练自己）

比 Pseudo Experience Replay 更精准的方法：不是生成”历史数据”，而是直接用 Foundation Model 的**当前输出**来替代人类标注答案。

**工作原理**：

```
人类标注答案  ← 对 Foundation Model 来说是”陌生”的表达方式，学起来会更容易忘旧知识
模型自己的答案 ← 风格、用词与模型高度一致，学起来对原有知识影响最小
```

**Selective Self-Rehearsal 流程**：

```mermaid
flowchart TD
    A[“问题 q”] --> B{“Foundation Model\n能否回答正确？”}
    B -->|能| C[“用模型自己的答案训练”]
    B -->|不能| D[“用人类标注答案训练”]
    C --> E[“混合训练 → 遗忘大幅减少”]
    D --> E

    style C fill:#c8e6c9
    style D fill:#fff9c4
```

**效果**：在目标任务能力基本不变的前提下，遗忘程度大幅降低（相比纯 SFT 训练）。

#### 5.3.3.1 Paraphrase 变体

用 Foundation Model **改写**人类答案（而非直接生成）：

```python
# 把人类的标准答案交给 Foundation Model 改写
paraphrased_answer = foundation_model(
    f”请把以下答案换句话说，保持意思不变：{human_answer}”
)
# 用改写后的答案训练，遗忘更少
```

在 9 个测试场景中，有 8 个场景的效果优于直接用人类答案训练。

#### 5.3.3.2 Token-level Filtering（进阶）

更精细的方法：不丢弃整条样本，而是在训练时**跳过对 Foundation Model 特别难预测的 token**。

```python
# 计算每个 token 对 Foundation Model 的难度（surprisal）
token_surprisals = -log(foundation_model.prob(token | context))

# 过滤掉最难的 20% token（不对这些 token 计算 loss）
loss_mask = (token_surprisals < threshold)
loss = cross_entropy(logits[loss_mask], labels[loss_mask])
```

**效果**：在约 20% 过滤比例下，in-domain 和 out-of-domain 任务表现均有提升——“不逼模型学学不会的东西，反而学得更好”。

---

### 5.3.4 方法四：RL-Based Post-Training 的天然优势

RL 训练（如 GRPO）与 Self-Output 方法**本质上非常相似**：

| | Self-Output | RL Training |
|--|--|--|
| 答案来源 | Foundation Model 自己生成 | Policy（当前模型）采样 |
| 正确答案处理 | 用自己的答案训练（提高概率） | 奖励为正 → 梯度更新提高概率 |
| 错误答案处理 | 用人类答案 | 奖励为负 → 降低概率 |

这可能解释了：为什么 **RL-based 训练通常放在训练流程的最后阶段**——它天然与 Self-Output 类似，对旧有能力的破坏更小，是一种天然抗遗忘的训练方式。

---

## 5.4 实践建议

> **Post-Training 黄金法则**：不要只看目标任务的表现，一定要同时评估模型原有能力是否保留。

**推荐流程**：

```
1. 建立基准 (Baseline)
   ├── 记录 Foundation Model 在各基准测试上的表现
   └── 重点关注：Safety、通用推理、原有擅长任务

2. Post-Training
   ├── 优先选择 Self-Output / RL-based 方法
   └── 若用人类标注数据，混入 3-5% Foundation Model 自生成数据

3. 全面评估
   ├── 目标任务：是否达到预期提升？
   ├── Safety 能力：ToxiGen / HarmBench 等
   └── 通用能力：MMLU / GSM8K / HumanEval 等
```

**常见陷阱**：

| 陷阱 | 表现 | 解决方案 |
|------|------|---------|
| 只看目标任务 | “在 Verilog 上打爆 GPT-4，但连注释都看不懂” | 加全面评估集 |
| 迷信 LoRA 防遗忘 | LoRA rank 大 → 遗忘和全参数差不多 | 配合 Self-Output |
| 忽视 Safety | 教了代码/数学后 Safety 崩了 | 混入 3% Safety 数据 |
| 数据全是人类标注 | 遗忘比用模型自己的答案更严重 | 改用 Paraphrase / Self-Output |

---

# 6. 分布式训练技术

大模型训练必须依赖分布式并行技术。

> **🎯 本章导读**
>
> 分布式训练是大模型训练的**核心工程技术**，没有分布式并行就无法训练超过单GPU显存容量的模型。本章介绍数据并行、张量并行、流水线并行、ZeRO等关键技术，以及如何选择合适的并行策略。**核心原则**：TP用于单层过大，PP用于层数过多，DP用于提升吞吐。

## 6.1 数据并行 (Data Parallelism, DP / DDP)

### 6.1.1 原理
数据并行是最直观的分布式训练方式。当单个 GPU 能够装下完整模型但显存无法容纳大 Batch Size 时，通过水平切分数据集，让多个 GPU 设备并行处理不同数据子集：
- 每个 GPU 设备都拥有一份独立的模型参数拷贝。
- 在前向传播中，各设备独立输入不同的 Batch 数据并计算 Loss 及本地梯度 $\mathbf{g}_i$。
- 在反向传播中，各设备通过卡间通信机制（如 `All-Reduce`）将所有梯度同步并求平均：$\mathbf{g}_{\text{avg}} = \frac{1}{N} \sum_{i=1}^N \mathbf{g}_i$。
- 各设备同步调用优化器，使用平均梯度 $\mathbf{g}_{\text{avg}}$ 同步更新自己的模型权重，确保各张卡上的模型始终一致。

### 6.1.2 底层工程优化：Bucket All-Reduce 与梯度通信重叠 (Overlapping)
在 PyTorch `DistributedDataParallel` (DDP) 的实际工程中，为了避免反向传播结束后一次性进行海量梯度同步带来的网络阻塞，系统采用了**梯度通信与计算重叠（Overlapping）**的技术：
1. **梯度分组（Buckets）**：DDP 在初始化时，根据反向传播的逆序（即从输出层向输入层），将梯度张量分配进多个固定大小的“桶”（Bucket，通常为 25MB）。
2. **异步同步**：在反向传播计算梯度时，一旦某一个 Bucket 中的梯度全部计算完毕，系统会立即在后台启动针对该 Bucket 的 `All-Reduce` 异步通信。与此同时，更前几层的反向传播梯度计算仍在 GPU 核心上继续运行。这成功实现了计算与网络通信的并发，掩盖了大量的卡间同步时间。

### 6.1.3 DDP 局限
- 模型所有参数、优化器状态及梯度必须能够完整装入单个 GPU 的物理显存中。
- 对于超过单卡显存容量的超大模型（如 7B 及以上），DDP 会直接导致 OOM (Out of Memory)，无法独立运行。

<div align="center">
  <img src="/images/llm-training/data-parallelism.png" width="80%" alt="数据并行架构" />
  <figcaption>图：数据并行(DP)架构 - 每个GPU持有完整模型副本（来源：PyTorch Distributed 论文 Figure 1）</figcaption>
</div>

---

## 6.2 张量并行 (Tensor Parallelism, TP)

### 6.2.1 原理与切分策略
当单层权重矩阵的大小超过单卡显存时，张量并行（如 Megatron-LM）通过将每一层的权重参数横向或纵向切分到同一节点（通常具有高速 NVLink 互联）的不同 GPU 上，实现层内的分布式矩阵乘法计算。

#### 1. MLP 层的切分策略
Transformer 的 MLP 层包含两个投影矩阵：门控/上投影 $W_{\text{gate/up}}$ 和下投影 $W_{\text{down}}$。设输入为 $X$，MLP 采用**列并行-行并行**的组合切分：
- **列并行（Column Parallelism）**：
  将第 1 层权重矩阵 $W_{\text{col}}$ 按列均匀拆分为 $p$ 个分片：$W_{\text{col}} = [W_1, W_2, \ldots, W_p]$。
  各卡直接输入完整 $X$，独立计算局部输出：
  $$Y_i = \text{Activation}(X W_i)$$
  此阶段无需任何通信。
- **行并行（Row Parallelism）**：
  将第 2 层权重矩阵 $W_{\text{row}}$ 按行均匀拆分为 $p$ 个分片：$W_{\text{row}} = [V_1; V_2; \ldots; V_p]$。
  各卡输入上一层局部输出 $Y_i$，独立计算矩阵乘法：
  $$Z_i = Y_i V_i$$
  此时，所有 GPU 卡通过一次 `All-Reduce (Sum)` 通信操作，将各卡的局部结果相加，获得完整的输出张量：
  $$Z = \sum_{i=1}^p Z_i + \text{bias}$$

#### 2. Attention 层的切分策略
- **QKV 投影**：同样使用列并行。将注意力头的参数均匀划分到各卡（例如，32 头模型在 8 卡 TP 下，每卡负责 4 头），各卡独立计算对应的 Query、Key 和 Value，不需要卡间通信。
- **注意力计算**：各 GPU 独立运行注意力运算，获得局部的 Context 向量。
- **Output 投影**：使用行并行。将各卡局部的输出权重进行行投影相乘，最后在输出端执行 1 次 `All-Reduce (Sum)` 合并，即可恢复完整的 Multi-Head Attention 输出。

### 6.2.2 数学推导与通信代价分析
对于 Transformer 的一个基本 Block，其前向和反向的通信算子调用非常明确：
- **前向传播 (Forward)**：
  - Attention 层的 Output 投影后：1 次 `All-Reduce` 通信。
  - MLP 层的下投影后：1 次 `All-Reduce` 通信。
  - **前向总开销**：$2 \times \text{All-Reduce}$。
- **反向传播 (Backward)**：
  - 反向传播对应的梯度流在行并行端（输入分发）天然需要一次 `All-Reduce` 汇聚梯度。
  - **反向总开销**：$2 \times \text{All-Reduce}$。

### 6.2.3 适用场景与局限
- **适用场景**：单层权重显存超限（如 70B 模型的 Attention 层及 MLP 层）。
- **局限**：TP 的通信频率极高（每个 Block 前反向共有 4 次 All-Reduce），网络传输必须极快。因此，TP 通常局限在**单节点内的 NVLink 通信**，TP 度数通常设为 2, 4 或 8，极少进行跨节点 TP。

<div align="center">
  <img src="/images/llm-training/tensor-parallelism.png" width="85%" alt="张量并行架构" />
  <figcaption>图：张量并行(TP)架构 - Transformer层的列/行切分策略（来源：Megatron-LM 论文 Figure 3）</figcaption>
</div>

---

## 6.3 流水线并行 (Pipeline Parallelism, PP)

### 6.3.1 原理
当模型层数过多，单节点显存已无法装下时，流水线并行采用“层间纵向切分”：将模型的 $L$ 层划分为 $p$ 个 Stage（阶段），分配到 $p$ 个不同的 GPU 上（可跨节点）。

### 6.3.2 调度策略与气泡占比公式
若直接将整个 Batch 送入流水线，会导致大部分 GPU 在前向和反向时处于闲置等待状态，称为流水线气泡（Bubble）。PP 通过将 Batch 细分为 $m$ 个更小的 Micro-Batches 来提高利用率。

#### 1. GPipe (F-then-B 调度)
- **调度逻辑**：前一 Stage 执行完所有 $m$ 个 Micro-Batches 的前向传播后，后一 Stage 才能执行。随后依次执行所有的反向传播。
- **气泡占比公式**：
  $$F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$$
- **缺点**：激活值（Activation）必须保存在显存中，直到反向传播到来。这造成显存占用随 Micro-Batch 数量 $m$ 线性增加，显存节省效果打折。

#### 2. 1F1B (One Forward, One Backward 调度)
- **调度逻辑**：当流水线启动填充完毕后，每个 Stage 都在交替执行 1 次前向计算与 1 次反向计算。
- **气泡占比公式**：
  $$F_{\text{bubble}} \approx \frac{p - 1}{m}$$
- **优势**：Micro-Batch $i$ 的激活值在其前向完成并执行对应的反向后，可以立即从显存中销毁。这使激活值显存占用与 Micro-Batch 数量 $m$ 彻底解耦，极大缓解了显存压力。

#### 3. Interleaved 1F1B (虚拟流水线)
- 每个 GPU 卡被虚拟分配负责非连续的多个 Stage（例如，GPU 0 负责第 1 层和第 9 层）。这能够进一步将 Bubble 时间减少达约 **2×**，但付出的代价是略微增加了点对点（P2P）的通信频率。

<div align="center">
  <img src="/images/llm-training/pipeline-parallelism.png" width="85%" alt="流水线并行架构" />
  <figcaption>图：流水线并行(PP)架构与1F1B调度策略（来源：GPipe 论文 Figure 1、PipeDream 论文 Figure 3）</figcaption>
</div>

---

## 6.4 序列并行 (Sequence Parallelism, SP)

### 6.4.1 原理
- **机制**：在注意力层的非张量并行区域（如 LayerNorm, Dropout, 残差连接），标准的 TP 仍然需要在每张 GPU 上冗余地存储完整的激活值（Activation Memory）。随着序列长度 $s$ 呈二次方增长，这一显存占用在长文本训练中尤为致命。序列并行（Sequence Parallelism）在**非注意力计算层**把序列维度进行切分（每张卡只负责 $\frac{s}{p}$ 长度的序列），而在进行 QKV 投影和 MLP 列投影前通过 `All-Gather` 拼回完整序列，计算完毕后通过 `Reduce-Scatter` 重新切分。

### 6.4.2 优势与长序列分布式优化
- **降本增效**：成功将 LayerNorm 和 Dropout 处的激活值显存分摊到了 $p$ 张 GPU 上。
- **支持超长上下文**：与 TP 配合（即 TP-SP），能将超长文本（如 128k - 1M Tokens）训练的激活值显存减小近一个数量级，使超长上下文训练不再受阻。

---

## 6.5 ZeRO (Zero Redundancy Optimizer)

微软 DeepSpeed 提出的 ZeRO 旨在完全消除 DDP 模式下多卡冗余的存储开销，将模型状态平摊到各 GPU。

### 6.5.1 模型静态显存数学推导（以 AdamW 优化器为例）
设模型参数量为 $\Psi$。在主流 FP16 混合精度训练下，一个 GPU 为存储模型状态（Model States）所消耗的静态显存包括：
1. **模型参数 (Parameters)**：FP16 存储，占用 $2\Psi$ 字节。
2. **梯度 (Gradients)**：FP16 存储，占用 $2\Psi$ 字节。
3. **优化器状态 (Optimizer States)**：使用 FP32 以保证计算精度，包括：
   - 主权重 (Master Weights)：$4\Psi$ 字节
   - 一阶动量 (Momentum)：$4\Psi$ 字节
   - 二阶变量 (Variance)：$4\Psi$ 字节
   - 优化器显存共计：$12\Psi$ 字节。

$$\text{静态总显存} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ 字节}$$

对于 7B 参数模型，静态显存占用即高达 $16 \times 7 = 112\text{GB}$，单张 H100 (80GB) 甚至连模型状态都装不下。

### 6.5.2 ZeRO 阶段性分片公式与 offload 技术 (设数据并行度为 $N_d$)

#### 1. ZeRO-1：优化器状态分片 (Optimizer States Partitioning)
- **机制**：将 $12\Psi$ 字节的 AdamW 优化器状态均匀分割并平摊到 $N_d$ 张卡上。每张 GPU 只负责更新和保存其中 $\frac{1}{N_d}$ 的优化器状态。
- **单卡显存公式**：
  $$M_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{12\Psi}{N_d}$$
  - *示例*：对于 7B 模型，$N_d=8$ 时，静态显存由 112GB 锐减到 **38.5GB**。

#### 2. ZeRO-2：梯度分片 (Gradient Partitioning)
- **机制**：在反向传播中，一旦某一层参数的梯度计算完毕，立即触发 `Reduce-Scatter` 将其分发给负责更新该层优化器状态的 GPU，其他 GPU 立即释放该梯度。
- **单卡显存公式**：
  $$M_{\text{ZeRO-2}} = 2\Psi + \frac{2\Psi + 12\Psi}{N_d} = 2\Psi + \frac{14\Psi}{N_d}$$
  - *示例*：对于 7B 模型，$N_d=8$ 时，静态显存降至 **26.25GB**。

#### 3. ZeRO-3：参数分片 (Parameter Partitioning)
- **机制**：把 $2\Psi$ 字节的模型参数同样平摊到 $N_d$ 张卡上。前向和反向传播执行到特定层时，所有 GPU 广播（`All-Gather`）获取该层的完整权重，使用完毕后立即丢弃。
- **单卡显存公式**：
  $$M_{\text{ZeRO-3}} = \frac{2\Psi + 2\Psi + 12\Psi}{N_d} = \frac{16\Psi}{N_d}$$
  - *示例*：对于 7B 模型，$N_d=8$ 时，静态显存仅需 **14GB**！

#### 4. ZeRO-Offload (显存-内存卸载)
- 利用 PCIe 通道，将分片后的优化器状态以及梯度卸载（Offload）到宿主机 CPU 的内存（CPU RAM）中，利用宿主机的 CPU 核心执行优化器计算更新。前向时再把更新后的权重写回 GPU。这显著拓宽了单卡能训练的模型参数上限。

#### 5. ZeRO-Infinity
- 在 ZeRO-Offload 基础上，利用 NVMe 固态硬盘（SSD）作三级缓存，可直接在低配 GPU 平台上微调千亿级大模型，打破物理硬件壁垒。

## 6.6 混合并行
————3D Parallelism

结合数据并行、张量并行、流水线并行：

$$
\text{总GPU数} = \text{DP度} \times \text{TP度} \times \text{PP度}
$$

```mermaid
graph TD
    A[选择并行策略] --> B{单层参数能否<br/>装入单GPU?}
    B -->|否| C[启用张量并行 TP]
    B -->|是| D{模型总层数<br/>是否很多?}
    C --> D
    D -->|是| E[启用流水线并行 PP]
    D -->|否| F{还有剩余GPU?}
    E --> F
    F -->|是| G[启用数据并行 DP<br/>提升吞吐量]
    F -->|否| H[完成配置]
    G --> H

    I[示例配置] --> J[1024 GPUs训练<br/>175B模型]
    J --> K[TP=8: 单层切8份]
    K --> L[PP=16: 分16个stage]
    L --> M[DP=8: 8个数据副本]
    M --> N[8×8×16=1024]

    style C fill:#ffcdd2
    style E fill:#fff9c4
    style G fill:#c8e6c9
    style H fill:#e3f2fd
```

<div align="center">
  <img src="/images/llm-training/3d-parallelism.png" width="90%" alt="3D混合并行架构" />
  <figcaption>图：3D混合并行架构(DP+TP+PP)训练大模型（来源：Megatron-LM 2021 论文 Figure 1）</figcaption>
</div>

### 6.6.1 策略选择原则

**决策流程**：
1. **首先考虑TP（张量并行）**：
   - 当单层参数 > 单GPU显存时必须使用
   - 典型配置：TP=2/4/8（同节点内，NVLink通信）
   - 例如：单层12GB，单GPU 80GB → 不需要TP

2. **其次考虑PP（流水线并行）**：
   - 当模型总层数很多时使用
   - 典型配置：PP=2/4/8/16
   - 例如：96层模型，PP=16 → 每个stage 6层

3. **最后考虑DP（数据并行）**：
   - 使用剩余所有GPU
   - 提升训练吞吐量
   - 例如：1024 GPU，TP=8，PP=16 → DP=8

**实际案例**：

| 模型规模 | TP | PP | DP | 总GPU | 说明 |
|---------|----|----|----|----|------|
| **7B参数** | 1 | 1 | 64 | 64 | 小模型，纯DP即可 |
| **13B参数** | 2 | 1 | 32 | 64 | 需要少量TP |
| **70B参数** | 8 | 4 | 4 | 128 | 需要TP+PP |
| **175B参数** | 8 | 16 | 8 | 1024 | 大模型，3D并行 |
| **540B参数** | 8 | 32 | 16 | 4096 | 超大模型 |

**Trade-off考虑**：
- **TP增大**：层内通信增多，需要高速互联（NVLink）
- **PP增大**：Pipeline bubble增大，GPU利用率下降
- **DP增大**：梯度同步通信增多，但可用Ring-AllReduce优化


# 7. ⚡ 训练优化技术

在大语言模型（LLM）的训练中，硬件资源（特别是GPU显存和带宽）与训练时间是核心瓶颈。优化技术不仅决定了模型能否在有限的资源下跑起来，还直接决定了训练的收敛速度与最终效果。本章将深入解析主流的优化器选择、学习率调度、梯度处理方法、正则化技术以及注意力加速算子 Flash Attention。

> **🎯 本章导读**
> 
> 大模型训练是一场**显存与计算效率的博弈**。优化器的选择决定了显存占用的下限，学习率和梯度策略决定了模型能否平稳收敛，而 Flash Attention 则是目前解决长文本注意力计算瓶颈的基石。本章旨在帮助读者建立从“数学公式”到“工程实现”的完整认知体系。

---

## 7.1 优化器选择（Optimizer Selection）

在大模型训练中，优化器状态（Optimizer States）是最大的显存消耗源之一。在常用的 FP16/BF16 混合精度训练中，虽然权重和梯度只需 2 字节（FP16/BF16），但 AdamW 优化器需要为每个参数存储一份 FP32 的 Master Weights、FP32 的一阶动量（Momentum）和 FP32 的二阶动量（Variance），这带来了巨大的显存开销。

### 7.1.1 AdamW：主导大模型训练的经典之作

AdamW 是当前大模型训练最主流的优化器（如 LLaMA, GPT, InternLM 等默认使用）。

#### 1. 核心公式与 Weight Decay 解耦
传统的 Adam 优化器在结合 L2 正则化时，会将权重梯度与正则化梯度混合在一起进行动量估计，导致对稀疏梯度的缩放异常。AdamW 将权重衰减（Weight Decay）直接与梯度更新解耦，在前一步更新参数时直接减去衰减项：

$$
\theta_{t+1} = \theta_t - \eta_t \lambda \theta_t - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \right)
$$

其中：
- $\theta_t$：第 $t$ 步的模型参数
- $\eta_t$：当前步的学习率
- $\lambda$：权重衰减率（通常为 0.1）
- $\hat{m}_t, \hat{v}_t$：经过偏差修正的一阶动量和二阶动量

#### 2. 显存开销分析
假设模型参数量为 $N$，采用混合精度（Mixed Precision）训练：
- **模型参数 (FP16/BF16)**：$2N$ 字节
- **梯度 (FP16/BF16)**：$2N$ 字节
- **AdamW 优化器状态 (FP32)**：
  - **Master Weights**（用于累加微小更新）：$4N$ 字节
  - **一阶动量 $m_t$**：$4N$ 字节
  - **二阶动量 $v_t$**：$4N$ 字节
  - **优化器状态总计**：$12N$ 字节

**结论**：仅优化器状态就需要 $12N$ 字节的显存，占混合精度训练基础显存（$16N$ 字节）的 **75%**。

---

### 7.1.2 Adafactor：低显存的自适应步骤优化器

Adafactor 主要是为了解决 AdamW 二阶动量占用 $4N$ 字节显存的痛点，常被用于 T5 等模型的训练。

#### 1. 低秩分解减小显存
Adafactor 的核心思想是将二阶动量矩阵 $V \in \mathbb{R}^N$（大小为参数量 $N=R \times C$）进行**低秩分解（Low-Rank Factorization）**，即通过行和 $V_R \in \mathbb{R}^R$ 和列和 $V_C \in \mathbb{R}^C$ 来近似二阶动量：

$$
\hat{V}_{i,j} = \frac{(V_R)_i \cdot (V_C)_j}{\sum_{k} (V_C)_k}
$$

这使得存储二阶动量的空间从 $O(RC)$ 降到 $O(R + C)$。对于一个大矩阵，这几乎将二阶动量的显存占用减少到了接近于 0。

#### 2. 优缺点分析
- **优点**：极大地节省了显存，使优化器状态显存从 $12N$ 字节降低到约 $4N$ 字节（如果禁用一阶动量并只存行/列二阶动量因子）。
- **缺点**：不存储完整的一阶动量和二阶动量，可能导致训练在某些任务上收敛变慢、不稳定。

---

### 7.1.3 Lion (Evolved Sign Momentum)：数据驱动的极简优化器

Lion 是通过 Google 的算法进化搜索（Symbolic Discovery）发现的新型优化器。

#### 1. 核心机制：Sign 函数与单动量
Lion 舍弃了二阶动量，仅保留一阶动量，且在更新参数时仅使用**符号函数（Sign Function）**，这使得更新步长更加均匀。其更新规则如下：

$$
u_t = \text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t)
$$
$$
m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t
$$
$$
\theta_{t+1} = \theta_t - \eta_t (u_t + \lambda \theta_t)
$$

其中：
- $g_t$：当前步梯度
- $m_t$：一阶动量
- $\text{sign}(\cdot)$：符号函数（取值为 $+1, -1, 0$）

#### 2. 显存开销分析
因为 Lion 删除了二阶动量：
- **一阶动量 $m_t$ (FP32)**：$4N$ 字节
- **Master Weights (FP32)**：$4N$ 字节
- **优化器状态总计**：$8N$ 字节（相比 AdamW 节省了 $4N$ 字节）

#### 3. 特点
- **高计算吞吐量**：`sign` 操作非常适合 GPU 向量化执行，且由于没有二阶动量的繁琐计算，每步迭代速度稍快。
- **超参数敏感**：Lion 相比 AdamW 更容易受学习率和权重衰减大小的影响，需要针对特定模型重新调优超参。

---

### 7.1.4 优化器系统对比与显存结构图

以下是常用优化器的综合对比表：

| 优化器 | 显存开销 (仅状态) | 核心特点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **AdamW** | $12N$ 字节 | 收敛极其平稳，对超参数不敏感，生态支持最完善 | 显存占用极大 | LLM 训练绝对默认选择 |
| **Lion** | $8N$ 字节 | 节省 33% 优化器显存，速度稍快，更新幅度均匀 | 超参难调，早期收敛可能有抖动 | 显存受限、追求更高吞吐量的场景 |
| **Adafactor** | $4N$ - $8N$ 字节 | 行列低秩分解，超长序列下显存优势明显 | 训练稳定性弱于 AdamW | 早期 T5 训练，极端显存受限场景 |

#### 优化器状态显存占用图解 (基于 FP16 混合精度训练，每参数字节数)

```mermaid
gantt
    title 优化器状态显存占用对比 (以每个模型参数为单位，单位：字节)
    dateFormat  X
    axisFormat %s
    
    section AdamW (12字节状态 + 4字节模型/梯度 = 16B)
    模型参数 & 梯度 (FP16) :active, 0, 4
    Master Weights (FP32) :crit, 4, 8
    一阶动量 (FP32) :active, 8, 12
    二阶动量 (FP32) :active, 12, 16
    
    section Lion (8字节状态 + 4字节模型/梯度 = 12B)
    模型参数 & 梯度 (FP16) :active, 0, 4
    Master Weights (FP32) :crit, 4, 8
    一阶动量 (FP32) :active, 8, 12
    
    section Adafactor (4字节状态 + 4字节模型/梯度 = 8B)
    模型参数 & 梯度 (FP16) :active, 0, 4
    Master Weights (FP32) :crit, 4, 8
```

---

## 7.2 学习率策略（Learning Rate Schedules）

在 Transformer 架构中，合理的学习率策略对防止模型梯度爆炸、加速收敛至关重要。目前业界标准采用 **Warmup (预热) + Decay (衰减)** 模式。

### 7.2.1 Learning Rate Warmup（预热）

#### 1. 为什么必须 Warmup？
在大模型训练初期（特别是使用 Pre-LN 结构或 AdamW 优化器时）：
- 随机初始化的权重导致网络前几层的梯度极不稳定。
- AdamW 优化器的二阶动量估计尚未建立（$\hat{v}_t$ 接近零，导致修正后的步长异常巨大）。
如果直接使用峰值学习率，极易引发数值溢出（Overflow）或不可逆的梯度爆炸。

#### 2. 实现方式
在训练的前 $T_{\text{warmup}}$ 步（通常占总步数的 1%–5%，约 2000–10000 步），学习率从 0 线性增加到最大峰值学习率 $\text{lr}_{\max}$：

$$
\text{lr}(t) = \text{lr}_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \le T_{\text{warmup}}
$$

---

### 7.2.2 Cosine Decay（余弦衰减）

Cosine Decay 是大模型最主流的衰减方式，它能使学习率在中间训练阶段保持平稳释放，在训练末期快速收敛。

$$
\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min}) \left(1 + \cos\left(\frac{\pi (t - T_{\text{warmup}})}{T - T_{\text{warmup}}}\right)\right), \quad t > T_{\text{warmup}}
$$

- **特点**：曲线顺滑。实验表明，Cosine 衰减在绝大多数语言建模任务上相比线性衰减能取得更低的困惑度（Perplexity）。
- **参数推荐**：$\text{lr}_{\min}$ 通常设为 $\text{lr}_{\max}$ 的 10%（或直接设为 0）。

---

### 7.2.3 WSD (Warmup-Stable-Decay) 调度策略

近年来，一些超大规模训练项目（例如 LLaMA 3, DeepSeek-V2/V3）为了应对**持续增量训练（Continual Training）**或动态调整数据量的需求，开始采纳 **WSD 调度策略**。

```mermaid
graph TD
    A[WSD 学习率策略] --> B[Warmup 阶段]
    A --> C[Stable 恒定阶段]
    A --> D[Decay 退火阶段]
    
    B --> B1[快速提升学习率到峰值]
    C --> C1[长时期以恒定最大学习率训练，方便中途随时终止或追加数据]
    D --> D1[在训练最后 5%–10% 步数内，急剧指数/余弦衰减，锁定权重收敛]
```

- **优势**：
  1. **高度灵活性**：在 Stable 阶段，如果发现模型表现好，可以随时延长 Stable 阶段的长度以塞入更多数据，而无需重跑 Cosine Decay 曲线。
  2. **快速收敛**：退火阶段（Decay Phase）在短时间内将学习率压低，模型效果在此阶段会迎来“二次飞跃”（PPL 骤降）。

---

## 7.3 梯度处理（Gradient Processing）

### 7.3.1 Gradient Clipping（梯度裁剪）

为防止在遇到异常长样本或极端梯度时引发梯度爆炸，需要对所有层梯度向量的模长进行截断。

#### 1. L2 范数全局裁剪（Global Norm Clipping）
这是大模型训练的标配。计算所有参数梯度拼接成的全局梯度向量 $\mathbf{g}$ 的 L2 范数，若超过阈值 $d_{\max}$，则进行等比例缩放：

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{d_{\max}}{\|\mathbf{g}\|_2}\right)
$$

- **最佳实践**：大模型训练中全局阈值 $d_{\max}$ 通常设为 `1.0`。
- **优点**：保持了梯度向量的方向不变，仅限制了步长，能很好地维持各层更新比例的协调。

---

### 7.3.2 Gradient Accumulation（梯度累积）

在大模型训练中，由于单卡显存受限，无法直接将很大的 Batch Size（例如百万级 tokens 级别）一次性喂入 GPU 进行前向传播。**梯度累积**通过“时间换空间”的方式，在物理显存受限时模拟大 Batch 训练。

#### 1. 工作原理
设目标 Batch Size 为 $B_{\text{global}}$，单卡单步处理的 Micro Batch Size 为 $B_{\text{micro}}$。
1. 在连续的 $N$ 个 Step 中，仅进行前向传播和反向传播，将计算出的梯度**累加（Add）**在梯度缓冲区中，而不调用 `optimizer.step()`。
2. 在第 $N$ 步，将累积的梯度除以 $N$（取平均），然后执行 `optimizer.step()` 更新参数并清空梯度。
3. 对应的等式：$B_{\text{global}} = B_{\text{micro}} \times N \times \text{DP\_degree}$。

```mermaid
sequenceDiagram
    autonumber
    participant GPU as GPU 显存 (Micro-Batch)
    participant Comm as 节点间通信 (All-Reduce)
    participant Opt as 优化器更新 (optimizer.step)
    
    Note over GPU: 运行 Micro-batch 1: 前向 + 反向
    Note over GPU: 梯度保留在显存中，不进行同步
    Note over GPU: 运行 Micro-batch 2: 前向 + 反向
    Note over GPU: 梯度与 Micro-batch 1 累加，不进行同步
    Note over GPU: ... 运行至 Micro-batch N
    Note over GPU: 累积完成 (N 步)
    GPU->>Comm: 触发 All-Reduce，跨卡同步累积梯度
    Comm-->>GPU: 同步完毕
    GPU->>Opt: 执行优化器参数更新
    GPU->>GPU: zero_grad() 清空梯度缓冲区
```

- **工程优化 (PyTorch `no_sync` 机制)**：在分布式训练中，默认的反向传播会在每一步自动触发卡间梯度同步（All-Reduce）。在梯度累积的前 $N-1$ 步中，应使用 `model.no_sync()` 上下文管理器，阻断无用的网络同步，只在最后一步执行 All-Reduce，能极大地提升网络带宽利用率。

---

### 7.3.3 Gradient Checkpointing（梯度检查点 / 激活重计算）

梯度检查点（Activation Checkpoint / Recomputation）是用**计算时间换显存空间**的经典技术。

#### 1. 背景：前向激活值显存瓶颈
在反向传播计算梯度时，公式需要用到前向传播计算出的激活值（Activation）。因此，标准的训练过程会在前向传播中把所有层的激活值保存在显存中，这造成了随模型层数 $L$ 和序列长度 $s$ 呈线性增长的巨大显存占用。

#### 2. 核心原理
- **选择性保存**：不再保存所有层的激活值，而是每隔 $k$ 层选择一层作为“检查点”（Checkpoint），只保存该层的激活值。
- **反向重计算**：反向传播到未保存激活值的层时，从最近的检查点开始，重新运行一次前向传播，实时计算出临时激活值用于梯度计算，算完后立即丢弃。

```mermaid
graph TD
    subgraph 标准训练 (Standard Training)
        A1[Layer 1 Forward] -->|保存 Activation 1| A2[Layer 2 Forward]
        A2 -->|保存 Activation 2| A3[Layer 3 Forward]
        A3 --> A4[Loss & Backward]
        A4 -->|使用 Activation 2| A5[Layer 2 Backward]
        A5 -->|使用 Activation 1| A6[Layer 1 Backward]
    end

    subgraph 激活值重计算 (Gradient Checkpointing)
        B1[Layer 1 Forward] -->|保存 Checkpoint 1| B2[Layer 2 Forward]
        B2 -.->|丢弃中间 Activation 2| B3[Layer 3 Forward]
        B3 --> B4[Loss & Backward]
        B4 -->|临时重计算: B1 -> Layer 2 Forward| B5[Layer 2 Backward]
        B5 -->|使用 Checkpoint 1| B6[Layer 1 Backward]
    end
    
    style A1 fill:#ffebee,stroke:#c62828
    style A2 fill:#ffebee,stroke:#c62828
    style B1 fill:#e8f5e9,stroke:#2e7d32
    style B2 fill:#eceff1,stroke:#37474f
```

#### 3. 代价与收益
- **收益**：激活值显存复杂度从 $O(L)$ 降至 $O(\sqrt{L})$，能极大地防止在超长序列训练时发生 OOM。
- **代价**：反向传播中多了一次前向计算，通常会带来大约 **30%–33%** 的额外计算开销。

---

## 7.4 正则化与数值稳定性技术

在大模型训练中，正则化不仅用于防止过拟合，更关键的作用在于提高混合精度训练下的数值稳定性。

### 7.4.1 Weight Decay（权重衰减）

- **在 AdamW 中的重要性**：在每一轮迭代更新中，模型权重乘以一个略小于 1 的系数（通常 Weight Decay = 0.1）：$\theta_{t} \leftarrow (1 - \eta_t \lambda)\theta_{t}$。
- **作用**：防止权重数值过大。在大模型训练中，权重过大容易导致激活值溢出（NaN）或引发注意力矩阵偏置过大而导致 Softmax 梯度饱和。

### 7.4.2 Dropout：在大模型中的逐渐淡出

- **现状**：在千亿（100B）以上规模模型的预训练阶段，Dropout 率（包括 Attention Dropout 和 Residual Dropout）通常被直接设为 **0**。
- **原因**：
  1. **数据充沛度**：大模型预训练的数据集往往极为庞大，模型参数相较于海量数据而言不易过拟合，本身已具备天然正则化。
  2. **效率损失**：Dropout 需要在随机状态发生器中采样 mask 矩阵并写入显存，降低了显卡计算吞吐（Throughput）。
- **特殊例外**：在小模型微调阶段（SFT）或在 Embedding 层后可能会保留 0.05–0.1 的 Dropout 以防止对特定微调模板产生过拟合。

### 7.4.3 Z-loss 正则化：抑制 Logits 爆炸

在大模型（如 PaLM, Gemini, DeepSeek）使用 fp16/bf16 混合精度进行超大规模分布式训练时，分类头的 Logits 容易变得极大，导致 Softmax 计算中指数项产生数值溢出（出现 NaN）。

#### 1. 核心机制
Z-loss 在原始的交叉熵损失中加入了一项辅助惩罚项，用于惩罚 Logits 的配分函数 $Z$（即 $\sum_i e^{x_i}$）的对数值：

$$
\mathcal{L} = \mathcal{L}_{\text{cross-entropy}} + \alpha \log^2 Z
$$

其中 $Z = \sum_{i} e^{x_i}$，$\alpha$ 通常设为 $10^{-4}$ 级。

#### 2. 作用与原理
- **约束 Logits 的绝对大小**：强制使输出的 Logits 均值处于合理范围（如 0 附近）。
- **极大地增强了稳定性**：避免在数十万步训练之后，由于分类层 Logits 漂移导致出现无法挽回的 NaN 崩溃。


## 7.5 Flash Attention

Flash Attention 是当前 LLM 训练和推理中最重要的注意力加速技术，通过改变计算顺序而非简化算法，在保持数值等价的前提下大幅减少显存占用、加快计算速度。

### 7.5.1 背景：标准 Attention 的内存瓶颈

标准 Scaled Dot-Product Attention 的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**问题所在**：
- 序列长度为 $n$ 时，$QK^T$ 矩阵大小为 $n \times n$，需完整写入 GPU 主显存（HBM）
- 内存复杂度 $O(n^2)$：$n=4096$ 时约占数百 MB，$n=32768$ 时高达数十 GB
- **真正的瓶颈不是计算量，而是数据搬运**：HBM 带宽远低于片上 SRAM，反复读写大矩阵成为性能瓶颈

### 7.5.2 GPU 内存层次：工作台 vs 仓库

```
SRAM（片上缓存）= 工作台
  • 容量：几十 MB
  • 速度：极快（~19 TB/s on A100）
  • 特点：运算单元直接读写

HBM（主显存）= 仓库
  • 容量：40–80 GB
  • 速度：较慢（~2 TB/s on A100）
  • 特点：所有 tensor 默认存放于此
```

标准 Attention 每步计算都需把中间矩阵写回 HBM，再从 HBM 读回，造成大量低效的数据搬运。Flash Attention 的核心目标：**尽量让数据留在 SRAM，减少往返 HBM 的次数**。

<div align="center">
  <img src="/images/llm-training/FlashAttention.png" width="90%" />
  <figcaption>图：GPU 内存层次（左）、FlashAttention 分块计算机制（中）、与标准 PyTorch 实现的速度对比（右）。来源：Dao et al., FlashAttention, NeurIPS 2022</figcaption>
</div>

### 7.5.3 核心思想：分块 + Online Softmax

Flash Attention 的两个关键创新：

**1. 分块计算（Tiling）**

将 $K$、$V$ 切成小块，逐块载入 SRAM 计算，避免 $n \times n$ 矩阵写入 HBM。当前块计算完后直接丢弃，不写回 HBM。

**2. Online Softmax（增量归一化）**

传统 Softmax 需先扫描全局最大值 $A_{\max}$，才能计算 $\exp$——这要求看到全部数据，无法分块。Online 方法的解决思路：

- 处理每一块时维护当前见过的最大值，遇到更大值时用修正因子还原历史累积：

$$\text{修正因子} = \exp\!\left(A_{\max}^{\text{旧}} - A_{\max}^{\text{新}}\right)$$

- 全程无需存储完整的 $n \times n$ 注意力矩阵

**3. 直接累积输出**

不显式存储注意力权重，直接累积输出 $O$：

$$O_k = O_{k-1} \times \text{修正因子} + \text{当前块贡献}$$

**4. 反向传播重计算**

反向传播时不从 HBM 读取中间矩阵，而是从 $Q/K/V$ 重新计算——用少量额外算力换取大量显存节省。

### 7.5.4 内存与速度对比

| 方法 | 注意力矩阵内存 | HBM 访问次数 |
|------|-------------|------------|
| 标准 Attention | $O(n^2)$ | 多次往返 |
| Flash Attention | $O(n)$（无完整矩阵）| 最少化 |

**实测性能数据**：
- 序列长度 4096：速度约提升 **8–9×**
- 数值精度差异：$< 10^{-7}$（与标准 Attention 数值等价）
- Yi-34B 实测：70K tokens 时 2.0s → 1.3s；730K+ tokens 时无 Flash Attention 直接 OOM

### 7.5.5 版本演进

| 版本 | 年份 | 核心改进 |
|------|-----|---------|
| FlashAttention v1 | 2022 | IO-aware 分块 + Online Softmax 原始实现 |
| FlashAttention v2 | 2023 | 更优并行策略，减少线程同步开销，~2× vs v1 |
| FlashAttention v3 | 2024 | 针对 H100/Hopper 架构，原生 FP8 支持 |

### 7.5.6 实际使用

**PyTorch 2.0+ 内置支持**（推荐，零额外依赖）：

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True  # 自回归训练必须设为 True
)
```

**通过 transformers 显式启用 Flash Attention 2**：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"  # 需先安装 flash-attn
)
```

```bash
pip install flash-attn --no-build-isolation
```

> **最佳实践**：Flash Attention 与标准 Attention 数值等价，序列越长收益越显著。PyTorch 2.0+ 可通过 `scaled_dot_product_attention` 零成本启用；所有 LLM 训练任务均建议开启。

---

# 8. 模型量化技术

模型量化是降低大模型计算和存储成本的核心技术之一。通过将高精度浮点数（FP32/FP16）转换为低精度整数（INT8/INT4），量化可以显著减少显存占用、加快推理速度，同时保持模型性能基本不变。

> **🎯 本章导读**
>
> 量化是大模型民主化的关键技术。**一个70B模型，FP16需要140GB显存，INT4量化后仅需35GB**——这意味着从8×A100降到1×A100即可运行。本章深入讲解量化的数学原理、主流方法（GPTQ、AWQ、QLoRA）、训练策略，以及工程实践。核心要点：**量化不是简单压缩，而是精心设计的精度-性能权衡艺术**。

## 8.1 🧮 量化基础概念

### 8.1.1 什么是量化？

量化（Quantization）是将连续的高精度数值映射到离散的低精度数值的过程：

$$
\text{FP32/FP16} \xrightarrow{\text{量化}} \text{INT8/INT4/INT2}
$$

**核心目标**：
- 📉 **降低显存占用**：FP16 → INT8 减少50%，FP16 → INT4 减少75%
- ⚡ **加速计算**：整数运算比浮点运算快2-4倍
- 💰 **降低成本**：更小的模型可以部署在更便宜的硬件上

**关键挑战**：
- 保持模型精度不显著下降
- 处理异常值（Outliers）
- 平衡量化粒度与精度损失

### 8.1.2 量化的数学原理

#### 8.1.2.1 对称量化（Symmetric Quantization）

将浮点数 $x \in [-\alpha, \alpha]$ 映射到整数 $q \in [-127, 127]$（INT8为例）：

$$
q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\alpha}{127}
$$

**反量化**（推理时恢复）：

$$
\hat{x} = s \cdot q
$$

其中：
- $s$：缩放因子（scale）
- $\alpha$：所有元素绝对值的最大值
- $q$：量化后的整数

**特点**：
- ✅ 实现简单，硬件友好
- ✅ 零点为0，无需额外存储
- ❌ 对非对称分布的数据浪费表示范围

#### 8.1.2.2 非对称量化（Asymmetric Quantization）

处理非对称分布的数据 $x \in [x_{\min}, x_{\max}]$：

$$
q = \text{round}\left(\frac{x - z}{s}\right)
$$

其中：
- $s = \frac{x_{\max} - x_{\min}}{255}$（INT8）
- $z = \text{round}\left(-\frac{x_{\min}}{s}\right)$：零点（zero-point）

**反量化**：

$$
\hat{x} = s \cdot (q - z)
$$

**特点**：
- ✅ 更好利用量化范围
- ✅ 适合激活值（通常非对称）
- ❌ 需要额外存储零点参数

#### 8.1.2.3 量化粒度

| 粒度 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| **Per-tensor** | 整个张量共享一个 $(s, z)$ | 内存小，速度快 | 精度损失较大 |
| **Per-channel** | 每个输出通道独立 $(s_i, z_i)$ | 精度更高 | 参数量增加 |
| **Per-group** | 每组参数共享（如128个元素） | 平衡精度与效率 | 实现复杂 |

**最佳实践**：
- **权重**：Per-channel 量化（精度关键）
- **激活值**：Per-tensor 量化（速度优先）

### 8.1.3 量化误差分析

量化引入的误差：

$$
\text{Error} = \mathbb{E}[(x - \hat{x})^2] = \mathbb{E}[(x - s \cdot \text{round}(x/s))^2]
$$

**误差来源**：
1. **舍入误差**：$\text{round}()$ 操作导致的精度损失
2. **裁剪误差**：超出 $[x_{\min}, x_{\max}]$ 的值被裁剪
3. **异常值影响**：少数极大值拉大缩放因子 $s$，压缩其他值的表示精度

**降低误差的策略**：
- 使用更细粒度的量化（Per-channel）
- 异常值单独处理（Mixed-precision）
- 量化感知训练（QAT）

### 8.1.4 量化精度与性能权衡（Precision-Performance Trade-offs）

不同的量化比特与数值格式在显存节省、计算加速和模型精度（困惑度 PPL）之间存在复杂的权衡。以下是主流数值格式的系统对比：

| 数值格式 | 参数显存占比 (以BF16为基准) | 困惑度 (PPL) 变化 | 硬件加速支持 | 典型应用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **BF16 / FP16** | 100% | 0 (基准) | Tensor Core 原生 | 标准模型训练、高精度推理基准 |
| **INT8 Weight-Only** | ~50% | 几乎无损 (< 0.05) | 需反量化到FP16计算 | 资源受限服务器部署，侧重保留精度 |
| **INT8 Weight & Act** | ~50% | 轻微下降 (< 0.1) | INT8 GEMM 硬件加速 | 高并发吞吐量推理（如 SmoothQuant） |
| **INT4 (NF4 / GPTQ)**| ~25% | 参数 > 7B 时无感；参数 < 3B 时轻微上升 | 需反量化计算或特定 kernel | 消费级显卡本地部署、QLoRA 训练 |
| **FP8 (E4M3 / E5M2)**| ~50% (训练) / 25% (推理) | 极小 (< 0.02) | Hopper/Blackwell 原生支持 | 极大规模分布式训练（如 DeepSeek-V3/R1）、新一代高吞吐推理 |

#### 8.1.4.1 关键结论与选型指南：
1. **大参数量对量化更具鲁棒性**：例如 70B 模型在 INT4 量化下的困惑度损失（Perplexity Degradation）几乎为零，而 3B/7B 模型在 4-bit 量化下会出现明显的常识和推理能力衰退。因此，小模型不建议使用低于 4-bit 的量化部署。
2. **FP8 是当前大模型训练的黄金标准**：DeepSeek-V3/R1 成功在 FP8（前向 E4M3，反向梯度 E5M2）格式下训练数万亿 Token。FP8 不仅能将权重和激活值显存减半，还能在 H100 等 GPU 上释放双倍的 Tensor Core 算力吞吐，基本做到了无损精度与速度的统一。
3. **NF4 (Normal Float 4) 专为正态分布设计**：它是 QLoRA 能够成功训练的基石。对于非均匀分布的权重矩阵，NF4 的信息熵明显高于普通 INT4，能最大化保留语言模型原有的表征精度。

---

## 8.2 📊 量化方法分类

### 8.2.1 按量化时机分类

```mermaid
graph LR
    A[量化方法] --> B[训练后量化<br>PTQ]
    A --> C[量化感知训练<br>QAT]

    B --> B1[动态量化<br>Dynamic]
    B --> B2[静态量化<br>Static]

    C --> C1[全量化训练<br>Full QAT]
    C --> C2[部分量化训练<br>Partial QAT]

    style B fill:#ffe0b2,stroke:#e65100
    style C fill:#c8e6c9,stroke:#1b5e20
```

### 8.2.2 训练后量化（Post-Training Quantization, PTQ）

**定义**：在模型训练完成后，直接对权重和激活值进行量化，无需重新训练。

**优点**：
- ⚡ **速度快**：几分钟到几小时即可完成
- 💰 **成本低**：无需训练数据和GPU资源
- 🛠️ **易部署**：可直接应用于任何预训练模型

**缺点**：
- 📉 **精度损失**：特别是低比特量化（INT4/INT2）
- ⚠️ **敏感性高**：某些层对量化非常敏感

**代表方法**：
- **GPTQ**：基于二阶信息的逐层量化
- **AWQ**：基于激活值重要性的权重量化
- **SmoothQuant**：迁移量化难度从激活到权重

### 8.2.3 量化感知训练（Quantization-Aware Training, QAT）

**定义**：在训练过程中模拟量化操作，让模型学习适应量化误差。

**核心思想**：
- 前向传播：使用量化后的权重和激活值
- 反向传播：使用浮点梯度（STE技巧）

**优点**：
- ✅ **精度最高**：模型主动适应量化
- ✅ **支持极低比特**：INT4、INT2甚至二值化

**缺点**：
- ⏰ **训练时间长**：需要重新训练或微调
- 💰 **成本高**：需要训练数据和GPU资源

**代表方法**：
- **QLoRA**：4-bit量化 + LoRA微调
- **LLM-QAT**：大模型量化感知训练
- **BitNet**：极低比特（1.58-bit）量化训练

### 8.2.4 按量化对象分类

| 量化对象 | 说明 | 难度 | 常用方法 |
|---------|------|------|---------|
| **仅权重** | 只量化模型参数 | 简单 | GPTQ, AWQ |
| **权重+激活** | 同时量化参数和中间结果 | 困难 | SmoothQuant, QAT |
| **KV Cache** | 量化推理阶段的注意力缓存，减少长上下文显存 | 中等 | INT8/INT4 Per-token, GQA/MQA |

---

## 8.3 🔧 训练后量化（PTQ）

### 8.3.1 GPTQ：基于二阶信息的权重量化

#### 8.3.1.1 核心原理

GPTQ（**G**PT **P**ost-**T**raining **Q**uantization）使用**二阶导数信息**优化量化误差，逐层量化权重矩阵。

**优化目标**：最小化量化前后输出差异

$$
\arg\min_{\hat{W}} \| WX - \hat{W}X \|^2
$$

其中：
- $W$：原始FP16权重矩阵
- $\hat{W}$：量化后的INT4权重
- $X$：校准数据的激活值

**关键技术**：Optimal Brain Quantization (OBQ)

逐列量化权重，每次量化一列时，将误差传播到剩余列：

$$
W_{\text{remaining}} \leftarrow W_{\text{remaining}} - \frac{w \cdot H^{-1}_{:,i}}{H^{-1}_{i,i}} \cdot e_i^T
$$

其中 $H = 2X^TX$ 是Hessian矩阵。

#### 8.3.1.2 实现流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Step 1: 加载预训练模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: 准备校准数据（通常128-1024个样本）
from datasets import load_dataset
calibration_dataset = load_dataset("c4", split="train[:1000]")

def prepare_calibration_data(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

calibration_data = calibration_dataset.map(prepare_calibration_data, batched=True)

# Step 3: 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,                      # 量化位数：4-bit
    group_size=128,              # 每128个参数共享一个scale
    damp_percent=0.01,           # Hessian阻尼系数（稳定性）
    desc_act=False,              # 激活值降序排列（可选优化）
    sym=True,                    # 对称量化
    true_sequential=True,        # 严格按层顺序量化
)

# Step 4: 执行GPTQ量化（耗时：7B模型约10-30分钟）
model_quantized = AutoGPTQForCausalLM.from_pretrained(
    model_name,
    quantize_config=quantize_config,
)
model_quantized.quantize(calibration_data)

# Step 5: 保存量化模型（7B: 140GB → 3.5GB）
model_quantized.save_quantized("./llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-2-7b-gptq-4bit")

# Step 6: 加载和使用量化模型
from auto_gptq import AutoGPTQForCausalLM
model_gptq = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-4bit",
    device="cuda:0",
    use_safetensors=True,
)

# 推理速度提升2-3倍，显存减少75%
inputs = tokenizer("The meaning of life is", return_tensors="pt").to("cuda:0")
outputs = model_gptq.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

#### 8.3.1.3 GPTQ 性能表现

| 模型 | 原始精度 | GPTQ-4bit | 显存占用 | 精度损失 |
|------|---------|-----------|----------|---------|
| **LLaMA-7B** | FP16 | INT4 | 14GB → 3.5GB (-75%) | <1% |
| **LLaMA-13B** | FP16 | INT4 | 26GB → 6.5GB (-75%) | <1% |
| **LLaMA-30B** | FP16 | INT4 | 60GB → 15GB (-75%) | ~1% |
| **LLaMA-65B** | FP16 | INT4 | 130GB → 33GB (-75%) | ~2% |

**适用场景**：
- ✅ 推理部署优化
- ✅ 资源受限环境
- ✅ 需要快速量化（无需训练）

---

### 8.3.2 AWQ：激活感知权重量化

#### 8.3.2.1 核心洞察

AWQ（**A**ctivation-aware **W**eight **Q**uantization）的核心发现：

> **并非所有权重都同等重要！** 对应大激活值的权重通道对模型性能影响更大，应该保持更高精度。

**量化策略**：

$$
\hat{W} = \text{Quantize}(W \cdot s), \quad \hat{Y} = \frac{\hat{W} \cdot (X / s)}{s}
$$

其中 $s$ 是per-channel的缩放因子，根据激活值大小自适应调整。

#### 8.3.2.2 确定通道重要性

1. **收集激活值统计**：

$$
s_i = \text{salient}(X_{:,i}) = \mathbb{E}[\|X_{:,i}\|]
$$

2. **保护重要通道**：

重要通道（激活值大）使用更大的缩放因子 → 量化误差更小

#### 8.3.2.3 实现代码

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-4bit"

# Step 1: 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Step 2: 配置量化参数
quant_config = {
    "zero_point": True,          # 使用零点（非对称量化）
    "q_group_size": 128,         # 分组大小
    "w_bit": 4,                  # 权重量化位数
    "version": "GEMM"            # 使用优化的GEMM kernel
}

# Step 3: 量化（AWQ比GPTQ快2-3倍）
model.quantize(tokenizer, quant_config=quant_config)

# Step 4: 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 加载和推理
model_awq = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
```

#### 8.3.2.4 AWQ vs GPTQ 对比

| 特性 | AWQ | GPTQ |
|------|-----|------|
| **量化速度** | ⚡ 快（5-10分钟/7B） | 慢（20-30分钟/7B） |
| **精度保持** | ✅ 更好（特别是极低比特） | ✅ 好 |
| **推理速度** | ⚡⚡ 更快（优化kernel） | ⚡ 快 |
| **显存占用** | 相同 | 相同 |
| **校准数据** | 少（~256样本） | 多（~1024样本） |

**推荐选择**：
- 追求极致精度 → **AWQ**
- 快速量化部署 → **AWQ**
- 需要广泛硬件支持 → **GPTQ**（生态更成熟）

---

### 8.3.3 SmoothQuant：平滑激活值分布

#### 8.3.3.1 问题：为什么激活值难量化？

大模型的激活值存在**严重的异常值（Outliers）**问题：

- 99.9%的值在 $[-10, 10]$ 范围
- 0.1%的异常值可达 $[-1000, 1000]$

如果用统一的scale量化 → 大部分值被压缩到很小范围 → 精度损失巨大。

#### 8.3.3.2 SmoothQuant 解决方案

**核心思想**：通过缩放因子将量化难度从激活值转移到权重：

$$
Y = (X \text{ diag}(s)^{-1}) \cdot (\text{diag}(s)W)
$$

其中：
- $s$ 是per-channel的平滑因子
- $X \text{ diag}(s)^{-1}$：压缩激活值的异常值
- $\text{diag}(s)W$：将缩放因子吸收到权重中（权重更易量化）

**确定平滑因子**：

$$
s_i = \max(|X_i|)^\alpha / \max(|W_i|)^{1-\alpha}
$$

其中 $\alpha \in [0, 1]$ 控制平滑强度（通常0.5）。

#### 8.3.3.3 实现示例

```python
import torch
from smoothquant import SmoothQuantForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"

# Step 1: 加载模型
model = SmoothQuantForCausalLM.from_pretrained(model_name)

# Step 2: 收集激活值统计（需要运行校准数据）
from datasets import load_dataset
calibration_data = load_dataset("c4", split="train[:512]")

# 运行forward获取激活值分布
model.collect_stats(calibration_data)

# Step 3: 应用SmoothQuant（计算并吸收平滑因子）
model.smooth_quant(alpha=0.5)

# Step 4: INT8量化（权重+激活值）
model.quantize_int8()

# 保存量化模型
model.save_quantized("llama-2-7b-smoothquant-int8")
```

#### 8.3.3.4 SmoothQuant 性能

| 任务 | FP16 | W8A8<br>(SmoothQuant) | W8A8<br>(Naive) |
|------|------|----------------------|----------------|
| **MMLU** | 45.3 | 44.8 (-0.5) | 38.2 (-7.1) |
| **GSM8K** | 15.2 | 14.7 (-0.5) | 8.3 (-6.9) |
| **HumanEval** | 12.8 | 12.2 (-0.6) | 7.3 (-5.5) |

**关键优势**：
- ✅ 首个成功的 **W8A8**（权重+激活都INT8）方案
- ✅ 硬件友好（无需特殊kernel）
- ✅ 端到端INT8推理（2-3倍加速）

---

## 8.4 🎓 量化感知训练（QAT）

### 8.4.1 QLoRA：4-bit量化 + LoRA微调

QLoRA 是当前最流行的量化训练方法，实现了**在单张A100上微调65B模型**。

#### 8.4.1.1 核心技术组合

1. **4-bit NormalFloat (NF4) 量化**
2. **双重量化（Double Quantization）**
3. **分页优化器（Paged Optimizers）**
4. **LoRA 适配器**

#### 8.4.1.2 NF4量化：信息理论最优量化

标准的4-bit整数量化范围：$[-7, 7]$（均匀分布）

**问题**：神经网络权重通常服从正态分布 $\mathcal{N}(0, \sigma^2)$，均匀量化浪费表示能力。

**NF4方案**：量化点按正态分布分位数分布

```python
# NF4的16个量化点（针对N(0,1)优化）
NF4_QUANT_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]
```

**归一化 + NF4量化**：

$$
\hat{W} = \text{NF4}\left(\frac{W}{\sigma_W}\right) \cdot \sigma_W
$$

**优势**：相比均匀量化，精度损失减少约30%。

#### 8.4.1.3 双重量化（Double Quantization）

**问题**：FP32的缩放因子 $s$ 占用大量内存（每64个参数1个FP32）。

**解决**：对缩放因子本身也量化！

```python
# 第一次量化：权重 → INT4
W_quant = quantize_nf4(W, scale_fp32)

# 第二次量化：缩放因子 FP32 → FP8
scale_quant = quantize_fp8(scale_fp32)

# 显存节省：64个参数的scale从4字节 → 0.5字节
```

#### 8.4.1.4 完整QLoRA训练流程

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ============================================================
# Step 1: 4-bit量化配置（NF4 + 双重量化）
# ============================================================
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 启用4-bit量化
    bnb_4bit_quant_type="nf4",              # 使用NF4量化（信息理论最优）
    bnb_4bit_use_double_quant=True,         # 双重量化（scale也量化）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用BF16（保持精度）
)

# ============================================================
# Step 2: 加载4-bit量化模型（70B模型仅需35GB显存！）
# ============================================================
model_name = "meta-llama/Llama-2-70b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",                      # 自动多卡分配
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================
# Step 3: 准备量化模型用于训练
# ============================================================
model = prepare_model_for_kbit_training(model)

# ============================================================
# Step 4: 配置LoRA参数（只训练0.1%参数）
# ============================================================
lora_config = LoraConfig(
    r=64,                                   # LoRA秩（越大越接近全参数微调）
    lora_alpha=16,                          # 缩放因子（通常为r的1/4）
    target_modules=[                        # 对哪些层应用LoRA
        "q_proj", "k_proj", "v_proj",       # 注意力层
        "o_proj",
        "gate_proj", "up_proj", "down_proj" # FFN层
    ],
    lora_dropout=0.05,                      # Dropout防止过拟合
    bias="none",                            # 不训练bias
    task_type="CAUSAL_LM"
)

# ============================================================
# Step 5: 应用LoRA到量化模型
# ============================================================
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例：
# trainable params: 134,217,728 || all params: 68,976,648,192 || trainable%: 0.19%

# ============================================================
# Step 6: 准备训练数据
# ============================================================
dataset = load_dataset("timdettmers/openassistant-guanaco")

# ============================================================
# Step 7: 配置训练参数
# ============================================================
training_args = TrainingArguments(
    output_dir="./qlora-llama-70b",
    num_train_epochs=3,
    per_device_train_batch_size=4,         # 小batch（4-bit量化后显存充足）
    gradient_accumulation_steps=4,         # 等效batch_size=16
    learning_rate=2e-4,                    # QLoRA典型学习率
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",              # 分页优化器（防止OOM）
    fp16=False,
    bf16=True,                             # 使用BF16（A100推荐）
)

# ============================================================
# Step 8: 开始训练
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=2048,
)

trainer.train()

# ============================================================
# Step 9: 保存LoRA适配器（仅~200MB！）
# ============================================================
model.save_pretrained("./qlora-adapter")

# ============================================================
# Step 10: 推理时合并LoRA（可选）
# ============================================================
from peft import PeftModel

# 加载基座模型（4-bit量化）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 加载并合并LoRA
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")
model = model.merge_and_unload()  # 合并LoRA权重到基座
```

#### 8.4.1.5 QLoRA 显存占用分析

以 **LLaMA-2-70B** 为例：

| 组件 | FP16全参数微调 | QLoRA (4-bit) | 节省 |
|------|---------------|---------------|------|
| **模型权重** | 140 GB | 35 GB | -75% |
| **优化器状态** | 280 GB | 7 GB (8-bit paged) | -97% |
| **梯度** | 140 GB | 0.28 GB (LoRA only) | -99% |
| **激活值** | ~40 GB | ~10 GB (小batch) | -75% |
| **总计** | **~600 GB** | **~52 GB** | **-91%** |

**结论**：QLoRA让70B模型微调从 **8×A100 降到 1×A100**！

---

### 8.4.2 BitNet：1.58-bit极限量化

BitNet 将量化推向极限：**每个权重只用1.58 bit**（三值：-1, 0, +1）。

#### 8.4.2.1 核心设计

**三值权重**：

$$
W \in \{-1, 0, +1\}
$$

**激活值**：8-bit量化（保持一定精度）

**为什么是1.58 bit？**

熵计算：$H = -\sum p_i \log_2 p_i$

如果 $p(-1) = p(+1) = 0.4, p(0) = 0.2$：

$$
H = -2 \times 0.4 \times \log_2(0.4) - 0.2 \times \log_2(0.2) \approx 1.52 \text{ bits}
$$

#### 8.4.2.2 训练方法

1. **前向传播**：使用三值权重

$$
W_{\text{ternary}} = \text{sign}(W_{\text{float}}) \cdot \gamma
$$

其中 $\gamma = \frac{1}{n}\sum |W|$ 是缩放因子。

2. **反向传播**：对浮点权重计算梯度（STE技巧）

3. **权重更新**：更新浮点权重，再投影到三值

#### 8.4.2.3 性能表现

| 模型 | 参数量 | 精度 | 困惑度 | 推理速度 |
|------|--------|------|--------|---------|
| LLaMA-13B | 13B | FP16 | 5.12 | 1× |
| BitNet-13B | 13B | 1.58-bit | 5.41 (+5.7%) | **4.5×** |

**适用场景**：
- 📱 端侧部署（手机、IoT设备）
- ⚡ 超低延迟推理
- 💰 极致成本优化

---

## 8.5 🛠️ 量化工程实践

### 8.5.1 量化工具生态

| 工具 | 支持方法 | 特点 | 推荐场景 |
|------|---------|------|---------|
| **Unsloth** | LoRA, QLoRA, 4-bit, 16-bit, FP8, GRPO | 2×加速，减少70% VRAM，兼容HF生态 | 消费级GPU高效微调 |
| **bitsandbytes** | QLoRA, 8-bit, 4-bit | 易用，Hugging Face集成 | QLoRA微调 |
| **auto-gptq** | GPTQ | 成熟，广泛支持 | PTQ部署 |
| **AutoAWQ** | AWQ | 速度快，精度高 | PTQ部署 |
| **llama.cpp** | GGUF/GGML | CPU推理优化 | 本地CPU部署 |
| **vLLM** | FP8, INT8 | 高吞吐推理 | 生产部署 |
| **TensorRT-LLM** | INT8, INT4, FP8 | NVIDIA优化 | NVIDIA GPU部署 |

### 8.5.2 量化流程最佳实践

#### 8.5.2.1 选择量化方法的决策树

```mermaid
graph TD
    A[需要量化?] -->|是| B{有训练资源?}
    A -->|否| Z[使用FP16/BF16]

    B -->|有| C{追求极致精度?}
    B -->|无| D[使用PTQ]

    C -->|是| E[QLoRA/QAT<br>4-bit训练]
    C -->|否| F[LoRA微调<br>BF16]

    D --> G{目标比特数?}

    G -->|INT8| H[SmoothQuant<br>权重+激活]
    G -->|INT4| I{优先级?}

    I -->|精度| J[AWQ]
    I -->|速度| K[GPTQ]

    style E fill:#c8e6c9
    style J fill:#ffe0b2
    style K fill:#ffe0b2
    style H fill:#fff9c4
```

#### 8.5.2.2 量化前的模型评估

```python
from transformers import AutoModelForCausalLM
from lm_eval import simple_evaluate

model_name = "meta-llama/Llama-2-7b-hf"

# 评估原始FP16模型（建立baseline）
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

results_fp16 = simple_evaluate(
    model=model_fp16,
    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge"],
    num_fewshot=0,
)

print("FP16 Baseline:", results_fp16)
# 输出示例：
# {
#   'hellaswag': {'acc': 0.5832},
#   'winogrande': {'acc': 0.7103},
#   'arc_easy': {'acc': 0.7742},
#   'arc_challenge': {'acc': 0.4616}
# }
```

#### 8.5.2.3 量化后的精度验证

```python
from auto_gptq import AutoGPTQForCausalLM

# 加载GPTQ-4bit量化模型
model_gptq = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-4bit",
    device="cuda:0"
)

results_gptq = simple_evaluate(
    model=model_gptq,
    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge"],
    num_fewshot=0,
)

# 计算精度损失
for task in results_fp16:
    fp16_acc = results_fp16[task]['acc']
    gptq_acc = results_gptq[task]['acc']
    loss = (fp16_acc - gptq_acc) / fp16_acc * 100
    print(f"{task}: {fp16_acc:.4f} → {gptq_acc:.4f} ({loss:+.2f}%)")

# 期望输出：
# hellaswag: 0.5832 → 0.5784 (-0.82%)
# winogrande: 0.7103 → 0.7056 (-0.66%)
# arc_easy: 0.7742 → 0.7701 (-0.53%)
# arc_challenge: 0.4616 → 0.4548 (-1.47%)
```

**可接受的精度损失**：
- ✅ **INT8**：<2%
- ✅ **INT4**：<5%（GPTQ/AWQ）
- ⚠️ **INT4**：5-10%（简单量化）
- ❌ **INT4**：>10%（量化失败，需调整策略）

#### 8.5.2.4 处理量化失败的层

某些层对量化极度敏感（如第一层embedding，最后一层LM head）。

**策略：混合精度量化**

```python
from auto_gptq import BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,

    # 指定不量化的层（保持FP16）
    modules_to_not_convert=[
        "model.embed_tokens",              # 第一层embedding
        "model.norm",                      # 最终LayerNorm
        "lm_head"                          # 输出层
    ],
)

# 这些关键层保持FP16，其他层4-bit量化
# 精度损失：10% → 2%，显存增加：3.5GB → 4.2GB（+20%，可接受）
```

### 8.5.3 量化的常见陷阱

#### 8.5.3.1 ❌ 陷阱1：使用过少的校准数据

```python
# 错误示例
calibration_data = dataset[:10]  # 只用10个样本 ❌

# 正确做法
calibration_data = dataset[:512]  # 至少128-1024个样本 ✅
```

#### 8.5.3.2 ❌ 陷阱2：校准数据分布不匹配

```python
# 错误示例：用代码数据校准对话模型 ❌
calibration_data = load_dataset("codeparrot/github-code")

# 正确做法：用目标任务相似的数据 ✅
calibration_data = load_dataset("allenai/c4", split="train[:1000]")  # 通用文本
# 或
calibration_data = load_dataset("OpenAssistant/oasst1")  # 对话数据
```

#### 8.5.3.3 ❌ 陷阱3：忽略量化后的数值稳定性

```python
# 量化后可能导致数值溢出或下溢
# 解决方案：在关键位置添加裁剪

def forward_with_clipping(x):
    x = self.attention(x)
    x = torch.clamp(x, min=-10, max=10)  # 防止异常值 ✅
    x = self.ffn(x)
    return x
```

---

## 8.6 📊 量化效果对比总结

### 8.6.1 不同量化方法的精度-效率权衡

```mermaid
graph LR
    A[精度<br>100%] --> B[FP32/FP16]
    B --> C[BF16<br>99.5%]
    C --> D[INT8 PTQ<br>98%]
    D --> E[INT8 QAT<br>99%]
    D --> F[INT4 AWQ<br>95%]
    F --> G[INT4 GPTQ<br>94%]
    F --> H[INT4 QLoRA<br>96%]
    G --> I[INT2 BitNet<br>85%]

    style B fill:#e8f5e9
    style C fill:#e8f5e9
    style E fill:#fff9c4
    style H fill:#fff9c4
    style F fill:#ffe0b2
    style I fill:#ffcdd2
```

### 8.6.2 推荐矩阵

| 场景 | 推荐方法 | 量化精度 | 预期损失 | 显存节省 |
|------|---------|---------|---------|---------|
| **微调训练** | QLoRA | 4-bit | &lt;3% | 75% |
| **推理部署（精度优先）** | AWQ | 4-bit | &lt;2% | 75% |
| **推理部署（速度优先）** | GPTQ | 4-bit | &lt;3% | 75% |
| **边缘设备** | INT8 QAT | 8-bit | &lt;1% | 50% |
| **端侧极限** | BitNet | 1.58-bit | ~10% | 90% |
| **生产高吞吐** | SmoothQuant W8A8 | 8-bit | &lt;2% | 75% |

---

## 8.7 🔮 量化技术的未来趋势

### 8.7.1 混合精度量化（Mixed-Precision Quantization）

不同层使用不同比特数：
- **敏感层**（注意力层）：INT8
- **普通层**（FFN）：INT4
- **不敏感层**：INT2

### 8.7.2 训练后零样本量化（Zero-Shot PTQ）

无需校准数据，直接从权重分布推断量化参数。

### 8.7.3 硬件-算法协同设计

- **NVIDIA Hopper**：原生支持FP8训练
- **Google TPU v5**：INT4矩阵乘法加速
- **专用NPU**：二值/三值神经网络专用芯片

### 8.7.4 大模型特定优化

- **稀疏性+量化**：结合剪枝，进一步压缩
- **KV Cache量化**：长上下文场景的关键优化（详见下节）
- **动态量化**：根据输入自适应调整量化精度

### 8.7.5 KV Cache 显存管理

#### 8.7.5.1 为什么 KV Cache 会占用大量显存？

大模型推理包含两个阶段：
- **Prefill（预填充）**：将整个输入序列一次性并行处理，生成初始 K/V 矩阵
- **Decode（解码）**：每次只生成一个 token，但需要访问所有历史 token 的 K/V

为避免重复计算历史 token 的 K/V，推理时将其缓存，这就是 KV Cache。随着序列变长，缓存持续增长，成为长上下文场景下显存的主要消耗来源。

<div class="mermaid">
flowchart LR
    subgraph prefill["① Prefill（并行处理）"]
        direction TB
        p1["输入序列<br/>token₁ token₂ … tokenₙ"] --> p2["Transformer<br/>并行计算所有 token"]
        p2 --> p3["KV Cache 初始化<br/>K₁V₁, K₂V₂, …, KₙVₙ"]
        p3 --> p4["输出第一个新 token"]
    end

    subgraph decode["② Decode（逐 token 自回归）"]
        direction TB
        d1["新 token<br/>tokenₙ₊₁"] --> d2["读取 KV Cache<br/>（跳过历史重算）"]
        d2 --> d3["追加<br/>Kₙ₊₁Vₙ₊₁ 到缓存"]
        d3 --> d4["输出下一个 token"]
        d4 -->|"继续循环"| d1
    end

    prefill -->|"缓存传递"| decode

    style p3 fill:#e3f2fd,stroke:#01579b
    style d2 fill:#e3f2fd,stroke:#01579b
    style d3 fill:#e8f5e9,stroke:#1b5e20
</div>

**显存计算公式**：

$$\text{KV Cache大小} = 2 \times L \times H \times d \times n \times \text{bytes\_per\_element}$$

其中 $L$ = 层数，$H$ = 注意力头数，$d$ = 每头维度，$n$ = 序列长度

以 **Gemma 2 27B** 为例（46 层，30 头，128 维，FP16）：
- 每 token 占用：$2 \times 46 \times 30 \times 128 \times 2 \approx \mathbf{0.72 \text{ MB}}$
- A100（80GB）最多容纳约 **114,000 tokens**

#### 8.7.5.2 KV Cache 量化

直接对缓存的 K/V 矩阵进行低精度量化，无需修改模型结构：

| 精度 | 方法 | 显存节省 | 精度影响 |
|------|-----|---------|--------|
| INT8 | Per-token 动态量化 | ~50% | 极小 |
| INT4 | Per-group 量化 | ~75% | 小 |
| FP8 | 硬件原生（H100）| ~50% | 极小 |

#### 8.7.5.3 GQA / MQA：从架构层面减少 KV Cache

比量化更根本的优化：减少 KV 头的数量。

<div align="center">
  <img src="/images/llm-training/GQA.png" width="85%" />
  <figcaption>图：MHA（左）、GQA（中）、MQA（右）三种注意力机制的 KV 头共享方式对比。来源：Ainslie et al., GQA, EMNLP 2023</figcaption>
</div>

```
MHA: [Q1 K1V1] [Q2 K2V2] [Q3 K3V3] [Q4 K4V4]  ← KV头数 = Q头数
GQA: [Q1 Q2] [K1V1]  [Q3 Q4] [K2V2]            ← KV头数 = Q头数 / 组大小
MQA: [Q1 Q2 Q3 Q4]   [K1V1]                    ← 所有 Q 共享 1 个 KV 头
```

| 方式 | 全称 | KV Cache 大小 | 代表模型 |
|------|-----|-------------|--------|
| MHA | Multi-Head Attention | 基准（1×）| GPT-2, BERT |
| GQA | Grouped-Query Attention | 减少 $g$ 倍 | LLaMA-2-70B, Mistral |
| MQA | Multi-Query Attention | 减少 $H$ 倍 | PaLM, Falcon |

GQA 是目前主流大模型的首选方案，在显存节省与模型质量之间取得较好平衡。

#### 8.7.5.4 其他 KV Cache 节省技术

- **Sliding Window Attention**：每层只保留最近 $w$ 个 token 的 KV，多层堆叠后感受野仍可覆盖较长上下文
- **Streaming LLM**：保留最近窗口 + 最初几个 token 的 KV（初始 token 的注意力分数对整体分布有锚定作用）
- **Prefix Caching（跨请求复用）**：相同前缀（如系统提示）的请求共享 KV Cache，在 API 服务场景中可节省 50%+ 成本

---

# 9. 数据工程

数据是大模型训练的基石。高质量的训练数据直接决定了模型的能力上限。本章系统介绍大模型训练中的数据工程技术。

> **🎯 本章导读**
>
> "Garbage in, garbage out"——数据质量直接决定模型上限。本章介绍数据采集、质量评估、去重、配比、Tokenization等完整数据处理流程。**核心理念**：高质量 > 大规模，多样性 > 单一来源。数据工程往往被低估，但它是训练成功的关键。

<div align="center">
  <img src="/images/llm-training/data-processing-pipeline.png" width="90%" alt="数据处理完整流程" />
  <figcaption>图：预训练数据处理完整流程 - 从采集到训练（来源：RefinedWeb 论文 Figure 1）</figcaption>
</div>

## 9.1 数据采集与来源

### 9.1.1 预训练数据来源

#### 9.1.1.1 网页数据
- **Common Crawl**：最大的开放网页爬取数据集，每月爬取数十亿网页，包含多语言、多领域内容
- **C4（Colossal Clean Crawled Corpus）**：基于Common Crawl清洗后的数据集，~750GB文本
- **RedPajama**：开源的LLaMA训练数据复现（1.2万亿token）

#### 9.1.1.2 代码数据
- **GitHub**：开源代码仓库（过滤星标、license）
- **Stack Overflow**：高质量代码问答
- **The Stack**：3TB源代码，30+编程语言

#### 9.1.1.3 学术与书籍
- **arXiv、PubMed**：学术论文
- **Books3、BookCorpus**：书籍数据集

#### 9.1.1.4 对话与社交媒体
- **Reddit**：高质量讨论和问答
- **Wikipedia**：结构化知识
- **StackExchange**：各领域专业问答

## 9.2 数据质量评估

### 9.2.1 启发式规则过滤

#### 9.2.1.1 文本长度与格式
- 最小/最大长度过滤
- 特殊字符、数字、标点比例控制
- 大写字母密度检测
- 语言检测（fastText、langdetect）

#### 9.2.1.2 重复内容检测
- 行级重复、段落重复
- 模板识别（网页模板、页眉页脚）

### 9.2.2 基于模型的质量评分

#### 9.2.2.1 困惑度（Perplexity）过滤
- 使用小型语言模型评分
- 过滤困惑度过高的文档（需谨慎）

#### 9.2.2.2 质量分类器
- 训练数据：人工标注高质量 vs 低质量样本
- 特征：文本流畅度、信息密度、语法正确性
- 模型：FastText、BERT分类器

#### 9.2.2.3 教育价值评分
- Phi系列的启发：评估"教科书质量"
- 使用GPT-4等强模型评分

### 9.2.3 毒性与有害内容检测

#### 9.2.3.1 毒性检测
- **Perspective API**：Google毒性评分API
- **Detoxify**：开源毒性检测模型

#### 9.2.3.2 PII（个人身份信息）去除
- 姓名、地址、电话、邮箱
- 使用NER模型识别
- 密码、密钥：正则表达式匹配

## 9.3 数据去重

去重是最关键的数据处理步骤，可显著提升模型性能并减少记忆效应。

### 9.3.1 精确去重
- **文档级**：基于MD5/SHA256 hash
- **URL去重**：处理重定向和规范化

### 9.3.2 模糊去重

#### 9.3.2.1 MinHash + LSH
- 估计Jaccard相似度
- 步骤：生成shingles → MinHash签名 → LSH找相似对
- 工具：datasketch库

<div align="center">
  <img src="/images/llm-training/minhash-deduplication.png" width="85%" alt="MinHash去重原理" />
  <figcaption>图：MinHash + LSH去重工作原理示意图（来源：CCNet 论文 Figure 1）</figcaption>
</div>

#### 9.3.2.2 SimHash
- 快速计算文档指纹
- 汉明距离判断相似度

#### 9.3.2.3 Suffix Array
- 寻找最长公共子串
- CCNet方法

### 9.3.3 跨数据集去重

#### 9.3.3.1 训练集与测试集去重
- **至关重要**：避免数据泄露
- 13-gram重叠检测（GPT-3）
- 影响评测可信度

### 9.3.4 去重Trade-off
- 过度去重：损失多样性
- 欠去重：浪费资源、增加记忆风险
- 平衡点：根据任务调整

## 9.4 数据配比与采样

### 9.4.1 静态配比策略

**GPT-3配比示例**：

```mermaid
pie title GPT-3预训练数据配比
    "Common Crawl" : 60
    "WebText2" : 22
    "Books" : 16
    "Wikipedia" : 3
```

**LLaMA配比示例**（更新的配比策略）：

| 数据源 | 占比 | Token数量 | 说明 |
|--------|------|-----------|------|
| CommonCrawl | 67% | ~1.34T | 网页数据，多样性最高 |
| C4 | 15% | ~300B | 清洗后的网页数据 |
| GitHub | 4.5% | ~90B | 代码数据 |
| Wikipedia | 4.5% | ~90B | 高质量百科知识 |
| Books | 4.5% | ~90B | 长文本，叙事能力 |
| ArXiv | 2.5% | ~50B | 数学、科学推理 |
| StackExchange | 2% | ~40B | 专业问答 |

**配比原则**：
- **高质量数据提权**：Wikipedia、Books、ArXiv虽然占比小，但多次采样
- **代码数据单独控制**：10-20%，提升代码能力但不过度
- **对话数据少量但重要**：StackExchange等问答数据培养对话能力
- **多样性优先**：Common Crawl占主导，保证知识广度

### 9.4.2 动态配比策略

#### 9.4.2.1 训练阶段调整
- **早期（0-70%）**：均衡配比
- **中期（70-90%）**：提升高质量数据
- **后期（90-100%）**：专注专业数据

#### 9.4.2.2 基于损失的调整
- 监控不同数据源loss
- 动态再平衡

### 9.4.3 课程学习（Curriculum Learning）
- 从易到难：按困惑度排序
- 从通用到专业

### 9.4.4 Temperature Sampling

采样概率计算公式：

$$
p_i = \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}
$$

其中：
- $n_i$：数据源 $i$ 的原始样本数量
- $\alpha$：温度参数
- $\alpha < 1$：提升小数据源采样概率（up-sampling）
- $\alpha > 1$：降低小数据源采样概率（down-sampling）
- $\alpha = 1$：按原始比例采样

## 9.5 Tokenization

### 9.5.1 算法选择

- **BPE（Byte Pair Encoding）**：GPT系列、LLaMA
  - 从字符开始迭代合并高频pair
  - 平衡词表大小和分词粒度
- **WordPiece**：BERT，基于最大似然
- **Unigram**：T5，从大词表剪枝
- **SentencePiece**：语言无关，多语言模型首选

### 9.5.2 词表大小
- 英语为主：32k - 50k
- 多语言：100k - 250k
- 代码模型：更大词表

**Trade-off**：
- 词表过小：序列长，训练慢
- 词表过大：embedding参数多，稀疏

### 9.5.3 特殊Token
- 标准：`<bos>`, `<eos>`, `<pad>`, `<unk>`
- 对话：`<|user|>`, `<|assistant|>`, `<|system|>`
- 多模态：`<image>`, `<video>`

### 9.5.4 最佳实践
- 预训练、微调、推理使用相同tokenizer
- 预先tokenize并缓存
- 使用fast tokenizer（Rust实现）
- 多语言平衡token数量


# 10. 评估与基准测试

评估体系是衡量训练效果和指导训练方向的重要工具。本章介绍大模型训练中常用的评测基准。

## 10.1 语言理解与知识

### 10.1.1 MMLU（Massive Multitask Language Understanding）
- **内容**：57个学科的多选题（数学、历史、法律、医学等）
- **规模**：~16,000道题目
- **评估维度**：知识广度、跨学科理解
- **难度**：大学到专业水平
- **意义**：衡量模型的通用知识储备

### 10.1.2 HellaSwag
- **任务**：常识推理句子补全
- **方法**：从4个选项中选择最合理的句子结尾
- **特点**：对人类简单（~95%），对早期模型困难
- **评估**：常识理解和情境推理

### 10.1.3 TruthfulQA
- **目标**：评估模型输出的真实性
- **设计**：包含常见误解和虚假信息的问题
- **评估维度**：
  - 模型是否会重复训练数据中的错误
  - 是否会产生幻觉（Hallucination）
- **重要性**：衡量模型可靠性

### 10.1.4 ARC（AI2 Reasoning Challenge）
- **内容**：小学科学考试题
- **难度**：Easy和Challenge两个版本
- **评估**：科学推理和知识应用

## 10.2 推理能力

### 10.2.1 GSM8K（Grade School Math 8K）
- **任务**：小学数学应用题
- **规模**：8,500道题
- **特点**：需要多步推理
- **评估方法**：
  - Direct答案评估
  - Chain-of-Thought推理过程评估
- **意义**：衡量基础数学推理能力

### 10.2.2 MATH
- **难度**：高中到大学竞赛级别数学
- **规模**：12,500道题
- **学科**：代数、几何、概率、数论等
- **评估**：复杂数学推理和问题解决
- **挑战**：即使最强模型也难以达到高分

### 10.2.3 HumanEval
- **任务**：Python函数实现
- **规模**：164个编程问题
- **评估方法**：单元测试通过率（pass@k）
- **特点**：
  - 独立函数，不涉及复杂系统
  - 明确的输入输出规范
- **变体**：HumanEval+（更严格测试）

### 10.2.4 MBPP（Mostly Basic Python Problems）
- **规模**：1,000个Python编程问题
- **难度**：入门到中级
- **评估**：实用编程能力

### 10.2.5 BigCodeBench
- **特点**：更复杂的真实世界编程任务
- **评估**：工具使用、API调用、复杂逻辑

## 10.3 多语言能力

### 10.3.1 FLORES（Facebook Low Resource Translation）
- **任务**：机器翻译
- **覆盖**：200+语言对
- **评估**：跨语言理解和生成
- **重要性**：衡量多语言模型能力

### 10.3.2 XNLI（Cross-lingual Natural Language Inference）
- **任务**：自然语言推理（蕴含、矛盾、中立）
- **语言**：15种语言
- **评估**：零样本跨语言迁移能力

### 10.3.3 Belebele
- **任务**：阅读理解
- **覆盖**：122种语言
- **评估**：广泛的多语言理解

## 10.4 长文本能力

### 10.4.1 RULER（Rule-based Evaluation of Long Context Understanding）
- **任务类型**：
  - 信息检索（Needle in a Haystack）
  - 多跳推理
  - 聚合统计
- **长度**：4k到128k+ tokens
- **评估**：长上下文建模能力

### 10.4.2 LongBench
- **任务**：单文档/多文档QA、摘要、代码等
- **长度**：平均5-15k tokens
- **语言**：英文和中文
- **评估**：真实长文本应用场景

### 10.4.3 Needle in a Haystack（大海捞针）
- **设计**：在长文本中插入关键信息
- **测试**：模型能否准确检索
- **变体**：
  - 单needle
  - 多needle
  - 不同位置和深度

## 10.5 安全性评估

### 10.5.1 ToxiGen
- **目标**：检测有害内容生成倾向
- **方法**：对抗性prompt测试
- **评估维度**：
  - 毒性
  - 仇恨言论
  - 暴力内容

### 10.5.2 BBQ（Bias Benchmark for QA）
- **任务**：检测社会偏见
- **维度**：
  - 性别
  - 种族
  - 宗教
  - 年龄
  - 性取向等
- **方法**：模糊和明确上下文对比

### 10.5.3 SafetyBench
- **覆盖**：多类安全风险
- **语言**：中英文
- **评估**：拒绝回答不当请求的能力

## 10.6 对齐评估

### 10.6.1 MT-Bench
- **任务**：多轮对话评估
- **评判**：使用GPT-4作为评委
- **维度**：
  - 写作
  - 角色扮演
  - 推理
  - 数学
  - 编程等

### 10.6.2 AlpacaEval
- **方法**：与参考模型（如GPT-4）对比
- **评估**：指令跟随质量
- **输出**：胜率（Win Rate）

### 10.6.3 Chatbot Arena
- **方法**：人类盲评，Elo评分
- **特点**：持续更新的实时排行榜
- **意义**：反映真实用户偏好

## 10.7 综合评测平台

### 10.7.1 LM Evaluation Harness
- **维护**：Eleuther AI
- **特点**：
  - 统一接口
  - 支持几十个基准测试
  - 标准化评测流程
- **使用**：研究社区广泛采用

### 10.7.2 HELM（Holistic Evaluation of Language Models）
- **维度**：
  - 准确性
  - 鲁棒性
  - 公平性
  - 偏见
  - 效率
- **特点**：多维度全面评估

### 10.7.3 OpenCompass
- **维护**：上海AI Lab
- **特点**：
  - 中文优化
  - 支持大规模评测
  - 可视化排行榜

<div align="center">
  <img src="/images/llm-training/benchmark-comparison.png" width="90%" alt="主流模型评估对比" />
  <figcaption>图：主流大模型在各基准测试上的性能对比（MMLU/GSM8K/HumanEval等）</figcaption>
</div>

## 10.8 评测的最佳实践

### 10.8.1 避免数据泄露
- 训练数据与测试集严格去重
- 使用新发布的基准测试
- 定期更新评测集

### 10.8.2 多维度评估
- 不依赖单一指标
- 综合考虑性能、安全、效率
- 关注长尾能力和边界情况

### 10.8.3 评测与训练的关系
- 评测结果指导训练方向
- 避免过度针对基准优化（刷榜）
- 关注实际应用场景表现

---

# 11. 实践指南与最佳实践

本章提供实际训练大模型的工程指导，包括硬件配置、成本估算、监控调试和代码示例。

## 11.1 硬件配置建议

### 11.1.1 不同规模模型的推荐配置

| 模型规模 | 参数量 | 推荐GPU | 数量 | 显存需求 | 互联网络 | 适用场景 |
|---------|--------|---------|------|---------|---------|---------|
| **极小(MiniMind)** | 26M-500M | RTX 3060/4060 | 1 | 8GB+ | PCIe 3.0 | 学习原理、极速验证 |
| **小型** | 1-3B | RTX 4090 | 4-8 | 24GB | PCIe 4.0 | 研究、原型开发 |
| **中型** | 7-13B | A100 | 16-32 | 40GB/80GB | InfiniBand | 企业应用 |
| **大型** | 30-70B | A100/H100 | 64-256 | 80GB | InfiniBand | 生产级模型 |
| **超大型** | 175B+ | H100 | 512-4096 | 80GB | InfiniBand/NVLink | 前沿研究 |

### 11.1.2 GPU选择指南

**训练场景**：
- **入门/个人实践**：RTX 3060 (12GB) / 4060 Ti (16GB) —— 能够完整跑通 MiniMind 等 SLM 项目。
- **预算有限**：RTX 4090（24GB，性价比高，适合小模型和LoRA微调）
- **企业级训练**：A100（80GB，成熟稳定，生态完善）
- **最新旗舰**：H100（80GB，性能最强，FP8支持，适合大规模训练）

**网络互联**：
- **节点内通信**：NVLink（900GB/s）> PCIe 5.0（128GB/s）
- **节点间通信**：InfiniBand（200-400Gb/s）> Ethernet（100Gb/s）


## 11.2 训练成本估算

### 11.2.1 成本构成

```mermaid
pie title 大模型训练成本构成
    "GPU租用/折旧" : 70
    "电力与冷却" : 15
    "存储与网络" : 5
    "人力成本" : 8
    "其他" : 2
```

### 11.2.2 详细成本计算

#### 11.2.2.1 预训练成本估算

**公式**：

$$
\text{总成本} = \text{GPU成本} \times \text{数量} \times \text{训练时长} + \text{其他成本}
$$

**具体计算示例**：

**案例1：训练一个7B模型**
```
模型规模：7B参数
训练数据：1T tokens
GPU配置：64 × A100 (80GB)
训练时长：约2周（336小时）

成本估算：
- GPU成本：$2.5/小时/卡（云服务）
- GPU总成本：64 × $2.5 × 336 = $53,760
- 存储成本：10TB数据 × $0.02/GB/月 ≈ $200
- 网络成本：忽略不计
- 人力成本：1人 × 2周 × $5000/周 = $10,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总成本：约 $64,000
```

**案例2：训练一个70B模型**
```
模型规模：70B参数
训练数据：2T tokens
GPU配置：512 × H100 (80GB)
训练时长：约6周（1008小时）

成本估算：
- GPU成本：$4/小时/卡（云服务H100）
- GPU总成本：512 × $4 × 1008 = $2,064,384
- 存储成本：50TB × $0.02/GB/月 × 1.5 ≈ $1,500
- 人力成本：3人 × 6周 × $5000/周 = $90,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总成本：约 $2,160,000 (216万美元)
```

**案例3：微调（LoRA）成本**
```
基础模型：70B参数
微调方法：QLoRA
数据规模：50k样本
GPU配置：1 × A100 (80GB)
训练时长：约12小时

成本估算：
- GPU成本：$2.5/小时
- GPU总成本：1 × $2.5 × 12 = $30
- 数据标注：50k × $0.5 = $25,000
- 人力成本：$2,000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总成本：约 $27,030
```

### 11.2.3 成本优化策略

#### 11.2.3.1 硬件优化
- 使用Spot/Preemptible实例（节省50-70%）
- 混合精度训练（BF16/FP8）
- Flash Attention减少显存

#### 11.2.3.2 算法优化
- 梯度检查点（Gradient Checkpointing）
- 激活重计算（Activation Recomputation）
- ZeRO优化器（DeepSpeed）

#### 11.2.3.3 数据优化
- 高质量数据 > 海量数据
- 提前进行数据清洗和去重
- 使用数据缓存和预处理

#### 11.2.3.4 训练策略
- 从小模型开始验证（消融实验）
- 使用Learning Rate Finder
- 早期终止（Early Stopping）

> **⚠️ 成本陷阱**
>
> - **过度预训练**：收益递减，建议监控validation loss
> - **盲目扩大规模**：先验证小模型效果
> - **忽视数据质量**：垃圾数据浪费计算资源
> - **缺乏监控**：未及时发现训练异常导致浪费

## 11.3 训练监控与调试

<div align="center">
  <img src="/images/llm-training/training-loss-curve.png" width="85%" alt="训练loss曲线示例" />
  <figcaption>图：健康的训练loss曲线（平滑下降，无spike）（来源：LLaMA 论文 Figure 2）</figcaption>
</div>

### 11.3.1 关键监控指标

#### 11.3.1.1 损失函数（Loss）

```mermaid
graph LR
    A[Training Loss] --> B{下降趋势?}
    B -->|正常下降| C[✅ 继续训练]
    B -->|突然上升| D[⚠️ Loss Spike]
    B -->|震荡不稳| E[⚠️ 不稳定]
    B -->|不再下降| F[⚠️ 收敛/过拟合]

    D --> G[降低学习率]
    E --> H[调整batch size<br/>或学习率]
    F --> I[早停或调整]

    style C fill:#c8e6c9
    style D fill:#ffcdd2
    style E fill:#ffcdd2
    style F fill:#fff9c4
```

**正常Loss曲线特征**：
- 稳定下降，无剧烈波动
- 预训练loss：通常从8-10降至2-3
- Training loss < Validation loss（轻微过拟合正常）

**异常Loss模式**：

| 异常现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| **Loss Spike** | 坏数据、学习率过大、数值不稳定 | 回退checkpoint，降低LR，检查数据 |
| **Loss不下降** | 学习率过小、模型容量不足 | 增大LR，检查模型架构 |
| **Loss震荡** | Batch size过小、LR过大 | 增大batch或降低LR |
| **Loss=NaN** | 梯度爆炸、数值溢出 | 梯度裁剪、混合精度、检查数据 |

#### 11.3.1.2 梯度相关指标

**监控指标**：
- **梯度范数**（Gradient Norm）：应在0.1-10之间
- **梯度裁剪比例**：<5%表示健康
- **参数更新比例**：更新量应为参数值的0.1-1%

```python
def monitor_gradients(model):
    """计算梯度的L2范数，用于监控训练稳定性"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# 每100步记录一次梯度范数
if step % 100 == 0:
    grad_norm = monitor_gradients(model)
    logger.log({"gradient_norm": grad_norm})
    # 健康范围：0.1-10，超过100需要警惕
```

#### 11.3.1.3 学习率监控

```python
# 记录当前学习率（追踪调度器是否正确工作）
current_lr = optimizer.param_groups[0]['lr']
logger.log({"learning_rate": current_lr})
# 期望曲线：Warmup↗ → Stable— → Decay↘
```

#### 11.3.1.4 性能指标

- **吞吐量**（Tokens/second）：衡量训练速度
- **GPU利用率**：应保持在80-95%
- **显存占用**：避免OOM，保留10-15%缓冲
- **通信时间占比**：<20%为佳

### 11.3.2 常见问题诊断与解决

#### 11.3.2.1 问题1：Out of Memory (OOM)

**症状**：`CUDA out of memory` 错误

**解决方案** (按效果排序，逐步尝试)：
```python
# 方案1: 启用梯度检查点（节省50-80%激活值显存）
model.gradient_checkpointing_enable()

# 方案2: 减小batch size + 梯度累积（保持等效batch size）
per_device_batch_size = 1
gradient_accumulation_steps = 32

# 方案3: CPU卸载优化器状态（节省Adam的8字节/参数）
from deepspeed.ops.adam import DeepSpeedCPUAdam
optimizer = DeepSpeedCPUAdam(model.parameters())

# 方案4: ZeRO-3全参数分片（最强，但训练速度会降低）
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    }
}
```

#### 11.3.2.2 问题2：训练速度慢

**诊断步骤**：
```bash
# 1. 检查GPU利用率
nvidia-smi dmon -s u

# 2. 分析性能瓶颈
python -m torch.utils.bottleneck train.py

# 3. 使用profiler
python -m torch.profiler train.py
```

**常见瓶颈及解决**：
- **数据加载慢**：增加`num_workers`，使用预处理
- **通信慢**：检查网络，使用NCCL优化
- **计算慢**：检查Flash Attention是否启用

#### 11.3.2.3 问题3：Loss Spike

**应对流程**：
```mermaid
graph TD
    A[发现Loss Spike] --> B[立即停止训练]
    B --> C[回退到上一个good checkpoint]
    C --> D{分析原因}
    D --> E[检查数据]
    D --> F[检查学习率]
    D --> G[检查梯度]
    E --> H[过滤坏数据]
    F --> I[降低学习率50%]
    G --> J[启用梯度裁剪]
    H --> K[恢复训练]
    I --> K
    J --> K

    style B fill:#ffcdd2
    style C fill:#fff9c4
    style K fill:#c8e6c9
```

**预防措施**：
```python
# 技巧1: 梯度裁剪（防止梯度爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 技巧2: WSD学习率调度（Warmup-Stable-Decay，比纯Cosine更稳定）
# 在peak LR保持一段时间，而非立即衰减

# 技巧3: Loss Spike自动应对机制
if current_loss > moving_avg_loss * 1.5:  # loss突增50%
    print(f"⚠️ Loss spike detected: {current_loss:.4f}")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5  # 自动降低学习率
    # 可选：自动回退到上一个checkpoint
```

### 11.3.3 Checkpoint管理策略

#### 11.3.3.1 保存策略

```python
# Checkpoint保存配置（平衡安全性和存储成本）
checkpoint_config = {
    "save_interval": 1000,           # 每1000步保存（约1-2小时）
    "save_total_limit": 5,           # 保留最近5个（防止磁盘爆满）
    "save_on_each_node": False,      # 只在主节点保存
    "save_optimizer_state": True,    # 必须保存，否则无法恢复
}

# 额外保存最佳模型（按validation loss）
if current_loss < best_loss:
    save_checkpoint(model, "best_model.pt")
    best_loss = current_loss
```

#### 11.3.3.2 恢复训练

```python
# 从checkpoint恢复
def resume_training(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_step = checkpoint['step']
    start_epoch = checkpoint['epoch']

    # 恢复学习率调度器
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 恢复随机数种子
    torch.manual_seed(checkpoint['random_seed'])

    return start_step, start_epoch
```

> **💡 最佳实践**
>
> - 在训练开始前进行**小规模试运行**（100-1000步）
> - 设置**自动化监控告警**（Loss异常、GPU故障等）
> - 保留**多个历史checkpoint**，不要只保留最新的
> - 定期**手动检查**训练日志和可视化图表

## 11.4 个人级实践案例：MiniMind

### 11.4.1 项目地址
```
https://github.com/jingyaogong/minimind
```

如果说训练 Llama-3 是一场“烧钱”的豪赌，那么 **MiniMind** 项目则向我们展示了大模型训练的“平民化”可能。

### 11.4.2 为什么关注 MiniMind？
MiniMind 是一个极简的开源 LLM 训练项目，旨在让开发者在**单张消费级显卡**上，从零开始完成 LLM 的全流程训练。

### 11.4.3 核心数据对比

| 维度 | MiniMind (26M版) | Llama-3 (8B版) |
|------|-----------------|---------------|
| **参数量** | 2.6千万 | 80亿 |
| **硬件要求** | 1 x RTX 3060 | 8 x A100 (SFT) / 1024+ (PT) |
| **训练时长** | ~2 小时 | 数周 |
| **数据规模** | ~1B Tokens | 15T Tokens |
| **成本** | 几块钱电费 | 数十万美元 |

### 11.4.4 给初学者的启示
*   **掌握全流程比堆算力更重要**：通过 MiniMind，你可以亲手训练分词器、编写 Transformer 结构、执行从预训练到 DPO 对齐的每一个 Python 脚本。
*   **快速验证的想法**：如果你有一个新的 Loss 函数或一种新的位置编码，在 MiniMind 这样的小模型上验证速度是极快的。
*   **SLM 的潜力**：在垂直领域（如特定格式转换、逻辑提取），经过精调的超小模型（Small Language Model）同样能爆发惊人的表现。


---
# 12. 常见问题
————FAQ

本章汇总大模型训练中**最常遇到的问题及解答**，帮助快速解决实践中的困惑。

> 💡 **使用技巧**：善用 Ctrl+F 搜索关键词快速定位问题

## 12.1 🏗️ 预训练相关

### 12.1.1 Q1: 需要多少数据才能训练一个有用的模型？

**A:** 取决于模型规模和目标：

- **小模型（1-3B）**：
  - 最少：10-50B tokens可得到基本能力
  - 推荐：100-300B tokens获得较好效果
  - 例如：Phi-1使用7B高质量tokens就很强

- **中型模型（7-13B）**：
  - 推荐：500B-1T tokens
  - LLaMA-1 7B使用1T tokens

- **大型模型（70B+）**：
  - 推荐：1.5-2T tokens
  - LLaMA-2 70B使用2T tokens

**💎 关键洞察**：**质量 > 数量**。Phi系列证明了高质量小数据可以打败低质量大数据（7B tokens训出1.3B模型，性能媲美13B）。

---

### 12.1.2 Q2: 如何判断预训练是否收敛？

**A:** 观察以下指标：

1. **Training Loss**：
   - 不再明显下降（变化<0.01/1000步）
   - 典型收敛值：2.0-3.0（取决于数据）

2. **Validation Loss**：
   - 与training loss趋势一致
   - 如果validation loss上升但training loss下降 → 过拟合

3. **下游任务性能**：
   - 在基准测试上的表现不再提升
   - 这是最终判断标准

4. **训练步数经验值**：
   - 7B模型：100-200k步
   - 70B模型：50-100k步
   - 规模越大，所需步数越少

**建议**：预训练通常**不追求完全收敛**，因为成本极高且收益递减。在loss曲线趋缓后即可停止。

---

### 12.1.3 Q3: 出现Loss Spike怎么办？

**A:** Loss突然上升的应对流程：

**立即行动**：
```
1. 停止训练
2. 回退到spike前的checkpoint（通常回退2-3个checkpoint）
3. 分析spike原因
```

**常见原因及解决**：

| 原因 | 症状 | 解决方案 |
|------|------|----------|
| **坏数据** | 单次大幅上升 | 跳过该batch，增强数据过滤 |
| **学习率过大** | 逐步上升 | 降低LR至原来的50% |
| **数值不稳定** | 随机出现 | 启用BF16，增加梯度裁剪 |
| **梯度爆炸** | 伴随grad norm飙升 | 降低LR，启用gradient clipping |

<div align="center">
  <img src="/images/llm-training/loss-spike-example.png" width="80%" alt="Loss Spike案例" />
  <figcaption>图：Loss Spike现象与恢复策略示意图（来源：OPT 论文 Figure 3）</figcaption>
</div>

**Loss Spike预防代码**：
```python
# 策略1: 自动检测与回退到安全checkpoint
if loss > moving_average * 2.0:  # loss突增2倍，危险！
    print("⚠️ Loss spike! Rolling back to previous checkpoint...")
    load_checkpoint(previous_good_checkpoint)
    learning_rate *= 0.5  # 降低学习率再试

# 策略2: 梯度裁剪（限制单步更新幅度）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 策略3: 更稳定的Adam配置（beta2=0.95比0.999更稳定）
optimizer = AdamW(lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
```

---

### 12.1.4 Q4: 预训练可以超过1个epoch吗？

**A:** **不推荐**，原因如下：

1. **记忆效应**：模型会记住训练数据，降低泛化能力
2. **收益递减**：第2个epoch的性能提升远小于成本
3. **行业惯例**：主流大模型（GPT-3、LLaMA等）都是<1 epoch

**例外情况**：
- 数据量极小（<10B tokens）时可以多epoch
- 领域专用模型（如医疗、法律）可以在少量高质量数据上多次训练
- 但即使如此，也很少超过3-5 epochs

**替代方案**：如果数据有限，优先考虑：
- 提升数据质量和多样性
- 增加模型容量
- 采用更好的数据配比策略

---

## 12.2 🎨 监督微调相关
————SFT

### 12.2.1 Q5: LoRA和全参数微调如何选择？

**A:** 根据场景选择：

**全参数微调**（Full Fine-Tuning）：
- ✅ **适用场景**：
  - 有充足GPU资源
  - 需要最佳性能
  - 任务与预训练差异大
- ❌ **劣势**：
  - 显存需求大（需存储完整梯度和优化器状态）
  - 训练速度慢
  - 每个任务需要完整模型副本

**LoRA微调**：
- ✅ **适用场景**：
  - GPU资源有限
  - 需要训练多个任务adapter
  - 快速实验和迭代
- ❌ **劣势**：
  - 性能略低于全参数（通常差距<2%）
  - 需要调整额外超参数（r, alpha）

**性能对比**：
```
任务类型          全参数微调    LoRA (r=16)    差距
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
指令遵循          96.5%        95.8%         -0.7%
对话质量          92.3%        91.5%         -0.8%
代码生成          88.7%        87.2%         -1.5%
数学推理          76.4%        74.9%         -1.5%
```

**推荐策略**：
- 资源受限或快速实验 → **LoRA**
- 生产环境追求极致性能 → **全参数微调**
- 折中方案 → **QLoRA**（量化+LoRA）

---

### 12.2.2 Q6: 需要多少SFT数据？

**A:** 远少于预训练，质量比数量重要：

**数据量指南**：

| 数据规模 | 效果 | 适用场景 |
|---------|------|----------|
| **1k-5k** | 基本遵循指令 | 快速原型、特定任务 |
| **10k-30k** | 良好对话能力 | 通用助手 |
| **50k-100k** | 优秀多任务能力 | 生产级模型 |
| **100k+** | 边际收益递减 | 追求极致性能 |

**真实案例**：
- **Alpaca**：52k合成数据，达到不错效果
- **Vicuna**：70k ShareGPT对话，接近ChatGPT
- **LLaMA-2-Chat**：27.5k数据，效果优异

**关键洞察**：
> 1000条高质量、多样化的数据 > 10000条低质量重复数据

**数据质量标准**：
- ✅ 指令清晰明确
- ✅ 回答准确、有帮助
- ✅ 覆盖多种任务类型
- ✅ 格式统一规范
- ❌ 避免模板化回答
- ❌ 避免错误信息
- ❌ 避免有害内容

---

### 12.2.3 Q7: 如何避免微调时的过拟合？

**A:** 多种策略组合使用：

**1. 控制训练轮数**
```python
# 通常1-3个epoch足够
num_train_epochs = 2  # 推荐起点

# 监控validation loss，早停
early_stopping_patience = 3
```

**2. 使用较小学习率**
```python
# SFT学习率应远小于预训练
learning_rate = 2e-5  # 而不是3e-4
```

**3. 数据增强**
```python
# 释义改写
# 指令变化
# 回答风格多样化
```

**4. Dropout和正则化**
```python
model_config = {
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "weight_decay": 0.01
}
```

**5. 混合训练数据**
```python
# 加入10-20%预训练数据
mixed_dataset = {
    "sft_data": 0.8,
    "pretrain_data": 0.2  # 防止灾难性遗忘
}
```

**过拟合的症状**：
- Training loss持续下降，但validation loss上升
- 在训练集上表现完美，但测试集表现差
- 模型开始"背诵"训练样本

**诊断命令**：
```python
# 定期评估
if step % eval_steps == 0:
    train_loss = evaluate(model, train_dataset)
    val_loss = evaluate(model, val_dataset)

    if val_loss > best_val_loss:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered!")
            break
```

---

## 12.3 🔁 Post-Training 相关

### 12.3.1 Q8: Post-Training 后模型旧能力下降怎么办？

**A:** 这是**灾难性遗忘（Catastrophic Forgetting）**，Post-Training 最常见的问题。

**核心原因**：训练时只优化目标任务 loss，模型会"忘记"旧知识。即使数据完全无害，即使使用 LoRA，遗忘仍然会发生。

**解决方案（优先级从高到低）**：

**方案1：Self-Output 训练**（推荐，无需历史数据）
```python
# 对每条训练数据，先用 Foundation Model 自己生成答案
foundation_output = foundation_model.generate(question)

# 如果模型能回答正确，用自己的答案训练
# 如果回答错误，用人类标注答案训练
answer = foundation_output if is_correct(foundation_output) else human_answer
```

**方案2：经验回放（Experience Replay）**
```python
# 混入约 3-5% 的旧任务数据（或 Foundation Model 自生成的数据）
mixed_dataset = {
    "new_task_data": 0.97,
    "replay_data": 0.03,  # 历史数据 or Magpie 自生成数据
}
```

**方案3：Magpie 伪经验回放**（拿不到历史数据时）
```python
# 让 Foundation Model 自问自答，生成伪历史数据
pseudo_data = []
for _ in range(n):
    q = foundation_model.generate("<|user|>")      # 自己生成问题
    a = foundation_model.generate(q + "<|assistant|>")  # 自己回答
    pseudo_data.append((q, a))
```

**关键结论**：
- LoRA **不能真正防止**遗忘（只是学得少所以忘得少）
- 即使训练数据无害，Safety Alignment 也会崩溃
- 训练后**必须**在多个 benchmark 上评估原有能力

---

## 12.4 🎯 对齐相关
————RLHF/DPO

### 12.4.1 Q9: RLHF和DPO如何选择？

**A:** DPO通常是更好的选择：

**DPO优势**（推荐）：
- ✅ 训练更稳定（无RL的探索-利用困境）
- ✅ 实现更简单（不需要PPO、Reward Model）
- ✅ 计算效率高（只需2个模型 vs 4个模型）
- ✅ 效果相当或更好
- ✅ 超参数更鲁棒

**RLHF优势**：
- ✅ 理论基础深厚
- ✅ 可以在线收集新数据
- ✅ 适合复杂奖励信号

**性能对比**：
```
评估维度         RLHF    DPO     备注
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
训练稳定性       ⭐⭐⭐   ⭐⭐⭐⭐⭐  DPO明显更稳定
实现复杂度       ⭐⭐     ⭐⭐⭐⭐⭐  DPO简单很多
计算效率         ⭐⭐     ⭐⭐⭐⭐⭐  DPO快2-3倍
最终效果         ⭐⭐⭐⭐  ⭐⭐⭐⭐   效果相当
```

**推荐策略**：
- **首选DPO**：适合绝大多数场景
- **考虑RLHF**：需要在线学习或复杂奖励函数
- **新方法**：ORPO、SimPO等DPO变体

---

### 12.4.2 Q9: 偏好数据如何构建？

**A:** 三种主要方法：

**方法1：人工标注**（最高质量）
```
流程：
1. 采样prompts（从真实用户或构造）
2. 生成多个回答（通常4-8个）
3. 人工排序或对比标注
4. 构建偏好对：(prompt, chosen, rejected)

成本：$0.5-2/样本
规模：10k-50k偏好对
质量：⭐⭐⭐⭐⭐
```

**方法2：AI标注**（性价比极高，推荐）
```python
# RLAIF: 使用GPT-4等强模型作为"评判者"
prompt = f"""
Given the question: {question}

Response A: {response_a}
Response B: {response_b}

Which response is better? Consider helpfulness, accuracy, and safety.
Answer: A or B
"""

# 成本优势明显
成本：$0.01-0.05/样本（vs 人工的$5-20/样本）
规模：轻松扩展到100k+
质量：⭐⭐⭐⭐（约90%接近人类标注）
```

**方法3：合成构建**（快速启动）
```python
# 从已有SFT数据自动构造偏好对
chosen = high_quality_response
rejected = synthesize_negative(chosen)  # 负样本来源：
    # - 截断回答（模拟不完整）
    # - 注入事实错误
    # - 违反指令要求
    # - 添加有害内容

# 零成本快速启动
成本：几乎免费（无需API或人工）
规模：无限（自动生成）
质量：⭐⭐⭐（有效，但不如真实对比）
```

**混合策略**（推荐）：
```
核心数据(20%)：人工标注，确保高质量
扩展数据(60%)：AI标注，快速扩展规模
补充数据(20%)：合成数据，增加多样性
```

---

## 12.5 ⚙️ 工程实践相关

### 12.5.1 Q10: 如何选择合适的并行策略？

**A:** 遵循决策树：

```mermaid
graph TD
    A[开始选择] --> B{模型能否<br/>放入单GPU?}
    B -->|能| C[使用数据并行 DP]
    B -->|不能| D{单层参数<br/>是否过大?}

    D -->|是| E[启用张量并行 TP]
    D -->|否| F{层数<br/>是否很多?}

    E --> F
    F -->|是| G[启用流水线并行 PP]
    F -->|否| H[评估总GPU数]

    G --> H
    H --> I{还有<br/>剩余GPU?}
    I -->|是| J[增加数据并行度]
    I -->|否| K[完成配置]

    J --> K

    style C fill:#c8e6c9
    style E fill:#ffcdd2
    style G fill:#fff9c4
    style K fill:#e3f2fd
```

**实用配置表**：

| GPU总数 | 模型规模 | 推荐配置 | 说明 |
|---------|---------|---------|------|
| 8 | 7B | DP=8 | 纯数据并行 |
| 16 | 13B | TP=2, DP=8 | 轻量TP |
| 64 | 30B | TP=4, PP=2, DP=8 | 2D并行 |
| 128 | 70B | TP=8, PP=4, DP=4 | 3D并行 |
| 512 | 175B | TP=8, PP=16, DP=4 | 深度3D并行 |

**配置验证公式**：
```python
# 关键公式：总GPU数 = 数据并行 × 张量并行 × 流水线并行
total_gpus = DP * TP * PP

# 示例：128张GPU跑70B模型
DP = 4   # 4组独立训练（提升吞吐）
TP = 8   # 每层切成8份（单层放不下）
PP = 4   # 分成4段流水线（层太多）
assert 4 * 8 * 4 == 128  # ✓ 刚好用满
```

---

### 12.5.2 Q11: 显存不够怎么办？

**A:** 多层优化策略：

**Level 1：基础优化** 💡 （零副作用，必开）
```python
# 技巧1: 混合精度（FP32→FP16/BF16，直接减半）
use_fp16 = True  # 或 bf16（大模型更推荐）

# 技巧2: 梯度检查点（时间换空间，重算激活值）
model.gradient_checkpointing_enable()

# 技巧3: Flash Attention（IO优化，又快又省）
use_flash_attention = True
```
💾 节省显存：~30-40%
⚡ 性能影响：几乎无（Flash甚至更快）

**Level 2：中级优化** ⚙️ （轻微性能trade-off）
```python
# 技巧4: 减小batch + 梯度累积（等效batch不变）
per_device_batch_size = 1      # 从4降到1
gradient_accumulation_steps = 32  # 累积32步更新一次

# 技巧5: ZeRO-2（分片优化器状态，省8字节/参数）
zero_stage = 2  # 优化器状态分布到多GPU
```
💾 节省显存：额外30-40%
⚡ 性能影响：~10-20%慢（通信开销）

**Level 3：激进优化** ⚡ （显著性能损失）
```python
# 技巧6: ZeRO-3全分片 + CPU卸载（参数也分片）
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,                              # 分片参数、梯度、优化器
        "offload_optimizer": {"device": "cpu"},  # 优化器→CPU
        "offload_param": {"device": "cpu"}       # 参数→CPU
    }
}

# 技巧7: 激活值也卸载到CPU（终极省显存）
"activation_checkpointing": {
    "cpu_checkpointing": True
}
```
💾 节省显存：额外40-50%（可训练超大模型）
⚡ 性能影响：~2-5x慢（CPU-GPU传输瓶颈）

**Level 4：终极方案** 🔧 （改变训练方式）
```python
# 技巧8: 量化训练（QLoRA: 4-bit量化）
load_in_4bit = True  # 7B模型从14GB→3.5GB

# 技巧9: 降低模型规模（最直接）
model_size = "13B"  # 从70B→13B，显存降5倍
```
💾 节省显存：量化4x，降模型规模按比例
⚡ 性能影响：量化略降（<2%），小模型能力降低

**显存占用分解**：
```
总显存 = 模型参数 + 优化器状态 + 梯度 + 激活值

示例（70B模型，BF16）：
- 模型参数：70B × 2字节 = 140GB
- 优化器（Adam）：70B × 8字节 = 560GB
- 梯度：70B × 2字节 = 140GB
- 激活值：取决于batch size和序列长度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总计：~840GB（未优化）

应用ZeRO-3 + Offload：
- 每个GPU只需：840GB / GPU数量
- 8×A100 (80GB)：每卡需105GB → 使用offload可行
```

---

### 12.5.3 Q12: 训练中断如何恢复？

**A:** 完整的恢复流程：

**1. 自动恢复机制**
```python
def save_checkpoint(model, optimizer, scheduler, step, epoch):
    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'random_seed': torch.initial_seed(),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    torch.save(checkpoint, f'checkpoint_step_{step}.pt')

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 恢复随机数状态（重要！）
    torch.manual_seed(checkpoint['random_seed'])
    np.random.set_state(checkpoint['numpy_random_state'])
    random.setstate(checkpoint['python_random_state'])

    return checkpoint['step'], checkpoint['epoch']
```

**2. 训练脚本支持恢复**
```python
# 启动参数
--resume_from_checkpoint ./checkpoint_step_50000.pt

# 训练循环
if args.resume_from_checkpoint:
    start_step, start_epoch = load_checkpoint(...)
    print(f"Resuming from step {start_step}")
else:
    start_step, start_epoch = 0, 0

for step in range(start_step, total_steps):
    # 训练逻辑
    ...
```

**3. 验证恢复正确性**
```python
# 恢复后，loss曲线应该平滑衔接
# 不应该有突变或跳跃

# 检查清单：
✓ Loss值连续
✓ 学习率正确
✓ 随机数种子恢复（数据顺序一致）
✓ Step计数正确
```

**4. DeepSpeed恢复**
```python
# DeepSpeed自动处理恢复
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# 恢复
_, client_sd = model_engine.load_checkpoint(checkpoint_dir)
step = client_sd['step']
```

> **⚠️ 注意事项**
>
> - 定期保存checkpoint（每1000-5000步）
> - 保留多个历史checkpoint（至少最近3-5个）
> - 恢复后先验证几步，确保loss正常
> - 记录每个checkpoint的验证指标

---

### 12.5.4 Q13: 如何判断训练是否正常？

**A:** 多维度监控清单：

**✅ 健康训练的特征**：

1. **Loss曲线**
   - ✓ 平稳下降，无大幅波动
   - ✓ Training loss < Validation loss（轻微）
   - ✓ 下降速度符合预期

2. **梯度指标**
   - ✓ Gradient norm在0.1-10范围
   - ✓ 梯度裁剪率<5%
   - ✓ 无NaN或Inf

3. **性能指标**
   - ✓ GPU利用率>80%
   - ✓ 吞吐量稳定
   - ✓ 通信时间<20%

4. **学习率**
   - ✓ 按调度正常变化
   - ✓ Warmup期平稳上升

**❌ 异常训练的症状**：

| 症状 | 可能原因 | 检查项 |
|------|---------|--------|
| Loss不下降 | 学习率过小、数据问题 | 检查LR、查看数据样本 |
| Loss震荡 | 学习率过大、batch小 | 降低LR或增大batch |
| Loss=NaN | 梯度爆炸、数值溢出 | 启用梯度裁剪和混合精度 |
| GPU利用率低 | 数据加载慢、通信瓶颈 | 增加workers、检查网络 |

**实战监控代码（W&B）**：
```python
import wandb

# 记录核心训练指标（每步记录）
wandb.log({
    "train/loss": loss,                     # 📉 最重要，应持续下降
    "train/grad_norm": grad_norm,           # 📊 监控训练稳定性
    "train/learning_rate": lr,              # 📈 验证调度器
    "system/gpu_utilization": gpu_util,     # ⚡ 应保持80-95%
    "system/tokens_per_second": throughput, # 🚀 吞吐量指标
    "step": step
})
```

**智能告警系统**（避免半夜被叫醒）：
```python
# 设置告警阈值（根据实际情况调整）
if loss > moving_avg * 1.5:
    send_alert("🚨 Loss spike! Current: {loss:.4f}, Avg: {moving_avg:.4f}")

if gpu_util < 50:
    send_alert("⚠️ GPU利用率低! 当前: {gpu_util}%（可能数据IO瓶颈）")

if grad_norm > 100:
    send_alert("💥 梯度爆炸! Norm={grad_norm:.2f}（正常<10）")
```

---

# 13. 迈向多模态：VLM 架构与融合训练
————Vision Language Model (VLM)

在纯文本 LLM 的基础上，如何让模型“看见”世界？Vision Language Model (VLM) 提供了将图像、视频等非文本模态整合进语言模型的标准方案。本章将从融合架构、动态分辨率处理、视觉 Token 编码以及 PyTorch 实战代码四个维度进行深度剖析。

---

## 13.1 视觉-语言融合架构（VLM Architecture）

主流 VLM（如 LLaVA、PaliGemma、Qwen-VL）主要由三个部分组成：

```mermaid
graph LR
    IMG[图像输入] --> VE[Visual Encoder<br/>如 SigLIP / ViT]
    VE --> PROJ[Projection Layer<br/>MLP / Perceiver]
    PROJ --> LLM[LLM Backbone<br/>如 Llama / Qwen]
    TXT[文本 Prompt] --> LLM
    LLM --> OUT[文本回答]

    style VE fill:#e1f5ff,stroke:#01579b
    style PROJ fill:#fff9c4,stroke:#f57f17
    style LLM fill:#c8e6c9,stroke:#1b5e20
```

### 13.1.1 视觉编码器 (Visual Encoder)
负责将原始图像转化为特征张量。当前主流选择是预训练好的 **ViT-L/14**（如 CLIP 或 SigLIP）。
* **SigLIP vs. CLIP**：早期的 VLM 多使用 CLIP，但近年 **SigLIP (Sigmoid Language-Image Pre-training)** 已成为主流。CLIP 使用 Softmax 损失进行全局对比学习，需要极大的 Batch Size 且对噪声敏感；而 SigLIP 将图文匹配视为独立的二分类任务，使用 Sigmoid 损失，在较小 Batch Size 下训练更稳定，且具有更强的零样本（Zero-Shot）图像分类与细粒度表征能力。

### 13.1.2 粘合投影层 (Projection Layer)
负责将视觉编码器输出的特征维度（如 ViT 的 1024 维）映射到大语言模型的词表向量维度（如 Llama-3 的 4096 维），并将视觉特征转化为 LLM 能够理解的“虚拟视觉 Tokens”。
* **线性投影/多层感知机 (Linear/MLP Projection)**：LLaVA 采用的极简方案，计算开销极低，但会将 ViT 的所有特征全部送入 LLM（如 576 个 Token），随着图像增加，极易占满 LLM 的上下文窗口。
* **感知机重采样器 (Perceiver Resampler)**：PaliGemma / Flamingo 采用的方案。使用一组固定数量的“查询向量（Queries）”通过交叉注意力（Cross-Attention）对 ViT 的海量特征进行聚合，将任意分辨率/任意数量的视觉特征压缩为固定长度（如 64 或 128 个 Token），极大节省了 LLM 的上下文窗口。
* **Q-Former**：BLIP-2 提出的基于两阶段预训练的 Query 变换器，结构较重但表征能力强。

### 13.1.3 语言模型基座 (LLM Backbone)
接收交织的视觉 Token 和文本 Token，输出最终的文本回答。训练时可以通过 LoRA 或全参数微调让其学习图文理解。

---

## 13.2 动态分辨率处理技术（Dynamic Resolution / Image Patching）

传统的视觉模型会将输入的任意尺寸图像强行缩放到固定分辨率（如 $224 \times 224$ 或 $336 \times 336$）。这对于普通分类任务可行，但对于大模型中的 **OCR（字符识别）**、**图表分析**或**细粒度目标检测**是灾难性的，缩放会导致小文字直接模糊不可读。

为了解决该痛点，现代 VLM（如 LLaVA-NeXT, Monkey, InternVL）采用了**动态切片（Image Patching）**技术：

```
+------------------------------------+
|                                    |
|          原始图像 (如 672x672)      |
|                                    |
+------------------------------------+
                  |  进行网格切分
                  v
+------------------+------------------+
|                  |                  |
|    子图 1 (336x336)|    子图 2 (336x336)|
|                  |                  |
+------------------+------------------+
|                  |                  |
|    子图 3 (336x336)|    子图 4 (336x336)|
|                  |                  |
+------------------+------------------+
                  +
+------------------------------------+
|  全局缩略图 (336x336, 提取宏观特征)   |
+------------------------------------+
```

* **处理流程**：
  1. 将一张高分辨率图像（如 $672 \times 672$）无重叠地切割为 $2 \times 2$ 个子图，每个子图为 $336 \times 336$。
  2. 另外将原始图强行缩放到 $336 \times 336$，作为“全局缩略图”，用于提供图像全局宏观布局。
  3. 将这 5 张图分别送入 Visual Encoder 提取特征，由 Projection Layer 映射后拼接为一串视觉 Tokens。
  4. 这种方式能让大模型无损看清大图中的所有像素细节。

---

## 13.3 视觉 Token 编码与序列交织

在 VLM 中，图像特征向量在送入 LLM 之前，必须在序列维度与文本进行拼接或交织（Interleaving）。

* **序列表示格式**：
  在 Token 级别，图像通常会被一对特殊的标志符包裹。例如，一张图映射为 $N$ 个视觉 Token（例如 $N=576$），在输入 LLM 时的表征形式为：
  ```
  <|begin_of_text|>system\nYou are a helpful assistant.\n
  user\nPlease describe this image: <image>Visual_Token_1, Visual_Token_2, ..., Visual_Token_N</image>\n
  assistant\n
  ```
* **Attention Mask 控制**：
  为了实现高效建模，视觉 Token 之间通常应用**双向注意力（Bidirectional Attention）**，即视觉 Token 内部能够互相看到，而不需要受自回归的因果遮蔽（Causal Mask）限制；文本 Token 对视觉 Token 则应用正常的**因果注意力（Causal Attention）**。

---

## 13.4 极简 VLM 前向传播 PyTorch 实现

以下代码演示了如何使用 PyTorch 从零构建一个支持 Vision-Language 融合的极简 VLM 模型：

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

class SimpleVLM(nn.Module):
    def __init__(self, llm_model_name_or_path, visual_dim=1024, embed_dim=4096):
        super().__init__()
        # 1. 实例化 LLM Backbone（此处使用Llama/Qwen架构）
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
        self.word_embeddings = self.llm.get_input_embeddings()
        
        # 2. 简易视觉特征提取器 (可以使用预训练 ViT/SigLIP 代替，此处用随机初始化模拟)
        # 假设输入图像为 336x336x3，ViT 提取后输出的 patch 特征数为 576，每个 patch 维度为 1024
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.GELU()
        )
        
        # 3. 粘合多层感知机 (MLP Projector)
        # 将 1024 维的视觉向量，映射到 LLM 词嵌入空间的 4096 维
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, text_input_ids, image_features, visual_token_start_idx):
        """
        Args:
            text_input_ids: 文本的 Token IDs, shape: [batch_size, seq_len]
            image_features: 视觉编码器提取的原始特征, shape: [batch_size, 576, 1024]
            visual_token_start_idx: 文本中插入图像占位符的起始索引位置
        """
        # Step 1: 将文本 Token IDs 转化为 LLM 的 Embedding 向量
        # text_embeds shape: [batch_size, seq_len, 4096]
        text_embeds = self.word_embeddings(text_input_ids)
        
        # Step 2: 提取图像特征并进行模态对齐投影
        # vis_features shape: [batch_size, 576, 1024] -> [batch_size, 576, 4096]
        vis_features = self.visual_encoder(image_features)
        vis_tokens = self.projector(vis_features)
        
        # Step 3: 在指定位置将视觉 Token 插入并替换文本序列中的占位符
        # 假设在 text_input_ids 中我们预留了长度为 576 的占位符（如特殊的 <image> token）
        batch_size, seq_len, embed_dim = text_embeds.shape
        num_vis_tokens = vis_tokens.shape[1]
        
        # 构建混合的输入 Embedding 序列
        mixed_embeds = []
        for i in range(batch_size):
            # 将第 i 个样本的视觉 Token，替换到预留的占位位置
            prefix = text_embeds[i, :visual_token_start_idx]
            suffix = text_embeds[i, visual_token_start_idx + num_vis_tokens:]
            
            # 拼接: prefix (文本) + vis_tokens (图像映射) + suffix (文本)
            sample_embeds = torch.cat([prefix, vis_tokens[i], suffix], dim=0)
            mixed_embeds.append(sample_embeds)
            
        # mixed_embeds shape: [batch_size, seq_len, 4096]
        mixed_embeds = torch.stack(mixed_embeds, dim=0)
        
        # Step 4: 送入 LLM 基座进行计算，得到输出 Logits
        outputs = self.llm(inputs_embeds=mixed_embeds)
        return outputs.logits
```

---

# 14. 参考资源
————References & Reading List

为了帮助深入探索和落地大语言模型训练，本章梳理了本综述提及的核心学术论文、开源社区标杆项目以及推荐的实践学习路径。

## 14.1 必读经典论文

* **模型架构与注意力**：
  * Vaswani et al. [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Transformer 奠基作)
  * Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (FlashAttention v1 原理)
* **缩放法则 (Scaling Laws)**：
  * Kaplan et al. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (OpenAI 早期缩放理论)
  * Hoffmann et al. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla 缩放公式，证明数据量与参数同等重要)
* **主流基座模型**：
  * Touvron et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (现代开源 LLM 的基石)
* **偏好对齐与强化学习**：
  * Ouyang et al. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT / RLHF)
  * Rafailov et al. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (DPO 算法)
  * DeepSeek-AI. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (GRPO 与推理模型训练范式)

## 14.2 标杆开源项目

* **分布式训练底座**：
  * [DeepSpeed](https://github.com/microsoft/DeepSpeed) (微软开源，ZeRO 系列并行技术的最佳载体)
  * [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (NVIDIA 官方出品，超大规模 3D 混合并行与 GPU 底层优化库)
* **微调与推理对齐工具**：
  * [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl) (Hugging Face 出品，内置 SFTTrainer, DPOTrainer 与 GRPOTrainer)
  * [Unsloth](https://github.com/unslothai/unsloth) (最强单卡微调加速库，CUDA kernel 重写，可将微调与 GRPO 训练的显存降低达 70-80%)
* **小模型教学实战**：
  * [MiniMind](https://github.com/wangr2018/minimind) (超轻量大模型全栈训练项目，非常适合在个人显卡上快速跑通预训练到 RL 完整 Pipeline)

## 14.3 推荐实践学习路径

```
[步骤1: 理论打底] -> 学习 Stanford CS224n 课程，读懂 Transformer、Causal Masking 机制与自回归解码。
      |
[步骤2: 玩转推理部署] -> 使用 llama.cpp 或 vLLM 部署主流开源模型（如 Qwen-7B），熟悉 INT4/INT8 量化后的显存占用变化。
      |
[步骤3: 玩具模型实训] -> 运行 MiniMind 项目，在单张消费级显卡（如 RTX 3090/4090）上跑通一次百兆参数模型的 Tokenizer 训练、Pre-train、SFT、DPO。
      |
[步骤4: 分布式工程进阶] -> 学习使用 DeepSpeed 配置文件，在多卡环境（如 8x A100）下配置 ZeRO-2/3、Offload 等策略，训练 7B-70B 规模模型。
```


