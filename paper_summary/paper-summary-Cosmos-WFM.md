## Cosmos World Foundation Model (2025)
———NVIDIA Cosmos World Foundation Model Platform for Physical AI

📄 **Paper**: [https://arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575)

### 精华

NVIDIA 发布的 Cosmos 物理 AI 世界模型平台，展示了构建通用物理世界模拟器的完整路径，值得借鉴的点包括：
1. **数据策展流水线**：开发了名为 Cosmos Video Curator 的大规模自动视频处理流水线，从 2 亿小时视频中筛选出 1000 万个高质量片段，解决了物理 AI 数据规模化的核心难题。
2. **多模态 Tokenizer**：设计了能够同时处理连续和离散表示的视觉 Tokenizer，通过时空分解和因果 3D 卷积实现了极高的压缩比和重建质量。
3. **分层训练范式**：采用先进行通用物理规律的大规模预训练，再针对特定机器人任务进行后训练（Post-training）的范式，显著提升了跨任务泛化能力。
4. **物理对齐验证**：通过在模拟环境中构建物理场景（如倾斜平面、U型槽等）并对比真实物理引擎结果，量化评估了生成模型对牛顿力学的遵循程度。
5. **安全护栏系统**：内置了完整的 Guardrail 系统，确保生成的物理模拟内容安全合规。

---

### 1. 研究背景/问题

物理 AI（Physical AI）的发展面临核心瓶颈：缺乏像语言模型那样的大规模高质量交互数据。虽然视觉生成模型近年来取得了巨大进步，但要在机器人、自动驾驶等物理交互领域应用，模型必须不仅能生成视觉逼真的图像，还必须深刻理解物理规律。现有的世界模型通常局限于特定环境或小规模数据，难以作为通用的“数字孪生”环境供物理 AI 训练和测试。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/wm/Cosmos-WFM-Overview.png" width="100%" />
<figcaption>
Cosmos 世界基础模型概览：包含扩散（Diffusion）和自回归（Autoregressive）两种架构。
</figcaption>
</div>

Cosmos 平台提供了一个完整的生态系统，用于构建和微调针对物理 AI 任务的世界基础模型（WFM）：

1. **平台组件**：
<div align="center">
  <img src="/images/wm/Cosmos-Platform-Components.png" width="100%" />
<figcaption>
Cosmos WFM 平台核心组件：视频策展、Tokenizers、预训练 WFM 和后训练样本。
</figcaption>
</div>

2. **Cosmos Tokenizer**：
<div align="center">
  <img src="/images/wm/Cosmos-Tokenizer-Architecture.png" width="100%" />
<figcaption>
Cosmos Tokenizer 架构：采用基于小波变换的编码器-解码器结构，通过因果 3D 卷积捕获时间相关性。
</figcaption>
</div>
Tokenizer 是系统的基石，支持连续（用于扩散模型）和离散（用于自回归模型）两种表示。它在保持高压缩比的同时，显著优于现有的 SOTA 方法（如 Video-MAGVIT2）。

3. **预训练模型架构**：
   - **扩散模型（Diffusion WFM）**：基于 DiT 架构，擅长生成高视觉质量的 3D 一致性视频。
<div align="center">
  <img src="/images/wm/Cosmos-Predict1-Diffusion-Architecture.png" width="100%" />
<figcaption>
Cosmos-Predict1 扩散模型整体架构：基于 DiT，整合了 T5 文本编码器和 3D RoPE。
</figcaption>
</div>

   - **自回归模型（Autoregressive WFM）**：将视频视为离散 Token 序列，擅长处理长序列预测和复杂的交互。
<div align="center">
  <img src="/images/wm/Cosmos-Predict1-Autoregressive-Architecture.png" width="100%" />
<figcaption>
Cosmos-Predict1 自回归模型架构：通过因果 Transformer 进行 Token 预测。
</figcaption>
</div>

4. **训练与微调范式**：
<div align="center">
  <img src="/images/wm/Cosmos-Training-Paradigm.png" width="100%" />
<figcaption>
预训练 WFM 作为通用物理学习器，通过后训练适应特定的物理 AI 任务。
</figcaption>
</div>
模型首先在大规模视频数据集上进行通用物理知识预训练，随后可以通过微调适应相机控制（Camera Control）、机器人操纵（Robotic Manipulation）和自动驾驶等任务。

---

### 3. 核心结果/发现

- **物理对齐能力**：通过构建受控的物理实验，验证了 Cosmos WFM 能够准确模拟物体在重力、碰撞下的运动轨迹，其预测精度接近专用物理引擎。
<div align="center">
  <img src="/images/wm/Cosmos-Physics-Alignment.png" width="100%" />
<figcaption>
物理场景仿真对比：展示了模型在模拟物理规律方面的能力。
</figcaption>
</div>

- **多任务泛化**：后训练后的模型在操纵、导航等任务上展示了极强的 Zero-shot 迁移能力，且生成质量优于 VideoLDM 等基准模型。
- **安全合规**：
<div align="center">
  <img src="/images/wm/Cosmos-Guardrail-Overview.png" width="100%" />
<figcaption>
Cosmos Guardrail 架构：涵盖了从输入 prompt 到输出内容的完整安全检测流程。
</figcaption>
</div>

---

### 4. 局限性

虽然模型展现了强大的物理模拟能力，但在处理极小尺度物体的精细交互（如指尖触感）方面仍有提升空间。此外，在大规模场景生成时，模型偶尔会出现物体凭空消失或突然出现的异常。

---
