---
## 1. ACoT-VLA (2026)
———Action Chain-of-Thought for Vision-Language-Action Models

📄 **Paper**: https://arxiv.org/abs/2601.11404

**精华**
这篇论文的核心创新在于将推理过程从语言/视觉空间转移到动作空间，值得借鉴的点包括：(1) 直接在动作空间进行推理，提供同质化的运动指导，弥合语义与运动学之间的鸿沟；(2) 显式推理器(EAR)与隐式推理器(IAR)的互补设计，同时提供轨迹级和语义级指导；(3) Teacher Forcing稳定化训练策略，避免推理模块对动作头的优化干扰；(4) 通过action-level guidance大幅提升长时域任务的鲁棒性和误差抗累积能力。

**研究背景/问题**
现有VLA模型主要在视觉-语言空间进行推理（如语言CoT预测子任务、视觉CoT合成目标图像），但这些推理形式对动作执行的指导是间接且次优的。VLM预训练主要聚焦语义理解而非物理动力学，世界模型虽能预测未来视觉状态但仍局限于视觉表征，两者都存在语义-运动学鸿沟（semantic-kinematic gap），难以为精确的低层动作生成提供充分的细粒度指导。

**主要方法/创新点**

本文提出 **Action Chain-of-Thought (ACoT)** 范式，将推理过程重新定义为结构化的动作意图序列，直接在动作空间进行deliberation。ACoT-VLA框架包含三个核心组件：

<div align="center">
  <img src="/images/ACoT-VLA-paradigm-comparison.png" width="100%" />
<figcaption>
不同CoT范式对比：(a) 语言CoT预测子任务，(b) 视觉CoT合成目标图像，(c) 本文提出的动作CoT直接在动作空间提供同质化指导
</figcaption>
</div>

<div align="center">
  <img src="/images/ACoT-VLA-architecture.png" width="100%" />
<figcaption>
ACoT-VLA整体架构，包含EAR、IAR和Action-Guided Prediction三大模块
</figcaption>
</div>

**1. Explicit Action Reasoner (EAR)**
- 设计为轻量级Transformer，以noisy action sequence作为输入
- 通过self-attention捕获时序依赖，cross-attention从VLM的key-value cache注入多模态上下文
- 采用flow matching训练，自主生成粗粒度参考轨迹 $a^{ref}_{t:t+H^{ref}-1}$
- 参考轨迹编码后形成显式动作空间指导 $Z^{ex}$

**2. Implicit Action Reasoner (IAR)**
- 直接操作VLM的key-value cache，提取隐式运动线索
- 对每层VLM特征，使用可学习query矩阵 $Q_i$ 通过cross-attention提取动作相关信息
- 下采样策略降低计算开销：将KV cache降维至 $d' \ll d$
- 跨层聚合后形成隐式动作指导 $Z^{im}$，捕获visual affordances和action semantics

**3. Action-Guided Prediction (AGP)**
- 将noisy action embedding视为query $Q_{action}$，与 $Z^{ex}$ 和 $Z^{im}$ 进行dual cross-attention
- 通过self-attention融合显式与隐式指导：$\bar{h} = \text{Self-Attn}([S^{ex}; S^{im}])$
- 最终action head $\pi^{head}_\theta$ 基于聚合表征预测去噪动作序列

**训练策略**：
- Flow matching损失同时优化EAR和action head
- Teacher Forcing稳定化：训练时 $Z^{ex}$ 直接从ground-truth轨迹计算，推理时切换为自条件模式

<div align="center">
  <img src="/images/ACoT-VLA-real-world-tasks.png" width="100%" />
<figcaption>
真实世界三项操作任务：擦拭污渍、倒水、开放集抓取
</figcaption>
</div>

**核心结果/发现**

**仿真实验**：
- LIBERO: 98.5%平均成功率（SOTA），相比π0.5提升1.6%，在LIBERO-Long（长时域）提升最显著（96.0% vs 92.4%）
- LIBERO-Plus: 84.1%，在鲁棒性测试中大幅超越，尤其在相机视角变化(+11.6%)、机器人初始状态扰动(+16.3%)、传感器噪声(+12.5%)上表现突出
- VLABench: Intention Score 63.5%、Progress Score 47.4%，在unseen-texture track上获得+12.6% IS和+7.2% PS的显著提升

<div align="center">
  <img src="/images/ACoT-VLA-real-world-results.png" width="100%" />
<figcaption>
真实世界实验结果对比
</figcaption>
</div>

**真实世界部署**：
- 在AgiBot G1机器人上平均成功率66.7%（vs π0.5的61.0%、π0的33.8%）
- 跨embodiment验证：在AgileX平台上同样有效，证明方法的通用性

**消融研究关键发现**：
- EAR单独使用提升1.4%（LIBERO），IAR单独提升1.2%
- EAR+IAR联合使用达到最优，证明显式与隐式指导的互补性
- 参考动作horizon在15-30时效果最佳，过长或过短均不利
- EAR参数量在300M时性能最优，过度参数化反而导致过拟合
- 推理延迟仅增加约20ms（91ms→112ms），性能-效率权衡优秀

**局限性**
该方法需要额外的推理模块，虽然计算开销相对较小但在资源受限平台上可能存在挑战。此外，当前动作表征仍采用action chunks（关节角度/末端执行器位姿），缺乏显式几何结构，未来可将动作表征扩展至几何可解释的3D空间，进一步释放ACoT的推理潜力。
