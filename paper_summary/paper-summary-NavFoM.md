---
## 1. NavFoM (2025)
——Embodied Navigation Foundation Model

📄 **Paper**: https://arxiv.org/abs/2509.12129

**精华**

这篇论文展示了如何构建跨任务、跨具身体的导航基础模型,值得借鉴的核心思想包括:(1) 引入 Temporal-Viewpoint Indicator (TVI) tokens 来统一编码不同相机配置和时间信息,使模型能够处理多视角输入;(2) 提出 Budget-Aware Temporal Sampling (BATS) 策略,通过遗忘曲线动态采样历史帧,平衡性能和推理速度;(3) 在 8.02M 导航样本(包括四足机器人、无人机、轮式机器人、汽车等多种具身体)上联合训练,展示了大规模多任务训练对泛化能力的提升;(4) 采用视觉特征缓存机制加速训练 2.9 倍;(5) 证明了无需针对特定任务微调即可在多个基准测试上达到 SOTA 或竞争性能。

**研究背景/问题**

当前导航系统主要聚焦于特定任务设定和具身体架构,缺乏跨任务和跨具身体的泛化能力。现有 VLM 虽然在零样本任务上表现出色,但导航任务仍然局限于狭窄的任务领域、固定的相机配置和特定的具身体平台。本文旨在构建一个统一的导航基础模型,能够处理来自不同具身体(四足机器人、无人机、轮式机器人、汽车)的多视角输入,并跨越多个导航任务(VLN、目标搜索、目标追踪、自动驾驶)。

**主要方法/创新点**

<div align="center">
  <img src="/images/NavFoM-pipeline-overview.png" width="100%" />
<figcaption>
NavFoM 整体架构:统一框架处理 Image QA、Video QA 和导航任务
</figcaption>
</div>

NavFoM 基于 Vision-Language Model 架构,扩展为双分支系统:一个用于导航,一个用于问答。核心创新包括:

**1. Temporal-Viewpoint Indicator (TVI) Tokens**
- 引入特殊 indicator tokens 来编码相机视角和时间信息,每个 TVI token 由三部分组成:
  - **可学习的 base embedding** (E_Base)
  - **时间编码** (Time PE): 使用正弦位置编码标识帧的时间顺序
  - **视角编码** (Angle PE): 使用正弦/余弦编码保持方位角的循环连续性
- 对于导航任务,使用: E_TVI = E_Base + Time PE + Angle PE
- 对于 Video QA,仅使用时间信息;对于 Image QA,仅使用 base embedding
- TVI tokens 使 LLM 能够区分不同时间步和不同视角的 tokens,实现多视角导航

**2. Budget-Aware Temporal Sampling (BATS)**
- 解决在线导航时视觉 tokens 数量激增的问题
- 基于遗忘曲线(exponential decay)的采样概率: P(t) = (1 - ε)e^(k(t-T)/T) + ε
- 动态调整历史帧采样,越近的帧采样概率越高
- 在 token budget 约束下,平衡短期上下文和长期历史信息
- 相比 Uniform Sampling,BATS 在保持性能的同时显著降低推理时间

**3. 观测编码**
- 使用预训练 vision encoders (DINOv2, SigLIP) 提取视觉特征
- 采用 Grid Average Pooling 策略生成两种分辨率的视觉 tokens:
  - **Fine-grained** (64×C): 用于当前最新观测和 Image QA
  - **Coarse-grained** (4×C): 用于导航历史和 Video QA
- 通过 cross-modality projector 将视觉特征映射到 LLM latent space

**4. Token 组织策略**
- 不同任务采用不同的 token 组织方式:
  - **Image QA**: fine-grained visual tokens + base TVI embedding
  - **Video QA**: coarse-grained visual tokens + base + time embedding
  - **Navigation**: coarse-grained + fine-grained tokens + base + time + angle embedding
- 这种设计实现了导航和 QA 数据的联合训练

**5. 轨迹预测**
- 使用三层 MLP 作为 planning head 从 LLM 隐藏状态预测轨迹
- 轨迹归一化到 [-1, 1] 分布,针对不同具身体(室内导航 vs 户外驾驶)采用不同的 scaling factor
- 对于室内机器人,预测 8 个航点;对于汽车和无人机,预测更长的轨迹

**6. 数据规模与来源**
- **导航数据** (8.02M): VLN-CE R2R/RxR (2.94M), OpenUAV (429K), 目标导航 (1.02M), 主动视觉追踪 (897K), 自动驾驶 (681K), Web 导航伪标签 (2.03M)
- **QA 数据** (4.76M): Image QA (3.15M) + Video QA (1.61M)
- 总计 12.7M 训练样本,覆盖四足机器人、无人机、轮式机器人、汽车等多种具身体

**7. 训练优化**
- 视觉特征缓存: 预先计算并缓存 coarse-grained visual tokens,训练加速 2.9 倍,GPU 内存减少 1.8 倍
- 使用 Qwen2-7B 作为 LLM backbone
- 单次训练所有参数(仅 designated trainable parameters),无需多阶段训练

**核心结果/发现**

**VLN 性能**:
- **VLN-CE R2R** (single-view): SR 5.01% → 64.9%, SPL 56.2%,无需任务特定微调即达到 SOTA
- **VLN-CE RxR** (four-view): SR 5.51% → 57.4%, SPL 49.4%,超越所有基线方法
- **OpenUAV** (四视角,UM split): SR 6.38% → 14.05%, OSRL 5.68% → 18.65%,显著优于 TravelUAV

**目标搜索**:
- **HM3D-OVON** (zero-shot): VAL SEEN SR 55.0%, VAL UNSEEN SR 45.2%,超越 MTU3D baseline

**主动视觉追踪**:
- **EVT-Bench** (four-view, zero-shot): Single Target SR 85.1%/TR 80.5%, Distracted Target SR 62.0%/TR 67.9%

**自动驾驶**:
- **NAVSIM** (eight-view): PDMS 84.3%, 与 SOTA 方法竞争性能
- **nuScenes** (six-view): CR 93%, 接近 SOTA

**Ablation 研究**:
- 多任务训练带来显著增益:联合训练使 VLN SR 从 57.3% 提升到 64.4%
- 相机数量对性能的影响:从单视角到四视角,SR 从 58.3% 提升到 65.8%,但增加到六视角略有下降
- BATS 相比 Uniform Sampling,在 RxR 上 nDTW 仅下降 1.4%,但保持稳定推理速度
- TVI tokens 相比其他替代方案(learned special tokens, handcraft tokens)显著提升性能

**实际部署**:
- 在 110 个真实世界测试场景(50 VLN + 30 搜索 + 30 追踪)中验证,成功率达到 72%~93%
- 支持跨具身体部署:四足机器人(Unitree Go2)、类人机器人、无人机、轮式机器人
- 0.5 秒内生成 8 航点轨迹(1600 token budget)

**局限性**

该方法在训练时需要大量计算资源(56 NVIDIA H100 GPUs,72 小时)。尽管引入了视觉特征缓存等优化策略,大规模训练仍然是资源密集型任务。此外,在需要遍历 300 米复杂邻域的 Unseen-Map 场景中表现较差,表明模型在大规模环境探索和长距离规划方面仍有改进空间。作者也指出 NavFoM 只是一个起点,未来需要更高质量的数据、更先进的技术以及新一代基准测试来推动泛化导航研究的发展。
