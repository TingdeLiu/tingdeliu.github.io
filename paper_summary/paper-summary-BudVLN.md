## BudVLN (2026)
———Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation

📄 **Paper**: [arXiv:2602.06356](https://arxiv.org/abs/2602.06356)

### 精华

1. **核心思想**：通过“回顾式纠偏”（Retrospective Rectification）解决 Vision-Language Navigation (VLN) 中的指令-状态不一致问题。
2. **训练范式**：引入了 **Adaptive Mutual Exclusion Strategy**，将样本动态分流为效率路径（GRPO）和鲁棒性路径（SFT），实现了精准训练。
3. **纠偏机制**：利用“回锚”机制合成语义一致的修正轨迹，避免了传统方法中强制回归导致的语义冲突。
4. **极致效率**：采用 GRPO 算法（借鉴自 DeepSeek-R1），无需价值网络，训练成本仅为传统 DAgger 的约 25%。
5. **性能卓越**：在 R2R-CE 和 RxR-CE 基准测试上刷新 SOTA，尤其在处理偏差和鲁棒性方面表现突出。

---

### 1. 研究背景/问题

当前的视觉-语言导航（VLN）系统面临严重的**曝光偏差（Exposure Bias）**问题：推理时的细微偏差会导致严重的累积误差。虽然 DAgger 类方法尝试通过纠正错误状态来缓解这一问题，但论文指出这些方法存在**指令-状态不一致（Instruction-State Misalignment）**的致命局限。如图 1 所示，强制智能体从离群状态回归往往会生成与其原始语言指令相冲突的监督信号（例如：指令要求直行，但为回归正轨必须掉头），这会损害智能体的指令遵循能力。

<div align="center">
  <img src="/images/VLN/BudVLN-misalignment-illustration.png" width="100%" />
<figcaption>
图 1：指令-状态不一致现象的图示，展示了传统 DAgger 如何产生语义冲突的监督。
</figcaption>
</div>

---

### 2. 主要方法/创新点

论文提出了 **BudVLN**，一个旨在通过统一的在线回顾式纠偏框架解决上述挑战的系统。

#### Adaptive Mutual Exclusion Strategy (自适应互斥策略)
BudVLN 并不对所有样本一视同仁，而是采用一种自适应策略进行动态路由：
- **Proficiency Pathway (效率路径)**：通过 Greedy Probe 评估。若智能体已能熟练完成任务，则利用 **GRPO (Group Relative Policy Optimization)** 进行组内相对优势学习，进一步优化路径效率。
- **Rectification Pathway (纠偏路径)**：若智能体在任务中失败，则触发**回顾式纠偏**。

<div align="center">
  <img src="/images/VLN/BudVLN-framework-overview.png" width="100%" />
<figcaption>
图 2：BudVLN 训练框架概览，展示了 GRPO 路径与回顾式纠偏（SFT）路径的动态分流。
</figcaption>
</div>

#### Retrospective Rectification (回顾式纠偏)
针对失败样本，BudVLN 执行以下操作：
1. **回锚（Anchor Identification）**：将状态回溯到发生偏差前的最后一个有效路径点（Valid Anchor）。
2. **语义一致性合成**：利用 Oracle 合成从该锚点出发的正确轨迹，以此作为 SFT 的监督信号。
这种方法确保了监督信号与原始指令的语义一致性，彻底解决了 DAgger 的语义冲突问题。

#### GRPO 优化
受到大规模推理模型成功的启发，BudVLN 引入了 GRPO 算法。它通过在一个采样组内计算相对优势，摆脱了对昂贵价值网络（Value Network）的依赖，极大地降低了计算开销，同时提升了探索效率。

---

### 3. 核心结果/发现

- **SOTA 性能**：在 R2R-CE 和 RxR-CE 两个主流基准测试中，BudVLN 全面超越了现有模型。在 R2R-CE 上，成功率 (SR) 达到 **57.6%**，SPL 达到 **51.1%**。
- **训练效率**：得益于 GRPO 算法和高效的纠偏机制，BudVLN 仅需 **27 GPU 小时** 即可完成训练，相比 DAgger 的 114 小时，效率提升了近 4 倍。
- **消融研究**：实验证明，单独添加纠偏机制能显著提升 SR，而 GRPO 算法则对 SPL 的提升和训练效率的优化起到了关键作用。

<div align="center">
  <img src="/images/VLN/BudVLN-main-results.png" width="100%" />
<figcaption>
表 1：BudVLN 与现有 VLN 模型在 R2R-CE 和 RxR-CE 测试集上的性能对比。
</figcaption>
</div>

---

### 4. 局限性

虽然 BudVLN 在离散和连续环境中均表现出色，但其鲁棒性目前仍受限于预定义 Oracle 的质量。在极度复杂的极端环境下，如何自主生成更高质量的“回顾性”知识仍是未来研究的方向。
