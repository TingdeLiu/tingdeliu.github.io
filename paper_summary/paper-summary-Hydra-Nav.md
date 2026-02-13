## 1. Hydra-Nav (2026)
——Object Navigation via Adaptive Dual-Process Reasoning

📄 **Paper**: https://arxiv.org/abs/2602.09972

---

**精华**

Hydra-Nav 最值得借鉴的核心思想是：将"慢思考"（CoT 推理）与"快行动"（低级反应控制）统一在**单个 VLM** 内，避免了多模型架构的碎片化问题。其关键创新在于通过 **Iterative Rejection Fine-Tuning (IRFT)** 让模型自主学习"何时触发推理"，而非固定频率触发，从而在成功率与推理开销之间取得最优平衡。三阶段课程训练（空间-动作对齐 → 记忆-推理集成 → 自适应推理）的渐进式设计，为构建具身导航智能体提供了可复用的训练范式。新提出的 SOT 指标（Success weighted by Operation Time）将推理延迟纳入评估，比 SPL 更贴近实际部署需求，值得在其他具身任务中推广使用。

---

**研究背景/问题**

Object goal navigation 要求机器人仅凭自我中心感知在真实环境中主动探索并定位目标物体。当前 VLM-based 方法存在两大核心缺陷：（1）时空推理能力不足，导致对已探索区域的记忆维护失效，引发重复探索；（2）在每步推理（chain-of-thought）的做法带来大量不必要的计算开销，而在关键"停滞点"又未能及时触发推理。现有双系统架构（slow-fast paradigm）依赖独立模型，存在架构割裂和切换灵活性不足的问题。

---

**主要方法/创新点**

Hydra-Nav 将高层规划与低层元动作统一在**单一 VLM**（基于 Qwen2.5-VL-7B）内，通过输出特殊 transition token `obs` 自主触发从快系统到慢系统的切换。

<div align="center">
  <img src="/images/Hydra-Nav-architecture.png" width="100%" />
<figcaption>
Hydra-Nav 整体架构：慢系统负责全局时空推理与高层规划，快系统负责低级元动作的高效执行，通过特殊 token obs 自适应切换。
</figcaption>
</div>

**双过程系统（Dual-process System）**

- **慢系统（Slow system）**：接收目标指令、当前全景观测（4 张 90° 间隔 RGB 图）和结构化长期记忆，生成 CoT 推理文本与高层计划，随后输出第一个元动作。
- **快系统（Fast system）**：基于上一慢系统的对话历史，利用 KV-caching 仅编码最新自我中心帧，自回归解码低级原子动作（MoveAhead 0.25m、TurnLeft/Right 30°），避免重复处理完整历史上下文。
- **自适应切换机制**：当智能体完成子目标或当前观测与现有计划矛盾时，输出 `obs` 触发全景扫描，构建新的地标节点并更新长期记忆，随后重新进入慢系统。

<div align="center">
  <img src="/images/Hydra-Nav-context-organization.png" width="100%" />
<figcaption>
推理期间的上下文组织方式：短期记忆为交错图像-动作对，遇到 obs token 时更新记忆并清空短期上下文。
</figcaption>
</div>

**三阶段课程训练（Curriculum Training Pipeline）**

**Stage 1 — 空间-动作对齐（Spatial-Action Alignment）**

使用 A* planner 在 HM3D、MP3D、OVON 训练集上生成 **500K 条轨迹**（20.1B tokens），训练 Qwen2.5-VL-7B 学习基本导航动作执行。每条轨迹格式化为多轮对话，通过单次前向-反向传播完成梯度计算。

**Stage 2 — 推理-记忆集成（Reasoning-Memory Integration）**

<div align="center">
  <img src="/images/Hydra-Nav-data-synthesis.png" width="100%" />
<figcaption>
Stage 2 数据合成流程：左侧为启发式路点选择的轨迹生成策略，右侧为用 Qwen3-VL-235B-Thinking 合成高质量推理文本的流程。
</figcaption>
</div>

- 使用启发式路点选择策略生成包含探索行为的轨迹（而非仅最短路径），每条轨迹选取分数最高的两个探索路点。
- 将轨迹分段（固定长度 16 步），在每段开头插入长期记忆和推理文本，段尾插入 `obs` token。
- 推理文本合成：先用 Qwen3-VL-235B-Thinking 对历史图像进行记忆摘要，再结合当前视图与"未来正确视图"（信息泄漏防止）生成前瞻性规划文本。
- 共生成 **565K 条混合样本（8.3B tokens）**，同时混入 VQA 数据防止过拟合。

**Stage 3 — 自适应推理（Adaptive Reasoning via IRFT）**

定义两类**停滞点（Stagnation Points）**：
1. **重复探索**：智能体在过去 $T_{stag}=20$ 步内回到距离 $\delta_{stag}=0.5$m 内的位置。
2. **缺乏进展**：在随机时间窗口 $\Delta t \sim \mathcal{U}(20,35)$ 内到目标距离未缩短。

IRFT 流程：在快系统模式下运行，于停滞点触发慢系统；对失败轨迹（超时或目标误识别）进行"拒绝-修复"——找到干预时间戳 $t^*$，用 A* 最优路径替换后续轨迹，重新合成修正段的推理文本；使用最新 checkpoint 迭代执行，每轮生成约 60K 条轨迹（4.5B tokens）。

---

**核心结果/发现**

<div align="center">
  <img src="/images/Hydra-Nav-performance-irft.png" width="100%" />
<figcaption>
多轮 IRFT 训练过程中 SR 和 SOT 在 HM3D、MP3D、OVON Val-Unseen 上的提升曲线。
</figcaption>
</div>

**与 SOTA 对比（Table 2）：**

| Benchmark | 指标 | Hydra-Nav-IRFT | 第二名 | 提升 |
|-----------|------|----------------|--------|------|
| HM3D Val  | SR   | **84.8%**      | 73.7%  | +11.1% |
| MP3D Val  | SR   | **64.0%**      | 46.6%  | +17.4% |
| OVON Val-Unseen | SR | **66.3%** | 45.2%  | +21.1% |

**SOT 指标分析（Table 5）：**

- Hydra-Nav-IRFT 推理触发比例仅 **3.0%**（HM3D），而 VLMnav/Nav-R²/WMNav 均为 100%。
- SOT 得分：Hydra-Nav-IRFT **24.0**（HM3D）vs Nav-R² 1.9（最高 SR 竞争者），提升约 12×。
- 说明频繁推理虽提高 SR，但严重拖累效率；自适应推理是实际部署的关键。

**消融实验关键发现：**
- 记忆模块对 SPL 提升显著（无记忆 SPL=13.9 vs 有记忆 28.8），说明长期空间记忆是路径效率的核心。
- 探索性轨迹数据 vs 最短路径数据：SR 下降 25.4%（HM3D），说明探索能力对高成功率不可或缺。
- Co-training with VQA 防止导航专有数据过拟合，维持泛化性（SR: 69.1→72.9，HM3D）。

<div align="center">
  <img src="/images/Hydra-Nav-realworld-demo.png" width="100%" />
<figcaption>
真实世界导航演示：机器人成功定位 Box、Trash Can、Oven，零样本迁移无需真实环境微调。
</figcaption>
</div>

---

**局限性**

评估仅在 Habitat 模拟器（HM3D/MP3D/OVON）中进行，缺乏在 Isaac Sim 等更高保真度仿真环境中的验证；当前框架专为 object navigation 设计，向移动操作等更复杂具身任务的扩展有待探索。
