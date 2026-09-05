---
layout: post
title: "具身导航周报（2026-08-26 ~ 2026-09-03）"
date:   2026-09-05
permalink: /vln-weekly-2026-09-05/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期 45 条新增全部来自 arXiv，公众号源因上游中转随 Deno Deploy Classic 下线而永久失效缺席。45 篇中仅 2 篇报告 R2R-CE：LookStep 给出可核验的 Val-Unseen 成功率 49.7% 且已开源，Revisiting Topological Graphs 把 VLN-CE 重构为分层 MDP、用拓扑图 frontier 宏动作让闭环 RL 可训练，但只声称 SOTA 不给数值。CGFM-Nav 在控制骨干变量（同 Qwen3-VL-8B）的 GOAT-Bench 对照下把成功率 53.2% 提到 63.0%、SPL 30.0% 提到 39.6%，是本期证据质量最高的导航结果。「验证」正成为跨任务共同部件，而基准重心继续偏离导航：45 项独立工作中 13 项以 LIBERO 为主基准。"
---
## 一、本期结论

- **连续环境 VLN 的训练范式出现向强化学习收敛的迹象。** 本期 45 篇中报告 R2R-CE 的仅 2 篇，其中 *Revisiting Topological Graphs* 把 VLN-CE 重构为分层 MDP，用拓扑图的 frontier 节点作为宏动作空间、训练无关的低层控制器作为状态转移，从而压缩决策视界、让闭环 RL 可解。论文称模仿学习在闭环下的两个老问题（行为克隆的分布偏移、DAgger 偏离轨迹后专家动作歧义）正是 RL 的动机。本报告判断：这条路线把「RL 在 VLN-CE 上样本效率太低」的既有结论重新推上台面，值得优先跟踪。
- **另一条并行路线是压成本，而非抬上限。** *LookStep* 在同等训练设置下用更少数据和更小记忆开销取得 R2R-CE Val-Unseen 成功率 49.7%，手段是放弃认知地图 / 历史帧堆叠 / 外部 3D 工具，改用语言标签生成粗粒度导航进度与未来状态，并自主决定是否把观测写入有界滚动记忆。本报告判断：这与上一条形成对照——一边抬上限，一边压部署成本，且两者评测口径同为 R2R-CE，可直接横向比较。
- **通用导航模型开始用单一 VLM 骨干覆盖多任务多本体。** *LightNav-0* 用统一 token 接口（双通道 pointing 表达与任务/场景/本体无关的空间意图 + 残差向量量化动作 tokenizer 映射到具体本体轨迹）替代任务专用预测头，论文称在全部 10 个公开导航仿真设置上取得单目成功率 state-of-the-art，并在真机上零样本跨本体泛化。本报告判断：该工作的语料规模（2K+ 场景、4K+ 小时）构成主要复现门槛。
- **记忆表征的争论从「容量」转向「更新纪律」。** *CGFM-Nav* 用多模态场景图 + 目标条件语义前沿场，在 GOAT-Bench 上同一 Qwen3-VL-8B 骨干下把成功率从 53.2% 提到 63.0%、SPL 从 30.0% 提到 39.6%；机械臂侧的 *AGM* 则明确主张可靠的具身记忆更依赖有纪律的状态更新而非记忆容量，只在子目标被物理证据验证后才推进进度指针。本报告判断：两项工作从不同任务给出同向证据，对长时序导航的子目标跟踪有直接借鉴价值。
- **本期公众号来源缺席，45 条全部来自 arXiv。** 原 wewe-rss 依赖的中转服务已随 Deno Deploy Classic 于 2026-07-20 下线而永久失效；已迁移至 we-mp-rss（走微信公众平台接口），但新授权账号触发微信频控，四轮采集均为 0 条。因此本期缺少公众号解读视角，跨源合并未发生，「45 条」即「45 项独立工作」。

## 二、优先阅读清单

| 优先级 | 工作 | 任务/场景 | 核心贡献 | 关键证据 | 阅读理由 |
|---|---|---|---|---|---|
| A1 | [Revisiting Topological Graphs](https://arxiv.org/abs/2609.03906v1) | VLN-CE（连续环境） | 分层 MDP + 拓扑图宏动作 + action-aware value head 支撑的 graph-based PPO | 摘要称在 R2R-CE 与 RxR-CE 上达到 state-of-the-art，**未提供可核验数字** | 把闭环 RL 在 VLN-CE 上变得可训练，属路线级变化 |
| A2 | [LookStep](https://arxiv.org/abs/2609.02350v1) | VLN-CE（连续环境） | 语言中心未来状态建模 + 事件驱动有界滚动记忆，去掉认知地图与外部 3D 工具 | R2R-CE Val-Unseen 成功率 49.7%；论文称同等训练设置下优于现有方法，且记忆效率更高、用数据更少 | 本期唯一给出可核验 R2R-CE 数字的工作，且[有开源代码](https://github.com/kunyang-YU/LookStep) |
| A3 | [LightNav-0](https://arxiv.org/abs/2608.30935v1) | 通用具身导航（指令跟随 / 开放词表 ObjectNav / 视觉跟踪） | 统一 token 接口：双通道 pointing + 残差 VQ 动作 tokenizer，无任务专用头 | 论文称 10 个公开导航仿真设置单目成功率均为 SOTA，LightNav-ER 在 8 个具身推理 benchmark 上完整集平均最高。**各设置具体数值摘要未列出** | 单一 VLM 骨干覆盖多任务多本体的完整方案 |
| A4 | [CGFM-Nav](https://arxiv.org/abs/2608.29114v1) | 终身多模态具身导航 | 多模态场景图（显式关系记忆）+ 目标条件语义前沿场（连续探索引导）构成闭环 | GOAT-Bench：同 Qwen3-VL-8B 骨干下成功率 53.2%→63.0%，SPL 30.0%→39.6%（论文自述为初步实验） | 记忆与探索耦合的具体设计，且数字口径清晰 |
| A5 | [VerNav](https://arxiv.org/abs/2609.00920v1) | **离散 R2R**（非 CE） | verifier-first：批量动作验证替代逐步自回归生成，仅对不确定决策调用生成器 | R2R 基准上决策阶段单步 LLM 延迟较自回归方法降低 **10 倍以上**，导航性能称 competitive | 延迟是 LLM-based VLN 落地的硬约束；注意其评测在离散 R2R，**不可与 A1/A2 的 R2R-CE 直接比较** |
| A6 | [CanonNav](https://arxiv.org/abs/2608.30242v1) | 跨平台视觉导航（真机相关） | 相机几何规范化，解耦导航行为与平台相机几何；用离线可通行性估计器的伪标签提供安全与局部进度监督 | 论文称仅用 RGB 推理即持续优于 RGB 基线，困难场景下超过 RGB-D 方法。**未提供数据集与数值** | 跨平台数据复用的机制层贡献，对多机器人数据汇聚有用 |

## 三、重点工作分析

### 1. Revisiting Topological Graphs：把 VLN-CE 的决策视界压到 RL 能训练的尺度

| 维度 | 分析 |
|---|---|
| 问题 | 论文指出闭环 VLN-CE 下模仿学习的两个具体瓶颈：行为克隆遭遇分布偏移；DAgger 在智能体偏离轨迹后专家动作变得歧义。而直接在微动作空间上做 RL 因奖励稀疏而样本效率极低。 |
| 方法 | 重构为分层 MDP，显式解耦高层规划与低层控制：把环境抽象成拓扑图，高层策略在 frontier 节点构成的宏动作空间上决策，训练无关（training-free）的低层控制器充当状态转移。为在动态 frontier 动作空间下评估状态价值，提出 action-aware value head，支撑 graph-based PPO。 |
| 证据 | 摘要称在 R2R-CE 与 RxR-CE 上达到 state-of-the-art，**但未给出任何数值、对照方法或分项指标**。 |
| 价值 | 若结论成立，这是把闭环 RL 引入 VLN-CE 的可行工程路径；宏动作 + 免训练低层控制器的分解方式，与地面机器人现有导航栈（拓扑图 + 局部规划器）天然对齐。 |
| 局限 | 全部证据依赖 SOTA 这一措辞，无法核验幅度；拓扑图构建质量与 frontier 选取策略的敏感性未在摘要中说明；低层控制器 training-free 意味着其能力上限直接约束整体性能。 |
| 建议 | 优先读正文的 value head 设计与 PPO 训练细节，重点核对 R2R-CE Val-Unseen 的 SR / SPL 具体数值及对照基线。 |

### 2. LookStep：用语言标签替代认知地图，压低 VLN-CE 的记忆与数据成本

| 维度 | 分析 |
|---|---|
| 问题 | 论文指出现有 MLLM 驱动的 VLN 遵循下一步动作预测范式，只监督专家动作，训练数据需求大；且依赖认知地图、累积历史帧或外部 3D 工具维护状态，计算与内存开销高。 |
| 方法 | 端到端统一框架，含两个部件：Language Centric Future State Modeling 用语言标签为每个候选动作生成粗粒度导航进度与未来状态；Event Driven Rolling Memory 自主决定是否把每帧观测以某种语义角色写入有界滚动记忆。 |
| 证据 | R2R-CE Val-Unseen 成功率 **49.7%**；论文称同等训练设置下优于现有方法，同时记忆效率更好、数据用量更少。**摘要未给出 SPL、nDTW 等其他指标，也未列出对照方法的具体数值。** |
| 价值 | 直接面向连续环境，且优化目标是部署成本（内存、数据量），与追求上限的路线互补；有界记忆的写入决策机制可独立移植到其他导航框架。 |
| 局限 | 仅单一指标（SR）且仅 Val-Unseen，无法判断是否以路径效率为代价；同等训练设置的界定需回正文确认；49.7% 的绝对水平需与同期 R2R-CE 榜单对照才有意义。 |
| 建议 | 本期最值得优先复现的工作——有[开源代码](https://github.com/kunyang-YU/LookStep)且目标明确。重点验证有界滚动记忆的写入策略在长指令上的表现。 |

### 3. LightNav-0：单一 VLM 骨干覆盖多任务与多本体

| 维度 | 分析 |
|---|---|
| 问题 | 论文指出现有导航系统依赖任务专用或本体专用组件，割裂感知、推理与动作，泛化受限；而 VLM 已编码视觉定位、空间推理、pointing 等空间先验，却很少被直接用于机器人控制。 |
| 方法 | 统一 token 接口：双通道 pointing 表达任务 / 场景 / 本体无关的空间意图；残差向量量化动作 tokenizer 把该意图映射为特定本体的精确轨迹。配合时序感知的视觉历史压缩、ER 中期训练、监督微调与强化学习。训练语料覆盖 2K+ 场景、4K+ 小时具身导航数据。 |
| 证据 | 论文称 LightNav-ER 在 8 个具身推理 benchmark 上取得最高完整集平均；LightNav-0 在全部 10 个公开导航仿真设置上取得单目成功率 state-of-the-art；真机评测显示跨本体、跨场景以及静态 / 动态目标的零样本泛化。**摘要未列出任一设置的具体数值。** |
| 价值 | 空间意图与本体动作解耦的接口设计，是把一个模型部署到多种地面平台的可行抽象；对同时维护多机器人的团队价值直接。 |
| 局限 | 数据规模构成主要复现门槛；10 个设置全 SOTA 缺少数值支撑，无法判断各任务提升是否均衡；真机结论只有定性描述。 |
| 建议 | 作为通用导航模型的架构参考精读，重点看双通道 pointing 的具体定义与动作 tokenizer 的本体适配方式；暂不列入复现计划。 |

### 4. CGFM-Nav：显式语义记忆与语义引导探索的闭环耦合

| 维度 | 分析 |
|---|---|
| 问题 | 论文指出现有环境表征难以同时支撑显式语义记忆与连续探索引导——要么能检索已见目标，要么能引导探索未见区域。 |
| 方法 | CGFM 是持久多模态场景表征：把物体、空间关系与视觉观测组织为多模态场景图，支撑目标检索与跨任务长时序推理；当无可靠目标匹配时，把图证据投影为目标条件的语义前沿场，引导探索走向语义上有希望的前沿与区域。CGFM-Nav 在其上叠加任务相关子图选择、VLM 推理与验证反馈，构成闭环决策。 |
| 证据 | GOAT-Bench 上，**同一 Qwen3-VL-8B 骨干**下总体成功率 53.2%→63.0%、SPL 30.0%→39.6%。论文自述为初步实验（preliminary）。 |
| 价值 | 控制了骨干变量的对照，使 +9.8 个百分点 SR、+9.6 个百分点 SPL 的增益可归因于表征与探索机制本身，而非模型规模；这是本期证据质量最高的导航结果之一。 |
| 局限 | 作者自述初步；仅 GOAT-Bench 单一 benchmark，未在 R2R-CE / RxR-CE 上验证；场景图构建的计算开销与错误累积未在摘要中讨论；无真机结果。 |
| 建议 | 跟踪其正式版本；语义前沿场的图证据投影部分可单独抽出，接到现有 ObjectNav 探索策略上做消融。 |

## 四、可迁移方法

| 来源方向 | 工作 | 可迁移机制 | 可接入地面导航的位置 | 风险/前提 |
|---|---|---|---|---|
| 机械臂 / 冻结 VLA | [AGM](https://arxiv.org/abs/2608.29537v1) | 成就锚定记忆：任务表示为带进度指针的子目标序列，**仅在当前子目标被物理证据验证后**才推进指针；本体感知交互线索决定何时验证，点跟踪与语言条件跨视角比较（2.43M 参数验证头）决定验证了什么 | 长时序 VLN 的子目标完成度判定，替代「发出动作即视为完成」的乐观更新 | 摘要中 RoboMME Counting 的 PickXTimes / BinFill 数值在原文中缺失，**无法核验提升幅度**；验证信号依赖操作任务的接触线索，导航需另找物理证据源 |
| 空地协同 VLN | [AGC-VLN](https://arxiv.org/abs/2609.03483v1) | 共享鸟瞰图作为协作接口：无人机在全局俯视图上把地面车上报位姿与 VLM 锚定目标渲染为 CAR / GOAL 标记并附距离标签，地面车据此获得第一人称视角无法提供的全局空间上下文，用冻结 VLM 规划沿路路径并闭环执行 | 地面车缺少全局上下文时的外部俯视信息注入（不限于无人机，也可来自固定摄像头或预建地图） | CARLA-Air Town10HD 100 个闭环 episode，联合成功率 77.0%，较较弱个体（无人机 50.0%）高 27.0 点、较最强已发表单智能体基线（Travel UAV 53.0%）高 24.0 点；结论限于仿真城市道路场景，室内可迁移性未验证 |
| 长时序智能体框架 | [EmbodiedSkills](https://arxiv.org/abs/2609.01281v1) | 把感知、规划、执行、**进度验证与恢复**统一编排的框架结构 | VLN 系统的失败恢复与重规划层 | 证据全在操作任务（RoboTwin 2.0 50 任务均值 86.20%、四个 LIBERO 套件 97.40%），导航侧无验证 |
| 失败纠正 | [Training-Free Action Correction](https://arxiv.org/abs/2608.29967v1) | 用语言反馈在不重训的前提下纠正 VLA 部署期失败 | 导航策略在线纠偏，避免为每类失败重新训练 | 证据为 LIBERO 上的操作任务；导航的失败模式（走错分岔、语义误绑定）与操作不同 |
| 评测方法 | [R2S-Eval](https://arxiv.org/abs/2609.03276v1) | 用 VLM 做 real-to-sim 标定后在仿真中评测真机策略 | 降低导航策略真机评测的人力成本与不稳定性 | 面向操作场景标定；导航需要的场景尺度与动态性更高 |
| 评测方法 | [GeoAgent](https://geoagent-benchmark.github.io) | 把静态图像任务改造为需主动探索的具身导航式评测；论文称 agentic 导航较静态图像基线显著提升准确率 | 评测设计思路：为感知类能力补上「先探索再判断」的具身环节 | 任务是地理定位而非目标导航；论文同时报告模型在发达 / 发展中地区上的显著偏差，以及先验错误时自我改进能力差 |

## 五、分类速览

### 5.1 地面 VLN / ObjectNav / 语义导航

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| Revisiting Topological Graphs | VLN-CE + 闭环 RL | 见重点分析 1 | A | [arXiv](https://arxiv.org/abs/2609.03906v1) |
| LookStep | VLN-CE 效率 | 见重点分析 2 | A | [arXiv](https://arxiv.org/abs/2609.02350v1) |
| LightNav-0 | 通用具身导航 | 见重点分析 3 | A | [arXiv](https://arxiv.org/abs/2608.30935v1) |
| CGFM-Nav | 终身多模态导航 | 见重点分析 4 | A | [arXiv](https://arxiv.org/abs/2608.29114v1) |
| VerNav | 离散 R2R 低延迟 | verifier-first 用批量动作验证替代逐步自回归，决策阶段单步 LLM 延迟降低 10 倍以上 | A | [arXiv](https://arxiv.org/abs/2609.00920v1) |
| CanonNav | 跨平台视觉导航 | 相机几何规范化解耦导航行为与平台几何，仅 RGB 推理称超过 RGB-D 方法 | A | [arXiv](https://arxiv.org/abs/2608.30242v1) |
| AGC-VLN | 空地协同 VLN | 共享鸟瞰图作为训练无关的协作接口，CARLA-Air 联合成功率 77.0% | B | [arXiv](https://arxiv.org/abs/2609.03483v1) |

### 5.2 记忆、地图、规划与评测

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| AGM | 具身记忆 | 子目标进度指针仅在物理证据验证后推进；主张记忆可靠性取决于更新纪律而非容量 | B | [arXiv](https://arxiv.org/abs/2608.29537v1) |
| EmbodiedSkills | 智能体编排 | 统一编排感知 / 规划 / 执行 / 进度验证 / 恢复的 VLA 框架 | B | [arXiv](https://arxiv.org/abs/2609.01281v1) |
| Training-Free Action Correction | 失败纠正 | 用语言反馈在不重训前提下纠正 VLA 部署期失败 | B | [arXiv](https://arxiv.org/abs/2608.29967v1) |
| R2S-Eval | 评测 | VLM 驱动的 real-to-sim 标定评测流程 | B | [arXiv](https://arxiv.org/abs/2609.03276v1) |
| GeoAgent | 评测 | Street View 具身导航式地理定位 benchmark；报告发达 / 发展中地区偏差 | B | [项目页](https://geoagent-benchmark.github.io) |
| Drive the Thoughts | 运行时监控 | 监控 VLA 推理链与轨迹一致性，分析称 33.3% 的 CoT 不可靠 | C | [arXiv](https://arxiv.org/abs/2608.29583v1) |
| LAVLA | 可解释性 | 对 GR00T N1.5 动作解码器做逐层隐空间聚类分析 | C | [arXiv](https://arxiv.org/abs/2609.02634v1) |

### 5.3 具身 VLA / 移动操作

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| MINERVA | 小模型 | 0.54M 参数策略在四个标准 LIBERO 套件 2,000 次 rollout 上均值 95.1%，仅低于所报 LeRobot π0.5 结果 2.4 点 | C | [arXiv](https://arxiv.org/abs/2609.03715v1) |
| DriftingVLA | 一步生成 | 逐维时间漂移的原生一步动作生成，LIBERO 98.32%、RoboTwin 2.0 81.09%、真机六任务 77.67% | C | [arXiv](https://arxiv.org/abs/2608.29749v1) |
| Temporal Forcing | 4D 表征 | 时序表征对齐缓解观测混淆，LIBERO 98.8%（较基座模型 +2.2 点） | C | [arXiv](https://arxiv.org/abs/2608.30643v1) |
| SMILE | 动作平滑 | 预测 B 样条系数抑制动作块抖动，LIBERO 98.0% 且 1.1 倍加速 | C | [arXiv](https://arxiv.org/abs/2608.29432v1) |
| GIFT | 中间特征监督 | 面向动作的结构化监督，零样本迁移 LIBERO-Plus 达 79.6% / 72.6% / 87.8% | C | [arXiv](https://arxiv.org/abs/2609.04193v1) |
| VLAct（Beyond Data Scaling） | 继续预训练 | 以表征为中心的继续预训练，LIBERO-Plus 82.6%、RoboTwin 2.0 92.5% | C | [arXiv](https://arxiv.org/abs/2608.27550v1) |
| PHR-VLA | 规划视界 | 腕部相机的接触中心潜在动力学监督，LIBERO 84.1%→88.4%，真机拆解任务 63.3%→82.5% | C | [arXiv](https://arxiv.org/abs/2608.27609v1) |
| PredVLA | 小模型 | 预测性感觉运动建模，LIBERO 短时序三套件 86.9%、四套件 75.4% | C | [arXiv](https://arxiv.org/abs/2608.26673v2) |
| AdaVLA | 推理加速 | 自适应步长流匹配，训练无关加速 | C | [arXiv](https://arxiv.org/abs/2608.29208v1) |
| Knowing When to Stop | 动作分块 | 用内部交叉注意力动态自适应决定动作块长度 | C | [arXiv](https://arxiv.org/abs/2609.00908v1) |
| REFACTOR-VLA | 技能库 | 无监督学习带类型的运动程序库；跨 provider 平均成对 NMI 0.705（95% 置信区间 [0.683, 0.729]） | C | [arXiv](https://arxiv.org/abs/2609.01215v1) |
| WISE | 后训练效率 | 世界模型引导的想象调度，π0 / π0.5 上 GPU 计算时间较全量想象减少约 80% | C | [arXiv](https://arxiv.org/abs/2609.03681v1) |
| PAVE | 表征对齐 | 轨迹相对的多视界转移对齐（剩余 episode 的 25 / 50 / 75 / 100%） | C | [arXiv](https://arxiv.org/abs/2608.30378v2) |
| CometVLA | 协同训练 | 具身数据金字塔协同训练以补足物理常识 | C | [arXiv](https://arxiv.org/abs/2608.30289v1) |
| SymVD | 蒸馏 | 对称视觉语言动作蒸馏，降低任务迁移的数据与重训成本 | C | [arXiv](https://arxiv.org/abs/2608.29828v1) |
| DREAM | 数据生成 | 部署期 real-to-sim 生成演示数据以适配新工作区 | C | [arXiv](https://arxiv.org/abs/2608.29078v1) |
| GRAFT | 在线 RL | 面向精细生物医学操作的在线强化适配 | C | [arXiv](https://arxiv.org/abs/2608.27079v2) |
| DeicticVLA | 指令模式 | 统一语言与指示性手势两种指令模式；未见类别上两种模式均 100% 成功，联合训练 LI 基线为 16.7% | C | [arXiv](https://arxiv.org/abs/2608.28108v1) |
| HINT | 长时序意图 | 从简单总体指令推断人类意图并随视觉观测持续适配 | C | [arXiv](https://arxiv.org/abs/2609.02653v1) |
| Evidence-Gated Regularization | 多模态鲁棒 | 缓解模态纠缠，SR 12.5%→16.4%（全模态）、9.4%→16.5%（无效传感器）、2.8%→6.1%（单传感器回退） | C | [arXiv](https://arxiv.org/abs/2609.03142v1) |
| ZETA | 跨本体迁移 | 受控研究表明预训练加入 5% 目标本体数据即提升目标本体平均进度 13.4 个百分点 | C | [arXiv](https://arxiv.org/abs/2609.02546v1) |
| FWBC-VLA | 力感知 | 面向接触密集 loco-manipulation 的全身力补偿 | C | [arXiv](https://arxiv.org/abs/2609.03889v1) |
| Scaling Bimanual Household Manipulation | 数据集 | 发布 1,500 小时双臂家务操作演示并做在策略纠正 | C | [arXiv](https://arxiv.org/abs/2609.03591v1) |

### 5.4 无人机、自动驾驶与其他低相关方向

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| MulDP | 四足 parkour 导航 | 多模态扩散策略生成导航速度指令，并发布四足 parkour 导航数据集 QPND；无语言指令成分 | C | [arXiv](https://arxiv.org/abs/2609.03984v1) |
| LaPla | 自动驾驶 | 潜空间对齐规划，nuScenes 上长视界 L2 误差较 SOTA VLA 方法降低 15.52% | C | [arXiv](https://arxiv.org/abs/2609.04070v1) |
| Rethinking Language's Role | 自动驾驶 | 讨论车载场景下语言模块的延迟与显存代价 | C | [arXiv](https://arxiv.org/abs/2608.30144v1) |
| Aligning Multi-Trajectory Supervision | 自动驾驶 | 多轨迹模仿与 GRPO 结合时的轨迹选择问题 | C | [arXiv](https://arxiv.org/abs/2608.30122v1) |
| Towards Zero-Shot Transfer for Driving VLAs | 自动驾驶 | 驾驶 VLA 的跨本体零样本迁移 | C | [arXiv](https://arxiv.org/abs/2609.02341v1) |
| Degradation-Tolerance Benchmark | 自动驾驶 | 纯相机端到端驾驶在模糊 / 噪声 / 弱光 / 天气 / 丢帧 / 内存故障下的容忍度基准 | C | [arXiv](https://arxiv.org/abs/2608.29005v1) |
| MLLMs as Drone VLA Agents | 无人机 | 评测 MLLM 直接进入无人机控制回路（指挥 / 接近 / 跟踪 / 搜索） | C | [arXiv](https://arxiv.org/abs/2609.01404v1) |
| Taxonomy of Construction Task Activities | 领域分析 | 建筑工人活动分类法及机器人所需能力清单 | C | [arXiv](https://arxiv.org/abs/2608.25395v2) |

### 5.5 资讯与非论文

本期无资讯类条目。公众号来源因上游服务故障未产出条目，原因见「一、本期结论」末条。

## 六、趋势判断与行动建议

### 趋势

- **VLN-CE 的方法竞争正在从「更好的表征」转向「更好的训练信号」。** Revisiting Topological Graphs 用宏动作压缩决策视界使闭环 RL 可训练，LookStep 用语言标签构造中间监督（导航进度、未来状态）替代仅监督专家动作——两者都在绕开纯模仿学习的监督稀疏问题，但手段不同。
- **「验证」正成为跨任务的共同部件。** VerNav 用 verifier 替代逐步生成以降延迟，AGM 用验证头决定记忆指针是否推进，CGFM-Nav 把验证反馈纳入闭环决策，EmbodiedSkills 把进度验证列为框架一级组件。本报告判断：验证器的设计（用什么证据、何时触发）正在成为独立的研究对象。
- **推理成本被当作一等指标。** LookStep（记忆效率、数据量）、VerNav（延迟 10 倍以上）、MINERVA（0.54M 参数达 LIBERO 95.1%）、AdaVLA 与 DriftingVLA（推理步数）分别从记忆、延迟、参数量、采样步数四个维度压成本。本报告判断：这类工作对真机部署的价值高于榜单名次。
- **本期机械臂操作（LIBERO 系 13 篇）在数量上远超地面导航工作。** 本报告判断：这是 arXiv 该时段的领域分布事实，不宜据此调整导航方向的投入。

### 研究空白

- **R2R-CE 的可核验数字稀缺。** 本期 45 篇中仅 2 篇报 R2R-CE，其中 1 篇（Revisiting Topological Graphs）只声称 SOTA 而不给数值，横向对比因此无法完成。
- **导航侧缺少 AGM 那样的物理证据来源。** 操作任务可用本体感知接触线索判定子目标完成，导航中等价的证据是什么（到达判定、语义匹配置信度、可通行性变化）尚无系统研究。
- **跨平台数据复用只有机制、没有规模验证。** CanonNav 提出相机几何规范化，但未给出数据集与数值；多机器人数据汇聚到底能带来多少增益仍是空白。

### 建议动作

| 动作 | 目标 | 优先级 |
|---|---|---|
| 复现 | LookStep：有开源代码，R2R-CE Val-Unseen SR 49.7% 口径明确，重点验证有界滚动记忆的写入策略 | 高 |
| 精读 | Revisiting Topological Graphs：核对正文中 R2R-CE / RxR-CE 的实际数值与对照基线，判断 SOTA 声称的成色 | 高 |
| 精读 | CGFM-Nav：控制骨干变量的 GOAT-Bench 对照是本期证据质量最高的导航结果之一，关注其正式版本 | 高 |
| 借鉴 | AGM 的「验证后才推进进度指针」更新纪律，设计导航子目标的物理证据判定 | 中 |
| 架构参考 | LightNav-0 的双通道 pointing 与残差 VQ 动作 tokenizer；因语料规模门槛暂不复现 | 中 |
| 跟踪 | VerNav 的 verifier-first 延迟优化；注意其为离散 R2R，迁移到 CE 需重新验证 | 中 |
| 暂缓 | LIBERO 系 13 篇操作工作与驾驶 / 无人机方向，仅保留 benchmark 动态感知 | 低 |
