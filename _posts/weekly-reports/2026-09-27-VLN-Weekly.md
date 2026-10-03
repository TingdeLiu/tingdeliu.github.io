---
layout: post
lang: zh-CN
translation_id: vln-weekly-2026-09-27
title: "具身导航周报（2026-09-16 ~ 2026-09-24）"
date:   2026-09-27
permalink: /vln-weekly-2026-09-27/
tags: [VLN, VLA, Embodied Navigation, Embodied Agent, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "R2R-CE 零样本出现 79.0% SR 的新声明（GPT-6-Astra，Codex harness + ultra 推理），但评测集、推理成本与停止口径均未在摘要中给出；一周前同一模型在直接工作流下为 52.0%，核验前不宜与训练式方法横向比较。同一基准上的数字越来越多地出自不同信息条件：SparseNav 免训练、自主，val-unseen 42.8%；Talk2Escape 借助 oracle 或人工纠正，66.0%。具身侧 harness 开始反哺训练，World Action Agent 用 harness 轨迹微调 Qwen3.5-9B，分布外成功率从 1.7% 升到 43.3%。"
---
## 一、本期结论

- **R2R-CE 零样本出现 79.0% 的新声明，但目前核验不了它说明了什么。** GPT-6-Astra 在 Codex harness 的最小接口下、只用单目 RGB，不做导航微调、不用训练过的航点预测器、不用预建地图，论文报告 ultra 推理档位 SR 79.0%，称比最强零样本高 13.0 点、比最强监督式高 6.9 点。一周前同一模型在直接调 API 的工作流里只有 52.0%（Open-Nav 100 episode 协议中的 50 条），且其中 16.0% 是步数上限触底而非主动停止。两篇之间 harness、推理档位、episode 集合与成功判定可能同时变了，27 点差距无法归因。本报告判断：若 79.0% 沿用 100 episode 子集，二项分布下 95% 置信区间半宽约 8 个百分点，「超过监督式 6.9 点」落在统计误差内，且监督式结果多在完整 val-unseen（1,839 episode）上报告，并非同集比较。
- **同一个 R2R-CE 上的数字，越来越多地出自互不相同的信息条件。** 本期三个新结果：SparseNav 免训练、自主、42.8%（val-unseen）；Talk2Escape 在打转或偏离时向算法 oracle 或人类求助，66.0%；GPT-6-Astra 用闭源前沿模型的 ultra 推理档位，79.0%。三者的差距首先来自「系统能拿到什么信息」，其次才是方法本身。来源之间还有直接冲突：Talk2Escape 称其 66.0% 超过当前监督式 SOTA，而按 GPT-6-Astra 的表述反推，作者采用的监督式最佳为 72.1%，上期 GroundingVLN 在完整 val-unseen 上也报告了 69.9%。
- **Harness 从导航扩散到整个具身栈，并开始反哺模型训练。** 上期 3 个 harness 工作，本期 5 个直接以 harness 为主题（HarnessPAI、RegenHarness、AdaHVLA、Robo-Harness K1、World Action Agent），另有 GPT-6-Astra 在 Codex harness 中评测。新出现的共同做法有两点。一是只改 harness 配置、不改模型权重，靠执行记录驱动修订。二是把 harness 的执行轨迹蒸馏回模型：HarnessPAI 用收敛程序采集的数据微调 π0.5，LIBERO-PRO 再涨 38.8 点；World Action Agent 用 harness 轨迹微调 Qwen3.5-9B，分布外成功率从 1.7% 升到 43.3%；Robo-Harness K1 的学生模型只用 107 个教师 episode。其中只有 AdaHVLA 报了导航侧数字（NaVILA-LH 从 22.5% 升至最高 57.5%）。
- **导航与 VLA 两边同时收敛到「有界记忆」：固定尺寸的状态，替代不断增长的上下文。** 导航侧，VNT-PA 用位姿索引的关键帧集合作环境表征，PointNav（HM3D）SR 93.3%、SPL 90.4%；SparseNav 只存当前子指令需要的稀疏地标；MemCtrl 用约一半上下文取得平均 +6%、长指令子集 +20%。VLA 侧，SmoLSTM、MemBodied、StateMem 三篇都用固定尺寸的记忆状态替代历史帧堆叠。SmoLSTM 的消融最直接：每步重置循环状态，完整任务成功率从 77.5% 跌到 7.0%。
- **评测开始系统性地质疑排行榜。** PopNavShift 显示社会导航策略的排名依赖指标：只改时间压力，按机器人用时排名有 8.6% 反转，按行人延迟有 22.4%、按最差十分位延迟有 23.9% 反转。RoboFollow 指出「低场景熵」下语言是冗余的，九个 VLA/WAM 在基础设定上的成绩不能可靠迁移到扰动设定。DPed-VLN 把社会安全指标与导航效率合并评估。VLN 可解释性工作则显示，策略内部编码了导航进度。
- 本期共 124 篇独立新工作，其中主池（导航 + 具身Agent）38 篇，另有 2 篇边界条目（ACE、WORLDS）一并分析；公众号源 0 条。arXiv 检索接口本期持续不可用，改用 OAI-PMH 元数据按原检索式本地匹配；在上期同一时间窗做对照，主池召回为 38/41。

## 二、优先阅读清单

1. **[GPT-6-Astra Lights Up](https://arxiv.org/abs/2609.29861v1)** · 零样本 VLN-CE，单目 RGB
   - 贡献：通用基础模型在 Codex harness 最小接口下自主决定观察、移动与停止。
   - 证据：R2R-CE SR 79.0%（ultra 推理）；episode 集合、SPL、推理成本未在摘要中给出。
   - 理由：当前零样本最高声明，需先核验口径再引用。
2. **[SparseNav](https://arxiv.org/abs/2609.26408v1)** · 免训练 VLN-CE + 四足真机
   - 贡献：按当前子指令决定感知什么，只维护几何 BEV 与稀疏地标记忆。
   - 证据：R2R-CE val-unseen SR 42.8%，RxR-CE val-unseen SR 40.7%。
   - 理由：自主、免训练、两个基准都报数，硬件栈常见，复现门槛低。
3. **[Talk2Escape](https://arxiv.org/abs/2609.28296v1)** · 对话式纠错 VLN
   - 贡献：检测到打转或偏离后，向 oracle 或人类发出接地提问，把开环导航变成闭环。
   - 证据：R2R-CE SR 66.0%；在 R2R-CE、RxR-CE、VLNVerse 上对多种基础智能体一致提升。
   - 理由：检测器可以单独复用；其 SOTA 声明与其他来源冲突。
4. **[VNT-PA](https://arxiv.org/abs/2609.21212v1)** · PointNav（HM3D）
   - 贡献：注意力由位姿差决定，关键帧集合即环境表征，可跨轨迹融合。
   - 证据：HM3D PointNav SR 93.3%，SPL 90.4%。
   - 理由：不建显式地图也能复用历史经验。
5. **[AdaHVLA](https://arxiv.org/abs/2609.29204v1)** · 长程 VLA 执行的自适应 harness
   - 贡献：以可检验的假设驱动 harness 修订，修订图保留证据与效果。
   - 证据：NaVILA-LH 平均测试成功率 22.5% → 最高 57.5%（仿真）。
   - 理由：本期 harness 工作中唯一有导航侧数字。
6. **[NaviScale](https://arxiv.org/abs/2609.27218v1)** · 语义地图 ObjectNav
   - 贡献：拼接真实户型与房间级语义地图，大规模生成补全预测的训练数据。
   - 证据：HM3D SR 64.3% / SPL 34.8%，MP3D SR 43.1% / SPL 16.8%。
   - 理由：不改预测架构、只扩数据的对照点。
7. **[What do VLM-Based VLN Models Rely on](https://arxiv.org/abs/2609.24576v1)** · VLN 可解释性与激活引导
   - 贡献：用干预式指标测各模态对决策的因果影响，并提取行为激活向量。
   - 证据：摘要未提供可核验数字。
   - 理由：给出「策略是否在用指令」的直接诊断手段。
8. **[DPed-VLN](https://arxiv.org/abs/2609.21504v1)** · 动态行人 VLN 基准
   - 贡献：Habitat 3.0 上 33,093 个 episode，联合评估导航效率与社会安全。
   - 证据：摘要只给相对结论，未提供可核验数字。
   - 理由：地面 VLN 进入动态人群场景的第一个成体量基准。

## 三、重点工作分析

### 1. GPT-6-Astra 两篇对读：79.0% 与 52.0% 之间，没有一个变量是被控制的

**问题。** 通用基础模型不加任何导航专用组件，能否完成连续环境 VLN。上期那篇（[How Far Can GPT-6-Astra Go?](https://arxiv.org/abs/2609.20116v2)，原题 GPT-6-Astra in a Navigation Workflow，本期以改题后的 v2 重新出现，不计入新增）已经暴露出：局部判断正确，不等于能持续推进并在正确位置停下。

**方法。** 新文用 Codex harness 下的最小接口，只给单目 RGB；模型自行决定何时观察、如何移动、何时停止；不做导航微调，不用训练过的航点预测器，不用预建地图。摘要只报了 ultra 推理档位，说明评测了多个档位。旧文是直接调 API 的观测—决策—执行工作流，每次请求带入选定观测、执行反馈和保留的进度记录。

**证据。** 新文报告 R2R-CE SR 79.0%，称比最强零样本高 13.0 点、比最强监督式高 6.9 点；摘要没有给 SPL、nDTW、episode 数和推理成本。旧文报告在 Open-Nav 100 episode 中的 50 条上 SR 52.0%、SPL 48.9%、nDTW 70.8%，其中主动 STOP 成功 36.0%，步数上限触底 16.0%。

**价值。** 本报告判断：如果 79.0% 在可比协议下成立，「基础模型 + 最小 harness」就已超过专门训练的 VLN-CE 系统，自研导航模块的价值需要重新论证。这正是作者第四点结论的主张。

**局限。**

- 评测集没有写明。零样本文献里的「常用零样本 R2R-CE 基准」多指 Open-Nav 的 100 episode 子集（本报告判断）。若如此，95% 置信区间半宽约 8 个百分点，「超过监督式 6.9 点」不具统计意义，且与全量 val-unseen 上的监督式结果不同集。
- ultra 推理的单步延迟与调用成本未报，真机可用性无从判断。
- 两篇之间 harness、推理档位、episode 集合、成功判定可能同时变化，27 点差距无法拆分归因。
- 作者自己也写明：即使用 ultra 推理，路线执行与目标验证仍不可靠，看似合理的局部地标匹配不一定导向正确完成。

**建议。** 精读正文，确认 episode 集合、各推理档位的 SR 与成本、成功是否要求主动 STOP；若日志公开，统计失败 episode 中「未停 / 错停 / 偏航」的构成。在核验前，不要把 79.0% 放进与训练式方法的横向对比。

### 2. SparseNav：少感知一点，反而是更可复现的零样本路线

**问题。** 基于地图的 VLN 若对所有可见物体都做语义标注，既增加感知开销，又会用无关物体淹没 VLM 规划器读到的空间表征。

**方法。** 持续维护轻量的几何 BEV 地图与稀疏地标记忆。指令管理器跟踪进度，给出当前子指令对应的地标查询；只有当被查询的地标可见、且其度量位置对下一步决策有用时，才调用开放词表分割。VLM 在「前沿点 + 局部方向航点」的混合候选中做选择。

**证据。** 论文报告：不做任何训练，R2R-CE val-unseen SR 42.8%，RxR-CE val-unseen SR 40.7%；另有感知策略与各组件的受控消融（摘要未给消融数字和 SPL）。真机部署在 Unitree Go2 上，RealSense D455 负责建图与地标接地，Livox MID-360 负责定位，不用预建地图，在多个室内环境验证。

**价值。** 本期三个 R2R-CE 新结果中，唯一同时满足自主、免训练、写明 val-unseen 分割、并在 RxR-CE 上同步报数的工作；硬件栈与常见四足平台一致。「按子指令决定感知什么」可以直接嫁接到任何基于地图的零样本 VLN 系统。

**局限。** 摘要未说明是否为完整 val-unseen，也未说明所用 VLM 与单步延迟；42.8% 不能与多在 100 episode 子集上报告的其他零样本结果直接比较。稀疏地标记忆会丢掉指令没提到、但对避障或重定位有用的物体。

**建议。** 复现指令条件化的感知触发模块，测量感知调用次数与延迟的实际下降；在同一 episode 集上与稠密语义地图基线做对照。

### 3. Talk2Escape：检测器比对话更值得带走

**问题。** 单轮 VLN 是开环执行：感知混叠、传感噪声、里程计漂移带来的小偏差持续累积，最终任务失败，而系统没有内建的恢复机制。

**方法。** 一个轻量视觉语言模块持续监测智能体运动学；检测到局部打转或严重轨迹偏离时，把第一人称观测转成简短的接地问题，向算法 oracle 或人类请求纠正反馈。框架与底层导航模型无关。

**证据。** 论文报告在 R2R-CE、RxR-CE、VLNVerse 上对多种基础智能体都有一致提升；R2R-CE SR 66.0%，称超过当前监督式与零样本 SOTA；在 Go2 四足上做了 sim-to-real 验证。摘要未给 episode 集合、SPL、每 episode 平均提问次数，也未区分 oracle 与人类反馈各自的成绩。

**价值。** 打转/偏离检测器本身不依赖外部反馈，可以单独拿来做「何时重规划 / 何时回溯」的触发器。这恰好对应上期指出的停止与恢复薄弱环节。

**局限。** 纠正信息来自 oracle 或人类，属于带特权信息的设定，其 SR 与自主智能体不在同一信息条件下。SOTA 声明存在来源冲突：GPT-6-Astra 新文反推的监督式最佳为 72.1%，上期 GroundingVLN 在完整 val-unseen 上报告 69.9%，都高于 66.0%。

**建议。** 读正文确认提问频次与 oracle 的具体设定；把检测器接到自有系统，做成不接人工的「自纠错」版本，量化没有外部反馈时的净收益。

### 4. VNT-PA：让注意力看位姿差，而不是看时间顺序

**问题。** 学习式导航策略按时间顺序编码观测历史，很难复用先前遍历同一环境时的经验；能复用经验的系统通常要先显式建图或建拓扑图，再在上面规划。

**方法。** Transformer 规划器的上下文是一组按相机位姿索引的深度关键帧。以位姿作位置编码，注意力由关键帧之间的位姿差决定，与时间顺序无关。训练时模仿真值网格上的最短路径规划器；推理时只用当前位姿和目标位置去查询空间上下文。

**证据。** 论文报告 HM3D 验证场景 PointNav SR 93.3%、SPL 90.4%；导航性能与训练效率都优于「把同一上下文编码为时间序列」和「把位姿当输入特征」的基线；定位噪声下的退化比基于显式地图的规划更平缓（摘要未给噪声实验的数字）。

**价值。** 上下文是位姿索引的集合，测试时可以融合来自不同轨迹的帧，为「多次进入同一环境时复用经验」提供了不建显式地图的方案。它与上期 Navi-Agent 形成两端对照：一个完全依赖位姿，一个完全放弃坐标。

**局限。** 任务是 PointNav（目标以坐标给出），不涉及语言；依赖位姿估计；示教来自真值网格上的最短路径，真实环境没有这种监督；上下文关键帧从何而来（预先遍历还是本回合积累）摘要未说明。

**建议。** 精读关键帧上下文的构造方式；在 VLN 系统里把「已访问区域记忆」换成位姿索引的关键帧集合，做小规模验证。

### 5. AdaHVLA：本期唯一给出导航侧数字的 harness 工作

**问题。** VLA 擅长局部控制与指令跟随，但长程任务需要持久记忆与规划；task harness 能保留历史、跟踪阶段进度，难点在于让两者对齐。

**方法。** 用机器人执行经验精化基于代码的协调策略。证据分析、harness 修订、行为评估由不同 agent 在相互隔离的上下文中完成；修订由可检验的「协调假设」驱动，再用后续 rollout 检验预期效果是否出现。状态化修订图把证据、假设、修订与观测效果连起来，保留备选 harness 与适配记忆，支持跨任务、跨环境的持续适配。

**证据。** 论文报告仿真中 NaVILA-LH 平均测试成功率从 22.5% 提升到最高 57.5%；三个 VLA 骨干上的操作任务，成功率比初始 harness 最多高 30.8 点；真机部署为定性展示。NaVILA-LH 的任务构成摘要未说明；「最高」表明 57.5% 是最优配置而非均值；对照是作者自己的初始 harness，不是外部方法。

**价值。** 「假设 → 修订 → rollout 检验」的闭环，与同期 [RegenHarness](https://arxiv.org/abs/2609.27612v1) 的「固定回归检查 + 发布授权后才接受修改」，是同一问题的两种约束强度：前者靠实验检验修订，后者靠门控限制修订。RegenHarness 还明确区分「模型提议 / 控制器终止 / 已验证完成」，并指出完成与否取决于执行历史而非离终点多近——这与 GPT-6-Astra 暴露的停止问题直接相关。

**局限。** 仅仿真有数字；适配需要多少次 rollout、成本多高未报；harness 修订是否会过拟合到测试任务不清楚。RegenHarness 只有真机案例，没有量化指标。

**建议。** 核实 NaVILA-LH 的定义与 57.5% 对应的配置；关注修订图的数据结构，它可以直接用来组织导航系统的失败案例库。

## 四、可迁移方法

只收录迁移路径能说清楚的工作。

- **VLA 记忆：[SmoLSTM](https://arxiv.org/abs/2609.22854v1)、[MemBodied](https://arxiv.org/abs/2609.28256v1)**
  - 机制：固定尺寸循环状态替代历史帧堆叠。SmoLSTM 在 LIBERO-Mem 上完整任务成功率 77.5%、子目标覆盖 85.1%，可训练参数 0.04B，每步重置状态则跌到 7.0%。MemBodied 用联想状态加首帧场景锚点，RMBench 五个记忆任务上平均成功率为无状态策略的 7.81 倍、普通循环记忆的 2.98 倍。
  - 接入位置：VLN 历史编码，替代滑窗或 KV 缓存式的历史上下文。
  - 前提与风险：操作任务 episode 短，记忆需求多为遮挡与同外观物体区分；导航需要的是长程空间记忆，需另行验证。
- **VLA 推理调度：[React When You Need To](https://arxiv.org/abs/2609.22587v1)**
  - 机制：按上次推理以来的场景变化量动态调整推理间隔，兼顾动作连贯与及时反应。
  - 接入位置：动作块执行期间的重规划触发，例如行人突然进入视野。
  - 前提与风险：证据来自操作场景，静态与动态真机设定下平均成功率 95%，比最强基线高 55 点。
- **感知工具化：[Robo-Harness K1](https://arxiv.org/abs/2609.29389v1)**
  - 机制：把感知暴露为工具供 agent 查询；工具调用轨迹可直接训练学生模型。Qwen3.5-9B 只用 107 个教师 episode，新初始状态准确率 44.2%（OpenVLA 30.2%），留出任务条件 13.9%（OpenVLA 0.0%）。
  - 接入位置：导航 agent 的感知接口设计；与本期 MCP 导航框架、上期 AnchorVLN 同构。
  - 前提与风险：数字全部来自操作任务。
- **Harness 轨迹蒸馏：[World Action Agent](https://arxiv.org/abs/2609.29964v1)、[HarnessPAI](https://arxiv.org/abs/2609.29166v1)**
  - 机制：用 harness 的成功执行轨迹微调小模型。WAA 在 LIBERO-Pro 上平均成功率 75.6%，用其轨迹微调 Qwen3.5-9B 后分布外成功率从 1.7% 升到 43.3%。HarnessPAI 用收敛程序采集的专家数据，使 π0.5 在 LIBERO-PRO 上再涨 38.8 点。
  - 接入位置：把零样本 VLN harness（如 GPT-6-Astra 类系统）的成功轨迹蒸馏成可部署的小模型。
  - 前提与风险：harness 自身成功率要足够高、轨迹覆盖要足够多样；蒸馏得到的是 harness 的行为，不是更强的空间能力。
- **语言标注效率：[LADA](https://arxiv.org/abs/2609.27747v1)**
  - 机制：先从无标注观测中学潜在动作码本，再把少量语言指令映射到码本上。不到 5% 的语言标注，Bench2Drive 闭环驾驶分 87.98、成功率 70.46%，与全监督基线持平或更高。
  - 接入位置：VLN 指令数据稀缺，可先用无标注导航轨迹学潜在动作码本。
  - 前提与风险：驾驶动作空间与室内导航差异大；Bench2Drive 是闭环仿真。
- **并行假设与取证：[WORLDS](https://arxiv.org/abs/2609.23841v1)（边界条目，空中）**
  - 机制：地理先验初始化的持久图，加上并行推理器保留竞争解释并主动请求证据，审查器处理观测，裁判决定选定目标或再取证一轮。CityNav 全部 5,311 个测试 episode 上 SR 51.8%，比已发表最佳高 15.7 点（OSM-only、高分辨率正射协议）；同模型同预算的 1,000 个共享 episode 上 50.0% 对 27.9%；审查器贡献 5.9 点；有四旋翼真机演示。
  - 接入位置：目标歧义大的 ObjectNav / 实例导航，先取证再定目标，替代「看到就走」。
  - 前提与风险：城市尺度空中场景，依赖地理先验地图；地面室内没有等价的先验源。
- **评测设计：[RoboFollow](https://arxiv.org/abs/2609.25636v1)**
  - 机制：「高场景熵」原则，每个训练场景都支持多条运动学上不同的任务分支，迫使策略依赖语言；四级扰动协议分离理解与执行。
  - 接入位置：自建 VLN 评测时，检查是否存在只有一条合理路线、语言可有可无的 episode。
  - 前提与风险：证据来自操作任务，九个 VLA/WAM 在 L0 的成绩不能可靠迁移到 L1–L3。

## 五、分类速览

每条标注相关度：A 为直接研究地面导航，B 有可迁移机制，C 仅作领域观察。

### 5.1 地面 VLN / ObjectNav / 语义导航

- **GPT-6-Astra Lights Up**（零样本 VLN-CE · A）：见重点分析。[原文](https://arxiv.org/abs/2609.29861v1)
- **How Far Can GPT-6-Astra Go?**（零样本 VLN-CE · A）：上期已分析的 v1 改题后的 v2，不计入本期新增；见重点分析。[原文](https://arxiv.org/abs/2609.20116v2)
- **SparseNav**（免训练 VLN-CE · A）：见重点分析。[原文](https://arxiv.org/abs/2609.26408v1)
- **Talk2Escape**（对话式纠错 VLN · A）：见重点分析。[原文](https://arxiv.org/abs/2609.28296v1)
- **VNT-PA**（PointNav · A）：见重点分析。[原文](https://arxiv.org/abs/2609.21212v1)
- **NaviScale**（语义地图 ObjectNav 数据 · A）：拼接 12,794 处房产的 24,000 个户型与 MP3D / HM3DSem 房间级地图，生成 19.2 万张语义地图训练补全预测器；HM3D SR 64.3% / SPL 34.8%，MP3D SR 43.1% / SPL 16.8%，另有真机部署。[原文](https://arxiv.org/abs/2609.27218v1)
- **物体-路径图**（开放词表实例导航 · A）：物体-路径图统一开放词表语义推理与拓扑导航，节点间用语义视觉伺服执行，无需稠密度量重建；HM3D / Replica 与真机验证，摘要未给数字。[原文](https://arxiv.org/abs/2609.24189v1)
- **DPed-VLN**（动态行人 VLN 基准 · A）：33,093 个 episode、ORCA 控制的人形行人、社会约束专家路径；LoRA 适配后 NaVILA、StreamVLN 在若干成功与安全指标上优于零样本版本，自提 DPet-RL 的 SR / SPL / STL 最高。[原文](https://arxiv.org/abs/2609.21504v1)
- **VLN 可解释性**（可解释性与激活引导 · A）：策略对视觉、指令、视觉记忆均敏感而不依赖单一模态；内部编码导航进度，行为激活向量可零样本迁移到分布外真实场景并提升表现。[原文](https://arxiv.org/abs/2609.24576v1)
- **NaViRrator**（地图到指令 · A）：人类可读地图上的起终点 → 地图坐标系路线骨架 → VLM 生成指令 → 交给预训练 VLN 策略执行；真机 SR / SPL 优于直接生成指令与 A* 骨架等对照，摘要未给数字。[原文](https://arxiv.org/abs/2609.21316v1)
- **Deploying FMs for Embodied Navigation**（基础模型部署 · A）：TAP 用场景中挖掘的人类习惯数据做个性化目标查找（Turtlebot 真机平均 +18%）；MemCtrl 用「记忆头」主动管理上下文，多任务平均 +6%、长指令子集 +20%，上下文约为基线一半。[原文](https://arxiv.org/abs/2609.25666v1)
- **CoRelNav**（多机器人关系型语义导航 · A）：任务条件化多机探索与候选驱动的协同验证耦合，跨拓扑节点聚合观测来判定空间关系；仿真优于基线，两台真机部署，摘要未给数字。[原文](https://arxiv.org/abs/2609.27720v1)
- **"Dear LLaVA, Please Drive"**（VLM 导航微调 · A）：用可微几何代价场代替标注轨迹，仅由双目深度学无碰撞路径；任务专属 LoRA 更新不到 1% 参数，未见环境 SPL「有竞争力」（未给数字）。[原文](https://arxiv.org/abs/2609.22925v1)
- **MCP 导航表征层**（LLM + ROS 导航 · A）：占据栅格转成带位姿的度量图像、航点级观测做语义标注，经 MCP 暴露为标准工具，不改 ROS 导航栈；仿真室内建图覆盖率超过 97%。[原文](https://arxiv.org/abs/2609.27340v1)
- **ACE**（具身探索，边界条目 · A）：证据接地感知与曝光感知移动结合，缓解过早终止与过度继续的矛盾；称导航任务成功率比先前 SOTA 高 18.0%、问答探索效率高 10.3%，摘要未写基准名。[原文](https://arxiv.org/abs/2609.22385v1)
- **ReVNM**（远端相机视觉导航 · B）：单个监控相机同时充当观测源与隐式地图，exo2ego 模块从远端视角预测机器人前方深度；只在随机生成的世界中训练，免微调迁移真机。[原文](https://arxiv.org/abs/2609.28976v1)

### 5.2 具身 Agent、记忆与规划

- **AdaHVLA**（自适应 harness · B）：见重点分析。[原文](https://arxiv.org/abs/2609.29204v1)
- **RegenHarness**（机器人 agent harness · B）：见重点分析第 5 项。[原文](https://arxiv.org/abs/2609.27612v1)
- **HarnessPAI**（物理 AI harness · B）：以代码为可执行、可演化的接口，回合内按程序开环执行，回合间用反馈修订程序并沉淀技能；LIBERO-PRO 比 π0.5 高 61.6 点，RoboCasa 原子任务比 WorldDreamer 高 27.2 点；覆盖扫地机与足式平台。[原文](https://arxiv.org/abs/2609.29166v1)
- **AquaMend**（信念失效恢复 · B）：在探测-信念-动作图上按期望损失在重探测、回滚、继续三者间选择；自建 32 个配对场景恢复 28 个，平均完全损失比重启低 21.6%，与决策论排障的差异经 Holm 校正不显著。[原文](https://arxiv.org/abs/2609.28973v1)
- **RoboFollow**（指令跟随诊断 · B）：见第四节。[原文](https://arxiv.org/abs/2609.25636v1)
- **OmniEcho**（空间音频 + 导航 · B）：OmniEchoBench 含 197 个真实空间音视频场景、2,972 个问答对、30 个真实环境中的 900 条一阶 Ambisonics 导航样本；声源引导导航「接近传统 VLN 水平」，摘要未给数字。[原文](https://arxiv.org/abs/2609.23407v2)
- **主动探索式操作**（Agent 化操作 · B）：规划 / 感知 / 执行三模块加细粒度感知-执行交错，目标初始不可见时先搜索再操作；Find-and-Place 任务验证，摘要未给数字。[原文](https://arxiv.org/abs/2609.29091v1)
- **WORLDS**（城市尺度语言搜索，边界条目 · B）：见第四节。[原文](https://arxiv.org/abs/2609.23841v1)
- **AquaCap**（水下代码即策略 agent · C）：双层 agent 把指令与观测转成条件化计划与可执行控制程序，失败感知记忆支持闭环重规划；仿真成功率 66.43%，ROV 真机抓取与搬运。[原文](https://arxiv.org/abs/2609.23133v1)
- **CE⁴L**（多视角持续学习基准 · C）：ego / exo / ego-exo 四任务持续学习基准，附参数高效的子空间路由适配器基线。[原文](https://arxiv.org/abs/2609.23492v1)
- **PUBG Ally**（游戏内对话式 agent · C）：语言模型 agent 用工具读取游戏信息并驱动更快的控制层；近 3.9 万局真人同玩数据迭代训练；推荐意愿正向回答比负向高 25.1 点。[原文](https://arxiv.org/abs/2609.29837v1)
- **Listening and Mirroring**（VR 共情对话 agent · C）：20 人被试内实验，言语调谐是感知共情最可靠的来源；属 VR 社交 agent，分池时被误归入具身Agent。[原文](https://arxiv.org/abs/2609.27246v1)

### 5.3 社会导航、越野与户外

- **PopNavShift**（社会导航评测 · B）：用 LLM 为 600 个人格记录生成行人运动参数，在 8 种人群条件、7,488 次配对运行下比较三类策略；排名反转比例随指标变化（见结论第 5 条）。[原文](https://arxiv.org/abs/2609.21838v1)
- **扩散引导在线适配**（社会导航 · B）：固定扩散策略、只训练噪声策略（DSRL），在部署环境微调时保住基础性能；多种子扩散 RL 策略集成为基础策略；硬件在环验证。[原文](https://arxiv.org/abs/2609.24317v1)
- **Where Should I Join?**（语言引导的加入人群 · B）：递归谱划分生成候选成员子集，语言条件图像-几何模型排序，再用人群队形先验预测社会合规的加入位姿；亚秒推理，真机验证。[原文](https://arxiv.org/abs/2609.28467v1)
- **AcousticDiffusion**（声源引导救援导航 · B）：麦克风阵列到达方向递归融合成 BEV 置信场，条件化扩散模型生成航点轨迹；四足真机平均方位误差 64.9°（A* 98.2°、RRT 90.4°），最终离呼救者 2.48 m（经典规划 3.96 m）。[原文](https://arxiv.org/abs/2609.21792v1)
- **Verti-WM**（越野世界模型 · C）：冻结 Transformer 处理刚性地形、神经符号地面力学处理可变形地形；预测误差比纯数据 / 纯物理基线低 34.6% / 21.7%，训练计算时间省 23.6 倍；真车成功率 80%（直接 sim-to-real 40%）。[原文](https://arxiv.org/abs/2609.23118v1)
- **TravPro**（越野可通行性排序 · C）：把现有标注转成区域偏好对，在冻结 VLM patch token 的原型上学偏好分数再蒸馏；五个未见域平均成对准确率 0.915（最强基线 0.783）。[原文](https://arxiv.org/abs/2609.23673v1)
- **户外 GNSS 导航栈**（ROS 2 户外导航 · C）：单 GNSS-IMU / 双天线前端可换，多种控制器共用统一状态接口；葡萄园 800 次实地运行，行内混合模式平均横向误差 0.95 cm（单 GNSS+IMU）/ 0.85 cm（双天线）。[原文](https://arxiv.org/abs/2609.28933v1)
- **Planning Trajectories that Bounce**（碰撞容忍规划 · C）：反射增广状态图按所用墙面序列划分路径类，受控撞墙可减少执行时间与控制量（仅仿真）。[原文](https://arxiv.org/abs/2609.27145v1)
- **BarrierFormer**（安全控制 · C）：Transformer 自回归生成预测滚动以替代模型，屏障评论器沿滚动检查 CBF 约束，推理时无需在线优化。[原文](https://arxiv.org/abs/2609.23896v1)
- **ZIL**（图像-点云配准 · C）：零样本非同步图像到 LiDAR 配准基础模型；7 个数据集 140 万帧训练，平移 / 旋转误差最多降低 87% / 76%。[原文](https://arxiv.org/abs/2609.22716v1)

### 5.4 水面、水下与无人机

- **RiverVLN**（无人水面艇 VLN · B）：河道连续运动下的长程 USV VLN 基准；PGT-NAV 把指令转成可视觉验证的语义阶段序列并在线维护当前阶段；Unity-ROS 闭环平均成功率 0.79，真船验证。[原文](https://arxiv.org/abs/2609.23423v1)
- **AquaWorld**（水下世界生成 · C）：共享地形结构的一致性随机化；同预算策略训练验证成功率高 21%，仿真训练的视觉导航策略在实体水槽中成功 95%。[原文](https://arxiv.org/abs/2609.22670v1)
- **PhysAI-Bench**（无人机 agent 决策基准 · C）：10,178 个决策实例，含 MCP 工具调用、A2A 交互与 6G 网络状态；29 个模型中 GPT-5.3 准确率最高，为 52.00%。[原文](https://arxiv.org/abs/2609.23695v1)

### 5.5 次池速览（VLA·操作 / 自动驾驶 / 其他）

次池共 84 篇（VLA·操作 81 篇，另 1 篇 AdaHVLA 因有导航侧数字移入 5.2；「其他」3 篇），不做深度分析。有迁移价值的已列入第四节。按主题归组如下：

- **记忆与长程执行（12）**：[SmoLSTM](https://arxiv.org/abs/2609.22854v1)、[StateMem](https://arxiv.org/abs/2609.22684v1)、[MemBodied](https://arxiv.org/abs/2609.28256v1)、[TaskAnchor](https://arxiv.org/abs/2609.23580v1)、[CommitFlow](https://arxiv.org/abs/2609.21908v1)、[CARE](https://arxiv.org/abs/2609.24118v1)、[H-VLA](https://arxiv.org/abs/2609.22895v1)、[X-Planner](https://arxiv.org/abs/2609.25187v1)、[TANDEM](https://arxiv.org/abs/2609.28314v1)、[LiMA](https://arxiv.org/abs/2609.28431v1)、[ActiveArena](https://arxiv.org/abs/2609.24124v2)、[SafeLoop](https://arxiv.org/abs/2609.26313v1)
- **Harness 与 agent 化执行（3）**：[Robo-Harness K1](https://arxiv.org/abs/2609.29389v1)、[World Action Agent](https://arxiv.org/abs/2609.29964v1)、[AR-WAM](https://arxiv.org/abs/2609.23578v1)
- **3D、几何与视觉表征（12）**：[Grounded Action Model](https://arxiv.org/abs/2609.23863v1)、[Bridge3D](https://arxiv.org/abs/2609.24525v1)、[GALA](https://arxiv.org/abs/2609.21948v1)、[FOCAL-VLA](https://arxiv.org/abs/2609.21228v1)、[HABILIS Brain 0](https://arxiv.org/abs/2609.25558v1)、[末端几何跨策略分析](https://arxiv.org/abs/2609.21659v2)、[拓扑视觉提示](https://arxiv.org/abs/2609.23944v1)、[3D 视觉语言对齐-融合](https://arxiv.org/abs/2609.28222v1)、[InfiNoVA](https://arxiv.org/abs/2609.27734v1)、[ActGaze](https://arxiv.org/abs/2609.28955v1)、[MaskVLA](https://arxiv.org/abs/2609.23565v1)、[PSR](https://arxiv.org/abs/2609.21753v1)
- **世界模型（5）**：[AffordanceWAM](https://arxiv.org/abs/2609.22332v2)、[Think Like a World Model](https://arxiv.org/abs/2609.24682v2)、[Imagine-RL](https://arxiv.org/abs/2609.24033v1)、[Prioritized Rollouts](https://arxiv.org/abs/2609.22879v1)、[MachEmbodied-U0](https://arxiv.org/abs/2609.25627v1)
- **RL 后训练与在线适配（13）**：[SynthDemo-RL](https://arxiv.org/abs/2609.21650v1)、[异步回放锚定在线后训练](https://arxiv.org/abs/2609.22888v1)、[BEE](https://arxiv.org/abs/2609.27450v1)、[RouteRLT](https://arxiv.org/abs/2609.26467v1)、[优势引导后训练剖析](https://arxiv.org/abs/2609.28161v1)、[不确定性门控探索噪声](https://arxiv.org/abs/2609.28838v1)、[ForceRFT](https://arxiv.org/abs/2609.22840v1)、[FAN](https://arxiv.org/abs/2609.21358v1)、[Self-Adaptive VLA](https://arxiv.org/abs/2609.30092v1)、[任务语义动作校准](https://arxiv.org/abs/2609.23650v1)、[BNN 全协方差平滑](https://arxiv.org/abs/2609.27244v1)、[类脑分层模块化持续学习](https://arxiv.org/abs/2609.25146v1)、[VLA 联邦微调测试床](https://arxiv.org/abs/2609.22973v1)
- **动作表征与推理效率（11）**：[KerColle](https://arxiv.org/abs/2609.22335v1)、[Catch Me If You Can](https://arxiv.org/abs/2609.21022v1)、[Fewer Steps, Better Actions](https://arxiv.org/abs/2609.21216v1)、[React When You Need To](https://arxiv.org/abs/2609.22587v1)、[FoldQuantVLA](https://arxiv.org/abs/2609.24433v1)、[VLAQuantBench](https://arxiv.org/abs/2609.25376v1)、[Decoupled Early Exits](https://arxiv.org/abs/2609.29382v1)、[CereVLA](https://arxiv.org/abs/2609.27468v1)、[动作块 VLA sim-to-real 流水线](https://arxiv.org/abs/2609.21817v1)、[动作分词：超越重建误差](https://arxiv.org/abs/2609.25820v1)、[方向-尺度分解动作表征](https://arxiv.org/abs/2609.28865v1)
- **评测与安全（9）**：[LIBERO-VPro](https://arxiv.org/abs/2609.24350v1)、[IndustrialVLA-Bench](https://arxiv.org/abs/2609.25562v1)、[VLA-Scope](https://arxiv.org/abs/2609.21246v1)、[SafeStage](https://arxiv.org/abs/2609.21223v1)、[VLPSA](https://arxiv.org/abs/2609.22462v1)、[噪声空间实时策略引导](https://arxiv.org/abs/2609.21220v1)、[CrossSafe](https://arxiv.org/abs/2609.28984v1)、[工业机械臂后门](https://arxiv.org/abs/2609.26868v1)、[ReVeal](https://arxiv.org/abs/2609.23910v1)
- **接触、力触觉与专用场景（13）**：[ForeTac-VLA](https://arxiv.org/abs/2609.20980v1)、[CompVLA](https://arxiv.org/abs/2609.23614v1)、[Opt2VLA](https://arxiv.org/abs/2609.23968v1)、[VisForce](https://arxiv.org/abs/2609.25785v1)、[VT-Bridge](https://arxiv.org/abs/2609.22606v1)、[HEARTH](https://arxiv.org/abs/2609.23418v1)、[CableVLA](https://arxiv.org/abs/2609.25606v1)、[Imperfection for Precision](https://arxiv.org/abs/2609.26672v1)、[SCULPT-VLA](https://arxiv.org/abs/2609.23275v1)、[能力感知共享控制](https://arxiv.org/abs/2609.25369v1)、[MATE](https://arxiv.org/abs/2609.26520v1)、[MedVLA](https://arxiv.org/abs/2609.25756v1)、[StenoVLA-3D（胃肠道内镜）](https://arxiv.org/abs/2609.24187v2)
- **自动驾驶（4）**：[Beyond the Leaderboard](https://arxiv.org/abs/2609.22582v1)、[PRIME](https://arxiv.org/abs/2609.22040v1)、[ZYT-World](https://arxiv.org/abs/2609.21712v2)、[LADA](https://arxiv.org/abs/2609.27747v1)
- **预训练数据与通用模型（2）**：[AtomEgo](https://arxiv.org/abs/2609.21461v1)、[ME-VLM](https://arxiv.org/abs/2609.24526v2)

### 5.6 资讯与非论文

本期无。公众号源入库 0 条。

## 六、趋势判断与行动建议

### 趋势

- **Harness 正在变成数据引擎。** 上期的 harness 工作只报系统整体指标；本期有三项工作（HarnessPAI、World Action Agent、Robo-Harness K1）把 harness 执行轨迹蒸馏回小模型，并都报告了显著提升。本报告判断：导航侧很可能出现同样的路线——先用闭源前沿模型加 harness 做高成功率的零样本系统，再把轨迹蒸馏成可部署的模型。GPT-6-Astra 的 79.0% 如果得到核实，就是这条路线的上游。
- **记忆设计从「存多少」转向「状态多大」。** 导航侧的 VNT-PA、SparseNav、MemCtrl，与 VLA 侧的 SmoLSTM、MemBodied、StateMem，都在用固定或有界的表征替代增长的上下文。SmoLSTM 重置状态后从 77.5% 跌到 7.0%，MemCtrl 用一半上下文反而提升，两组证据共同说明：瓶颈不在上下文长度，在于记什么。
- **零样本 VLN-CE 的比较维度从「方法」扩展到「信息条件」。** 是否使用外部纠正（Talk2Escape）、用哪个基础模型和推理档位（GPT-6-Astra）、是否免训练自主（SparseNav），已经比方法细节更能解释成绩差异。上期的问题是 episode 子集不统一，本期又叠加了信息条件不统一。
- **非视觉信号进入导航。** 本期 OmniEcho（空间音频导航样本）、AcousticDiffusion（麦克风阵列引导四足）、ReVNM（远端监控相机）都让导航依赖机载视觉以外的信号源。三者都还没有与标准 VLN 基准对接。

### 研究空白

- **零样本 R2R-CE 缺少信息条件的登记规范。** 同一基准上已经并存 episode 子集、外部纠正、基础模型与推理档位三类未对齐的变量，没有一篇论文同时报告这三项。
- **停止与完成判定仍没有专门指标。** GPT-6-Astra 作者承认 ultra 推理下目标验证仍不可靠，RegenHarness 主张完成取决于执行历史而非终点距离，上期旧文有 16.0% 的触底成功。三处都指向同一环节，但本期仍无人单独报告停止相关的指标。
- **人在回路的成本不透明。** Talk2Escape 没有报告提问频次，也没有给出「SR 随干预次数变化」的曲线，因此无法判断 66.0% 需要多少人工。

### 建议动作

**高优先级**

- **精读并核验** GPT-6-Astra Lights Up：79.0% 对应的 episode 集合、各推理档位的 SR 与成本、成功是否要求主动 STOP；核验前不进入横向对比。
- **精读并复现** SparseNav：复现指令条件化的感知触发，测感知调用次数与延迟的下降。
- **借鉴** Talk2Escape 的打转/偏离检测器：不接人工，改为触发自身重规划或回溯，测净收益。
- **更新跟踪表**：R2R-CE 登记表在「评测子集 + episode 数 + 是否零样本」之外，新增「是否用外部纠正」「基础模型与推理档位」「成功是否需主动 STOP」三项。

**中优先级**

- **精读** VNT-PA：关键帧上下文的构造方式，评估能否替代自有系统中的拓扑记忆。
- **核实** AdaHVLA：NaVILA-LH 的任务定义与 57.5% 对应的配置。
- **最小验证**：在 VLN 历史编码处试固定尺寸循环状态（SmoLSTM / MemBodied 思路），与滑窗上下文对照。
- **跟踪** DPed-VLN 基准释出情况，以及 NaVILA / StreamVLN 在动态行人下的 LoRA 适配结果。

**低优先级**

- **跟踪** NaviScale 数据集是否公开。
- **暂缓**次池 VLA 量化与推理加速类工作：机制可借鉴，但均未在 VLN-CE 上验证。

> **更新（2026-10-01）**：GPT-6-Astra Lights Up 已于 2026-09-25 发布 v2，本报告所引 79.0% / 76.0% 为 v1 的单次运行结果；v2 改为每档三次运行取均值，ultra 为 81.3±2.5%（SPL 71.5±1.7），medium 为 75.7±1.5%（SPL 65.6±2.1），仍只在 R2R-CE-100 上评测。详见 [VLN-Papers 的 GPT-6-Astra 条目](/VLN-Papers/#gpt-6-astra)。
