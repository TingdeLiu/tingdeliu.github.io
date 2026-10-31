---
layout: post
lang: zh-CN
translation_id: vln-weekly-2026-10-31
title: "具身导航周报（2026-10-21 ~ 2026-10-29）"
date: 2026-10-31
permalink: /vln-weekly-2026-10-31/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期新增 102 项独立工作，VLN 与具身 Agentic 导航各 9 篇。MOON 报告 R2R-CE / RxR-CE SR 提升 5.2 / 6.0 个百分点；EvoLife 将主动求助与持久地点记忆接入导航闭环，精读同时核对其引导条件。"
---

> **本期数字**：新增 102 篇 / 102 项独立工作 · 主线：VLN 9 篇 / 具身 Agentic 导航 9 篇 · 其他方向 84 篇 · 公众号 0 条 · 另更新已有论文 1 篇

## 一、本期要点
{: id="key-points"}

- **记忆维护开始覆盖“保留、清空、反思”三个环节。** [时空记忆](#spatiotemporal-memory)组织历史证据，[MOON](#moon)处理偏航后的历史惯性，[RetroNav](#retronav)把失败解释连回场景记录。
- **规划与执行的接口进入训练目标。** [MATCH](#match)联合优化航点请求和原子动作执行；同批出现的语义安全评测也把“识别风险”和“改变运动”分开检查。
- **主动求助进入终身导航框架。** [EvoLife](#evolife)将交互、动作工具和地点记忆接在一起，其结果同时依赖引导条件；首访、重访和连续任务需要分别阅读。

## 二、精读
{: id="deep-reads"}

### 1. MOON
{: id="moon"}

**偏航恢复需要判断何时清空历史。**

**[论文](https://arxiv.org/abs/2610.33546v1)** · 主线：VLN · 指令跟随 · R2R-CE / RxR-CE（连续环境）

**问题。** 历史既帮助跟随路线，也可能在走错后维持原来的动作惯性。

**方法。** 冻结基础策略，用偏航判别器触发历史清空，再由经 GRPO 训练的恢复策略接管；新增参数不足基础策略的 1%。

<div align="center">
<img src="/images/vln/MOON-overview.webp" alt="MOON 架构图" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1577/871" />
<figcaption>判别器报警后清空历史，恢复策略接管；下半部为恢复训练流程。（<a href="https://arxiv.org/html/2610.33546v1#S3.F4">论文原图</a>）</figcaption>
</div>

**证据。** 摘要报告 R2R-CE、RxR-CE 的 SR 相对冻结基础策略分别提高 5.2、6.0 个百分点；随机触发消融支持清空时机的重要性。

**判断。** 适合拆分记忆错误与动作错误。清空会损害仍在正确路线上的回合，应同时检查误报警与恢复收益；摘要未给真机结果。

**下一步。** 复现：在同一偏航状态比较保留历史、定时清空、判别器触发与恢复策略。

### 2. MATCH
{: id="match"}

**规划目标应随执行器能力共同调整。**

**[论文](https://arxiv.org/abs/2610.38248v1)** · 主线：VLN · 分层规划与执行 · R2R-CE / RxR-CE（连续环境）

**问题。** 只训练规划器、固定执行器，会持续提出实际难以完成的航点请求。

**方法。** 执行感知拓扑图结合价值估计与结果记忆，允许连续微调目标；奖励平衡按序路线进展和执行残差，两级策略同步更新。

<div align="center">
<img src="/images/vln/MATCH-overview.webp" alt="MATCH 架构图" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/520" />
<figcaption>规划器提出执行请求，执行器输出原子动作，同批轨迹更新两级策略。（<a href="https://arxiv.org/html/2610.38248v1#S2.F2">论文原图</a>）</figcaption>
</div>

**证据。** 摘要无数值，只称 0.5B 模型在 R2R-CE / RxR-CE 取得领先结果；原文真机展示仅 4 个回合。

**判断。** 适合研究航点是否可执行，同时保留指令约束。4 个真机回合不足以说明部署稳定性；局部到达率也不能替代整条指令完成。

**下一步。** 精读：核对碰撞设置，分离目标偏移、联合训练与路线奖励的作用。

### 3. EvoLife
{: id="evolife"}

**把主动求助与位置记忆接进导航闭环。**

**[论文](https://arxiv.org/abs/2610.39446v1)** · 主线：具身 Agentic 导航 · 交互导航 / ObjectNav / 连续多任务

**问题。** 移动、澄清意图和保存经验分离时，一次交互所得的信息很难在后续任务复用。

**方法。** 统一运行框架提供航点、离散、像素与物体导航工具，将问答及观测绑定到地点，供后续访问检索与修订。

<div align="center">
<img src="/images/vln/EvoLife-overview.webp" alt="EvoLife 架构图" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1977/1350" />
<figcaption>混合动作、主动交互与持久空间记忆共享同一个决策循环。（<a href="https://arxiv.org/html/2610.39446v1#S3.F2">论文原图</a>）</figcaption>
</div>

**证据。** 摘要报告四组各 100 回合的 VLN / ObjectNav 首访 SR 为 100%，平均每回合主动交互 1.87 次；各 50 序列的 LHPR-VLN、GOAT-Bench 完整序列 SR 为 98%、94%。

**判断。** 这是带求助的系统结果。原文仿真引导端可见特权路线与目标信息，不能与无帮助 VLN 等同；仍需评估现实求助成本及错误回答的影响。

**下一步。** 精读：固定导航器与交互预算，对读无引导、受限引导和完整引导。

### 4. RetroNav
{: id="retronav"}

**失败反思需要连回可核验的场景证据。**

**[论文](https://arxiv.org/abs/2610.35172v1)** · 主线：具身 Agentic 导航 · 终身导航 / 多次尝试

**问题。** 只保存场景知识，难以解释历史决策为何失败，以及重试时应改变什么。

**方法。** 统一记忆关联几何、语义与导航经历，并区分客观观测和任务解释；独立评价器决定重试，反思 Agent 生成有证据支持的重试简报。

<div align="center">
<img src="/images/vln/RetroNav-overview.webp" alt="RetroNav 架构图" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1054" />
<figcaption>导航积累场景和行为记忆，评价与反思模块据此生成下一次尝试的简报。（<a href="https://arxiv.org/html/2610.35172v1#S1.F1">论文原图</a>）</figcaption>
</div>

**证据。** 摘要无数值，只称在五个导航基准取得领先表现。

**判断。** 适合检查反思是否引用真实证据。多次尝试增加了信息与预算，应同时统计首试成功、重试恢复和累计路径及调用成本。

**下一步。** 复现：相同重试预算下比较盲目重试、只读场景记忆和证据化简报。

### 5. 时空记忆
{: id="spatiotemporal-memory"}

**记忆要同时支持任务进度、历史取证和几何执行。**

**[论文](https://arxiv.org/abs/2610.38318v1)** · 主线：具身 Agentic 导航 · 零样本 VLN · R2R-CE val-unseen（连续环境）

**问题。** 保留历史不代表推理器能读懂它；仅凭当前视图也容易重复探索。

**方法。** 任务记忆保存进度，时间记忆压缩带位姿的观测，空间记忆累积几何；8B 观察器与 8B 推理器协作，选择航点或请求补充视角，再由几何控制器执行。

<div align="center">
<img src="/images/vln/SpatiotemporalMemory-overview.webp" alt="SpatiotemporalMemory 架构图" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1259" />
<figcaption>三种记忆支持航点选择与 PREVIEW 补充观察，执行后的观测继续更新记忆。（<a href="https://arxiv.org/html/2610.38318v1#S2.F1">论文原图</a>）</figcaption>
</div>

**证据。** 摘要报告 R2R-CE val-unseen 消融中，去除时间记忆使平均路径由 16.9 m 增至 39.1 m；未给 SR 数值。

**判断。** 路径效率值得跟踪，但不能把整个系统压成单个“8B 模型”。原文真机依赖远程 A100 与预建图定位，尚不能说明纯机载、无先验地图部署能力。

**下一步。** 精读：对齐定位和动作接口，再比较路径、SR、词元及补充观察成本。

## 三、也值得一看
{: id="also-worth"}

- **[RepresentNav](https://arxiv.org/abs/2610.31762v1)** · VLN：自建视觉表征与潜空间预测；摘要报告 VLNVerse 细粒度任务 SR 83.80%。
- **[One Scan is Enough](https://arxiv.org/abs/2610.33794v1)** · VLN：单次扫描合成训练环境；摘要报告五个真机场景 SR 85%，同楼未扫描区域 82%。
- **[MEMVLN](https://arxiv.org/abs/2610.35066v2)** · VLN：有界情节记忆与路线重建训练；真机仅六条留出指令，需核对重复试验口径。
- **[ASPEN](https://arxiv.org/abs/2610.36285v1)** · VLN：事件触发语义推理；RxR-CE val-unseen 的模型调用减少 9.5%，估计推理时间减少 8.1%。
- **[Natural-Language to Nav2](https://arxiv.org/abs/2610.32633v1)** · Agentic 导航：语言编译为 JSON 策略，经确定性检查后投影到 Nav2 代价地图。
- **[PIFR](https://arxiv.org/abs/2610.35576v1)** · Agentic 导航：PIFR 分开评估插入任务完成、原任务恢复与整段序列完成。
- **[HGWM](https://arxiv.org/abs/2610.32554v3)** · Agentic 导航：房间—物体先验与持久图记忆配合图匹配；HM3D v0.1 SR 62.6%、SPL 31.5%。
- **[ThinkLocoManip](https://arxiv.org/abs/2610.34370v1)** · Agentic 导航：足式移动操作框架编排导航、运动与操作工具，以关键帧保留恢复上下文。
- **[VISTA](https://arxiv.org/abs/2610.35051v1)** · 可迁移机制：主动生成新回合验证框架修改，可借鉴到导航反思后的再测试。
- **[LOGIV](https://arxiv.org/abs/2610.40016v1)** · 可迁移机制：持续核验任务依赖并局部修复，可借鉴到导航子目标的前置条件维护。

## 四、趋势与行动
{: id="trends-and-actions"}

**趋势**

- **记忆的收益取决于使用时机与证据组织。** [MOON](#moon)、[RetroNav](#retronav)与[时空记忆](#spatiotemporal-memory)分别覆盖清空、反思和检索，适合在共同预算下对读，而非简单比较缓存长度。
- **导航框架需要为执行、求助和恢复分别记账。** [MATCH](#match)约束执行请求，[EvoLife](#evolife)引入外部指导，PIFR 处理运行时中断；最终到达率不能独自解释各环节的贡献。

**研究空白**

- 缺少同时固定动作接口、历史预算、模型调用和重试次数的导航恢复评测，难以分离策略变强与系统获得更多机会。
- 带特权信息的引导、远程推理及预建图定位，与普通地面机器人部署条件仍有距离，需要逐项移除后检验能力边界。

**下一步**

- **高 · 复现** [MOON](#moon)：建立同状态偏航干预集，同时记录误报警、恢复 SR 与累计路径。
- **高 · 精读** [EvoLife](#evolife)：列清引导端可见信息，复核无引导与受限引导条件，再判断哪些收益可迁移。
- **中 · 对照** [MATCH](#match)：固定执行器和接触设置，比较只调目标、只训练执行器与联合训练。
- **中 · 原型** Natural-Language to Nav2：从禁行区与事件通行策略做起，分别统计解析错误、错误接受和执行偏差。
- **低 · 跟踪** [RetroNav](#retronav)与[时空记忆](#spatiotemporal-memory)：等预算测首试、重试及词元成本，暂缓仅凭最终 SR 排序。

## 五、全部工作索引
{: id="full-index"}

以下覆盖全部 102 项新增独立工作；A 为两条主线，B 为可迁移机制，C 为领域观察，精读工作链接回对应卡片。

<details markdown="1">
<summary>VLN 与语义导航 · A · 9 篇</summary>

- [RepresentNav](https://arxiv.org/abs/2610.31762v1)：策略自建视觉表征并预测未来潜状态。
- [MOON](#moon)：见精读。
- [One Scan is Enough](https://arxiv.org/abs/2610.33794v1)：扫描重建场景，自动合成指令与训练任务。
- [Stage-Aware Attribution](https://arxiv.org/abs/2610.34028v1)：按指令阶段汇聚当前及历史视觉归因。
- [ScheNav](https://arxiv.org/abs/2610.34286v1)：粗略示意图与观测融合为探索价值场。
- [ASPEN](https://arxiv.org/abs/2610.36285v1)：预测命令持续性，按需重新调用语义模型。
- [OSMA-Bench](https://arxiv.org/abs/2610.34674v1)：按障碍语义检验间距、速度和运动平滑性。
- [MEMVLN](https://arxiv.org/abs/2610.35066v2)：有界情节记忆配合掩码路线重建。
- [MATCH](#match)：见精读。

</details>

<details markdown="1">
<summary>具身 Agentic 导航 · A · 9 篇</summary>

- [HGWM](https://arxiv.org/abs/2610.32554v3)：房间—物体先验与持久图记忆引导找物。
- [Natural-Language to Nav2](https://arxiv.org/abs/2610.32633v1)：确定性验证语言策略，再接入 Nav2 代价地图。
- [ThinkLocoManip](https://arxiv.org/abs/2610.34370v1)：编排移动、导航与操作工具，关键帧支持恢复。
- [OmniBrain-0](https://arxiv.org/abs/2610.36013v1)：蒸馏导航与操作专家，统一高层决策模型。
- [PIFR](https://arxiv.org/abs/2610.35576v1)：分开评测任务抢占、插入执行与恢复。
- [RetroNav](#retronav)：见精读。
- [Spatiotemporal Memory](#spatiotemporal-memory)：见精读。
- [EvoLife](#evolife)：见精读。
- [LLM-Guided Loco-Manipulation](https://arxiv.org/abs/2610.40248v1)：按技能、对象与几何目标编排交互导航。

</details>

<details markdown="1">
<summary>可迁移机制 · B · 19 篇</summary>

- [ASTEROID](https://arxiv.org/abs/2610.30105v1)：用学生自身历史训练探索，可借鉴部分可观测决策。
- [FP32 Master Copy](https://arxiv.org/abs/2610.31859v1)：审计低精度更新丢失，可借鉴导航策略微调检查。
- [Horizon–Depth Coupling](https://arxiv.org/abs/2610.32102v1)：联合考察求解深度与开环执行长度。
- [SceneBloom](https://arxiv.org/abs/2610.32978v1)：生成物理有效室内布局，支持 ObjectNav 训练。
- [TrajMamba](https://arxiv.org/abs/2610.33916v1)：统一训练与部署的人体轨迹预处理链。
- [Interaction-Aware GNN](https://arxiv.org/abs/2610.34060v1)：图网络修正航点，利用行人及目标上下文。
- [LegoOcc++](https://arxiv.org/abs/2610.34093v1)：开放词汇占据保持跨视角实例语义一致。
- [RoboDuo](https://arxiv.org/abs/2610.34108v1)：检查整任务完成后调用原子技能恢复。
- [FISRF](https://arxiv.org/abs/2610.35686v2)：按能力选择反思与导师指导策略。
- [EBench Agent Profiling](https://arxiv.org/abs/2610.35832v1)：比较直接控制与调用策略后的框架协同。
- [RobotAgentBench](https://arxiv.org/abs/2610.34649v1)：以任务发现、执行与恢复衡量机器人自主性。
- [The Trinity](https://arxiv.org/abs/2610.36077v1)：能力树组织可验证环境与框架迭代。
- [TP-JSCC](https://arxiv.org/abs/2610.35489v1)：压缩并传输 3D 问答视觉词元。
- [VISTA](https://arxiv.org/abs/2610.35051v1)：主动探索验证修改，保留框架版本谱系。
- [LSMM](https://arxiv.org/abs/2610.37203v1)：位置索引矩阵记忆支持足式迷宫探索。
- [AtomTree](https://arxiv.org/abs/2610.38744v1)：按当前符号状态分解子目标并分层回溯。
- [Object Motion Anticipation](https://arxiv.org/abs/2610.38993v1)：在线适配新物体的运动启动预测。
- [EmbodiExp](https://arxiv.org/abs/2610.39180v1)：用适用条件与可核验结果组合子目标经验。
- [LOGIV](https://arxiv.org/abs/2610.40016v1)：持续核验任务依赖并约束局部修复。

</details>

<details markdown="1">
<summary>VLA / 操作 · C · 43 篇</summary>

- **示范、数据、技能与仿真（16）**：[EgoForesight](https://arxiv.org/abs/2610.31845v1)、[MAST-VLA](https://arxiv.org/abs/2610.32312v1)、[Learned Speed Prior](https://arxiv.org/abs/2610.32676v1)、[EXE](https://arxiv.org/abs/2610.33685v1)、[Speech Conditioning](https://arxiv.org/abs/2610.33866v1)、[Dexterous Manipulation Survey](https://arxiv.org/abs/2610.33486v1)、[TAD](https://arxiv.org/abs/2610.33650v1)、[HuGe](https://arxiv.org/abs/2610.34186v1)、[RynnVLA-Latent](https://arxiv.org/abs/2610.34356v1)、[Robot-Aligned 3DGS](https://arxiv.org/abs/2610.35626v1)、[Self-Labeled Preference Transfer](https://arxiv.org/abs/2610.38000v1)、[Tokens of Motion](https://arxiv.org/abs/2610.37330v1)、[BURGER](https://arxiv.org/abs/2610.40532v1)、[Gym-Anything for Robotics](https://arxiv.org/abs/2610.40565v1)、[DEED](https://arxiv.org/abs/2610.40430v1)、[TouchBridge](https://arxiv.org/abs/2610.40349v1)
- **记忆、世界表征与动作生成（11）**：[CogMemory](https://arxiv.org/abs/2610.34251v1)、[PDRVLA](https://arxiv.org/abs/2610.37381v1)、[Retrieve to Act](https://arxiv.org/abs/2610.37215v1)、[ThinkWorld](https://arxiv.org/abs/2610.37666v1)、[JAM](https://arxiv.org/abs/2610.37219v1)、[World Dreamer v5-lite](https://arxiv.org/abs/2610.37518v1)、[SaiVLA-JEPA](https://arxiv.org/abs/2610.38508v1)、[CoRAct](https://arxiv.org/abs/2610.39543v1)、[HiFOR](https://arxiv.org/abs/2610.40373v1)、[PRIME W0](https://arxiv.org/abs/2610.39528v1)、[Smooth Action Chunking](https://arxiv.org/abs/2610.39451v1)
- **量化、推理与机载硬件（6）**：[AGQ](https://arxiv.org/abs/2610.33074v1)、[PoT-LIF](https://arxiv.org/abs/2610.33658v1)、[TRIAGE](https://arxiv.org/abs/2610.34519v1)、[JevVLA](https://arxiv.org/abs/2610.35010v1)、[VLA-HWBench](https://arxiv.org/abs/2610.34717v2)、[QbD-VLA](https://arxiv.org/abs/2610.38661v1)
- **鲁棒性、安全与评测（9）**：[VLA Failure Features](https://arxiv.org/abs/2610.32208v1)、[Backtrack](https://arxiv.org/abs/2610.32941v1)、[RobustRLlib](https://arxiv.org/abs/2610.32971v1)、[Federated VLA Safety](https://arxiv.org/abs/2610.32888v1)、[MART-GRPO](https://arxiv.org/abs/2610.32431v1)、[Stimulus–Response Safety](https://arxiv.org/abs/2610.34837v1)、[VLA-Stem](https://arxiv.org/abs/2610.37378v1)、[BEHAVIOR-1K Inference Settings](https://arxiv.org/abs/2610.37917v1)、[RoboSeed](https://arxiv.org/abs/2610.40026v1)
- **移动操作实践（1）**：[Tabletop to Walking Bot](https://arxiv.org/abs/2610.32271v1)

</details>

<details markdown="1">
<summary>飞行、控制与其他 · C · 22 篇</summary>

- **飞行与水下（6）**：[AirPlayground](https://arxiv.org/abs/2610.29699v1)、[SP-Guide](https://arxiv.org/abs/2610.32697v1)、[Quadrotor Trajectory Control](https://arxiv.org/abs/2610.34456v1)、[Omni-IGN](https://arxiv.org/abs/2610.36968v1)、[Embodied at Sea](https://arxiv.org/abs/2610.38199v1)、[OpenFlyV2](https://arxiv.org/abs/2610.39287v1)
- **几何、控制、定位与驾驶（6）**：[Continuous-Time Safe Planning](https://arxiv.org/abs/2610.35670v1)、[LECO](https://arxiv.org/abs/2610.34929v1)、[SG²SLAM](https://arxiv.org/abs/2610.36312v1)、[Geomagnetic Gradient Flow](https://arxiv.org/abs/2610.36812v1)、[Value-Guided MPPI](https://arxiv.org/abs/2610.38731v1)、[CoC-CFG](https://arxiv.org/abs/2610.40002v1)
- **实验室、通用 Agent 与其他（10）**：[VERID](https://arxiv.org/abs/2610.32755v1)、[Whale You Go Green?](https://arxiv.org/abs/2610.33191v1)、[Patchwork](https://arxiv.org/abs/2610.33582v1)、[EMMA](https://arxiv.org/abs/2610.35966v1)、[Intrinsic Goals / Drosophila](https://arxiv.org/abs/2610.34638v1)、[Language to Drives](https://arxiv.org/abs/2610.35629v1)、[LongChang](https://arxiv.org/abs/2610.37649v1)、[Auditable Self-Driving Labs](https://arxiv.org/abs/2610.36761v1)、[MCP-Net](https://arxiv.org/abs/2610.37127v1)、[Live iOS Navigation](https://arxiv.org/abs/2610.39264v1)

</details>
