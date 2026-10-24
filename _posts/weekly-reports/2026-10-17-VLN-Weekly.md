---
layout: post
lang: zh-CN
translation_id: vln-weekly-2026-10-17
title: "具身导航周报（2026-10-08 ~ 2026-10-15）"
date: 2026-10-24
permalink: /vln-weekly-2026-10-17/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期新增 175 篇独立工作，自动主池 64 篇，5 篇涉及 R2R-CE。RSI-Nav 报告 SR 68.6%；VLN-Safe 揭示成功轨迹中的碰撞，I2O2I-Nav 的完整任务成功率仍仅 4.02%。重点关注证据验证、阶段交接与失败驱动改进。"
---

## 一、本期结论
{: id="key-conclusions"}

- **连续 VLN 的提升需要与碰撞行为一起看。** 本期 175 篇新增工作中，5 篇明确涉及 R2R-CE，其中 4 篇摘要给出相关数值，VLN-AIEP 仅列出评测基准。[RSI-Nav](https://arxiv.org/abs/2610.15782v1) 报告 R2R-CE SR 68.6%；[VLN-Safe](https://arxiv.org/abs/2610.14682v1) 则发现，允许碰撞后滑动时，38.5–56.8% 的成功轨迹包含接触。本报告判断：成功率与安全性必须分别报告，连续 R2R-CE 也不能与离散 R2R 混比。
- **记忆的关键是维护可信状态，并控制何时执行。** [MemoryVLN](https://arxiv.org/abs/2610.18478v1) 研究空间锚定和跨上下文复用，[S3-Nav](https://arxiv.org/abs/2610.19431v1) 分离持久状态、决策可见证据和执行权限，[AVALON](https://arxiv.org/abs/2610.21586v1) 区分人的通行许可与机器人观测到的可通行性。本报告判断：值得验证的是过期证据如何撤销、停止条件如何复核，而不只是记忆容量。
- **长任务的瓶颈出现在阶段交接。** [I2O2I-Nav](https://arxiv.org/abs/2610.18307v1) 的完整任务成功率从 1.15% 提高到 4.02%，仍远低于单阶段表现；[MultINav](https://arxiv.org/abs/2610.17089v1) 把开门、开容器引起的连通性与可见性变化纳入导航。本报告判断：用上一阶段的实际终点启动下一阶段，才能暴露被独立子任务评测掩盖的困难。
- **自我改进开始针对具体失败，而非统一追加数据。** [WEDGE](https://arxiv.org/abs/2610.19420v1) 根据策略弱点生成环境，[EnactOpt](https://arxiv.org/abs/2610.14977v1) 联合修改上下文构造与文字技能，[AdaRSI](https://arxiv.org/abs/2610.20474v1) 先排查调用方式，再微调持续失败的子步骤。本报告判断：导航研究可先比较“修证据、修调度、修策略”三种干预的成本与收益；操作任务上的收益不能直接当成导航收益。

本期新增 175 条，对应 175 篇独立 arXiv 工作，发表日期覆盖 2026-10-08 ~ 2026-10-15；另更新 5 篇既有论文版本，不计入新增。公众号新增 0 条。自动主池为导航 45 篇、具身 Agent 19 篇，共 64 篇；次池为 VLA·操作 90 篇、自动驾驶 7 篇、其他 14 篇，共 111 篇。自动标签存在边界错误，例如 StereoEngine 被标为自动驾驶，Jev-Guard 进入具身 Agent，而 ASD 与 VP-EVT 落在“其他”。深度分析仅选自主池中的直接相关工作；分类速览按实际内容整理，覆盖全部 175 篇。以下数值来自归档摘要，未提供划分、对照或绝对值的地方不补全。

## 二、优先阅读清单
{: id="priority-reading"}

1. **[VLN-Safe](https://arxiv.org/abs/2610.14682v1)** · 连续 R2R-CE 碰撞评测
   - 贡献：对照 sliding 与 blocking 两种碰撞处理，并加入独立碰撞保护器。
   - 证据：切换为 blocking 后，6 个已发表智能体的 SR 下降 11.7–36.1 个百分点。
   - 理由：直接影响既有导航结果的解释和真机迁移判断。
2. **[GuidedNav](https://arxiv.org/abs/2610.13122v1)** · 连续 R2R-CE / RxR-CE 表征监督
   - 贡献：训练时监督当前子任务、地标注意力与空间表示。
   - 证据：摘要在两套 val-unseen 上分别写 SR 提升 8.1% / 13.1%，未给绝对 SR；不擅自换算为百分点。
   - 理由：关注如何增强指令与动作的联系，同时减少在线推理负担。
3. **[RSI-Nav](https://arxiv.org/abs/2610.15782v1)** · 连续导航的语言—动作数据闭环
   - 贡献：分批生成指令、执行筛选并立即更新同一模型。
   - 证据：R2R-CE SR 68.6%、RxR-CE SR 72.4%；摘要未注明数据划分。
   - 理由：适合检查合成指令是否随当前策略同步改善。
4. **[MemoryVLN](https://arxiv.org/abs/2610.18478v1)** · 连续导航长期记忆
   - 贡献：时间压缩、空间锚定和受置信度控制的记忆写回。
   - 证据：摘要称 R2R-CE / RxR-CE SR 改善 2% / 6.5%，未注明绝对值、划分及相对/绝对增幅口径。
   - 理由：检查历史压缩与跨上下文复用是否各自有效。
5. **[VLN-AIEP](https://arxiv.org/abs/2610.13104v1)** · 指令执行进度
   - 贡献：由导航决策标签学习进度，抑制已执行指令内容。
   - 证据：评测覆盖离散 R2R、RxR、CVDN 及连续 R2R-CE，摘要未给可核验指标数值。
   - 理由：与需要额外进度标注的表征监督方法形成对照。
6. **[I2O2I-Nav](https://arxiv.org/abs/2610.18307v1)** · 室内—室外—室内连续任务
   - 贡献：按实际交接状态评测完整任务，并定位阶段失败。
   - 证据：完整任务 SR 1.15% → 4.02%；底层策略保持相同。
   - 理由：适合作为运行时交接与完成验证的压力测试。
7. **[S3-Nav](https://arxiv.org/abs/2610.19431v1)** · 证据与执行权限分离
   - 贡献：压缩决策上下文，同时保留持久空间语义证据并独立验证执行。
   - 证据：摘要未给环境名称或可核验数值。
   - 理由：贴近导航 Agent 运行时设计，但效果仍需全文实验确认。
8. **[WEDGE](https://arxiv.org/abs/2610.19420v1)** · 失败驱动环境生成
   - 贡献：验证弱点后再调整控制该弱点的环境因素。
   - 证据：在 ProcTHOR 比较随机和困难样本替代方案，并报告部分策略向 HM3D 迁移；摘要无数值。
   - 理由：比单纯扩充环境数量更适合做针对性消融。

## 三、重点工作分析
{: id="deep-analysis"}

### 1. VLN-Safe
{: id="vln-safe"}

**先明确碰撞如何改变动作。**

**问题。** 到达终点不等于没有碰撞；sliding 允许碰撞后继续沿障碍物移动，可能掩盖危险动作的后果。

**方法。** [论文](https://arxiv.org/abs/2610.14682v1) 在相同导航任务上对照 sliding 与 blocking，并用深度几何和近期运动检测风险；保护器否决危险前进、执行脱困后归还控制权。

**证据。** 在 R2R-CE blocking 下，保护器对 5 个被评测智能体将碰撞步数降低 62–98%，SR 提高 6.9–29.3 个百分点；这是跨智能体范围，不是单一模型的分数。

**价值。** 本报告判断：它适合优先复现，因为无需先重训策略，就能检查成功是否依赖碰撞后的滑动。

**局限。** 仿真碰撞事件不等价于真机接触损伤；摘要也未提供传感器误差下的保护器可靠性。

**建议。** 固定策略和轨迹集合，比较两种碰撞机制，记录无碰撞成功率、干预次数与额外路程。

### 2. GuidedNav
{: id="guidednav"}

**把任务进度和几何监督留在训练阶段。**

**问题。** 只用动作标签训练，模型可能学到动作相关性，却没有可靠识别当前指令片段及其地标。

**方法。** [论文](https://arxiv.org/abs/2610.13122v1) 引导动作生成前的注意力，并以几何特征约束表示；提供 Guided-R2R 和 Guided-RxR 标注数据。

**证据。** 摘要报告相对于所比较单目 RGB 方法的改善和更快推理，但没有绝对 SR、延迟、硬件或训练成本。

**价值。** 本报告判断：适合检验训练时结构监督能否替代部署时额外文字推理与几何编码。

**局限。** 新增标注带来数据成本；摘要无法区分收益来自标注规模、监督设计还是模型配置。

**建议。** 在固定骨干和数据量下，分别消融子任务、地标与空间监督，并单独测量闭环延迟。

### 3. RSI-Nav
{: id="rsi-nav"}

**数据生成器必须跟上被训练的策略。**

**问题。** 用固定模型一次性标注全部轨迹，生成指令可能逐渐偏离更新后的导航策略能力。

**方法。** [论文](https://arxiv.org/abs/2610.15782v1) 让一个共享模型承担指令生成与指令跟随；逐批生成语言、执行验证、按完成与路径效率筛选，然后立即更新。

**证据。** 两套连续 VLN 基准给出明确 SR，但摘要没有划分、置信区间或与固定生成器的配对增益。

**价值。** 本报告判断：值得复现的是同步更新数据与策略的闭环，而非只追求合成语料数量。

**局限。** 自己生成、自己筛选可能保留策略擅长的轨迹，遗漏困难指令；这是待检验风险，并非论文已经证实的失败。

**建议。** 固定轨迹预算，比较一次性标注与逐批更新，并测量长指令、困难路线和指令多样性的变化。

### 4. MemoryVLN
{: id="memoryvln"}

**压缩历史之后仍需保留空间来源。**

**问题。** 长历史同时受到上下文预算、冗余观测与跨时间空间对齐不稳定的影响。

**方法。** [论文](https://arxiv.org/abs/2610.18478v1) 以近密远疏采样压缩历史，将语义、几何与位姿关联；分层记忆支持上下文间预热、选择性检索和置信度控制写回。

**证据。** 摘要报告两套连续 VLN 基准上的改善，但未给检索与几何编码开销，也未说明跨上下文记忆的实验隔离方式。

**价值。** 本报告判断：复用“在哪里看到什么”的证据，比无位置的历史摘要更贴合回访与重复搜索问题。

**局限。** 必须确认测试环境、轨迹与持久记忆的边界，避免把先验接触场景的收益误当成陌生环境泛化。

**建议。** 区分单次任务内记忆与跨任务记忆；固定 token 预算，比较压缩、空间锚定和写回机制。

### 5. I2O2I-Nav
{: id="i2o2i-nav"}

**原子导航能力并不能保证完整任务。**

**问题。** 独立评测每个阶段，忽略了不理想终点、阶段切换与错误完成判断的累积。

**方法。** [论文](https://arxiv.org/abs/2610.18307v1) 构建物理可执行的复合场景，让后续阶段从前一阶段实际终点开始；模块化智能体增加完成验证、交接与有限恢复。

**证据。** 基准包含 20 个复合场景、1,374 个评测回合；单阶段 SR 为 40.5–76%，完整任务仍仅 4.02%。

**价值。** 本报告判断：它给地面配送机器人提供了比单独室内或室外到达更接近部署的诊断目标。

**局限。** 这些是基准结果，不能当成真实配送成功率；完整任务仍低，系统尚未证明可靠部署。

**建议。** 优先记录阶段条件成功率、交接状态和错误停止，定位末端 ObjectNav 的瓶颈。

## 四、可迁移方法
{: id="transferable-methods"}

- **操作记忆：[REMM](https://arxiv.org/abs/2610.12710v1) / [GAMMA](https://arxiv.org/abs/2610.12793v1)。** 前者撤销失去证据支持的任务结论，后者核验记忆写入与子目标。接入导航记忆写回和完成判定；需要物体身份、位置和证据依赖关系，不能直接复用机械臂评测收益。
- **操作调度：[When to Ask for Help](https://arxiv.org/abs/2610.21160v1)。** 区分触发器质量与慢模块干预价值。可用于导航云边调用预算；前提是计入延迟与失败干预，不能只凭预测告警准确率判断系统收益。
- **语言评测：[Language2Drive](https://arxiv.org/abs/2610.16634v1)。** 固定初始场景，仅改变可行指令，检验行为是否随语言改变。可迁移到左右转、绕行和顺序指令；需先证明两条指令都可执行。
- **空中 VLN：[X-Fly](https://arxiv.org/abs/2610.21450v1)。** 从策略实际到达的状态学习恢复和停止。可借鉴到地面动作块执行；OpenFly 上的结果不能直接外推到 R2R-CE，动力学和停止区域都需重设。
- **记忆评测：[MeMoREval](https://arxiv.org/abs/2610.12561v1)。** 在不知道后续请求时先积累经验，再评测记忆的任务效用与写入、读取成本。可用于比较导航文字、帧与场景图记忆；其无新观测请求协议与在线导航不同。

## 五、分类速览
{: id="categorized-index"}

A 表示直接地面导航相关；B 表示有明确可迁移机制；C 表示领域观察或低相关工作。分类是阅读用途，不替换资料库的自动领域标签。重点工作不重复展开；次池按主题合并列出链接。

### 5.1 VLN 与 ObjectNav
{: id="ground-navigation"}

- **[Two-Stage Multi-modal Learning for Dynamic Audio-Visual Navigation](https://arxiv.org/abs/2610.14077v1)**（A）：动态音视导航：以声强—几何对齐提高未见声音泛化。
- **[Visual-Wi-Fi Navigation to Signal-Emitting Devices](https://arxiv.org/abs/2610.12733v1)**（A）：ViFiNav：结合移动中的 Wi-Fi 方位证据与深度，定位小型或遮挡设备。
- **[GuidedNav](https://arxiv.org/abs/2610.13122v1)**（A）：训练时地标、进度与空间监督；见重点分析。
- **[VLN-AIEP](https://arxiv.org/abs/2610.13104v1)**（A）：VLN-AIEP：以地标锚定轨迹，抑制已经执行的指令片段。
- **[VLN-Safe](https://arxiv.org/abs/2610.14682v1)**（A）：碰撞机制会改变成功率解释；见重点分析。
- **[Training-Free Multi-Agent Cooperative Embodied Search from Uncertain Vision-Language Priors](https://arxiv.org/abs/2610.15038v1)**（A）：将语言模型先验视为假设，以图优化联合规划多机器人搜索。
- **[VbotNav-0](https://arxiv.org/abs/2610.14634v1)**（A）：异步云端子目标与边缘轨迹策略连接室内外路线。
- **[ADAPT](https://arxiv.org/abs/2610.15956v1)**（A）：DarkNav-Bench / ADAPT：用空间音频与深度支持无可见光导航。
- **[RSI-Nav](https://arxiv.org/abs/2610.15782v1)**（A）：RSI-Nav：语言生成与导航策略逐批同步更新；见重点分析。
- **[FireNav](https://arxiv.org/abs/2610.15595v1)**（A）：火灾视觉退化下按导航决策效用选择重建证据。
- **[RefBeliefNav](https://arxiv.org/abs/2610.15375v1)**（A）：RefBeliefNav：区分目标信念和证据不确定性，统一探索与回访。
- **[LERR-Nav](https://arxiv.org/abs/2610.15362v1)**（A）：学习何时探索、恢复及回顾历史线索。
- **[Estimating the Causal Impact of Social Navigation Signals in Corridor Scenarios](https://arxiv.org/abs/2610.16411v1)**（A）：社会导航信号的观察相关性可能与干预效果相反。
- **[G2WAM](https://arxiv.org/abs/2610.16558v1)**（A）：G2WAM：以共享空间坐标锚定历史与目标图像，在潜空间联合预测状态和动作。
- **[MultINav](https://arxiv.org/abs/2610.17089v1)**（A）：MultINav：交互改变路径连通性或目标可见性，执行后需更新状态图。
- **[PathLift](https://arxiv.org/abs/2610.16103v1)**（A）：使用传感器深度逐航点修正无尺度导航轨迹。
- **[CoSMO](https://arxiv.org/abs/2610.18321v1)**（A）：用校准类别集合驱动语义安全距离；覆盖保证不等于零碰撞。
- **[DR-VLA](https://arxiv.org/abs/2610.18373v1)**（A）：人类第一视角视频迁移到四足导航；摘要主要报告伪闭环与开环指标。
- **[I2O2I-Nav](https://arxiv.org/abs/2610.18307v1)**（A）：跨室内外实际阶段交接评测；见重点分析。
- **[LynxScenes](https://arxiv.org/abs/2610.18093v1)**（A）：轮足机器人户外多传感器数据与伪闭环评测；数据和工具计划公开。
- **[MemoryVLN](https://arxiv.org/abs/2610.18478v1)**（A）：空间锚定与跨上下文记忆复用；见重点分析。
- **[SkipWorld](https://arxiv.org/abs/2610.18135v1)**（A）：先用跳步潜预测筛选轨迹，再用自回归模型重排。
- **[ASD](https://arxiv.org/abs/2610.17593v1)**（A）：ASD：用训练时特权轨迹反馈蒸馏语言目标与导航推理。
- **[Cracking Mobile Manipulation](https://arxiv.org/abs/2610.19600v1)**（A）：移动操作按规划、底盘定位和局部操作分解失败并逐项干预。
- **[F2A-ObjectNav](https://arxiv.org/abs/2610.19618v1)**（A）：事件驱动的目标验证、候选管理、恢复与策略切换。
- **[WEDGE](https://arxiv.org/abs/2610.19420v1)**（A）：WEDGE：验证策略弱点并生成针对性训练环境。
- **[HM3D-M3](https://arxiv.org/abs/2610.20121v1)**（A）：HM3D-M3：在同一任务中联合多机器人、多模态与多目标导航。
- **[MoCaP](https://arxiv.org/abs/2610.20998v1)**（A）：MoCaP：从人类视频提取空间记忆及可操作底盘位置的先验。
- **[S3-Nav](https://arxiv.org/abs/2610.19431v1)**（A）：持久状态、决策证据和执行授权分离；需核实全文效果。
- **[VLX-Go](https://arxiv.org/abs/2610.19361v1)**（A）：多阶段学习与策略访问状态上的细化，用于干扰目标中的跟随。
- **[AVALON](https://arxiv.org/abs/2610.21586v1)**（A）：以用户提供的平面图和地点线索初始化拓扑记忆，并在现场核验更新。
- **[MuSTMM](https://arxiv.org/abs/2610.21429v1)**（A）：跨场景声音触发移动操作，以位姿对齐深度构建导航空间记忆。
- **[Training-Free Target-Aware Refinement of 3D Gaussian Language Map](https://arxiv.org/abs/2610.21619v1)**（A）：在已有语言地图中最小量注入目标先验，同时抑制查询无关的偏置。
- **[VP-EVT](https://arxiv.org/abs/2610.21999v1)**（A）：将目标可见性设为运动目标，主动绕开持续遮挡。

### 5.2 记忆、规划与评测
{: id="memory-planning"}

- **[HorizonNav](https://arxiv.org/abs/2610.13849v1)**（B）：由生成视频重建可规划空间；想象场景可规划不代表现实可通行。
- **[EnactOpt](https://arxiv.org/abs/2610.14977v1)**（B）：联合演化上下文构造代码和文字技能；保持模型与运行时固定。
- **[PVFH-Bench](https://arxiv.org/abs/2610.14635v1)**（B）：诊断错误文字前提诱发的视觉幻觉；可用于导航目标不存在的测试。
- **[StateSkill-Agent](https://arxiv.org/abs/2610.14601v1)**（B）：StateSkill：按有限状态组织技能及阶段相关上下文。
- **[Observe, Share, or Predict?](https://arxiv.org/abs/2610.16895v1)**（B）：PARTNR 对照观察、共享知识与预测伙伴动作；更多预测不保证更好协作。
- **[Code2Worlds-2](https://arxiv.org/abs/2610.17651v1)**（B）：从视觉场景编译物理属性，再经模拟、诊断与修正生成可执行环境。
- **[MemAct](https://arxiv.org/abs/2610.17640v1)**（B）：MemAct：在同一循环中写记忆、检索历史并主动获取新视点。
- **[SEAR](https://arxiv.org/abs/2610.18792v1)**（B）：区分保留经验的收益与重复尝试、预算和熟悉任务的收益。
- **[Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2610.19108v1)**（B）：空间推理数据构建与评测综述，关注监督和指标口径。
- **[SCOPER](https://arxiv.org/abs/2610.17657v1)**（B）：SCOPER：联合检查证据记忆、执行进度与故障恢复。
- **[AdaRSI](https://arxiv.org/abs/2610.20474v1)**（B）：先修技能调用，再针对持续失败的子步骤微调执行器。
- **[Dungeon and Visibility World Generator](https://arxiv.org/abs/2610.19523v1)**（B）：可控几何与可见性环境生成工具，适用于探索评测。
- **[C-Slerp](https://arxiv.org/abs/2610.20409v1)**（B）：C-Slerp：用球面表示组合中间目标，并按难度调整航点数量。
- **[PhySGA](https://arxiv.org/abs/2610.20560v1)**（B）：用物理仿真反馈验证组合场景，可借鉴环境生成的可执行性检查。
- **[HiMeco](https://arxiv.org/abs/2610.20312v1)**（B）：HiMeco：区分感知、探索与规划失败，再保留经过执行验证的技能。
- **[Towards Agentic Robots](https://arxiv.org/abs/2610.20442v1)**（B）：从模型、数据、环境与运行时四个方向梳理 Agent 化机器人。
- **[ACT3D](https://arxiv.org/abs/2610.21783v1)**（B）：ACT3D：通过问答结果学习选择视点；空间问答成绩不是导航成功率。
- **[GAMMA](https://arxiv.org/abs/2610.12793v1)**（B）：GAMMA：分离记忆写入与推理，并用机械视觉证据审核主张。
- **[MeMoREval](https://arxiv.org/abs/2610.12561v1)**（B）：MeMoREval：在后续任务未知时积累经验，比较记忆模态与推理成本。
- **[EmbodiedCLBench](https://arxiv.org/abs/2610.13021v1)**（B）：EmbodiedCLBench：评测基础任务经验能否支持未见技能组合。
- **[JoyInside-VLM-Agent](https://arxiv.org/abs/2610.12957v1)**（B）：产品约束下的多模态策略训练，包含目标修改与执行反馈。
- **[DySCOVER](https://arxiv.org/abs/2610.14302v1)**（B）：DySCOVER：以预测误差引导机器人主动采集视觉学习经验。
- **[MAGMA-Bench](https://arxiv.org/abs/2610.13534v1)**（B）：MAGMA-Bench：在技能接口层评测任务中断、执行失败与异构机器人协作。
- **[MPT](https://arxiv.org/abs/2610.22322v1)**（B）：MPT：联合学习竞价参与和行动策略，用于分散式协作。

### 5.3 具身 VLA / 移动操作
{: id="vla-manipulation"}

- **操作记忆、技能与运行时（C）**：[Embodied Zero-Shot Task Planning via Hardware-Aware LLM-VLM Integration for Grounded Robotic Manipulation](https://arxiv.org/abs/2610.14106v1)、[REMM](https://arxiv.org/abs/2610.12710v1)、[Dual-Stream Memory for Vision-Language-Action Policies](https://arxiv.org/abs/2610.13262v1)、[Memory Without Curation](https://arxiv.org/abs/2610.14406v1)、[A Persian Vision-Language-Action Dataset and Benchmark for Ambiguity- and Infeasibility-Aware Instruction-to-Plan Grounding](https://arxiv.org/abs/2610.15666v1)、[Exploring the Limits of Combinatorial Generalization in Embodied Manipulation](https://arxiv.org/abs/2610.15425v1)。
- **操作记忆、技能与运行时（C）**：[SurgiCLAW](https://arxiv.org/abs/2610.15766v1)、[Can a Frontier VLM Serve as a Robot Policy for Memory-Dependent Manipulation?](https://arxiv.org/abs/2610.17350v1)、[ChunkAct](https://arxiv.org/abs/2610.18874v1)、[EqC](https://arxiv.org/abs/2610.18790v1)、[ForkCast](https://arxiv.org/abs/2610.18650v1)、[Let the Mind Conceive, Let the Hand Achieve](https://arxiv.org/abs/2610.18352v1)。
- **操作记忆、技能与运行时（C）**：[ACRO](https://arxiv.org/abs/2610.19677v1)、[Back on Track](https://arxiv.org/abs/2610.19579v1)、[Robota-Embed](https://arxiv.org/abs/2610.19495v1)、[VLRLA](https://arxiv.org/abs/2610.19517v1)、[CarryOn](https://arxiv.org/abs/2610.21453v1)、[Memory as Weights](https://arxiv.org/abs/2610.22070v1)。
- **操作记忆、技能与运行时（C）**：[REKI](https://arxiv.org/abs/2610.22178v1)。
- **操作评测、指令与偏好（C）**：[Spa-Bench](https://arxiv.org/abs/2610.12696v1)、[JitterVLA](https://arxiv.org/abs/2610.13628v1)、[Red-Teaming of Vision-Language-Action Models via Quality Diversity Environment Generation](https://arxiv.org/abs/2610.14415v1)、[LIBERO-Industry](https://arxiv.org/abs/2610.16782v2)、[VLA-GAUGE](https://arxiv.org/abs/2610.16567v1)、[LIBERO-RIG](https://arxiv.org/abs/2610.18843v1)。
- **操作评测、指令与偏好（C）**：[What Matters in Quality-Aware VLA Pretraining](https://arxiv.org/abs/2610.18869v1)、[Paraphrase Consensus Is Not Self-Consistency](https://arxiv.org/abs/2610.19438v1)、[Do VLAs Really Follow Instructions? Diagnosing How-Level Instruction Following in Vision-Language-Action Models](https://arxiv.org/abs/2610.21251v1)、[LIBERO-PeRM](https://arxiv.org/abs/2610.22372v1)。
- **操作数据、迁移与策略优化（C）**：[StereoEngine](https://arxiv.org/abs/2610.12734v1)、[Axiom-0](https://arxiv.org/abs/2610.14155v1)、[Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies](https://arxiv.org/abs/2610.12861v1)、[Noise Space Learning for VLA Adaptation](https://arxiv.org/abs/2610.13943v1)、[OGPO](https://arxiv.org/abs/2610.12914v1)、[RARA](https://arxiv.org/abs/2610.13578v1)。
- **操作数据、迁移与策略优化（C）**：[Splat2Act](https://arxiv.org/abs/2610.13945v1)、[VLMs are Native Action Models](https://arxiv.org/abs/2610.12832v1)、[CFG-Policy](https://arxiv.org/abs/2610.14936v1)、[Flow-Support-Weighted Proposals for Test-Time Verified VLA Policies](https://arxiv.org/abs/2610.14588v1)、[Verify, Act, Distill](https://arxiv.org/abs/2610.14885v1)、[CoLA](https://arxiv.org/abs/2610.15337v1)。
- **操作数据、迁移与策略优化（C）**：[Trust-Calibrated VLA Policy Post-Training with Evolving Imagination](https://arxiv.org/abs/2610.16517v1)、[UniLAP](https://arxiv.org/abs/2610.16969v1)、[Breaking the Average](https://arxiv.org/abs/2610.17983v1)、[Post-Training, Finetuning, and Steering](https://arxiv.org/abs/2610.19143v1)、[Graph-Reweighted Behavior Cloning with Human Interventions for Precise Robotic Manipulation](https://arxiv.org/abs/2610.20504v1)、[Human Demonstrations as Post-Training Data for Any Robot Body](https://arxiv.org/abs/2610.20351v1)。
- **操作数据、迁移与策略优化（C）**：[Multi-Dimensional Featurizers Yield Fine-Grained Steering and Data Shaping in Robot Foundation Models](https://arxiv.org/abs/2610.20767v1)、[One-shot Human Video-Guided Adaptation for Vision-Language-Action Models](https://arxiv.org/abs/2610.19746v1)、[Multi-Teacher On-Policy Distillation for Flow-Based Vision-Language-Action Models](https://arxiv.org/abs/2610.21522v1)。
- **未来预测与动作表示（C）**：[Beyond Actions](https://arxiv.org/abs/2610.13212v2)、[Evo-World](https://arxiv.org/abs/2610.12936v1)、[Forward Dynamics Models](https://arxiv.org/abs/2610.13117v1)、[Vision-Predictor-Action](https://arxiv.org/abs/2610.15556v1)、[Delta Flow Matching](https://arxiv.org/abs/2610.15794v1)、[FlowVLA](https://arxiv.org/abs/2610.16160v1)。
- **未来预测与动作表示（C）**：[GreenWAM](https://arxiv.org/abs/2610.18774v1)、[WAVE-0](https://arxiv.org/abs/2610.18568v1)、[WoVR+](https://arxiv.org/abs/2610.17922v1)、[A Phenomenology of Neural Geometry in Vision-Language-Action Models](https://arxiv.org/abs/2610.20840v1)、[Coordi-pi](https://arxiv.org/abs/2610.19753v1)、[ObjDream](https://arxiv.org/abs/2610.20223v1)。
- **未来预测与动作表示（C）**：[What Matters for World Modeling in Vision-Language-Action Models?](https://arxiv.org/abs/2610.19825v1)、[TS-WAM](https://arxiv.org/abs/2610.21922v1)、[VLA-Delta](https://arxiv.org/abs/2610.21306v1)、[What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning](https://arxiv.org/abs/2610.21915v1)。
- **精细操作、多模态与本体控制（C）**：[DEFT-VLA](https://arxiv.org/abs/2610.14854v1)、[Learning Dynamic Grasping](https://arxiv.org/abs/2610.14848v1)、[PRISM](https://arxiv.org/abs/2610.14704v1)、[Recovering Low-Bit Failures in Vision-Language-Action Policies via Quantization-Conditioned Reinforcement Learning](https://arxiv.org/abs/2610.15675v1)、[Listen Before Act](https://arxiv.org/abs/2610.16313v1)、[Interaction Latents as Actions for Contact-Rich Manipulation](https://arxiv.org/abs/2610.18855v2)。
- **精细操作、多模态与本体控制（C）**：[TacitVLA](https://arxiv.org/abs/2610.17540v1)、[TRACE](https://arxiv.org/abs/2610.17999v1)、[Causal Analysis of Robot Foundation Models Enables Efficiency Gains](https://arxiv.org/abs/2610.20750v1)、[ForceBench](https://arxiv.org/abs/2610.19604v1)、[FoveaVLA](https://arxiv.org/abs/2610.20064v1)、[SPECTRA-VLA](https://arxiv.org/abs/2610.19651v1)。
- **精细操作、多模态与本体控制（C）**：[KODA](https://arxiv.org/abs/2610.21152v1)。
- **推理压缩、动作块与验证（C）**：[Accelerating VLA Inference with Perturbation Informed Execution](https://arxiv.org/abs/2610.19656v2)、[Fast-VLA — You Only Think Once](https://arxiv.org/abs/2610.21389v1)、[FastVLA — Adaptive Visual Token Compression](https://arxiv.org/abs/2610.22146v1)、[Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution](https://arxiv.org/abs/2610.21434v1)、[VeriVLA](https://arxiv.org/abs/2610.21985v1)、[When to Ask for Help](https://arxiv.org/abs/2610.21160v1)。

### 5.4 无人机与其他
{: id="lower-relevance"}

- **空中导航与交互（C）**：[Verifier-Mediated Multi-LLM Coordination for Multi-Robot Path Planning](https://arxiv.org/abs/2610.12761v1)、[BGC-Nav](https://arxiv.org/abs/2610.14702v1)、[ClearNav](https://arxiv.org/abs/2610.17419v1)、[Benchmarking Interaction-Oriented Aerial Mobility across Heterogeneous Environments](https://arxiv.org/abs/2610.18381v1)、[Towards bio-inspired visual navigation](https://arxiv.org/abs/2610.21978v1)、[X-Fly](https://arxiv.org/abs/2610.21450v1)。
- **几何控制、路径规划与估计（C）**：[CeDR-Nav](https://arxiv.org/abs/2610.13007v1)、[TAME-Nav](https://arxiv.org/abs/2610.12877v1)、[Load-Aware Quadruped Navigation with Online Contact-based Geometry Estimation](https://arxiv.org/abs/2610.14596v1)、[Learning Reachability-Based Safety Representations for Real-Time Robot Navigation in Unknown and Dynamic Environments](https://arxiv.org/abs/2610.16522v1)、[LiDAR-based Humanoid Navigation in Dynamic, Non-Convex Environments via Spatio-Temporal Safe Corridors](https://arxiv.org/abs/2610.17232v1)、[End-to-End Vision-Language-Action Policies for Autonomous Lunar Rover Docking](https://arxiv.org/abs/2610.19224v1)。
- **几何控制、路径规划与估计（C）**：[Reinforcement Learning for Large-Scale Lifelong Multi-Agent Path Finding with Shared Spatial Fields](https://arxiv.org/abs/2610.18186v1)、[RingNav](https://arxiv.org/abs/2610.19044v1)、[RoPE-Flow](https://arxiv.org/abs/2610.21086v1)、[Vision-Guided Closed-Loop Autonomous Navigation of Light-Driven Janus Microbots](https://arxiv.org/abs/2610.21222v1)。
- **驾驶与移动视觉适应（C）**：[Reflect-VLA](https://arxiv.org/abs/2610.13453v1)、[YOUDrive](https://arxiv.org/abs/2610.14802v1)、[Mobility Helps Learning](https://arxiv.org/abs/2610.15846v1)、[AUGUstus](https://arxiv.org/abs/2610.16985v1)、[Language2Drive](https://arxiv.org/abs/2610.16634v1)、[SafeDriveVLA](https://arxiv.org/abs/2610.20609v1)。
- **通用 Agent、非导航与检索噪声（C）**：[A4-MR](https://arxiv.org/abs/2610.13056v1)、[MemForm](https://arxiv.org/abs/2610.14431v1)、[Jev-Guard](https://arxiv.org/abs/2610.18326v1)、[Two Routes to Interoceptive Prioritisation](https://arxiv.org/abs/2610.18617v1)、[Average-Reward Policy Identification from a Single Continuing Trajectory](https://arxiv.org/abs/2610.20486v1)、[Beyond the Trial-and-Error Loop](https://arxiv.org/abs/2610.19423v1)。
- **通用 Agent、非导航与检索噪声（C）**：[Generative World Simulation with DAO and JING](https://arxiv.org/abs/2610.19726v1)、[MolHop](https://arxiv.org/abs/2610.19920v1)、[Context Matters! Social, Spatial, and Semantic Coordination with an Embodied Agent in Immersive Analytics](https://arxiv.org/abs/2610.21484v1)、[Evaluating Inquiry Learning in LLM-Based Agents](https://arxiv.org/abs/2610.21604v1)。

### 5.5 资讯与非论文
{: id="news"}

- 本期无新增公众号资讯；上述条目均为 arXiv 归档，包含综述、基准与学位论文等不同文献类型。


## 六、趋势判断与行动建议
{: id="trends-and-actions"}

### 趋势
{: id="trends"}

- **执行验证正在成为独立研究对象。** VLN-Safe、S3-Nav 与 I2O2I-Nav 分别从碰撞、证据权限和阶段交接暴露问题。本报告判断：应单独衡量“提议正确”“允许执行”“执行完成”。
- **记忆与主动补证据开始结合。** MemoryVLN、RefBeliefNav 与 MemAct 关注历史复用、重访和新视点。本报告判断：记忆评测应记录减少了多少重复搜索，以及是否撤销错误信念。
- **数据与系统改进需要失败归因。** WEDGE、EnactOpt 和 AdaRSI 对应环境、运行时与执行器干预。本报告判断：自动主池 64 篇只占全部 175 篇的一部分，次池数量多不意味着本期主要进展都在操作；应按问题而非论文数量分配精读时间。

### 研究空白
{: id="research-gaps"}

- 跨上下文记忆对场景变化、证据冲突和测试信息隔离的影响，尚不能由本期摘要中的 SR 增益回答。
- 室内外连续任务需要同时评测停止、阶段交接和碰撞机制；当前单阶段成绩不足以推断完整任务可靠性。
- 缺少在相同硬件、传感器、交互和训练预算下，对“加监督、加记忆、加保护器”的直接比较。

### 建议动作
{: id="recommended-actions"}

**高优先级**

- **复现 VLN-Safe 的评测对照。** 先确定本地 R2R-CE 是否启用 sliding，再检查 blocking 与碰撞保护器。
- **精读 GuidedNav、RSI-Nav、MemoryVLN 与 VLN-AIEP。** 补齐划分、基线、绝对 SR/SPL、增幅口径和计算预算，再形成可比较的记录。
- **建立阶段交接日志。** 借鉴 I2O2I-Nav，记录前阶段终点、后阶段是否可启动、失败恢复和最终停止依据。

**中优先级**

- **跟踪 S3-Nav、WEDGE、RefBeliefNav。** 分别针对执行授权、困难环境生成和不确定性驱动回访设计小规模消融。
- **检查可迁移记忆协议。** 对照 MeMoREval 与 REMM，加入证据失效、错误完成撤销和再观察任务。

**低优先级**

- **暂缓整套引入纯操作、空中导航及通用 Agent 系统。** 先验证具体接口或评测机制；不把跨任务成功率直接外推为地面 VLN 效果。
