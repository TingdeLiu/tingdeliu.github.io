---
layout: post
lang: zh-CN
translation_id: vln-weekly-2026-10-11
title: "具身导航周报（2026-10-02 ~ 2026-10-08）"
date:   2026-10-11
permalink: /vln-weekly-2026-10-11/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期新增 147 篇独立工作，自动主池 49 篇，3 篇报告 R2R-CE 结果。NavGPT-3 报告 R2R-CE SR 81.51，StageVLN 在 val-unseen 报告 SR 56.3%；PG-VP 的避险引导伴随 SR 下降。重点关注运行时调度、视点锚定记忆与可视化探索状态，并区分完成率、安全性和部署成本。"
---

## 一、本期结论
{: id="key-conclusions"}

- **本期 147 篇新增工作中，3 篇明确报告 R2R-CE 结果，但数字衡量的对象不同。** [NavGPT-3](https://arxiv.org/abs/2610.10787v1) 报告连续环境 R2R-CE / RxR-CE 的 SR 为 81.51 / 90.43；[StageVLN](https://arxiv.org/abs/2610.05664v1) 在 R2R-CE val-unseen 报告 SR 56.3%、SPL 51.4%。[PG-VP](https://arxiv.org/abs/2610.07558v1) 的 84.9% / 83.2% 是两套连续基准上的低风险动作引导比例，同时导航 SR 分别下降 6.8 / 7.9 个百分点。本报告判断：先对齐任务、划分与安全目标，再讨论性能，不能把这些数字直接排成榜单；这里均不是离散 R2R。
- **导航运行时的核心问题变成“谁在何时接管运动”。** NavGPT-3 把推理、行动和监控分成可调度线程；[SuperNav](https://arxiv.org/abs/2610.12126v1) 用视觉点接口连接通用模型与导航工具。[RT-SAFE](https://arxiv.org/abs/2610.09294v1) 在世界持续演化的仿真中发现，实时执行的碰撞数可达配对静态评估的 12.3 倍。本报告判断：模型答得对，还需要系统及时执行、中断和恢复，任务完成率不能单独代表闭环可靠性。
- **长期记忆开始同时回答“在哪里看见过”和“这段记忆如何影响下一步”。** [LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1) 将小物体记忆绑定到观察视点，并评估物体搬动后的重新确认；[MarvisNav](https://arxiv.org/abs/2610.06510v1) 将探索状态直接标在候选路线的图像位置。本报告判断：对地面机器人，更值得验证的是记忆能否减少重复搜索、纠正过期判断，而非上下文能存多少帧。
- **降低计算量需要保留决策所需的信息，并测量完整闭环。** StageVLN 把几何辅助模块限制在训练阶段；[LiteNWM](https://arxiv.org/abs/2610.12368v1) 用未来潜表示给候选轨迹评分；[LaTraNav](https://arxiv.org/abs/2610.11622v1) 用慢语义模块配合快规划器。本报告判断：训练时增加结构监督、推理时压缩预测、运行时异步更新，是不同的优化位置；RTX 5090 上的加速不能直接视为机载性能。

本期新增 147 条，对应 147 篇独立 arXiv 工作，实际发表日期覆盖 2026-10-02 ~ 2026-10-08；另更新 5 篇既有论文版本，不计入新增。公众号新增 0 条。自动主池为导航 32 篇、具身 Agent 17 篇，共 49 篇；次池为 VLA·操作 87 篇、自动驾驶 1 篇、其他 10 篇，共 98 篇。自动标签存在边界误判：PG-VP、SpikingVLA 等导航工作落在次池，IndexAct 等非具身工作进入主池。下文按实际研究内容安排阅读顺序，以上统计保留自动分池口径；分类速览覆盖全部 147 篇。

## 二、优先阅读清单
{: id="priority-reading"}

1. **[NavGPT-3](https://arxiv.org/abs/2610.10787v1)** · 连续 R2R-CE / RxR-CE 与运行时调度
   - 贡献：分别维护推理、行动和监控上下文，支持线程中断与运动控制权切换。
   - 证据：摘要报告完整系统 SR 81.51 / 90.43；未注明对应数据划分，比较前须核对评测协议。
   - 理由：同时涉及当前关注的连续 VLN 与 AgentOS 式运行机制。
2. **[StageVLN](https://arxiv.org/abs/2610.05664v1)** · 连续导航的训练期空间监督
   - 贡献：几何、相对朝向与路线进度监督只用于训练。
   - 证据：4B 主干，R2R-CE val-unseen SR 56.3%、SPL 51.4%；RxR-CE SR 54.3%，摘要未注明其划分。
   - 理由：适合检验不增加部署模块的表征改进。
3. **[PG-VP](https://arxiv.org/abs/2610.07558v1)** · 连续导航中的非视觉风险
   - 贡献：将温度或辐射风险转为动态虚拟障碍，驱动冻结 OmniNav 避让。
   - 证据：R2R-CE / RxR-CE val-unseen 低风险动作引导比例 84.9% / 83.2%，代价是 SR 下降 6.8 / 7.9 个百分点。
   - 理由：直接展示安全目标与到达目标之间的权衡；不应将引导比例写成 SR。
4. **[MarvisNav](https://arxiv.org/abs/2610.06510v1)** · 零样本 ObjectNav
   - 贡献：把候选拓扑节点及其探索状态投影到当前视图。
   - 证据：HM3D SR 81.2%、SPL 42.5%；摘要称 VLM 调用量为 WMNav 的 7.5%。
   - 理由：记忆接口明确，可设计保持信息量不变的对照。
5. **[LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1)** · 动态物体布局中的终身搜索
   - 贡献：空记忆起步，通过多视点检查与视点锚定记忆搜索小物体。
   - 证据：LiSoNav-Eval 含 28 个室内场景、45 类小物体；摘要未提供可核验的 SR / SPL。
   - 理由：任务假设贴近家庭机器人长期使用。
6. **[SuperNav](https://arxiv.org/abs/2610.12126v1)** · 通用请求与陌生环境
   - 贡献：不对通用 MLLM 做导航专用微调，以技能、工具和进度管理支持持续导航。
   - 证据：摘要报告实例级、多物体、需求驱动任务、HM3D 和四足真机评估，未给绝对指标。
   - 理由：适合对照不同运行时如何连接通用推理与专用执行。
7. **[LiteNWM](https://arxiv.org/abs/2610.12368v1)** · 视觉导航的候选轨迹评估
   - 贡献：共享视觉编码，联合预测多时域未来潜表示并评分。
   - 证据：RECON / SCAND / SACSoN 离线宏平均轨迹误差相对 NoMaD+NWM-XL 降低 17.56%；RTX 5090 加速 128.00 倍。真机导航相对 NoMaD 的成功率为 43.3% → 83.3%。
   - 理由：值得核查候选数量、硬件、闭环试验规模及评分器迁移条件。

## 三、重点工作分析
{: id="featured-analysis"}

### 1. NavGPT-3：把中断与控制权作为运行时能力
{: id="navgpt-3"}

**问题。** 长时推理与低延迟运动需要不同的执行节奏。

**方法。** [论文](https://arxiv.org/abs/2610.10787v1) 将推理、行动、监控组织成拥有独立上下文、工具和权限的线程，由运行时调度。

**证据。** 摘要称最低反应时间由每次语言模型决策的 3–19 秒降至每个动作策略步骤的 0.5–1 秒。上述 SR 结果尚需结合具体划分解读。

**价值。** 本报告判断：可借鉴的是中断、上下文隔离与执行反馈接口。

**局限。** 最低反应时间不代表端到端最坏响应时延；基准成功不能证明通用物理环境中的人类水平。

**建议。** 优先审查运动控制权交接、监控触发条件和失败恢复日志，再决定复现范围。

### 2. StageVLN：在训练中学习空间结构
{: id="stagevln"}

**问题。** 单靠动作监督未必保留几何、朝向和全局进度信息。

**方法。** [论文](https://arxiv.org/abs/2610.05664v1) 用冻结几何模型提供分层空间监督，结合相对朝向与专家路线进度目标，部署时移除辅助模块。

**证据。** R2R-CE val-unseen 的 SR / SPL 已在清单列出；摘要未提供同主干基线与训练成本。

**价值。** 本报告判断：适合部署结构已固定、仍能调整训练流程的系统。

**局限。** 移除辅助模块不等于降低原主干计算量，也不意味着训练免费。

**建议。** 分别移除几何、朝向和进度监督，并固定数据与训练预算比较。

### 3. MarvisNav：让记忆直接参与候选路线选择
{: id="marvisnav"}

**问题。** 文本历史与当前图像分离时，模型还需推断记忆与可选路线的对应关系。

**方法。** [论文](https://arxiv.org/abs/2610.06510v1) 用拓扑图维护局部探索进度，将候选节点和状态一起显示在自我中心视图中。

**证据。** 摘要报告 HM3D 的 SR / SPL 与调用量优势，也包含真机验证，但未给出真机试验规模。

**价值。** 本报告判断：收益可能来自信息的呈现方式，可先做接口级复现。

**局限。** 更少模型调用不自动等于更短总时延；建图与投影仍有成本。

**建议。** 保持节点、探索信息和模型相同，对照文本、独立地图与图像叠加三种接口。

### 4. LiSoNav / IVAM-Nav：把“没看见”放回观察条件
{: id="lisonav"}

**问题。** 小物体易被遮挡、经常搬动，旧位置记忆不足以支持持续搜索。

**方法。** [论文](https://arxiv.org/abs/2610.10125v1) 从互补视点检查支撑面，并将记忆与观察视点关联，以便复用和重新核验。

**证据。** 基准区分未移动与已移动目标，要求空场景记忆起步；规模见清单，摘要没有性能数值。

**价值。** 本报告判断：它把搜索效率、可见性和记忆失效放入同一个任务。

**局限。** 目前条目不足以判断感知成本、位姿误差影响或真机可复现性。

**建议。** 分开记录旧位置复查成本、有效观察次数和搬动后的恢复时间。

### 5. SuperNav：检查通用请求如何落到可执行目标
{: id="supernav"}

**问题。** 导航专用微调的覆盖范围可能限制新请求、新环境的泛化。

**方法。** [论文](https://arxiv.org/abs/2610.12126v1) 保留通用 MLLM，借助技能、工具、上下文管理和统一视觉点接口完成导航。

**证据。** 摘要称在多种任务上超过四个基线，并报告四足部署；缺少可比较的绝对值。

**价值。** 本报告判断：可重点检查语义目标转换成可执行目标的接口。

**局限。** “任意任务、任意场景”是论文标题，现有摘要证据不能支持无限泛化。

**建议。** 在未知物体、含歧义请求与目标不可达时检查反馈和重规划，而不只看成功演示。

## 四、可迁移方法
{: id="transferable-methods"}

- **间歇感知：[ALONE](https://arxiv.org/abs/2610.11591v1)。** 通过动作传播空间信念，可靠性不足且新观测有帮助时再看。两类无人机仿真成功率为 98% / 97%；仅在成功试验内，新深度观测步骤占比中位数为 0.9% / 1.3%。本报告判断：可借鉴到共享相机的地面导航，但需重测轮式运动模型、动态障碍和失败试验的感知需求。
- **紧凑历史：[LightVLN](https://arxiv.org/abs/2610.05024v1)。** 用每帧单 token 历史与局部聚合减少输入。摘要报告 Orin NX 16 GB 上 14.61 Hz 推理、11.13 Hz 端到端更新；这是 real-to-sim 硬件在环评估。本报告判断：适合参考输入压缩方式，不能作为真实环境自主飞行或地面 VLN 成功证据。
- **因果记忆评测：[EMBER-Bench](https://arxiv.org/abs/2610.05013v1)。** 同时测下一动作选择和历史原因回溯；16 个模型中最高准确率 61.2%，两位人类评估者均值 98.3%。本报告判断：可在地面导航加入“此前门已关闭、目标被搬动”等历史约束，验证记住原因后能否实际改变动作；问答成绩仍不等于闭环控制。
- **子任务上下文：[RobotUse](https://arxiv.org/abs/2610.04929v2)。** 由后端处理几何、规划和控制，子任务保留细节并返回决策所需信息，执行经验更新持久操作手册。RoboLab 任务成功率 45%，高于 CaP-X 6.7 个百分点。本报告判断：可借鉴到导航子目标的上下文交接，但需另测移动过程的时延与连续状态一致性。

## 五、分类速览
{: id="categorized-index"}


A = 直接服务地面导航；B = 有明确可迁移机制或评测价值；C = 低相关领域观察。操作论文按主题压缩；分类位置不改变自动分池统计。

### 5.1 地面 VLN / ObjectNav / 语义导航
{: id="ground-navigation"}


- **[LiteNWM](https://arxiv.org/abs/2610.12368v1)**（A）：未来潜表示评估轨迹，见优先清单。

- **[LaTraNav](https://arxiv.org/abs/2610.11622v1)**（A）：慢 VLM 与快规划器异步协作，摘要报告同语义更新率下路径更新加速 6.05 倍。

- **[SuperNav](https://arxiv.org/abs/2610.12126v1)**（A）：通用请求到导航工具，见重点分析。

- **[TAPNAV](https://arxiv.org/abs/2610.10748v1)**（A）：视觉不可用时主动触碰环境，以信息增益辅助定位与路线规划。

- **[AirGroundVLN](https://arxiv.org/abs/2610.10421v1)**（A）：空地协同目标导航，强调跨视角记忆与区域到局部规划。

- **[LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1)**（A）：动态小物体终身搜索，见重点分析。

- **[SpikingVLA](https://arxiv.org/abs/2610.09710v1)**（A）：脉冲模型的异步推理；摘要提到导航 SR / SPL，但未注明基准，不据此横比。

- **[MixVPR Teach-and-Repeat](https://arxiv.org/abs/2610.09631v1)**（A）：以 MixVPR 做视觉地点识别，降低示教重放系统的硬件需求。

- **[SiGNgapore](https://arxiv.org/abs/2610.09488v2)**（A）：真实公共空间的标识导航数据，支持连续的标识决策研究。

- **[NavGPT-3](https://arxiv.org/abs/2610.10787v1)**（A）：导航线程调度与中断，见重点分析。

- **[COOL](https://arxiv.org/abs/2610.09358v1)**（A）：从人—物交互推断所有权，主动更新记忆以支持个性化找物。

- **[RT-SAFE](https://arxiv.org/abs/2610.09294v1)**（A）：仿真在推理期间持续演化，分别测任务完成和安全事件。

- **[CUSP](https://arxiv.org/abs/2610.07882v1)**（A）：用风险起始标注和累积告警识别越野危险，不能把人工接管时刻等同于风险起点。

- **[PG-VP](https://arxiv.org/abs/2610.07558v1)**（A）：非视觉风险的动态提示，见优先清单；自动标签为 VLA·操作。

- **[Risk-Sensitive Crowd Navigation / AECP](https://arxiv.org/abs/2610.07474v1)**（A）：以方向相关的不确定性椭球及尾部风险约束适应人群运动分布变化。

- **[Semantic-Aware Humanoid Navigation](https://arxiv.org/abs/2610.07396v1)**（A）：联合处理高程图难以表达的风险与指令—执行偏差；仿真导航和真机步态验证须区分。

- **[Ackermann VLN Sim-to-Real](https://arxiv.org/abs/2610.07192v1)**（A）：在阿克曼转向平台上研究连续 VLN 迁移，摘要未给 SPL / nDTW 数值。

- **[MarvisNav](https://arxiv.org/abs/2610.06510v1)**（A）：可视化探索记忆，见重点分析。

- **[Dual-VAE Sim-to-Real](https://arxiv.org/abs/2610.06327v1)**（A）：双 VAE 对齐仿真与真实特征；摘要的近 91% 是图像分类指标，不是闭环导航 SR。

- **[JESSI](https://arxiv.org/abs/2610.05733v2)**（A）：从 LiDAR 到可执行控制，联合学习人群感知与社会导航。

- **[StageVLN](https://arxiv.org/abs/2610.05664v1)**（A）：训练期空间与进度监督，见重点分析。

- **[Tour-guide Social Navigation](https://arxiv.org/abs/2610.05455v1)**（A）：面向跟随导览机器人的社会力模型，摘要主要提出实验设计。

- **[TACET](https://arxiv.org/abs/2610.03828v1)**（A）：将空间礼让与行走噪声联合控制，避免仅把社会导航理解为保持距离。

### 5.2 记忆、地图、规划与评测
{: id="memory-planning-evaluation"}


- **[2DGS-Planner](https://arxiv.org/abs/2610.11752v1)**（B）：在高斯地图上以光栅化查询可通行几何，属于几何规划组件。

- **[Mine Odyssey](https://arxiv.org/abs/2610.11328v1)**（B）：在 Minecraft 重建场景测长程空间探索，不等同于真机导航。

- **[Arena 5.0](https://arxiv.org/abs/2610.11220v1)**（A）：ROS2 社会导航仿真与场景生成平台。

- **[SCOPE](https://arxiv.org/abs/2610.12431v1)**（B）：为轨迹扩散模型生成可用于控制的不确定性估计，包含人群导航评估。

- **[Ctrl-CWM](https://arxiv.org/abs/2610.09438v1)**（B）：通过世界模型规划生成可受目标控制的人群行为。

- **[iAm.md](https://arxiv.org/abs/2610.10962v1)**（B）：用部署证据与持久对象记录帮助机器人判断技能是否可执行。

- **[System Switch](https://arxiv.org/abs/2610.09683v1)**（B）：研究何时请求慢推理；Doom 闭环实验无方案到达出口，离线收益不能替代任务成功。

- **[ActiveLang](https://arxiv.org/abs/2610.09518v1)**（B）：以语义不确定性选观察视点，主动建立开放词汇 3D 地图。

- **[VeriFine](https://arxiv.org/abs/2610.08761v1)**（B）：让策略、课程和验证器共同改进，关注反馈判断本身的瓶颈。

- **[SPW-Nav](https://arxiv.org/abs/2610.08941v1)**（B）：语言控制的流式全景生成，生成视频质量不是导航策略成绩。

- **[Evidence-Driven Human-Agent-Robot Teaming](https://arxiv.org/abs/2610.08933v1)**（B）：以有权限边界的服务编排证据采集，当前证据为场景示例与硬件在环原型。

- **[SpaTime](https://arxiv.org/abs/2610.08713v1)**（B）：因果几何 token 与响应时间监督支持流式空间推理。

- **[EMHO](https://arxiv.org/abs/2610.08432v1)**（B）：从执行轨迹修订冻结模型之外的 harness，并处理子任务间取舍。

- **[Attacca](https://arxiv.org/abs/2610.07785v1)**（B）：在 Minecraft 中训练搜索—接近—交互连续过程，强调上个任务留下的状态。

- **[OntoPlan](https://arxiv.org/abs/2610.07649v1)**（B）：以符号场景与动作前提支持长程机器人规划；自动标签为其他。

- **[Inspect Robots](https://arxiv.org/abs/2610.06306v1)**（B）：用于物理评估的模块化任务、执行与终止基础设施。

- **[Embodied Guardrail Benchmark](https://arxiv.org/abs/2610.06122v1)**（B）：将防护效果、正常任务误拦截和运行时延分开评估。

- **[Direction-Conditioned Policies](https://arxiv.org/abs/2610.05087v1)**（B）：用表征空间方向与距离组织目标条件策略，非语言导航专用方法。

- **[EMBER-Bench](https://arxiv.org/abs/2610.05013v1)**（B）：历史原因与下一动作的联合评测，见可迁移方法。

- **[RobotUse](https://arxiv.org/abs/2610.04929v2)**（B）：子任务上下文与持续经验手册，见可迁移方法。

- **[PreAct-Nav](https://arxiv.org/abs/2610.04916v1)**（A）：持续子目标与动作条件未来预测支持城市导航，摘要未给绝对指标。

- **[ROMA](https://arxiv.org/abs/2610.06955v2)**（B）：通过交互主动取得视觉、声音、触觉和力觉证据，面向对象感知。

### 5.3 具身 VLA / 移动操作
{: id="vla-manipulation"}


- **多视角、几何与感知—动作接口（6）· C**：[VersaCamVLA](https://arxiv.org/abs/2610.12451v1)、[WARP-VLA](https://arxiv.org/abs/2610.11508v1)、[PAIR](https://arxiv.org/abs/2610.09016v1)、[Wiring Matters](https://arxiv.org/abs/2610.06318v1)、[GeoBridge-VLA](https://arxiv.org/abs/2610.05026v1)、[ExStereo](https://arxiv.org/abs/2610.04805v1)。

- **潜世界模型与未来监督（9）· C**：[PLaW-VLA](https://arxiv.org/abs/2610.12285v1)、[HWAM](https://arxiv.org/abs/2610.12026v1)、[ACT3](https://arxiv.org/abs/2610.11416v1)、[Juno](https://arxiv.org/abs/2610.09940v1)、[RobotAPO](https://arxiv.org/abs/2610.09454v1)、[ViDAL](https://arxiv.org/abs/2610.08150v1)、[SimForcing](https://arxiv.org/abs/2610.06598v1)、[ForeAct3D](https://arxiv.org/abs/2610.04607v1)、[SUAVE](https://arxiv.org/abs/2610.04009v1)。

- **训练、适应与自改进（12）· C**：[HT-Policies](https://arxiv.org/abs/2610.12231v1)、[CAPABLE](https://arxiv.org/abs/2610.11971v1)、[SQAM](https://arxiv.org/abs/2610.10437v1)、[DRIVE](https://arxiv.org/abs/2610.09943v1)、[EmbodiedRSI](https://arxiv.org/abs/2610.10498v1)、[Robo-COP](https://arxiv.org/abs/2610.09228v1)、[VLA-ZO](https://arxiv.org/abs/2610.06271v1)、[Data Augmentation in VLA Post-Training](https://arxiv.org/abs/2610.05994v1)、[TIGER](https://arxiv.org/abs/2610.07527v1)、[ProactiveVLA](https://arxiv.org/abs/2610.06999v1)、[RoboIRS](https://arxiv.org/abs/2610.04681v3)、[PermVLA](https://arxiv.org/abs/2610.04659v1)。

- **语言依赖、指令与捷径（7）· C**：[NegaAlign](https://arxiv.org/abs/2610.11952v1)、[Task Scrubbing](https://arxiv.org/abs/2610.10912v1)、[Rephrase Before You Act](https://arxiv.org/abs/2610.10526v1)、[Do VLAs Understand Instructions?](https://arxiv.org/abs/2610.10178v1)、[YUBI-STAG](https://arxiv.org/abs/2610.09718v1)、[Encoded but Not in Control](https://arxiv.org/abs/2610.06235v1)、[PerturBot](https://arxiv.org/abs/2610.04616v1)。

- **动作表示、节奏与执行（9）· C**：[REACT](https://arxiv.org/abs/2610.12007v1)、[PathTime-VLA](https://arxiv.org/abs/2610.11771v1)、[RoboPace](https://arxiv.org/abs/2610.09696v1)、[TempoBridge](https://arxiv.org/abs/2610.09451v1)、[ProAct](https://arxiv.org/abs/2610.09170v1)、[StairVLA](https://arxiv.org/abs/2610.07756v2)、[ESP](https://arxiv.org/abs/2610.07696v1)、[RACE](https://arxiv.org/abs/2610.05719v1)、[Vela](https://arxiv.org/abs/2610.05230v1)。

- **技能、记忆、路由与验证（8）· C**：[FLOWMEM](https://arxiv.org/abs/2610.12090v1)、[EVIS](https://arxiv.org/abs/2610.11418v1)、[RV-ICL](https://arxiv.org/abs/2610.06843v1)、[EvoMem-VLA](https://arxiv.org/abs/2610.05418v1)、[TUD](https://arxiv.org/abs/2610.05025v1)、[DiVeR](https://arxiv.org/abs/2610.04933v1)、[Local Predictive Sufficiency](https://arxiv.org/abs/2610.04303v1)、[SWAP](https://arxiv.org/abs/2610.06926v1)。

- **剪枝、认证加速与系统开销（6）· C**：[DIVA](https://arxiv.org/abs/2610.09144v1)、[CARE](https://arxiv.org/abs/2610.08917v1)、[ActTune](https://arxiv.org/abs/2610.08444v1)、[VLA-ACL](https://arxiv.org/abs/2610.08133v2)、[SAPrune](https://arxiv.org/abs/2610.05273v1)、[Beyond LLM Serving](https://arxiv.org/abs/2610.05062v1)。

- **鲁棒性、监测与恢复（8）· C**：[Ancestor-VLM Patch Attacks](https://arxiv.org/abs/2610.09708v1)、[SOUL](https://arxiv.org/abs/2610.09496v1)、[TMT](https://arxiv.org/abs/2610.09462v1)、[SALT](https://arxiv.org/abs/2610.07946v1)、[OGAM](https://arxiv.org/abs/2610.05878v1)、[Implied Harm in VLA Instructions](https://arxiv.org/abs/2610.05818v1)、[VICS-G](https://arxiv.org/abs/2610.05166v2)、[Learned Corrector vs. Simple Retreat](https://arxiv.org/abs/2610.06921v1)。

- **数据、移动操作与长程评测（11）· C**：[ManiUnit](https://arxiv.org/abs/2610.12089v1)、[RoboQuest](https://arxiv.org/abs/2610.10388v1)、[VOMMI](https://arxiv.org/abs/2610.08220v1)、[SMART](https://arxiv.org/abs/2610.07652v1)、[BiGym 2.0](https://arxiv.org/abs/2610.07594v1)、[ACG-Bench / AE-VLA](https://arxiv.org/abs/2610.06184v1)、[When Does Retrieval Help?](https://arxiv.org/abs/2610.05492v1)、[RMMBench](https://arxiv.org/abs/2610.05414v1)、[ArticuTable](https://arxiv.org/abs/2610.05249v1)、[Grounded in Time](https://arxiv.org/abs/2610.04255v1)、[Hybrid Flow Task-and-Motion Planning](https://arxiv.org/abs/2610.04771v1)。

- **接触、触觉与近距离探索（4）· C**：[OpenViTac](https://arxiv.org/abs/2610.10384v1)、[MIM-VLA](https://arxiv.org/abs/2610.08425v1)、[Reactive Exploration with Virtual Model Control](https://arxiv.org/abs/2610.08110v1)、[AgenticTactileVLA](https://arxiv.org/abs/2610.04391v1)。

### 5.4 无人机、自动驾驶与其他低相关方向
{: id="other-directions"}


- **[ALONE](https://arxiv.org/abs/2610.11591v1)**（B）：无人机间歇感知，见可迁移方法。

- **[TGIT](https://arxiv.org/abs/2610.10635v1)**（B）：冻结空中导航器前的指令翻译，可借鉴语言接口；仍需地面任务验证。

- **[Sensor-Layout-Agnostic Navigation](https://arxiv.org/abs/2610.08306v1)**（B）：将不同深度相机布局归一化到共同坐标并显式标记盲区，实证来自飞行平台。

- **[LightVLN](https://arxiv.org/abs/2610.05024v1)**（B）：空中导航的紧凑历史与硬件在环效率，见可迁移方法。

- **[WAND](https://arxiv.org/abs/2610.11809v1)**（C）：四旋翼抗风与障碍控制，未研究语言或语义任务。

- **[WareFly-VLA](https://arxiv.org/abs/2610.08526v1)**（C）：仓储无人机的语言条件人员搜索与跟踪。

- **[Learning minimum-time navigation policies in two-dimensional flows with a genetic algorithm](https://arxiv.org/abs/2610.12177v1)**（C）：已知二维流场中的最短时间控制，无语言或语义导航成分。

- **[Visual Swarm Navigation](https://arxiv.org/abs/2610.06400v1)**（C）：视觉群体探索与控制，无语言任务。

- **[EM Digital Twin Calibration](https://arxiv.org/abs/2610.07081v2)**（C）：通过机器人测量路径校准电磁数字孪生，目标为参数估计。

- **[Visual Prosthesis Navigation](https://arxiv.org/abs/2610.05772v1)**（C）：面向视觉假体的人类导航辅助，属于不同的任务和用户群体。

- **自动驾驶与道路生成（3）· C**：[GeoCoTDrive](https://arxiv.org/abs/2610.10390v1)、[Odyssey](https://arxiv.org/abs/2610.06469v1)、[Controllable Road Marking Generation](https://arxiv.org/abs/2610.05771v1)。

- **视觉理解、生成与游戏交互（4）· C**：[ContourVLA](https://arxiv.org/abs/2610.12107v1)、[WorldBench](https://arxiv.org/abs/2610.10622v1)、[Humanity's Sixth Sense](https://arxiv.org/abs/2610.08966v2)、[PlaySuite](https://arxiv.org/abs/2610.07127v1)。

- **非具身检索噪声：仅保留归档链接（5）· C**：[DRL-DCO](https://arxiv.org/abs/2610.11546v1)、[Pelvic Fractures Technology Review](https://arxiv.org/abs/2610.09884v1)、[Energy Transition Investment RL](https://arxiv.org/abs/2610.10768v1)、[Agentic RCA / E4](https://arxiv.org/abs/2610.08622v1)、[IndexAct](https://arxiv.org/abs/2610.07960v1)。

### 5.5 资讯与非论文
{: id="news"}


- 本期没有新增公众号文章或独立资讯；以上均为论文条目。

## 六、趋势判断与行动建议
{: id="trends-actions"}

### 趋势
{: id="trends"}

- **运行时成为可单独比较的研究对象。** NavGPT-3、SuperNav 与 [EMHO](https://arxiv.org/abs/2610.08432v1) 分别研究调度、执行接口和经验驱动的 harness 修订。本报告判断：同一底层模型下，系统组织方式需要独立消融，不能全部归功于模型能力。
- **观察、记忆、动作应形成可核验的因果链。** IVAM-Nav、MarvisNav 与 EMBER-Bench 从可见性、空间呈现和历史约束切入。本报告判断：记忆评估应问“它改变了哪次决策、是否改变正确”，而不只比较检索或回答分数。
- **效率目标必须与安全和完成率一起报告。** 本期 49 篇自动主池与 98 篇次池数量悬殊；大量操作模型的加速结果不能直接转成导航结论。LiteNWM、LaTraNav 和 RT-SAFE 提示，应同时记录全链路时延、观测新鲜度、任务完成与碰撞。

### 研究空白
{: id="research-gaps"}

- 连续 VLN 尚缺本期可直接对齐的“划分、观测、干预方式、计算预算、动态环境”统一比较。尤其不能把最低响应时间当作最坏响应保证。
- 视点锚定记忆与可视化路线记忆如何在位姿漂移、目标搬动和感知误检同时存在时协作，本期摘要尚未给出联合验证。
- 对避险引起的任务失败，如何区分合理拒绝、必要绕行与策略退化，仍需要与 SR 分开的评价。

### 建议动作
{: id="recommended-actions"}

**高优先级**

- **精读 NavGPT-3 与 StageVLN。** 先核对 R2R-CE / RxR-CE 划分、训练数据、输入条件和中断机制，再选择基线；不要混用离散 R2R 结果。
- **复现 MarvisNav 的记忆呈现对照。** 固定底层模型与信息量，测重复访问、搜索路径和真实总时延。
- **跟踪 LiSoNav 的数据与代码。** 从空记忆起步，分别检查目标搬动与未搬动时的搜索效率。

**中优先级**

- **评估 PG-VP 的安全—成功权衡。** 保留低风险动作引导比例与导航 SR 两种指标，不用单个加权总分掩盖代价。
- **检查 SuperNav、LiteNWM 与 LaTraNav 的部署接口。** 在目标硬件上记录端到端延迟、观测到执行的时间差和失败恢复。
- **补充持续运行测试。** 参考 RT-SAFE，让行人与障碍在推理期间继续运动；保留静态配对实验作为对照。

**低优先级**

- **暂缓将次池操作成绩迁入导航排行榜。** 空中导航、游戏智能体、桌面操作与自动驾驶只在明确任务映射和验证协议后迁移；本期未提供闭环指标的工作先跟踪资料。
