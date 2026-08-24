---
layout: post
title: "具身导航周报（2026-08-01 ~ 2026-08-15）"
date:   2026-08-15
permalink: /vln-weekly-2026-08-15/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
published: false
excerpt: "地面导航的记忆从「单次 episode 内的地图」转向「跨 episode 复用的持久结构」，SSTG-Nav、LifelongCrossNav、SAIN 三项互不相关的工作指向同一目标；VLN-CE 执行范式由开环转向闭环并显式拆出纠错点（SC²-WM、WNM-3D、Route2Step）；跨本体导航的核心设计变量收敛到「用什么中间接口连接语义与本体」。"
---

* 目录
{:toc}


## 一、本期结论

- **地面导航的记忆开始从"单次 episode 内的地图"转向"跨 episode 复用的持久结构"。** [SSTG-Nav](https://arxiv.org/abs/2608.00527v1)（一次测绘、多次复用的度量语义拓扑图）、[LifelongCrossNav](https://arxiv.org/abs/2608.07079v1)（跨楼层多目标共享稀疏 3D 语义体素记忆）、[SAIN](https://arxiv.org/abs/2608.09196v1)（把对话答案编译成常驻的空间/物体状态）三项工作互不相关，却都在解决同一件事：让上一次导航获得的信息在下一次查询时仍然可用。本报告判断，这是本期对地面 VLN 最具方向性的信号 —— 长期驻场机器人（家庭、医院、写字楼）的评测协议正在从"单条指令成功率"扩展到"序列查询下的信息复用率"。
- **VLN-CE 的执行范式正在从开环转向闭环，且纠错点被显式拆出来。** [SC²-WM](https://arxiv.org/abs/2608.07548v1) 用世界模型前瞻在动作执行前做状态级计划修正，[WNM-3D](https://arxiv.org/abs/2608.07267v1) 用几何感知表征条件化未来视图与动作的联合生成，[Route2Step](https://arxiv.org/abs/2608.03143v1) 则把"进度跟踪错误"与"执行错误"分离成两个可分别监督的模块。三者的共同前提是：仅靠 next-action 监督无法区分智能体是走错了还是理解错了。
- **"用什么中间接口连接语义与本体"成为跨本体导航的核心设计变量。** [CrossTracer](https://arxiv.org/abs/2608.06688v1) 选择归一化图像平面路点作为统一接口，再用本体条件残差修正；[HumanoidVLN](https://arxiv.org/abs/2608.12860v1) 则从评测侧指出，轮式基准掩盖了双足运动约束与运动引起的相机抖动。本报告判断，跨本体是本期少数同时出现"方法"与"基准"两侧进展的子方向。
- **VLA 侧的大量工作实际是可迁移到导航的运行时监控与记忆机制，而非操作专属。** [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1) 证明任务进度可由线性探针从残差流读出并用作无标签 OOD 检测，[Continue or Replan?](https://arxiv.org/abs/2608.03483v1) 把重规划时机变成可学习决策，[SAFECAST](https://arxiv.org/abs/2608.04246v1) / [GUARD](https://arxiv.org/abs/2608.04510v1) 给出部署期失败检测。这些机制接入 VLN-CE 的成本低于重训导航策略。
- **覆盖边界说明：** 本期 arXiv 侧新增条目达到抓取上限 100 条，最早只回溯到 2026-08-01；07-31 及更早的提交见上一期报告，本期未重新覆盖。公众号侧 14 条中 2 条为非论文资讯，其中 1 条正文抓取失败。本次共 114 条原始条目，跨源合并 5 组后为 109 项独立工作（107 篇论文 + 2 条资讯）。

## 二、优先阅读清单

| 优先级 | 工作 | 任务/场景 | 核心贡献 | 关键证据 | 阅读理由 |
|---|---|---|---|---|---|
| A1 | [SSTG-Nav](https://arxiv.org/abs/2608.00527v1) | 室内可复用 ObjectNav，含真机 | 一次无目标测绘建立度量接地的语义拓扑图，后续查询只做检索与轻量规划 | 公众号解读称仿真基准成功率 97.5%，并完成 ROS2 实体机器人部署；未在条目中给出基准名称与对照方法 | 唯一同时提出"复用范式 + 评测协议 + 真机落地"的地面导航工作 |
| A2 | [LifelongCrossNav](https://arxiv.org/abs/2608.07079v1) | 未知多楼层室内序列多目标 ObjectNav | 共享稀疏 3D 语义体素记忆 + 楼梯专用感知与跨层遍历；提出 HM3D-MFMON 基准 | 摘要在基准描述处截断，未提供可核验数值 | 把"持久记忆"与"跨楼层"两条此前分开处理的线合并，且自带基准 |
| A3 | [WNM-3D](https://arxiv.org/abs/2608.07267v1) | 连续环境 VLN（VLN-CE） | 冻结前馈几何编码器 + 3D Scene-to-Token Adapter，将单目 RGB 历史转为固定长度前缀条件化世界-动作 DiT | 摘要在方法描述处截断，未提供可核验数值 | 本期世界模型类工作中机制描述最完整、与地面 VLN-CE 最直接对口的一篇 |
| A4 | [Route2Step](https://arxiv.org/abs/2608.03143v1) | VLN 指令跟随 | 用显式 step-level 接口解耦语义进度跟踪与动作生成；E-SPA 无需人工时间标注即可监督进度状态 | 摘要在监督流程处截断，未提供可核验数值 | 直接针对"纠错动作标签掩盖进度错误"这一评测与训练顽疾 |
| A5 | [SAP-Nav](https://arxiv.org/abs/2608.12707v1) | 分层开放词汇物体导航（OVON） | 在线构建可查询空间-语义表征 + 主动视点验证，零样本、无需预计算场景图 | 摘要在实验部分截断，未提供可核验数值 | "证据不足就换视点再确认"这一主动感知回路可直接嫁接到现有 ObjectNav 栈 |
| A6 | [HumanoidVLN](https://arxiv.org/abs/2608.12860v1) | 人形机器人 VLN 仿真与基准 | Isaac Sim 上的物理接地平台，RL 运动策略 + 可替换 PD/MPC 路径跟踪 | 4 台机器人（Unitree G1、H1、Internal-A/B），下肢 10–12 DoF、身高 1.17–1.80 m；声明兼容 NaVILA、DualVLN、StreamVLN、JanusVLN；场景要求可导航面积 >100 m² | 若要评估自研策略的跨形态稳健性，这是本期唯一可直接使用的物理接地平台 |
| A7 | [SAIN](https://arxiv.org/abs/2608.09196v1) | 交互式实例目标导航（IIGN） | 把 oracle 回答编译为目标证据、走廊记忆与候选物体标签，存入结构化记忆供统一策略消费 | VL-LN IIGN 基准：SR 20.2 → 25.4，SPL 13.07 → 14.17（对比"已报告的最强方法"，摘要未列出该方法名） | 本期少数给出完整可核验数值的地面导航工作；绝对成功率之低也说明该任务远未解决 |
| A8 | [Embodied Agents Take Control](https://arxiv.org/abs/2607.26148) | 零样本 VLN-CE | 直接复用通用代码 Agent 框架，仅单目 RGB + 4 个基础动作，无地图/深度/全景/路点模块 | 公众号解读称零样本 R2R-CE 成功率区间 68.3%–78%，并称对标大规模训练策略；条目未给出逐项对照表 | 若结论成立，会改变"是否值得继续投入导航专项训练"的判断，值得优先证伪 |

## 三、重点工作分析

### 1. SSTG-Nav：把 ObjectNav 从"每次重探索"改写为"一次测绘、多次检索"

| 维度 | 分析 |
|---|---|
| 问题 | 主流 ObjectNav 采用单次探索模式，每条指令都要重扫环境，长期驻场场景下历史观测被完全丢弃；同时存在"识别到物体但无法抵达"的断层 —— 相机观测点位不等于机器人合法停靠点位 |
| 方法 | 公众号解读称包含三块：度量接地（将 2D 检测反投影到 3D 并偏移 0.8 m 生成安全停靠区）、源感知 3D 软融合（跨视角证据聚合 + 置信度过滤误检）、多候选故障自愈；查询阶段只做检索与轻量路径规划，不重复建图 |
| 证据 | 公众号解读称仿真基准成功率 97.5%，并完成 ROS2 实机部署。**本报告判断该数字暂不可用于横向比较**：条目未给出基准名称、场景划分与基线方法。作者另设 3 套隔离信息边界的评估协议，用以分离"地图几何覆盖"与"语义识别误差"各自的影响 —— 这一协议设计比成功率数字更值得关注 |
| 价值 | 直接对应家庭/办公长期驻场机器人的真实工况；"观测点 → 可通行停靠点"的几何转换是现有语义地图方案（VLMaps、ConceptFusion 等）普遍缺失的一环 |
| 局限 | 依赖环境静态性 —— 一次测绘的前提是布局与物体位置不频繁变化，条目未说明物体移动或场景改变后的地图失效与更新策略；97.5% 的绝对高值也提示基准可能偏易 |
| 建议 | 优先精读其**评估协议部分**而非成功率；[项目页](https://daojiepeng.github.io/SSTG-Nav) 与 [代码](https://github.com/DaojiePENG/sstg-nav-bench) 已给出，0.8 m 停靠偏移这一具体机制可在自有 ObjectNav 栈上单独消融 |

### 2. LifelongCrossNav：持久记忆与跨楼层被合并成同一个问题

| 维度 | 分析 |
|---|---|
| 问题 | ObjectNav 的持久记忆（多目标序列查询）与跨楼层导航此前是两条分开处理的线；单目标单层假设与真实住宅/办公楼不符 |
| 方法 | 每个 episode 内智能体接收有序的物体目标查询序列，持续维护共享稀疏 3D 语义体素记忆，增量累积几何结构、可通行状态与视觉-语言特征，后续查询直接检索而不重建地图；跨层部分由支撑面感知的 3D 可通行性建图、楼梯专用感知与方向感知楼梯遍历组成；统一策略协调同层前沿探索、实时/历史 POI 检索、楼梯导航与目标接近 |
| 证据 | 提出 HM3D-MFMON（基于 HM3D 场景的序列多楼层多目标导航基准）。**摘要在基准描述处截断，本期条目未提供任何可核验的成功率或 SPL 数值** |
| 价值 | "记忆能否降低后续查询的探索成本"是可直接量化的指标，比单目标成功率更贴近长期部署；楼梯感知是把仿真结论推向真实住宅的必要模块 |
| 局限 | 稀疏体素 + VL 特征的内存增长与长期漂移在摘要中未涉及；跨层可通行性高度依赖深度/几何质量，真机迁移风险未知 |
| 建议 | 跟踪 HM3D-MFMON 是否开放；若开放，可与 SSTG-Nav 的复用协议对照，判断"episode 内记忆"与"跨 episode 记忆"两种设定的指标是否可互换 |

### 3. WNM-3D：给世界-动作模型补上几何条件

| 维度 | 分析 |
|---|---|
| 问题 | VLN 系统日益把预训练 VLM 改造成直接输出动作的 VLA，语义能力强但不显式建模"观测在预测动作下应如何演化"；已有的连续 VLN 世界-动作模型（WAM）在联合生成未来视图与动作时，未以从历史推断的几何感知表征为条件 |
| 方法 | 冻结的前馈几何编码器从单目第一视角 RGB 历史提取几何感知表征，可训练的 3D Scene-to-Token Adapter 将其转为世界-动作 Diffusion Transformer token 空间中的定长前缀；通过 block-causal attention，该前缀条件化每一个未来"视频-动作"块，提供共享的几何上下文 |
| 证据 | **摘要在此处截断，本期条目未提供数据集、指标或对照数值。** 公众号同题解读（2026-08-10）正文抓取为空，无法交叉验证 |
| 价值 | 单目 RGB 输入 + 冻结几何编码器的组合对地面平台的传感器要求最低；"持久场景上下文以定长前缀注入"是比逐帧几何融合更省算力的接法 |
| 局限 | 无任何公开数值，当前只能视为机制候选而非已验证方案；扩散式世界-动作模型的推理开销与闭环控制频率的矛盾未在条目中说明 |
| 建议 | 待正文可获取后复核实验；机制层面可先借鉴"冻结几何编码器 + 轻量 Adapter 转 token 前缀"这一接口设计，它与现有 VLN-CE 策略正交 |

### 4. Route2Step：把"走错了"和"理解错了"分开监督

| 维度 | 分析 |
|---|---|
| 问题 | 基于 VLM 的导航器通常只用 next-action 预测同时监督"进度跟踪"和"步骤执行"两种能力。智能体偏离路线时，一个纠正性动作标签可以恢复下一步移动，却无法指明它是选错了子指令还是没执行好正确的子指令 —— 于是智能体继续从错误的进度状态出发做决策 |
| 方法 | 指令分析模块 M_IA 从全局指令与视觉历史预测 step-level 状态；动作生成模块 M_AG 以该状态与近期观测为条件生成局部动作块；E-SPA 步骤对齐流程在无人工时间标注的情况下监督进度状态 |
| 证据 | **摘要在监督流程描述处截断，未提供可核验数值** |
| 价值 | 这是本期最贴近工程诊断需求的一篇：显式进度状态本身就是可观测量，可直接用于失败归因、早停与人工接管触发，价值不止于成功率 |
| 局限 | 显式接口引入误差传播风险 —— M_IA 判断错误会系统性污染下游动作；无标注对齐（E-SPA）的质量决定整体上限，条目未给出对齐精度 |
| 建议 | 优先复现其**接口定义与 E-SPA 对齐流程**；即便不采用完整框架，"输出当前子指令编号"也可作为轻量辅助头加进现有策略 |

### 5. SAP-Nav：把"看不清就换个角度再确认"写进策略

| 维度 | 分析 |
|---|---|
| 问题 | 分层 OVON 要求跟随可能通过场景级、房间级、区域级、实例级线索指定目标的自由形式指令。部分观测下存在一对矛盾需求：空间接地需要环境级的持久证据，而目标验证需要清晰、可判别的候选视图 |
| 方法 | 从主动获取的房间视图增量构建可查询空间-语义表征，使得在任何已探索位置都能发起空间语义查询；主动视点验证（Active Viewpoint Verification）评估当前观测证据是否充分，不足时先将智能体移动到更有信息量的视点，再依类别与属性约束验证候选 |
| 证据 | 声明为完全在线、零样本，无需任务专用训练或预计算场景地图，同时支持分层与标准类别级 OVON。**摘要在实验部分截断，未提供可核验数值** |
| 价值 | 主动视点验证针对的是 ObjectNav 中占比很高的一类失败 —— 远距离误判导致提前终止；该模块可作为独立组件插入既有零样本导航流水线 |
| 局限 | 换视点带来额外路径开销，摘要未说明 SPL 层面的代价；"证据是否充分"的判据依赖 VLM 置信度，本身可能是不可靠信号 |
| 建议 | 与 SSTG-Nav 的多视角证据融合对照阅读 —— 两者都在处理"单次识别不可信"，前者靠离线多视角融合，后者靠在线主动重定位，取舍值得量化 |

## 四、可迁移方法

| 来源方向 | 工作 | 可迁移机制 | 可接入地面导航的位置 | 风险/前提 |
|---|---|---|---|---|
| 操作 VLA 可解释性 | [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1) | 任务进度（归一化剩余时间）可由单个线性探针从 π0.5 残差流线性读出，并作为无标签 OOD 检测器识别进度停滞 | 导航策略的运行时进度监控与卡死检测，可替代人工设定的超时阈值 | 报告称该探针无法有效引导策略，只能读不能控；导航任务的"进度"定义（路径完成度 vs 子指令序号）与操作不同，需重新验证 |
| 操作 VLA 执行调度 | [Continue or Replan? (BCP)](https://arxiv.org/abs/2608.03483v1) | 把固定执行 horizon 换成一串"继续/重规划"的伯努利决策，基座策略冻结、即插即用 | VLN-CE 中动作块的重规划时机；当前多数实现按固定步数重规划，与关键转向点无关 | 最优 horizon 不可直接观测，其监督构造方式需在导航数据上重做 |
| 部署期安全 | [SAFECAST](https://arxiv.org/abs/2608.04246v1)、[GUARD](https://arxiv.org/abs/2608.04510v1) | 前者用对比集扰动改进隐状态风险探针的训练与校准，后者通过消融 KV 缓存条目度量动作对视觉-语言证据的接地程度 | 导航失败检测与接管触发；GUARD 不修改预训练策略，接入成本低 | 两者均在操作基准（LIBERO / DROID / SimplerEnv 类）验证，导航的失败模式（绕圈、错层、提前停止）未覆盖 |
| 长时序记忆 | [AtlasVLA](https://arxiv.org/abs/2608.06729v1)、[Skills in Weights, Memory in Code (HyMeS)](https://arxiv.org/abs/2608.09410v1) | 前者用 4D 体素哈希持久世界状态 + ego 工作状态双记忆解决"物体移出视野即遗忘"；后者让编码 Agent 以可执行启发式系统承担记忆管理，底层策略保持马尔可夫 | 对应导航中的"回头找不到刚才路过的物体"；HyMeS 的分工方式适合把地图/记忆逻辑留在可审查的代码侧 | AtlasVLA 面向腕部单相机操作场景，体素哈希在房间尺度的内存表现未知；HyMeS 依赖 Agent 的迭代反馈循环，实时性存疑 |
| 3D 视觉-语言效率 | [HiSC](https://arxiv.org/abs/2608.04610v1)、[CoverPrune](https://arxiv.org/abs/2608.13226v1)、[3DZip](https://arxiv.org/abs/2608.01185v1) | 三种互补的 3D token 压缩思路：空间图合并的层次聚类（免训练）、最优传输意义下的证据覆盖保持、体素化 + 特征多样性锚点选择 | 语义地图查询与场景级 VLM 推理的算力瓶颈，尤其是持久地图规模随探索增长的场景 | 均在 3D QA / 场景理解上验证，导航所需的是"可通行性 + 目标定位"而非问答精度，压缩取舍标准可能不同 |
| 空间推理接地 | [Chain of Spatial Thoughts / Space Tokens](https://arxiv.org/abs/2608.10278v1) | 将场景级 3D 几何与物体级空间属性蒸馏为连续潜在 token，直接进入 CoT 推理，推理时不需额外空间编码器 | 需要空间关系判断的指令解析（"绕过桌子走到窗边的椅子"） | 条目标题（Chain of Spatial Thoughts）与摘要中的方法名（Space Tokens）不一致，**归属待核验**；未见导航任务实验 |
| 数据质量审计 | [Auditing Instruction-Trajectory Mismatches (MMPF)](https://arxiv.org/abs/2608.07895v1) | 免训练的多模态概率融合，检测"轨迹正确但配错语言指令"的样本并纠正标签 | VLN 演示数据的指令-轨迹配对审计，尤其是自动生成或众包指令 | 在 LIBERO 注入错配与真机噪声数据上验证；VLN 指令的粒度更长，局部邻域一致性假设是否成立需验证 |

## 五、分类速览

### 5.1 地面 VLN / ObjectNav / 语义导航

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| SSTG-Nav | 可复用 ObjectNav | 见重点分析 | A | [链接](https://arxiv.org/abs/2608.00527v1) |
| LifelongCrossNav | 跨楼层持久记忆 | 见重点分析 | A | [链接](https://arxiv.org/abs/2608.07079v1) |
| WNM-3D | VLN-CE 世界模型 | 见重点分析 | A | [链接](https://arxiv.org/abs/2608.07267v1) |
| Route2Step | 进度与执行解耦 | 见重点分析 | A | [链接](https://arxiv.org/abs/2608.03143v1) |
| SAP-Nav | 分层 OVON | 见重点分析 | A | [链接](https://arxiv.org/abs/2608.12707v1) |
| SC²-WM | VLN-CE 闭环 | 世界模型前瞻做状态级计划修正，反馈显示模型能力不足时在测试期选择性更新世界模型；[代码](https://github.com/sunrise-ikun/SC2_WM) 已公开 | A | [链接](https://arxiv.org/abs/2608.07548v1) |
| CompactNav | VLN-CE 表征 | 公众号解读称首次引入"最小充分表征"，以文本指令为先验筛选图像、低秩跨模态信息瓶颈、压缩世界模型三模块串联 | A | [链接](https://arxiv.org/abs/2607.23181) |
| Embodied Agents Take Control | 零样本 VLN-CE | 通用代码 Agent 直接接管导航闭环，仅单目 RGB + 4 动作 | A | [链接](https://arxiv.org/abs/2607.26148) |
| SAIN | 交互式实例目标导航 | 把对话答案编译为常驻结构化记忆而非一次性文本提示 | A | [链接](https://arxiv.org/abs/2608.09196v1) |
| HumanoidVLN | 人形 VLN 基准 | Isaac Sim 物理接地的多形态人形 VLN 仿真平台 | A | [链接](https://arxiv.org/abs/2608.12860v1) |
| CrossTracer | 跨本体导航 | 归一化图像平面路点作统一接口，CE-Adapter 预测本体条件残差修正；CE-RRT* 自动生成训练标注 | A | [链接](https://arxiv.org/abs/2608.06688v1) |
| ULVN | 无序图像目标导航 | 仅 RGB、无时序与里程先验，从无序图像集合构建 2D 拓扑图 + 基于图的置信传播定位 | A | [链接](https://arxiv.org/abs/2608.06833v2) |
| UniNav | 图像目标导航 | 单个扩散过程内联合去噪视觉 token 与连续路点，可利用无路点标注的纯视频数据训练 | A | [链接](https://arxiv.org/abs/2608.03244v1) |
| Latent World Models with Monotone Planning Costs | 图像目标导航规划 | 指出规划代价排序错误会误导 CEM 采样规划器，提出单调代价排序损失 | A | [链接](https://arxiv.org/abs/2608.09073v1) |
| SpikingNav | 鲁棒具身导航 | 脉冲感知编码器 + 脉冲策略网络，面向资源受限平台与视觉退化条件 | A | [链接](https://arxiv.org/abs/2608.05078v1) |
| 360CityArena | 城市导航基准 | 秋叶原 602 段 360° 视频、85 条街道、175 项人工任务；评测称主流 LMM 智能体空间推理低于人类专家 | A | [链接](https://arxiv.org/abs/2608.08814v1) |
| Can VLMs Assess Proxemic Risk | 导航安全评估 | 三个开源 VLM 对第一视角机器人图像做四级危险分类，微调后提升有限；正确分类不等于正确的人物空间定位 | B | [链接](https://arxiv.org/abs/2608.12515v1) |
| Embodied Multimodal Grounding via Semantic-3DGS | 开放词汇移动操作 | 主动多视角语义 3DGS + 可达性感知的底盘位姿选择，3D 语义线索仅注入动作专家后段 | B | [链接](https://arxiv.org/abs/2608.10756v1) |

### 5.2 记忆、地图、规划与评测

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| AtlasVLA | 持久状态记忆 | 4D 体素哈希世界状态 + ego 工作状态双记忆 | B | [链接](https://arxiv.org/abs/2608.06729v1) |
| ChainVLA | 跨查询执行状态 | 递归工作状态 + 稀疏事件记忆携带任务进度，未执行动作续接进下次生成 | B | [链接](https://arxiv.org/abs/2608.02326v2) |
| Explicit Language Memory | 长时序规划 | 把离散时序观测转成带时间逻辑的文本记忆序列 | B | [链接](https://arxiv.org/abs/2608.04765v1) |
| SkillMemo | 技能记忆 | MoE 引导的轨迹分割 + 技能级动态情节记忆库 | B | [链接](https://arxiv.org/abs/2608.05970v1) |
| Skills in Weights, Memory in Code (HyMeS) | 混合记忆 | 底层技能靠模仿学习，高层记忆管理交给编码 Agent 的可执行启发式 | B | [链接](https://arxiv.org/abs/2608.09410v1) |
| BridgeVLA++ | 3D 操作记忆 | 统一时空记忆架构建模持久空间上下文与时间交互 | B | [链接](https://arxiv.org/abs/2608.05042v1) |
| Continue or Replan? (BCP) | 自适应执行 horizon | 见可迁移方法 | B | [链接](https://arxiv.org/abs/2608.03483v1) |
| RTCF | 免训练测试期纠正 | 渐进式记忆对齐检索成功轨迹，在频域而非时域做纠正融合 | B | [链接](https://arxiv.org/abs/2608.04527v2) |
| VANE | 测试时训练 | 候选更新与在线策略隔离，用后续观测验证后才提交，使适应可选择、可回退 | B | [链接](https://arxiv.org/abs/2608.09448v2) |
| SAFECAST | 失败检测 | 见可迁移方法 | B | [链接](https://arxiv.org/abs/2608.04246v1) |
| GUARD | 失败检测 | 见可迁移方法 | B | [链接](https://arxiv.org/abs/2608.04510v1) |
| ValueFormer | 逐帧价值信号 | 冻结 DINOv3 之上的因果 transformer，一次前向同时输出平滑价值与二值纠错信号 | B | [链接](https://arxiv.org/abs/2608.02958v1) |
| Decoding Task Progress | 进度可解释性 | 见可迁移方法 | B | [链接](https://arxiv.org/abs/2608.13474v1) |
| Auditing ITM (MMPF) | 数据审计 | 见可迁移方法 | B | [链接](https://arxiv.org/abs/2608.07895v1) |
| From Recovery to Drop-off | VLA 表征退化 | 动作后训练使深度可解码性在每一层下降，且末层出现额外塌缩，可定位到末层 MLP 干扰 | B | [链接](https://arxiv.org/abs/2608.08904v1) |
| Positional Blind Spots | 空间能力盲区 | 仅移动与任务无关的干扰物即可在局部区域显著抬高失败率，提出定位与 LoRA 缓解流程 | B | [链接](https://arxiv.org/abs/2608.01573v1) |
| Suppression Sticks, Locality Is Fragile | 模型编辑审计 | 任务向量减法在 LIBERO-Goal 十个技能上呈现分离/抵抗/全局塌缩三种状态，局部性不可靠 | C | [链接](https://arxiv.org/abs/2608.04692v1) |
| 3DZip | 3D token 压缩 | 体素化去冗余 + 特征多样性锚点选择 | B | [链接](https://arxiv.org/abs/2608.01185v1) |
| HiSC | 3D token 压缩 | 免训练层次空间聚类，把压缩从 token 级抬到簇级 | B | [链接](https://arxiv.org/abs/2608.04610v1) |
| CoverPrune | 3D token 剪枝 | 以最优传输形式化"保持证据覆盖"，取代最大化多样性 | B | [链接](https://arxiv.org/abs/2608.13226v1) |
| Chain of Spatial Thoughts / Space Tokens | 空间接地 | 见可迁移方法；标题与方法名不一致，归属待核验 | B | [链接](https://arxiv.org/abs/2608.10278v1) |
| World Tokens | 训练期世界建模 | World Adapter 把 VLM 特征转成定长世界 token，训练期接未来视频去噪器、部署期丢弃 | B | [链接](https://arxiv.org/abs/2608.09730v1) |
| GWM-VLA | 几何感知世界模型 | VGGT-Ω 聚合多视角构建几何感知状态，预测目标视图的下一步 patch token | B | [链接](https://arxiv.org/abs/2608.07619v1) |
| SLIM-0.5B | 紧凑潜在交互模型 | 0.5B 参数，自监督掩码轨迹预测学习动作接地的预测性潜变量 | B | [链接](https://arxiv.org/abs/2608.09771v1) |
| Weights or Skills? | 综述 | 沿"冻结权重策略 vs 自写可执行技能"轴梳理机器人学习，按自我改进程度排列 code-as-policy 方法 | B | [链接](https://arxiv.org/abs/2608.01851v1) |
| StellaVLA | 上下文适应 | 离线把原始轨迹转成含任务计划、子目标描述与口述 3D 运动的结构化示范，测试期检索单条示范做条件 | B | [链接](https://arxiv.org/abs/2608.11671v1) |
| In-Context VLA | 语言消费能力 | 论证自由形式文本 CoT 会损害底层控制，主张让 VLA 消费而非生成语言 | B | [链接](https://arxiv.org/abs/2608.05738v1) |
| PhyAI | 推理引擎 | 单运行时统一 VLA 与 WAM 在车载/边缘/云的推理，报告相对官方实现 1.40×–4.65× 加速 | B | [链接](https://arxiv.org/abs/2608.03682v2) |
| EMS (Fast and Accurate) | 双系统解耦 | 环境感知的模型选择，在两个完全解耦的大小系统间切换，无需端到端联合训练 | B | [链接](https://arxiv.org/abs/2608.06434v1) |
| WA-SpecDec | 投机解码 | 把世界模型导出的物理场景感知注入 prefill，使接受阈值随场景风险变化 | B | [链接](https://arxiv.org/abs/2608.08725v1) |
| Temporal GRPO | RL 信用分配 | 构造可检测任务阶段，只比较进入同一阶段的 rollout，缓解轨迹级信用混叠 | B | [链接](https://arxiv.org/abs/2608.13026v1) |
| TEMPO | RL 后训练 | 冻结 VLM 主干，语义投影层与动作专家以不同速率更新的双时间尺度优化 | B | [链接](https://arxiv.org/abs/2608.07314v1) |
| HiRoC | 分层后训练 | 规划器分解子目标、执行器在线交互中持续改进子目标条件动作 | B | [链接](https://arxiv.org/abs/2608.05999v1) |
| DyPES-VLA | 跨本体 | 共享动力学先验（未来预测目标）+ 本体专属控制，减少人工动作格式对齐 | B | [链接](https://arxiv.org/abs/2608.06374v1) |

### 5.3 具身 VLA / 移动操作

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| G0.5 | 统一自回归 VLA | 单一 transformer 解码器在同一目标下输出推理与动作 token，含跨本体动作 tokenizer 与视觉记忆模块 | B | [链接](https://arxiv.org/abs/2608.11739v1) |
| Ego2Robot | 数据合成 | 第一视角人类视频转机器人训练数据，报告 18,561 小时、15 种形态 | C | [链接](https://arxiv.org/abs/2608.02580v1) |
| DreamTrajectory | 移动操作 | 联合规划任务空间运动再生成全身动作，并校验预测动作能否实现意图运动 | B | [链接](https://arxiv.org/abs/2608.01381v1) |
| Panorama-Aware VLA | 移动操作 | 全身遥操作系统 + 全景感知策略，采集 5.5 小时轮式双臂多模态演示 | B | [链接](https://arxiv.org/abs/2608.02257v1) |
| MVUCF | 多相机表征 | 仅训练期注入深度与跨视角对应目标，部署时移除辅助头 | C | [链接](https://arxiv.org/abs/2608.01826v1) |
| Mind-VLA | 指令感知空间对齐 | 只对齐语言指定目标物体的三视图几何，而非整场景 | B | [链接](https://arxiv.org/abs/2608.04633v1) |
| CofactVLA | 因果去混淆 | 单次前向内构造语言掩码反事实分支，抑制"视觉压过语言"的因果混淆 | B | [链接](https://arxiv.org/abs/2608.04396v1) |
| Grounded Semantic Re-Binding | 指令泛化 | 指出改写指令导致的性能崩塌源于架构（视觉文本联合编码引起特征漂移）而非语义理解缺失 | B | [链接](https://arxiv.org/abs/2608.02497v1) |
| SALT | 动作 tokenizer | 要求冻结 VLM 从量化动作潜变量恢复指令，报告 SimplerEnv 平均成功率 71.9% vs 重建式 42.7% | B | [链接](https://arxiv.org/abs/2608.10484v1) |
| LIRA | 层间信息路由 | 每个融合块对齐一个以对应 VLM 层为中心的局部深度窗口 | C | [链接](https://arxiv.org/abs/2608.07596v1) |
| How Should VLAs Use Proprioceptive State | 消融研究 | 固定主干与数据，对比五种本体状态接入方式与历史长度的作用 | B | [链接](https://arxiv.org/abs/2608.03052v1) |
| Cross-View Action Consistency | 视角鲁棒 | 对动作流速度场做正则，用同一 MuJoCo 状态渲染的视角对构造监督 | B | [链接](https://arxiv.org/abs/2608.06965v1) |
| Track4Action | 世界中心蒸馏 | 把冻结 3D 追踪器的已实现转移蒸馏进当前观测策略，部署时不需追踪器 | C | [链接](https://arxiv.org/abs/2608.03727v1) |
| World-to-Wrist | 细粒度操作 | 任务条件下预测未来腕部潜变量作为动作预测上下文 | C | [链接](https://arxiv.org/abs/2608.05369v1) |
| ReTouch | 触觉 VLA | 触觉 patch 编码保留手指身份与局部接触结构，在线细化触觉预测 | C | [链接](https://arxiv.org/abs/2608.01824v1) |
| FACT (Demystifying VLA Failures) | 接触密集失败分析 | 区分精度失败（流匹配训练错配）与力失败（力信号结构），报告五任务 66% vs 最佳基线 41% | C | [链接](https://arxiv.org/abs/2608.01402v1) |
| SpaceVLA | 用户标注锚点 | XR 界面让用户标注抓取与放置区域并渲染成图像叠加，闭环 Unity 抓取成功率 91.25% | C | [链接](https://arxiv.org/abs/2608.05730v1) |
| RoboSynChallenge | 竞赛基准 | 合成数据训练 + 真实环境评测的统一竞赛设置 | C | [链接](https://arxiv.org/abs/2608.12416v1) |
| Policy-Induced Hand Priors | 人形双臂 | 17 种初始构型下量化初始位姿依赖与手部选择偏置 | C | [链接](https://arxiv.org/abs/2608.11769v1) |
| RL Bootstrapping of OpenVLA-OFT | 零演示本体对齐 | 无本体演示条件下用 PPO + GRPO 两阶段适配绳驱并联机器人 | C | [链接](https://arxiv.org/abs/2608.01013v1) |
| Trajectory Divergence Horizon | 手术双臂 | 把手术 VLA 部署形式化为自适应执行 horizon 决策问题 | C | [链接](https://arxiv.org/abs/2608.09125v1) |
| Deltoris | 推理加速 | 位级稀疏 + 投机推理的算法-硬件协同，面向扩散 VLA 的 50–200 Hz 控制 | C | [链接](https://arxiv.org/abs/2608.04428v1) |
| Neural Introspection Gating | KV 缓存复用 | 用 top-2 动作 token 的 logit margin 作零成本置信信号触发缓存失效 | C | [链接](https://arxiv.org/abs/2608.10824v1) |
| The Gate, Not the Cache | 加速可靠性 | 门控信号来自自身加速前向时，LIBERO-Object 0.9 跳过率下成功率降至 0.68（复用）/0.31（删除），且动作级检测器无法发现 | B | [链接](https://arxiv.org/abs/2608.00391v1) |
| CloudEdgeVLA | 云边协同 | 把时序错配当表征学习问题，云端慢特征与边端最新视觉组合 | B | [链接](https://arxiv.org/abs/2608.00569v1) |
| Hermite Curves as Trajectory Priors | 动作块结构 | 用分段三次 Hermite 曲线参数化动作块以强制平滑与端点连续 | C | [链接](https://arxiv.org/abs/2608.01265v2) |

### 5.4 无人机、自动驾驶与其他低相关方向

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| FreqNav | 无人机 VLN | 按飞行阶段在低频全局结构与高频目标细节间路由视觉 token | C | [链接](https://arxiv.org/abs/2608.00970v1) |
| AeroDPO | 无人机 VLN | 论证感知质量重于语言推理容量，2B + 高保真视觉匹配 7B 基线 | C | [链接](https://arxiv.org/abs/2608.07557v1) |
| CoNav-UAV | 无人机协同 | 双高度双机通过 Stackelberg 学习显式建模协作 | C | [链接](https://arxiv.org/abs/2608.01802v1) |
| DBFly | 无人机 VLN | 航点生成前显式空间推理，公众号解读称平均成功率提升 25.07 个百分点 | C | [链接](https://arxiv.org/abs/2608.04825) |
| FlowPilot | 无人机避障 | 双流世界-动作模型，7 阶伯恩斯坦多项式动作表征；公众号解读称 Jetson Orin NX 推理 18 ms 内、实测 5.5 m/s | C | [链接](https://arxiv.org/abs/2608.00635) |
| RecoverFly | 无人机 RL | 失败感知的 token 级 RL 后训练 | C | [链接](https://arxiv.org/abs/2608.09467v1) |
| AirForesight | 无人机 VLN | 当前地图表征同时受当前重建与未来轨迹预测监督 | C | [链接](https://arxiv.org/abs/2608.12835v1) |
| ARIES-Mission2 | 无人机任务生成 | 零样本目标定位后把像素位置转 GPS 航点，用 TSP 优化访问顺序 | C | [链接](https://arxiv.org/abs/2608.12763v1) |
| DreamFly | 无人机 VLN | 因果对齐历史记忆 + 滚动时域扩散规划 | C | [链接](https://arxiv.org/abs/2608.12308v1) |
| GRASP | 无人机跨模态 | 区域聚焦对齐 + 语义原型，应对俯视视角的背景干扰与视觉同构 | C | [链接](https://arxiv.org/abs/2608.09270v1) |
| 语义接地到决策优化统一框架 | 无人机 VLN | 指令接地语义增强 + 相关性感知的历史动态聚合 | C | [链接](https://arxiv.org/abs/2608.09564v1) |
| SkyAnchor | 航拍流式分割 | 语义 token 路由 + 双层记忆库，配套 DroneEyes 像素级流式数据集 | C | [链接](https://arxiv.org/abs/2607.19857) |
| DisasterBench | 无人机灾害推理 | 5330 张航拍图、29300 道多选推理样本，配 2B 端侧模型 | C | [链接](https://arxiv.org/abs/2606.06217v1) |
| DaViNCi | 室外 VLN 数据集 | 首个同时含连续动作与动态元素的室外 VLN 数据集，6 张地图、6933 条轨迹 | C | [链接](https://arxiv.org/abs/2608.11901v1) |
| WAM-Diff2 | 驾驶 VLA | 把预训练自回归通才蒸馏成多任务离散扩散模型 | C | [链接](https://arxiv.org/abs/2608.01035v2) |
| Deferred Exposure of Future Trajectories | 驾驶 CoT | 指出标注时暴露真值未来轨迹会诱发轨迹锚定偏置 | C | [链接](https://arxiv.org/abs/2608.01755v2) |
| XCoT-VLA | 驾驶 CoT | 用紧凑可执行 CoT token 取代自然语言推理 | C | [链接](https://arxiv.org/abs/2608.10976v1) |
| BrainWAM | 驾驶规划 | 指出语义捷径会在共享注意力中压制预测动力学，改为动作空间协调 | C | [链接](https://arxiv.org/abs/2608.12854v1) |
| FlashDrive | 驾驶推理加速 | 同时针对视觉编码、prefill、推理 token 串行、去噪四级瓶颈 | C | [链接](https://arxiv.org/abs/2608.12932v1) |
| FIRE-VLA | 驾驶 RL | 低奖励低多样性组触发自蒸馏，把未解决失败转为特权监督 | C | [链接](https://arxiv.org/abs/2608.13395v1) |
| DriveVLA-M0 | 驾驶记忆 | 失败案例潜在记忆池 + 解耦静态路结构与动态交互的检索模型 | C | [链接](https://arxiv.org/abs/2608.10413v1) |
| CMU-Drive / V2V-VLA | 协同驾驶 | 多网联车闭环协同基准与单次前向联合生成动作、路点、推理与通信策略 | C | [链接](https://arxiv.org/abs/2608.07621v1) |
| Depth-Wise Probing of Planning Token | 驾驶可解释性 | 导航命令在首层后即可线性解码（97.7%），但与原生规划器的兼容性到末层才最优 | C | [链接](https://arxiv.org/abs/2608.07361v1) |
| VLAGuard | 物理攻击防御 | 注意力保护微调把 LIBERO 仿真中 OpenVLA 失败率从 100.0% 降至 25.9% | C | [链接](https://arxiv.org/abs/2608.01028v1) |
| SARF | 物理攻击防御 | 结构感知鲁棒微调，零推理开销 | C | [链接](https://arxiv.org/abs/2608.03231v1) |
| DRIFT | 对抗攻击 | 只攻击流匹配 VLA 的第一步去噪比攻击更宽窗口更强也更省 | C | [链接](https://arxiv.org/abs/2608.03207v1) |
| DURA | 对抗攻击 | 基于扩散生成视觉自然的对抗补丁，支持黑盒设置 | C | [链接](https://arxiv.org/abs/2608.10393v1) |
| UniTexture | 对抗攻击 | 单个纹理化 3D 物体跨任务诱导目标偏移 | C | [链接](https://arxiv.org/abs/2608.13453v1) |
| Text-Guided Glioma Segmentation | 医学影像 | 与具身导航无关，为关键词命中噪声（vision-language 相关） | — | [链接](https://arxiv.org/abs/2608.05389v1) |

### 5.5 资讯与非论文

| 日期 | 事件 | 类型 | 与研究的关系 | 链接 |
|---|---|---|---|---|
| 2026-08-07 | 深蓝学院与上海交大秦通团队"四足机器人 VLN 线下实训营"招生，宣称覆盖运动控制、SLAM、零样本目标导航与 TravExplorer 框架，使用宇树 Go2 + Mid-360 + RealSense + Orin NX | 培训招生 | 商业课程信息，非研究成果。可留意其提到的 TravExplorer 导航框架是否有公开论文或代码 | [链接](https://mp.weixin.qq.com/s/yXNDgROEEe0A7xBuXnzesg) |
| 2026-08-09 | 公众号文章《从理解世界到走进现场 — ROBOT 还差哪一步》 | 观点/综述 | **正文抓取失败（未找到正文容器），内容未知**，本期不做判断 | [链接](https://mp.weixin.qq.com/s/l7oCKh7jQJkbS6mGf0aCgQ) |

## 六、趋势判断与行动建议

### 趋势

- **"记忆"从模型内部隐状态外移为可检查的显式结构。** AtlasVLA 的体素哈希世界状态、LifelongCrossNav 的稀疏语义体素、SSTG-Nav 的拓扑图、SAIN 的结构化对话记忆、HyMeS 的代码化启发式系统，五项工作分别来自操作与导航两个社区，但都选择把记忆放在网络之外。本报告判断这一取向部分源于工程可诊断性需求，而非纯粹的性能考量 —— 隐式记忆无法回答"机器人为什么忘了刚才那把椅子"。
- **失败不再只在训练期处理，而是被搬到运行时检测与纠正。** 本期至少 8 项工作围绕部署期可靠性：SC²-WM（导航状态漂移）、VANE（可回退的测试期适应）、SAFECAST / GUARD（失败检测）、ValueFormer / Decoding Task Progress（逐帧进度信号）、RTCF（免训练测试期纠正）、BCP（重规划时机）。它们的共同假设是基座策略冻结不动。
- **效率工作开始暴露可靠性代价，而非只报告加速比。** The Gate, Not the Cache 显示 token 跳过在高跳过率下会闭环塌缩且动作级检测器无法察觉；Suppression Sticks 显示任务向量减法的局部性不可靠；From Recovery to Drop-off 显示动作后训练系统性削弱 VLM 的深度可解码性。本报告判断，对导航而言这类"代价审计"比新的加速方法更值得跟踪 —— 导航的失败往往是缓慢累积而非单步崩溃。
- **无人机与自动驾驶占据本期条目的大多数（5.4 节 29 项，约占独立工作的 27%），但可迁移到地面 VLN 的机制有限。** 多数集中在飞行动力学、频域 token 分配、驾驶轨迹表征与推理加速，与地面平台的语义导航问题不共享瓶颈。本报告判断，检索关键词的当前配置在无人机方向的噪声比偏高。

### 研究空白

- **地图/记忆失效后的更新策略缺失。** SSTG-Nav 与 LifelongCrossNav 都建立在"环境结构相对稳定"的前提上，但两者的条目均未说明物体被移动、房间重新布置后如何检测并局部更新地图。这是长期驻场部署最先暴露的问题。
- **主动感知的代价没有被计入指标。** SAP-Nav 的主动视点验证与 SAIN 的主动提问都会增加路径长度与交互轮数，但现有 SPL 类指标无法反映"多问一句省了十米路"的权衡，也无法反映提问对人的打扰成本。
- **公开数值严重缺位。** 本期 A 级工作中，仅 SAIN 与 HumanoidVLN 给出可核验的量化信息，其余多为摘要截断或来自公众号解读。跨方法比较在本期基本不可行。

### 建议动作

| 动作 | 目标 | 优先级 |
|---|---|---|
| 精读 + 复现评估协议 | SSTG-Nav 的三套隔离信息边界评估协议（分离地图几何覆盖与语义识别误差的影响），比其 97.5% 成功率更值得移植到自有 ObjectNav 栈 | 高 |
| 精读 + 局部复现 | Route2Step 的 step-level 接口与 E-SPA 对齐流程；先以"当前子指令编号"辅助头的形式做低成本验证 | 高 |
| 加入 benchmark 跟踪 | HumanoidVLN（可用性最明确）与 LifelongCrossNav 的 HM3D-MFMON（待确认是否开放） | 高 |
| 机制借鉴 | 把 Decoding Task Progress 的线性进度探针接到自有导航策略上，作为卡死/绕圈检测器，与现有超时阈值对照 | 中 |
| 优先证伪 | Embodied Agents Take Control 声称的零样本 R2R-CE 68.3%–78%；若成立需重估导航专项训练的投入，若不成立需搞清差距来源 | 中 |
| 待正文可获取后复核 | WNM-3D、SAP-Nav、LifelongCrossNav 的完整实验部分（本期摘要均被截断） | 中 |
| 调整抓取配置 | 检索关键词在无人机方向噪声比偏高（5.4 节占本期约 27%）；同时 arXiv 单次 100 条上限在 15 天窗口下已被打满，建议缩短抓取间隔或提高上限 | 中 |
| 暂缓 | 5.3 节中的操作专属工作（触觉、灵巧手、接触密集任务）与 5.4 节的驾驶推理加速方向，本期未见对地面导航的明确迁移路径 | 低 |

---

**免责声明**：本报告基于 RSS 抓取的公开摘要与公众号解读整理，多数 arXiv 摘要在抓取时被截断，实验结论未经原文核验。所有标注"公众号解读称"的数字均来自第三方解读而非论文原文。引用前请回查原始文献。
