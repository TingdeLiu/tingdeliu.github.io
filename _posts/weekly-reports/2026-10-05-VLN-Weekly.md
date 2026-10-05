---
layout: post
title: "具身导航周报（2026-09-25 ~ 2026-10-02）"
date:   2026-10-05
permalink: /vln-weekly-2026-10-05/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期新增 242 篇独立工作，自动主池 81 篇。EdgeVLN 在完整 R2R-CE val-unseen 1,839 个 episode 上报告 4-bit SR 58.02%；AVERT-VLN 的 76.2% 则包含人工纠错。跨任务记忆、主动取证和可通行姿态成为重点，但不同信息条件下的成功率仍须分开比较。"
---

## 一、本期结论

- **本期最值得复现的是“可解释的执行改进”，而非单一最高成功率。** [PACE](https://arxiv.org/abs/2609.32292v2) 把楼梯、门口等过渡区域的语义意图转成可通行姿态；[SeekVLN](https://arxiv.org/abs/2609.37353v1) 在证据不足时主动观察；[EdgeVLN](https://arxiv.org/abs/2609.35570v1) 同时测完整连续导航基准和板端资源。本报告判断：这三条路线分别对应“知道往哪走却过不去”“尚未看清却继续走”和“仿真能跑却上不了板”的不同瓶颈，应分开诊断。
- **R2R-CE 结果继续增加，但信息条件与评测范围差异明显。** 本期 242 篇独立工作中，8 篇命中 R2R-CE 字段；另有 EdgeVLN 明确使用“R2R VLN-CE”，合计 9 项涉及该连续环境基准，其中 8 项提供 SR/SPL 绝对值或提升声明，1 项仅说明评测基准。EdgeVLN 在完整 val-unseen 1,839 个 episode 上报告 SR 58.02%；[InsightMap](https://arxiv.org/abs/2609.37187v1) 在 R2R-CE/RxR-CE val-unseen 报告 56.9%/54.9%；[AVERT-VLN](https://arxiv.org/abs/2609.39579v1) 的 76.2%/66.3% 包含人工纠错。PACE 只评估跨楼层子集。[PanoVLN](https://arxiv.org/abs/2609.34759v1) 与 SeekVLN 的提升在摘要中以百分号表述，尚不能确定是相对增幅还是百分点。本报告判断：这些数字不足以排成统一排行榜。
- **Harness 开始围绕“经验如何跨任务复用”接受检验。** [终身导航 NavHarness](https://arxiv.org/abs/2609.34276v1) 在使用 SLAM 估计位姿时报告 GOAT-Bench s-SR 83.7、e-SR 36.9，并用结构化恢复交接对照等长度摘要；[MemTransfer](https://arxiv.org/abs/2609.32313v1) 则发现，原示范起点成功率很高的轨迹记忆，换起点后下降 48–49 个百分点。本报告判断：长期记忆的价值取决于换起点、改路径、换目标后还能否使用，存得更多本身不是充分证据。
- **空间记忆正在加入“何时看过、是否有机会看见”的证据条件。** [ECROM](https://arxiv.org/abs/2610.00330v2) 按观测机会解释检测与未检测，在十个 HM3D 家庭环境的长期检索基准中，相对对应指标最强记忆基线提高 support-level AP 4.5 点、搜索 SPL 4.2 点；[EvolvingNav](https://arxiv.org/abs/2609.39166v2) 预测到达检查时刻的目标位置，并保留目标迁出已知候选位置的概率。本报告判断：持续导航需要维护带时间与可见性条件的信念，静态“物体最后一次在哪里”不足以处理搬动和遮挡。
- **安全、正确停止与连续运行正在成为独立评测目标。** EdgeVLN 的停止头直接复用主干隐藏状态；[社会导航世界模型](https://arxiv.org/abs/2609.40177v2) 从名义动作与实际可执行动作的差距推断风险；[STARS](https://arxiv.org/abs/2609.40245v2) 显示社会场景理解仍存在相对规则基线的不足。本报告判断：完成任务、及时响应、避免碰撞和停止正确应分别报告，尤其不能把暂停物理仿真的推理能力直接视为部署能力。

本期新增入库 242 条，对应 242 篇独立 arXiv 工作；公众号新增 0 条。自动主池为导航 48 篇、具身 Agent 33 篇，合计 81 篇；次池为 VLA·操作 133 篇、自动驾驶 8 篇、其他 20 篇，合计 161 篇。自动主池仍含代码检索、损失景观和电脑操作等边界条目，本报告将其降为低相关记录，不作为地面导航证据。覆盖日期为本次新增工作实际发布日期；最新条目为 10 月 2 日。

## 二、优先阅读清单

1. **[EdgeVLN](https://arxiv.org/abs/2609.35570v1)** · 连续 R2R-CE 与板端部署
   - 贡献：联合优化量化、运行时、记忆裁剪和停止判定。
   - 证据：完整 val-unseen 1,839 个 episode，4-bit 模型 SR 58.02%；Orin NX 16 GB 上常驻内存 11.35 GB，停止头每步新增 0.013 秒。
   - 理由：本期评测范围与资源条件交代最清楚，适合作为部署对照。
2. **[PACE](https://arxiv.org/abs/2609.32292v2)** · 连续 R2R-CE/RxR-CE 跨楼层子集
   - 贡献：用可通行姿态连接冻结语义规划器与短程动作，并从失败偏好中学习纠正。
   - 证据：六个零样本导航器的平均 SR，R2R-CE 子集 16.35% → 27.65%，RxR-CE 子集 4.76% → 12.06%。
   - 理由：直接研究地面机器人经常失败的门口、楼梯和狭窄过渡。
3. **[SeekVLN](https://arxiv.org/abs/2609.37353v1)** · 连续 R2R-CE/RxR-CE 的主动取证
   - 贡献：先判断进度证据是否充分，再决定观察或移动；用同状态反事实分支给取证动作分配奖励。
   - 证据：摘要称相对基础模型的 SR 提升分别为 12.7%/7.5%，未给绝对 SR，也未解释百分号口径。
   - 理由：把感知动作的价值与后续导航收益直接联系起来。
4. **[PanoVLN](https://arxiv.org/abs/2609.34759v1)** · 全景 RGB 连续导航
   - 贡献：联合设计长动作序列、置信度执行、分岔路线监督和语义几何表征。
   - 证据：4B、RGB-only，摘要称 R2R-CE/RxR-CE val-unseen SR 超过此前最佳 11.9%/8.7%；绝对结果和增量单位待核验。
   - 理由：全景感知的收益来自完整策略设计，值得逐项做消融。
5. **[InsightMap](https://arxiv.org/abs/2609.37187v1)** · 显式地图与连续导航
   - 贡献：共享主干同时学习导航动作与动作后的地图生成。
   - 证据：R2R-CE/RxR-CE val-unseen SR 56.9%/54.9%；地图预测监督使 R2R-CE SR/SPL 分别提高 4.3/3.2 个百分点。
   - 理由：有直接导航消融，适合核验辅助空间监督的因果贡献。
6. **[FINE](https://arxiv.org/abs/2609.32855v1)** · 连续导航的数据效率
   - 贡献：利用既有示范中的未来地标，学习语义、几何与反事实未来表征。
   - 证据：完整训练数据下，InternVLA-N1 的 R2R-CE/RxR-CE val-unseen SR 分别提高 2.6/4.5 个百分点。
   - 理由：新增监督从已有轨迹中提取，适合示范预算有限的场景。
7. **[AVERT-VLN](https://arxiv.org/abs/2609.39579v1)** · 连续导航的人工辅助恢复
   - 贡献：独立监控器异步识别偏离，触发人工帮助，并把纠错交互转成局部偏好学习。
   - 证据：人工辅助设定下，R2R-CE/RxR-CE val-unseen SR 76.2%/66.3%。
   - 理由：适合研究“何时求助”，引用数字时必须带人工辅助条件。
8. **[NavHarness：Towards Lifelong Embodied Navigation](https://arxiv.org/abs/2609.34276v1)** · 跨任务导航与持续记忆
   - 贡献：跨会话维护地图、任务记录和纠正，使用结构化交接支持恢复。
   - 证据：GOAT-Bench、SLAM 估计位姿条件下 s-SR 83.7、e-SR 36.9；IR2R-CE s-SR 85.9，属于另一个连续导航任务。
   - 理由：具身 Agent 方向的直接导航证据；需核验整体任务完成与子任务完成的差距。

## 三、重点工作分析

### 1. EdgeVLN：量化收益必须连同执行路径与停止判定一起测

**问题。** 模型压小后仍可能因内存搬运、不断增长的历史和停止错误无法部署。

**方法。** 量化 StreamVLN，重建流式上下文并裁剪记忆 token；LATTE 停止头复用隐藏状态，不增加视觉编码器或第二次主干前向。

**证据。** [论文摘要](https://arxiv.org/abs/2609.35570v1) 报告完整 R2R-CE val-unseen 的 4-bit SR 58.02%，板端常驻内存 11.35 GB。同为四位格式，执行路径不同，每步能耗最多相差 36.8 倍；20.8 倍速度与 13.3 倍能耗改善以需要存储流式加载的 BF16 为对照。两位量化失效。

**价值。** 本报告判断：部署收益取决于模型与运行时的共同选择，停止头也应在同一资源预算内评测。

**局限。** 导航 SR 来自仿真，资源测量来自 Orin NX，不能视作同规模真机路线验证；摘要未给绝对每步总延迟或完整 SPL。

**建议。** 复现四位执行路径与停止头，统一记录 SR、SPL、停止误判、内存、每步延迟及能耗。

### 2. SeekVLN：主动观察需要证明能改善后续动作

**问题。** 机器人可能在未见转弯地标时仍自信前进，把错误进度判断带入后续决策。

**方法。** 先从离线专家轨迹生成补充视图与证据标签，再比较同状态下取证分支和直接移动分支的后续收益，训练何时观察。

**证据。** [摘要](https://arxiv.org/abs/2609.37353v1) 在 R2R-CE/RxR-CE 报告 12.7%/7.5% 的 SR 提升声明，未明确增量单位、绝对结果和观察成本。

**价值。** 本报告判断：反事实分支为取证动作提供较直接的信用分配，可用于验证观察是否避免后续偏航。

**局限。** 摘要未量化取证步数、推理成本和真机收益；训练时使用未来专家动作，需核验推理输入与数据预算。

**建议。** 在同一策略上对照固定观察、置信度触发观察和反事实训练，计入转头、停顿及总耗时。

### 3. PACE：规划正确之后，局部执行仍需要空间接地

**问题。** 到达楼梯或门口并不等于能够通过，高层语义目标缺少可执行的姿态和路径条件。

**方法。** 生成以机器人为中心的可通行姿态，再据此产生短程动作；从执行失败中构造正常、恢复与放大偏差行为的偏好对。

**证据。** [论文](https://arxiv.org/abs/2609.32292v2) 在六个冻结零样本导航器上报告跨楼层子集平均 SR：R2R-CE 16.35% → 27.65%，RxR-CE 4.76% → 12.06%。

**价值。** 本报告判断：可通行姿态提供了高层意图与低层控制之间可检查的接口，便于区分识别错误和执行错误。

**局限。** 数字限于跨楼层子集；局部模块经过监督学习和偏好训练，整体系统不能称为完全免训练。真机摘要未给数量或 SR。

**建议。** 单独复现过渡区域，并区分目标定位、通过失败及偏离后恢复；保留纯几何执行对照。

### 4. AVERT-VLN：监控器的价值不能由人工辅助 SR 单独证明

**问题。** 偏离路线后，控制器可能继续错误执行；持续人工监督又难以扩展。

**方法。** 独立监控器根据指令、历史与当前观察识别语义偏离，异步给出 LOST 判定；控制器接受后暂停并请求指导，再用对应决策学习偏好。

**证据。** [摘要](https://arxiv.org/abs/2609.39579v1) 给出 R2R-CE/RxR-CE val-unseen 人工辅助 SR 76.2%/66.3%；风险数据有 20K 条反事实轨迹，进度预训练使用 40K 条正常轨迹。

**价值。** 本报告判断：监控、求助与离线改进是可分别复用的模块，适合检验低频人工介入是否具有成本效益。

**局限。** 摘要未给求助次数、人工信息量、误报率或同预算自主结果，不能把恢复系统成绩归因于导航策略。

**建议。** 固定人工分钟数或纠错次数，对照随机求助、策略置信度和独立监控；报告误报、漏报与恢复成功率。

### 5. 终身导航 NavHarness：子任务成功与整轮成功应分开看

**问题。** 跨任务复用会遇到地图不完整、旧记录与新观察冲突，以及失败后重新启动丢失状态。

**方法。** 会话内查验并纠正地图、搜索记录和房屋知识，跨新会话保存经验；恢复时交接结构化状态，任务结束后整合。

**证据。** [该工作](https://arxiv.org/abs/2609.34276v1) 在 GOAT-Bench、SLAM 估计位姿条件下报告 s-SR 83.7、e-SR 36.9；相对仅保留上下文的独立会话，Astra/Opus 5 的 s-SR 提高 18.6/22.6 点。它与本期另一篇[自适应目标 NavHarness](https://arxiv.org/abs/2609.39915v1) 是不同 arXiv 工作，不能按方法名合并。

**价值。** 本报告判断：结构化恢复交接对等长度摘要的对照，比仅比较是否使用记忆更能解释经验复用机制。

**局限。** 子任务成绩高而整轮成绩低，长期稳定性仍须核验；IR2R-CE 不能代替 R2R-CE 路线跟随结果。闭源模型成本与经验访问预算未在摘要中完整给出。

**建议。** 与 MemTransfer 的换起点、阻断路线和无关历史扰动结合，检查收益来自可迁移知识还是重复目标的记录积累。

## 四、可迁移方法

- **进度判断：[ProgressCompass](https://arxiv.org/abs/2609.36684v1)**。
  - 来源：操作任务 ContextProgress-Bench，24 个任务、120 个 episode；作者报告补入正确上下文后，同五个进度模型的误差降低 77–82%。
  - 接入位置：导航子指令完成判断，为监控器显式提供已完成动作、重复房间与历史地标状态。
  - 本报告判断：先用人工核验上下文做诊断上限，再测试自动生成上下文；操作进度误差不是导航 SR。
- **可执行技能迁移：[RoboBridge](https://arxiv.org/abs/2610.02717v1)**。
  - 来源：LIBERO-PRO 与对应物理操作任务，将任务意图、观察、工具调用和结果验证写成可修订程序；摘要没有可核验结果数字。
  - 接入位置：把导航恢复流程表达为有前置条件和结果检查的技能，保留通用结构、修订环境相关步骤。
  - 本报告判断：迁移要求导航工具的动作、位姿与停止接口稳定；操作实验不能证明路线泛化。
- **异步时间管理：[DiffWAM / FastDreamer](https://arxiv.org/abs/2609.39763v1)**。
  - 来源：无人机导航，将冻结视频模型的预测特征直接接到轨迹生成，按时间戳异步交接；Jetson AGX Thor 上 Flash 管线延迟 1.08 秒。
  - 接入位置：地面机器人高层视觉推理与低层执行重叠时，校验候选轨迹适用的观察时间和当前执行进度。
  - 本报告判断：先复用时间戳与轨迹交接机制；飞行轨迹 RMSE 或 endpoint success 不能替代 VLN 路线指标。
- **局部交互证据：[JRDB-AVR](https://arxiv.org/abs/2609.35032v1)**。
  - 来源：真实机器人视频的主动视觉问答，同时评估答案与支持答案的观察证据；摘要未给可核验准确率。
  - 接入位置：为“已到转弯处”“已识别目标门口”等导航判断建立可见证据评价。
  - 本报告判断：适合诊断凭常识猜对的回答；其问答成绩不能直接代表闭环导航成功。

## 五、分类速览

A 表示直接地面语言/语义导航，B 表示迁移机制明确但任务不同，C 表示低相关观察。以下覆盖全部 242 篇独立工作；自动分池数量与研究相关度不是同一概念。主池中的边界条目已按实际任务放入低相关方向，次池按主题合并索引。

### 5.1 地面 VLN / ObjectNav / 语言与语义导航

- **[PACE](https://arxiv.org/abs/2609.32292v2)**（A）：跨楼层可通行姿态执行；见重点分析。
- **[FINE](https://arxiv.org/abs/2609.32855v1)**（A）：以未来地标的语义、几何与反事实表征改善示范利用率。
- **[RAO-Nav](https://arxiv.org/abs/2609.32224v1)**（A）：使用全模态语言模型做零样本语义视听导航，潜在推理引导相关观察获取。
- **[RECAST](https://arxiv.org/abs/2609.32595v1)**（A）：把可通行表面、风险、朝向与通道判断接地成代价地图。
- **[Query, Align, and Distill](https://arxiv.org/abs/2609.33097v1)**（A）：显式可导航查询构成师生蒸馏接口；摘要未说明具体基准与导航数值。
- **[EdgeVLN](https://arxiv.org/abs/2609.35570v1)**（A）：完整连续基准与 Orin 板端资源联测；见重点分析。
- **[NavHarness：Lifelong Navigation](https://arxiv.org/abs/2609.34276v1)**（A）：跨会话维护地图和结构化恢复交接；见重点分析。
- **[NavJev](https://arxiv.org/abs/2609.34969v1)**（A）：将每步多模态生成压成候选动作证据与结构化选择；R2R-CE SR 27.0%、SPL 22.4%、每步 0.65 秒，摘要未给 split。
- **[PanoVLN](https://arxiv.org/abs/2609.34759v1)**（A）：将全景可见性与长动作、置信度执行、分岔监督联合设计。
- **[Reliability-Aware Route Memory](https://arxiv.org/abs/2609.34163v1)**（A）：逆序查询出程几何锚点，结合动作仲裁与终点验证；仅 50 个逆向配对 episode。
- **[SOR-Nav](https://arxiv.org/abs/2609.34707v1)**（A）：显式决定继续搜索当前区域还是跨区重定位；完整 MP3D 验证集 SR/SPL 61.8%/38.5%。
- **[BCNav](https://arxiv.org/abs/2609.37084v1)**（A）：将声音方向估计与深度导航策略解耦，输出地面机器人连续速度。
- **[CGPI](https://arxiv.org/abs/2609.37591v1)**（A）：由动作引起的观察变化提取信用，保留经验证的适应更新并回滚不受支持的更新。
- **[InsightMap](https://arxiv.org/abs/2609.37187v1)**（A）：显式地图兼作历史参照和动作后辅助预测目标。
- **[Risk-Aware Semantic Grounding](https://arxiv.org/abs/2609.37554v1)**（A）：在规划前区分歧义、幻觉和语义冲突，决定执行、澄清或拒绝。
- **[SeekVLN](https://arxiv.org/abs/2609.37353v1)**（A）：对进度证据不足主动取证；见重点分析。
- **[Astra 跨域具身策略评测](https://arxiv.org/abs/2609.38537v1)**（A）：摘要报告 RxR SR 92%、HM3D 物体搜索 82%，未说明完整 split 或 CE；控制例中物理仿真等待推理，不能当实时导航证据。
- **[ASENA](https://arxiv.org/abs/2609.39207v1)**（A）：编码 Agent 调用可选 4B 导航策略并保存技能；摘要 R2R/RxR 未明确 CE，十轮循环 100-task 子集结果不能替代一次独立测试。
- **[AVERT-VLN](https://arxiv.org/abs/2609.39579v1)**（A）：异步偏离监控与人工指导恢复；见重点分析。
- **[NavHarness：Adaptive Goals](https://arxiv.org/abs/2609.39915v1)**（A）：目标、验证、记忆、执行分工；R2R-CE/RxR-CE 摘要无数值，真机八条路线各评三次，SR 83.3%、导航误差 1.51 米。
- **[UniTrackPLA](https://arxiv.org/abs/2610.00878v1)**（A）：全景时空编码与未来一致性检查统一语言导航和动态人物跟踪。
- **[GeoScaffold](https://arxiv.org/abs/2610.02697v1)**（A）：训练时重建深度、连通性和可通行性，部署时移除几何监督组件；摘要没有明确基准数值。

### 5.2 记忆、地图、规划、社会导航与评测

- **[TRACKGRAPH](https://arxiv.org/abs/2609.31005v1)**（B）：在图像流中跟踪短期 mask 身份，再融合进开放词汇 3D 场景图。
- **[VideoSocNav](https://arxiv.org/abs/2609.37476v2)**（A）：从网络行走视频重建策略状态空间中的可通行地图和行人运动，减少照片级仿真依赖。
- **[MemTransfer](https://arxiv.org/abs/2609.32313v1)**（A）：用换起点、阻断路径和历史相关性对照检验记忆的实际迁移。
- **[3D Point Tracking with State Space Models](https://arxiv.org/abs/2609.34035v1)**（B）：固定尺寸循环状态修正单目公制深度，可作为动态几何前端。
- **[EM-EQA Viewpoint Selection](https://arxiv.org/abs/2609.33288v1)**（B）：将全景投影为透视视图，再按问题相关性和多样性选择历史证据。
- **[HEIR](https://arxiv.org/abs/2609.35955v1)**（B）：联合评估完整人实体事件与局部关系，尚未验证闭环导航收益。
- **[JRDB-AVR](https://arxiv.org/abs/2609.35032v1)**（B）：同时评估主动视觉问答与支持回答的证据，见可迁移方法。
- **[General Asynchronous Agents](https://arxiv.org/abs/2609.35427v1)**（B）：以并发推理协程处理流式视频、游戏和监控；摘要没有地面导航实验。
- **[Multi-Scale Semantic Mapping](https://arxiv.org/abs/2609.34833v1)**（B）：按类别与距离校准城市观测，并降低建图策略间冗余耦合。
- **[SAIL](https://arxiv.org/abs/2609.34347v1)**（B）：按声音源保留事件、方向和距离的对应关系，适合视听导航前端观察。
- **[MAVLN / TRISS](https://arxiv.org/abs/2609.35965v1)**（A）：多机器人导航引入依赖和资源约束，结合共享拓扑记忆与冲突处理。
- **[VCN-Bench](https://arxiv.org/abs/2609.34687v1)**（A）：MP3D 上区分先验视频中的目标识别与闭环到达，含 1,250 个评测 episode。
- **[Human Motion Prediction During Daily Tasks](https://arxiv.org/abs/2609.37971v1)**（B）：分析惯性、占据、语义、注视与明确意图对室内人运动预测的作用。
- **[DeCOD LiDAR SLAM](https://arxiv.org/abs/2609.36753v1)**（B）：用走廊截面地标约束退化轴向漂移，不涉及语言导航策略。
- **[RGB-Only CBF Distillation](https://arxiv.org/abs/2609.36520v1)**（B）：把特权教师的动态避障行为蒸馏为仅用 RGB 历史和速度的安全过滤器。
- **[BRAID / Generative Interactions](https://arxiv.org/abs/2609.37708v1)**（B）：分开建模群体互动状态与个体变化，为社会导航提供候选上下文接口。
- **[Learning to Plan from Random Exploration](https://arxiv.org/abs/2609.38383v1)**（B）：从随机探索中的时间关系学习多尺度可达性，未使用动作或奖励标签训练该关系模型。
- **[ECROM](https://arxiv.org/abs/2610.00330v2)**（A）：以观测机会校准检测和未检测，支持查询时才确定概念的长期物体搜索。
- **[EvolvingNav](https://arxiv.org/abs/2609.39166v2)**（A）：时间索引位置信念、到达时刻预测和可见性条件的负观察更新。
- **[DODGER](https://arxiv.org/abs/2609.38873v1)**（B）：训练中用 CBF 参考和约束违背引导策略，部署不使用运行时安全过滤器。
- **[Terrain Traversability Continual Learning](https://arxiv.org/abs/2609.39755v1)**（B）：从接触经验学习滑移等可通行指标，以验证门限制历史性能退化。
- **[Hallway Legibility](https://arxiv.org/abs/2609.40158v1)**（A）：两项各 45 人研究比较会车侧意图、目标意图与行人分心条件。
- **[Social-WM](https://arxiv.org/abs/2609.40177v2)**（A）：潜在未来预测与实际可执行动作差距提供社会导航安全信号。
- **[STARS / SocialNav-SUB](https://arxiv.org/abs/2609.40245v2)**（B）：真实社会导航场景问答检验时空关系与意图理解，最佳 VLM 仍低于部分规则和人类基线。
- **[Uruqi](https://arxiv.org/abs/2609.39195v1)**（B）：连续视觉经验联合监督自运动跟踪、物体持久建图与空间推理。
- **[Spatial Memory Intelligence](https://arxiv.org/abs/2610.02521v1)**（B）：世界模型长期记忆的空间聚类、稀疏化、动作相关检索与可靠性过滤。
- **[Token Communication for CEAI](https://arxiv.org/abs/2610.01826v1)**（B）：研究任务驱动的语义 token 通信与协作物体搬运，导航迁移尚待验证。
- **[OmniAct3D](https://arxiv.org/abs/2610.03015v1)**（B）：处理全景与透视几何差异，以局部证据支持三维检测。
- **[Representational Alignment](https://arxiv.org/abs/2610.02985v1)**（B）：理论上只对影响当前互动结果的表征差异增加传感运动锚点。

次池中的导航和空间感知边界工作保留索引，尚未据此做主池深度结论：

- **导航与空间感知（B/C），7 项**：[InfraVLA](https://arxiv.org/abs/2609.33647v1)、[TUDF Scene Completion](https://arxiv.org/abs/2609.36543v1)、[S4VY](https://arxiv.org/abs/2609.36875v1)、[PERSEPHONE Spatial Perception](https://arxiv.org/abs/2609.37419v1)、[WayFinder](https://arxiv.org/abs/2609.37922v1)、[GroundingPI](https://arxiv.org/abs/2609.39601v1)、[PAGER](https://arxiv.org/abs/2610.01589v1)。

### 5.3 具身 Agent、VLA 与移动操作

主池中的 Agent 与操作相关工作：

- **[SciHorizon-eLab](https://arxiv.org/abs/2609.30971v1)**（C）：将实验协议编译为可验证的长程实验室操作任务。
- **[RoboFoundry](https://arxiv.org/abs/2609.32862v1)**（B）：将上下文和技能系统作为可验证的整体策略进化，主要证据来自综合具身与操作任务。
- **[Beyond Tasks](https://arxiv.org/abs/2609.33165v1)**（C）：持续行为协调与长期互动的观点论文，未给导航基准证据。
- **[Robot-GST](https://arxiv.org/abs/2609.33872v1)**（B）：RGB-D 重建的 Gaussian-SAM 环境用于执行前仿真与结果检验。
- **[SkillWeaver](https://arxiv.org/abs/2609.36171v1)**（B）：Agent 在闭环交互技能上探索，通过验证器引导树搜索生成操作示范。
- **[RoboSkill](https://arxiv.org/abs/2609.37810v1)**（B）：探索、执行、复用与进化闭环用文本和代码保存操作经验。
- **[ProgressCompass](https://arxiv.org/abs/2609.36684v1)**（B）：显式补入进度判断所需上下文；见可迁移方法。
- **[RobotEQ 3.0](https://arxiv.org/abs/2609.36618v1)**（C）：根据个体特征预测用户期望的主动辅助行为。
- **[Video2Skill](https://arxiv.org/abs/2609.36691v1)**（B）：诊断流式观察中的技能归类、复用与扩展；熟悉技能整合不等于学习新技能。
- **[ChronoGraph](https://arxiv.org/abs/2609.39665v1)**（B）：把过去和预计的动作、可供性部件及状态变化写成共同四维图接口。
- **[Game-Guided Skill Discovery](https://arxiv.org/abs/2609.40137v1)**（B）：自博弈产生可组合、可供人操作的技能，尚非语言导航策略。
- **[Embodied Agent Arena](https://arxiv.org/abs/2610.00854v1)**（B）：1,000 个案例分开测几何精度、功能接地与完整任务成功。
- **[PyRUA-Lean](https://arxiv.org/abs/2610.01939v1)**（B）：条件程序组合原语并按需反馈；700 个操作案例同调用预算下成功率 63.1% → 71.7%。
- **[RoboBridge](https://arxiv.org/abs/2610.02717v1)**（B）：通过经过验证的技能修订延续 sim-to-real 学习；见可迁移方法。

以下为次池主题索引，均未作为地面 VLN 结果比较；同一工作只归入一个主题：

- **失败恢复与系统进化（C），18 项**：[Causeway](https://arxiv.org/abs/2609.30913v1)、[FIND](https://arxiv.org/abs/2609.32069v2)、[Kintsugi-VLA](https://arxiv.org/abs/2609.31048v1)、[SEES](https://arxiv.org/abs/2609.32698v1)、[ActionGround](https://arxiv.org/abs/2609.33256v1)、[Recursive Harness Distillation across Agents for Robot Manipulation](https://arxiv.org/abs/2609.33378v1)、[F4R](https://arxiv.org/abs/2609.35575v2)、[FailPatch](https://arxiv.org/abs/2609.34175v1)、[Self-Evolving Coding Agents](https://arxiv.org/abs/2609.35432v1)、[MotorMind](https://arxiv.org/abs/2609.38078v1)、[ProAct-VLM](https://arxiv.org/abs/2609.37681v1)、[Skill-Space Shooting for Autonomous Robot Policy Improvement](https://arxiv.org/abs/2609.38178v1)、[FailBank](https://arxiv.org/abs/2609.39820v1)、[InterEvolve](https://arxiv.org/abs/2610.02196v1)、[Recova](https://arxiv.org/abs/2610.01178v1)、[SocialVLA](https://arxiv.org/abs/2610.02360v1)、[MobiAgent](https://arxiv.org/abs/2610.03476v1)、[Execution Error Compensation](https://arxiv.org/abs/2609.37334v1)。
- **动作接口、分块与策略结构（C），11 项**：[Fast Plans, Faithful Actions](https://arxiv.org/abs/2609.30833v1)、[DS-VLA](https://arxiv.org/abs/2609.32253v1)、[ActionUNet](https://arxiv.org/abs/2609.34982v1)、[Alignment-Guided Flow Transformer for Efficient Vision-Language-Action Policy Learning](https://arxiv.org/abs/2609.34467v2)、[Quantile Head for Vision-Language-Action Models](https://arxiv.org/abs/2609.34061v1)、[CATok](https://arxiv.org/abs/2609.35469v1)、[Discrete Forcing](https://arxiv.org/abs/2609.39526v1)、[DSDyn-VLA](https://arxiv.org/abs/2609.39198v1)、[ChunkVLA-AM](https://arxiv.org/abs/2610.01856v1)、[TOAST](https://arxiv.org/abs/2610.00899v1)、[Linear Representation Hypothesis](https://arxiv.org/abs/2609.30996v1)。
- **量化、蒸馏与低延迟推理（C），12 项**：[FRAM](https://arxiv.org/abs/2609.30965v2)、[Action Upcycling](https://arxiv.org/abs/2609.34911v2)、[EdgeDAE](https://arxiv.org/abs/2610.00311v1)、[RAVEL](https://arxiv.org/abs/2609.34170v1)、[Token Caching](https://arxiv.org/abs/2609.34319v1)、[DriftOPD](https://arxiv.org/abs/2610.00317v1)、[Asynchronous Distribution Alignment](https://arxiv.org/abs/2609.36540v1)、[Urgency-Aware Denoising](https://arxiv.org/abs/2609.37772v1)、[Spike-driven VLA](https://arxiv.org/abs/2609.39514v1)、[Two-Step Flow Denoising](https://arxiv.org/abs/2609.39822v1)、[CHASE-VLA](https://arxiv.org/abs/2610.02666v1)、[FastOPD](https://arxiv.org/abs/2610.02832v1)。
- **语言接地与组合泛化（C），10 项**：[GT-VLA](https://arxiv.org/abs/2609.31904v1)、[Spatial Grafting](https://arxiv.org/abs/2609.35249v1)、[Layer Selection](https://arxiv.org/abs/2609.36118v1)、[Referential Guidance](https://arxiv.org/abs/2609.38616v1)、[RawVLA](https://arxiv.org/abs/2609.37530v1)、[Cue the Flow](https://arxiv.org/abs/2609.38989v1)、[Same Scene, Different Task](https://arxiv.org/abs/2610.00524v1)、[Instruction-Action Binding / ECT](https://arxiv.org/abs/2609.39971v1)、[WorldAuditBench](https://arxiv.org/abs/2609.40325v1)、[MixVLA](https://arxiv.org/abs/2610.02898v1)。
- **未来预测与世界模型（C），16 项**：[Towards VLA-Dreamer](https://arxiv.org/abs/2609.31313v1)、[Devol-ONE](https://arxiv.org/abs/2609.32193v2)、[SLIP-VLA](https://arxiv.org/abs/2609.33575v1)、[RoboFL](https://arxiv.org/abs/2609.34968v1)、[WorldGuide](https://arxiv.org/abs/2609.34206v1)、[DILL](https://arxiv.org/abs/2609.37165v1)、[V-JEPA Policy](https://arxiv.org/abs/2609.37250v1)、[Predictive Supervision Placement](https://arxiv.org/abs/2609.36645v2)、[EWAM](https://arxiv.org/abs/2609.39973v1)、[MotionWeave](https://arxiv.org/abs/2609.39324v1)、[Planning Limits of Latent World Models](https://arxiv.org/abs/2609.39235v1)、[Token-World](https://arxiv.org/abs/2610.00575v1)、[ATI-VLA](https://arxiv.org/abs/2610.01741v1)、[UniWAM](https://arxiv.org/abs/2610.02054v1)、[World-Calibrated Proposal-to-Action](https://arxiv.org/abs/2610.02323v1)、[IG-VLA](https://arxiv.org/abs/2610.02626v1)。
- **强化学习与策略优化（C），14 项**：[VLaRL](https://arxiv.org/abs/2609.30868v1)、[PF-RL](https://arxiv.org/abs/2609.32634v1)、[SAMBAR](https://arxiv.org/abs/2609.32108v1)、[Principal Steering Subspaces for Online Adaptation of Frozen Generative Robot Policies](https://arxiv.org/abs/2609.33765v1)、[TimelyDAgger](https://arxiv.org/abs/2609.33157v2)、[PolicyWeave](https://arxiv.org/abs/2609.33125v1)、[Adjoint Guidance Flow](https://arxiv.org/abs/2609.34944v1)、[ChronoSRL](https://arxiv.org/abs/2609.36238v1)、[StructRL](https://arxiv.org/abs/2609.36352v1)、[Low-Rank RL](https://arxiv.org/abs/2609.34599v1)、[UTMPO](https://arxiv.org/abs/2609.34688v1)、[Online-ES](https://arxiv.org/abs/2609.38855v1)、[PRICE the Action Chunks](https://arxiv.org/abs/2609.38890v1)、[eRLT](https://arxiv.org/abs/2610.00913v1)。
- **安全、故障与行为诊断（C），17 项**：[One-Step Observation Perturbations](https://arxiv.org/abs/2609.32550v1)、[Adversarial Training / View Collapse](https://arxiv.org/abs/2609.33707v1)、[Do Not Cut When Uncertain](https://arxiv.org/abs/2609.35039v1)、[MAIL-Bench](https://arxiv.org/abs/2609.35003v1)、[State Readout Diagnostics](https://arxiv.org/abs/2609.34684v1)、[RoboIRGBench](https://arxiv.org/abs/2609.34384v1)、[Acceleration Benchmark Diagnostics](https://arxiv.org/abs/2609.37771v1)、[Memorize, Adapt, Ignore](https://arxiv.org/abs/2609.38401v1)、[Blackout vs. Freeze](https://arxiv.org/abs/2609.39145v1)、[Exploiting Vulnerabilities](https://arxiv.org/abs/2609.39178v1)、[Multi-Link Safety Filtering for VLA Policies Around Moving Hazards](https://arxiv.org/abs/2609.40007v1)、[Behavioural Robustness Evaluation](https://arxiv.org/abs/2610.01351v1)、[WBAG](https://arxiv.org/abs/2610.01083v1)、[Detect and Suppress](https://arxiv.org/abs/2610.03498v1)、[ManiPhysicsBench](https://arxiv.org/abs/2610.02802v1)、[Multi-Agent Action Collapse](https://arxiv.org/abs/2610.02848v1)、[Reactive Obstacle Avoidance](https://arxiv.org/abs/2609.35231v2)。
- **触觉、人类输入与专项操作（C），9 项**：[CLAP](https://arxiv.org/abs/2609.32767v1)、[BrainVLA](https://arxiv.org/abs/2609.34561v1)、[Gaze Prompts](https://arxiv.org/abs/2609.34550v1)、[mmHRI](https://arxiv.org/abs/2609.34220v2)、[Tactile Curiosity](https://arxiv.org/abs/2609.40134v2)、[EMG Task Conditioning](https://arxiv.org/abs/2610.01794v1)、[RoboChemGym](https://arxiv.org/abs/2610.02708v1)、[SARI](https://arxiv.org/abs/2610.02804v1)、[SimpleTouch](https://arxiv.org/abs/2610.02784v1)。
- **记忆与长期状态（C），11 项**：[RecastVLA](https://arxiv.org/abs/2609.32155v1)、[SSVR](https://arxiv.org/abs/2609.33412v1)、[D²-VLA](https://arxiv.org/abs/2609.34792v2)、[Ledger](https://arxiv.org/abs/2609.34554v1)、[LexiconVLA](https://arxiv.org/abs/2609.36774v1)、[Action-History Memory](https://arxiv.org/abs/2609.37307v1)、[T²Mem](https://arxiv.org/abs/2609.36720v1)、[ECoMEM](https://arxiv.org/abs/2610.00801v1)、[Optimus-R](https://arxiv.org/abs/2609.39794v1)、[MIKASA-Robo-VLA](https://arxiv.org/abs/2610.00604v1)、[Divide-and-Remember](https://arxiv.org/abs/2610.00982v1)。
- **人形、双臂、移动操作与协作（C），11 项**：[Fiatlux](https://arxiv.org/abs/2609.38216v1)、[TAO-DA](https://arxiv.org/abs/2609.33197v1)、[Humanoid Loco-Manipulation With Discrete VLA Model](https://arxiv.org/abs/2609.35709v1)、[Uni-VLaT](https://arxiv.org/abs/2609.35450v2)、[Cooperative Multi-Agent VLA](https://arxiv.org/abs/2609.36588v1)、[EgoAlign](https://arxiv.org/abs/2609.38046v3)、[EgoHumanoid-V2](https://arxiv.org/abs/2609.37181v1)、[FineART](https://arxiv.org/abs/2609.36416v2)、[IronMind](https://arxiv.org/abs/2609.39403v1)、[Whole-Body Human Pretraining](https://arxiv.org/abs/2610.00438v1)、[DuoMind](https://arxiv.org/abs/2610.02161v1)。

### 5.4 无人机、自动驾驶与其他低相关方向

- **[SatNav](https://arxiv.org/abs/2609.31507v1)**（C）：卫星图像构造城市级无人机 VLN，非地面视角评测。
- **[SemNav：Code Repository](https://arxiv.org/abs/2609.31176v1)**（C）：代码仓库问题定位中的“导航”，不属于具身导航。
- **[Wireless Evidence Acquisition](https://arxiv.org/abs/2609.31428v1)**（C）：截止时间约束的多传感器无线调度，未验证闭环 VLN。
- **[AquaBEV-Nav](https://arxiv.org/abs/2609.32156v1)**（C）：水下单目直接预测 BEV 占据并用于探索。
- **[AquaWAM](https://arxiv.org/abs/2609.33299v2)**（C）：水下动作与被动动力学共同预测，不能直接映射为地面导航结果。
- **[DroneWAM](https://arxiv.org/abs/2609.33148v1)**（B）：无人机潜在世界模型按场景调整预测深度，地面适用性需另测。
- **[ForeFly](https://arxiv.org/abs/2609.33581v1)**（B）：无人机 VLN 区分近端未来与路线关键未来，作为机制观察。
- **[Just-In-Time Agent Memory](https://arxiv.org/abs/2609.34385v1)**（C）：通用历史检索与查询时上下文构建，摘要未给具身导航证据。
- **[Normative Loss Landscape Navigation](https://arxiv.org/abs/2609.35926v1)**（C）：参数损失景观中的“导航”，不属于机器人空间导航。
- **[VehicleArena](https://arxiv.org/abs/2609.35916v1)**（C）：独立目标的多车驾驶及交通外部性评测。
- **[Pixels to Keys](https://arxiv.org/abs/2609.37907v2)**（C）：游戏视频动作反推，非地面机器人实验。
- **[DiffWAM](https://arxiv.org/abs/2609.39763v1)**（B）：无人机预测特征接轨迹、异步时间戳交接；见可迁移方法。
- **[OSWorld-Science](https://arxiv.org/abs/2609.39903v1)**（C）：科学软件电脑操作，非物理具身导航。
- **[Reward as Observation](https://arxiv.org/abs/2610.00729v1)**（C）：依赖奖励和动作历史进行迁移，不能直接接入通常没有密集奖励的部署导航。
- **[MEAN Movement and Compression](https://arxiv.org/abs/2610.02334v1)**（C）：移动、压缩和功率联合优化的通信模型，未给导航基准实验。
- **[LiDARFlow](https://arxiv.org/abs/2610.01573v1)**（C）：机载 LiDAR 势流式几何避障，无语言或语义导航机制。

其余次池方向：

- **飞行与纯控制（C），4 项**：[LQR-ArUco Fusion](https://arxiv.org/abs/2609.35700v1)、[AeroManip-VLA](https://arxiv.org/abs/2609.36915v1)、[Scene-Scale Aerial Manipulation](https://arxiv.org/abs/2609.39670v1)、[RL-Guided PAC-NMPC](https://arxiv.org/abs/2609.39854v1)。
- **驾驶与交通（C），9 项**：[CausalDriveBench](https://arxiv.org/abs/2609.32157v1)、[RCVLA](https://arxiv.org/abs/2609.32681v1)、[CAR-VLA](https://arxiv.org/abs/2609.34387v2)、[RefineDrive](https://arxiv.org/abs/2609.35078v1)、[Embodied Reasoning Interfaces](https://arxiv.org/abs/2609.34794v1)、[Class 8 Truck VLA](https://arxiv.org/abs/2609.38570v1)、[Speed in the Blind Spot](https://arxiv.org/abs/2609.37046v1)、[Vision-Language-Action Autonomous Driving Agent with Language-based Memory](https://arxiv.org/abs/2609.38641v1)、[VLALight](https://arxiv.org/abs/2609.36934v1)。
- **通用 Agent、游戏与资料综述（C），9 项**：[CueKFS](https://arxiv.org/abs/2609.31873v1)、[GameBoyWorlds](https://arxiv.org/abs/2609.32093v1)、[ExpVoyager](https://arxiv.org/abs/2609.32630v1)、[Reflect Reverse](https://arxiv.org/abs/2609.38536v1)、[AutoDataBench](https://arxiv.org/abs/2609.40097v1)、[EngramBench](https://arxiv.org/abs/2609.39284v1)、[JevSpawn](https://arxiv.org/abs/2610.00437v1)、[Coco](https://arxiv.org/abs/2610.02376v1)、[Evolutionary Computation Survey](https://arxiv.org/abs/2610.02996v1)。
- **形态、群体及医疗机器人（C），3 项**：[Bridging Body and Brain](https://arxiv.org/abs/2609.31329v1)、[Fish Schools](https://arxiv.org/abs/2609.35554v1)、[Endovascular BCI Navigation](https://arxiv.org/abs/2610.03537v1)。

### 5.5 资讯与非论文

- 本期未新增公众号文章或独立非论文资讯；Docker 未运行，手动公众号链接清单为空。

## 六、趋势判断与行动建议

### 趋势

- **训练监督与推理预算开始分开优化。** FINE 从未来地标提取监督，GeoScaffold 在训练时内化几何，EdgeVLN 在部署时测运行路径，NavJev 将每步生成改为结构化动作选择。本报告判断：需要同时报告训练新增资源和实际闭环成本。
- **记忆从记录轨迹转向带条件的决策证据。** MemTransfer 揭示起点与路径变化的影响，ECROM 校准观测机会，EvolvingNav 推进时变信念，终身导航 NavHarness 检查旧记录与新观察。本报告判断：后续评测应覆盖记忆失效与纠正，不能只测重复任务。
- **监控、取证与恢复可以作为独立研究对象。** SeekVLN 的主动取证、AVERT-VLN 的求助监控、CGPI 的更新回滚分别针对信息不足、执行偏离和错误适应。本报告判断：模块效果应在相同基础策略和相同额外预算下比较。
- **较大的具身模型成绩仍受执行与评测条件约束。** 本期 Astra 跨域评测的 RxR/HM3D 结果未在摘要中给完整 split 或 CE 口径，运动控制示例又暂停物理仿真等待推理；Embodied Agent Arena 与 VCN-Bench 也区分局部判断和最终任务完成。本报告判断：精准估计、正确目标定位和完整行动成功应分别检验。

### 研究空白

- **共同预算下，观察、全景和人工帮助各值多少？** PanoVLN 扩大视野，SeekVLN 增加取证，AVERT-VLN 引入人工纠错；缺少同一路线、相同感知/推理/人工成本下的对照。
- **地图或记忆出错时，系统能否自己发现并修复？** 主动纠正旧经验的代价、误纠正风险和最坏任务损失，尚未从摘要中的平均成绩得到解释。
- **板端延迟如何改变动态环境中的结果？** 需把监控、停止判定、取证和低层安全控制都纳入真实时钟，避免只在暂停仿真的条件下验证推理能力。

### 建议动作

**高优先级**

- **复现 EdgeVLN 的停止与部署对照**：先锁定完整 R2R-CE val-unseen、运行格式及统一硬件，记录 SR/SPL、错误停止、资源和总耗时。
- **精读 PACE 与 SeekVLN**：前者先核验子集定义及局部模块训练数据，后者先核验提升单位、绝对结果和主动取证成本，再决定做执行模块还是取证策略。
- **核验 PanoVLN 和 FINE 的资源条件**：检查全景获得方式、分岔路线数据以及未来监督是否引入额外专家信息；FINE 的低数据预算提升未在摘要中明确对应基准，暂不作定量横比。
- **复现终身导航 NavHarness 的经验交接**：统一任务序列、历史访问与重启预算，加入 MemTransfer 风格的换起点、阻断路径及无关记忆条件。

**中优先级**

- **跟踪 InsightMap、ECROM、EvolvingNav**：分别对应动作后的空间监督、观测机会校准和到达时刻信念，先验证单独模块，再考虑组合。
- **诊断 AVERT-VLN 监控器**：固定人工介入预算，拆开检测、恢复和离线训练的贡献。
- **核验两篇 NavHarness 的代码与数据归属**：终身导航工作是 2609.34276，自适应目标工作是 2609.39915；复现记录保留完整标题和 ID。
- **检查 ASENA 的协议**：条目使用 R2R/RxR 名称，未明确 CE；十轮反复访问的 100-task 子集成绩应与一次独立测试分开记录。

**低优先级**

- **观察 VLA 记忆、量化与恢复动态**：本期次池保持主题索引；只有明确能接入地面导航、且有对应闭环对照时再升级阅读。
- **暂缓以最高 SR 选型**：人工辅助、子集、全景、RGB-D、反复经验学习、不同模型及推理档位下的数字先核验条件。
- **暂缓纯控制和非具身“navigation”条目**：它们不构成本期语言/语义导航趋势的证据。
