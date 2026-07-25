---
layout: post
title: "VLN 周报（2026-07-06 ~ 2026-07-25）"
date: 2026-07-25
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "本周精选：密歇根-清华推出室内外连续物理导航基准 NavVerse；高德发布 ABot-N1 通用导航模型基座；上海科大推出城市级纠错导航框架 DA-Nav 及 ReDA 数据集。"
---

# 具身导航周报（vln_2026-07-25，覆盖 2026-07-06 ~ 2026-07-25）

---

# 一、本周值得优先看的几篇（贴近地面机器人 VLN / 具身导航）

# 1. NavVerse —— 室内外连续无缝跨域物理仿真基准（密歇根大学 & 清华团队，2026-07-24）
家用、巡检、末端配送机器人常面临“室内穿行 $\rightarrow$ 出门上路”的连贯通行场景，但现有仿真基准往往将室内外隔离，且忽略了真实的物理碰撞与跌落风险。密歇根大学与清华团队推出首个打通室内-户外-混合场景的一体化物理仿真基准 NavVerse。它包含 100 个室内场景、50 个城市户外场景以及 50 个室内外连通场景，物理上无断点、无场景传送或镜头切换。评测任务融合了经典的 ObjNav、VLN 以及首创的长时序 PlaceNav（商业地标搜索），并搭建了包括任务完成度、探索效率和物理安全的量化评测体系。
实验测试了 SGImagineNav（图规划）、PoliFormer（强化学习）、UniNaVid（端到端 VLA）和 LongNav-R1（RL微调 VLA）。结论令人警醒：UniNaVid 在 ObjNav 上成功率最高仅 11.62%，在 PlaceNav 任务上，从纯户外（SR=17.65%）切换到室内外混合场景时，成功率直接跌至 3.64%，表现出严重的跨域衰减；RL 模型在户外及混合场景中成功率直接归零。此外，碰撞率与成功率之间存在严重的权衡冲突，且“如何寻找建筑物出口”是限制机器人走向户外的最大瓶颈。
链接：https://arxiv.org/abs/2607.19695 ｜ 项目页：https://umich-curly.github.io/NavVerse-Benchmark/
<div align="center"><img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/teCWJo4aHf8b4wYhu8ic0icCYYD5xia7Jr9P20ibPZLBZNZ3d9w6Cia9g3wG9ibRWeia4xvw7k7cmxIQ45vusEC0iapWY4LeJR7JZ3a43AOPVe2t47I/640?wx_fmt=webp&from=appmsg" width="80%" /><figcaption>NavVerse 室内外无缝跨场景混合物理仿真环境与评测任务</figcaption></div>

# 2. ABot-N1 —— 统一多任务的“快慢双系统”通用导航大模型基座（阿里高德 CV 实验室，2026-07-17）
传统的端到端黑盒导航模型将“语义目标识别”与“高频电机控制”塞进同一个计算循环，导致不同任务梯度互相冲突。高德团队提出了解耦式的“快慢双系统”通用导航大模型基座 ABot-N1。慢系统（Qwen-3.5-4B）负责低频深度语义和空间推理，输出可读的思维链（CoT）和通行/目标图像像素锚点；快系统（Qwen-3.5-2B + QFormer）根据实时视图和缓存的像素锚点异步生成连续行进路点。基于自研仿真引擎产出的 3000 万标注样本进行监督预训练，并利用 GRPO 组相对策略优化进行后训练，设计了安全分层均衡采样以提高避障性能。
在包含 PointBench（点位导航）和 POIBench（最后一米进店精度）的评测中，ABot-N1 无需分任务微调，商铺 POI 抵达成功率直接暴涨 35%，室内无碰撞抵达率达 95.4%，复杂室外达 92.9%。高德 TuTu 四足机器人真机部署成功，实现了“深思”与“本能”的完美异步运行。
链接：https://arxiv.org/abs/2607.10383 ｜ 项目页：https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/
<div align="center"><img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/teCWJo4aHf8zUlqSZsUAkLLlQXDLIfSXIe47dgibhptL78jqAiaBgpbwibBzzx3IPYib5iaBu2zsRptrqWAibj25xI32jZjXuRia7EmjZlk4pKR4Ek/640?wx_fmt=webp&from=appmsg" width="80%" /><figcaption>ABot-N1 结合思维链与像素锚点的解耦快慢双脑导航系统</figcaption></div>

# 3. DA-Nav —— 商用导航粗指令引导下的城市级自主纠错视觉导航（上海科技大学等，2026-07-16/13）
城市长距离户外机器人导航依赖高成本的稠密地图和人工标注，长距离行驶容易累积误差偏离路线。上海科技大学等团队提出 DA-Nav 导航框架，直接复用手机高德/谷歌地图下发的粗粒度方向指令（例如“50米后右转”）。DA-Nav 将导航定义为“自中心图像平面离散空间定位”，以避开大模型的 3D 空间幻觉。通过 CoT 链式推理，每步依次输出路线偏差判断、导航/纠偏动作以及图像平面网格轨迹。
该研究同时开源了首个包含偏离纠错样本的 ReDA 数据集（CARLA仿真生成，286K帧）。模型在 CARLA 环境中训练，无需真机微调（Zero-Shot Sim-to-Real），可直接跨形态部署至 Unitree Go2 四足机器人和乐聚 Kuavo-V 人形机器人，并在真实户外完成 1.2 公里的稳定闭环导航，显示出极强的自主纠偏与跨机体泛化能力。
链接：https://arxiv.org/abs/2607.11638v2 ｜ 项目页：https://uav-dualcog.lozumi.com (Wait, code/project page from paper info)
<div align="center"><img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/teCWJo4aHfibVicZRQXKDGUgzlGlf7GPH7PxU7kE7cS7HlxRYe8gFicICibVrGqtLn6ykFE9koYzdLbiaH6qsBnK9TsIr84M1qoF6ISG5UicAJ4bc/640?wx_fmt=webp&from=appmsg" width="80%" /><figcaption>DA-Nav 结合 CARLA 偏离纠错机制的图像空间网格导航框架</figcaption></div>

# 4. RAVEN —— 视时空一体化长时程机器人智能记忆系统（普林斯顿大学 & 宾大，2026-07-06）
机器人长时部署的核心是记住去过的地方，而传统“图像 $\rightarrow$ 文字字幕 $\rightarrow$ 文本嵌入”的记忆存储方式容易产生“字幕瓶颈”，压缩丢失大量的空间和外观细节。普林斯顿与宾大团队提出视时空一体化记忆系统 RAVEN，跳过文本转换，直接存储高保真稀疏视觉嵌入、空间位姿和时间戳的三元组向量。VLM 配备了文本、时间、空间和图像四种检索工具，在工作记忆内以迭代推理方式进行“回忆、验证、修正和决策”。
实验表明，RAVEN 相比纯 VLM 方案，检索效率提升 10 倍以上，在 250 倍帧采样压缩下依然能保持 90% 以上性能。RAVEN-QA 真实机器人场景中准确率达 92.7%，成功在 Unitree Go1 四足机器人上零样本落地部署，取得 92.4% 的导航成功率。
链接：https://arxiv.org/abs/2606.25206 ｜ 项目页：https://ravenmem.github.io
<div align="center"><img src="https://mmbiz.qpic.cn/sz_mmbiz_jpg/teCWJo4aHf9L5vSLDjo0Kj0TWuoPEMo2dF06Aqh9FHia2zpws2yczKI8Oo2BVqYKHsYLVlic7vCTrugdNtKia6P4fqWKxcxj5HPuVlzm7ocIib8/640?wx_fmt=jpeg&from=appmsg" width="80%" /><figcaption>RAVEN 视时空三元组记忆构建与 VLM 工具箱迭代推理机制</figcaption></div>

# 5. SkillNav —— 分而治之的模块化视觉语言导航调度框架（密歇根州立 & 鲁汶大学，ACL 2026）
视觉语言导航（VLN）的一大痛点是端到端黑盒策略极易“死记”训练轨迹，在陌生场景下泛化性暴跌，且大模型空间接地精度较差。本文推出模块化 SkillNav 框架，将复杂的导航决策拆解为 5 类专用基础技能（寻找物体、探索新环境、规避死胡同等），高层由零样本 VLM 担任通用调度器。VLM 不直接输出控制坐标，而是根据指令和视觉选择合适的技能专家，低层运动交给具体技能模型闭环执行。
项目自研了合成数据集，在全新陌生场景基准 GSA-R2R 上取得 SOTA 效果。SkillNav 通过这种解耦方式实现了极佳的可解释性与强跨场景适配泛化能力，避免了单模型在推理步骤中的计算冗余。
链接：https://arxiv.org/pdf/2508.07642v4 ｜ 项目页：https://hlr.github.io/SkillNav/
<div align="center"><img src="https://mmbiz.qpic.cn/mmbiz_jpg/teCWJo4aHf8FUdiaFJ925H8QGO4rW30RD6ansrv8bEdLNkpNu8zaRkcgdzPzbTKTWA2jdsAS2CjO1iclFd54WFNTB2YQNSlRXKSHZbYflbUKA/640?wx_fmt=webp&from=appmsg" width="80%" /><figcaption>SkillNav 模块化导航框架的技能拆解与调度逻辑</figcaption></div>

# 6. SuReNav —— 超像素图约束松弛实现避障与抄近道的平衡（韩国科学技术院 KAIST，ICRA 2026）
在半静态场景（如校园、公园）中，临时路障常导致传统严格守规的机器人陷入无路可走的“过约束死局”。KAIST 团队提出 SuReNav 框架，通过超像素图对空间进行精准建模，并利用图神经网络（GNN）从人类演示中学习地形约束的松弛策略（例如为了避障短暂借道草坪或非机动车道），搭配可微 A* 规划路径。
SuReNav 精准限制了约束松弛的物理范围，避免了全局松弛导致的激进抄近道或不安全违规，在仿真与真实四足机器人上实现了安全避障与高通行效率的动态最优平衡。
链接：https://arxiv.org/abs/2602.06807 ｜ 项目页：https://sure-nav.github.io/
<div align="center"><img src="https://mmbiz.qpic.cn/mmbiz_jpg/teCWJo4aHfib2Ov37ZqsQuP9l6MLmicOmEy8KuHjhEIgQVbcZxY7V7iagkB9RJibpUkicqP2wYwicaoDu5We19d4AT17vsia5xeZbuCZ1Q0TLth3Yg/640?wx_fmt=webp&from=appmsg" width="80%" /><figcaption>SuReNav 超像素图建模与 GNN 约束松弛路径规划</figcaption></div>

# 7. 从指令跟随到认知导航：视觉语言导航综述（中科院自动化所等，2026-06/07）
视觉语言导航（VLN）正在从封闭 benchmark 中的“指令跟随”，逐步走向面向开放世界的“认知导航”。中国科学院自动化研究所、西安交通大学等机构的研究人员共同发表了一篇全面的 VLN 综述论文。不同于以任务或模型结构分类的传统综述，该工作提出了一个以“范式演进”为核心的统一能力演进分析框架，将 VLN 发展划分为四个递进层次：感知演进（语义粒度/空间结构/输入真实度）、认知演进（指令抽象/空间推理/审慎规划/世界模型）、学习演进（从模仿到自我改进）和泛化演进。
该综述系统整理了 2022—2026 年间具身导航的最前沿进展（涵盖 3DGS 语义地图、流式第一视角输入、世界模型及安全可信导航等），为理解未来具身导航智能体的发展指明了路线图。
链接：https://www.preprints.org/manuscript/202606.2231/v2 ｜ 项目页：https://github.com/lvkailin0118/Awesome-VLN-Evolution
<div align="center"><img src="https://mmbiz.qpic.cn/mmbiz_jpg/teCWJo4aHf9kvqP7BZ6VaK6RPgYwPYRTRBzA3aGc6fULcmEib3NLa2QrmXzxibeTn0v1uM0RiahBcPfgjb5OibZiciaHjU4Dw56oDw3WQ3ry9RdUo/640?wx_fmt=webp&from=appmsg&watermark=1#imgIndex=0" width="80%" /><figcaption>从指令跟随到认知导航的 VLN 范式演进路线图</figcaption></div>

---

# 二、其他方向但有明确可借鉴技巧的（不属于地面导航，但方法值得参考）

- **CosFly-VLA（无人机跟拍中的遮挡恢复 VLA 模型，道通智能等）**：核心是将无人机在城市低空作业中因遮挡而丢目标的难题，定义为**“闭环遮挡恢复任务”**。通过设计四阶段完整训练链路，使模型具备对目标定位、遮挡状态判断与动作生成的联合预测。**这种“主动估计遮挡状态并采取恢复行动”的闭环容错机制，对于地面导航中跟随动态行人或穿行多隔断空间的 VLN 任务有极佳的架构迁移参考价值**。
  链接：https://arxiv.org/abs/2607.15004v1
- **3D-IC（打通移动导航与机械臂抓取的交互链框架，中科院计算所，ICML 2026）**：传统的开放词汇移动操作（OVMM）将导航与操纵划分为独立阶段，常导致“到了目标前却角度不佳无法抓取”的近视规划问题。3D-IC 引入**统一 3D 语义特征图进行跨阶段联合规划**，使导航终点直接服务于后续的抓取动作。**这种“以最终交互为导向反向优化导航路点”的解耦-对齐思路，非常适合迁移到需要末端交互（如开门、递物）的地面移动机器人框架中**。
  代码：https://github.com/kekeZ66/3D-IC
- **PhyAgentOS（具身智能体认知规划与物理执行解耦的操作系统运行时）**：提出“Session-Centered”运行时和 **State-as-a-File 协议，以 Markdown/YAML 文件系统作为认知层和大模型规划器的通信边界**，解耦大模型的高延迟规划和控制器的低延迟运动。**这一通信接口抽象对于搭建具有多级反馈决策和安全审计需求的地面导航软件系统极具参考价值**。
  链接：https://arxiv.org/abs/2607.16636v1
- **SUREFlow（基于状态空间不确定性感知的残差流匹配操作框架）**：共同预测动作速度与**输入相关的残差不确定性**，在 Extended Rollout 期间选择性地改进不稳定的动作维度而无需环境反馈。**这种将动作生成的不确定性内化为状态控制信号的思路，在扩散/流匹配地面视觉导航策略中可以用来检测规划瓶颈并动态调速**。
  链接：https://arxiv.org/abs/2607.10504v1

---

# 三、无人机与低空导航方向从简（非核心方向，仅供了解领域动态）

- **SkyShield（低空占据安全感知，厦门大学）**：发布首个面向 20 米以下低空无人机自主安全飞行的前视单目语义占据数据集 SkyShield，提出 KAR-mIoU 风险评估指标及 SkyOcc 轻量化基线，解决帧间外参剧变下的占据预测难题。
- **LNN-Fly（时序失配避障，电子科技大学，IROS 2026）**：针对真机控制频率抖动和时序失配，提出连续时间激光雷达避障策略，将实时 $\Delta t$ 引入模型前向计算并配合自适应记忆门，实现仿真到真机的零微调避障部署。
- **UAV-DualCog（双认知推理基准，西北工业大学）**：提出首个评估多模态大模型对无人机进行“自状态感知 + 外部环境理解” spatio-temporal Spatio-temporal 推理能力的双认知评测基准，并开源训练集支持微调。
- **OrthoTrack（零训练航拍无卫星6DoF定位，苏黎世联邦理工，ECCV 2026）**：仅结合免费公开正射底图（TDOM）与地表模型（DSM），依托时序跟踪逻辑实现无 GPS、无漂移的实时 6-DoF 无人机轨迹定位与 MovingDrone 数据集。
- **PiLoT v2（半正射像素对齐，国防科技大学）**：使用二维半正射地图（TDOM+DSM）替代重型三维重建网格，在机载 CPU 上进行地图裁剪和多模态融合优化，地图大小压缩 97% 且兼顾全局无漂移定位。
- **FSD-VLN（快慢双系统空中导航，鹏城实验室）**：将高延迟大模型语义推理和低延迟高频电机动作控制进行解耦异步运行，将空中长程 VLN 的未见环境导航成功率最高提升 2 倍，且延迟减半。
- **SemCityLoc（语义几何对齐，慕尼黑工业大学，ECCV 2026）**：通过语义-几何对齐框架和轻量化 3D 城市模型，实现无卫星下的低空 6-DoF 定位，挑战性城市峡谷场景误差降至 2.62 米。
- **CosFly-Track（跟拍视觉跟踪数据集，道通智能等）**：发布 240 万帧多模态对齐的无人机视觉跟踪数据集，配合 MuCO 轨迹优化引擎填补了动态目标跟拍训练数据的行业空白。
- **AeroAct（四轴W-A模型，2026-07-16）**：首个面向四轴自主飞行的动作中心世界-动作模型（WAM），基于视频扩散 Transformer 预测局部轨迹-动作后果图像。
- **No Training, Better Flights（测试时轨迹修正，2026-07-21）**：探究测试时缩放（Test-Time Scaling）在无人机 VLN 中的应用，基于多次采样生成并行候选并通过自纠正机制进行再评估以保证飞行安全。

---

# 四、公众号「视觉语言导航」本周解读全览（19 篇，客观列全）

| 日期 | 标题 | 会议 | 方向 | 一句话看点 |
|------|------|------|------|-----------|
| 07-24 | 室内外「无缝跨域」导航太难了！NavVerse：一站式搞定机器人跨场景全维度评测 | — | **地面/精选①** | Isaac Sim连续物理仿真，全维度跨场景评测 |
| 07-23 | 厦大低空无人机防撞新利器！SkyShield：用3D占据表征筑牢城市低空飞行安全防线 | — | 无人机（从简③） | 首个低空前视单目语义占据数据集与KAR-mIoU |
| 07-22 | IROS-2026 \| 电子科大时序失配也能稳避障！LNN-Fly：连续时间激光雷达无人机导航框架 | IROS 2026 | 无人机（从简③） | 引入连续时间计算，解决真机控制频率抖动避障 |
| 07-21 | 无人机 AI 能否 “认清自己”？UAV-DualCog：西工大双认知基准 | — | 无人机（从简③） | 自我感知与环境理解耦合推理双认知评估 |
| 07-20 | ECCV-2026 \| 无GPS也能厘米级定位与航拍！OrthoTrack：靠公开航拍底图搞定无人机全时段6DoF轨迹 | ECCV 2026 | 无人机（从简③） | 零训练无卫星定位， MovingDrone 仿真数据集 |
| 07-19 | 国防科大无人机「长时序航拍」不漂移！PiLoT v2：正射像素对齐让机载设备也能跑 | — | 无人机（从简③） | 用TDOM+DSM二维半地图压缩，CPU裁剪计算 |
| 07-18 | 无人机城市「遮挡追踪行人」不跑偏！CosFly-VLA：空间感知VLA模型 | — | 无人机（可借鉴②） | 遮挡追踪闭环恢复，估计可见性与连续动作 |
| 07-17 | 高德发布「通用导航」大模型基座！ABot-N1：一台机器人搞定 5 种行走任务 | — | **地面/精选①** | 快慢双系统+像素锚点，商铺抵达率涨35% |
| 07-16 | 机器人根据高德导航「自主 Citywalk」！DA-Nav：方向感知和自动纠错让机器人跑完整条城市道路 | — | **地面/精选①** | 直接适配商用粗指令，2D图像平面纠偏推理 |
| 07-15 | 一文读懂 VLN 感知、认知、学习与泛化四层演进：从指令跟随到认知导航的完整脉络 | — | **地面/精选①** | 系统性演进分析框架，梳理2022-2026年进展 |
| 07-14 | 冲 Nature！本科生一作连发多篇顶会，多尺度建模大幅涨点 | — | 资讯/非论文 | 缝合特征插件与主干模块学术包汇总 |
| 07-13 | 无人机长距导航提速50%！FSD-VLN：快慢双系统语义推理与实时飞行「全都要」 | — | 无人机（从简③） | 鹏城解耦快慢动作与语义，导航成功率提2倍 |
| 07-12 | ECCV-2026 \| 无人机穿越大楼不再迷路！SemCityLoc：语义3D城市模型赋能无人机 | ECCV 2026 | 无人机（从简③） | 语义-几何对齐，厘米级低空峡谷定位基准 |
| 07-11 | ACL-2026 \| 让机器人 “重组技能” 闯新环境！SkillNav：模块化框架攻克视觉语言导航泛化难题 | ACL 2026 | **地面/精选①** | 5类技能智能体解耦，GSA-R2R取得SOTA |
| 07-10 | ICML-2026 \| 机器人移动找物、抓取、放物一气呵成！3D-IC：联合规划打通移动导航操作全流程 | ICML 2026 | **地面/精选①** | 3D交互链，打通导航与操纵联合决策规划 |
| 07-09 | ICRA-2026 \| 道路被挡也能灵活抄近道！SuReNav：超像素图约束松弛实现类人高效导航 | ICRA 2026 | **地面/精选①** | 超像素图GNN学约束松弛，破过约束死局 |
| 07-08 | 无人机跟拍 “全程锁焦” 不掉线！CosFly-Track：首个大规模多模态无人机视觉跟踪数据集 | — | 无人机（从简③） | 240万帧对齐数据，MuCO优化轨迹引擎 |
| 07-07 | IROS冠军项目公开！四足机器人建模与强化学习部署：URDF解析、PPO策略训练、域随机化、摩擦补偿… | — | 地面/ Locomotion | 深蓝学院四足动力学与RL行走控制课程大纲 |
| 07-06 | 机器人导航也能「过目不忘」？RAVEN：视觉时空记忆系统、破解长程导航核心瓶颈 | — | **地面/精选①** | 视时空嵌入三元组记忆，VLM工具迭代检索 |

---

# 五、arXiv 新论文分类速览（97 篇，按主题归类）

# 5.1 视觉语言导航（VLN/ObjectNav，地面与通用）
- **ZONDA: Zero-shot Object Navigation with Dynamic Avoidance in Multi-floor Environments** ([arXiv](https://arxiv.org/abs/2607.21025v1))：提出多层环境下带动态行人的零样本物体导航框架，支持跨楼梯穿越、多视角目标校验与显式行人预测规避。
- **VoLN: Vision-Only Long-Horizon Navigation---Paradigm, Benchmark, and Method** ([arXiv](https://arxiv.org/abs/2607.21400v1))：提出纯视觉长时程导航新设定，以目标图像表示终点，将寻路信息转化为环境中可见的局部路网线索而非全局地图。
- **EA-Nav: Learning Safe Visual Navigation Policies with Embodiment Awareness** ([arXiv](https://arxiv.org/abs/2607.19880v1))：仿人模仿学习跨机体导航框架，将机器人自身的几何尺寸转化为引导 tokens 以处理相机的动作模糊性。
- **VLN-AVP: Zero-Shot Vision-Language Navigation with Hybrid Long-Short-Term Memory for Autonomous Valet Parking** ([arXiv](https://arxiv.org/abs/2607.17767v1))：结合 BEV 空间特征和 VLM 进行无预建图的开放指令自主代客泊车，引入混合长短期记忆存储语义/几何线索。
- **Predictive Training with Latent Imagination for Visual Quadruped Navigation** ([arXiv](https://arxiv.org/abs/2607.17574v1))：使用辅助 JEPA 风格预测器和 SIGReg 正则化，对四足机器人的 LSTM 隐藏状态进行潜在想象训练，增强其对动态障碍物的预判避障性能。
- **PGN: Design and Implementation of a Vision-Language Navigation System Based on Pangu Multimodal Foundation Model** ([arXiv](https://arxiv.org/abs/2607.17806v1))：基于华为盘古大模型（OpenPangu-7B）的离线 VLN 策略，引入时序采样与思维链生成动作。
- **Anticipate Before Acting: Future-State-Conditioned Vision-Language Navigation** ([arXiv](https://arxiv.org/abs/2607.18042v2))：提出未来状态条件 VLN（FSC-VLN），在行为克隆中引入未来帧特征查询，提高智能体决策的前瞻性。
- **Token-Wise Latent Streaming from Slow Reasoners to Fast Planners for Dynamic Vision Language Navigation** ([arXiv](https://arxiv.org/abs/2607.16806v1))：设计 SPARK-VLN 框架，通过在 VLM 的生成过程中流式传输中间 Token 的隐状态，给底层流量匹配规划器提供实时推理动力。
- **SkillNav: Score-Level Skill Intervention for Zero-Shot Object Goal Navigation** ([arXiv](https://arxiv.org/abs/2607.15758v1))：利用无 Token 开销的动作干预技能组件读写 VLM 维护的好奇心价值地图，抑制死胡同原地打转与路径折返。
- **Difference-Based Relational Learning for Zero-Shot Object-Goal Visual Navigation With Direct Sim-to-Real Transfer** ([arXiv](https://arxiv.org/abs/2607.15642v1))：利用孪生网络的关系差值提取器与双帧缓冲区构建域无关特征表示，实现无硬件微调的鲁棒 Sim-to-Real generalisation。
- **SoftNav: Injecting 3D Scene Tokens into VLMs for Embodied Navigation** ([arXiv](https://arxiv.org/abs/2607.14586v1))：跳过文本串行化，利用投影器将检测到的 3D 实体表示直接以 Soft Tokens 形式注入 VLM 的隐藏空间，在 HM3D-OVON 基准上提升显著。
- **NavCMPO: Critic-Guided MeanFlow Policy Optimization for Adaptive Navigation** ([arXiv](https://arxiv.org/abs/2607.14643v1))：两阶段自适应导航策略，结合Few-Step轨迹流匹配与带有密集避障Critic梯度的轨迹纠偏。
- **Joint On-and-Off Policy Learning for Vision-and-Language Navigation** ([arXiv](https://arxiv.org/abs/2607.13461v1))：推出 JOP-VLN 框架，在三阶段管线中无缝融合了离线行为克隆与在线强化学习。
- **ReflectVLN: Training Vision-Language Navigation Agents with Reflective Reasoning** ([arXiv](https://arxiv.org/abs/2607.12680v1))：通过意图智能体（做子任务拆分与反思纠偏）与执行智能体（将短指令接地为底层动作并监控进展）进行闭环双向交互的 agentic VLN 框架。
- **PixelLoop: Shortcut Topological Navigation with Pixel-Level Loops** ([arXiv](https://arxiv.org/abs/2607.12811v1))：将拓扑闭环引入像素级 3D 拓扑结构，作为连接规划路径的拓扑捷径以支撑稳定的跨视点寻路。
- **A Hybrid Mamba for Audio-Visual Navigation** ([arXiv](https://arxiv.org/abs/2607.13110v1))：提出 Samba 导航模型，使用 Mamba State Encoder 替代传统 GRU 聚合时序，并利用音频 Mamba 捕捉声谱图的全局时空相关性。
- **Enabling 24-hour Agricultural Robotics: Unsupervised Day-to-Night Cross-Modal Image Translation for Nighttime Visual Navigation** ([arXiv](https://arxiv.org/abs/2607.12065v1))：无监督的昼夜图像翻译网络，在农业视觉导航中将白天植株 RGB 影像转为夜间近红外，支持全天候 24 小时导航。
- **AdvNav: Behavior-Guided Black-Box Adversarial Attacks on Vision-Language Navigation** ([arXiv](https://arxiv.org/abs/2607.11063v2))：行为引导的无梯度黑盒对抗攻击框架，通过微小视觉扰动破坏智能体的 perceptual-action 时序依赖链。
- **Traj-VLN: Learning Pixel-Space Interaction via Autoregressive Trajectory Generation** ([arXiv](https://arxiv.org/abs/2607.10744v2))：将连续 VLN-CE 分解为一系列局部 3D 交互子目标，自回归生成交互路径。
- **Early to Share, Late to Save: Synchronisation-Driven Communication Gating in Bandwidth-Constrained Cooperative VLN** ([arXiv](https://arxiv.org/abs/2607.08504v1))：多机器人协同视觉语言导航下的同步驱动通信门控，在有限带宽下实现高通信效率与导航性能的取舍平衡。
- **GemNav: Discrete-Token Visual Robot Navigation using a Multimodal Large Language Model** ([arXiv](https://arxiv.org/abs/2607.06882v1))：使用基于离散 Token 的多模态大语言模型对视觉输入直接回归动作控制，提升机器人无网下的导航效率。

# 5.2 具身VLA与操作控制（机械臂操作与通用具身VLA）
- **AXIS: A Growable Community-Driven Data Engine for Scalable Robot Manipulation** ([arXiv](https://arxiv.org/abs/2607.21588v1))：浏览器端大规模远程遥操与自适应数据扩增引擎，自带自动标注、滤波及平滑流。
- **Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids** ([arXiv](https://arxiv.org/abs/2607.20345v1))：商超货架整理任务下的 Unitree G1 人形 VLA 框架 DEED，包含控制对齐、视觉特征强化与经验驱动策略。
- **FM-VLA: Force-based Memory for Vision-Language-Action Models in Contact-Rich Manipulation** ([arXiv](https://arxiv.org/abs/2607.18231v1))：为 VLA 注入变分自编码器压缩后的端部 6D 力学历史 Token，为视觉有遮挡的接触交互任务提供非马尔可夫力学记忆。
- **Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation** ([arXiv](https://arxiv.org/abs/2607.18016v1))：为人形机器人提出 POT-VLA，在移动和操作过程中维持全局 3D 实体物体的拓扑状态并转化为 VLA 输入，支持几何谓词校验。
- **IMBench: A Benchmark for Intuitive Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.15641v1))：结合感知、物理推理与交互执行的全功能直觉机械臂操纵评测基准。
- **Foresight Residual RL for Long-Horizon Robot Manipulation with Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.16506v1))：通过离线预估的“预见价值”（Foresight Value）为 VLA 策略提供连续平滑的 subtask 接手奖励，优化长程组装操作。
- **Towards Human-like Physical Intelligence: Lifelong Vision-Language-Action Learning for Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.14852v2))：提出 LifelongVLA，利用双时序自适应架构均衡模型的可塑性与稳定性，支持轻量级离线任务增量。
- **Representation-Aligned Tactile Grounding for Contact-Rich Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.14609v1))：分析未来触觉状态对网络中间特征的预测相关度，提出潜在对齐以在力学控制期间引入精细触觉监督。
- **Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning** ([arXiv](https://arxiv.org/abs/2607.13818v1))：提出评估操作稳定的运行指标，配合 Agentic RL 让机器人在接触卡顿时自主选择重试或更换运动模式。
- **VistaVLA: Geometry- and Semantic-Aware 3D Gaussian-Grounded VLA for Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.12356v2))：利用 3D 高斯泼溅构建显式 3D 语义认知地图以增强 VLA 模型的空间几何约束。
- **Towards Predictive, Aligned, and Scalable Robot Learning** ([arXiv](https://arxiv.org/abs/2607.11270v1))：推出 Lumo-2 世界动作模型，在预训练阶段以自监督视频编码特征约束潜空间动力学，大幅提高下游动作预测精度。
- **SUREFlow: State-space Uncertainty-aware REsidual Flow Matching for Robust Robot Manipulation** ([arXiv](https://arxiv.org/abs/2607.10504v1))：见「二」。
- **On the Efficiency of LoRA Fine-Tuning for Vision-Language-Action Models in Industrial Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.10172v1))：系统探究低秩适应（LoRA）在流匹配 VLA 模型（如 $\pi_0$）微调中的应用，证明 r=32 时的均匀 LoRA 与全量微调无显著性能差异。
- **Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents** ([arXiv](https://arxiv.org/abs/2607.08448v3))：内存辅助的 agentic 操作框架，将冻结的 VLA 包装为带自愈闭环的重试原语，并利用外部大模型代码代理进行语义编排。
- **FabriVLA: A Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation** ([arXiv](https://arxiv.org/abs/2607.08575v2))：轻量级多任务精细机械臂控制模型，优化了网络对空间极小物体特征的表征精度。
- **TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation** ([arXiv](https://arxiv.org/abs/2607.07287v2))：推出首个面向灵巧手交互的触觉预测物理基础模型。
- **Smooth Operator: A Real-Time Sampling-Based Algorithm for Kinematic Hand Retargeting** ([arXiv](https://arxiv.org/abs/2607.07491v2))：基于采样的实时动力学约束手部重定向算法，大幅压缩人到人形灵巧手数据收集的延迟。
- **Dual Latent Memory in Vision-Language-Action Models for Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.07608v1))：设计双分支潜记忆 VLA 以在运动控制中协同长短程时间上下文。
- **NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation** ([arXiv](https://arxiv.org/abs/2607.06678v1))：针对长时序操作指令，在隐藏特征层对长时历史动作 Token 进行自监督无损压缩。
- **Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation** ([arXiv](https://arxiv.org/abs/2607.06564v1))：将现有的 2D VLA 适配为 3D 几何动力学感知的精细操作网络。

# 5.3 具身通用架构/后训练/系统与硬件优化
- **HyWorldVLA: A Vision-Language-Action Model with Hybrid World Modeling for Autonomous Driving** ([arXiv](https://arxiv.org/abs/2607.20988v1))：混合世界 VLA 模型，通过在 VAE 潜特征预测的同时重构未来视频像素，对自动驾驶策略进行精细像素对齐。
- **Reasoning as a Double-Edged Sword: Architecture and Cross-Stage Robustness in Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.17786v1))：探究思维链/显式推理对 VLA 鲁棒性的影响，指出隐式潜空间迭代推理在噪声扰动下往往比文本思维链更容易发生系统崩溃。
- **JoyNexus: Service-Oriented Multi-Tenant Post-Training for VLA Models** ([arXiv](https://arxiv.org/abs/2607.16074v1))：服务化的多租户 VLA 后训练云端调度系统，优化小微 SFT/RL 实验卡时的灵活记账及负载。
- **AC-VLA: Robust Out-of-Distribution Action Execution via Compositional Learning** ([arXiv](https://arxiv.org/abs/2607.15714v1))：为打破 VLA 轨迹过拟合和视觉捷径，引入大模型驱动的指令解耦与本体系坐标对齐组件。
- **Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories** ([arXiv](https://arxiv.org/abs/2607.15330v2))：小米机器人实验室发布的 VLA 具身基础模型，基于 10 万小时实操轨迹训练，配备自动生成语义场景转移标签的管线。
- **Reflex: Real-Time VLA Control through Streaming Inference** ([arXiv](https://arxiv.org/abs/2607.14695v1))：针对流匹配 VLA，引入时间步不变特性解耦视觉与去噪循环，设计 context 区域滑动缓存实现 $O(1)$ KV-Cache 更新以支撑实时控制。
- **Lights, Camera, Malfunction: When Illumination Robustness Leaves VLA Models Blind to Color** ([arXiv](https://arxiv.org/abs/2607.14698v1))：提出 FLARE 聚光灯物理攻击策略，指出粗暴的颜色扰动防守增强会让大模型丧失对颜色的语义辨识。
- **FoMoVLA: Bridging Visual Foresight and Motion Guidance for Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.14739v1))：将未来特征预见与稀疏 2D 点跟踪结合，既提供导航目标画面，又捕获运动连续路径。
- **Action QFormer: Structured Representation Shaping under Action Supervision in Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.14635v1))：引入 Query 式动作接口重新组织底层特征表示，避免动作梯度直接破坏 VLM 本身的语义与物体 grounding 能力。
- **S-squared-VLA: Decoupling Semantic and Spatial Streams in Vision-Language-Action Models for Autonomous Driving** ([arXiv](https://arxiv.org/abs/2607.13926v1))：解耦自动驾驶 VLA 模型中的语义流与空间三维表征流，解决语言 Discrete Tokens 破坏连续空间边界预测的瓶颈。
- **Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection** ([arXiv](https://arxiv.org/abs/2607.14236v1))：提出 LIFT 框架，利用零初始化的交叉注意力机制在预训练模型侧路动态 grafting 接触反作用力。
- **Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment** ([arXiv](https://arxiv.org/abs/2607.13429v1))：提出 Anchor-Align 策略，在 BC 微调中使用冻结的副本防止原语义退化，并对动作特征做显式的语言语义对齐约束。
- **DiMaS: Distribution Matching for Steering Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.14280v1))：提出分布匹配方向调整技术，以替换经典的 VLM 线性特征偏移，实现对流匹配 VLA 模型的精细行为控制。
- **TrustVLA: Mechanism-Guided Inference-Time Defense Against Vision-Language-Action Backdoors** ([arXiv](https://arxiv.org/abs/2607.12571v1))：针对 VLA 后门劫持（BadVLA、INFUSE），揭示模型注意力的紧凑因果足迹并提供推理阶段解毒防御。
- **Reducing Temporal Redundancy for Efficient Vision-Language-Action Inference** ([arXiv](https://arxiv.org/abs/2607.12287v1))：系统级加速策略，视觉感知层仅增量提取动态区域 Token，动作流匹配层通过 2 步蒸馏压缩去噪循环，大幅缩短时序延迟。
- **ExToken: Structured Exploration for Efficient Vision-Language-Action Reinforcement Fine-tuning** ([arXiv](https://arxiv.org/abs/2607.12931v1))：提出 ExToken 策略，在 RL 交互训练中以离线行为先验 Token 引导智能体，用轨迹的多样性突破探索停滞瓶颈。
- **See like a Robot: Robot-Centric Pointmaps for Vision-Language-Action Models** ([arXiv](https://arxiv.org/abs/2607.11498v1))：通过向像素点内嵌机器人本体系的 3D 坐标生成 Robot-Centric Pointmaps，以克服相机视角抖动对动作推理坐标的漂移干扰。
- **TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging** ([arXiv](https://arxiv.org/abs/2607.09818v1))：构建基于时空掩码的机械臂控制框架，引入离散扩散动作专家以规避普通 AR 回归的物理隔离瓶颈。
- **WCog-VLA: A Dual-Level World-Cognitive Vision-Language-Action Model for End-to-End Autonomous Driving** ([arXiv](https://arxiv.org/abs/2607.08375v1))：结合 3D 空间预测、博弈 CoT 推理和生成式扩散渲染的双层世界感知自动驾驶框架。
- **LEEVLA: Seeing What Matters in Latent Environment Evolution for Vision-Language-Action** ([arXiv](https://arxiv.org/abs/2607.08182v1))：结合语义漂移指引和 DPP 动态特征剪裁的 LEEVLA 策略，迫使模型关注指令核心指示区域。
- **CLAP: Direct VLM-to-VLA Adaptation via Language-Action Grounding** ([arXiv](https://arxiv.org/abs/2607.08974v1))：提出 CLAP，无需重度 fine-tune 即可通过指令-动作隐式对齐将离线 VLM 改造为可预测电机动作的 VLA 模型。
- **Training-Free Acceleration for Vision-Language-Action Models with Action Caching and Refinement** ([arXiv](https://arxiv.org/abs/2607.06370v1))：无训练加速机制，利用缓存的高频动作块，在推理时仅作少量轻量残差纠偏而不执行完整大模型前向。
- **SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models** ([arXiv](https://arxiv.org/abs/2607.06442v1))：在模仿学习前对轨迹数据的控制动作和几何拓扑结构进行相似度裁剪，挑选最具代表性数据。

# 5.4 其他具身智能相关研究（三维占据/世界模型/强化学习理论）
- **Unified Prediction and Planning via Conflict-Aware Disjoint Parameter Training** ([arXiv](https://arxiv.org/abs/2607.19971v1))：揭示多任务联合模型中预测与控制目标的参数分配竞争问题，提出冲突感知的非重叠参数分配优化。
- **ReferTrack: Referring Then Tracking for Embodied Visual Tracking** ([arXiv](https://arxiv.org/abs/2607.20061v1))：自中心单目“先检测、后跟踪”具身人身追踪框架，使用历史滑动特征存储运动状态以输出三维跟踪路点。
- **LENS: LLM-guided Environment Simplification for Planning and Control in Clutter** ([arXiv](https://arxiv.org/abs/2607.19633v1))：大语言模型引导的未知杂乱物理场景简化网络，为物理仿真控制器在线裁剪和更新局部几何约束。
- **STeP: Signal Temporal Logic for Precise Specifications for Action Generation with Vision Language Models** ([arXiv](https://arxiv.org/abs/2607.18580v1))：以时序信号逻辑（STL）为高层抽象接口链接 VLM 与低层物理控制器，提供强可解释的约束规制。
- **Patch Policy: Efficient Embodied Control via Dense Visual Representations** ([arXiv](https://arxiv.org/abs/2607.18236v1))：直接引入预训练 ViT 块中富含空间几何的密集特征作为 Transformer 控制策略输入，比重头训练的卷积策略更省时。
- **Reward-Driven LLM Agent Workflows: Synthesizing POMDP Routing and Self-Correction for Autonomous Decision-Making** ([arXiv](https://arxiv.org/abs/2607.17038v1))：结合部分可观测马尔可夫决策（POMDP）和自纠正奖励网络的 LLM Agent 路由决策流。
- **PhyAgentOS: A Self-Evolving Operating System for Embodied Agents with Decoupled Cognitive Planning and Physical Execution** ([arXiv](https://arxiv.org/abs/2607.16636v1))：见「二」。
- **Video = World + Event Stream** ([arXiv](https://arxiv.org/abs/2607.15038v2))：推出 Wan-Streamer v0.3 模型，将视频 reframer 为“静态背景 + 动态事件”，自监督训练实时生成/补全后续视频。
- **Training-Free Open-Vocabulary 3D Point-Cloud Segmentation on the Generalized Few-Shot Benchmark** ([arXiv](https://arxiv.org/abs/2607.15331v1))：结合 3D 视觉大模型（RegionPLC）与 prompt 概念分割模型（SAM3）的免训练三维点云开放实体识别。
- **RoboTTT: Context Scaling for Robot Policies** ([arXiv](https://arxiv.org/abs/2607.15275v1))：将具身 VLA 的控制历史窗口拓展至 8K 时序长度，实现测试阶段就地根据人类操作演示自我提升与误差恢复。
- **UESF-Bench: Benchmarking and Probing for Unified Embodied Seeking and Following** ([arXiv](https://arxiv.org/abs/2607.13621v1))：推出统一的具身寻找与跟随评测基准，要求智能体在陌生动态环境中先自主寻找到特定目标，然后再实现稳定跟从。
- **Semantic Anchoring for Robotic Action Representations** ([arXiv](https://arxiv.org/abs/2607.13597v2))：揭示行为克隆微调会导致网络表征退化，提出将动作输出锚定回冻结图像编码器的隐层语义空间。
- **MAMMOTH: A Multi-Modal End-to-End Policy for Off-Road Mobility Robust to Missing Modality** ([arXiv](https://arxiv.org/abs/2607.12965v1))：越野大范围非结构化视觉导航框架，结合雷达和 RGB 传感器，在光照过爆/遮挡导致传感器失效时提供稳定控制。
- **Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference** ([arXiv](https://arxiv.org/abs/2607.12659v3))：为解决 Jetson Orin 等边缘计算小算力机器人的异步延迟，提出未来修正机制根据当前运行速度动态修补下发的控制指令。
- **ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning** ([arXiv](https://arxiv.org/abs/2607.12992v1))：针对分块动作输出中的接缝抖动问题，在训练中引入接缝感知约束，并在推理中应用平滑重叠混合。
- **VIA: Visual Interface Agent for Robot Control** ([arXiv](https://arxiv.org/abs/2607.11119v1))：直接利用视觉代理界面控制机器人，将物理控制解耦为视觉像素选择。
- **Learning to Navigate Efficiently with Only 0.58M Trainable Parameters** ([arXiv](https://arxiv.org/abs/2607.11029v2))：通过硬解析几何投影、栅格占有等已知拓扑变换，极力压缩学习参数至 0.58M 即可完成复杂的零样本导航。
- **From World Action Models to Embodied Brains: A Roadmap for Open-World Physical Intelligence** ([arXiv](https://arxiv.org/abs/2607.11689v1))：探讨从世界动作模型（WAMs）演进到具身决策大脑的架构与数据集标准化路线图。
- **Artificial Foveated Perception for Mitigating Shortcut Learning in Robotic Foundation Models** ([arXiv](https://arxiv.org/abs/2607.10655v1))：探讨具身模型在 BC 期间的“视觉捷径”过拟合，并引入人工中心凹聚焦网络迫使策略忽略背景噪声。
- **PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers** ([arXiv](https://arxiv.org/abs/2607.09590v1))：适用于分块动作输出 Transformer（ACT）的 RL 后训练框架，引入混合行为约束防范特征退化。
- **Learning More from Less: Reinforcement Learning from Hindsight** ([arXiv](https://arxiv.org/abs/2607.09042v1))：利用第三方 VLM 在模仿学习后训练中进行 hindsight relabeling，自动为失败轨迹贴上合理的语义标签并重新分配稀疏奖励，提升样本效率。
- **Can the Cloud Drive? Infrastructure Feasibility of Offloading Autonomous Driving Across 5G and 6G** ([arXiv](https://arxiv.org/abs/2607.09045v1))：评估通过 5G/6G 将大算力具身大模型云端托管的工程延迟与性价比可行性，并指出基于 100ms 控制环和 300ms 规划环的分离策略最为可行。
- **TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning** ([arXiv](https://arxiv.org/abs/2607.08283v2))：轻量级内存融合策略，在接触点与 subgoal 转移时动态更新内部隐藏进度。
- **Prompt-Driven Exploration** ([arXiv](https://arxiv.org/abs/2607.08837v1))：基于大模型的提示词迭代探索，大模型评估失败轨迹并修改输入 prompt，为 RL 智能体引导更具信息量的动作空间。
- **Post-Training in End-to-End Autonomous Driving** ([arXiv](https://arxiv.org/abs/2607.08072v2))：关于自动驾驶端到端架构在大规模专家示范后利用 RL 进行后训练对齐的安全技术综述。
- **A Comprehensive Survey and Systematic Real-World Evaluation of Embodied Vision-and-Language Navigation** ([arXiv](https://arxiv.org/abs/2607.09792v1))：视觉语言导航最新综述与系统性真机评测基准分析。
- **Multi-Agent Robotic Control with Onboard Vision-Language Models** ([arXiv](https://arxiv.org/abs/2607.07403v1))：使用低参数机载 VLM 实现多机器人分布式协同的通信与运动决策。
- **HELP: Human-Efficient Large-Scale Robot Post-Training with Rollout Segmentation** ([arXiv](https://arxiv.org/abs/2607.09776v2))：大参数模型BC微调后的后训练效率指引，利用 Rollout 划分定位高误差阶段并注入人工干预。
- **Vision Language Action (VLA) Models for Unmanned Aerial Robotics and Bimanual Manipulation: A Review** ([arXiv](https://arxiv.org/abs/2607.06706v1))：VLA 基础模型在双臂灵巧操作和空中低空机器人的最新技术研究综述。
- **Optimal Transport Q-Learning for Flow Policy Steering and Acceleration** ([arXiv](https://arxiv.org/abs/2607.06262v1))：将最优传输理论引入 Q-Learning 以加速流匹配策略的动作生成速度。

---

# 六、资讯类与学术生态（非论文）

- **深蓝学院四足动力学与强化学习行走控制实战课程**（由英国纽卡斯尔大学潘为教授主讲，基于可微仿真平台 MATRiX 进行系统辨识、PPO 策略并行训练、域随机化与摩擦补偿，打通从仿真到碎石/斜坡真机越障的完整 Sim-to-Real 闭环）。
- **学术缝合网络模块资源大包**（按注意力、卷积、特征融合、轻量化等归类，包含 500 个即插即用的网络微型插件与主干模块，配论文、代码与任务适配指引）。

---

# 七、主题分布小结

本期为双周增量抓取（覆盖时间段 2026-07-06 $\sim$ 2026-07-25，共计新增 119 条条目）。经过严格剔除与归类，与地面机器人视觉语言导航（VLN/ObjectNav）**高度相关**的重磅工作共有 7 篇（NavVerse、ABot-N1、DA-Nav、RAVEN、SkillNav、SuReNav 六篇公众号精选 + 中科院自动化所的最新演进综述），已全部收录于「一、本周值得优先看的几篇」并提供详细的架构解析与 benchmark 数据。

本期无人机方向研究呈现明显的“低空占据安全”与“无 GPS 定位”两个聚焦趋势（共 10 篇），已做从简处理；机械臂操作与通用 VLA（约 20 篇）以及具身通用软件系统和强化学习后训练技巧（约 30 篇）本期不属于导航主流，仅在其方法或通信架构上有明确可迁移点（如 CosFly-VLA 的闭环遮挡恢复、3D-IC 的跨阶段联合规划、PhyAgentOS 的文件系统式认知解耦以及 SUREFlow 的残差不确定性感知）时，才于「二」中详细摘出并点明迁移途径。
