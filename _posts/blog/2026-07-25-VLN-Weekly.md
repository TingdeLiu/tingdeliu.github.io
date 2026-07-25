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

# 具身导航周报（2026-07-25，覆盖 2026-07-06 ~ 2026-07-25）

---

# 一、本期结论

- **室内外连续过渡成为地面导航新痛点**：由 NavVerse 等工作表明，传统在纯室内或纯室外独立优化的算法在面对建筑大门、光照和尺度突变的跨域场景时，任务成功率出现显著滑坡。
- **慢速语义规划与快速控制模块的异步解耦趋向主流**：ABot-N1 与 FSD-VLN 等框架均采用双脑（双系统）架构，大模型进行低频的思维链推理或动作指示，轻量控制器执行高频避障控制，以解决计算延迟与控制频率不匹配的瓶颈。
- **图像平面离散空间定位替代连续 3D 回归**：DA-Nav 等前沿方法将三维航点预测转换为自中心图像平面的二维离散网格选取，使导航动作与视觉语言模型（VLM）的二维空间推理本能对齐，从而降低空间幻觉并提高了控制稳定性。

---

# 二、优先阅读清单

| 优先级 | 工作 | 任务/场景 | 核心贡献 | 关键证据 | 阅读理由 |
|---|---|---|---|---|---|
| A1 | [NavVerse](https://arxiv.org/abs/2607.19695) | 室内外无缝过渡导航基准 | 搭建连续无缝物理仿真平台并量化碰撞及跨域退化 | 混合场景 PlaceNav 任务成功率降至 3.64%，碰撞率与效率呈显著权衡 | 地面机器人跨场景部署的首个一站式 Isaac Sim 物理评测基准 |
| A2 | [ABot-N1](https://arxiv.org/abs/2607.10383) | 通用导航大模型基座 | “快慢双系统”解耦语义规划与运动控制，像素锚点对齐 | POIBench 抵达成功率相比基线提高 35%，真机无碰撞率超 92.9% | 高德首个打通POI/点位/指令/物体/行人五类任务的通用导航框架 |
| A3 | [DA-Nav](https://arxiv.org/abs/2607.11638v2) | 城市级粗指令引导自纠偏导航 | 基于自中心图像网格预测动作与 CoT 显式纠偏，首创 ReDA 数据集 | 零微调部署于 Go2 与 Kuavo-V，完成真实户外 1.2 公里闭环导航 | 探究如何直接复用商业地图粗方向指令并在运动偏离时自主纠偏 |
| A4 | [RAVEN](https://arxiv.org/abs/2606.25206) | 视时空记忆数据库检索 | 构建视觉嵌入+位姿+时间戳的稀疏记忆库，设计工具化 VLM 迭代检索 | 检索仅消耗 7.43% 帧，250倍采样压缩下仍保留 90% 以上性能，真机 SR 达 92.4% | 提供了一种无需文本字幕压缩、高保真且计算低开销的机器人记忆方案 |
| A5 | [SkillNav](https://arxiv.org/pdf/2508.07642v4) | 模块化导航技能智能体 | 解耦为 5 大基础导航技能并由零样本 VLM 担任高层激活调度器 | 在 GSA-R2R 基准上取得优于传统端到端和纯 VLM 的泛化效果 | 解决大模型空间接地差与端到端网络盲目记忆训练轨迹的缺陷 |
| A6 | [SuReNav](https://arxiv.org/abs/2602.06807) | 半静态场景路障避障规划 | 基于超像素图建模障碍边界，利用 GNN 从人导数据学约束松弛 | 仿真与真实四足机器人测试中有效松弛规则通过路障（未见具体数字） | 解决过约束场景下严格遵守规则导致导航无解的“死局” |
| A7 | [一文读懂 VLN 演进](https://www.preprints.org/manuscript/202606.2231/v2) | 视觉语言导航范式综述 | 提出感知、认知、学习、泛化四维演进框架系统梳理 2022-2026年进展 | 覆盖 3D 拓扑、流式视频、世界模型、安全可信导航等前沿工作（无特定实验数值） | 全面理解 VLN 从封闭“指令跟随”走向开放“认知导航”的技术脉络 |

---

# 三、重点工作分析

## 1. NavVerse：一站式搞定机器人跨场景全维度物理评测

| 维度 | 分析 |
|---|---|
| 问题 | 现有导航基准将室内（离散、无碰撞动力学）与室外（离线街景、无运动控制）割裂，忽略真实碰撞和跌落动力学，无法系统评测跨边界过渡及 PlaceNav（城市地标大范围检索）性能。 |
| 方法 | 基于 NVIDIA Isaac Sim 搭建包含 100 室内、50 户外、50 连通混合场景的平台，机器人无需镜头传送连续通行。构建 ObjNav、VLN 以及长程地标搜索 PlaceNav，评测 SR（成功率）、SPL（路径加权成功率）、CE（覆盖效率）、CR（碰撞率）、NSR（可通行路面占比）等物理安全指标。 |
| 证据 | 大量轨迹受挫于出口识别。UniNaVid (VLA) 的 ObjNav 成功率最高仅 11.62%，在 PlaceNav 任务上，从纯户外（SR=17.65%）切换到混合场景时成功率骤降至 3.64%（降幅达 14.01%）；PoliFormer (RL) 离开室内后户外成功率直接归零。Spot 四足底盘成功率为 100% 时，轮式底盘由于路沿碰撞降至 47.5%。 |
| 价值 | 首次用物理引擎量化了“成功率、通行效率、运行安全”三者之间无法调和的取舍矛盾，明确将“建筑出口定位”定位为当下端到端算法的主要痛点。 |
| 局限 | 当前场景仅限于单层建筑，没有动态行人与车辆，且仅支持室内至户外的单向评估。 |
| 建议 | 建议任何从事户外机器人落地、多形态机体适配的研究小组在算法评估阶段加入该基准的离线诊断。 |

## 2. ABot-N1：通过快慢系统与像素锚点统一多行走场景

| 维度 | 分析 |
|---|---|
| 问题 | 传统的端到端黑盒策略将复杂的语义寻找与高频运动控制揉合，多任务梯度梯度冲突导致坐标跑偏漂移、难以调试，且对禁行区域（如机动车道）无修正能力。 |
| 方法 | 解耦为慢速推理大脑（Qwen-3.5-4B）与高速控制专家（Qwen-3.5-2B + QFormer）。慢系统输出可读思维链与 2D 通行/目标像素锚点；快系统接收锚点蒸馏特征并以 10Hz 输出 SE(2) 运动路点。基于自研 NSsim 重建引擎的 3000万样本预训练并结合 GRPO 强化学习以三层安全奖励约束轨迹。 |
| 证据 | 在自研 PointBench 和 POIBench 测试中，POI 抵达率提升 35%；室内无碰撞抵达率达 95.4%，复杂室外达 92.9%。部署于 TuTu 四足机器人进行实景测试。 |
| 价值 | 论证了利用 2D 图像平面上的“像素锚点”作为高层决策与底层控制的通信中介，是解决大模型控制频率错配与黑盒不可靠的有效中间态。 |
| 局限 | 推理骨干参数量共约 6B，且基于多目相机输入，在有限的嵌入式边缘芯片上部署开销仍然较高。 |
| 建议 | 建议关注并借鉴其基于 2D 像素锚点对齐的慢脑-快脑解耦架构，作为解决高延迟决策与高频避障冲突的开发组件。 |

## 3. DA-Nav：粗指令引导与离线纠错训练的长距离城市导航

| 维度 | 分析 |
|---|---|
| 问题 | 传统 VLN 需要专家高精度轨迹标注及高成本稠密建图，在城市长距离导航中极易累积运动偏差，模型因缺乏偏离路线后的恢复样本导致轨迹崩塌。 |
| 方法 | 输入粗粒度离散方向指令，使用 CoT 显式判断机器人是否偏离路线；将输出建模为自中心图像二维网格的离散选择，以贴合大模型本能。设计 ReDA 户外导航数据集，专门混合了 158K 标准专家帧与 128K 偏离纠错帧。 |
| 证据 | CARLA 仿真训练后直接零微调（Zero-Shot Sim-to-Real）部署至 Unitree Go2 四足机器人和乐聚 Kuavo-V 人形机器人，在真实户外完成 1.2 公里的稳定闭环导航。 |
| 价值 | 验证了无需 3D 全局高精地图与重度人工精细标注，仅利用低成本离散网格轨迹自回归及离线纠偏数据集训练即可实现大范围城市场景下的轨迹自愈。 |
| 局限 | 对于图像平面 2D 网格规划精度依赖相机的画面朝向，快速偏航或镜头晃动时容易发生纠偏失效。 |
| 建议 | 该工作的 ReDA 偏离纠错样本构建范式和图像 2D 网格运动表示直接适用于需要提升抗噪鲁棒性的地面长程视觉导航方案。 |

## 4. RAVEN：高保真视时空稀疏记忆系统

| 维度 | 分析 |
|---|---|
| 问题 | 传统记忆系统采用“视觉转字幕文本”的有损压缩导致空间、纹理等关键细节丢失（字幕瓶颈），而长视频 VLM 检索的计算成本又面临二次方膨胀。 |
| 方法 | 跳过文本，将帧图像多模态视觉嵌入、空间 3D 位姿、时间戳直接绑定为三元组存入向量记忆库。VLM 工作记忆模块搭配文本、时间、空间和图像检索四类工具，以迭代推理闭环进行回忆、检索和决策。 |
| 证据 | 在 RAVEN-QA 仿真与实机基准中，QQMM-v2 + Gemini-3-Pro 准确率达 92.7%；相比纯 VLM 方案，检索效率提升 10 倍以上，每次检索仅使用 7.43% 帧；250 倍帧采样压缩下保留 90% 性能，Unitree Go1 真机 SR 达 92.4%。 |
| 价值 | 证明了无需过早将具身机器人的视觉观察抽象为自然语言，使用高保真稀疏视觉嵌入结合向量检索，对长程多任务推理是更轻量且更精确的选择。 |
| 局限 | 检索的对齐质量强依赖于预训练多模态编码器的泛化边界；对高频变化的动态障碍物场景，记忆的实时覆盖更新尚未处理。 |
| 建议 | 建议精读和复现该框架的三元组 FAISS 向量检索库和迭代式工作记忆推理闭环。 |

---

# 四、可迁移方法

| 来源方向 | 工作 | 可迁移机制 | 可接入地面导航的位置 | 风险/前提 |
|---|---|---|---|---|
| 无人机 VLA / 目标跟踪 | [CosFly-VLA](https://arxiv.org/pdf/2607.15004) | 遮挡追踪定义为“闭环恢复任务”，估计遮挡可见性并联合预测动作 | 地面跟随型 VLN 中的目标识别与动作恢复决策器 | 需要高精度目标位置与可见性标注训练 |
| 具身移动操作 (OVMM) | [3D-IC](https://github.com/kekeZ66/3D-IC) | 采用统一的 3D 语义交互特征图对导航终点与操纵路径进行联合规划 | 目标交互驱动型底座寻路（如开门、递物）的终点价值评估器 | 维护稠密 3D 语义交互图的内存与算力开销较高 |
| 具身智能运行基础 | [PhyAgentOS](https://arxiv.org/abs/2607.16636v1) | State-as-a-File 协议，以 Markdown/YAML 格式作为慢脑规划与快脑控制器的通信抽象 | 分布式导航决策流中的跨硬件通信接口层 | 文件读写带来的时空开销（适用于低频规划回路） |
| 机械臂操作控制 | [SUREFlow](https://arxiv.org/abs/2607.10504v1) | 动作流预测结合残差不确定性感知，仅在此维度上执行反向迭代优化 | 基于流匹配/扩散的连续视觉导航底座运动控制器 | 需要有监督或自监督预测动作不确定性 |

---

# 五、分类速览

## 5.1 地面 VLN / ObjectNav / 语义导航

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| NavVerse | 室内外跨场景导航评测 | 见重点分析 | A | [arXiv](https://arxiv.org/abs/2607.19695) |
| ABot-N1 | 通用解耦导航模型基座 | 见重点分析 | A | [arXiv](https://arxiv.org/abs/2607.10383) |
| DA-Nav | 城市级离散指令纠偏导航 | 见重点分析 | A | [arXiv](https://arxiv.org/abs/2607.11638v2) |
| RAVEN | 视时空记忆导航检索 | 见重点分析 | A | [arXiv](https://arxiv.org/abs/2606.25206) |
| SkillNav | 模块化导航技能智能体 | 见重点分析 | A | [arXiv](https://arxiv.org/pdf/2508.07642v4) |
| SuReNav | 超像素图约束松弛规划 | 见重点分析 | A | [arXiv](https://arxiv.org/abs/2602.06807) |
| 一文读懂 VLN 演进 | 具身导航能力范式综述 | 见重点分析 | A | [Preprint](https://www.preprints.org/manuscript/202606.2231/v2) |
| ZONDA | 跨多层 object 寻找避障 | Heuristic 地面高度差建图配合 VLM 目标校验应对跨楼层及行人规避 | A | [arXiv](https://arxiv.org/abs/2607.21025v1) |
| VoLN | 纯视觉长程拓扑规划 | 避开全局高精定位，将寻路信息转化为环境中可见的局部路网特征 | A | [arXiv](https://arxiv.org/abs/2607.21400v1) |
| Difference-Based Relational Learning | 关系差异零样本迁移 | 提取目标物与局部图像的孪生差异特征构建域无关特征用于 ObjectNav | A | [arXiv](https://arxiv.org/abs/2607.15642v1) |
| SoftNav | 三维实体 Tokens 注入 | 用投影器将 3D 物体与前沿边界直接以 tokens 投影送入 VLM 推理空间 | A | [arXiv](https://arxiv.org/abs/2607.14586v1) |
| NavCMPO | Critic引导均值流策略优化 | 基于Few-Step生成辅以点云 Critic 避障梯度优化避障控制精度 | A | [arXiv](https://arxiv.org/abs/2607.14643v1) |
| Joint On-and-Off Policy | 模仿学习与强化学习结合 | 三阶段训练管线，无缝融合行为克隆、DAgger 算法与 RL 探路 | A | [arXiv](https://arxiv.org/abs/2607.13461v1) |
| ReflectVLN | 双 agent 意图执行对齐 | 意图智能体（ subtask 规划）与执行智能体（短时序避障控制）对齐纠偏 | A | [arXiv](https://arxiv.org/abs/2607.12680v1) |
| A Hybrid Mamba (Samba) | 视听多模态时空导航 | 引入 Mamba 状态编码器处理视听时间序列，克服传统 GRU 的限制 | A | [arXiv](https://arxiv.org/abs/2607.13110v1) |
| Agricultural Robotics | 农业大棚夜间图像翻译 | 无监督对齐日间 RGB 与夜间 NIR 植株图像，实现 24h 夜间自主导航 | A | [arXiv](https://arxiv.org/abs/2607.12065v1) |
| AdvNav | 时序动作对抗视觉攻击 | 探究在多步 perception-action 闭环中通过图像扰动干扰决策的机制 | A | [arXiv](https://arxiv.org/abs/2607.11063v2) |
| Traj-VLN | 离散二维平面航点预测 | 将 VLN-CE 任务切分为在自中心二维平面图像上的自回归轨迹路径生成 | A | [arXiv](https://arxiv.org/abs/2607.10744v2) |
| Early to Share | 受限带宽协同通讯决策 | 同步驱动下的通讯门控机制优化多机器人合作视觉语言寻路表现 | A | [arXiv](https://arxiv.org/abs/2607.08504v1) |
| GemNav | 离散大模型动作预测 | 将移动底盘控制条件化，通过大语言模型直接回归离散化运动 Token | A | [arXiv](https://arxiv.org/abs/2607.06882v1) |
| Comprehensive Survey Real-World | 真实世界 VLN 评估综述 | 系统回顾 VLN 具身控制的发展现状并在多种机器人形态上真实测试 | A | [arXiv](https://arxiv.org/abs/2607.09792v1) |

## 5.2 记忆、地图、规划与评测

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| LENS | 杂乱场景物理规划简化 | 大模型引导的在线环境剪裁，去除 distractor 保证避障控制器专注 | B | [arXiv](https://arxiv.org/abs/2607.19633v1) |
| STeP | 时序逻辑语言指示约束 | 利用 Signal Temporal Logic 将高层语言指令转化为低层控制硬约束 | B | [arXiv](https://arxiv.org/abs/2607.18580v1) |
| Patch Policy | 密集图像表征控制策略 | 跳过 VLM 骨干开销，提取预训练 ViT 密集 patch 特征供机器人训练 | B | [arXiv](https://arxiv.org/abs/2607.18236v1) |
| Reward-Driven LLM Workflows | POMDP 启发决策路由 | 结合部分可观测马尔可夫决策（POMDP）和自纠正奖励网络优化决策流 | B | [arXiv](https://arxiv.org/abs/2607.17038v1) |
| PhyAgentOS | 具身解耦操作系统 | 见重点分析 / 见“四” | B | [arXiv](https://arxiv.org/abs/2607.16636v1) |
| Video = World + Event Stream | 视频流式背景生成 | Wan-Streamer 将视频解耦为静态背景世界与动态事件流自监督学习 | B | [arXiv](https://arxiv.org/abs/2607.15038v2) |
| 3D Point-Cloud Segmentation | 开放词汇点云无监督分割 | RegionPLC 结合 SAM3 进行跨视角一致性校验的免训练点云分类 | B | [arXiv](https://arxiv.org/abs/2607.15331v1) |
| RoboTTT | 时序上下文扩展 TTT 学习 | 采用测试时训练（Test-Time Training）将动作历史扩展至 8K 帧 | B | [arXiv](https://arxiv.org/abs/2607.15275v1) |
| See like a Robot | 机器人坐标 Pointmap 表征 | 像素存储机器人本体系坐标，削减视角漂移对底座决策的影响 | B | [arXiv](https://arxiv.org/abs/2607.11498v1) |
| From WAMs to Embodied Brains | 世界动作模型演进路线 | 从 WAM 框架与自监督预测未来物理转移探讨物理具身的落地阻碍 | B | [arXiv](https://arxiv.org/abs/2607.11689v1) |
| Artificial Foveated Perception | 中心凹视场感知防捷径 | 模仿中心凹视力引入动态视网膜剪裁，迫使模型忽略无关环境背景 | B | [arXiv](https://arxiv.org/abs/2607.10655v1) |
| Optimal Transport Q-Learning | 最优传输流控动作加速 | 最优传输理论指导扩散流 matching 策略以加快连续规划动作生成 | B | [arXiv](https://arxiv.org/abs/2607.06262v1) |

## 5.3 具身 VLA / 移动操作

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| 3D-IC | 导航与操作联合规划 | 见重点分析 / 见“四” | B | [WeChat](https://mp.weixin.qq.com/s/Czg4zAIKMi4by2Aug7OG2A) |
| AXIS | 大规模遥操及数据引擎 | 部署浏览器端的动作遥操平台并提供自检测和轨迹平滑数据标注 | C | [arXiv](https://arxiv.org/abs/2607.21588v1) |
| Closing the Lab-to-Store Gap | G1 人形货架整理 VLA | 在货架摆放任务中，基于 DEED 架构探索控制频率与数据对齐后训练 | C | [arXiv](https://arxiv.org/abs/2607.20345v1) |
| FM-VLA | 力反馈潜记忆 VLA | 见重点分析 / 见“四” | B | [arXiv](https://arxiv.org/abs/2607.18231v1) |
| Closing the Loop in Humanoid | 持久几何三维实体 Tokens | 在 Loco-manipulation 中使用 persistent 3D records 校验动作完成 | C | [arXiv](https://arxiv.org/abs/2607.18016v1) |
| IMBench | 直觉物理操作评测基准 | 统一评测感知、物理直觉、操纵控制和重试闭环的操作新基准 | C | [arXiv](https://arxiv.org/abs/2607.15641v1) |
| Foresight Residual RL | 长程预见强化学习 | 在 VLA 策略上引入离线估计的预见价值概率，优化多步装配任务 | C | [arXiv](https://arxiv.org/abs/2607.16506v1) |
| Towards Human-like VLA | 持续学习塑性平衡 | LifelongVLA 结合双时序增量更新，缓解机械臂微调中的遗忘问题 | C | [arXiv](https://arxiv.org/abs/2607.14852v2) |
| Representation-Aligned Tactile | 触觉对齐 VLA 训练 | 将未来触觉状态对齐至 VLA 隐藏表征中以引导接触力学学习 | C | [arXiv](https://arxiv.org/abs/2607.14609v1) |
| Learning Robust Execution | 操作受阻高层重试决策 | 设计运行时稳定性指标并利用 Agentic RL 调度高层恢复行动 | C | [arXiv](https://arxiv.org/abs/2607.13818v1) |
| VistaVLA | 3D高斯抓取 VLA | 引入 explicit 3D Gaussian-grounded 特征以防范 2D 动作的几何穿模 | C | [arXiv](https://arxiv.org/abs/2607.12356v2) |
| Towards Predictive Robot Learning | 潜空间世界模型学习 | Lumo-2 优化自监督特征潜空间以保证高层动作生成对齐动力学 | C | [arXiv](https://arxiv.org/abs/2607.11270v1) |
| SUREFlow | 触觉不确定性感知的流匹配 | 见重点分析 / 见“四” | B | [arXiv](https://arxiv.org/abs/2607.10504v1) |
| On the Efficiency of LoRA | 工业 UR5e 微调 LoRA 探究 | 系统性评估 LoRA 秩与冻结策略在工业流匹配机器人中的效果 | C | [arXiv](https://arxiv.org/abs/2607.10172v1) |
| Harness VLA | VLA 动作包装重试 primitive | 利用 LLM 动态编写控制代码，把 VLA 当作可调用和自重试的局部技能 | C | [arXiv](https://arxiv.org/abs/2607.08448v3) |
| FabriVLA | 轻量级操作模型 | 针对多任务精细机械臂控制的轻量级 VLA 架构设计 | C | [arXiv](https://arxiv.org/abs/2607.08575v2) |
| TouchWorld | 灵巧手触觉预测世界模型 | 灵巧手多指接触操作中的触觉时序预测物理世界基础大模型 | C | [arXiv](https://arxiv.org/abs/2607.07287v2) |
| Smooth Operator | 人到手部姿态重定向 | 实时在动力学约束下将人类手部动作映射至机器人灵巧手的算法 | C | [arXiv](https://arxiv.org/abs/2607.07491v2) |
| Dual Latent Memory | 长短时序潜记忆 VLA | 结合长短期潜特征融合的 VLA 控制网络以处理历史依赖操作 | C | [arXiv](https://arxiv.org/abs/2607.07608v1) |
| NativeMEM | 长时历史特征压缩无损 | 在特征层面使用自监督压缩机制缩减历史动作 tokens 以支持长程指令 | C | [arXiv](https://arxiv.org/abs/2607.06678v1) |
| Lift3D-VLA | 机械臂 3D 物理操作 | 提取 3D 点云与姿态特征提升 VLA 模型的空间交互和动力学表现 | C | [arXiv](https://arxiv.org/abs/2607.06564v1) |

## 5.4 无人机、自动驾驶与其他低相关方向

| 工作 | 主题 | 一句话贡献 | 相关度 | 链接 |
|---|---|---|---|---|
| SkyShield | 无人机低空占据安全感知 | 低空视觉占据预测前视单目数据集与 KAR-mIoU 飞行安全风险指标 | C | [arXiv](https://arxiv.org/abs/2606.00747) |
| LNN-Fly | 连续时间避障模型 | 引入实测控制周期 $\Delta t$ 并结合自适应记忆门克服控制抖动撞墙 | C | [arXiv](https://arxiv.org/pdf/2606.28827) |
| UAV-DualCog | 双认知航拍时空推理基准 | 包含自状态感知与环境理解耦合推理的无人机时空大模型评测 | C | [arXiv](https://arxiv.org/abs/2607.16193) |
| No Training, Better Flights | 测试时导航轨迹修正 | 在无人机导航中使用测试时缩放机制自评估并选择最佳轨迹 | C | [arXiv](https://arxiv.org/abs/2607.19288v1) |
| OrthoTrack | 航拍底图厘米级6DoF定位 | 无卫星场景下仅依赖公开正射影像（TDOM）进行时序位姿厘米级跟踪 | C | [arXiv](https://arxiv.org/pdf/2606.25245) |
| PiLoT v2 | 压缩正射地图全局定位 | 采用二维半地图（TDOM+DSM）代替重型 3D 网格实现低资源机载定位 | C | [arXiv](https://arxiv.org/abs/2606.31098) |
| CosFly-VLA | 遮挡追踪闭环恢复控制 | 见重点分析 / 见“四” | B | [arXiv](https://arxiv.org/pdf/2607.15004) |
| AeroAct | 四轴 W-A 模型动作生成 | 空中语言条件控制下，利用扩散 Transform 预测连续动作后果图像 | C | [arXiv](https://arxiv.org/abs/2607.14997v1) |
| FSD-VLN | 鹏城空中导航快慢解耦 | 将空中 VLN 的高层语义规划与高频动作生成彻底异步解耦进行控制 | C | [arXiv](https://arxiv.org/abs/2607.08359) |
| SemCityLoc | 城市 3D 定位几何对齐 | 利用语义-几何对齐框架和轻量城市底图降低低空 6DoF 的位姿漂移 | C | [arXiv](https://arxiv.org/abs/2606.27444) |
| CosFly-Track | 多模态无人机跟踪数据 | 发布 240 万帧多模态对齐的无人机视觉跟踪数据集填补产业空白 | C | [arXiv](https://arxiv.org/abs/2605.17776) |
| HyWorldVLA | 双路世界模型自动驾驶 | 视频 VAE 潜特征预测结合未来像素重构双重约束的自动驾驶 WAM | C | [arXiv](https://arxiv.org/abs/2607.20988v1) |
| S-squared-VLA | 自动驾驶语义空间流解耦 | 解耦 VLA 中离散决策 tokens 与连续轨迹生成控制流，防表征 collapse | C | [arXiv](https://arxiv.org/abs/2607.13926v1) |
| WCog-VLA | 博弈 Chain-of-Thought 驾驶 | 融合 3D 空间、博弈推理与生成式扩散渲染的世界认知端到端驾驶 | C | [arXiv](https://arxiv.org/abs/2607.08375v1) |
| Post-Training Driving | 自动驾驶 RL 对齐综述 | 系统梳理端到端自动驾驶在专家克隆后应用 RL 提升安全舒适的方案 | C | [arXiv](https://arxiv.org/abs/2607.08072v2) |
| Can the Cloud Drive? | 5G/6G 云托管延迟可行性评估 | 从 GPU 服务与队列随机性评估云托管具身智能模型的可行成本与延迟 | C | [arXiv](https://arxiv.org/abs/2607.09045v1) |
| Reasoning as Double-Edged Sword | 推理鲁棒性跨阶段测试 | 评估对比无推理/文字 CoT/潜空间循环 VLA，发现潜空间循环极易崩溃 | C | [arXiv](https://arxiv.org/abs/2607.17786v1) |
| JoyNexus | 云端多租户后训练服务 | 针对 VLA 模型研发的多租户云端强化学习及 SFT 调度微调后端系统 | C | [arXiv](https://arxiv.org/abs/2607.16074v1) |
| AC-VLA | 行为克隆微调泛化退化诊断 | 评估指出微调会损害 VLM 隐藏特征，引入指令解耦与坐标对齐防退化 | C | [arXiv](https://arxiv.org/abs/2607.15714v1) |
| Xiaomi-Robotics-1 | 小米具身 VLA 大模型 | 结合自监督场景转移伪标注，在 10 万小时遥操轨迹下训练的通用 VLA | C | [arXiv](https://arxiv.org/abs/2607.15330v2) |
| Reflex | 去噪流匹配 VLA 部署运行时 | 利用时间步不变性滑动缓存 KV-Cache，支持流去噪端侧 10Hz 输出 | C | [arXiv](https://arxiv.org/abs/2607.14695v1) |
| Lights, Camera, Malfunction | 光照聚光灯物理攻击与防守 | 物理聚光灯对抗攻击框架，揭示常规抗噪增强会导致颜色感知崩塌 | C | [arXiv](https://arxiv.org/abs/2607.14698v1) |
| FoMoVLA | 点跟拍未来特征对齐 | 机械臂 VLA 框架，融合未来特征预见与点跟拍约束提升时间平滑 | C | [arXiv](https://arxiv.org/abs/2607.14739v1) |
| Never Too Late for Force | LIFT 端部力学反馈注入 | 在 VLA 骨干上引入力学侧路 tokens 和交叉注意力提高接触力反应 | C | [arXiv](https://arxiv.org/abs/2607.14236v1) |
| Generalizable VLA Finetuning | 动作语义对齐训练 | 机械臂模仿学习中，使用 frozen VLM 特征蒸馏与离散动作语义对齐 | C | [arXiv](https://arxiv.org/abs/2607.13429v1) |
| DiMaS | 机械臂行为引导控制 | 提出分布匹配策略调整流匹配 VLA 动作生成中间态以干预动作偏好 | C | [arXiv](https://arxiv.org/abs/2607.14280v1) |
| TrustVLA | VLA 安全后门主动防御 | 诊断 INFUSE 后门，定位其注意力的 compact causal footprint 实行防御 | C | [arXiv](https://arxiv.org/abs/2607.12571v1) |
| Reducing Temporal Redundancy | 增量特征与2-step动作去噪 | Perceptual tokens 增量更新结合动作流去噪蒸馏加速 VLA 推理 | C | [arXiv](https://arxiv.org/abs/2607.12287v1) |
| ExToken | 后训练行为 discrete Condition | 在 RL 后训练中使用先验 discrete condition tokens 提升探索效率 | C | [arXiv](https://arxiv.org/abs/2607.12931v1) |
| TS-Mask VLA | 时空掩码扩散动作回归 | 融合 2D 时空掩码与离散扩散专家，克服 next-token 自回归开销 | C | [arXiv](https://arxiv.org/abs/2607.09818v1) |
| CLAP | VLM-to-VLA 对齐直接映射 | 隐式对齐语言动作对，以无损保留预训练隐藏层特征泛化性 | C | [arXiv](https://arxiv.org/abs/2607.08974v1) |
| Training-Free VLA Acceleration | 动作块重用加速 | 推理时重复使用高频动作 tokens 块，仅在侧分支作残差修正避障 | C | [arXiv](https://arxiv.org/abs/2607.06370v1) |
| SIEVE | 控制几何相似度轨迹剪裁 | 对遥操数据进行控制动作与几何相似度筛选以优化模仿训练效率 | C | [arXiv](https://arxiv.org/abs/2607.06442v1) |
| Unified Prediction Planning | 共享骨干参数分配冲突 | 分析预测邻居动作和自车安全规划的 Skill Conflict 现象及分配解决 | C | [arXiv](https://arxiv.org/abs/2607.19971v1) |
| ReferTrack | 自中心单目检测人身追踪 | 滑动记忆特征存储运动序列以输出 3D 轨迹点执行持久跟从 | C | [arXiv](https://arxiv.org/abs/2607.20061v1) |
| RoboTTT | TTT时序上下文自更新 | 使用 Test-Time Training 在推理阶段就地根据人类演示进行策略修正 | C | [arXiv](https://arxiv.org/abs/2607.15275v1) |
| UESF-Bench | 人身寻找跟随联合基准 | 联合测试机器人在陌生环境中寻找目标人并稳定跟随的长时序任务 | C | [arXiv](https://arxiv.org/abs/2607.13621v1) |
| Semantic Anchoring | 微调语义结构保护 | 机械臂模仿微调导致语义对齐破坏，提出特征锚定回 frozen VLM 隐藏层 | C | [arXiv](https://arxiv.org/abs/2607.13597v2) |
| MAMMOTH | 越野多模态视觉导航寻路 | 恶劣光照或单雷达/相机故障下通过端到端 traversability 鲁棒寻路 | C | [arXiv](https://arxiv.org/abs/2607.12965v1) |
| Jetson-PI | 边缘端动作异步偏角纠正 | 利用预测器纠正因边缘设备异步推理延迟造成的机体指令偏角 | C | [arXiv](https://arxiv.org/abs/2607.12659v3) |
| ChunkFlow | 分块动作接缝重叠平滑 | 在 VLA 动作块输出中加入 seam-aware 训练和平滑重叠以规避动作边界抖动 | C | [arXiv](https://arxiv.org/abs/2607.12992v1) |
| VIA | 界面界面交互机器人控制 | 验证大模型的视觉 UI 操作能力直接用于操纵机器人关节控制 | C | [arXiv](https://arxiv.org/abs/2607.11119v1) |
| 0.58M Parameter Navigation | 几何解析硬编码控制策略 | 将透视变换等几何关系硬编码为特征接口以大幅缩减可学习参数 | C | [arXiv](https://arxiv.org/abs/2607.11029v2) |
| World Action Models Roadmap | 世界动作模型预测路线 | 研讨 WAM 动作预测后果范式的技术难点与数据集规范倡议 | C | [arXiv](https://arxiv.org/abs/2607.11689v1) |
| PAC-ACT | 动作 ACT RL 后训练 | 引入混合行为约束，使用强化学习优化 ACT 动作分块大模型 | C | [arXiv](https://arxiv.org/abs/2607.09590v1) |
| Learning More from Less | 失败轨迹 hindsight 重标 | 利用第三方大模型 relabel 失败轨迹的目标与奖励提升后训练采样率 | C | [arXiv](https://arxiv.org/abs/2607.09042v1) |
| TFP | 阶段进度感知隐藏层状态 | 机械臂 VLA 控制中，在接触点与 subgoal 转换时动态刷新隐状态 | C | [arXiv](https://arxiv.org/abs/2607.08283v2) |
| Prompt-Driven Exploration | 滚动交互 Rollout 自修正 | 探索时由 VLM 根据 Rollout 动态纠偏动作 Prompt 以优化 RL 表现 | C | [arXiv](https://arxiv.org/abs/2607.08837v1) |
| Multi-Agent VLMs | 多机低参大模型协同决策 | 低参机载大模型在机器人底盘群体中分布式协同通信与避障决策 | C | [arXiv](https://arxiv.org/abs/2607.07403v1) |
| HELP | 阶段误差分割及人工干预 | 评估模仿学习在闭环 rollout 中高误差阶段并引导局部高价值对齐 | C | [arXiv](https://arxiv.org/abs/2607.09776v2) |
| Vision Language Action Review | UAV与双臂操作大模型综述 | UAV 飞行控制与双臂灵巧操作大模型的技术分类与现状综述 | C | [arXiv](https://arxiv.org/abs/2607.06706v1) |

## 5.5 资讯与非论文

| 日期 | 事件 | 类型 | 与研究的关系 | 链接 |
|---|---|---|---|---|
| 2026-07-14 | 缝合特征插件与主干模块学术包汇总 | 学术资源宣传 | 介绍 500 个学术插接注意力/卷积/多尺度融合等模块，辅助模型微调 | [WeChat](https://mp.weixin.qq.com/s/oQdfIks9bQGY1R_u_0amcA) |
| 2026-07-07 | 深蓝学院四足动力学与强化学习行走控制课程大纲 | 教育培训宣传 | 介绍四足机器人系统辨识、PPO策略训练、域随机化与真机部署 | [WeChat](https://mp.weixin.qq.com/s/xR35JJDgoTJCpaKkDNLX7w) |

---

# 六、趋势判断与行动建议

## 趋势

- **大模型长程规划（慢脑）与轻量化避障（快脑）的异步解耦成为标准范式**：本期 ABot-N1 与 FSD-VLN 均证明，通过引入中间层（如 2D 像素锚点或离散网格轨迹），可以完美规避大模型推理延迟对电机控制高实时性的干扰，避免单一端到端网络内部的梯度负迁移。
- **物理真实度对 sim-to-real 评估的影响力被重新强调**：如 NavVerse 中关于轮式底盘动力学失效和路沿碰撞的消融所示，仅依赖完美的离线几何路径无法预测真实部署中的卡死和跌落。必须将机器人的本体物理边界与真实的连续物理控制引擎（如 NVIDIA Isaac Sim）融合，才能客观反应算法水平。
- **动作表示向“图像空间接地”过渡**：DA-Nav 等工作表明，直接回归 3D 运动坐标易引入空间幻觉，而回归至自中心图像平面 2D 离散网格上不仅贴合多模态大模型的视觉感知先验，还天然适合融合离线轨迹偏差以执行闭环纠正。

## 研究空白

- **室内到户外无缝跨域导航中的光照与尺度动态适应机制**：在跨出大门的物理边界时，环境从结构化（室内走廊）转为非结构化（开阔路段），且光照可能发生数量级的变化。现有模型在这一临界点的特征对齐极其脆弱，容易导致寻找出口超时或目标识别崩塌。
- **基于高保真稀疏特征的长时记忆在高频动态场景下的自我进化与覆盖机制**：RAVEN 提出了高效的三元组记忆，但其仅针对静态物体和已探索空间。在人流密集或物品频繁搬动的动态半静态环境中，如何识别“环境要素发生了改变”并进行有选择的局部记忆擦除或重写，仍缺乏有效的数学边界。

## 建议动作

| 动作 | 目标 | 优先级 |
|---|---|---|
| 复现 | 搭建 [RAVEN](https://arxiv.org/abs/2606.25206) 的三元组 FAISS 记忆数据库与 VLM 工作记忆迭代检索机制，评估其在复杂多隔断地面导航中的真实召回率 | 高 |
| 复现 | 复现 [ABot-N1](https://arxiv.org/abs/2607.10383) 的慢脑 Qwen-4B 思维链 CoT 输出像素锚点与快脑控制器的通信接口，用于我们本地底盘的控频对齐 | 高 |
| 跟踪 | 将我们的地面导航算法接入 [NavVerse](https://arxiv.org/abs/2607.19695) 物理仿真基准以进行一站式跨场景闭环安全评估 | 中 |
| 暂缓 | 暂缓复现 [0.58M Parameter Navigation](https://arxiv.org/abs/2607.11029v2) 的极小参数学习导航，因其过度依赖已知的刚性几何投影硬编码，开放世界泛化性有待验证 | 低 |
