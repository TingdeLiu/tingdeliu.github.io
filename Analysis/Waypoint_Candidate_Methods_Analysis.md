# 具身导航 (ENav / VLN) Waypoint Candidate (候选航点) 全景对比与统计分析报告

> **整理/合并**: Antigravity  
> **日期**: 2026-07-28  
> **文档位置**: [Waypoint_Candidate_Methods_Analysis.md](file:///C:/Github/Tingde.Liu.github.io/analysis/Waypoint_Candidate_Methods_Analysis.md)  
> **合并说明**: 本文档系合并 `Waypoint_Candidate_Methods_Analysis.md` 与 `waypoint_candidate_report.md` 后的全量统计报告，以多维度对比表格为核心，全面覆盖经典及 2026 年最新 21 项代表性工作。

---

## 1. 概述与核心范式统计

在视觉语言导航 (Vision-Language Navigation, VLN) 及具身智能 (Embodied AI / ENav) 领域，**航点候选 (Waypoint Candidate)** 的生成、表达与选择是连接高层语义/视觉决策与低层控制的关键环节。

为了克服早期的离散拓扑图局限，近年研究形成了 5 大主流技术范式。以下为 5 大范式的大类对比统计：

| 范式编号 | 范式名称 (Paradigm) | 代表论文/方法 (21项全集) | 候选点/轨迹生成机制 | 大模型 (LLM/VLM) 角色 | 核心优势 | 关键局限 / 瓶颈 | 统计占比 (约) |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **P1** | **学习型预测器**<br>(Learned Predictor) | CMA, ETPNav | 专用网络 (Depth/RGB) 在线预测极坐标 $\Delta (r, \theta)$ 或拓扑节点 | 传统 Transformer 预测概率 / 离散动作分发 | 端到端训练，计算速度快 | 易发生 OOD 泛化失效；若预测器漏看候选区域则高层决策无法救回 | 10% |
| **P2** | **VLM + 粗到细/两阶段过滤**<br>(Two-Stage & Coarse-to-Fine) | Open-Nav, EvoMemNav, SEDualVLN | 预测器 (Depth+NMS) 或分层记忆图生成候选集/视图 | VLM 进行 CoT 推理选择 Top-K 点，或在精筛阶段做语义路由 | 显著降低 VLM 的 Prompt Token 消耗与计算开销 | 依然依赖前置候选推荐，存在前置漏检风险 | 15% |
| **P3** | **地图 / 几何 / 前沿候选**<br>(Map/Frontier/Topology) | VL-Nav, Skill-Nav, GS-VLN, HSGM, Abstract Obstacle Map, ReasonNavi, TravExplorer | 基于占用网格(2D Occupancy)、TSDF、3DGS 或前沿(Frontier)聚类提取候选点 | 结合视觉提示 (Visual Prompt) 或 HVL 权重打分择优 | 无需训练航点预测器，几何连通性与安全性保障强 | 依赖建图质量，在动态/无深度传感器场景表现受限 | 35% |
| **P4** | **生成式轨迹池与 Critic 打分**<br>(Generative Trajectory Pool) | NavDP, GeniNav, DriveVLN, WAM-Nav | 扩散模型 (Diffusion) 或一致性流匹配 (CFM) 批量生成候选轨迹池 | Critic 网络多维打分，或 VLM 根据视图彩虹光栅化挑选 | 轨迹运动学平滑，避障能力极强，支持零样本迁移 | 采样计算开销较高，对实时性控制要求极高 | 20% |
| **P5** | **去预测器 / 像素指向 / 交互链**<br>(Predictor-Free & Pixel Pointing) | AgenticNav, ABot-N1/CISPO, LaViRA, Traj-VLN, 3D-IC | **不预测 3D 候选集**；直接在视角图像上预测像素 $(u,v)$ / BBox，或构建阶段交互链 | VLM/VLA 直接调用 `move_to(u,v)` 工具或做 2D 网格选择 | 摆脱 3D 坐标回归幻觉与候选集覆盖瓶颈，成功率高 (SR+5%) | 依赖精确的 2D-to-3D 几何投影或底层 A* / 逆运动学解算 | 20% |

---

## 2. 全量 21 项代表性论文/方法对比主表 (Master Statistical Table)

以下表格汇总了涵盖 ICCV, EMNLP, CoRL, ICRA, CVPR, ICML 及 arXiv 最新发表的 21 项 Waypoint Candidate 核心工作：

| 序号 | 方法/论文名称 | 年份/会议 | 论文链接 (arXiv / OpenAccess) | 范式分类 | 候选航点/轨迹生成机制 | 候选选择/筛选/打分机制 | 输入与空间表达 | 关键特点与创新突破 |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **1** | **CMA** | ICCV 2021 | [arXiv:2004.02852](https://arxiv.org/abs/2004.02852) | P1 (预测器) | 深度网络在线预测极坐标偏移 $\Delta(r, \theta)$ | 启发式/动作分类器输出选择概率 | Depth + RGB (极坐标) | VLN-CE 连续环境开山之作，奠定 Waypoint Predictor 基础 |
| **2** | **ETPNav** | EMNLP 2023 | [arXiv:2310.16381](https://arxiv.org/abs/2310.16381) | P1 (预测器) | 动态拓扑预测网络预测局部可达邻近节点 | 跨模态 Transformer 局部与全局注意力打分 | RGB-D (连续拓扑图) | 在线实时构建连续拓扑 Candidate 节点 |
| **3** | **VL-Nav** | 2024 | [arXiv:2311.17387](https://arxiv.org/abs/2311.17387) | P3 (地图/前沿) | 混合生成：①Frontier边界聚类点；②目标检测中心点(IBTP) | HVL Score (视觉高斯混合 + 语言匹配度) 综合打分 | RGB-D + 开放词汇检测 | 结合探索前沿与目标实例候选，模拟人类搜索逻辑 |
| **4** | **Skill-Nav** | CoRL 2024 | [arXiv:2409.09841](https://arxiv.org/abs/2409.09841) | P3 (地图/几何) | A* 在占用图上按 0.5–3m 步幅采样航点序列 | 低层 Controller 计算物理控制，高层 LLM 规划 | 2D Occupancy Grid | Waypoint 作为高层规划与底层四足运动策略接口 |
| **5** | **NavDP** | 2024 | [arXiv:2410.10803](https://arxiv.org/abs/2410.10803) | P4 (生成轨迹) | Diffusion Policy 一次性并行生成多条候选轨迹/航点 | 独立 Critic 网络评估每条轨迹的安全裕度分值 | RGB-D (轨迹池) | 零样本 Sim-to-Real 迁移，复杂障碍物避障鲁棒 |
| **6** | **Open-Nav** | ICRA 2025 | [arXiv:2409.11210](https://arxiv.org/abs/2409.11210) | P2 (两阶段) | 独立预测模块 (深度图+NMS) 提取 Top-K 极坐标候选点 | 结合 SpatialBot/RAM 感知，由 LLM CoT 推理选择 | Depth + RGB | 首个用开源 LLM 替代闭源 API 的连续 VLN 框架 |
| **7** | **GS-VLN** | 2025 | [arXiv:2411.18247](https://arxiv.org/abs/2411.18247) | P3 (3DGS) | 3D Gaussian 节点扩展属性作为 3D 候选 Waypoint | 多层 Transformer ($\mathcal{F}^{\text{MLT}}$) 结合 3D 特征预测 | 3D Gaussian Splatting | 显式结合 3D 几何与语义进行 Candidate 概率预测 |
| **8** | **ABot-N1 / CISPO** | 2025/2026 | - | P5 (像素指向) | **不生成候选集**：直接预测视角目标像素 $(u,v)$ 及朝向 $\theta$ | 轻量级 VLM/VLA (System 1) 直接解算连续动作 | 三相机 RGB (像素坐标) | 摆脱物理深度与候选框依赖，适应多构型机器人部署 |
| **9** | **AgenticNav** | 2026 | [arXiv:2601.03254](https://arxiv.org/abs/2601.03254) | P5 (像素指向) | **不生成候选集** (Waypoint-Free)：直接预测视角像素 $(u,v)$ | VLM 直接调用 `move_to(k, u, v)` 工具控制 | 单目 RGB / RGB-D | 打破预测器限制，避开候选漏检，成功率提高 5% SR |
| **10** | **LaViRA** | ICRA 2026 | [arXiv:2510.19655](https://arxiv.org/abs/2510.19655) | P5 (Predictor-Free) | **放弃 3D 航点预测器**；将目标降维为 2D Bounding Box | 语言 MLLM 确定方向 $\rightarrow$ 视觉 MLLM 选 2D 目标 $\rightarrow$ A* 导航 | 2D 透视图像 (BBox) | 消除 3D 预测误诊，提升复杂语境下动作选择精准度 |
| **11** | **HSGM** | CVPR 2026 | [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Bridging_the_2D-3D_Gap_A_Hierarchical_Semantic-Geometric_Map_for_Vision_CVPR_2026_paper.html) | P3 (地图投影) | 分层地图决策图中维护与更新几何可行候选路点 | 将候选路点投影至 RGB 视觉视野 (Visual Prompt) 由 VLM 挑选 | 分层语义地图 (2D-3D) | Visual Prompt 点阵标记，让 VLM “看见”候选点 |
| **12** | **GeniNav** | CVPR 2026 | [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_GeniNav_Generative_Model_Driven_Image-Goal_Navigation_via_Imagination-Guided_Consistency_Flow_CVPR_2026_paper.html) | P4 (流匹配) | 多段一致性流匹配 (MS-CFM) 单次采样 5 条连续候选轨迹 | 混合排序模块 (HRM)：联合语义、几何安全与视野增益打分 | 连续动作轨迹池 | 生成式高平滑性轨迹，解决 Image-Goal 漂移问题 |
| **13** | **DriveVLN** | CVPR 2026 | [CVPR OpenAccess](https://openaccess.thecvf.com/content/CVPR2026/html/Guo_DriveVLN_Towards_Mapless_Vision-and-Language_Navigation_in_Autonomous_Driving_CVPR_2026_paper.html) | P4 (轨迹池) | 轨迹锚定高斯采样与去噪，批量生成 Top-K 候选驾驶轨迹 | 候选轨迹以不同颜色光栅化叠加在透视图上，由 VLM 选择 | 前视透视图 (彩虹轨迹) | 适用于无图自动驾驶，利用视觉 Prompt 实现显式交互 |
| **14** | **EvoMemNav** | arXiv 2026 | [arXiv:2606.03509](https://arxiv.org/abs/2606.03509) | P2 (粗精双阶) | 维护分层记忆图谱：锚点视图与前沿候选视图 ($C^A, C^F$) | 预算式粗筛 (房间/物体可见性) $\rightarrow$ VLM 细精精筛选 | Posed RGB-D | 极大压缩候选数量，大幅缩减 VLM Token 与推理耗时 |
| **15** | **3D-IC** | ICML 2026 | [ICML 2026](https://icml.cc/) | P5 (交互链) | 定义交互航点，按 OVMM 4 阶段（导航/抓取/容器/放置）生成候选 | 组装为候选交互链 (Candidate Chains)，路径成本粗筛+VLM评分 | 3D 拓扑与几何 | 扩展至移动操纵 (OVMM)，导航与操作联动闭环 |
| **16** | **Abstract Obstacle Map** | ICRA 2026 | [ICRA 2026](https://icra2026.org/) | P3 (几何候选) | 基于全局/局部抽象障碍物占据地图预测几何安全航点 | 滤除几何不可达与障碍重叠区域，纯几何连通性校验 | 抽象障碍物 Occupancy | 显著提高零样本在复杂未包含环境下的碰撞安全性 |
| **17** | **Traj-VLN / DA-Nav** | arXiv 2026 | [arXiv:2607.10744](https://arxiv.org/abs/2607.10744) | P5 (2D网格) | 放弃 3D 坐标回归，映射为自中心 2D 平面图像离散网格候选 | 与 VLM 原生 2D 空间推理能力自然对齐，大模型选点 | 2D Egocentric Grid | 大幅降低 3D 空间坐标计算带来的空间幻觉 |
| **18** | **ReasonNavi** | arXiv 2026 | [arXiv:2602.15864](https://arxiv.org/abs/2602.15864) | P3 (均匀节点) | 全局地图中均匀生成与墙体保持安全距离的候选导航节点 | 锁定目标房间后，MLLM 结合地图语义与指令选择全局坐标 | 全局语义地图 | 避免局部碰撞，高层长程推理逻辑清晰 |
| **19** | **TravExplorer** | arXiv 2026 | [arXiv:2605.19958](https://arxiv.org/abs/2605.19958) | P3 (可通行前沿) | RGB-D 点云与梯度特征转化为跨楼层可通行前沿探索候选 | 3D 拓扑图与楼层可达性分析，指导狗机器人跨楼层探索 | 3D RGB-D 点云 | 专为四足机器人跨楼层复杂地形前沿挑选设计 |
| **20** | **SEDualVLN** | arXiv 2026 | [arXiv:2605.17249](https://arxiv.org/abs/2605.17249) | P2 (路径渲染) | 慢速系统基于 3D 俯视图与沿候选路径渲染出的图像序列建模 | GPT-4o 逐一对比候选路径渲染图序列与指令选最佳前沿 | 3D 俯视+路径渲染图 | 渲染未来路径视野供大模型评估，决策准确率高 |
| **21** | **WAM-Nav** | arXiv 2026 | [arXiv:2606.04907](https://arxiv.org/abs/2606.04907) | P4 (世界模型) | WAM-Nav (Diffusion/Flow) 在线批量滚动采样 16 条候选动作轨迹 | 前向世界模型预测未来状态，选择综合表现最优的首条轨迹 | 多相机 RGB / 状态序列 | 基于 World Action Model 预测未来环境演化与奖励 |

---

## 3. 多维度专项交叉对比与统计表格

### 3.1 候选航点/轨迹生成机制 (Candidate Generation Mechanism) 维度对比

| 机制类型 | 代表方法 | 空间表征维度 | 生成算法 / 原理 | 优势 | 主要短板 |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **1. 极坐标 3D 预测器** | CMA, Open-Nav | 3D 极坐标 $(r, \theta, z)$ | Depth/RGB 图像过神经网络或 NMS 筛选极坐标点 | 独立轻量，符合传统机器人运动学输入 | OOD 环境易失效，易产生局部漏检 |
| **2. 2D 透视 / 网格 / BBox 指向** | AgenticNav, ABot-N1, LaViRA, Traj-VLN | 2D 像素坐标 $(u,v)$ / BBox | 图像像素点选、Bounding Box 或 2D Egocentric Discrete Grid | 无需 3D 回归与候选预测器，与 VLM 原生 2D 空间能力完美对齐 | 需底层深度/A*将 2D 转化为 3D 机器人控制 |
| **3. 占用网格 / 前沿 (Frontier) 采样** | VL-Nav, Skill-Nav, Abstract Obstacle Map, TravExplorer | 2D/3D Occupancy Grid | 占用地图聚类提取未探索边界 (Frontier) 或 0.5-3m 采样 | 物理可达性与安全裕度有严格保障，零样本泛化强 | 依赖建图质量，在动态环境建图易掉帧 |
| **4. 隐式生成式轨迹池 (Diffusion/Flow)** | NavDP, GeniNav, DriveVLN, WAM-Nav | 连续动作轨迹序列 $\tau = \{s_t, a_t\}$ | 扩散去噪 (Diffusion) 或一致性流匹配 (Consistency Flow) 采样 | 动力学平滑，碰撞避障能力极强 | 采样迭代步数消耗算力，推理延迟较高 |
| **5. 显式 3D Gaussian / 分层地图** | GS-VLN, HSGM, EvoMemNav, ReasonNavi | 3D 空间节点 / 分层图 | 3DGS 属性扩展、分层记忆图谱或全局均匀节点采样 | 几何与语义信息极其丰富，可进行长程路径规划 | 内存占用较大，初始化地图建构时间较长 |

---

### 3.2 候选打分与决策机制 (Selection & Scoring Mechanism) 维度对比

| 打分/选择机制 | 代表方法 | 打分核心指标 / 价值函数 | 选择器载体 (Decision Engine) | 特性与适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 文本思维链 CoT 筛选** | Open-Nav, EvoMemNav | 语言指令与候选节点语义相关度 | LLM (Qwen, Llama, GPT-4o) | 处理自然语言逻辑强，但处理纯几何坐标容易产生幻觉 |
| **2. 视网膜提示 (Visual Prompt) 视觉直选** | HSGM, DriveVLN | 视角内彩虹轨迹/标记点的语义匹配度 | MLLM / VLM (GPT-4o, Qwen-VL) | **极其直观**，利用大模型视觉定位能力，解决文本坐标困难 |
| **3. 多维混合 Value / Critic 显式打分** | NavDP, GeniNav (HRM) | $\text{Score} = w_1 \cdot \text{Safety} + w_2 \cdot \text{Semantic} + w_3 \cdot \text{View Gain}$ | 专用 Critic 神经网络 | 严格数学加权，安全性极高，适合无图避障与轨迹优化 |
| **4. 路径渲染与世界模型预测打分** | SEDualVLN, WAM-Nav | 沿候选路径未来图像渲染帧的匹配度 / WAM 状态预测 | GPT-4o / World Action Model | 前瞻性最高，能预判盲区背后的景象 |
| **5. 交互链拓扑与路径成本粗精筛选** | 3D-IC, EvoMemNav | 拓扑图距离成本 + 目标语义存在概率 | 分级过滤器 + VLM | 解决大规模长程复杂场景下的算力与 Token 爆炸问题 |

---

### 3.3 硬件平台与应用场景适配矩阵 (Platform & Scenario Matrix)

| 场景分类 | 代表方法 | 机器人/硬件平台 | 先验地图依赖 | 特殊环境挑战与解决方案 |
| :--- | :--- | :--- | :---: | :--- |
| **室内单层 VLN-CE** | CMA, ETPNav, Open-Nav, AgenticNav, Traj-VLN | 轮式/履带机器人 (Matterport3D / Habitat) | 无先验地图 (Mapless) | 解决局部死胡同与视角遮挡问题 |
| **跨楼层 / 狗机器人** | Skill-Nav, TravExplorer | 四足机器狗 (Unitree Go1/B1) | 局部点云建图 | 处理梯段、高低起伏与跨楼层拓扑可达性 |
| **无图自动驾驶 (Mapless AD)** | DriveVLN | 自动驾驶车辆 (Multi-Camera Autonomous Driving) | 无地图 / 单目/多相机透视图 | 高速动态障碍避障与车道语义指令对齐 |
| **移动操纵 (OVMM)** | 3D-IC | 机械臂移动机器人 (Fetch, Stretch, Everyday Robots) | 局部 3D 拓扑图 | 导航与抓取/放置候选动作的跨阶段联合优化 |
| **零样本泛化 / 无图避障** | NavDP, GeniNav, Abstract Obstacle Map, WAM-Nav | 泛用移动机器人 (Real-world Sim-to-Real) | 实时 RGB-D / 占用网格 | 处理 Sim-to-Real 鸿沟与未见过的障碍物 |

---

## 4. 关键洞察与技术演进趋势

通过对 21 项核心工作的表格统计分析，当前 Waypoint Candidate 技术呈现出以下 4 大演进趋势：

```
                              ┌── 1. 空间表达：3D 极坐标预测 ───► 2D 视角像素 / 视觉 Prompt 标注
                              ├── 2. 采样范式：启发式/单节点预测 ──► 生成式轨迹池 (Diffusion / Flow Matching)
Waypoint Candidate 演进趋势 ───┼── 3. 大模型交互：纯文本 Prompt CoT ──► 彩虹光栅化 (Rainbow Overlay) 视觉直选
                              └── 4. 任务边界：纯移动导航 Candidate ──► 导航-操作联合 Candidate Chains (OVMM)
```

1. **从 3D 坐标回归走向 2D 视角指向 (Pixel Pointing / Predictor-Free)**：
   早期方法 (CMA, Open-Nav) 迫使模型在 3D 极坐标空间回归候选点，经常出现“候选框漏看关键目标”的问题。2026 年最新趋势 (AgenticNav, LaViRA, Traj-VLN) 证明：**直接在 2D 视角图像上指定目标像素/BBox 或使用 2D 网格，彻底摆脱 3D 候选预测器，成功率可提升 5% 以上**。

2. **从单点候选走向生成式连续轨迹池 (Generative Trajectory Pools)**：
   单纯的航点 (Waypoint) 缺乏运动学平滑性与速度约束。基于 Diffusion 和 Consistency Flow 的方法 (NavDP, GeniNav, DriveVLN, WAM-Nav) 可单次生成多条高质量候选轨迹，结合 Critic 打分能极大提高避障鲁棒性。

3. **视觉 Prompt (Visual Prompting / Rainbow Overlays) 成为大模型交互新标准**：
   直接将候选点的物理坐标放入 Prompt 容易导致 LLM 产生空间坐标幻觉。HSGM 和 DriveVLN 通过在第一人称 RGB 图像上**叠加彩虹线条或标记点**，让 VLM 像人类一样在视觉画面上“点选”路线，大幅降低了推理误判率。

4. **粗到细 (Coarse-to-Fine) 筛选与长链条 (Candidate Chains) 扩展**：
   在全屋或多楼层大规模环境中，直接输入所有候选点会导致大模型 Token 与算力爆表。EvoMemNav 通过图谱粗筛降低候选基数；而 3D-IC 则进一步将候选机制从“导航航点”拓展至“导航-抓取-放置”的完整交互链条。
