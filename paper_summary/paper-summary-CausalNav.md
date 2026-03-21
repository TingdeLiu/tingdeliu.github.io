## CausalNav (2026)
———First Scene Graph-based Semantic Navigation for Dynamic Outdoor Environments

📄 **Paper**: https://arxiv.org/abs/2601.01872

### 精华

CausalNav 的核心亮点在于将多层级场景图（Embodied Graph）与 RAG 机制深度结合，实现了支持开放词汇查询的长程语义导航——"图即知识库"的设计范式值得借鉴。其次，层次化 Embodied Graph 构建策略（从细粒度对象节点到粗粒度建筑物与聚类节点）展示了如何在多空间尺度上统一语义表示与检索。第三，基于时空走廊（Spatial-Temporal Corridor）的动态对象过滤机制，无需额外标注即可区分静态、准静态与动态障碍物，是处理室外动态场景的实用方案。第四，使用本地开源 LLM 替代商业 API 完成层次语义检索，证明了在自主平台上脱离云端仍可实现高质量语义推理。

---

### 1. 研究背景/问题

室外大规模动态环境中的自主语义导航面临三大挑战：开放词汇的语义理解、动态环境适应（行人、车辆等移动障碍物）以及长期稳定性。现有 VLN 研究主要聚焦于静态室内场景，依赖高精度地图或大规模训练数据，在真实室外动态场景中的长程导航鲁棒性未得到充分验证。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/vln/CausalNav-overview.png" width="100%" />
<figcaption>
CausalNav 整体工作流：集成语义推理、动态环境适应和 Embodied Graph 规划三大模块
</figcaption>
</div>

CausalNav 提出了一个由三个核心模块构成的语义导航框架：

**模块一：开放词汇目标跟踪与自我运动估计**

使用 YOLO-World 从 RGB 图像中提取开放词汇的 2D 检测框和分割掩码，通过 ByteTrack 进行多目标跟踪。结合 LiDAR 点云将 2D 检测投影至 3D 空间，获得目标的 3D 姿态 $^w\mathbf{T}_{obj}$。自车运动通过 LiDAR-IMU 里程计（FAST-LIO2）估计，提供精确的定位与坐标变换基础。

**模块二：动态对象过滤与 Embodied Graph 构建**

<div align="center">
  <img src="/images/vln/CausalNav-architecture.png" width="100%" />
<figcaption>
CausalNav 三模块流水线架构：目标跟踪与自我运动估计 → 动态过滤与图构建 → 图更新与自然语言导航
</figcaption>
</div>

- **时空走廊过滤**：将每个目标的历史轨迹编码为时空走廊 $\mathcal{T} = \{^w\mathbf{T}^n_{obj}, \text{3DBBox}_i, t_i\}_{i=1}^n$。若目标在 $k$ 步内位移超过阈值，则认定为动态目标并从图中移除，有效消除运动引起的虚假节点。

- **Embodied Graph 层次构建**：静态环境由两类节点组成——建筑物节点 $\nu_i^{build}$ 来自离线地图，对象节点 $\nu_i^{obj}$ 来自实时感知。使用 LLM 对节点进行层次聚类（spatial-semantic similarity），形成多级抽象：对象层（Level $L-1$）→ 建筑物/Place 层（Level $L$）→ 聚类节点（Clustering Node）。每次自车移动超过距离阈值 $d$，新增自车节点 $\nu_i^l$ 记录历史轨迹。

- **RAG 语义检索**：基于 LLM 打分的层次化检索，结合空间相似性 $\kappa^{spatial}$ 和语义相似性 $\kappa^{semantic}$，在图中逐层选择最匹配查询的节点路径，支持开放词汇目标定位。

**模块三：Embodied Graph 动态更新与自然语言导航**

<div align="center">
  <img src="/images/vln/CausalNav-embodied-graph.png" width="100%" />
<figcaption>
仿真环境中构建的 Embodied Graph：粗粒度建筑物节点与细粒度对象节点（消火栓、邮箱等）的多层次融合
</figcaption>
</div>

- **全局规划**：解析自然语言指令，通过 RAG 检索 Embodied Graph 推断目标位置，优先使用历史轨迹中的 Dijkstra 最短路径；若目标不可达，则调用离线地图或 Google Maps 生成粗粒度路线，结果表示为路点序列 $\mathcal{W} = \{w_1, w_2, \ldots, w_n\}$。

- **局部规划**：采用 RH-Map 进行实时动态局部地图构建，通过 Informed-RRT* 生成初始轨迹，再使用 NMPC-CBF（Nonlinear Model Predictive Control with Control Barrier Function）进行轨迹跟踪与动态避障，保证对移动行人/车辆的安全性。

---

### 3. 核心结果/发现

<div align="center">
  <img src="/images/vln/CausalNav-real-world-results.png" width="100%" />
<figcaption>
真实环境中不同距离尺度的导航实验：短程（130m 对象级指令）与长程（512m 建筑级指令）
</figcaption>
</div>

**仿真实验**（对比 ViNT、NoMaD、GNM、CityWalker）：
- Small 任务：SR 100%，SPL 88.9%，CC 0.2（所有方法中最优）
- Medium 任务：SR 92%，SPL 82.2%
- Large 任务：SR 80%，SPL 66.0%，CC 1.2，TL 141.82m

**真实世界实验**：
- 短程（130m）：ViNT 和 CausalNav 均成功，其他方法失败
- 长程（512m）：仅 CausalNav 成功完成任务，其他方法因碰撞失败
- CityWalker 在真实世界表现显著差于仿真，对光照变化和动态障碍物敏感

**消融实验**：
- 启用 Embodied Graph 动态更新：SR 从 78% 提升至 90%，SPL 从 54.7% 提升至 80.1%
- 最优超参数：$\alpha=\beta=0.5$，$\gamma=1.5$（空间-语义平衡点）
- 运行时延：105ms/cycle（10Hz），比 NoMaD 仅多 11% 开销

---

### 4. 局限性

CausalNav 在极端光照/天气条件下的鲁棒性有待提升，且长程图记忆的压缩与遗忘机制尚未完善，可能在超长时间运行后出现图膨胀和检索精度下降问题。
