## 1. MapNav (2025)
———A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation

📄 **Paper**: https://arxiv.org/abs/2502.13451

---

**精华**

MapNav 的核心创新在于用轻量级的 Annotated Semantic Map (ASM) 替代传统的历史 RGB 帧序列作为记忆表示，实现了恒定 0.17MB 的内存占用（与步数无关），推理速度提升 79.5%。值得借鉴的关键思想：将语义地图与自然语言标注相结合，使 VLM 能够直接理解空间信息，而无需额外的解码器；用结构化的 top-down 地图取代时序帧，把"历史信息"从时间维度转移到空间维度，大幅降低计算开销。这种"语言化地图"的思路为 VLM 赋能导航提供了一个清晰且高效的范式。

---

**研究背景/问题**

Vision-and-Language Navigation (VLN-CE) 要求 agent 在连续三维环境中跟随自然语言指令导航。现有方法大量依赖历史 RGB 帧作为时序上下文，导致内存随轨迹长度线性增长（Navid 在 300 步时高达 276MB），且无法充分利用 VLM 对语言的理解能力。设计一种高效的记忆表示以替代历史帧，成为本工作的核心动机。

---

**主要方法/创新点**

MapNav 提出一个端到端的 VLM-based VLN 框架，核心组件是在线更新的 Annotated Semantic Map (ASM)。

<div align="center">
  <img src="/images/MapNav-framework-overview.png" width="100%" />
<figcaption>
MapNav 整体框架：ASM 与当前 RGB 观测、指令一起输入 VLM，直接生成导航动作
</figcaption>
</div>

**ASM 生成流程**

<div align="center">
  <img src="/images/MapNav-ASM-generation.png" width="100%" />
<figcaption>
ASM 生成过程：RGB-D → 点云 → 语义地图 → 文字标注地图
</figcaption>
</div>

ASM 是一个多通道张量 **M**（维度 $C \times W \times H$，$C = C_n + 4$），其中：
- **基础通道（1-4）**：编码障碍物分布、已探索区域、agent 当前位置、历史轨迹
- **语义通道（n个）**：存储各目标物体的空间分布

生成流程：
1. 用 Mask2Former 对当前 RGB 帧做语义分割，提取目标 mask
2. 结合深度图将 3D 点云投影到 2D 俯视平面，对齐语义 mask
3. 对每个语义区域做连通分量分析，计算区域质心并在地图上添加文字标签（如 "chair"、"potted plant"）
4. 生成最终 ASM，包含物体位置、轨迹、障碍物等结构化信息

**为什么 ASM 优于普通语义地图？**

<div align="center">
  <img src="/images/MapNav-map-format-comparison.png" width="100%" />
<figcaption>
不同地图格式的 VLM 理解对比：ASM 的文字标注使 VLM 能精确识别物体位置和语义
</figcaption>
</div>

实验证明，VLM（GPT-4o 和 MapNav）处理 ASM 时表现出对物体位置的精准理解（注意力峰值 > 0.8），而处理原始 top-down 地图（峰值 < 0.3）或语义地图（峰值 < 0.4）时注意力极为分散。ASM 通过显式文字标注将抽象语义转化为语言基础，充分激活 VLM 预训练的语言理解能力。

**双流编码器架构**

MapNav 基于 LLaVA-Onevision 框架，使用 SigLIP-so400m 视觉编码器：

$$\mathbf{F}_t = \Phi_{spatial}(\mathbf{X}_t, \mathcal{G}), \quad \mathbf{F}_t^M = \Phi_{spatial}(\mathbf{X}_t^M, \mathcal{G})$$

两路特征分别通过 MLP 投影对齐到语言空间，最终拼接为统一表示：

$$\mathbf{V}_t = [\text{TASK}; \mathbf{E}_t; \text{OBS}; \mathbf{E}_t^M; \text{MAP}]$$

**动作预测**

VLM 直接输出自然语言动作，通过正则表达式匹配解析为 {前进, 左转, 右转, 停止} 四类动作，无需额外动作解码器。

**训练数据（~1M 样本）**

三阶段数据收集：
- Phase I：来自 R2R + RxR 的 GT 轨迹（~300k × 2）
- Phase II：DAgger 在线交互采集（~200k × 2）
- Phase III：碰撞恢复专项数据（~25k × 2）

**历史帧数量的消融**

<div align="center">
  <img src="/images/MapNav-historical-frames-ablation.png" width="100%" />
<figcaption>
不同历史帧数量对性能的影响：加入 ASM 的提升远大于增加历史帧数量
</figcaption>
</div>

加入 ASM 后，SR 从 27% 提升至 36%，SPL 从 23% 提升至 34%；继续增加历史 RGB 帧带来的提升则相当有限，说明核心增益来自 ASM 的空间表示能力，而非时序帧累积。

---

**核心结果/发现**

**模拟环境（R2R-CE & RxR-CE Val-Unseen）**

| 方法 | R2R SR↑ | R2R SPL↑ | RxR SR↑ | RxR SPL↑ |
|------|---------|----------|---------|----------|
| NaVid (All RGB Frames) | 49.1 | 37.4 | 23.8 | 21.2 |
| MapNav (w/o ASM + Cur. RGB) | 41.2 | 27.1 | 15.6 | 12.2 |
| **MapNav (w/ ASM + Cur. RGB)** | **50.3** | **36.5** | **22.1** | **20.2** |
| MapNav (w/ ASM + Cur. + 2 His. RGB) | 53.0 | 39.7 | 32.6 | 27.7 |

- 仅用 ASM + 单帧 RGB，性能即可媲美使用全部历史帧的 NaVid
- 加入 2 帧历史 RGB 后超越所有 SOTA，R2R SPL 提升 1.3%，RxR SPL 提升 6.5%

**效率对比（关键优势）**

| 方法 | 1步 | 10步 | 100步 | 300步 | 平均推理时间 |
|------|-----|------|-------|-------|------------|
| Navid | 0.92MB | 9.2MB | 92MB | **276MB** | 1.22s |
| **MapNav** | **0.17MB** | **0.17MB** | **0.17MB** | **0.17MB** | **0.25s** |

- 内存占用恒定 0.17MB，与轨迹长度完全解耦
- 推理速度提升 **79.5%**（1.22s → 0.25s）

**真实世界（5种室内场景）**

在 Office、Meeting Room、Lecture Hall、Tea Room、Living Room 中，MapNav 在简单指令和语义指令下均全面超越 WS-MGMAP 和 Navid，SR 提升最高达 30%。

---

**局限性**

语义分割模块在遮挡或光照变化等复杂条件下可能产生不准确的物体标签，从而影响 ASM 质量。未来计划扩展到更复杂的具身 AI 任务（如交互导航和操作），需将物体可供性和物理交互能力整合进 ASM 表示。
