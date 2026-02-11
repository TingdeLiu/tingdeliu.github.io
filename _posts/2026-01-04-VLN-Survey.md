---
layout: post
title: "Vision-Language Navigation (VLN) 综述"
27date:   2026-02-03
tags: [VLN, VLA, Robotics, Computer Vision, Deep Learning]
comments: true
author: Tingde Liu
toc: true
excerpt: "视觉语言导航（VLN）是计算机视觉、自然语言处理和机器人导航交叉领域的前沿研究方向。本文对VLN的基本概念、任务类型、主要挑战和最新进展进行全面综述。"
---

# 引言

视觉语言导航（Vision-Language Navigation, VLN）是一个多学科交叉的研究领域，涵盖了自然语言处理、计算机视觉、多模态信息融合以及机器人导航等多个学科。在该领域，研究人员致力于开发能够理解自然语言指令，并在复杂环境中实现自主导航的智能体。

VLN任务的核心挑战在于如何让机器人或智能体理解人类的自然语言指令，并通过视觉感知在真实或虚拟环境中进行导航。这项技术在服务机器人、自动驾驶、智能家居等领域有着广泛的应用前景。

本博文旨在系统梳理VLN领域的研究进展，为学习和研究VLN提供参考。

# VLN基本概述

## 什么是VLN？

VLN任务的定义是：给定一个自然语言指令（natural language instruction），智能体（agent）被放置在模拟器中的初始位置，需要通过理解指令并观察视觉环境，按照指令给定的路线移动到目的地。

这个任务最早起源于2018年，作者认为让一个5岁左右的孩子去拿一个勺子（"bring me a spoon"）是一个很简单的任务，但是如果想通过语言指令去指导机器人去拿一个勺子却非常困难。

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/WX20250824-184006.png" width="80%" />
<figcaption>
VLN任务示意图
</figcaption>
</div>

### VLN 在具身智能体中的角色

近年来，VLN 越来越多地被视为通用 Vision–Language–Action 智能体的一项核心能力，而非孤立的导航任务。在此背景下，VLN 常作为复杂具身任务（如任务执行或协作导航）的子模块进行研究。

*相关任务*：TEACh 等具身任务执行基准

## VLN的三个核心要素

一个完整的VLN系统包含三个核心要素：

1. **Instruction Source（指令源 / Oracle）**：指令源用于生成或提供自然语言导航指令，模拟人类用户对导航目标的描述。在部分交互式 VLN 设定中，智能体可向指令源请求额外信息或澄清指令，从而形成更接近真实人机交互的导航过程。

2. **Agent（智能体/执行者）**：Agent 是 VLN 系统的核心执行主体，负责感知环境、理解语言指令并输出导航动作。智能体需要根据当前视觉观察、历史状态以及语言指令，与环境进行连续交互，完成从起点到目标位置的导航任务。

3. **Environment（环境）**：Environment 定义了智能体执行导航任务的空间。由于真实环境中的数据采集与训练成本较高，现有研究通常依赖高保真模拟器进行训练与评测。例如，在 Room-to-Room（R2R）任务中，Matterport3D 数据集被广泛用作室内三维仿真环境。

## VLN 系统的基本组成

随着具身智能的发展，VLN 系统正从单纯的“指令匹配”向 **VLA (Vision-Language-Action)** 全能模型演进。其架构通常由感知、大脑、行动三个核心模块组成，并呈现出明显的**分层控制（Hierarchical Control）**趋势：

#### 1. 感知模块 (Perception Module) —— 从单一特征到语义-几何融合
* **功能**：负责从环境中提取基于视觉和语言的观察信息。
* **趋势**：从传统的视觉骨干网络（如 CNN）转向**视觉-语言对齐的 Transformer**（如 SigLIP），以获取更强的指令对齐能力；同时结合具有强空间先验的**几何表示模型**（如 DINOv2），以提高在复杂环境中的空间感知与操作精度。

#### 2. 大脑模块 (Reasoning Module) —— 慢思考：高层逻辑与战略规划
* **功能**：作为系统的“慢系统”，负责融合多模态输入，进行高级逻辑推理、常识判断与任务规划。
* **交互逻辑**：**低频率输出**。大脑模块（通常基于预训练的 VLM）利用大规模互联网知识进行推理，不需要实时输出电机信号。它以较低的频率输出高层决策指令或环境中的**目标点像素坐标（Goal Point Coordinates）**。
* **优势**：支持零样本（Zero-shot）泛化，能够将复杂的自然语言指令分解为可执行的中间目标。

#### 3. 行动模块 (Action Module) —— 快行动：高频生成与物理控制
* **功能**：作为系统的“快系统”，将大脑模块的决策指令转化为具体的物理动作。
* **交互逻辑**：**高频率执行**。行动模块接收来自大脑的目标点坐标，利用**连续生成建模**（如扩散模型 Diffusion Model）以高频率（如 30Hz+）预测平滑、无碰撞的**运动轨迹（Trajectory）**。
* **控制闭环**：基于生成的轨迹，利用 **MPC（模型预测控制）** 或底层控制器精准驱动电机（如输出扭矩、位移信号），实现平滑且多模态的动作分布建模。


### VLN 在 VLA 范式下的独特研究价值
虽然 VLN 属于 VLA 体系，但它与传统的“基于 VLA 的机械臂控制（Manipulation）”在任务逻辑上有着本质区别：

* **长时序环境建模 (Long-Horizon Exploration)**：机械臂控制通常关注近场操作，视角相对固定；而 VLN 涉及长距离、多房间甚至跨楼层的移动（如 LHPR-VLN），要求智能体在移动中动态维护环境记忆（如拓扑图或语义地图），处理因位移产生的空间迷失风险。
* **指令流与视觉流的“时空动态对齐”**：在机械臂任务中，指令（如“抓起杯子”）与目标通常是静态对应的；而在 VLN 中，指令解析是随着位移**动态演进**的。
* **分层异步协同需求**：这决定了 VLN 必须采用“大脑模块（慢系统）”与“行动模块（快系统）”的异步协作——大脑负责高层语义状态跟踪，并周期性地将抽象指令转化为行动模块所需的**像素级局部目标点**。
* **常识推理与物理约束的博弈**：VLN 的独特之处在于如何利用生成式策略（如扩散模型）将大脑模块可能存在的“语义幻觉”转化为符合物理规律的连续轨迹，并由 **MPC（模型预测控制）** 处理碰撞与环境摩擦。

### VLN 区别于传统机器人导航
VLN 与传统的机器人导航（Navigation Stack，如基于 SLAM 的系统）在核心驱动力上有显著不同：

* **从“几何坐标”转向“语义路标” (Semantic vs. Geometric)**：
    * **传统导航**：依赖预建的高精度几何地图（点云或占据栅格），通过全局坐标（XY 坐标）驱动。
    * **VLN**：智能体通常置于**未见环境（Unseen Environment）**中，必须通过理解自然语言中的“语义地标”（如“走过红色的椅子后左转”）进行在线决策，而非单纯的坐标追踪。
* **从“被动避障”转向“主动常识搜索”**：
    * **传统导航**：主要解决“如何不撞到障碍物并到达 A 点”。
    * **VLN**：要求智能体具备具身常识。当指令提到“去厨房拿咖啡”时，即便厨房不在视野内，系统也能利用 VLM 的常识预测厨房的方位并规划搜索路径，这超出了传统导航栈的范畴。
* **端到端语义理解的集成**：
    * **传统导航**：感知、规划、执行是解耦的模块。
    * **VLN**：在 VLA 范式下，视觉感知与语言指令在“大脑模块”中深度融合，直接影响行动模块生成的轨迹分布，实现了从高层语义到低层物理动作的端到端映射。

---

## VLN 的主要挑战

### 1. 语言–视觉–行动（Language–Vision–Action）的一致性与可控推理

尽管大规模视觉-语言预训练模型在跨模态对齐方面取得了显著进展，但如何在导航过程中实现语言指令、视觉观测与动作决策之间的语义一致性，仍然是 VLN 的核心挑战之一。特别是在基于大语言模型的 VLN 系统中，如何保证高层语义推理结果能够被可靠地转化为可执行的低层导航动作，是当前研究亟需解决的问题。

### 2. 开放世界场景下的泛化能力

VLN 模型通常在有限数量的场景和指令分布上进行训练，但在实际应用中需要面对开放世界环境，包括未见过的空间布局、物体组合以及多样化的自然语言表达。如何提升模型在跨环境、跨任务和跨指令分布下的零样本或少样本泛化能力，是当前 VLN 研究的关键瓶颈之一。

### 3. 分层规划中长期目标与短期决策的协同

VLN 任务天然具有长时序和部分可观测的特性，智能体需要在理解全局导航目标的同时，根据局部观测进行实时决策。近年来引入的大语言模型在高层规划和子目标分解方面展现出强大的能力，但如何在分层架构中实现高层推理与低层控制策略之间的稳定协同，以及在规划失败时进行有效恢复，仍然是一个具有挑战性的问题。

### 4. 长时记忆与环境建模能力

在复杂室内或室外环境中，智能体需要整合跨时间的多次观测以形成对环境的整体理解。如何构建有效的记忆机制和世界模型，以支持长期导航、路径回溯和错误纠正，是提升 VLN 系统鲁棒性和效率的重要方向。

### 5. 面向真实部署的可靠性与安全性

尽管大多数 VLN 方法仍主要在模拟环境中进行评测，但随着研究逐步向真实机器人系统过渡，导航过程中的安全性、稳定性以及对异常情况的应对能力变得尤为重要。如何在保证导航成功率的同时，避免潜在的危险行为，并提升系统决策过程的可解释性，是 VLN 走向真实应用必须面对的问题。

## VLN研究发展趋势

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/2025-vln-research-timeline.png" width="100%" />
<figcaption>
VLN Research Timeline (Refer from "Thinking-VLN")
</figcaption>
</div>

从整体发展脉络来看，VLN 研究经历了从任务驱动模型到具身智能体的重要转变，其研究重心也随之不断演进。

### 1. 早期阶段：任务驱动的多模态建模（2018–2019）

该阶段的研究主要关注如何构建有效的视觉与语言联合表示，并探索基于循环神经网络或早期 Transformer 架构的导航模型，为 VLN 任务奠定了基础方法论。

### 2. 中期阶段：数据集与评测基准扩展（2020–2021）

随着研究的深入，多个更大规模、更复杂的 VLN 数据集和评测基准被提出，推动模型从简单场景向更具挑战性的真实环境分布扩展，并促进了对泛化能力的系统性研究。

### 3. 过渡阶段：大规模预训练模型的引入（2022–2023）

在这一阶段，预训练视觉-语言模型以及大语言模型被引入 VLN 任务，使模型具备更强的语义理解、推理与指令跟随能力。VLN 开始从任务特定模型向通用多模态模型能力迁移。

### 4. 当前阶段：面向具身智能体的统一建模（2024–至今）

最新研究趋势表明，VLN 正逐步被视为通用具身智能体的重要能力之一，而非孤立的导航任务。研究重点从单一任务性能提升，转向统一感知、推理、规划与执行的多模态智能体框架，并探索其在更广泛现实场景中的应用潜力。

## 关键技术方向

### 1. 统一的视觉-语言-行动（Vision-Language-Action, VLA）架构

近年来，VLN 研究逐渐从传统的模块化设计转向端到端的统一建模范式，将视觉感知、语言理解、空间推理与动作决策统一在同一模型中。基于 Transformer 或大语言模型的 VLA 架构能够有效处理长时序决策问题，并提升复杂自然语言指令下的导航性能，成为当前 VLN 研究的重要趋势。

### 2. 大规模预训练与指令对齐

VLN 模型越来越依赖大规模视觉-语言预训练，以获得更强的跨模态语义对齐与通用表示能力。通过在图文、视频-语言及导航轨迹等多源数据上进行预训练，并结合指令微调（Instruction Tuning），模型在未见环境和复杂指令下的泛化能力显著提升。

### 3. 基于大语言模型的推理与规划

引入大语言模型进行高层语义推理和路径规划已成为 VLN 的重要研究方向。大语言模型可用于子目标分解、动作规划以及轨迹评估，将 VLN 从纯感知控制问题扩展为具备推理与决策能力的导航任务，从而提升复杂场景和长路径下的导航成功率。

### 4. 记忆机制与世界模型构建

为应对部分可观测环境和长距离导航挑战，研究者引入显式或隐式记忆机制以及可学习的世界模型，用于累积环境信息并进行跨时间推理。语义地图和视觉记忆模块能够帮助智能体构建对环境的结构化理解，是实现高效 VLN 的关键组成部分。

### 5. 数据驱动的训练范式与学习策略

相比传统强化学习方法，当前 VLN 更倾向于采用模仿学习、离线强化学习及数据驱动的训练策略。通过轨迹重标注、数据增强以及利用大语言模型生成伪示例，能够有效提升样本效率并降低真实环境交互成本。

### 6. 从 VLN 到通用具身智能任务

VLN 正逐步与目标导航、具身问答和任务执行等具身智能任务融合，形成统一的具身指令跟随问题设定。这一趋势推动 VLN 从单一任务向多任务、多模态的通用具身智能体发展。

## 未来研究方向

### 1. 开放世界环境下的泛化能力

未来 VLN 研究需要突破对封闭场景和固定指令分布的依赖，实现对未知环境、未见物体及组合式指令的零样本或少样本泛化能力。

### 2. 面向真实机器人的导航能力

缩小模拟环境与真实世界之间的差距是 VLN 走向实际应用的关键挑战。未来研究需充分考虑传感器噪声、动态环境和连续控制等现实因素，以提升 VLN 系统在真实机器人平台上的可靠性。

### 3. 基于多模态大模型的统一具身智能体

将 VLN 视为多模态大模型的重要能力之一，而非独立任务，是当前的重要发展方向。通过统一感知、推理、规划与执行，构建具备多任务能力的具身基础模型，有望显著提升 VLN 的通用性和扩展性。

### 4. 持续学习与自主探索

未来的 VLN 智能体需要具备持续学习能力，能够在长期运行过程中自主收集经验、更新环境认知并不断优化导航策略，从而适应不断变化的环境和任务需求。

### 5. 自然人机交互与协作导航

支持更自然的人机交互形式，如对话式指令修正、不完整或模糊指令理解，以及人类实时干预，将显著提升 VLN 系统在真实应用场景中的可用性。

### 6. 安全性、可解释性与失败恢复

随着 VLN 系统逐步走向真实部署，其安全性和可解释性问题愈发重要。未来研究需关注导航决策的可解释性、失败检测与自我纠错机制，以保障系统在复杂环境中的安全运行。

# VLN 任务类型

随着研究范式的演进，VLN 的任务划分逐渐从早期基于指令形式的分类，转向以智能体能力需求和交互方式为核心的分类方式。

## 按推理与决策复杂度划分

**1. 指令跟随型 VLN（Instruction-Following VLN）**

该类任务要求智能体根据给定的自然语言指令，在环境中完成从起点到目标位置的导航，通常不涉及显式目标搜索或复杂语义推理。此类任务主要用于评估模型的语言理解能力和基本导航能力。

*代表性数据集*：Room-to-Room（R2R）、Room-for-Room（R4R）

---

**2. 语义推理驱动的 VLN（Reasoning-Oriented VLN）**

该类任务在导航过程中引入目标物体搜索或语义约束，智能体需要将语言指令与环境中的语义信息进行推理匹配，从而完成导航与定位任务。

*代表性数据集*：REVERIE、SOON

---

**3. 长时序与组合式 VLN（Long-Horizon VLN）**

该类任务强调长距离导航和复杂指令组合，要求智能体具备长期规划、记忆和错误恢复能力，是评估 VLN 系统长期决策能力的重要设定。

---

## 按交互方式划分

**1. 非交互式 VLN**

智能体在接收到初始指令后独立完成导航任务，过程中不与用户进行额外交互。这是当前最常见的 VLN 评测设定。

---

**2. 交互式与对话式 VLN**

该类任务允许智能体在导航过程中与用户进行多轮交互，通过提问或反馈不断优化导航目标，更接近真实人机协作场景。

*代表性数据集*：CVDN

---



# VLN的应用场景

## 室内场景

室内VLN主要关注家庭或办公环境内的导航。环境通常较为复杂，包含多个房间和各种家具，对智能体的空间理解能力要求较高。

<div align="center">
  <img src="/images/vln_indoor.png" width="100%" />
<figcaption>
室内VLN示例
</figcaption>
</div>

**应用示例**：
- 家庭服务机器人
- 室内物流配送
- 智能导览系统

## 室外场景

室外VLN面临更大的环境复杂度，需要处理动态障碍物、天气变化等因素。

<div align="center">
  <img src="/images/vln_outdoor.png" width="100%" />
<figcaption>
室外VLN示例
</figcaption>
</div>

**应用示例**：
- 自动驾驶
- 户外服务机器人
- 城市导航系统

## 空中场景

空中VLN涉及无人机等飞行器的导航控制。

<div align="center">
  <img src="/images/vln_aerial.png" width="100%" />
<figcaption>

室外VLN示例
</figcaption>
</div>
**应用示例**：
- 无人机巡检
- 空中搜救
- 航拍导航



# VLN主流数据集

VLN研究依赖高质量的数据集来训练和评估导航模型。以下是VLN领域最具影响力的主流数据集（含最新进展）：

---

## 指令导向数据集 

指令导向任务(Instruction-guided)是 VLN 的核心，重点在于将复杂的自然语言指令映射到具体的环境动作序列中。

---
## 1. VLN (2018)
### 1.1 R2R (Room-to-Room)

* **发布时间**：2018 (CVPR)
* **环境表示**：**离散拓扑图 (Discrete Graph)**。基于 Matterport3D 扫描的真实场景。
* **核心挑战**：跨模态对齐（Cross-modal Alignment），要求智能体在复杂的真实图像中识别指令提及的地标。

<div align="center">
  <img src="/images/R2R.png" width="100%" />
<figcaption>
R2R数据集概览
</figcaption>
</div>


**[数据集目录结构]**

```text
R2R/
├── data/
│   ├── R2R_train.json          # 训练集：14,025 条指令
│   ├── R2R_val_seen.json       # 已见环境：与训练集场景重合，考量记忆力
│   ├── R2R_val_unseen.json     # 未见环境：全新场景，考量泛化性 (最关键指标)
│   └── R2R_test.json           # 测试集：榜单评测专用，隐藏 GT 路径
├── connectivity/               # 拓扑连接图 (定义 Agent 可移动的范围)
│   └── <Scan_ID>_connectivity.json 
└── img_features/               # 视觉特征 (主流采用 ViT-B/16 或 ResNet 离线提取)
    └── <Scan_ID>.tsv           # 存储各视点 (viewpoint) 的全景特征向量

```

**[数据条目与底层逻辑解析]**
R2R 的 JSON 不仅仅是文本，它包含了导航初始化的关键位姿信息：

```json
{
  "scan": "2n8P_example",          // 场景 ID (对应 Matterport3D 中的房屋)
  "path": ["vp_1", "vp_2", "vp_3"],// 离散路径节点序列 (Ground Truth)
  "heading": 1.57,                 // 初始水平偏航角 (Radians)，决定 Agent 第一眼看哪
  "instructions": [                // 每条路径对应的 3 条独立人类标注 (多样性)
    "Leave the bedroom and go into the hallway...",
    "Walk past the bathroom and stop near the stairs.",
    "Go through the door and walk to the end of the hall."
  ],
  "instr_id": "1234_0"             // 格式：{path_id}_{instruction_index}
}

```

**[关键技术细节：拓扑连接文件]**
这是离散 VLN 的核心，`connectivity.json` 定义了智能体在每个点位可以看到的邻居节点：

```json
// <Scan_ID>_connectivity.json 内部逻辑示例
{
  "image_id": "vp_1",
  "rel_heading": 0.52,             // 目标点相对于当前的水平夹角
  "rel_elevation": 0.1,            // 目标点相对于当前的俯仰角
  "distance": 2.1,                 // 节点间欧氏距离 (米)
  "unobstructed": true             // 路径是否通畅 (无墙壁阻隔)
}

```

**[核心评估指标 (Metrics)]**
在整理 R2R 时，必须包含这四个核心指标：

* **NE (Navigation Error)**: 预测终点与真值终点的平均距离 (m)，越低越好。
* **SR (Success Rate)**: 终点误差小于 3m 的比例，越高越好。
* **SPL (Success weighted by Path Length)**: **核心指标**。权衡导航效率与准确度，避免智能体通过“乱绕路”碰巧到达终点。
* **OSR (Oracle Success Rate)**: 路径中任意一点靠近过目标的比例，衡量模型是否曾“经过”正确答案。


---

### 1.2 R4R (Room-for-Room)

* **发布时间**：2019 (EMNLP)
* **核心特点**：通过拼接 R2R 路径形成更长的轨迹。
* **技术突破**：引入了 **CLS (Coverage weighted by Length Score)** 指标，要求模型必须“严格遵循指令路径”而不仅仅是到达终点。

**[数据格式差异]**

* **路径构成**：将两条 R2R 路径首尾相连，平均路径步数从 4-6 步增加到 10-15 步。
* **JSON 补充**：增加了 `path_id` 追踪原始 R2R 路径来源。

---

### 1.3 RxR (Room-across-Room)

* **发布时间**：2020 (EMNLP)
* **核心特点**：多语言支持（英语、印地语、泰卢固语）及**细粒度对齐**。

**[数据集目录结构]**

```text
RxR/
├── annotations/
│   ├── en-US/                  # 英语指令文件夹
│   ├── hi-IN/                  # 印地语指令文件夹
│   └── te-IN/                  # 泰卢固语指令文件夹
├── poses/                      # 指令与视点的细粒度对齐数据 (Pose Trace)
└── rxr_train_guide.json        # 训练引导文件

```

**[关键技术点：Pose Trace]**

* **对齐数据**：RxR 不仅提供指令，还记录了标注员在写指令时视线停留的时间戳。
* **JSON 字段**：包含 `pose_trace` 数组，记录了 `(time, view_index)`，允许进行多模态的时间序列对齐训练。

---

---
## 2. VLN-CE (2020)
————Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments

* **发布时间**：2020 (ECCV)
* **环境表示**：**连续环境 (Continuous Environment)**。基于 Habitat 模拟器渲染 Matterport3D 场景，使用低层动作控制（0.25m 前进，15° 转向）。
* **核心特点**：将离散拓扑图导航转换为连续空间导航，移除了预先构建导航图、完美定位和瞬移假设，更贴近真实机器人场景。

📄 **Paper**: https://arxiv.org/abs/2004.02857

<div align="center">
  <img src="/images/VLN-CE-comparison.png" width="100%" />
<figcaption>
VLN 与 VLN-CE 的对比: VLN 基于固定拓扑的全景图节点(左)，而 VLN-CE 在连续环境中使用低层动作(右)
</figcaption>
</div>

**[数据集目录结构]**

```text
data/
├── datasets/
│   ├── R2R_VLNCE_v1-3/              # R2R 数据集转换版本
│   │   ├── train/
│   │   │   └── train.json.gz        # 训练集（4,475 条轨迹）
│   │   ├── val_seen/
│   │   │   └── val_seen.json.gz     # 已见环境验证集
│   │   └── val_unseen/
│   │       └── val_unseen.json.gz   # 未见环境验证集
│   │
│   ├── RxR_VLNCE_v0/                # RxR 多语言版本
│   │   ├── train/
│   │   │   ├── train_guide.json.gz           # Guide 轨迹
│   │   │   ├── train_guide_gt.json.gz        # Ground Truth
│   │   │   ├── train_follower.json.gz        # Follower 轨迹
│   │   │   └── train_follower_gt.json.gz
│   │   ├── val_seen/
│   │   ├── val_unseen/
│   │   └── text_features/                    # BERT 预编码特征
│   │
├── scene_datasets/
│   └── mp3d/                        # Matterport3D 场景资源
│       ├── <scan_id>.glb            # 场景网格模型
│       └── <scan_id>.navmesh        # 可导航网格
│
└── ddppo-models/                    # 预训练强化学习模型
```

**[数据格式示例]**

VLN-CE 保留 R2R 的指令和路径信息，但将离散节点路径转换为连续轨迹：

```json
{
  "episode_id": 1234,
  "scene_id": "2n8kARJN3HM",
  "trajectory_id": "4321",
  "instruction": {
    "instruction_text": "Walk past the bathroom and stop near the stairs.",
    "instruction_tokens": ["walk", "past", "the", "bathroom", ...]
  },
  "reference_path": [              // 离散参考路径（来自 R2R）
    "viewpoint_1",
    "viewpoint_2",
    "viewpoint_3"
  ],
  "start_position": [1.2, 0.15, 3.4],  // 连续空间起始坐标 (x, y, z)
  "start_rotation": [0, 1.57, 0, 0],   // 四元数表示的初始朝向
  "goals": [                           // 目标位置（可能有多个）
    {
      "position": [5.6, 0.15, 8.2],
      "radius": 3.0                    // 成功判定半径（米）
    }
  ],
  "shortest_paths": [                  // 预计算的最短路径动作序列
    [
      {"action": "MOVE_FORWARD", "rotation": 0},
      {"action": "TURN_LEFT", "rotation": 15},
      {"action": "MOVE_FORWARD", "rotation": 0},
      ...
    ]
  ],
  "info": {
    "geodesic_distance": 9.89,         // 最短路径长度（米）
    "euclidean_distance": 7.32
  }
}
```

**[关键技术特性]**

* **轨迹转换方法**：通过射线投射和 A* 路径验证，将 77% 的 R2R 离散路径成功转换为连续环境轨迹
* **动作空间**：`MOVE_FORWARD (0.25m)`, `TURN_LEFT (15°)`, `TURN_RIGHT (15°)`, `STOP`
* **观测空间**：RGB (480×640) + Depth (480×640)，视场角 (FoV) 79°
* **物理约束**：支持碰撞检测、可导航网格 (NavMesh)、Agent 高度 1.5m
* **Habitat 集成**：利用 Habitat-Sim 高性能渲染（1000+ FPS），支持分布式训练

**[核心评估指标]**

VLN-CE 采用与 R2R 一致的评估指标，但在连续空间中重新定义：

* **NE (Navigation Error)**: 最终位置与目标的欧式距离（米），越低越好
* **SR (Success Rate)**: 终点误差 < 3m 的轨迹比例，越高越好
* **SPL (Success weighted by Path Length)**: 路径效率加权成功率 = SR × (最短路径长度 / 实际路径长度)
* **OSR (Oracle Success Rate)**: 轨迹中任意位置曾接近目标（< 3m）的比例

**[性能基准]**

| 模型 | Val Unseen SR | Val Unseen SPL | 备注 |
|------|--------------|----------------|------|
| Seq2Seq | 18% | 0.16 | 基础模型 |
| CMA (Cross-Modal Attention) | 32% | 0.30 | 最佳基线 |
| 无深度输入 | ≤1% | - | 性能崩溃 |
| 无指令输入 | 17% | - | 单模态基线 |

**核心发现**：深度信息对 VLN-CE 至关重要，移除深度导致性能崩溃；平均轨迹长度从 VLN 的 4-6 步增加到 55.88 步。

---
## 3. VLN-PE (2025)
————Rethinking the Embodied Gap: Physical and Visual Disparities in VLN

* **发布时间**：2025 (ICCV)
* **环境表示**：**物理真实连续环境 (Physically Realistic Environment)**。基于 GRUTopia 物理模拟器 (Isaac Sim)，支持真实的运动动力学和物理交互。
* **核心特点**：首个支持多种机器人具身（人形/四足/轮式）的 VLN 平台，引入物理控制器和真机部署验证，揭示了仿真到真实的具身化差距。

📄 **Paper**: https://arxiv.org/abs/2507.13019v2

<div align="center">
  <img src="/images/VLN-PE-evolution.png" width="100%" />
<figcaption>
VLN 任务的演进: 从 oracle-based 导航(2018)到 VLN-CE 连续导航(2020)，再到 VLN-PE 物理真实导航(2025)
</figcaption>
</div>

**[数据集目录结构]**

```text
VLN-PE/
├── datasets/
│   ├── R2R-filtered/                # 过滤楼梯场景的 R2R
│   │   ├── train/                   # 8,679 个 episodes
│   │   ├── val_seen/                # 658 个 episodes
│   │   └── val_unseen/              # 1,347 个 episodes
│   │
│   ├── GRU-VLN10/                   # 新增合成家居场景
│   │   ├── train/                   # 441 个 episodes
│   │   ├── val_seen/                # 111 个 episodes
│   │   └── val_unseen/              # 1,287 个 episodes
│   │
│   └── 3DGS-Lab-VLN/                # 3D Gaussian Splatting 渲染实验室
│       ├── train/                   # 160 个 episodes
│       └── val/                     # 640 个 episodes
│
├── scenes/
│   ├── mp3d/                        # 90 个 Matterport3D 场景
│   ├── GRScenes/                    # 10 个高质量合成场景
│   └── 3DGS/                        # 3DGS 在线渲染场景
│
├── robots/
│   ├── humanoid/                    # 人形机器人配置
│   │   ├── unitree_h1/              # Unitree H1 (相机高度 ~1.5m)
│   │   └── unitree_g1/              # Unitree G1
│   ├── quadruped/                   # 四足机器人
│   │   └── unitree_aliengo/         # Unitree Aliengo (相机高度 ~0.5m)
│   └── wheeled/                     # 轮式机器人
│       └── jetbot/                  # NVIDIA Jetbot
│
└── controllers/
    ├── physical_controller/         # RL-based 物理控制器
    └── simple_controller/           # 简化运动控制器
```

**[数据格式特点]**

VLN-PE 扩展了 VLN-CE 数据格式，新增机器人具身和物理状态信息：

```json
{
  "episode_id": 5678,
  "scene_id": "GRScene_001",
  "instruction": "Walk to the living room and find the red pillow.",
  "robot_type": "humanoid_h1",           // 机器人类型
  "controller_type": "physical",         // 控制器类型
  "camera_height": 1.5,                  // 相机高度（米）
  "start_position": [2.3, 0.0, 4.1],
  "start_rotation": [0, 0.785, 0, 0],
  "goal_position": [8.7, 0.0, 9.2],
  "goal_radius": 3.0,
  "lighting_condition": "normal",        // 光照条件: normal/low/high
  "sensor_config": {
    "rgb": true,
    "depth": true,                       // 是否包含深度
    "resolution": [270, 480]
  }
}
```

**[关键技术特性]**

* **跨具身支持**：统一接口支持人形（H1, G1）、四足（Aliengo）和轮式（Jetbot）机器人，各具身可独立训练或联合训练
* **物理控制器**：基于 RL 训练的低层控制器，模拟真实运动动力学（步态、平衡、碰撞响应）
* **多场景融合**：101 个场景（90 MP3D + 10 GRScene + 1 定制），支持光照变化和 3DGS 渲染
* **真机验证**：在 Unitree Go2 四足机器人上进行 14 个室内场景的实际部署测试
* **标准化格式**：兼容 LeRobot v2.1 格式（InternData-N1），便于跨平台使用

**[核心评估指标]**

VLN-PE 保留传统指标并新增物理真实性指标：

* **TL (Trajectory Length)**: 轨迹总长度（米）
* **NE (Navigation Error)**: 最终距离目标的误差（米）
* **SR (Success Rate)**: 成功率（< 3m）
* **SPL (Success weighted by Path Length)**: 路径效率加权成功率
* **OSR (Oracle Success Rate)**: 曾接近目标的比例
* **FR (Fall Rate)**: 机器人跌倒的比例（物理真实性指标）⭐
* **StR (Stuck Rate)**: 机器人卡住的比例（碰撞/动力学失败）⭐

**[性能基准 - Humanoid H1 on R2R-filtered Val Unseen]**

| 模型 | 参数量 | SR (%) | SPL | FR (%) | StR (%) | 备注 |
|------|--------|--------|-----|--------|---------|------|
| Seq2Seq-Full (VLN-CE) | 36M | 15.2 | 0.13 | 8.3 | 12.1 | 零样本迁移 |
| CMA-Full (VLN-CE) | 36M | 18.7 | 0.16 | 7.5 | 10.8 | 零样本迁移 |
| NaVid (零样本) | 7B | 22.4 | 0.19 | 6.2 | 9.3 | 大模型 |
| CMA (VLN-PE 训练) | 36M | 25.8 | 0.22 | 3.8 | 5.2 | 域内训练 |
| RDP (Diffusion Policy) | 6M | 27.1 | 0.23 | 2.9 | 4.7 | 新方法 |
| CMA+ (跨具身训练) | 36M | **28.7** | **0.24** | **2.1** | **3.9** | 最佳性能 |

**核心发现**：
1. **零样本迁移失败**：VLN-CE 模型迁移到 VLN-PE 时 SR 相对下降 34%
2. **跨具身泛化**：联合训练单一模型可在所有机器人类型上达到 SOTA
3. **多模态鲁棒性**：RGB+Depth 在低光照下性能下降仅 1-2%，而纯 RGB 下降 12.47%
4. **真机验证成功**：VLN-PE 训练模型在真实 Unitree Go2 上 SR 达到 28.57%


---

## 4. VLN-N1(2025)
————Synthetic Data for InternVLA-N1

* **发布时间**：2025
* **环境表示**：**连续环境 (Continuous Environment)**。基于 VLN-CE 等导航数据集转换，采用统一的 LeRobotDataset 格式。
* **核心特点**：标准化的机器人学习数据格式，支持视频、指令、动作和元数据的结构化存储，兼容多种导航基准测试。

<div align="center">
  <img src="/images/InternNav.png" width="100%" />
<figcaption>
InternNav 数据集架构
</figcaption>
</div>

**[数据集组成]**
* 3dfront_d435i / 3dfront_zed
  - 说明：基于 3D-FRONT 室内设计数据集。包含大量合成的家具布置，适合训练物体识别与空间布局理解。

* gibson_d435i / gibson_zed
  - 说明：基于 Gibson 环境。由真实建筑扫描而成，是机器人导航和具身智能（Embodied AI）最主流的数据集之一。

* hm3d_d435i / hm3d_zed
  - 说明：Habitat-Matterport 3D 数据集。由 Meta 发布，包含 1000 个超高分辨率的 3D 扫描场景，模型精细度极高。

* hssd_d435i / hssd_zed
  - 说明：Habitat Synthetic Scene Dataset。Meta 开发的高质量合成数据集，场景布局更符合真实居家逻辑，用于提升导航算法泛化性。

* matterport3d_d435i / matterport3d_zed
  - 说明：经典的 Matterport3D 场景。包含 90 个大型建筑的完整扫描数据，是早期视觉导航研究的基石。

* replica_d435i / replica_zed
  - 说明：Replica 数据集。只有 18 个场景，但重建质量极高，拥有极其密集的网格和语义标注，适合测试高精度 SLAM。

* mp3d_d435i / mp3d_zed
  - 说明：Matterport3D 的缩写版本，通常与全景图处理或特定的导航任务相关联。

* scannet_d435i / scannet_zed
  - 说明：ScanNet 数据集。包含超过 1500 个扫描房间，重点在于室内语义分割和物体实例标注。

**[数据合成流程]**

| 阶段 | 流程名称 | 核心操作与技术实现 |
| :--- | :--- | :--- |
| **01** | **轨迹数据渲染合成** | 基于场景资产、全局地图和本体信息，利用传统运动控制方法（Motion Control）设置规则，自动化合成机器人移动轨迹。 |
| **02** | **语料标注与改写** | 利用大语言模型（LLM）对轨迹视频进行语义解析，生成初版导航指令；随后根据特定任务需求进行指令微调与润色。 |
| **03** | **数据质量筛选** | 基于轨迹中包含的有意义语义信息及物体数量进行分档打分，强制滤除 0 分数据。 |

---

**详细阶段说明**

**（1）轨迹数据渲染合成 (Trajectory Rendering)**
* **输入支撑**：场景资产 (Assets)、全局地图 (Global Map)、机器人本体参数 (Robot Configuration)。
* **合成逻辑**：通过预设规则的运动控制算法，在仿真环境中生成符合物理规律的导航路径。
* **自定义建议**：在此阶段可配置自定义相机内参（如 $f_x, f_y, c_x, c_y$）以匹配实际硬件。

**（2）语料标注和改写 (Instruction Generation)**
* **描述生成**：调用 LLM 对合成的轨迹视频进行“视觉到语言”的转换，形成初始自然语言指令。
* **指令优化**：针对复杂场景进行语言改写，提升指令的丰富度与对环境特征的覆盖率。

**（3）数据筛选 (Data Filtering & Quality Control)**
* **量化评分**：
  - 依据轨迹内涉及的有效语义信息、地标物体数量进行打分。
  - 评分体系分为三档，设定阈值过滤无效样本。
* **成效总结**：
  - **效率提升**：最终滤除 23% 的低质量数据，显著降低训练成本。
  - **性能表现**：筛选后的高质量、多元化场景数据确保了模型性能的可扩展性（Scalability）。


**[数据集目录结构]**

```text
<datasets_root>/
│
├── <sub_dataset_1>/              # 环境级数据集 (如 3dfront_zed)
│   ├── <scene_dataset_1>/        # 场景级数据集
│   │   ├── <traj_dataset_1>/     # 轨迹级数据集
│   │   │   ├── data/             # 结构化 episode 数据 (.parquet)
│   │   │   │   └── chunk-000/
│   │   │   │       └── episode_000000.parquet
│   │   │   │
│   │   │   ├── meta/             # 元数据与统计信息
│   │   │   │   ├── episodes_stats.jsonl  # 每个 episode 的特征统计
│   │   │   │   ├── episodes.jsonl        # Episode 元数据 (任务、指令等)
│   │   │   │   ├── info.json             # 数据集级别配置信息
│   │   │   │   └── tasks.jsonl           # 任务定义
│   │   │   │
│   │   │   └── videos/           # 观测视频
│   │   │       └── chunk-000/
│   │   │           ├── observation.images.depth/    # 深度图序列
│   │   │           │   ├── 0.png
│   │   │           │   ├── 1.png
│   │   │           │   └── ...
│   │   │           ├── observation.images.rgb/      # RGB 图像序列
│   │   │           │   ├── 0.jpg
│   │   │           │   ├── 1.jpg
│   │   │           │   └── ...
│   │   │           ├── observation.video.depth/     # 深度视频
│   │   │           │   └── episode_000000.mp4
│   │   │           └── observation.video.trajectory/# RGB 轨迹视频
│   │   │               └── episode_000000.mp4
│   │   │
│   │   ├── <traj_dataset_2>/
│   │   └── ...
│   │
│   ├── <scene_dataset_2>/
│   └── ...
│
├── <sub_dataset_2>/
└── ...
```

**[核心元数据文件解析]**

**1. episodes_stats.jsonl** - 每个 episode 的特征统计

```json
{
  "episode_index": 0,
  "stats": {
    "observation.images.rgb": {
      "min": [[[x]], [[x]], [[x]]],      // 最小像素值
      "max": [[[x]], [[x]], [[x]]],      // 最大像素值
      "mean": [[[x]], [[x]], [[x]]],     // 平均值
      "std": [[[x]], [[x]], [[x]]],      // 标准差
      "count": [300]                      // 帧数
    },
    "observation.images.depth": {...},
    "action": {...}
  }
}
```

**2. episodes.jsonl** - Episode 索引与任务描述

```json
{
  "episode_index": 0,
  "tasks": [
    "Go straight down the hall and up the stairs. When you reach the door to the gym, go left into the gym and stop..."
  ],
  "length": 57                           // 该 episode 的总帧数
}
```

**3. info.json** - 数据集全局配置

```json
{
  "codebase_version": "v2.1",            // LeRobot 格式版本
  "robot_type": "unknown",               // 机器人平台类型
  "total_episodes": 1,
  "total_frames": 152,
  "fps": 30,                             // 视频与状态采集帧率
  "splits": {"train": "0:503"},          // 数据集划分
  "features": {                          // 特征模式定义
    "observation.images.rgb": {
      "dtype": "image",
      "shape": [270, 480, 3],            // [height, width, channels]
      "names": ["height", "width", "channel"]
    },
    "observation.camera_intrinsic": {    // 相机内参矩阵 (3×3)
      "dtype": "float32",
      "shape": [3, 3]
    },
    "observation.path_points": {         // 轨迹点云 (N×3)
      "dtype": "float64",
      "shape": [36555, 3],
      "names": ["x", "y", "z"]
    },
    "action": {                          // 动作变换矩阵 (4×4)
      "dtype": "float32",
      "shape": [4, 4]
    }
  }
}
```

**4. tasks.jsonl** - 任务自然语言描述

```json
{
  "task_index": 0,
  "task": "Go straight to the hallway and then turn left. Go past the bed. Veer to the right and go through the white door. Stop when you're in the doorway."
}
```

**[关键技术特性]**

* **格式统一化**：将离散节点路径转换为连续的相机轨迹 + 动作序列
* **多模态融合**：同时存储 RGB、深度图、点云、相机参数
* **高效存储**：Parquet 格式支持快速索引，MP4 视频便于可视化
* **扩展性强**：通过继承 `NavDataset` 和 `NavDatasetMetadata` 类适配导航任务特性

**[核心评估指标]**

InternNav 保留 VLN-CE 的标准指标，同时支持 LeRobot 框架的训练评估：

* **SR (Success Rate)**: 终点误差 < 3m 的成功率
* **SPL (Success weighted by Path Length)**: 路径效率加权成功率
* **Oracle Success Rate**: 轨迹中任意点接近目标的比例
* **DtG (Distance to Goal)**: 最终距离目标的平均距离
* 
---
## 3D 情景数据

---
## 目标导向数据集

目标导向任务(Object-grounded)在路径导航的基础上增加了物体定位和语义理解的要求，更接近真实应用场景。


### 1. REVERIE (Remote Embodied Visual Referring Expression in Real Indoor Environments)

* **发布时间**：2020 (CVPR)
* **环境表示**：基于 Matterport3D 的离散拓扑图
* **核心挑战**：远程物体定位 + 跨模态指代消解（Referring Expression + Navigation）

**[任务定义与创新点]**

REVERIE 是 VLN 领域首个将 **导航** 与 **物体定位** 深度融合的数据集，智能体需要：
1. 根据自然语言指令导航到目标房间
2. 在全景视图中识别并定位指令中提及的远程目标物体（目标物体在初始位置不可见）
3. 物体候选来自所有可能视点的全景图像，而非单张图片

**[数据集目录结构]**

```text
REVERIE/
├── data/
│   ├── REVERIE_train.json       # 10,466 条训练指令
│   ├── REVERIE_val_seen.json    # 已见环境验证集
│   └── REVERIE_val_unseen.json  # 未见环境验证集
├── annotations/
│   └── bbox/                     # Matterport3D 物体边界框标注
│       └── <Scan_ID>_bbox.json  # 每个场景的物体实例信息
└── img_features/                # 物体区域特征（Faster R-CNN 提取）
    └── <Scan_ID>_obj.tsv

```

**[核心数据解析]**

REVERIE 在 R2R 基础上扩展了物体接地（grounding）标注：

```json
{
  "id": 1234,
  "scan": "2n8P_example",
  "path": ["vp_1", "vp_2", "vp_3"],       // 导航路径（与 R2R 相同）
  "heading": 1.57,
  "instructions": [
    "Walk to the living room and find the red pillow on the couch."
  ],
  "objId": 78,                            // 目标物体 ID（关键新增字段）
  "obj_name": "pillow",                   // 物体类别名称
  "viewpoint": "vp_3",                    // 目标物体所在的最佳观测视点
  "bbox": {                               // 物体边界框（像素坐标）
    "image_id": "vp_3_idx_12",            // 全景图中的视角索引
    "x": 120, "y": 200, "w": 50, "h": 60
  }
}

```

**[关键技术点：物体标注机制]**

* **物体库**：每个 Matterport3D 场景包含预标注的物体实例（来自 Matterport3D Object Annotations），共涉及 4,140 个不同物体实例，21,702 条指令。
* **全景视图挑战**：与传统 RefExp 任务在单张图片中选择不同，REVERIE 要求从 **所有可能视点的 36 个方向** 中定位物体。
* **视点依赖性**：同一物体从不同视点观察外观会显著变化（遮挡、光照、角度），增加了视觉识别难度。

**[核心评估指标]**

REVERIE 使用 **三级评估体系**：

* **RGS (Remote Grounding Success)**：**核心指标**。同时满足两个条件：
  1. 导航成功（终点与目标视点距离 < 3m）
  2. 物体定位成功（预测物体 ID 与真实 objId 一致）
* **RGSPL (RGS weighted by Path Length)**：在 RGS 基础上加入路径效率惩罚。
* **SR (Success Rate)**：仅评估导航部分，与 R2R 中的 SR 定义相同（终点误差 < 3m）。

**[技术难点]**

1. **长距离指代消解**：物体在初始位置不可见，需要结合语言推理和空间记忆。
2. **多模态对齐**：需要同时理解"房间级导航指令"（如"去客厅"）和"物体级描述"（如"沙发上的红色枕头"）。
3. **视点选择**：智能体需要学会在目标房间选择最佳观测角度来识别物体。

---

### 2. SOON (Scenario Oriented Object Navigation)

* **发布时间**：2021 (CVPR)
* **环境表示**：基于 Matterport3D 的连续 3D 环境
* **核心挑战**：场景级描述理解 + 任意起点导航（From Anywhere to Object）

**[任务定义与创新点]**

SOON 突破了传统 ObjectNav 固定起点的限制，提出了更贴近真实场景的任务设定：
* **场景描述导航**：不提供逐步指令，仅给出目标物体及其周围环境的语义描述（如"客厅角落的书架旁边有一个蓝色花瓶"）
* **任意起点**：智能体可以从场景中的任意位置开始导航，而非固定起点
* **零样本泛化**：强调对未见过的物体类别和场景布局的理解能力

**[数据集目录结构]**

```text
SOON/
├── data/
│   └── FAO/                      # From Anywhere to Object 数据集
│       ├── train.json            # 训练集
│       ├── val_seen.json         # 已见场景验证集
│       └── val_unseen.json       # 未见场景验证集
├── scene_datasets/               # Matterport3D 场景文件
└── semantic_annotations/         # 语义场景图标注
    └── <Scan_ID>_semantic.json  # 物体关系与属性标注

```

**[核心数据解析]**

SOON 引入了富含语义信息的场景描述，避免目标歧义：

```json
{
  "episode_id": "FAO_001",
  "scene_id": "17DRP5sb8fy",
  "target_object": {
    "object_id": "obj_42",
    "category": "vase",
    "attributes": "blue, ceramic"          // 物体属性
  },
  "scene_description": "In the corner of the living room, next to the bookshelf, there is a blue ceramic vase on a small round table.",
  "description_components": {               // 结构化描述
    "object_attribute": "blue ceramic vase",
    "object_relationship": "next to the bookshelf",
    "region_description": "corner of the living room",
    "nearby_region": "near the fireplace"
  },
  "start_position": [x, y, z],              // 任意起点（非固定）
  "start_rotation": [qw, qx, qy, qz]
}

```

**[关键技术点：语义场景图]**

* **四级描述体系**：
  1. **物体属性**（Object Attribute）：颜色、材质、尺寸等
  2. **物体关系**（Object Relationship）：空间关系（旁边、上方、里面）
  3. **区域描述**（Region Description）：所在房间或区域
  4. **邻近区域**（Nearby Region）：周围地标或参考物

* **FAO 数据集规模**：3,848 条指令，词汇量 1,649 个单词，覆盖多种物体类别和场景配置。

**[核心评估指标]**

* **Success Rate (SR)**：智能体到达目标物体 1m 范围内的成功率。
* **SPL (Success weighted by Path Length)**：结合路径效率的成功率。
* **DTS (Distance To Success)**：失败案例中，终点与目标的平均距离。
* **Zero-shot Generalization**：在未见物体类别上的成功率，评估语义理解能力。

**[技术难点]**

1. **语义推理**：需要理解物体属性、空间关系等高层语义概念。
2. **场景记忆**：由于起点不固定，智能体需要快速建立场景的全局认知。
3. **描述消歧**：在包含多个相似物体的场景中，精确定位符合描述的目标。

---

### 3. LHPR-VLN (Long-Horizon Planning and Reasoning in VLN)

* **发布时间**：2025 (CVPR)
* **环境表示**：Habitat Simulator + 连续 3D 环境（216 个复杂场景）
* **核心挑战**：超长程规划（150步） + 多阶段任务分解 + 决策一致性

**[任务定义与创新点]**

LHPR-VLN 是首个专门针对 **长视距导航** 设计的数据集，填补了 VLN 领域在长程规划研究上的空白：
* **超长路径**：平均 150 个动作步（相比 R2R 的 4-6 步，增长 25 倍）
* **多阶段任务**：指令包含多个连贯的子任务（如"先去厨房拿杯子，然后去客厅，最后到卧室"）
* **决策一致性**：要求智能体在长时间导航过程中保持对任务目标的记忆和理解

**[数据集目录结构]**

```text
LHPR-VLN/
├── episodes/
│   ├── train/                   # 3,260 个长视距任务
│   │   └── episode_*.json.gz
│   ├── val_seen/
│   └── val_unseen/
├── scenes/                      # 216 个复杂 3D 场景
│   └── <Scene_ID>/
│       ├── mesh.ply             # 场景网格
│       └── semantic.ply         # 语义标注
└── data_generation/             # NavGen 自动生成平台配置
    └── config.yaml

```

**[核心数据解析]**

LHPR-VLN 引入了多阶段任务结构和细粒度步骤标注：

```json
{
  "episode_id": "LHPR_001",
  "scene_id": "scene_complex_42",
  "instruction": "First, go to the kitchen and pick up a cup from the counter. Then, walk to the living room and place it on the coffee table. Finally, head to the bedroom and sit on the bed.",
  "instruction_length": 18.17,              // 平均指令长度（单词数）
  "num_steps": 152,                         // 总步数（平均 150 步）
  "sub_tasks": [                            // 多阶段任务分解
    {
      "task_id": 1,
      "description": "Go to kitchen, pick up cup",
      "start_step": 0,
      "end_step": 45,
      "goal_position": [x1, y1, z1]
    },
    {
      "task_id": 2,
      "description": "Walk to living room, place cup",
      "start_step": 46,
      "end_step": 98,
      "goal_position": [x2, y2, z2]
    },
    {
      "task_id": 3,
      "description": "Head to bedroom, sit on bed",
      "start_step": 99,
      "end_step": 152,
      "goal_position": [x3, y3, z3]
    }
  ],
  "start_position": [x0, y0, z0],
  "start_rotation": [qw, qx, qy, qz],
  "action_sequence": [                      // 完整的动作序列
    "MOVE_FORWARD", "TURN_LEFT", ...        // 150 个动作
  ]
}

```

**[关键技术点：NavGen 数据生成平台]**

* **双向生成**：结合 top-down（从场景语义生成任务）和 bottom-up（从路径生成指令）两种策略
* **多粒度标注**：包含任务级、子任务级、步骤级三层标注
* **复杂场景构建**：216 个场景专门设计为包含多个房间和复杂空间结构

**[核心评估指标]**

* **SR (Success Rate)**：完成所有子任务并到达最终目标的成功率（< 3m）
* **PSPL (Progressive Success weighted by Path Length)**：**新指标**。评估每个子任务的完成情况和路径效率
* **Task Completion Rate (TCR)**：完成的子任务占总子任务的比例
* **Decision Consistency Score (DCS)**：衡量智能体在长程导航中是否保持对目标的一致理解

**[技术难点]**

1. **记忆管理**：在 150 步的导航过程中保持对初始指令和中间目标的记忆
2. **层次化规划**：需要将长指令分解为多个子目标，并协调执行
3. **累积误差**：长路径中的小错误会累积，导致偏离正确轨迹
4. **计算资源**：训练和推理成本显著高于短路径任务

---

## 对话式导航数据集

对话式导航(Dialog-based Navigation)允许智能体通过多轮交互主动获取信息，模拟人类在不确定情况下的问询行为。

---

### 1. CVDN (Cooperative Vision-and-Dialog Navigation)

* **发布时间**：2019 (CoRL - Conference on Robot Learning)
* **环境表示**：Matterport3D 离散拓扑图（基于 R2R 环境）
* **核心挑战**：主动问询 + 对话历史建模 + 不确定性下的导航决策

**[任务定义与创新点]**

CVDN 引入了 **人机协作** 的导航范式，智能体（Navigator）可以在导航过程中向 Oracle 提问：
* **Navigator**：只能看到当前视觉观测，需要通过提问获取导航帮助
* **Oracle**：拥有最短路径的特权信息，但不能主动提供，只能回答 Navigator 的问题
* **对话交互**：平均 4.5 轮对话，Navigator 需要学会何时提问、问什么问题

**[数据集目录结构]**

```text
CVDN/
├── data/
│   ├── train/
│   │   ├── dialogs.json         # 2,050+ 条人类对话标注
│   │   └── navigation.json      # 对应的导航路径
│   ├── val_seen/
│   └── val_unseen/
├── tasks/
│   └── NDH/                      # Navigation from Dialog History 任务
│       ├── train.json            # 基于对话历史的导航数据
│       └── val.json
└── pretrained/
    └── oracle_model/             # 预训练的 Oracle 模型

```

**[核心数据解析]**

CVDN 数据包含 **完整的对话过程** 和 **导航轨迹**：

```json
{
  "dialog_id": "CVDN_001",
  "scan": "2n8P_example",
  "target": {
    "object": "blue chair",
    "viewpoint": "vp_final"
  },
  "start_viewpoint": "vp_1",
  "start_heading": 0.0,
  "dialog_history": [              // 人类标注的对话过程
    {
      "turn": 1,
      "message": "I'm in a bedroom. Where should I go?",
      "speaker": "navigator",
      "viewpoint_at_turn": "vp_1"
    },
    {
      "turn": 2,
      "message": "Go through the door and turn right.",
      "speaker": "oracle",
      "oracle_action": "vp_2"       // Oracle 知道的最佳下一步
    },
    {
      "turn": 3,
      "message": "I see a hallway. Am I close?",
      "speaker": "navigator",
      "viewpoint_at_turn": "vp_2"
    },
    {
      "turn": 4,
      "message": "Yes, the chair is in the next room on your left.",
      "speaker": "oracle",
      "oracle_action": "vp_final"
    }
  ],
  "trajectory": ["vp_1", "vp_2", "vp_final"],
  "success": true
}

```

**[关键技术点：NDH 任务]**

CVDN 提出了 **Navigation from Dialog History (NDH)** 子任务：
* 给定目标物体和人类对话历史
* 智能体需要理解对话内容，推断目标位置
* 在未探索的环境中执行导航
* 核心难点：对话指代消解（"那个房间"、"左边"等指代如何映射到环境）

**[核心评估指标]**

* **Goal Progress (GP)**：智能体是否向目标位置移动（距离减少）
* **SR (Success Rate)**：到达目标 3m 范围内的成功率
* **SPL (Success weighted by Path Length)**：路径效率惩罚的成功率
* **Dialog Efficiency**：平均需要多少轮对话才能成功导航（越少越好）
* **Question Quality**：提问是否有效（是否获得了有用信息）

**[技术难点]**

1. **主动学习**：智能体需要学会在何时提问（不确定性高时）以及提问策略
2. **对话历史建模**：需要记忆和理解多轮对话的上下文
3. **指代消解**：对话中的"这里"、"那边"等指代需要映射到视觉环境
4. **Oracle 建模**：训练时需要模拟 Oracle 的回答策略

---

### 2. TEACh (Task-driven Embodied Agents that Chat)

* **发布时间**：2022 (AAAI)（arXiv 首次发布于 2021 年 10 月）
* **环境表示**：AI2-THOR 模拟器 + 可交互家居环境
* **核心挑战**：任务级对话 + 物体交互 + 状态变化（如切菜、煮咖啡）

**[任务定义与创新点]**

TEACh 是首个支持 **物体交互和状态变化** 的对话式导航数据集：
* **Commander（指挥者）**：拥有任务的完整信息，通过对话指导 Follower
* **Follower（执行者）**：从第一人称视角观察环境，执行导航和物体操作动作
* **任务复杂度**：从简单的"煮咖啡"到复杂的"准备早餐"（包含多个子任务）
* **物体交互**：支持拾取（PickUp）、放置（Place）、切片（Slice）、加热（Heat）等 20+ 种动作

**[数据集目录结构]**

```text
TEACh/
├── data/
│   ├── train/                   # 3,000+ 人类对话任务
│   │   ├── edh_instances/       # Execution from Dialog History
│   │   └── tfd_instances/       # Talk-through, then Follow-through Demonstration
│   ├── valid_seen/
│   └── valid_unseen/
├── images/                      # 第一人称视角图像序列
│   └── <episode_id>/
│       └── frame_*.jpg
├── object_states/               # 物体状态变化追踪
│   └── <episode_id>.json
└── evaluation/
    └── metrics/                 # 任务完成度评估脚本

```

**[核心数据解析]**

TEACh 数据包含 **完整的任务执行过程** 和 **对话交互**：

```json
{
  "instance_id": "TEACh_train_001",
  "task_type": "Coffee",                   // 任务类型
  "task_description": "Make a cup of coffee and place it on the dining table.",
  "scene_id": "FloorPlan1",
  "dialog": [
    {
      "turn": 1,
      "utterance": "First, go to the coffee machine on the counter.",
      "speaker": "commander",
      "timestamp": 0.0
    },
    {
      "turn": 2,
      "utterance": "I see the coffee machine. Should I press the button?",
      "speaker": "follower",
      "timestamp": 5.2
    },
    {
      "turn": 3,
      "utterance": "Yes, fill the mug with coffee, then take it to the table.",
      "speaker": "commander",
      "timestamp": 7.5
    }
  ],
  "actions": [                             // 执行的动作序列
    {
      "action": "MoveAhead",
      "success": true,
      "position": [x, y, z],
      "rotation": [rx, ry, rz],
      "frame": "frame_001.jpg"
    },
    {
      "action": "PickupObject",
      "object_id": "Mug_001",
      "success": true,
      "frame": "frame_015.jpg"
    },
    {
      "action": "PourInto",                // 状态变化动作
      "object_id": "Mug_001",
      "receptacle": "CoffeeMachine_001",
      "success": true,
      "frame": "frame_032.jpg"
    },
    {
      "action": "PutObject",
      "object_id": "Mug_001",
      "receptacle": "DiningTable_001",
      "success": true,
      "frame": "frame_078.jpg"
    }
  ],
  "initial_state": {                       // 初始环境状态
    "Mug_001": {"isFilled": false, "isHot": false, "position": [x1, y1, z1]}
  },
  "goal_state": {                          // 目标状态
    "Mug_001": {"isFilled": true, "isHot": true, "receptacle": "DiningTable_001"}
  }
}

```

**[关键技术点：EDH 与 TFD 任务]**

* **EDH (Execution from Dialog History)**：
  * 给定 Commander 和 Follower 的对话历史
  * Follower 需要理解对话并执行任务
  * 类似于 CVDN 的 NDH 任务，但增加了物体交互

* **TFD (Two-stage Task)**：
  1. **Talk-through**：Commander 先演示任务，边做边讲解
  2. **Follow-through**：Follower 根据之前的讲解在新场景中执行相同任务
  * 测试从演示中学习的能力

**[核心评估指标]**

* **GC (Goal-Condition Success Rate)**：**核心指标**。所有目标状态是否达成：
  * 正确的物体被放置在正确的位置
  * 物体状态正确（如咖啡是热的、面包被切片）
* **Task Success Rate (TSR)**：主要任务目标是否完成
* **Dialog Score**：对话质量和效率
* **Action Efficiency**：完成任务所需的动作步数
* **State Change Accuracy**：物体状态变化的准确性

**[技术难点]**

1. **长期依赖**：任务平均包含 50+ 个动作步骤，需要长期规划
2. **状态追踪**：需要记忆物体的当前状态（杯子是否装满、炉子是否开启等）
3. **多模态融合**：结合对话、视觉、动作历史做决策
4. **任务泛化**：在未见过的场景和物体配置上执行相同任务类型

---

### 3. HA-VLN (Human-Aware Vision-Language Navigation)

* **发布时间**：2025 (NeurIPS 2024 Datasets and Benchmarks Track, HA-VLN 2.0 发布于 2025年3月)
* **环境表示**：离散（Matterport3D）+ 连续（Habitat）双模式支持
* **核心挑战**：社交感知导航 + 人群避让 + 个人空间保护 + Sim2Real 迁移

**[任务定义与创新点]**

HA-VLN 是首个将 **人类社交行为约束** 引入 VLN 的数据集：
* **社交感知**：智能体需要尊重人类的个人空间（personal space），避免碰撞和过近接触
* **动态人群**：环境中包含移动的人类，执行各种日常活动（walking, sitting, talking）
* **真实验证**：包含真实机器人实验数据，验证 Sim2Real 迁移能力
* **统一基准**：同时支持离散和连续环境，便于不同方法对比

**[数据集目录结构]**

```text
HA-VLN/
├── data/
│   ├── HAPS_2.0/                # Human Activity Pose Sequences 2.0
│   │   ├── motion_sequences/    # 172 种活动的 3D 人体运动序列
│   │   │   └── activity_*/
│   │   │       ├── frames/      # 58,320 帧精确对齐的姿态
│   │   │       └── annotations.json
│   │   └── descriptions/        # 486 个详细的动作描述
│   ├── episodes/
│   │   ├── discrete/            # 离散环境（Matterport3D）
│   │   │   ├── train.json       # 16,844 条社交导航指令
│   │   │   └── val_*.json
│   │   └── continuous/          # 连续环境（Habitat）
│   │       └── episodes.json.gz
│   └── real_world/              # 真实机器人实验数据
│       ├── robot_trajectories/
│       └── human_tracking/
└── simulators/
    ├── HA3D_discrete/           # 离散环境模拟器
    └── HA3D_continuous/         # 连续环境模拟器

```

**[核心数据解析]**

HA-VLN 在导航指令中增加了 **社交约束** 和 **人群信息**：

```json
{
  "episode_id": "HA-VLN_001",
  "scan": "2n8P_example",
  "instruction": "Walk through the living room to the kitchen, but avoid getting too close to the person sitting on the couch.",
  "path": ["vp_1", "vp_2", "vp_3"],
  "humans": [                              // 动态人类信息
    {
      "human_id": "person_01",
      "activity": "sitting on couch",      // 当前活动
      "motion_sequence": "HAPS_sitting_01", // 对应的运动序列
      "trajectory": [                      // 时空轨迹
        {"time": 0.0, "position": [x1, y1, z1], "orientation": [r1]},
        {"time": 1.0, "position": [x2, y2, z2], "orientation": [r2]},
        ...
      ],
      "personal_space_radius": 1.2         // 个人空间半径（米）
    },
    {
      "human_id": "person_02",
      "activity": "walking to kitchen",
      "motion_sequence": "HAPS_walking_03",
      "trajectory": [...]
    }
  ],
  "social_constraints": {                  // 社交约束
    "min_distance_to_humans": 1.0,         // 最小保持距离
    "avoid_blocking_paths": true,          // 避免阻挡他人路径
    "priority_to_humans": true             // 人类优先通行
  }
}

```

**[关键技术点：HAPS 2.0 数据集]**

* **活动类别**：172 种日常活动（walking, sitting, reaching, talking, reading 等）
* **精确对齐**：486 个高质量 3D 人体运动模型，经过人工验证确保动作-描述对齐
* **时空标注**：58,320 帧姿态数据，包含精确的时间戳和空间坐标
* **多人交互**：支持多人协同活动（如对话、传递物品）

**[核心评估指标]**

HA-VLN 2.0 引入了 **社交感知评估体系**：

* **SA-SR (Social-Aware Success Rate)**：**核心新指标**。同时满足：
  1. 导航成功（到达目标 < 3m）
  2. 无社交违规（未进入他人个人空间）
  3. 无碰撞（与人类保持安全距离）

* **Personal Space Violation Rate (PSVR)**：违反个人空间的频率
* **Collision Rate (CR)**：与人类发生碰撞的次数
* **Path Efficiency with Social Cost (PESC)**：结合路径长度和社交代价的综合指标
* **Sim2Real Transfer Success**：真实机器人实验的成功率

**[技术难点]**

1. **动态预测**：需要预测人类未来的移动轨迹，提前规划避让路径
2. **社交规范建模**：不同文化和场景下的个人空间定义可能不同
3. **实时性**：需要在运动的人群中快速做出导航决策
4. **Sim2Real Gap**：模拟器中的人类行为与真实世界存在差异
5. **多目标优化**：在导航效率和社交安全之间权衡

**[真实世界验证]**

HA-VLN 2.0 包含真实机器人实验：
* 在实际室内环境部署导航机器人
* 与真实人类交互，验证算法的安全性和有效性
* 提供了宝贵的 Sim2Real 迁移数据

---

## 需求导向数据集

需求导向导航(Demand-driven Navigation)要求智能体理解用户的抽象需求（如"我想喝咖啡"），并自主推理需要找到的物体。

---

### DDN (Demand-driven Navigation)

* **发布时间**：2023-2024（基于 ProcThor 数据集）
* **环境表示**：AI2-THOR + ProcThor 程序化生成的室内环境
* **核心挑战**：需求理解 + 常识推理 + 物体功能性映射

**[任务定义与创新点]**

DDN 突破了传统"明确物体导航"的限制，模拟真实场景中的高层需求：
* **抽象需求输入**：用户不说"找到咖啡机"，而是说"我想喝咖啡"或"我需要清洁工具"
* **物体功能推理**：智能体需要理解哪些物体可以满足需求（咖啡机、速溶咖啡、法式压壶都能满足"喝咖啡"的需求）
* **常识知识**：需要丰富的常识知识库（如"咖啡机通常在厨房""清洁工具可能在储藏室"）

**[数据集目录结构]**

```text
DDN/
├── data/
│   ├── train.json               # 1,692 条需求导向指令
│   ├── val.json                 # 241 条验证指令
│   └── test.json                # 485 条测试指令
├── scenes/
│   ├── train/                   # 600 个场景（200个/split）
│   │   └── <Scene_ID>.json      # ProcThor 场景配置
│   ├── val/
│   └── test/
├── demand_ontology/             # 需求本体（知识图谱）
│   ├── demand_categories.json   # 需求分类（饮食、清洁、娱乐等）
│   └── object_functions.json    # 物体-功能映射表
└── object_categories/           # 109 个物体类别定义
    └── category_definitions.json

```

**[核心数据解析]**

DDN 数据强调 **需求到物体的映射**：

```json
{
  "episode_id": "DDN_001",
  "scene_id": "ProcThor_train_042",
  "demand": "I want to make coffee.",        // 用户需求（自然语言）
  "demand_category": "food_beverage",        // 需求类别
  "acceptable_objects": [                    // 可接受的目标物体（多个）
    "CoffeeMachine",
    "InstantCoffee",
    "FrenchPress"
  ],
  "preferred_object": "CoffeeMachine",       // 首选物体
  "required_properties": {                   // 物体需满足的属性
    "functional": true,                      // 必须可用
    "accessible": true                       // 必须可触及
  },
  "common_locations": [                      // 常见位置（常识）
    "Kitchen",
    "DiningRoom"
  ],
  "start_position": [x, y, z],
  "start_rotation": [rx, ry, rz],
  "ground_truth_path": [...]                 // 参考路径（到首选物体）
}

```

**[关键技术点：需求本体]**

* **需求分类体系**：
  * 饮食需求（Food & Beverage）：喝咖啡、吃饭、切菜
  * 清洁需求（Cleaning）：打扫、擦地、洗碗
  * 娱乐需求（Entertainment）：看电视、读书
  * 工作需求（Work）：打电话、使用电脑

* **物体-功能映射**：
  ```json
  {
    "demand": "clean floor",
    "objects": [
      {"name": "VacuumCleaner", "priority": 1, "effectiveness": 0.9},
      {"name": "Mop", "priority": 2, "effectiveness": 0.7},
      {"name": "Broom", "priority": 3, "effectiveness": 0.5}
    ]
  }
  ```

* **常识推理链**：
  * 需求："我想喝咖啡" → 物体推理："需要咖啡机或速溶咖啡" → 位置推理："通常在厨房" → 导航规划

**[核心评估指标]**

* **DSR (Demand Success Rate)**：**核心指标**。找到任意可满足需求的物体（< 1m）
* **PSR (Preferred Success Rate)**：找到首选物体的成功率
* **Reasoning Accuracy**：需求→物体映射的准确性
* **Location Prediction Accuracy**：预测物体位置的准确性
* **SPL (Success weighted by Path Length)**：结合路径效率

**[技术难点]**

1. **需求歧义消解**：同一需求可能对应多个物体，需要根据场景选择最合适的
2. **常识知识集成**：需要大量常识知识（物体功能、常见位置、使用场景）
3. **零样本泛化**：对未见过的需求类型进行推理
4. **多目标决策**：当多个物体都可满足需求时，如何选择最优目标
5. **知识库构建**：如何构建和维护需求-物体-位置的知识图谱

**[与 VLN 的区别]**

| 维度 | 传统 VLN | DDN |
|------|----------|-----|
| 输入 | "去厨房找咖啡机" | "我想喝咖啡" |
| 目标 | 明确的物体/位置 | 抽象的需求 |
| 推理 | 语言→路径映射 | 需求→物体→路径多级映射 |
| 知识 | 视觉-语言对齐 | 常识知识 + 物体功能性 |

---

## 特殊场景数据集

特殊场景数据集突破了室内导航的限制，探索无人机、城市航拍等新兴应用场景。

---

### 1. AerialVLN (Vision-and-Language Navigation for UAVs)

* **发布时间**：2023 (ICCV)
* **环境表示**：3D 模拟器 + 近真实感城市场景渲染（25 个城市场景）
* **核心挑战**：三维空间推理 + 高度控制 + 城市地标识别

**[任务定义与创新点]**

AerialVLN 是首个专为 **无人机（UAV）** 设计的 VLN 数据集：
* **三维导航**：需要同时控制水平位置和飞行高度
* **空中视角**：俯视和斜视视角与地面导航完全不同
* **城市环境**：包含建筑物、道路、公园、工厂等多样化城市场景
* **高密度物体**：870+ 种不同物体类别，远超室内数据集

**[数据集目录结构]**

```text
AerialVLN/
├── data/
│   ├── AerialVLN-S/             # AerialVLN-Simulator 数据集
│   │   ├── train.json           # 8,446 条飞行轨迹
│   │   ├── val_seen.json
│   │   └── val_unseen.json
│   └── trajectories/
│       └── <Episode_ID>/
│           ├── waypoints.json   # 轨迹关键点
│           └── actions.json     # 飞行动作序列
├── scenes/
│   ├── downtown/                # 市中心场景
│   ├── factory/                 # 工厂区场景
│   ├── park/                    # 公园场景
│   └── village/                 # 乡村场景
├── annotations/
│   ├── landmarks/               # 地标标注（建筑名称、特征）
│   └── objects/                 # 870+ 物体类别标注
└── pilot_data/                  # AOPA 持证飞行员标注数据
    └── human_trajectories.json

```

**[核心数据解析]**

AerialVLN 需要处理 **三维空间的飞行路径**：

```json
{
  "episode_id": "AerialVLN_001",
  "scene_id": "downtown_city_01",
  "instruction": "Fly over the blue rooftop building, then descend to 15 meters and head towards the park with the fountain.",
  "instruction_length": 22,
  "trajectory": [                          // 三维轨迹
    {
      "waypoint_id": 0,
      "position": [x0, y0, z0],            // z 轴为高度
      "heading": 90.0,                     // 水平朝向（度）
      "pitch": -15.0,                      // 俯仰角（负值为向下看）
      "altitude": 30.0,                    // 海拔高度（米）
      "timestamp": 0.0
    },
    {
      "waypoint_id": 1,
      "position": [x1, y1, z1],
      "heading": 120.0,
      "pitch": -20.0,
      "altitude": 25.0,
      "timestamp": 5.3
    },
    ...
  ],
  "landmarks_mentioned": [                 // 指令中提及的地标
    {
      "name": "blue rooftop building",
      "category": "building",
      "position": [xb, yb, zb],
      "visibility_range": 50.0             // 可见距离（米）
    },
    {
      "name": "park with fountain",
      "category": "outdoor_area",
      "position": [xp, yp, zp]
    }
  ],
  "action_space": {                        // 飞行动作空间
    "horizontal": ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "HOVER"],
    "vertical": ["ASCEND", "DESCEND", "MAINTAIN_ALTITUDE"]
  },
  "pilot_certified": true                  // 是否由持证飞行员标注
}

```

**[关键技术点：AOPA 认证飞行员标注]**

* **专业性**：所有轨迹由 AOPA（Aircraft Owners and Pilots Association）持证飞行员记录
* **安全性**：轨迹符合飞行安全规范（避障、高度控制、速度限制）
* **真实性**：飞行模式符合真实无人机的物理特性

**[多样化场景类型]**

* **Downtown（市中心）**：高楼林立，需要在建筑间导航
* **Factory（工厂区）**：大型工业设施，烟囱、仓库等地标
* **Park（公园）**：开阔区域，树木、池塘、雕塑等自然地标
* **Village（乡村）**：低密度建筑，农田、道路等特征

**[核心评估指标]**

* **SR (Success Rate)**：到达目标位置的成功率（3D 欧氏距离 < 5m）
* **ALT-E (Altitude Error)**：**新指标**。高度控制误差（米）
* **SPL (Success weighted by Path Length)**：3D 路径长度惩罚
* **Landmark Recognition Accuracy**：地标识别准确率
* **Collision Rate**：与建筑物或障碍物的碰撞率

**[技术难点]**

1. **三维空间推理**：需要同时理解"向前飞"和"上升/下降"的空间关系
2. **视角变化**：不同高度和俯仰角下，同一地标的外观差异巨大
3. **地标消歧**：城市中可能有多个相似的建筑物（如多个蓝色屋顶）
4. **安全约束**：需要避免碰撞、保持安全高度、遵守飞行限制区域
5. **长距离导航**：城市环境尺度大，导航距离远超室内场景

**[与室内 VLN 的对比]**

| 维度 | 室内 VLN (R2R) | AerialVLN |
|------|----------------|-----------|
| 空间维度 | 2D（平面移动） | 3D（含高度） |
| 视角 | 第一人称水平视角 | 俯视 + 斜视 |
| 地标密度 | 稀疏（房间、家具） | 密集（870+ 物体） |
| 场景尺度 | 小（单个建筑） | 大（城市街区） |
| 动作空间 | 前进 + 旋转 | 前进 + 旋转 + 升降 |

---

### 2. CityNav (Language-Goal Aerial Navigation Dataset with Geographic Information)

* **发布时间**：2025 (ICCV)（arXiv 于 2024 年 6 月首次发布）
* **环境表示**：真实城市航拍图像 + 地理语义地图（GSM）
* **核心挑战**：真实世界泛化 + 地标空间关系理解 + 地理信息融合

**[任务定义与创新点]**

CityNav 是首个基于 **真实城市** 的大规模空中 VLN 数据集：
* **真实场景**：覆盖 4.65 km² 实际城市区域（英国剑桥和伯明翰）
* **人类演示**：32,637 条人类飞行员标注的真实轨迹
* **地理语义地图（GSM）**：结合地理信息（地标位置、道路网络）辅助导航
* **零样本挑战**：需要在真实世界的复杂性和不确定性下导航

**[数据集目录结构]**

```text
CityNav/
├── data/
│   ├── trajectories/
│   │   ├── cambridge/           # 剑桥市轨迹（16,000+ 条）
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   └── test.json
│   │   └── birmingham/          # 伯明翰市轨迹（16,000+ 条）
│   │       └── ...
│   └── geographic_maps/
│       ├── GSM_cambridge.json   # 剑桥地理语义地图
│       └── GSM_birmingham.json  # 伯明翰地理语义地图
├── aerial_images/               # 真实航拍图像序列
│   └── <Episode_ID>/
│       ├── frame_*.jpg          # 第一人称视角航拍图像
│       └── metadata.json        # GPS 坐标、时间戳
├── landmarks/                   # 城市地标数据库
│   ├── landmark_database.json   # 地标名称、类别、GPS 坐标
│   └── landmark_images/         # 地标参考图像
└── annotations/
    ├── spatial_relations.json   # 地标间的空间关系标注
    └── instruction_annotations.json

```

**[核心数据解析]**

CityNav 结合了 **真实航拍图像** 和 **地理信息**：

```json
{
  "episode_id": "CityNav_Cambridge_001",
  "city": "Cambridge",
  "instruction": "Fly from the market square towards King's College Chapel, then turn left at the River Cam and follow it northward.",
  "instruction_length": 25,
  "trajectory": [
    {
      "waypoint_id": 0,
      "gps": {"lat": 52.2053, "lon": 0.1218, "alt": 50.0},  // GPS 坐标
      "heading": 45.0,
      "image": "frame_000.jpg",
      "timestamp": "2024-06-15T10:30:00Z"
    },
    {
      "waypoint_id": 1,
      "gps": {"lat": 52.2042, "lon": 0.1167, "alt": 48.0},
      "heading": 38.0,
      "image": "frame_015.jpg",
      "timestamp": "2024-06-15T10:30:23Z"
    },
    ...
  ],
  "landmarks_in_instruction": [            // 指令中的地标
    {
      "name": "Market Square",
      "type": "public_space",
      "gps": {"lat": 52.2054, "lon": 0.1190},
      "osm_id": "way/123456789"            // OpenStreetMap ID
    },
    {
      "name": "King's College Chapel",
      "type": "historic_building",
      "gps": {"lat": 52.2042, "lon": 0.1165},
      "osm_id": "way/987654321"
    },
    {
      "name": "River Cam",
      "type": "waterway",
      "gps": {"lat": 52.2035, "lon": 0.1180},  // 中心线坐标
      "osm_id": "way/111222333"
    }
  ],
  "geographic_semantic_map": {             // 地理语义地图信息
    "landmark_locations": [...],           // 地标位置列表
    "road_network": [...],                 // 道路网络拓扑
    "spatial_relations": [                 // 地标间的空间关系
      {
        "landmark_1": "Market Square",
        "landmark_2": "King's College Chapel",
        "relation": "southwest_of",
        "distance": 580.0                  // 米
      },
      {
        "landmark_1": "King's College Chapel",
        "landmark_2": "River Cam",
        "relation": "east_of",
        "distance": 120.0
      }
    ]
  }
}

```

**[关键技术点：地理语义地图（GSM）]**

* **地标定位**：提供城市中所有主要地标的精确 GPS 坐标
* **空间关系**：预计算的地标间方位关系（north_of, southwest_of 等）
* **道路网络**：城市道路的拓扑结构，辅助路径规划
* **多模态输入**：GSM 可作为额外的输入模态，与视觉观测结合

**[GSM 的作用]**

```json
// GSM 提供的辅助信息示例
{
  "query": "Where is King's College Chapel relative to Market Square?",
  "gsm_response": {
    "direction": "southwest",
    "distance": 580.0,
    "intermediate_landmarks": ["Senate House", "Great St Mary's Church"]
  }
}
```

**[核心评估指标]**

* **SR (Success Rate)**：到达目标区域的成功率（GPS 误差 < 10m）
* **GPS-DTG (GPS Distance To Goal)**：终点与目标的 GPS 距离（米）
* **SPL (Success weighted by Path Length)**：基于 GPS 路径长度的 SPL
* **Landmark Recognition Accuracy**：正确识别指令中地标的准确率
* **Spatial Relation Understanding**：理解地标间空间关系的准确率

**[技术难点]**

1. **真实世界复杂性**：
   * 天气变化（阴天、晴天、雨天）
   * 光照变化（不同时间、季节）
   * 遮挡（树木、云层、建筑阴影）

2. **地标歧义**：
   * 城市中可能有多个相似建筑
   * 地标外观随视角变化显著

3. **长距离导航**：
   * 覆盖 4.65 km²，导航距离可达数千米
   * 需要全局路径规划能力

4. **跨城市泛化**：
   * 不同城市的建筑风格、道路布局差异大
   * 需要泛化到未见过的城市

5. **多模态融合**：
   * 如何有效融合视觉观测和地理语义地图
   * 在 GPS 不可用时如何纯视觉导航

**[CityNav vs AerialVLN]**

| 维度 | AerialVLN | CityNav |
|------|-----------|---------|
| 场景 | 模拟场景（近真实感） | 真实城市航拍 |
| 规模 | 25 个场景, 8,446 轨迹 | 2 个城市, 32,637 轨迹 |
| 覆盖面积 | 相对较小 | 4.65 km² |
| 地理信息 | 无 | GSM（地标、道路网络） |
| 挑战重点 | 三维空间推理 | 真实世界泛化 |
| 数据来源 | 持证飞行员标注 | 真实飞行数据 |

**[应用场景]**

* 城市无人机配送导航
* 无人机巡检（基础设施、建筑）
* 搜索救援任务（根据语言描述的位置快速定位）
* 航空摄影（根据拍摄需求规划飞行路径）

---

### 3. OpenFly (A Comprehensive Platform for Aerial Vision-Language Navigation)

* **发布时间**：2025 (arXiv 首次发布于 2025 年 2 月)
* **环境表示**：多引擎集成（Unreal Engine + GTA V + Google Earth + 3D Gaussian Splatting）
* **核心挑战**：大规模数据 + 多样化场景 + 自动化工具链 + 关键帧感知

**[任务定义与创新点]**

OpenFly 是迄今为止 **最大规模** 的空中 VLN 平台：
* **海量数据**：100,000 条飞行轨迹，是 AerialVLN 和 CityNav 总和的 3 倍
* **多引擎支持**：整合 4 种不同的渲染引擎，覆盖从游戏级到照片级的真实感
* **自动化工具链**：高度自动化的数据采集、场景分割、轨迹生成、指令标注流程
* **18 个场景**：覆盖城市、乡村、山区、海岸等多种地形
* **多样化高度和长度**：轨迹高度从 10m 到 200m，长度从 50m 到 5km

**[数据集目录结构]**

```text
OpenFly/
├── data/
│   ├── trajectories/
│   │   ├── unreal_engine/       # Unreal Engine 渲染场景（30,000 条）
│   │   ├── gta_v/               # GTA V 场景（25,000 条）
│   │   ├── google_earth/        # Google Earth 真实场景（25,000 条）
│   │   └── 3d_gaussian/         # 3D Gaussian Splatting 场景（20,000 条）
│   └── split/
│       ├── train.json           # 训练集（80,000 条）
│       ├── val.json             # 验证集（10,000 条）
│       └── test.json            # 测试集（10,000 条）
├── scenes/                      # 18 个多样化场景
│   ├── urban_downtown/
│   ├── suburban_residential/
│   ├── rural_countryside/
│   ├── mountain_region/
│   ├── coastal_area/
│   └── ...
├── toolchain/                   # 自动化数据生成工具链
│   ├── point_cloud_processor/   # 点云获取与处理
│   ├── semantic_segmentation/   # 场景语义分割
│   ├── trajectory_generator/    # 飞行轨迹创建
│   └── instruction_generator/   # GPT-4o 指令生成
├── keyframe_annotations/        # 关键帧标注
│   └── <Episode_ID>_keyframes.json
└── openfly_agent/               # OpenFly-Agent 模型代码
    ├── model/
    └── configs/

```

**[核心数据解析]**

OpenFly 引入了 **关键帧（Keyframe）** 的概念：

```json
{
  "episode_id": "OpenFly_UE_12345",
  "engine": "unreal_engine",            // 渲染引擎
  "scene": "urban_downtown_02",
  "instruction": "Take off from the parking lot, fly north along Main Street, ascend to 50 meters when you reach the clock tower, then circle around the stadium and land on the rooftop helipad.",
  "instruction_source": "GPT-4o",       // 指令由 GPT-4o 生成
  "trajectory_stats": {
    "length_meters": 1250.0,
    "duration_seconds": 180.0,
    "max_altitude": 52.0,
    "min_altitude": 5.0,
    "num_waypoints": 85
  },
  "keyframes": [                        // 关键帧（重点观测点）
    {
      "keyframe_id": 0,
      "waypoint_id": 0,
      "description": "parking lot - takeoff point",
      "importance": 0.95,               // 重要性评分（0-1）
      "reason": "navigation_start",
      "position": [x0, y0, z0],
      "image": "frame_000.jpg"
    },
    {
      "keyframe_id": 1,
      "waypoint_id": 22,
      "description": "clock tower - altitude reference",
      "importance": 0.88,
      "reason": "landmark_mentioned",   // 指令中提及的地标
      "position": [x1, y1, z1],
      "image": "frame_022.jpg"
    },
    {
      "keyframe_id": 2,
      "waypoint_id": 57,
      "description": "stadium - circling point",
      "importance": 0.92,
      "reason": "action_change",        // 动作模式变化（直飞→盘旋）
      "position": [x2, y2, z2],
      "image": "frame_057.jpg"
    },
    {
      "keyframe_id": 3,
      "waypoint_id": 84,
      "description": "rooftop helipad - landing zone",
      "importance": 0.98,
      "reason": "navigation_goal",
      "position": [x3, y3, z3],
      "image": "frame_084.jpg"
    }
  ],
  "full_trajectory": [
    {"waypoint_id": 0, "position": [x0, y0, z0], ...},
    {"waypoint_id": 1, "position": [...], ...},
    ...
    {"waypoint_id": 84, "position": [x84, y84, z84], ...}
  ],
  "engine_metadata": {
    "rendering_quality": "high",
    "weather": "clear",
    "time_of_day": "noon"
  }
}

```

**[关键技术点：自动化工具链]**

OpenFly 的核心创新是 **高度自动化** 的数据生成流程：

1. **点云获取（Point Cloud Acquisition）**：
   * 从不同引擎提取 3D 场景点云
   * 支持多种格式（.pcd, .ply, .las）

2. **场景语义分割（Semantic Segmentation）**：
   * 自动识别建筑物、道路、树木、水体等类别
   * 生成语义标签用于地标识别

3. **飞行轨迹创建（Trajectory Generation）**：
   * 基于场景拓扑自动生成可行飞行路径
   * 考虑安全高度、避障、平滑度等约束

4. **指令生成（Instruction Generation）**：
   * 将轨迹和第一人称图像输入 GPT-4o
   * 生成自然语言描述："从...起飞，沿着...飞行，到达..."
   * 确保指令与视觉观测一致

**[OpenFly-Agent：关键帧感知模型]**

OpenFly 提出了 **关键帧感知（Keyframe-Aware）** 的 VLN 模型：
* **动机**：长轨迹中并非所有帧都同等重要，关键帧包含更多导航信息
* **方法**：
  * 自动识别关键观测帧（地标出现、动作变化、导航节点）
  * 对关键帧赋予更高的注意力权重
  * 减少计算开销（只处理关键帧而非所有帧）

**[多引擎对比]**

| 引擎 | 真实感 | 物理准确性 | 场景多样性 | 数据量 |
|------|--------|-----------|-----------|--------|
| Unreal Engine | 高 | 高 | 中 | 30,000 |
| GTA V | 中-高 | 中 | 高（城市） | 25,000 |
| Google Earth | 照片级 | 低（静态） | 最高（全球） | 25,000 |
| 3D Gaussian | 照片级 | 低 | 中 | 20,000 |

**[核心评估指标]**

* **SR (Success Rate)**：标准成功率（< 5m）
* **KF-SR (Keyframe Success Rate)**：**新指标**。在关键帧位置的导航准确性
* **SPL (Success weighted by Path Length)**：路径效率
* **Keyframe Attention Score**：模型对关键帧的注意力分配准确性
* **Cross-Engine Generalization**：跨引擎泛化能力（在一个引擎训练，在另一个测试）

**[技术难点]**

1. **跨引擎泛化**：
   * 不同引擎的渲染风格、物理特性差异大
   * 需要学习引擎无关的导航策略

2. **关键帧识别**：
   * 如何自动识别哪些帧是关键帧
   * 关键帧的重要性如何量化

3. **长距离规划**：
   * 轨迹长度跨度大（50m - 5km）
   * 需要多尺度的规划策略

4. **指令质量控制**：
   * GPT-4o 生成的指令可能包含幻觉或不一致
   * 需要自动化验证和过滤机制

5. **计算效率**：
   * 100,000 条轨迹的训练规模巨大
   * 需要高效的数据加载和模型训练策略

**[OpenFly 的独特价值]**

* **规模最大**：100k 轨迹是目前空中 VLN 数据集中最大的
* **工具开源**：提供完整的数据生成工具链，便于社区扩展
* **多引擎支持**：可以研究跨领域迁移和鲁棒性
* **关键帧创新**：引入新的建模思路，提高长轨迹导航效率

---

## 数据集对比总览

| 数据集 | 发布年份 | 任务类型 | 环境类型 | 数据规模 | 核心创新 | 主要指标 |
|--------|----------|----------|----------|----------|----------|----------|
| **R2R** | 2018 | 指令导向 | 室内离散 | 14,025 指令 | VLN 奠基数据集 | SR, SPL, NE |
| **R4R** | 2019 | 指令导向 | 室内离散 | 长路径拼接 | 路径忠诚度评估 | CLS, nDTW, SDTW |
| **RxR** | 2020 | 指令导向 | 室内离散 | 126k 指令（多语言） | 细粒度时空对齐 | SR, SPL, DTW |
| **VLN-CE** | 2020 | 指令导向 | 室内连续 | 基于 R2R 转换 | 连续动作空间 | SR, SPL, DTS |
| **REVERIE** | 2020 | 目标导向 | 室内离散 | 10,466 指令, 4,140 物体 | 导航+物体定位 | RGS, RGSPL |
| **SOON** | 2021 | 目标导向 | 室内连续 | 3,848 指令 | 场景描述+任意起点 | SR, SPL, DTS |
| **LHPR-VLN** | 2025 | 目标导向 | 室内连续 | 3,260 任务, 平均 150 步 | 长程多阶段规划 | SR, PSPL, TCR |
| **CVDN** | 2019 | 对话导航 | 室内离散 | 2,050+ 对话 | 主动问询+Oracle | SR, SPL, GP |
| **TEACh** | 2022 | 对话导航 | 室内交互 | 3,000+ 任务对话 | 物体交互+状态变化 | GC, TSR |
| **HA-VLN** | 2025 | 对话导航 | 室内/混合 | 16,844 指令 | 社交感知+人群避让 | SA-SR, PSVR |
| **DDN** | 2023-24 | 需求导向 | 室内连续 | 1,692 需求指令 | 抽象需求推理 | DSR, PSR |
| **AerialVLN** | 2023 | 空中导航 | 城市模拟 | 8,446 轨迹, 25 场景 | 三维空间+无人机 | SR, SPL, ALT-E |
| **CityNav** | 2025 | 空中导航 | 真实城市 | 32,637 轨迹, 4.65 km² | 真实航拍+地理地图 | SR, GPS-DTG |
| **OpenFly** | 2025 | 空中导航 | 多引擎 | 100k 轨迹, 18 场景 | 大规模+关键帧感知 | SR, KF-SR, SPL |

**图例说明**：
- **SR**: Success Rate（成功率）
- **SPL**: Success weighted by Path Length（路径效率加权成功率）
- **CLS**: Coverage weighted by Length Score（路径覆盖度评分）
- **RGS**: Remote Grounding Success（远程物体定位成功率）
- **GC**: Goal-Condition Success（目标条件成功率）
- **SA-SR**: Social-Aware Success Rate（社交感知成功率）
- **ALT-E**: Altitude Error（高度误差）
- **KF-SR**: Keyframe Success Rate（关键帧成功率）


# VLN主流模拟器

VLN研究需要高质量的3D仿真环境来训练和测试导航模型。以下是VLN领域最常用的主流模拟器（含最新更新和趋势）：

## Matterport3D Simulator

**基本信息：**
- **开发者**：Peter Anderson et al.
- **发布时间**：2018年
- **开源地址**：[GitHub](https://github.com/peteanderson80/Matterport3DSimulator)

**核心特点：**
- **真实场景扫描**：基于Matterport3D数据集，包含90个真实室内环境的高精度3D扫描
- **全景视图**：提供360度全景RGB-D图像
- **离散导航**：采用预定义的导航图，智能体在固定视点间移动
- **高效渲染**：优化的渲染引擎，支持快速视觉观测生成
- **经典基准**：R2R、R4R等经典数据集的官方模拟器

**应用场景：**
- 指令导向的室内导航任务（R2R、R4R）
- 离散动作空间的VLN研究
- 基于真实场景的导航模型训练

**优势：**
- 真实感强，场景来自实际建筑扫描
- 与经典VLN数据集无缝集成
- 社区支持完善，大量研究基于此平台

**局限性：**
- 仅支持离散导航，灵活性受限
- 物理交互能力有限
- 场景数量相对较少（90个环境）

---

## Habitat

**基本信息：**
- **开发者**：Facebook AI Research (FAIR)
- **发布时间**：2019年（最新3.1版本2024–2025年更新）
- **开源地址**：[GitHub](https://github.com/facebookresearch/habitat-lab)

**核心特点：**
- **高性能仿真**：超快速渲染（10,000+ FPS）
- **连续环境**：支持连续动作空间和自由移动
- **多数据集支持**：兼容Matterport3D、Gibson、HM3D、LHPR-VLN等
- **模块化设计**：灵活的任务定义和传感器配置
- **Sim2Real支持**：提供真实机器人部署工具链
- **新特性**：
  - 动态环境支持（移动物体/人群）
  - 空中和户外环境支持
  - 长程任务和复杂子任务支持

**应用场景：**
- 连续动作空间导航研究（VLN-CE）
- 长视距任务（LHPR-VLN）
- 目标导航（ObjectNav）、语义导航（SemanticNav）
- 具身AI和Sim2Real研究

**优势：**
- 仿真速度极快，训练效率高
- 支持连续导航，更贴近真实机器人控制
- 大规模数据集（HM3D 800+场景）
- 动态场景、空中任务支持
- 强大的扩展性和社区生态

**局限性：**
- 配置复杂，学习曲线陡
- 对硬件要求较高（GPU加速）

---

## Isaac Sim / Isaac Lab

**基本信息：**
- **开发者**：NVIDIA
- **核心组件**：
  - **Isaac Sim**：基于 NVIDIA Omniverse 的高保真机器人仿真环境
  - **Isaac Lab**：基于 Isaac Sim 的模块化机器人学习与强化学习框架（GPU 加速）
- **开源地址**：
  - Isaac Lab 文档： https://isaac-sim.github.io/IsaacLab/main/index.html :contentReference[oaicite:0]{index=0}
  - Isaac Sim 官方页面： https://developer.nvidia.com/isaac-sim :contentReference[oaicite:1]{index=1}

**核心特点：**
- **高保真物理与渲染**：基于 RTX 加速的 PhysX 物理引擎与真实感渲染，可模拟碰撞、摩擦、传感器噪声等真实物理特性 :contentReference[oaicite:2]{index=2}
- **机器人学习集成**：Isaac Lab 提供强化学习、模仿学习、策略训练等端到端机器人学习工作流，可批量训练数千个并行环境 :contentReference[oaicite:3]{index=3}
- **多机平台资产库**：包括四旋翼、差分驱动机器人、步态机器人、机械臂等多种机器人模型，可自定义场景与任务 :contentReference[oaicite:4]{index=4}
- **导航与控制支持**：
  - 支持 ROS2、Nav2 等机器人导航栈集成，可用于路径规划与多机器人导航测试 :contentReference[oaicite:5]{index=5}
  - 虽主要用于强化学习与策略训练，但同样可用于评估视觉导航策略、连续控制与视觉感知组合任务
- **数据生成与 Sim‑to‑Real**：结合 Omniverse Replicator，可生成训练用合成数据并辅助现实迁移训练 :contentReference[oaicite:6]{index=6}

**应用场景：**
- 连续控制与导航策略训练（强化学习 / 模仿学习）
- 多传感器 SLAM、视觉感知与导航策略评估
- 多机器人协作与动态环境测试
- 合成数据生成与 Sim‑to‑Real 迁移训练

**优势：**
- **高保真模拟**：比传统离散图导航能更真实模拟连续物理行为与多传感器数据
- **学习框架支持**：内置强化学习训练工作流，可扩展到大规模并行环境
- **集成生态**：与 Omniverse、ROS、RTX GPU 加速等生态联动良好

**局限性：**
- **复杂度高**：上手门槛比简易模拟器如 Habitat、AI2‑THOR 更陡峭
- **计算资源要求高**：需要强 GPU 才能充分利用高保真渲染与物理仿真
- **目前在视觉导航（VLN）Benchmark 领域的专用数据集支持较少**：相比 Matterport/Habitat 等，社区内 VLN benchmark 评测还不如它们成熟

---

## AI2-THOR

**基本信息：**
- **开发者**：Allen Institute for AI
- **发布时间**：2017年（持续更新，最新4.0版本）
- **开源地址**：[官网](https://ai2thor.allenai.org/)

**核心特点：**
- **物理交互**：基于Unity3D，支持完整物理模拟
- **可交互对象**：环境中的物体可抓取、移动、操作
- **多样化场景**：厨房、卧室、客厅、浴室等200+场景
- **语义分割**：内置语义标注和实例分割
- **多智能体支持**：支持同时多个智能体任务
- **新特性**：
  - 多智能体协作
  - 可定制动作和交互
  - 可与VLN-CE、TEACh、EQA数据集结合

**应用场景：**
- 具身问答（EQA）
- 视觉语言导航+操作任务
- 家庭服务机器人研究

**优势：**
- 强大的物理引擎和真实物体交互
- 可多模态任务训练
- API友好，易上手

**局限性：**
- 渲染速度较慢
- 场景规模相对较小
- 资源消耗大

---

## Gibson / iGibson

**基本信息：**
- **开发者**：Stanford University
- **Gibson发布时间**：2018年
- **iGibson发布时间**：2021–2024（最新3.0版本）
- **开源地址**：[iGibson GitHub](https://github.com/StanfordVL/iGibson)

**核心特点：**

**Gibson 1.0**：
- 基于真实建筑扫描（1000+）
- 快速光栅化渲染
- 支持基础物理模拟

**iGibson 3.0**：
- **交互式场景**：完整物理交互和对象操作
- **逼真渲染**：PBR物理渲染
- **语义信息**：丰富的语义标注和物体属性
- **大规模场景**：完整房屋、办公楼等
- **任务多样性**：导航、操作、家务任务
- **新特性**：
  - 动态物体与人群模拟
  - 多智能体与社交导航约束
  - Sim2Real优化

**应用场景：**
- 大规模室内导航
- 导航+操作任务
- Sim2Real迁移研究
- 家庭服务机器人仿真

**优势：**
- 场景数量多，环境多样性高
- 真实感强，基于实际建筑扫描
- iGibson 3.0功能全面，支持复杂交互

**局限性：**
- 安装复杂
- 部分场景质量参差不齐

---

## AirSim

**基本信息：**
- **开发者**：Microsoft
- **发布时间**：2017年（持续更新）
- **开源地址**：[GitHub](https://github.com/microsoft/AirSim)

**核心特点：**
- **无人机/车辆仿真**：面向飞行器和地面车辆
- **高保真物理**：基于Unreal或Unity
- **多传感器支持**：相机、LiDAR、IMU、GPS
- **新特性**：
  - 城市大规模航拍场景
  - 长航程导航、多机协作
  - 与CityNav/OpenFly数据集配合

**应用场景：**
- 空中VLN（AerialVLN）
- 无人机导航与控制
- 自动驾驶与户外导航任务

**优势：**
- 专业飞行器仿真平台
- 高精度物理模拟
- 支持大规模户外环境

**局限性：**
- 室内导航支持有限
- 配置复杂，对硬件要求高

---

## InternUtopia

**基本信息：**
- **开发者**：Shanghai AI Laboratory (上海人工智能实验室)
- **发布时间**：2024年
- **开源地址**：[GitHub](https://github.com/OpenGVLab/InternUtopia)

**核心特点：**
- **大规模开放世界**：支持超大规模城市场景模拟（10+ km²）
- **高保真渲染**：基于Unreal Engine 5的照片级真实感渲染
- **物理交互**：完整的物理引擎，支持动态物体和环境交互
- **多智能体支持**：支持多智能体协同导航和交互任务
- **丰富的动态元素**：包含动态交通流、行人、天气变化等
- **语义信息**：提供详细的场景语义标注和3D边界框
- **多模态感知**：支持RGB、深度、语义分割、LiDAR等多种传感器
- **新特性**：
  - 大规模城市场景的自动生成
  - 实时物理模拟与照片级渲染
  - 支持VLN、具身智能、自动驾驶等多种任务
  - 可扩展的任务定义框架

**应用场景：**
- 大规模城市导航任务
- 开放世界具身智能研究
- 多智能体协作与社交导航
- 自动驾驶与户外导航
- 长距离导航规划

**优势：**
- 超大规模场景支持，适合长程导航研究
- 高保真视觉渲染，接近真实世界
- 动态环境模拟，更贴近实际应用
- 灵活的任务定义和可扩展性
- 多模态传感器支持

**局限性：**
- 计算资源需求极高（需要高性能GPU）
- 配置和使用复杂度较高
- 社区生态相对较新，文档和资源仍在完善

---

## 模拟器对比

| 模拟器 | 环境类型 | 动作空间 | 物理交互 | 渲染速度 | 主要应用 | 场景数量 | 新增特性 (2024–2025) |
|--------|----------|----------|----------|----------|----------|----------|---------------------|
| Matterport3D | 室内 | 离散 | 有限 | 快 | R2R/R4R | 90 | 保持经典基准 |
| Habitat 3.1 | 室内/空中/户外 | 连续 | 基础 | 极快 | VLN-CE, LHPR-VLN | 800+ (HM3D) | 动态物体、空中/长程导航、Sim2Real强化 |
| Isaac Sim / Lab | 室内/室外/空中 | 连续 | 强 | 高 | 强化学习、连续VLN | 可定制 | 高保真物理、动态环境、多机协作、Sim2Real |
| AI2-THOR 4.0 | 室内 | 离散/连续 | 强 | 中等 | 交互任务 | 200+ | 多智能体、可定制交互、家庭场景扩大 |
| iGibson 3.0 | 室内 | 连续 | 强 | 快 | 综合任务 | 1000+ | 动态人群、社交导航、Sim2Real强化 |
| AirSim | 室内外 | 连续 | 强 | 中等 | 无人机/车辆 | 可定制 | 城市航拍、大规模航程、多机协作 |
| InternUtopia | 开放世界/城市 | 连续 | 强 | 中等 | 大规模城市导航 | 可定制 | 超大规模场景、照片级渲染、动态环境 |
| iThorAir / Aerial Sim | 室外/空中 | 连续 | 基础 | 中等 | 空中VLN | 可定制 | 多机协作、长程规划、动态障碍物 |

---

## 选择建议

- **经典VLN基准（R2R/R4R）**：Matterport3D Simulator
- **连续环境与长程任务**：Habitat 3.1
- **需要物理交互任务**：AI2-THOR 4.0 / iGibson 3.0
- **无人机/空中导航**：AirSim / Isaac Sim
- **大规模场景训练**：Gibson/iGibson 或 Habitat + HM3D
- **大规模城市/开放世界导航**：InternUtopia
- **Sim-to-Real部署**：Habitat 3.1 / iGibson 3.0 / Isaac Sim


# 评估指标

评估指标是衡量VLN模型性能的关键工具。一个完善的评估体系应该同时考虑导航的**精度**、**效率**和**轨迹质量**。以下是VLN领域最常用的评估指标：

## 导航精度指标

### Success Rate (SR)

**定义：**
成功到达目标位置的episode比例。

**计算方法：**
$$
SR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[d_i < \tau]
$$

其中：
- $N$ 是测试episode的总数
- $d_i$ 是第 $i$ 个episode结束时智能体与目标位置的距离
- $\tau$ 是成功阈值（通常设为3米）
- $\mathbb{1}[\cdot]$ 是指示函数，条件满足时为1，否则为0

**取值范围：**
- 0 到 1（或0%到100%）
- 越高越好

**优点：**
- 直观易懂，反映任务完成率
- 最常用的主要评估指标

**缺点：**
- 忽略了导航效率（路径长度）
- 对成功阈值敏感

### Navigation Error (NE)

**定义：**
智能体停止时与目标位置的平均距离误差（米）。

**计算方法：**
$$
NE = \frac{1}{N} \sum_{i=1}^{N} d_i
$$

其中：
- $d_i$ 是第 $i$ 个episode结束时智能体与目标的欧氏距离

**取值范围：**
- $[0, +\infty)$ 米
- 越低越好

**优点：**
- 提供连续的性能度量
- 不依赖成功阈值的设定

**缺点：**
- 受场景规模影响较大
- 难以跨数据集比较

### Oracle Success Rate (OSR)

**定义：**
在整个导航轨迹中，智能体曾经距离目标最近的位置是否满足成功条件。

**计算方法：**
$$
OSR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\min_{t} d_i^{(t)} < \tau]
$$

其中：
- $d_i^{(t)}$ 是第 $i$ 个episode在时间步 $t$ 时智能体与目标的距离
- $\min_{t} d_i^{(t)}$ 是整个轨迹中与目标的最小距离

**取值范围：**
- 0 到 1（或0%到100%）
- 越高越好

**优点：**
- 评估智能体是否"到过"目标附近
- 反映路径规划的潜在能力
- 有助于区分"到达但没停"和"从未到达"两种失败情况

**缺点：**
- 不能反映最终导航结果
- 通常作为辅助指标使用

## 导航效率指标

### Success weighted by Path Length (SPL)

**定义：**
考虑路径效率的成功率，同时衡量成功率和路径长度。

**计算方法：**
$$
SPL = \frac{1}{N} \sum_{i=1}^{N} S_i \frac{l_i^*}{\max(p_i, l_i^*)}
$$

其中：
- $S_i$ 是成功指示符（到达目标为1，否则为0）
- $l_i^*$ 是最短路径长度（从起点到终点的理论最短距离）
- $p_i$ 是智能体实际走过的路径长度
- $\max(p_i, l_i^*)$ 确保分母不小于最短路径

**取值范围：**
- 0 到 1
- 越高越好
- SPL = 1 表示以最短路径成功到达目标

**优点：**
- **最重要的综合指标**，同时考虑成功率和效率
- 惩罚绕路行为，鼓励高效导航
- 被广泛用作主要性能指标（与SR并列）

**缺点：**
- 需要计算最短路径（需要环境图信息）
- 对失败的episode惩罚较重（直接计为0）

**典型取值：**
- 早期模型（2018-2019）：R2R上SPL约20-30%
- 中期模型（2020-2021）：R2R上SPL约30-40%
- 近期模型（2022-2024）：R2R上SPL约40-50%
- SOTA模型（2024+）：R2R上SPL可达50%以上

### Coverage weighted by Length Score (CLS)

**定义：**
衡量智能体按照指令覆盖参考路径的程度，同时考虑效率。

**计算方法：**
$$
CLS = \frac{1}{N} \sum_{i=1}^{N} \frac{C_i \times l_i^*}{\max(p_i, l_i^*)}
$$

其中：
- $C_i$ 是覆盖率（智能体经过的参考路径节点比例）
- $l_i^*$ 是参考路径长度
- $p_i$ 是实际路径长度

**取值范围：**
- 0 到 1
- 越高越好

**优点：**
- 评估轨迹与参考路径的匹配度
- 适合评估指令跟随能力

**缺点：**
- 依赖参考路径的质量
- 计算相对复杂

## 轨迹质量指标

### normalized Dynamic Time Warping (nDTW)

**定义：**
衡量智能体轨迹与参考路径之间的相似度，使用动态时间规整算法。

**计算方法：**
$$
nDTW = e^{-\frac{DTW(\mathcal{P}_{agent}, \mathcal{P}_{ref})}{\sigma}}
$$

其中：
- $DTW(\cdot, \cdot)$ 是动态时间规整距离
- $\mathcal{P}_{agent}$ 是智能体的实际轨迹
- $\mathcal{P}_{ref}$ 是参考路径
- $\sigma$ 是归一化参数

**取值范围：**
- 0 到 1
- 越高越好
- nDTW = 1 表示轨迹完全匹配

**优点：**
- 评估轨迹的时序一致性
- 对轨迹的局部偏差容忍度高
- 考虑了路径的整体形状

**缺点：**
- 计算复杂度较高
- 需要时序对齐，计算开销大

### Success weighted by normalized Dynamic Time Warping (SDTW)

**定义：**
结合成功率和轨迹相似度的综合指标。

**计算方法：**
$$
SDTW = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot nDTW_i
$$

**优点：**
- 同时考虑成功率和轨迹质量
- 更全面的性能评估

## 其他辅助指标

### Trajectory Length (TL)

**定义：**
智能体实际走过的平均路径长度。

**用途：**
- 分析导航效率
- 检测模型是否过度探索或原地打转

### Steps Taken

**定义：**
智能体完成任务所需的平均步数。

**用途：**
- 评估导航速度
- 分析决策效率

### Collision Rate

**定义：**
发生碰撞的步数占总步数的比例。

**用途：**
- 评估导航安全性（在连续环境中）
- 检测路径规划质量

### Human Collision Rate

**定义：**
与动态行人发生碰撞的次数（用于社交导航）。

**用途：**
- 评估社交导航能力
- 测试动态避障性能

## 评估指标速查表

### 核心指标总览

| 指标 | 英文全称 | 定义 | 取值范围 | 方向 | 首次提出 | 备注 |
|:----:|:---------|:-----|:--------:|:----:|:--------:|:-----|
| **SR** | **Success Rate** | **终点距目标≤3m的任务比例** | **0-100%** | **↑** | **Anderson et al., 2018** | **最核心指标，反映任务完成率** |
| **SPL** | **Success weighted by Path Length** | **SR × (最短路径/实际路径)** | **0-100%** | **↑** | **Anderson et al., 2018** | **综合成功率和路径效率** |
| **NE** | **Navigation Error** | **智能体终点与目标点的距离（米）** | **0-∞** | **↓** | **Anderson et al., 2018** | **衡量定位精度** |
| **OSR** | **Oracle Success Rate** | **轨迹中任意点距目标≤3m的比例** | 0-100% | ↑ | Anderson et al., 2018 | 评估是否路过正确位置 |
| nDTW | normalized Dynamic Time Warping | 预测轨迹与真实轨迹的归一化对齐距离 | 0-1 | ↑ | Ilharco et al., 2019 | 评估轨迹相似度 |
| SDTW | Success-normalized DTW | nDTW × SR | 0-100 | ↑ | Jain et al., 2019 | 同时考虑成功和轨迹质量 |
| CLS | Coverage weighted by Length Score | 指令覆盖率 × 路径效率 | 0-100% | ↑ | Jain et al., 2019 | 评估指令跟随细粒度 |
| TL | Trajectory Length | 实际行走的路径长度（米） | 0-∞ | - | - | 分析路径效率 |
| **CR** | **Collision Rate** | **发生碰撞的任务比例** | **0-100%** | **↓** | **Krantz et al., 2020** | **VLN-CE核心安全指标** |
| **HCR** | **Human Collision Rate** | **与人类碰撞的任务比例** | **0-100%** | **↓** | **Wei et al., 2025** | **Social-VLN关键指标** |
| FR | Fall Rate | 机器人跌倒的任务比例 | 0-100% | ↓ | Wang et al., 2025 | VLN-PE物理仿真指标 |
| StR | Stuck Rate | 机器人卡住无法移动的比例 | 0-100% | ↓ | Wang et al., 2025 | VLN-PE鲁棒性指标 |

---

### 指标选择速查

**标准评估（必须报告）：**
- **SR + SPL**（所有VLN任务）
- **NE**（定位精度要求高时）

**特定场景补充：**
- **连续环境（VLN-CE）**：+ CR
- **物理仿真（VLN-PE）**：+ FR + StR + TL
- **社交导航（Social-VLN）**：+ HCR
- **轨迹质量研究**：+ nDTW / SDTW
- **指令跟随研究**：+ CLS

---

### 指标权衡关系

**常见矛盾：**
- **SR ↑ vs SPL ↑**：高成功率可能伴随低效路径
- **SR ↑ vs CR ↓**：激进策略提高成功率但增加碰撞
- **SR ↑ vs FR/StR ↓**：探索更多区域增加失败风险

**理想模型特征：**
- SR > 60%（R2R Val-Unseen基准）
- SPL/SR > 0.85（路径效率高）
- CR < 5%（安全导航）
- NE < 5m（精确定位）

---

### 评估最佳实践

**报告规范：**

```markdown
1. 必须分别报告 Val-Seen 和 Val-Unseen
2. 标注成功阈值（默认3m，如有不同需说明）
3. 说明是否使用 ground truth 路径（OSR计算）
4. 标注传感器配置（RGB-only / RGB-D / Panoramic）
```

**公平对比检查清单：**
- ✅ 相同数据集划分
- ✅ 相同成功阈值
- ✅ 相同传感器输入
- ✅ 相同评估环境（Habitat / 真实世界）

---

### 历史演进

| 阶段 | 年份 | 代表工作 | 核心指标 | 新增关注点 |
|:----:|:----:|:---------|:---------|:-----------|
| 1.0 | 2018-2019 | R2R, RxR | SR, SPL, NE | 基础任务完成 |
| 2.0 | 2020-2021 | VLN-CE, nDTW | + CR, nDTW | 连续环境 + 轨迹质量 |
| 3.0 | 2022-2024 | VLN-PE, REVERIE | + FR, StR | 物理真实性 + 多任务 |
| 4.0 | 2025- | Social-VLN, DualVLN | + HCR | 动态环境 + 社交感知 |

**未来趋势：**
- 真实世界部署指标（能耗、时间）
- 长期任务鲁棒性评估
- 人机交互质量指标

---

# 学习资源与框架

**[VLN-Survey-with-Foundation-Models](https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models)** ⭐⭐⭐⭐⭐
- **类型**：GitHub资源仓库
- **重点**：专注于LLM/VLM时代的VLN方法（2023-至今），持续更新最新论文
- **适合**：想了解大模型如何革新VLN领域的研究者

**[Awesome-Embodied-AI](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)** ⭐⭐⭐⭐⭐
- **类型**：全栈资源合集
- **重点**：涵盖VLN、VLA、机器人操作等完整具身智能技术栈
- **适合**：系统学习具身AI全貌的研究者

**[Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide)** ⭐⭐⭐⭐⭐
- **类型**：入门教程 + 实践指南
- **重点**：提供代码实践、论文解读、学习路径规划
- **适合**：零基础入门或需要结构化学习路径的新人

**[Vision-and-Language Navigation: A Survey](https://arxiv.org/abs/2203.12667)** ⭐⭐⭐⭐
- **类型**：综述论文（IJCV 2023）
- **重点**：系统梳理VLN发展脉络，截至2022年的方法总结
- **适合**：需要全面了解VLN历史演进的研究者

---


### 🎓 重要会议与研讨会

**具身智能专项会议：**
- **[Embodied AI Workshop](https://embodied-ai.org/)** - CVPR官方Workshop，最新趋势和挑战赛发布地
- **[CoRL](https://www.corl.org/)** (Conference on Robot Learning) - VLN向真实机器人迁移的主要阵地
- **[RSS](https://roboticsconference.org/)** (Robotics: Science and Systems) - 顶级机器人会议，强调Sim-to-Real

**主流顶会VLN论文分布：**

| 会议 | 侧重点 | 近期代表作 |
|:----:|:-------|:-----------|
| **CVPR** / **ICCV** / **ECCV** | 视觉-语言方法创新 | NaVILA (CVPR'25), DualVLN (投稿中) |
| **NeurIPS** | 基础模型 + 强化学习 | StreamVLN (NeurIPS'24) |
| **CoRL** | 真实机器人部署 | GNM (CoRL'23), NoMad (CoRL'24) |
| **ICRA** / **IROS** | 导航算法工程化 | VLN-PE (ICRA'25), ViPlanner (ICRA'24) |

---

# 主流 VLN 研究路线

## 0. 技术演进脉络与历史发展

VLN 研究从 2018 年至今经历了显著的范式演进：

**早期范式 (2018-2020)：判别式跨模态匹配**
- **代表工作**：VLN-BERT, Recurrent VLN-BERT, PREVALENT
- **核心思想**：通过文本编码器与视觉编码器提取特征，利用跨模态注意力实现语言-视觉对齐，预测动作 $$P(a_t \mid o_t, I)$$
- **本质**：reactive policy learning，无显式长期规划
- **关键局限**：缺乏空间记忆、泛化能力弱、难以处理复杂指令

**演进路径**：
- **Matching → Planning (2020-2022)**：引入语义地图与拓扑规划，赋予智能体显式空间建模能力
- **Planning → Reasoning (2023-2024)**：基于大模型的双系统/单系统架构，实现从匹配到推理的升级
- **Reasoning → Imagination (2024-2025)**：生成式世界模型，智能体在"想象空间"中前瞻性规划

---


## 1. 语义地图与拓扑规划框架
——Semantic Map & Graph-based Planning

- **起始时间**：2020–2022  
- **代表工作**：DUET, DualVLN, ETPNav, LagMemo  

### 核心思想  

该类方法引入显式环境建模机制，在导航过程中构建 **拓扑图或语义地图**，将空间连通关系与语义信息存储在外部记忆中。智能体基于该地图进行全局路径规划与局部执行。

以 LagMemo 为代表的方法进一步将语言中涉及的目标（如 sofa, kitchen）投影到所构建的语义地图上，实现 **language-grounded SLAM-style navigation**。

---

<div align="center">
  <img src="/images/lagmemo-language-injection.png" width="100%" />
  <figcaption>Semantic Map + Planning VLN 架构示意（LagMemo范式）</figcaption>
</div>

---
### 典型 Pipeline  

```

RGB / Depth → Mapping Module → Topological / Semantic Map
↑
Instruction → Language Parser → Goal / Constraint Projection
↓
Global Planner → Local Policy → Action

```

---

### 优势  

- 支持长距离规划与路径回溯。  
- 显式空间结构提升可解释性。  
- 有效缓解 partial observability。  

---

### 关键缺陷  

- ❌ 地图构建误差会累积传播。  
- ❌ 对动态环境与遮挡鲁棒性有限。  
- ❌ 语义投影通常依赖检测器或规则。  
- ❌ 缺乏高层语言推理能力。

---


## 2. VLN双系统架构
——Dual-System VLN

- **起始时间**：2023–2024
- **代表工作**：DualVLN, InternVLN

### 核心思想

该类方法受认知科学"双过程理论"（快思考/慢思考）启发，采用 **双系统分层架构**：

- **System-2（慢思考，"大脑"）**：基于 VLM/LLM 的高层推理与规划模块
  - 负责：语言理解、指令分解、全局规划、子目标生成
  - 频率：低频（约2 Hz），深度推理
  - 代表：Qwen-VL, GPT-4V, InternVL

- **System-1（快行动，"小脑"）**：基于 VN System（Vision-Navigation）的快速轨迹执行模块
  - 负责：高频轨迹生成、局部避障、快速导航决策
  - 频率：高频（30 Hz），实时响应
  - 代表：DD-PPO, iPlanner, ViPlanner, GNM, ViNT, NoMad, NavDP, InternVLA-N1（DiT策略）

两个系统异步协作：System-2 提供高层语义引导，System-1 负责流畅的低层执行，实现 **"Ground Slow, Move Fast"**。

---

<div align="center">
  <img src="/images/dualvln-framework-overview.png" width="100%" />
  <figcaption>DualVLN 双系统架构示意：System-2（VLM）负责慢推理规划，System-1（VN）负责快轨迹执行</figcaption>
</div>

---
### 典型 Pipeline

```
Instruction + Observation → System-2 (VLM/LLM) → Pixel Goal / Subgoal
                                ↓
                         Latent Features
                                ↓
Current RGB Observation → System-1 (VN System / DiT) → High-Freq Trajectory
                                ↓
                         Low-Level Controller (MPC)
                                ↓
                            Action Execution

```

---

### 优势

- ✅ 强语言理解与常识推理能力（System-2）
- ✅ 高频流畅导航，低延迟实时响应（System-1）
- ✅ 优秀的 zero-shot 泛化性能
- ✅ 支持复杂指令分解与多步规划
- ✅ 异步推理流水线，充分利用 KV-cache

---

### 关键缺陷

- ❌ 系统架构复杂，需要两个独立模型
- ❌ 系统间信息传递需要精心设计（如像素目标、潜在特征）
- ❌ System-2 易产生 hallucination 行为
- ❌ 需要大规模轨迹数据训练 System-1

---


## 3. 单系统端到端 VLN 架构
——Single-System End-to-End VLN

- **起始时间**：2019–2024
- **代表工作**：Seq2Seq, CMA, RDP, StreamVLN

### 核心思想

该类方法采用 **统一的端到端模型**，直接从多模态输入（视觉观测 + 语言指令 + 历史动作）生成下一步动作，无需显式的系统分层或模块划分。

以 StreamVLN 为代表的最新方法将 VLN 建模为 **多轮对话式的交错生成任务**：$$o_1 \to a_1 \to o_2 \to a_2 \to ... \to o_T \to a_T$$，通过滑动窗口 KV cache 和体素化空间剪枝实现高效的流式导航。

---

<div align="center">
  <img src="/images/StreamVLN-framework-overview.png" width="100%" />
  <figcaption>StreamVLN 单系统架构示意：统一模型直接生成观测-动作交错序列</figcaption>
</div>

---
### 典型 Pipeline

```
[Instruction, o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t] → Unified VLM
↓
Autoregressive Generation
↓
Action Token a_t

```

---

### 优势

- ✅ 端到端优化，无模块间信息损耗
- ✅ 推理延迟低（StreamVLN: 0.27s/step），适合实时部署
- ✅ 训练简洁，无需多阶段训练或手工设计模块
- ✅ 通过 KV cache 复用高效处理长对话历史

---

### 关键缺陷

- ❌ 决策过程黑盒，可解释性较弱
- ❌ 需要大规模多源数据联合训练（VLA + VQA + 通用视觉数据）
- ❌ grounding 稳定性依赖数据质量与模型规模
- ❌ 对复杂多步推理的建模能力弱于双系统架构

---

### 架构对比：双系统 vs 单系统

| 维度 | 双系统（DualVLN） | 单系统（StreamVLN） |
|:---:|:---|:---|
| **架构设计** | System-2（VLM/LLM）+ System-1（VN System）分离 | 统一端到端 VLA |
| **推理方式** | VLM规划（2Hz）→ VN高频执行（30Hz） | 观测-动作交错自回归生成 |
| **训练方式** | 分阶段训练（System-2微调 + System-1监督） | 端到端联合训练 |
| **推理延迟** | System-2: 0.7s, System-1: 0.03s | 单次推理: 0.27s |
| **控制频率** | 高频（30 Hz）流畅轨迹 | 低频（约4 Hz）离散动作 |
| **可解释性** | 强（显式像素目标 + 推理链） | 弱（黑盒） |
| **复杂推理** | 强（VLM/LLM显式推理） | 弱（隐式编码） |
| **部署复杂度** | 高（两个模型 + 异步协调） | 低（单模型 + KV cache） |



--- -->

## 4. 生成式世界模型框架
——Generative World Models for VLN

- **起始时间**：2024–2025
- **代表工作**：
  - [Dynam3D](https://openreview.net/forum?id=s6k9l5yX8e) (NeurIPS'25 Oral)
  - [Navigation World Models (NWM)](https://www.amirbar.net/nwm/) (CVPR'25, Best Paper Honorable Mention, Meta AI)
  - [DreamVLA](https://zhangwenyao1.github.io/DreamVLA/) (NeurIPS'25)
  - [WMNav](https://b0b8k1ng.github.io/WMNav/) (IROS'25 Oral)

### 核心思想

该类方法引入 **Predictive World Modeling**，使智能体能够：
1. **预测未来**：在执行动作前，在"想象空间"中模拟可能的未来观测
2. **前瞻性规划**：通过评估多条候选轨迹的未来结果，选择最优路径
3. **减少试错**：显著降低真实环境中的碰撞和探索成本

VLN 从 **reactive execution** 转向 **deliberative planning with imagination**，更接近人类"先想象后行动"的认知方式。

---

<div align="center">
  <img src="/images/world_model_architecture.png" width="100%" />
  <figcaption>Generative World Model VLN 架构示意</figcaption>
</div>

---

### 典型 Pipeline

```
Current Observation + Action Candidate → World Model (Diffusion/Autoregressive)
                                           ↓
                                    Future Visual Rollouts
                                           ↓
                                    Trajectory Evaluation
                                           ↓
                                    Optimal Action Selection

```

---

### 主流技术路线

#### 1. **3D动态表示 + 世界模型**（Dynam3D）

- **核心创新**：多层级 patch-instance-zone 3D 表示，动态在线更新
- **技术特点**：
  - 将 2D CLIP 特征投影到 3D 空间
  - 在线编码和定位 3D 实例
  - 动态适应环境变化，提供长期记忆
- **性能**：SOTA on R2R-CE, REVERIE-CE, NavRAG-CE
- **优势**：大规模探索 + 长期记忆 + 动态环境适应

#### 2. **可控视频生成模型**（Navigation World Models, Meta AI）

- **核心创新**：1B 参数 Conditional Diffusion Transformer (CDiT)
- **技术特点**：
  - 基于过去观测和导航动作预测未来视觉观测
  - 在熟悉环境中模拟轨迹并评估是否达到目标
  - 从单张图像想象未知环境的轨迹
- **训练数据**：多样化的自我中心视频（人类 + 机器人）
- **优势**：灵活的约束规划 + 单图像泛化能力
- **荣誉**：CVPR 2025 Best Paper Honorable Mention

#### 3. **多模态世界知识预测**（DreamVLA）

- **核心创新**：动态区域引导的世界知识预测机制
- **预测内容**：视觉 + 深度 + 几何 + 语义 + 分割
- **技术特点**：
  - 扩散 Transformer 建模动作条件分布
  - 先形成抽象多模态推理链，再执行动作
- **性能**：76.7% 真实机器人任务成功率
- **应用**：操作任务，但范式可迁移到 VLN

#### 4. **VLM + 世界模型融合**（WMNav）

- **核心创新**：将 Vision-Language Models 集成到世界模型中
- **关键组件**：
  - PredictVLM：预测决策的可能结果
  - Curiosity Value Map：构建记忆并提供反馈
  - 导航策略模块：动态决策
- **性能**：
  - HM3D: +3.2% SR, +3.2% SPL（zero-shot SOTA）
  - MP3D: +13.5% SR（所有方法中最优）
- **优势**：模块化设计 + zero-shot 泛化

---

### 优势

- ✅ 支持前瞻性规划，在行动前"想象"结果
- ✅ 显著降低真实试错成本和碰撞率
- ✅ 更接近人类认知导航方式（先思考再行动）
- ✅ 可以评估多条候选轨迹，选择最优路径
- ✅ 单图像即可想象未知环境的导航轨迹（NWM）

---

### 关键缺陷

- ❌ 想象误差会累积，长期预测不稳定
- ❌ 训练成本极高（需要大规模视频数据 + 大模型）
- ❌ 推理延迟较高（视频生成模型较慢）
- ❌ 与语言约束的融合仍需改进（语义漂移问题）
- ❌ sim2real gap：模拟的未来可能与真实不符

---



### 总结

VLN 的研究已从早期的判别式匹配演进为当前的四大主流路线：

1. **语义地图与拓扑规划**：显式空间建模，支持长距离规划与回溯
2. **VLN双系统架构**：认知分层，System-1快速感知 + System-2推理规划
3. **单系统端到端架构**：统一优化，低延迟流式导航
4. **生成式世界模型**：前瞻性规划，在"想象空间"中搜索最优路径

未来趋势是融合 **语言理解、空间建模、双系统推理、端到端优化与世界模型预测**，形成"语言引导 + 想象规划 + 连续执行"的统一具身智能体架构。

---

### 2024-2026 技术趋势 (Key Trends)

| 技术趋势 | 描述 | 关键论文/项目 |
| :--- | :--- | :--- |
| **Long-Horizon** | 处理超长距离路径（>150步）及涉及多房间、跨楼层的复杂多阶段导航任务。 | LHPR-VLN (CVPR '25), L-VLN |
| **Dynamic Environment** | 针对真实世界中移动行人、开关门、光照突变等非稳态场景的适应性导航。 | DynamicVLN (2025), DynaNav |
| **World Models** | 引入预测学习，智能体通过生成未来视觉帧预测动作结果，实现"脑内模拟规划"。2025年重大突破：Dynam3D (NeurIPS'25 Oral)、NWM (CVPR'25 Best Paper)、DreamVLA (NeurIPS'25)、WMNav (IROS'25 Oral)。 | Dynam3D, NWM (Meta AI), DreamVLA, WMNav |
| **VLA Models** | **视觉-语言-动作 (VLA) 一体化**。直接从多模态输入输出底层控制指令，消除感知与动作的鸿沟。 | OpenVLA (2024), Helix, RT-2 |
| **Self-Evolving VLN** | **2025-2026 新趋势**。智能体在测试阶段通过自我反思和经验检索，在无需重新训练的情况下实现性能进化。 | **SE-VLN** (ICLR '26 / 2025), Reflection-Nav |
| **3DGS-based Map** | 利用 **3D高斯泼溅 (3D Gaussian Splatting)** 进行环境重建，提供比点云更精细、渲染更快的神经导航地图。 | GS-Nav (2025), Splat-Nav |

---

### 趋势深度解析

#### 1. 从“固定模型”到“自我演进 (Self-Evolving)”
这是目前最前沿的方向（如 **SE-VLN**）。传统的 VLN 模型在部署后能力是固定的，而最新的研究让智能体拥有一个“经验库”。当它导航失败时，它会记录失败原因并在下次遇到类似场景时进行**自我修正**。这种“闭环进化”让模型更像真实的生物。

#### 2. 生成式 AI 赋予智能体“想象力”
**World Models** 的引入改变了 VLN 的本质。以前智能体是“应激式”移动，现在它能利用 **Diffusion** 或 **Video Generation** 技术预判：“如果我右转，我会看到什么？”这种前瞻性规划显著降低了碰撞率。

#### 3. 具身大模型 (VLA) 的统领地位
随着 **OpenVLA** 等模型的成熟，VLN 不再是一个独立的视觉匹配任务，而是被纳入了通用的具身机器人大脑中。现在的趋势是：一个模型既能做 R2R 导航，也能在到达终点后完成“把杯子放进微波炉”的操作任务。

---

# VLN经典论文

## 1. DualVLN/InternVLN (2025)
——Ground Slow, Move Fast

<div align="center">
  <img src="/images/dualvln-framework.png" width="100%" />
<figcaption>
DualVLN双系统框架总览
</figcaption>
</div>

**研究背景/问题**

VLN领域存在基本矛盾：强大的推理能力需要"慢思考"，而流畅的导航行动需要"快反应"。传统端到端模型存在三大瓶颈：
- **动作碎片化**：每一步都需调用大模型，产生离散的短视野动作（如"前进0.25m"）
- **响应延迟高**：无法实现高频控制（30Hz+），导致运动不自然
- **缺乏层次协调**：语义理解、全局规划和局部避障耦合在一起，难以应对动态障碍物

**主要方法/创新点**



DualVLN提出**首个双系统VLN基础模型**，将高级语义理解与低级轨迹执行解耦，形成互补的快慢系统：

<div align="center">
  <img src="/images/dualvln-framework-overview.png" width="100%" />
<figcaption>
DualVLN双系统框架架构
</figcaption>
</div>

### 系统2（慢思考的"大脑"）

**核心功能**：
- **全局规划器**：基于Qwen-VL-2.5（7B参数），以约2 Hz频率运行
- **像素级目标预测**：将3D导航任务转化为2D像素级目标定位问题
- **自动生成训练数据**：
  - 通过3D→2D投影，将未来轨迹点投影到当前视角
  - 利用深度图过滤遮挡点（距离>深度值的点被视为不可见）
  - 选择最远可见点作为"像素目标"（farthest pixel goal）
- **智能视角调整**：
  - 当未来轨迹无法投影到当前视角时（如连续转弯动作的起点）
  - 自主预测视角调整动作（Turn Left/Right 15°, Look Up/Down 15°）
  - 最多支持4次连续视角调整，模仿人类"环顾四周、低头看路"的行为

**训练策略**：
- 完全解冻视觉编码器和LLM骨干网络，进行一个epoch的全量微调
- 使用StreamVLN的数据配方（多源VLN数据集）
- 优化器：AdamW，学习率2e-5，批量大小128，训练14,000步

### 系统1（快行动的"小脑"）


**核心功能**：
- **高频轨迹生成**：轻量级扩散Transformer策略，以30 Hz频率运行
- **多模态条件扩散**：
  - **显式像素目标**：提供可解释的空间引导
  - **隐式潜在目标**：从系统2的隐藏状态中提取丰富的任务相关语义
- **语义特征提取**：
  - 在像素目标文本后追加4个可学习的特殊token `<TRAJ>`
  - 通过prompt tuning优化这些潜在查询向量
  - 从冻结的VLM最后一层隐藏状态中提取紧凑特征
- **双时间步RGB融合**：
  - 编码系统2最后一帧（时间t）和当前帧（时间t+k）
  - 通过自注意力融合两个时间步的ViT特征
  - 用Q-Former压缩为32个token，作为高频视觉条件
- **Flow Matching训练**：
  - 噪声轨迹：$X_u = \alpha_u X_0 + \sigma_u \epsilon$
  - 速度预测：$\hat{\dot{X}}_u = f_\theta(X_u, u, Z' \oplus F)$
  - 损失函数：$\mathcal{L}_{flow} = \mathbb{E}_{u,X_0,\epsilon}[\|\hat{\dot{X}}_u - \dot{X}_u\|_2^2]$

**架构细节**：
- RGB编码器：DepthAnythingV2-Small的ViT骨干
- DiT设计：隐藏维度384，12层Transformer，6个注意力头
- 潜在嵌入：从3584维线性投影到768维后进行交叉注意力
- 输出：32个密集路径点的平滑轨迹

**训练策略**：
- 冻结系统2（Qwen-VL），仅训练潜在查询和DiT策略
- 优化器：AdamW，学习率1e-4，批量大小128，训练15,000步
- **关键设计**：仅使用像素目标grounding样本进行监督（不包含视角调整样本）

<div align="center">
  <img src="/images/dualvln-system1-trajectory.png" width="100%" />
<figcaption>
系统1的高频轨迹生成
</figcaption>
</div>

### 协同机制

**异步推理流水线**：
- 系统2（2Hz）：每0.5秒生成新的像素目标和潜在特征
- 系统1（30Hz）：每0.03秒基于最新RGB和缓存的潜在特征更新轨迹
- 低级控制器（200Hz）：MPC控制器跟踪系统1生成的轨迹
- **关键优势**：利用KV-cache复用，将系统2推理时间从1.1s降至0.7s；系统1用TensorRT并行生成32个轨迹点仅需0.03s

**核心结果/发现**

### 1. 仿真基准测试（VLN-CE）

#### R2R Val-Unseen（仅单视角RGB输入）

| 方法 | SR ↑ | SPL ↑ | NE ↓ | OS ↑ |
|:------:|:------:|:-------:|:------:|:------:|
| StreamVLN（前SOTA） | 56.9% | 51.9% | 4.98m | 64.2% |
| **DualVLN** | **64.3%** | **58.5%** | **4.05m** | **70.7%** |
| **提升幅度** | **+7.4%** | **+6.6%** | **-0.93m** | **+6.5%** |

#### RxR Val-Unseen（多语言指令）

| 方法 | SR ↑ | SPL ↑ | NE ↓ | nDTW ↑ |
|:------:|:------:|:-------:|:------:|:--------:|
| NaVILA（前SOTA） | 49.3% | 44.0% | 6.77m | 58.8% |
| **DualVLN** | **61.4%** | **51.8%** | **4.58m** | **70.0%** |
| **提升幅度** | **+12.1%** | **+7.8%** | **-2.19m** | **+11.2%** |

**关键观察**：
- 在RxR基准上，DualVLN的优势更明显（+12.1% SR），说明双系统设计对复杂多语言指令有更好的泛化能力
- 与使用全景RGB+深度+里程计的多传感器方法（如ETPNav）相比，DualVLN仅用单视角RGB即达到更高的64.3% SR（ETPNav仅57.0%）

### 2. 物理仿真基准（VLN-PE）

#### R2R Val-Unseen（Humanoid H1机器人）

| 方法 | SR ↑ | SPL ↑ | NE ↓ | FR ↓ | StR ↓ |
|:------:|:------:|:-------:|:------:|:------:|:-------:|
| NaVid（零样本迁移） | 22.42% | 18.58% | 5.94m | 8.61% | 0.45% |
| RDP（在VLN-PE上训练） | 25.24% | 17.73% | 6.72m | 24.57% | 3.11% |
| **DualVLN（零样本迁移）** | **51.60%** | **42.49%** | **4.66m** | **12.32%** | **2.23%** |

**关键发现**：
- 尽管DualVLN未在VLN-PE上微调，但仍大幅超越所有基线（包括在VLN-PE上训练的方法）
- 成功率提升超过2倍（51.60% vs 22.42%），证明双系统设计对物理真实环境有更强的泛化能力
- 跌倒率（FR）虽高于NaVid，但成功率显著提升，表明DualVLN在探索效率和安全性之间取得了更好的平衡

### 3. Social-VLN基准（动态障碍物）

**新基准设计**：
- 在R2R-CE基础上，沿ground-truth轨迹放置动态人形机器人（Habitat 3.0）
- 引入Human Collision Rate（HCR）指标，量化与行人的不安全交互
- 收集763K社交导航样本（60个MP3D场景），用改进的A*算法生成避障轨迹

**性能对比（R2R Val-Unseen）**：

| 方法 | 静态VLN SR | Social-VLN SR | SR下降 | HCR |
|:------:|:------------:|:---------------:|:--------:|:-----:|
| StreamVLN | 56.9% | 31.4% | -25.5% | 36.4% |
| DualVLN | 64.3% | 37.2% | -27.1% | 35.4% |

**关键观察**：
- 两种方法的成功率都大幅下降（~26%），说明Social-VLN极具挑战性
- DualVLN在动态场景中仍保持6%的绝对优势，HCR略低于StreamVLN
- **改进空间**：尽管DualVLN表现最佳，但仍有很大提升空间（37.2% SR远低于静态场景的64.3%）

### 4. 真实世界跨具身实验

**实验设置**：
- **机器人平台**：轮式（Turtlebot4）、四足（Unitree Go2）、人形（Unitree G1）
- **传感器配置**：Intel RealSense D455（不同安装高度，下倾15°）
- **推理硬件**：远程服务器（RTX 4090 GPU，占用20GB显存）
- **控制流程**：机器人流式传输RGB-D图像 → 服务器异步推理 → MPC控制器跟踪轨迹

**定量评估（3种场景难度）**：
- **走廊（简单）**：DualVLN SR 100% vs 基线25%-80%
- **单卧室（中等）**：DualVLN SR 100% vs 基线0%-70%
- **R2R办公室（困难，跨房间）**：DualVLN SR 85% vs 基线0%-60%

**导航误差对比**：

| 场景 | CMA | NaVid | NaVILA | StreamVLN | DualVLN |
|:------:|:-----:|:-------:|:--------:|:-----------:|---------:|
| 走廊 | 3.2m | 0.9m | 0.3m | 0.2m | **0.2m** |
| 卧室 | 5.3m | 2.5m | 0.6m | 0.3m | **0.3m** |
| 办公室 | 15.4m | 10.1m | 2.2m | 0.5m | **0.4m** |

**定性分析（见补充视频）**：
- **场景多样性**：办公室、食堂、街道、便利店，零样本设置（无场景特定微调）
- **像素目标精准**：准确选择安全、可达的像素目标
- **轨迹流畅性**：生成平滑、避障的连续轨迹，避免频繁停止或转向
- **地形适应性**：成功处理楼梯、斜坡、门槛等复杂地形
- **动态避障**：实时躲避行走的行人，保持任务轨迹
- **跨平台鲁棒性**：在不同相机高度、振动、跟踪精度下表现稳定

### 5. 消融实验

#### 5.1 目标表征的作用

**实验设计**（R2R Val-Unseen）：
- **w/o Sys.2 Train**：系统1和系统2联合端到端训练，不使用显式像素目标
- **w/o Pixel Goal**：训练系统1时，移除像素目标文本（潜在查询无法关注显式目标）
- **w/o Latent Goal**：仅使用冻结VLM的像素目标文本的最后一层隐藏状态

**结果对比**：

| 配置 | SR | SPL | OS | NE |
|:------:|:-----:|:-----:|:-----:|:-----:|
| DualVLN（完整） | 64.3% | 58.5% | 70.7% | 4.05m |
| w/o Sys.2 Train | 55.2% | 51.5% | 60.9% | 4.98m |
| w/o Pixel Goal | 62.2% | 55.8% | 68.0% | 4.22m |
| w/o Latent Goal | 60.9% | 55.1% | 67.7% | 4.26m |

**关键发现**：
1. **解耦训练至关重要**（w/o Sys.2 Train -9.1% SR）：
   - 联合训练导致扩散策略收敛缓慢
   - 系统2的泛化能力严重退化
   - 证明了显式像素目标作为中间监督的必要性

2. **显式像素目标增强可解释性**（w/o Pixel Goal -2.1% SR）：
   - 为扩散策略提供明确的空间引导
   - 提升系统2的可解释性和泛化能力
   - 仅依赖隐式特征会损失部分性能

3. **隐式潜在目标提供丰富语义**（w/o Latent Goal -3.4% SR）：
   - 被动使用固定VLM特征限制了信息流
   - 可学习的潜在查询能主动提取任务相关表征
   - 两种目标表征互补，缺一不可

#### 5.2 与SOTA点目标导航策略对比

**实验设计**（VLN-PE R2R Val-Unseen）：
- 移除隐式潜在目标，将显式像素目标转换为点目标（使用额外深度信息）
- 用SOTA点目标导航策略替换系统1：
  - **iPlanner**：命令式路径规划（Yang et al., 2023）
  - **NavDP**：导航扩散策略（Cai et al., 2025）

**结果对比**：

| 局部规划器 | SR | SPL | NE | OS |
|:------------:|:-----:|:-----:|:-----:|:-----:|
| iPlanner | 47.07% | 41.09% | 4.91m | 55.53% |
| NavDP | 58.72% | 50.98% | 4.22m | 67.33% |
| System 1（完整） | **63.62%** | **56.49%** | **3.90m** | **69.93%** |

**性能差距分析**：
1. **轨迹分布不匹配**：
   - 点目标规划器生成的轨迹与系统2训练数据分布不同
   - 导致系统2的像素目标预测质量下降

2. **像素目标误差的鲁棒性差异**：
   - **系统1**：对方向正确但位置偏差的像素目标具有鲁棒性，能通过实时RGB修正轨迹
   - **点目标方法**：直接将像素投影到世界坐标，对微小像素误差高度敏感
   - **特例**：当像素目标语义错误（如目标在障碍物上）或机器人靠近障碍物时，系统1的鲁棒性也会失效

3. **避障能力**：
   - 系统1展现出强大的视觉避障行为（基于高频RGB输入）
   - 点目标方法更依赖精确的几何信息，对传感器噪声更敏感

#### 5.3 系统1的数据缩放规律

**实验设计**：
- 使用系统2轨迹数据的不同比例训练系统1：1%, 5%, 10%, 30%, 50%
- 评估SR和SPL在R2R Val-Unseen上的变化

**结果曲线**：
- **1%**：SR ~54%, SPL ~54%（已具备竞争力）
- **10%**：SR ~62%, SPL ~58%（接近饱和）
- **50%**：SR ~64%, SPL ~58.5%（边际收益递减）

**关键洞察**：
1. **系统1是轻量级的**：
   - 设计为快速、简单的轨迹生成器
   - 目标跟踪任务本质上比语义理解简单

2. **与系统2的数据缩放对比**：
   - 系统2遵循VLM的数据饥饿特性（更多多样化数据→更好泛化）
   - 系统1快速饱和，表明其性能上限取决于系统2的质量

3. **训练效率**：
   - 仅需系统2数据的10%即可训练出高性能系统1
   - 进一步证明了解耦训练的优势

#### 5.4 像素目标与轨迹的一致性分析

**实验设计**：
- 随机采样1000个样本，来自不同成功率的DualVLN模型（64.3%, 60.9%, 58.2%, 56.8%）
- 将预测轨迹投影到图像平面，计算与像素目标的：
  - **像素距离**：投影轨迹点到像素目标的欧氏距离
  - **角度偏差**：轨迹方向与像素目标方向的夹角

**可视化结果**（见Figure 10）：
- **密度集中在左下角**：大多数点的像素距离和角度偏差都很小
- **趋势一致**：所有成功率模型都显示轨迹朝向像素目标并到达其附近区域
- **性能相关性**：成功率越高的模型，密度集中度越高（轨迹-目标一致性越强）

**结论**：
- 系统1的轨迹预测强烈受像素目标引导
- 验证了双系统设计的有效性：系统2提供明确目标，系统1忠实执行

### 6. 注意力机制分析

**可视化方法**（见Supplementary Material Figure 11）：
- 提取Qwen-VL不同层（第6、15、24层）的注意力图
- 关注两个模态：语言指令token 和 视觉token（历史帧+当前观察）

**层级分析**：
1. **浅层（Layer 6）**：
   - 注意力分散在整个场景和指令的多个词汇
   - 关注通用的上下文和空间线索（如物体、场景布局、方向词）

2. **中层（Layer 15）**：
   - 注意力开始聚焦到目标相关区域
   - 指令中的关键词（如"table", "bridge"）获得更高权重

3. **深层（Layer 24）**：
   - 注意力高度集中在精确的像素目标区域
   - 同时对STOP token分配显著权重（用于任务完成判断）
   - 证明模型在最后阶段整合视觉和语言线索进行最终决策

**关键发现**：
- **逐层精化**：从广泛的语义理解 → 逐步精确的空间定位
- **多模态融合**：深层同时关注视觉目标和语言指令（特别是STOP信号）
- **任务完成感知**：模型能自主判断何时到达目标（通过STOP token的注意力）

### 7. 推理效率分析

**系统2优化**：
- **KV-cache复用**：将轨迹token推理时间从1.1s降至0.7s（提速36%）
- **视角调整缓存**：连续视角调整时重用已编码的历史图像特征

**系统1优化**：
- **TensorRT加速**：并行生成32个轨迹点仅需0.03s
- **异步推理**：系统1持续运行在30Hz，不等待系统2更新

**端到端延迟**：
- 系统2更新周期：0.5s（2Hz）
- 系统1更新周期：0.033s（30Hz）
- 控制器频率：200Hz（MPC跟踪）
- **实际效果**：机器人始终有最新轨迹可用，实现近实时、流畅的导航

**局限性与未来方向**

1. **极端扰动鲁棒性**：
   - 在强烈相机抖动、光照剧变、遮挡等极端情况下性能下降
   - 未来可引入更鲁棒的视觉编码器（如事件相机、多模态传感器融合）

2. **Sim-to-Real迁移效率**：
   - 虽然DualVLN展现出强大的零样本泛化能力，但仿真到真实的域差距仍存在
   - 可探索域随机化、域自适应等技术进一步缩小差距

3. **跨层表征对齐**：
   - 当前的潜在查询机制是单向的（系统2 → 系统1）
   - 未来可探索双向反馈机制，让系统1的执行结果反馈到系统2的规划

4. **Social-VLN性能**：
   - 在动态场景中成功率仍有很大提升空间（37.2% vs 静态64.3%）
   - 需要更多社交导航数据和显式的人-机器人交互建模

5. **长视程泛化**：
   - 论文未详细评估超长指令（如跨楼层导航）的性能
   - 未来可扩展到更大规模环境（如整栋建筑、园区级导航）

6. **计算资源需求**：
   - 系统2（7B VLM）需要20GB显存，限制了边缘设备部署
   - 可探索模型压缩、量化、知识蒸馏等技术实现轻量化

**方法论贡献总结**

1. **首个双系统VLN基础模型**：将认知科学的"双过程理论"引入具身导航
2. **解耦训练范式**：保留VLM泛化能力的同时，高效训练低级策略
3. **显隐式目标协同**：兼顾可解释性（像素目标）和表征丰富性（潜在特征）
4. **异步推理架构**：实现高频控制（30Hz）的同时保持低级别的感知-行动延迟
5. **Social-VLN基准**：填补了VLN领域在动态场景评估上的空白

---

## 2. NavDP (2025)
——基于扩散模型的零样本导航规划

<div align="center">
  <img src="/images/navdp.png" width="100%" />
<figcaption>
NavDP双阶段推理框架
</figcaption>
</div>

**研究背景/问题**

机器人导航面临的核心挑战是如何在保证安全的前提下实现跨场景、跨平台的泛化能力。传统的导航方法往往依赖于显式地图构建或需要在目标环境中大量采集真实数据，这限制了其在实际场景中的部署效率。NavDP（Navigation with Diffusion Policy，上海AI Lab）提出通过大规模模拟数据训练，结合扩散模型生成候选轨迹和Critic网络评估安全性，实现零样本sim-to-real迁移。

**主要方法/创新点**

<div align="center">
  <img src="/images/navdp-framework.png" width="100%" />
<figcaption>
NavDP双阶段推理框架
</figcaption>
</div>

### 核心思路

扩散模型负责生成候选轨迹，Critic负责挑选安全路线

### 两阶段推理框架

- **第一阶段（策略生成）**：用RGB-D图像+导航目标，经策略Transformer编码后，通过扩散生成候选轨迹
- **第二阶段（安全评估）**：将生成轨迹与RGB-D token融合，再经共享Transformer与critic head，选择与目标无关的安全轨迹

### 模拟特权信息利用

- **生成器训练**：利用模拟环境中的全局最优规划器指导轨迹生成
- **Critic训练**：利用模拟环境的全局ESDF，从负样本轨迹中学习精细空间理解
- **数据增强**：对原始轨迹进行随机旋转和插值，生成混合轨迹增加多样性

### 多模态输入编码

- **输入**：单帧RGB-D图像+导航目标（四种类型：点目标、图像目标、轨迹目标、无目标）
- **深度处理**：裁剪至0.1-3.0 m，RGB经预训练DepthAnything ViT编码，深度由自训练ViT编码
- **Transformer解码器**：将512个RGB-D token压缩为16个融合token

### Real-to-Sim增强

- 采用Gaussian Splatting重建真实环境，提供高真实感的训练与评测平台
- 在训练集中加入27%的real-to-sim样本，可使目标场景成功率提升30%，且不损害泛化能力

<div align="center">
  <img src="/images/navdp-trajectory-visualization.png" width="100%" />
<figcaption>
NavDP模拟数据生成流程
</figcaption>
</div>

**核心结果/发现**

- **跨机器人平台泛化**：在不同机器人（Dingo、Go2、Galaxea R1）上稳定高于基线（GNM, ViNT, NoMad, iPlanner, ViPlanner, EgoPlanner）
- **零样本Sim-to-Real**：成功在Unitree Go2、Galaxea R1、Unitree G1上部署，室内外场景均表现良好，含动态行人干扰
- **数据规模与效率**：模拟数据生成速度约2,500条轨迹/GPU/天，比真实采集快20倍；数据集覆盖1244个场景、总长度363.2 km
- **模型组件贡献**：Critic模块是性能提升的关键，移除后性能显著下降；No-goal训练目标对整体避障行为影响最大
- **Real-to-Sim效果**：真实场景成功率提高30%，证明real-to-sim数据能显著提升sim-to-real成功率
- **高速避障**：>10Hz推理，支持2.0 m/s高速避障，动态场景下优于传统地图规划方法

**局限性**

NavDP的性能高度依赖于高质量的模拟数据训练，Real-to-Sim数据比例需要仔细平衡以避免过拟合特定场景。虽然在多种机器人平台上展现了良好的泛化能力，但在极端复杂环境（如密集人群、高度动态场景）下的鲁棒性仍有提升空间。此外，扩散模型的多步推理虽然提供了多样化的轨迹候选，但相比直接回归方法计算开销更大，对实时性要求极高的应用场景可能存在挑战。

---

## 3. NoMaD (2023)
——目标掩码扩散策略实现统一导航

**研究背景/问题**

传统机器人导航系统通常为探索（exploration）和目标导航（goal-conditioned navigation）分别训练独立的策略模型，这不仅增加了系统复杂度，也限制了跨任务的知识共享和泛化能力。NoMaD（Nomadic Multi-task Agent with Diffusion，伯克利，ICRA2024 Best Paper）提出通过统一的扩散策略框架，使用目标掩码机制同时建模任务特定行为（目标导向）和任务无关行为（探索），实现单一策略胜任多种导航任务。

**主要方法/创新点**

<div align="center">
  <img src="/images/nomad-framework.png" width="100%" />
<figcaption>
NoMaD目标掩码扩散策略框架
</figcaption>
</div>

### 核心思路

通过统一的扩散策略，同时建模任务特定和任务无关行为

### 两个关键组件

**目标掩码（Goal Masking）**
- 通过二值掩码控制策略是否关注目标图像，实现任务条件的灵活切换
- **训练时**：目标掩码以50%概率随机设置，使模型同时学习目标导向行为和探索行为
- **推理时**：根据任务需要设置掩码（探索时掩盖目标，导航时提供目标）

**扩散策略（Diffusion Policy）**
- 利用扩散模型生成多模态、无碰撞的动作序列
- 从随机噪声逐步迭代生成预测动作序列
- 动作分布既可在无目标条件下表达探索行为，也可在提供目标条件下收敛到目标导向行为

### 统一框架设计

- 通过Transformer编码视觉观测并结合扩散模型生成未来动作序列
- 同时支持任务特定行为（目标导向）和任务无关行为（探索）
- 使用大规模多样化数据集（GNM和SACSoN）进行端到端监督训练

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/nomad-goal-masking.png" width="100%" />
<figcaption>
NoMaD目标掩码机制示意图
</figcaption>
</div> -->

**核心结果/发现**

- **探索未知环境**：成功率达到98%，平均碰撞数仅0.2，超过最优基线Subgoal Diffusion约25%，且参数量仅为其1/15
- **目标导航**：在已知环境的目标导航任务中，成功率与最优基线相当，但计算资源需求更少
- **计算效率**：比现有方法计算效率提升约15倍，是首个成功在物理机器人上部署的目标条件动作扩散模型
- **统一策略优势**：联合训练能够学习共享表示和环境可操作性，单一策略即可胜任多种行为
- **编码器选择**：ViNT编码器配合注意力目标掩码效果最佳，成功率98%，碰撞数最少
- **多场景验证**：在6个复杂的室内外环境中表现优异

**局限性**

NoMaD的视觉编码器选择对性能影响较大，需要仔细调优以达到最佳效果。虽然ViT编码器具有更大的容量和表达能力，但其训练优化难度较高，收敛速度相对较慢。此外，目标掩码机制的随机采样比例（训练时50%）是一个关键超参数，在不同场景下可能需要针对性调整。尽管在多个室内外环境中表现优异，但在极端复杂、高度动态的场景（如密集人流、快速变化的障碍物）下的鲁棒性仍有进一步提升空间。

---
## 4. ODYSSEY (2025)
——Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks

**研究背景/问题**

在动态、非结构化环境中，机器人需要将移动性、操作和实时感知紧密结合才能执行复杂任务。现有研究大多局限于桌面场景，未能解决移动平台特有的感知受限和执行器范围有限的问题，且在开放世界环境中的泛化能力不足。

**主要方法/创新点**

ODYSSEY提出了一个统一的移动操作框架，包含分层规划和全身控制两大核心模块：

<div align="center">
  <img src="/images/odyssey-framework-overview.png" width="100%" />
<figcaption>
ODYSSEY框架整体架构
</figcaption>
</div>

**长期任务规划器：**
- **全局任务级规划**：融合RGB和LiDAR流构建场景的空-语义表示，利用预训练基础模型将实例图映射到场景中
- 使用GPT-4.1将自然语言指令分解为原子动作序列（导航、抓取、放置等），并输出粗略目标航路点
- 航路点投影到2D占用图，通过局部搜索确定无碰撞目标姿态

**局部操作：**
- 使用腕部安装的深度观测数据指导视觉-语言模型生成精确末端执行器姿态
- Qwen2.5-VL-72B-Instruct模型根据RGB观测和文本描述推断任务相关接触点
- 根据目标物体主轴和表面法线施加几何约束，确定末端执行器朝向

<div align="center">
  <img src="/images/odyssey-whole-body-control.png" width="100%" />
<figcaption>
两阶段全身控制策略训练流程
</figcaption>
</div>

**全身控制策略：**
- 单一网络将观测向量（运动指令、末端执行器目标、地面高度图、重力向量、本体感知状态等）映射到目标动作
- **两阶段训练**：第一阶段固定机械臂关节训练运动；第二阶段控制全部18个关节，采用地形不变末端执行器采样策略
- 引入步态奖励、频率奖励和末端执行器跟踪项，运用领域随机化增强适应性

**模拟基准测试：**
- 构建包含50个刚体物体、15个容器、30个关节结构、10个可拖动物体的多样化资产库
- 基准测试包括10个真实场景（室内家居、超市、餐厅、室外庭院等）
- 长期任务包含246个室内和58个室外变化，涉及抓取、重新定向、容器放置、关节操作等多种技能

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/odyssey-results-comparison.png" width="100%" />
<figcaption>
与基线方法的性能对比
</figcaption>
</div> -->

**核心结果/发现**

- **短期任务**：在ARNOLD基准测试上优于PerAct基线，仅依赖单个自我中心摄像头实现更强的泛化能力，在未见数据集上性能保持稳定
- **长期任务**：在8个长期移动操作任务上实现40%以上整体成功率，每个原子技能类别保持60%以上成功率，展现出可靠的协调能力
- **低层策略**：在基座速度跟踪方面优于RoboDuet基线，末端执行器姿态跟踪性能相当，且在不同地形上具有更强的适应性
- **Sim-to-Real迁移**：成功在Unitree Go2+Arx5平台上实现现实世界部署，在"导航到抓取"和"抓取和放置"任务中验证了框架的实用性

**局限性**

模型在物体几何形状的空间推理方面存在局限，导致夹爪对齐不佳和细长手柄或部分遮挡物品的定位不准确。此外，抓取小物体时偶尔失败，主要由于末端执行器跟踪和视觉感知精度不足。

---

## 5. PanoNav (2025)
——Mapless Zero-Shot Object Navigation

**研究背景/问题**

现有目标导航方法大多依赖深度传感器或预建地图来构建2.5D场景表示，限制了在真实环境中的适用性和泛化能力。零样本目标导航要求机器人识别和导航到超出预定义类别范围的对象，现有方法在开放词汇场景中表现有限。无地图方法通常只基于当前观测进行决策，忽略历史轨迹信息，容易陷入局部死锁。

**主要方法/创新点**

PanoNav是一个无地图、仅使用RGB图像的零样本目标导航框架，包含两个核心模块：

<div align="center">
  <img src="/images/panonav-framework-overview.png" width="100%" />
<figcaption>
PanoNav框架整体架构
</figcaption>
</div>

**全景场景解析（Panoramic Scene Parsing）：**

*局部方向解析：*
- **点阵图像增强**：将每个RGB图像转换为点阵图像，通过Scaffold方法增强平面位置理解，与RGB图像共同作为MLLM输入
- **空间关系图构建**：MLLM利用几何距离关系和平面位置关系，构建空间关系图，生成每个方向的详细描述（物体存在、空间关系、房间类型等）

<div align="center">
  <img src="/images/panonav-panoramic-parsing.png" width="100%" />
<figcaption>
全景场景解析模块：从RGB输入到局部方向描述
</figcaption>
</div>

*全局全景总结：*
- **环境整体感知**：对机器人周围环境进行整体分析，识别环境中存在的物体类型和当前房间类型（如厨房、走廊）
- **隐式自我定位**：通过全局总结提供隐式自我定位信息，帮助机器人理解其在更大环境中的位置

**动态记忆引导决策（Dynamic Memory-guided Decision-Making）：**

- **动态有界记忆队列**：存储最近的全局场景总结，队列长度固定，当队列满时新元素加入会移除最旧元素
- **决策过程**：
  - 记忆队列未满时：决策仅基于当前的局部描述和全局总结
  - 记忆队列满时：决策结合当前信息和历史记忆信息，避免重复探索已访问区域
- **动作选择**：决策结果包括导航方向和是否找到目标的标志，由运动控制器执行相应动作

<div align="center">
  <img src="/images/panonav-dynamic-memory.png" width="100%" />
<figcaption>
动态记忆引导决策机制
</figcaption>
</div>

**任务设置：**
- **观测数据**：每个时间步获取六个方向的RGB图像（间隔60度），形成全景视图，不依赖深度传感器或GPS
- **动作空间**：停止、前进（0.25米）、左转/右转（30度）、抬头/低头
- **任务目标**：在未见过的环境中根据语言指令找到目标对象，并导航至目标位置

**核心结果/发现**

- **性能优势**：在HM3D数据集上，PanoNav的成功率（SR）达到43.5%，SPL达到23.7%，显著优于PixNav（SR=37.9%，SPL=20.5%）和ZSON（SR=25.5%，SPL=12.6%），甚至超过部分依赖地图和闭词汇表的方法
- **死锁避免**：在高度欺骗性环境中，通过动态记忆机制实现48.0%成功率和19.2% SPL，逃离局部区域的逃逸率达82.0%
- **消融实验验证**：
  - 全景视图的重要性：仅使用三视图时性能显著下降（SR=19.5%，SPL=9.97%）
  - 解耦解析与决策的优势：解耦方法（SR=43.5%，SPL=23.7%）优于直接从MLLM输出决策（SR=38.5%，SPL=22.57%）
  - 动态记忆的关键作用：移除动态记忆后性能大幅下降（SR=38.5%，SPL=22.57%）

**局限性**

虽然PanoNav显著提升了无地图零样本导航性能，但未来仍需探索利用多模态信息（如语音、手势等）构建更强大的记忆队列，以进一步提高无地图目标导航的鲁棒性和泛化能力。

---

## 6. VLN-R1 (2025)
——基于GRPO与Time-Decayed Reward的端到端导航

**研究背景/问题**

VLN是具身人工智能领域的一项核心挑战，要求智能体根据自然语言指令在真实世界环境中进行导航。传统的导航方法通常依赖离散的拓扑图和预定义的节点连接，限制了智能体在连续环境中的泛化能力。

**主要方法/创新点**

VLN-R1提出了一种创新的端到端框架，利用大型视觉-语言模型（LVLM）直接处理自我中心视频流，生成连续的导航动作。

<div align="center">
  <img src="/images/vln-r1-framework-overview.png" width="100%" />
<figcaption>
VLN-R1端到端框架整体架构
</figcaption>
</div>

**核心设计理念：**
- 构建能够实时处理自我中心视频流并生成连续导航动作的端到端框架
- 与传统方法依赖导航图或额外传感器不同，VLN-R1直接将视觉输入和自然语言指令转化为动作输出
- 提高系统通用性，增强在未见过环境中的适应能力

**主要组件：**

*VLN-Ego数据集：*
- **数据生成**：通过Habitat模拟器生成，包含自我中心视频流与未来动作预测的配对数据
- **三部分文本注释**：
  - 指令部分：自然语言导航指令（如"走到客厅的沙发旁"）
  - 视觉部分：包括历史帧和当前观察，提供自我中心的视觉信息
  - 动作部分：未来动作选择（前进、左转、右转、停止四种基本动作）
- **数据规模**：
  - 从Room-to-Room生成了60K个训练样本
  - 从Room-Across-Room生成了1.2M个训练样本
  - 覆盖了61个训练场景

<div align="center">
  <img src="/images/vln-ego-dataset.png" width="100%" />
<figcaption>
VLN-Ego数据集构建流程
</figcaption>
</div>

*长短期记忆采样：*
- 新颖的视频输入处理策略，用于动态平衡历史帧的重要性与当前观察的实时性
- 确保模型既能利用历史信息，又能快速响应当前环境变化
- 相比单一动作预测，多步动作预测结合历史上下文显著提升性能

**两阶段训练策略：**

<div align="center">
  <img src="/images/vln-r1-training-pipeline.png" width="100%" />
<figcaption>
VLN-R1两阶段训练流程
</figcaption>
</div>

*监督微调（SFT）阶段：*
- 模型的动作序列预测与专家演示对齐，通过监督学习优化输出文本
- 模型生成的多步动作序列文本与地面真值对齐，通过交叉熵损失进行优化
- 给定历史观察序列H_t、指令Z和当前观察O_t，模型预测n步未来动作序列

*强化微调（RFT）阶段：*
- 引入基于GRPO（Group Relative Policy Optimization）的强化学习方法
- 结合时间衰减奖励机制（TDR），进一步优化模型在长时程导航中的性能
- 超参数经过消融实验确定，生成次数选择8作为默认值

**时间衰减奖励机制（TDR）：**
- **核心思想**：通过引入衰减因子，平衡短期和长期奖励
- **作用机制**：使模型能够更关注近期的动作，同时考虑长期目标
- **优势**：用于评估多步动作预测的长期效果，优化长时程导航性能

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-r1-tdr-mechanism.png" width="100%" />
<figcaption>
时间衰减奖励（TDR）机制示意图
</figcaption>
</div> -->

**模型架构：**
- **输入**：自我中心视频流和指令
- **输出**：多步未来动作序列
- **端到端设计**：消除了对导航图的依赖，使其在连续环境中表现出色

**核心结果/发现**

VLN-R1在VLN-CE（视觉-语言导航连续环境）基准上进行了全面测试：

**测试平台：**
- **Room-to-Room（R2R）**：要求智能体在单个房间内导航
- **Room-Across-Room（R4R）**：要求智能体跨房间导航，任务更具挑战性

**性能表现：**
- **R2R数据集**：展现了高效的导航能力和准确的任务完成率
- **R4R数据集**：通过强化微调显著提升了跨域适应性，小型2B模型的性能甚至接近7B模型
- **模型可扩展性**：证明了端到端框架在不同模型规模下的有效性

**消融实验验证：**
- **长短期记忆采样**：多步动作预测结合历史上下文显著提升性能，优于单一动作预测
- **TDR机制**：与传统奖励函数相比，TDR显著提高了长时程任务的成功率
- **生成次数**：从6增加到8时性能提升有限，因此选择8作为默认值

**技术优势：**
- 端到端设计实现了实时导航
- 结合LVLM的视觉-语言理解能力和强化学习的优化策略
- 展示了在任务特定推理中的潜力

**局限性**

论文内容相对简短，未详细说明具体的性能指标数值（如SR、SPL等）和与其他SOTA方法的详细对比。此外，对于Real-world部署的讨论较少，主要集中在仿真环境（Habitat）测试，缺乏真实机器人平台上的验证实验。

---

## 7. LagMemo (2025)
——Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation

**研究背景/问题**

传统视觉导航方法受限于单目标、单模态和封闭类别设置，无法满足实际应用中多模态开放词汇表多目标导航的需求。现有方法如端到端强化学习依赖隐式状态编码导致泛化能力差，而模块化方法基于2D语义地图仅支持预定义类别，无法适应开放词汇场景。

**主要方法/创新点**

LagMemo提出了首个将语言特征融入3D Gaussian Splatting（3DGS）的视觉导航系统：

<div align="center">
  <img src="/images/lagmemo-framework-overview.png" width="100%" />
<figcaption>
LagMemo系统框架：先进行前沿探索构建语言3DGS记忆，再基于记忆进行多目标导航
</figcaption>
</div>

**语言3DGS记忆重建：**

*前沿探索与几何重建：*
- **前沿探索策略**：智能体首先进行基于前沿的环境探索，收集RGB-D图像和姿态信息
- **3DGS几何重建**：3D高斯由空间位置μ、颜色c、半径r和不透明度o参数化，通过颜色和深度渲染的几何损失优化
- **关键帧检索机制**：针对导航场景中帧间重叠有限导致的遗忘和表面空洞问题，引入帧池存储历史帧，周期性渲染并按PSNR评估，优先优化低保真帧

<div align="center">
  <img src="/images/lagmemo-language-injection.png" width="100%" />
<figcaption>
语言特征注入流程：SAM实例分割→CLIP特征提取→2D-3D特征关联→离散化码本
</figcaption>
</div>

*语言特征注入：*
- **实例级特征提取**：使用SAM生成实例掩码，通过特征splatting渲染逐像素语义特征，聚合掩码级特征并优化
- **两级码本量化**：粗分区联合考虑3D位置和语言特征，细分区仅基于语言特征细化类别
- **2D-3D特征关联**：对每个实例类别，渲染其高斯并评估与2D实例掩码的空间语义一致性，将高维CLIP实例特征分配给离散化的3D高斯语言类别
- **码本构建**：每个码本条目对应一簇3D高斯，富含CLIP特征，支持多模态查询（文本/图像）

**记忆引导的视觉导航：**

<div align="center">
  <img src="/images/lagmemo-navigation-pipeline.png" width="100%" />
<figcaption>
记忆引导导航管线：目标定位→路径点导航→目标验证→终点导航
</figcaption>
</div>

- **目标定位**：多模态输入目标（文本/图像）通过CLIP编码器编码，与码本计算余弦相似度定位候选实例，计算高斯质心并投影到2D障碍地图生成路径点
- **路径点导航**：使用Fast Marching Method（FMM）从当前位置规划无碰撞路径至路径点
- **目标验证与匹配**：到达路径点后全景扫描验证目标，对象/文本目标使用SEEM开放词汇实例分割和CLIP相似度二次验证，图像目标使用LightGlue特征匹配
- **终点导航**：确认目标可见后，利用SEEM掩码和深度信息将目标点投影到障碍地图，再次使用FMM导航至终点并执行STOP

**GOAT-Core基准数据集：**
- 针对GOAT-Bench质量问题策划高质量核心子集：每集20个子任务（原7.88个），平均13.37个独特类别（原4.82个），子任务间平均距离6.89m（原5.18m）
- 手动修正不准确文本描述，优先语义清晰的对象，限制所有子任务在单层楼
- 包含480个多模态子任务（163图像、158对象、159文本目标）

**核心结果/发现**

**目标定位任务：**
- 在GOAT-Core上总体成功率70.8%，显著优于VLMaps（58.8%）
- 分模态性能：对象88.4% vs 69.7%，图像56.4% vs 43.3%，文本66.8% vs 61.0%
- 语言3DGS保留详细3D空间上下文，实现精确定位，而VLMaps的2D网格压缩丢失关键几何信息

**多模态多目标视觉导航：**
- 平均成功率SR=56.3%，SPL=35.3%，在所有四个场景中均取得最高成功率
- 相比次优基线CoWs*，SR提升10%，相比Modular GOAT提升18%
- 文本导航任务优势尤为显著，充分展现丰富3D语义表示的优势
- 消融实验验证：移除关键帧机制导致PSNR从27.20降至21.15，定位准确率从70.8%降至66.3%；移除码本定位准确率骤降至34.6%
- 目标验证机制至关重要：无验证SR=41.3%，通用CLIP验证SR=46.7%，模态特定验证SR=56.3%

**局限性**

当前方法依赖静态固定容量记忆，在动态环境（重新排列、增删物体）中适应性不足。全场景3D表示内存密集，需要层次化或多分辨率高斯、不确定性感知剪枝和特征压缩。此外，目标定位能力依赖几何保真度，视角覆盖不足会留下盲区，未来需开发记忆感知的主动探索策略。

---



## 8. GaussNav (2025)
——Gaussian Splatting for Visual Navigation

**研究背景/问题**

Instance ImageGoal Navigation (IIN)要求智能体在未探索环境中定位并导航至目标图像所描绘的特定对象实例，需要跨视角识别目标对象同时忽略干扰物。现有基于BEV地图的导航方法缺乏详细纹理表示，难以胜任实例级任务，无法保留场景的实例感知特征，不足以区分同类别的多个对象。

**主要方法/创新点**

GaussNav首次将3D Gaussian Splatting（3DGS）引入具身视觉导航，提出语义高斯地图表示：

<div align="center">
  <img src="/images/gaussnav-framework-overview.png" width="60%" />
<figcaption>
GaussNav整体框架：前沿探索→语义高斯构建→高斯导航
</figcaption>
</div>

**前沿探索（Frontier Exploration）：**
- 智能体同时维护探索地图和障碍地图，探索地图标记已探索区域，障碍地图标记场景中的障碍物
- 检测探索地图轮廓并排除障碍地图区域，将最近的前沿点设为路径点，迭代覆盖整个环境

**语义高斯构建（Semantic Gaussian Construction）：**

*几何重建：*
- **3DGS简化表示**：每个高斯由9个参数特征化：RGB颜色向量c、质心µ∈R³、半径r、不透明度o∈[0,1]、类别标签l
- **可微渲染**：通过alpha合成渲染RGB、深度和轮廓图像，支持新视角合成（NVS）
- **关键帧检索机制**：针对导航场景帧间重叠有限问题，存储历史帧并周期性渲染评估PSNR，优先优化低保真帧，采用两阶段优化（p1=30迭代新视点，p2=60迭代关键帧视点）

<div align="center">
  <img src="/images/gaussnav-semantic-gaussian-construction.png" width="60%" />
<figcaption>
语义高斯构建流程：高斯密集化与语义高斯更新交替进行
</figcaption>
</div>

*语义特征注入：*
- **实例分割**：使用Mask-RCNN为每个高斯分配语义标签
- **特征优化**：通过特征splatting渲染逐像素语义特征，优化特征损失以鼓励实例内一致性和实例间可分性
- **高斯聚类**：基于语义标签和3D位置聚类高斯，将场景中的对象分割为不同语义类别下的不同实例

**高斯导航（Gaussian Navigation）：**

<div align="center">
  <img src="/images/gaussnav-navigation-pipeline.png" width="80%" />
<figcaption>
高斯导航流程：分类器→渲染描述性图像→匹配与定位→路径规划
</figcaption>
</div>

- **分类器**：使用ResNet50对目标图像分类预测语义标签ˆlg，显著缩小搜索空间（如场景CrMo8WxCyVb从648个潜在观测减少到33个）
- **匹配与定位**：
  - 为每个候选实例通过NVS生成描述性图像（nv=1/3/5，θ=±15°/±30°水平和垂直旋转）
  - 使用DISK提取关键点和特征描述符，通过LightGlue匹配，选择匹配关键点数最多的候选对象
  - 使用DBSCAN聚类去除语义分割误差导致的离群点，精确定位目标实例
- **路径规划**：将语义高斯转换为点云并体素化投影到2D BEV网格，使用FMM生成最短距离场并规划路径

**创新要点：**
- 统一几何、语义和实例感知特征的地图表示，首次将3DGS应用于具身视觉导航
- 通过渲染描述性图像直接定位目标对象，无需额外探索或验证步骤
- 关键帧检索机制有效缓解导航场景中的遗忘和表面空洞问题

**核心结果/发现**

- **HM3D数据集性能**：SPL从0.347大幅提升至0.578（提升66.6%），成功率达72.5%，显著超越所有基线方法
- **效率优势**：运行帧率超过20 FPS，在模块化方法中效率最高，搜索空间优化显著（如CrMo8WxCyVb场景从648个观测点减少至33个）
- **消融实验验证**：
  - 移除分类器导致Success降至37.5%，SPL降至29.1%，但使用分类器后匹配时间减少2.5倍
  - 移除匹配模块Success降至44.4%，SPL降至35.3%
  - NVS对识别成功率有益，GT NVS可进一步提升性能（Success从72.3%升至74.7%）
  - 使用GT匹配模块Success提升至85.0%，GT目标定位Success达94.6%
- **渲染质量分析**：在HM3D验证集上PSNR最高可达40，深度渲染误差接近零，但部分高纹理场景重建质量欠佳
- **跨场景泛化**：在36个验证场景中表现稳定，语义高斯可视化展示了对多种场景复杂度和对象组成的鲁棒性

**局限性**

当前方法在高纹理环境中重建质量欠佳，导致NVS可能产生孔洞等伪影。错误源分析显示匹配失败和目标定位不准确仍有改进空间。语义高斯不适合直接路径规划，需转换为2D BEV网格，增加了计算开销。

---


## 9. VLFM (2023)
——Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

**研究背景/问题**
零样本语义导航要求机器人在未见环境中高效定位目标对象，现有方法（如ESC、SemUtil）依赖物体检测器将视觉线索转化为文本后再用LLM/BERT进行语义推理，存在计算瓶颈且无法充分利用视觉-语言联合表征。如何直接从RGB观测中提取语义价值以指导前沿探索成为关键挑战。

**主要方法/创新点**

VLFM提出语言驱动的前沿价值图框架，实现端到端视觉-语义推理：

<div align="center">
  <img src="/images/vlfm-system-overview.png" width="100%" />
<figcaption>
VLFM系统架构：初始化、语义前沿探索、目标导航三阶段流程
</figcaption>
</div>

**核心机制：**

1. **前沿航点生成（Frontier Waypoint Generation）**
   - 利用深度和里程计构建2D占用地图，识别已探索与未探索区域边界作为前沿候选点
   - 每个前沿中点作为潜在导航航点

2. **价值图生成（Value Map Generation）**
   - 使用预训练BLIP-2视觉-语言模型直接从RGB图像计算语义价值分数
   - 文本提示："Seems like there is a <target object> ahead"
   - 输出余弦相似度分数并投影到俯视图价值图（双通道：语义分数+置信度分数）

<div align="center">
  <img src="/images/vlfm-value-map-generation.png" width="100%" />
<figcaption>
价值图生成流程：BLIP-2计算语义分数并投影到俯视图
</figcaption>
</div>

1. **置信度加权更新（Confidence-Weighted Averaging）**
   - 置信度分数基于像素相对光轴位置：$c_{i,j} = \cos^2(\theta/(\theta_{fov}/2) \times \pi/2)$
   - 重叠区域的语义值更新：$v_{i,j}^{new} = (c_{i,j}^{curr}v_{i,j}^{curr} + c_{i,j}^{prev}v_{i,j}^{prev})/(c_{i,j}^{curr} + c_{i,j}^{prev})$
   - 置信度更新偏向高置信值：$c_{i,j}^{new} = ((c_{i,j}^{curr})^2 + (c_{i,j}^{prev})^2)/(c_{i,j}^{curr} + c_{i,j}^{prev})$

<div align="center">
  <img src="/images/vlfm-confidence-weighting.png" width="100%" />
<figcaption>
置信度评分机制：光轴附近像素置信度最高，边缘递减
</figcaption>
</div>

1. **物体检测与导航**
   - YOLOv7用于COCO类别，Grounding-DINO用于开放词汇检测
   - Mobile-SAM提取目标轮廓，确定最近点作为目标航点
   - 使用VER训练的PointNav策略执行航点导航（纯几何理解，不依赖语义）

**关键创新：**
- 直接视觉-语义推理：绕过物体检测器，BLIP-2直接从RGB生成语义分数
- 空间化价值表征：将语义价值映射到俯视图网格，支持前沿选择
- 置信度驱动融合：动态平衡当前观测与历史信息

**核心结果/发现**
- **基准测试表现**：在Gibson、HM3D、MP3D三个数据集上均达到SOTA零样本性能
  - Gibson：SPL 52.2%、SR 84.0%（相比SemUtil提升+11.7% SPL、+14.7% SR）
  - HM3D：SPL 30.4%、SR 52.5%（相比ESC提升+8.1% SPL、+13.3% SR）
  - MP3D：SPL 17.5%、SR 36.4%（相比ESC提升+3.3% SPL、+7.7% SR）
- 超越部分有监督方法：在Gibson和MP3D数据集上优于SemExp、PONI等ObjectNav训练方法
- **消融实验**：置信度加权平均（Weighted avg.）在所有数据集上均优于简单替换（Replacement）和无权平均（Unweighted avg.）
- **真实世界部署**：成功在Boston Dynamics Spot机器人上部署，在办公楼环境中高效导航至未见目标对象，所有模型（BLIP-2、GroundingDINO、MobileSAM、ZoeDepth）实时运行于RTX 4090 MaxQ笔记本

**局限性**
仅支持单层楼导航（缺少z坐标里程计导致价值图重置困难），HM3D和MP3D中14.6%和9.6%的跨楼层任务失败；假定目标物体在默认相机高度可见，未来可探索主动相机控制、操作式搜索（如打开抽屉）及可复用的语义地图表征以支持长时程多任务规划。

## 10. Motus (2025)
——A Unified Latent Action World Model

**研究背景/问题**

当前具身智能体的理解、世界建模和控制能力被孤立地建模在不同模型中,这种碎片化阻碍了统一多模态生成能力的实现,也限制了从大规模异构数据中学习。现有方法将本应统一的系统分割为5个独立的建模任务:VLA(视觉-语言-动作模型)、WM(世界模型)、IDM(逆动力学模型)、VGM(视频生成模型)和视频-动作联合预测模型。两个核心挑战包括:如何在单一框架中统一这些多模态生成能力,以及如何利用大规模异构数据(互联网视频、自我中心人类演示、多机器人轨迹)进行动作专家的预训练。

**主要方法/创新点**

<div align="center">
  <img src="/images/motus-architecture-overview.png" width="90%" />
<figcaption>
Motus整体架构:Mixture-of-Transformer结构整合理解专家、视频生成专家和动作专家
</figcaption>
</div>

Motus提出了统一的潜在动作世界模型,通过以下创新实现五种建模范式的融合:

**1. Mixture-of-Transformer (MoT)架构:**
- **三模态联合注意力(Tri-model Joint Attention)**:将三个专家的多头自注意力层连接起来,在保留各专家特定功能的同时实现跨模态知识融合
  - 理解专家(Understanding Expert):基于Qwen3-VL-2B(253.5M参数),具备3D定位和空间理解能力
  - 视频生成专家(Video Generation Expert):采用Wan 2.2 5B作为视频基础模型
  - 动作专家(Action Expert):Transformer结构(641.5M参数),与Wan相同深度
- **总模型规模**:8B参数(VGM 5.00B + VLM 2.13B + Act Expert 641.5M + Und Expert 253.5M)

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-tri-model-attention.png" width="100%" />
<figcaption>
三模态联合注意力机制详解
</figcaption>
</div> -->

**2. UniDiffuser式调度器:**
- 为视频和动作分配不同的时间步τ_o、τ_a和噪声尺度
- 支持五种推理模式的灵活切换:VLA、世界模型、IDM、VGM、视频-动作联合预测
- 使用rectified flow目标函数:
  - $$l_{\text{action}} = \mathbb{E} \left[ \left\| v^{\theta_a} - (\epsilon_a - a_{t+1:t+k}) \right\|^2 \right]$$
  - $$l_{\text{obs}} = \mathbb{E} \left[ \left\| v^{\theta_o} - (\epsilon_o - o_{t+1:t+k}) \right\|^2 \right]$$
  
<!-- - l_action = E[||v^θ_a - (ε_a - a_{t+1:t+k})||²]   - l_obs = E[||v^θ_o - (ε_o - o_{t+1:t+k})||²] -->

**3. 潜在动作(Latent Actions) - 像素级"增量动作":**

<div align="center">
  <img src="/images/motus-latent-action-vae.png" width="60%" />
<figcaption>
潜在动作VAE架构:从光流到潜在动作表示
</figcaption>
</div>

- **光流表示**:使用DPFlow计算光流作为通用运动表示,将其转换为RGB图像
- **深度压缩自编码器(DC-AE)**:将高维光流压缩为4×512维token,再通过轻量级编码器投影到14维潜在动作向量
- **训练策略**:混合90%无标注数据(自监督重建)+10%有标注轨迹(任务无关数据+标准演示)
- **分布对齐**:引入任务无关数据(AnyPos方法),使用Curobo随机采样目标机器人动作空间
- **损失函数**:$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_a \left\| a_{\text{real}} - a_{\text{pred}} \right\|^2 + \beta \mathcal{L}_{\text{KL}}$$
<!-- - **损失函数**:L = L_recon + λ_a||a_real - a_pred||² + βL_KL -->

**4. 动作密集-视频稀疏预测策略:**
- 视频帧率:8帧 @ 5Hz
- 动作块:48步 @ 30Hz
- 通过下采样视频帧平衡token数量,防止过拟合视频预测而削弱动作预测能力

**5. 三阶段训练流程:**

<!-- <div align="center">
  <img src="/images/motus-training-pipeline.png" width="100%" />
<figcaption>
Motus三阶段训练流程与数据金字塔
</figcaption>
</div> -->

- **阶段1(视频生成)**:使用多机器人轨迹、自我中心人类视频和合成数据适配VGM(仅训练VGM,约8000 GPU小时)
- **阶段2(潜在动作统一训练)**:冻结VLM,在视频、语言和潜在动作上预训练整个Motus模型(约10000 GPU小时)
- **阶段3(监督微调)**:在目标机器人数据上使用真实动作微调(约400 GPU小时)

**6. 六层数据金字塔:**
- **Level 1**: Web数据(VGM和VLM预训练)
- **Level 2**: 自我中心人类视频(Egodex: 230,949样本)
- **Level 3**: 合成数据(RoboTwin: 27,500样本)
- **Level 4**: 任务无关数据(AnyPos: 1,000样本)
- **Level 5**: 多机器人任务轨迹数据(Agibot: 728,209 + RDT: 6,083 + RoboMind: 16,861)
- **Level 6**: 目标机器人任务轨迹数据(In-house: 2,000样本)

<div align="center">
  <img src="/images/motus-embodied-data-pyramid.png" width="100%" />
<figcaption>
具身数据金字塔:从Level 1到Level 6数据量递减但质量递增
</figcaption>
</div>

**核心结果/发现**

**仿真环境(RoboTwin 2.0)性能:**
- 随机化场景平均成功率:87.02%(Motus) vs 72.84%(X-VLA) vs 43.84%(π0.5)
- 相比X-VLA提升15%,相比π0.5提升45%
- 在50个任务上评估,包含强背景和环境随机化(随机背景、杂乱桌面、桌高扰动、随机光照)
- 清洁场景成功率:88.66%(Motus) vs 72.80%(X-VLA) vs 42.98%(π0.5)

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-robotwin-results.png" width="100%" />
<figcaption>
RoboTwin 2.0仿真基准测试结果对比
</figcaption>
</div> -->

**真实世界实验:**
- **两个平台**:AC-One和Agilex-Aloha-2双臂机器人
- **9个复杂任务**:测试空间理解、可变形物体操作、精确流体控制、视觉理解、长时域规划
  - 任务包括:叠毛巾、使用滴滤咖啡机煮咖啡、研磨咖啡豆、将面包放入烤箱、从饮水机取水、倒水浇花、按键盘按键

- **AC-One平台**:平均部分成功率63.22%(Motus) vs 25.86%(无预训练) vs 14.79%(π0.5)
  - 突出任务:研磨咖啡豆92% vs 0%(无预训练),煮咖啡62% vs 0%,放立方体入盘100% vs 60%

- **Agilex-Aloha-2平台**:平均59.30%(Motus) vs 26.60%(无预训练) vs 48.60%(π0.5)
  - 突出任务:从饮水机取水96% vs 8%(无预训练),叠毛巾39% vs 0%

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-real-world-tasks.png" width="100%" />
<figcaption>
Motus在真实世界复杂任务上的执行展示
</figcaption>
</div> -->

**其他基准测试:**
- **LIBERO-Long**:97.6%成功率(与X-VLA并列最优,达到state-of-the-art)
- **VLABench**: In Distribution平均0.48(vs π0.5的0.43),Cross Category平均0.25(vs π0.5的0.22)

**消融实验验证:**
- **训练阶段重要性**:完整Motus(阶段2预训练) 87.02% vs 仅阶段1 81.86%(+10.02%提升)
- **IDM模式性能**:动作MSE 0.014(Motus) vs 0.044(ResNet18+MLP) vs 0.122(DINOv2+MLP),显著优于专门训练的IDM基线
- **VLA模式竞争力**:83.90%成功率,与联合模式87.02%性能接近
- **世界模型生成质量**:FID 11.209,FVD 61.209,SSIM 0.866,PSNR 25.07(在两个平台上评估)

**五种统一模式实证验证:**
$$
\begin{aligned}
\text{1. VLA:} & \quad p(a_{t+1:t+k} \mid o_t, \ell) && \text{--- 从观察和语言预测动作} \\
\text{2. 世界模型:} & \quad p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k}) && \text{--- 从当前观察和动作预测未来观察} \\
\text{3. IDM:} & \quad p(a_{t+1:t+k} \mid o_{t:t+k}) && \text{--- 从观察序列推断动作} \\
\text{4. VGM:} & \quad p(o_{t+1:t+k} \mid o_t, \ell) && \text{--- 从观察和语言生成未来视频} \\
\text{5. 联合预测:} & \quad p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, \ell) && \text{--- 同时生成视频和动作}
\end{aligned}
$$

<!-- 1. VLA: p(a_{t+1:t+k} | o_t, ℓ) - 从观察和语言预测动作
2. 世界模型: p(o_{t+1:t+k} | o_t, a_{t+1:t+k}) - 从当前观察和动作预测未来观察
3. IDM: p(a_{t+1:t+k} | o_{t:t+k}) - 从观察序列推断动作
4. VGM: p(o_{t+1:t+k} | o_t, ℓ) - 从观察和语言生成未来视频
5. 视频-动作联合预测: p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, ℓ) - 同时生成视频和动作 -->

<!-- <div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-unified-modes-visualization.png" width="100%" />
<figcaption>
Motus五种统一模式的可视化展示
</figcaption>
</div> -->

**局限性**

当前方法需要大量计算资源(总计约18,400 GPU小时训练)。某些复杂任务(如叠毛巾)的性能仍有限,部分成功率仅为39%。尽管通过潜在动作改进了跨具身泛化,但仍需进一步研究。未来工作将探索更先进的统一模型架构,追求更通用的运动先验,并从互联网规模的通用视频中学习潜在动作。此外,需要研究如何降低部署成本并提升模型在极端条件下的鲁棒性。

---
## 11. NavGPT （2024）
——利用大语言模型进行视觉语言导航的显式推理
📄 **Paper**: https://arxiv.org/abs/2305.16986

**研究背景/问题**

现有的视觉语言导航(VLN)方法虽然性能较好,但其决策过程是隐式的、不可解释的。大语言模型(LLMs)在训练中展现出强大的推理能力和知识储备,但其在具身导航任务中的推理能力尚未被充分探索。研究的核心问题是:LLMs能否理解以文本形式描述的交互世界、动作及其后果,并利用这些信息解决导航任务?

**主要方法/创新点**

NavGPT是一个完全基于LLM的指令跟随导航系统,通过零样本方式执行视觉语言导航任务。系统包含四个核心组件:

<div align="center">
  <img src="/images/navgpt-architecture.png" width="100%" />
<figcaption>
NavGPT系统架构图
</figcaption>
</div>

1. **导航系统原则(Navigation System Principle)**: 定义VLN任务和LLM的基本推理格式与规则
2. **视觉基础模型(Visual Foundation Models)**: 使用BLIP-2将视觉观察转换为自然语言描述,使用Faster-RCNN检测物体并提取深度信息
3. **导航历史(Navigation History)**: 维护观察、推理和动作的三元组历史,使用GPT-3.5进行摘要以控制长度
4. **提示管理器(Prompt Manager)**: 将所有信息整合为LLM可理解的自然语言提示

关键创新在于**协同推理与动作(Synergizing Reasoning and Actions)**:
- 扩展动作空间为 A˜ = A ∪ R,其中R为推理轨迹
- 在每步导航前先生成推理(Thought),再做出动作决策(Action)
- 推理不触发环境交互,但能增强LLM的问题解决能力

<div align="center">
  <img src="/images/navgpt-visual-perception.png" width="100%" />
<figcaption>
视觉感知转换为语言描述的过程
</figcaption>
</div>

NavGPT展现出多种高级导航规划能力:
- 将指令分解为子目标
- 整合与导航相关的常识知识
- 从观察场景中识别地标
- 跟踪导航进度
- 处理异常情况并调整计划

**核心结果/发现**

在R2R val unseen数据集上的实验结果(使用GPT-4):
- Success Rate (SR): 34%
- SPL: 29%
- Oracle Success Rate (OSR): 42%

虽然与训练模型仍有约40%的性能差距,但NavGPT展示了LLM的强大能力:
- **高级规划能力**: 能够分解指令、识别地标、跟踪进度、适应异常
- **生成能力**: 可根据导航历史生成高质量的导航指令
- **空间意识**: 能够绘制准确的俯视图轨迹

人类评估显示LLM生成的推理质量可接受但仍有提升空间(准确性1.66/3.0,信息量1.93/3.0,合理性1.78/3.0)。

**局限性**

主要瓶颈在于:(1)视觉信号转换为自然语言时的信息损失;(2)历史观察摘要时的信息损失;(3)零样本性能与训练模型相比仍有较大差距(SR差距约40%)。未来方向建议采用多模态输入的LLMs或利用LLMs的高级推理能力来增强基于学习的模型。

---

## 12. NavGPT-2 （2024）
——释放大型视觉语言模型的导航推理能力
📄 **Paper**: https://arxiv.org/abs/2407.12366

**研究背景/问题**

虽然有将LLMs集成到VLN任务的努力,但存在两个极端方法的局限:(1)零样本方法依赖复杂的提示工程,存在信息损失且性能差距大(约40% SR);(2)微调方法虽然使用大规模LLMs,但性能仍落后于VLN专用模型,且丧失了LLMs的语言能力和可解释性。研究目标是在保持LLMs解释能力的同时,消除LLM-based agents与SOTA VLN专用模型之间的性能差距。

**主要方法/创新点**

NavGPT-2采用**冻结LLM + 导航策略网络**的混合架构,分两阶段训练:

<div align="center">
  <img src="/images/navgpt2-architecture.png" width="100%" />
<figcaption>
NavGPT-2模型架构
</figcaption>
</div>

**阶段一:视觉指令调优(Visual Instruction Tuning)**
- 基于InstructBLIP架构,使用Q-former将多视图图像编码为固定长度的视觉tokens
- 使用GPT-4V自动生成10K导航推理数据
- 仅微调Q-former和投影层,保持LLM和视觉编码器(EVA-CLIP ViT-g/14)冻结

<div align="center">
  <img src="/images/navgpt2-data-generation.png" width="100%" />
<figcaption>
GPT-4V导航推理数据生成流程
</figcaption>
</div>

**阶段二:图基础导航策略学习**
- 提取LLM隐藏层表示作为视觉-语言表征
- 采用拓扑图导航策略网络(源自DUET),包含:
  - 节点嵌入(Node Embedding):整合视觉特征、方向嵌入、步数嵌入
  - 跨模态编码(Cross-Modal Encoding):图感知自注意力(GASA)机制
  - 全局动作预测(Global Action Prediction):从整个构建的图中选择下一步
- 使用DAgger损失训练,保持VLM冻结

关键创新点:
1. **VLM隐表示作为视觉-语言表征**:将视觉特征投影到LLM的语言空间,实现更强的跨环境对齐
2. **数据高效**:利用LLM预训练权重,在50%数据量下即可达到DUET全量数据的性能
3. **保留语言能力**:冻结LLM使其保持生成导航推理和与人类交互的能力

**核心结果/发现**

在R2R数据集上的性能(NavGPT-2FlanT5-XXL, 5B参数):

| Split | SR | SPL | NE | OSR |
|-------|----|----|-----|-----|
| Val Unseen | 71% | 60% | 3.18 | 80% |
| Test Unseen | 72% | 60% | 3.33 | 80% |

主要发现:
- **消除性能差距**:在相同训练规模下,超越所有LLM-based方法,与DUET(SOTA VLN专用模型)性能相当
- **数据效率**:使用50% R2R数据即可达到DUET使用全量数据的性能
- **泛化能力**:
  - RxR数据集(细粒度指令):SR提升3.67%
  - HM3D数据集(未见环境):SR提升21.6%(47.2% vs 25.6%)
- **可解释性**:能够生成描述周围环境、识别导航进度、规划下一步的自然语言推理

<div align="center">
  <img src="/images/navgpt2-reasoning-examples.png" width="100%" />
<figcaption>
NavGPT-2生成的导航推理示例
</figcaption>
</div>

消融实验表明:
- 移除导航策略网络后性能大幅下降(SR从68%降至21%)
- FlanT5系列模型优于Vicuna系列(编码器-解码器架构优于纯解码器架构)
- 更强的视觉编码器对性能提升有限,主要增益来自LLM隐表示

**局限性**

(1)导航推理基于局部观察,未在VLM中建模历史,一致性有待提高;(2)推理与动作预测未严格同步;(3)存在幻觉问题(识别不存在的物体或误判方向);(4)交互能力未经充分评估。未来工作应聚焦于推理-动作同步机制、历史建模以及交互导航能力的开发。

### 系列对比总结

| 维度 | NavGPT (AAAI-2024) | NavGPT-2 (ECCV-2024) |
|------|-------------------|---------------------|
| **核心思路** | 纯LLM零样本导航 | 冻结LLM + 微调导航策略 |
| **训练方式** | 无需训练(零样本) | 两阶段训练(VLM微调+策略学习) |
| **性能(R2R SR)** | 34% | 72% (test unseen) |
| **推理能力** | 显式,基于提示工程 | 显式,基于指令调优 |
| **主要贡献** | 揭示LLM导航推理能力 | 消除LLM-agent与SOTA的性能差距 |
| **局限** | 性能差距大,信息损失严重 | 推理-动作同步不足,存在幻觉 |

两篇工作共同展示了LLMs在具身导航中的巨大潜力,从探索性的零样本方法发展到实用的混合架构,为构建可解释、可交互的通用导航智能体指明了方向。


---

## 13. GaussianAD (2024)
——Gaussian-Centric End-to-End Autonomous Driving
📄 **Paper**: https://arxiv.org/abs/2412.10371   🛖**代码仓库**：https://github.com/wzzheng/GaussianAD

**研究背景/问题**

现有的端到端自动驾驶方法在场景表示上存在困境：密集表示（如鸟瞰图BEV、体素）虽然全面但计算开销大，影响决策推理的资源分配；稀疏表示（如实例框、地图元素）虽然高效但无法捕获细粒度的3D结构信息，可能遗漏关键信息（如不规则障碍物、交通信号灯、人体姿态等），不符合端到端驾驶"全面信息流"的理念。

**主要方法/创新点**

<div align="center">
  <img src="/images/gaussianad-framework-overview.png" width="100%" />
<figcaption>
GaussianAD整体框架：使用统一3D高斯初始化场景，通过4D稀疏卷积和可变形交叉注意力融合多帧环视图像特征，支持密集任务（3D占用）和稀疏任务（3D检测、地图构建、运动预测），并通过高斯流预测进行轨迹规划
</figcaption>
</div>

本文提出GaussianAD框架，核心创新包括：

1. **3D语义高斯表示**：使用稀疏的3D语义高斯（每个高斯由均值、协方差和语义logits表征）作为场景中间表示，既保持稀疏性又具备高斯混合的通用逼近能力和显式3D结构。

2. **高斯场景构建**：
   - 用统一分布的3D高斯初始化场景（默认25600个）
   - 采用4D稀疏卷积实现高斯间的时序交互
   - 通过可变形交叉注意力逐步融合多帧环视图像特征来细化高斯

3. **灵活的感知架构**：
   - 密集任务（3D占用预测）：使用高斯到体素映射（Gaussian-to-Voxel Splatting）
   - 稀疏任务（3D检测、地图构建）：使用3D稀疏卷积+稀疏最大池化，直接从高斯预测
   - 运动预测：通过agent tokens和map tokens的交叉注意力获得motion tokens

<div align="center">
  <img src="/images/gaussianad-training-pipeline.png" width="100%" />
<figcaption>
训练流程：支持不同标注组合的灵活训练，利用显式高斯表示进行仿射变换预测未来场景，用未来感知标签或未来场景表示作为监督
</figcaption>
</div>

4. **高斯流（Gaussian Flow）预测**：
   - 为每个高斯预测3D流（future displacement）来建模场景演化
   - 覆盖动态和静态元素的全面场景预测
   - 利用规划轨迹进行全局仿射变换，预测自车观察到的未来场景表示
   - 用未来场景表示差异作为预测和规划的显式监督

5. **端到端训练**：
   - 总损失：J_GaussianAD = J_perc + J_pred + J_plan
   - 支持不同标注组合：3D占用、3D检测、地图、运动、场景预测
   - 场景预测监督无需额外标注，只需未来感知标签

<div align="center">
  <img src="/images/gaussianad-visualization.png" width="100%" />
<figcaption>
可视化结果：包含3D占用预测、3D目标检测、语义地图构建和轨迹规划的综合结果
</figcaption>
</div>

**核心结果/发现**

在nuScenes数据集上的实验结果：

1. **端到端规划性能（SOTA）**：
   - L2位移误差：1s/2s/3s分别为0.40/0.64/0.88米，平均0.64米
   - 碰撞率：1s/2s/3s分别为0.09%/0.38%/0.81%，平均0.42%
   - 相比使用相同监督信号的OccNet，大幅优于其2.14米的平均L2误差

2. **3D感知能力**：
   - 3D占用预测：mIoU 22.12%，IoU 33.81%（优于GaussianFormer等专用方法）
   - 3D目标检测：mAP 0.19（略低于专注检测的方法，但作为多任务模型表现合理）

3. **4D占用预测**：
   - 未来1s/2s/3s的mIoU分别为6.29%/5.36%/4.58%
   - 验证了高斯流的场景预测能力，尽管GaussianAD是端到端多任务模型

4. **消融研究**：
   - 使用运动监督对降低碰撞率特别有效
   - 基于流的场景预测监督可达到类似效果，且无需额外标注
   - 高斯剪枝（-40%）轻微降低感知性能但改善规划性能

**局限性**

GaussianAD无法预测完全准确的场景演化，因为它没有考虑自车向前移动时新观察到的区域的补全。这导致在4D占用预测任务上的性能相比专用方法有所不足。

---


## 14. FSR-VLN (2025)
——基于层次化多模态场景图的快慢推理视觉语言导航

📄 **Paper**: https://arxiv.org/abs/2509.13733v3

**研究背景/问题**

视觉语言导航（VLN）是具身智能中的基础任务，但现有方法在长距离空间推理方面存在严重局限，特别是在长距离导航任务中表现出较低的成功率和较高的推理延迟。关键瓶颈在于缺乏持久的长距离空间记忆来编码、组织和检索环境知识。现有几何语义地图和3D场景图依赖预提取的视觉特征，缺乏与VLM的直接交互；而基于图像的拓扑方法虽然成功率高，但由于依赖视频字幕处理长序列而效率低下。

**主要方法/创新点**

<div align="center">
  <img src="/images/fsr-vln-system-overview.png" width="100%" />
<figcaption>
系统总览：HMSG构建与基于FSR的导航推理流程
</figcaption>
</div>

FSR-VLN提出了结合两大核心创新的新型导航系统：

**1. 层次化多模态场景图（HMSG）表示**

HMSG将环境组织为四个层级：
- **楼层节点（Floor nodes）**：存储楼层标识符、名称、最小/最大高度、PLY点云以及所含房间节点的引用
- **房间节点（Room nodes）**：包含ID、2D多边形边界、点云、语义属性（名称、CLIP嵌入）以及关联的视图和对象节点链接
- **视图节点（View nodes）**（新颖贡献）：表示房间内的特定视觉视角，存储CLIP嵌入、VLM生成的描述、相机位姿以及与对象的可见性关系。该层支持使用VLM对图像视图进行推理，同时增强对象级定位能力
- **对象节点（Object nodes）**：表示离散实例，具有几何属性（3D边界框、点云）、语义嵌入以及与父房间和可见视图的链接

每个节点编码多模态特征，包括几何属性、语义信息和拓扑连接。HMSG使用FAST-LIVO2 SLAM系统提取RGBD数据和位姿来构建，然后进行开放词汇实例映射。GPT-4o从图像视图推断房间名称，系统计算每个对象的平均深度以选择最佳代表视图。

<div align="center">
  <img src="/images/hmsg-representation.png" width="100%" />
<figcaption>
四层HMSG层次结构，每个节点包含多模态特征
</figcaption>
</div>

**2. 快慢导航推理（FSR）**

受人类认知双过程理论启发，FSR分三个阶段运行：

**阶段1：基于LLM的用户指令理解**
- 对于空间指令（如"办公室里的蓝色圆柱形凳子"）：LLM充当层次化概念解析器，将输入分解为楼层、区域和对象组件
- 对于非空间指令（如"我累了"）：LLM充当目标推理代理，根据用户意图识别最相关的对象或区域

**阶段2：快速匹配（直觉检索）**
- 在查询文本与HMSG视图层嵌入之间进行基于CLIP的相似度匹配以识别目标视图
- 并行进行对象级匹配，使用查询文本与对象嵌入之间的CLIP特征
- 通过层次化特征匹配高效检索候选房间、视图和对象

**阶段3：慢速推理（深思熟虑的精化）**
- VLM（GPT-4o）验证匹配的对象是否出现在其最佳视图中
- 如果验证失败，系统会：
  - 使用LLM对未匹配视图的文本描述进行推理
  - 比较快速匹配视图与LLM选择的视图
  - 应用VLM推理确定最终最优目标图像
  - 通过重新计算与最终视图中对象的CLIP相似度来更新目标对象

<div align="center">
  <img src="/images/fsr-navigation-reasoning.png" width="100%" />
<figcaption>
三阶段FSR流程：LLM指令理解、基于CLIP的快速匹配、基于VLM的慢速推理
</figcaption>
</div>

这种多阶段架构无缝集成了高效的特征空间匹配与鲁棒的VLM驱动视觉验证。慢速推理仅在快速直觉失败时激活，大幅减少推理时间的同时提高准确性。

**核心结果/发现**

FSR-VLN在长距离真实室内环境中对87条机器人采集的指令进行评估，涵盖四个不同类别（无需推理、需要推理、小物体、空间目标）：

- **成功率（SR）**：92%（80/87），显著优于基线方法：
  - 比MobilityVLA（34.5%）高167%
  - 比OK-Robot（60.9%）高51%
  - 比HOVSG（51.7%）高77%

- **检索成功率（RSR@Top1）**：在4-5米距离阈值下达到96.6%，在所有距离阈值下始终保持最佳性能

- **响应时间**：使用慢速推理平均5.5秒，仅使用快速匹配平均1.5秒
  - 与MobilityVLM（30秒）相比响应时间减少82%
  - 通过仅在快速匹配失败时激活慢速推理，实现高效实时性能

- **HM3D-SEM数据集**：RSR@Top1在1米处达到87%，显著优于HOVSG（52%）和osmAG-LLM（28%）

- **消融实验**：添加空间目标（ST）指令使RSR从72.4%提升到81.6%，结合导航推理（NR）进一步提升到92%，验证了两个组件的有效性

该系统已成功集成到Unitree-G1人形机器人的语音交互、规划和控制模块中，展示了具备自然语言交互能力的真实世界部署能力。

**局限性**

HMSG构建耗时，不适合实时建图。系统假设静态环境，限制了在动态场景中的适用性。未来工作将重点提高场景图构建效率、扩展对动态环境的鲁棒性，以及集成探索性导航能力以处理新颖或模糊的场景。


---

## 15. VLingNav (2026)
Embodied Navigation with Adaptive Reasoning and Visual-Assisted Linguistic Memory
——A Unified VLA Model with Adaptive CoT and Visual-Assisted Linguistic Memory

📄 **Paper**: https://wsakobe.github.io/VLingNav-web/

<div align="center">
  <img src="/images/VLingNav_architecture.png" width="100%" />
<figcaption>
VLingNav 整体架构概述，展示了AdaCoT推理和VLingMem记忆模块。
</figcaption>
</div>

**精华**

该论文提出了VLingNav框架，通过自适应链式思考（AdaCoT）和视觉辅助语言记忆（VLingMem）赋予具身智能体认知能力，实现了高效且可解释的具身导航。其核心亮点在于动态推理机制和跨模态记忆，使其在各种具身导航基准测试中达到SOTA性能，并展示了强大的零样本迁移能力和跨任务泛化能力，为资源受限机器人平台上的智能导航提供了启发。

**研究背景/问题**

当前的具身导航VLA模型在复杂、长周期任务中缺乏明确的推理能力和持久性记忆，难以泛化到不同环境和任务变体。现有模型多为被动式系统，缺少自适应推理机制，并且依赖有限的上下文窗口，导致在复杂场景下无法有效规划和避免重复探索。

**主要方法/创新点**

本文提出了VLingNav，一个以语言驱动的VLA框架，旨在通过两个核心组件赋予具身智能体认知能力：

<div align="center">
  <img src="/images/VLingNav_framework.png" width="100%" />
<figcaption>
VLingNav 整体架构。
</figcaption>
</div>

1.  **自适应链式思考 (Adaptive Chain-of-Thought, AdaCoT)**：
受人类双进程理论启发，AdaCoT机制在必要时动态触发显式推理，使智能体能够根据任务复杂性在快速、直观执行和缓慢、深思熟虑的规划之间灵活切换。这解决了现有CoT方法中推理频率固定导致效率低下的问题。

<div align="center">
  <img src="/images/VLingNav-CoT-labeling-pipeline.png" width="100%" />
<figcaption>
VLingNav的自适应CoT标注流程图。
</figcaption>
</div>

2.  **视觉辅助语言记忆 (Visual-Assisted Linguistic Memory, VLingMem)**：
为了处理长周期的空间依赖性，VLingMem构建了一个持久的、跨模态的语义记忆，使智能体能够回忆过去的观察结果，防止重复探索，并推断动态环境中的移动趋势，从而确保在长时间交互中的连贯决策。


**训练数据和策略**：
- **Nav-AdaCoT-2.9M数据集**：构建了目前最大的具身导航数据集，包含推理标注和自适应CoT标注。
- **在线专家引导强化学习 (Online Expert-guided RL)**：在模仿学习（SFT）之后引入了在线专家引导RL阶段，使模型能够获得更鲁棒、自探索的导航行为，超越监督演示的局限性。

<div align="center">
  <img src="/images/VLingNav-online-training.png" width="100%" />
<figcaption>
在线后训练的混合rollout过程。
</figcaption>
</div>

**核心结果/发现**
- VLingNav在多项具身导航基准测试（如ObjectNav, EVT, ImageNav）上实现了最先进的性能。
- 在HM3Dv1 ObjectNav上，SR和SPL显著优于Uni-NaVid，展现了强大的探索和记忆能力。
- 在HM3D OVON上，VLingNav在所有测试拆分中均表现最佳，证明了其强大的跨领域泛化能力。
- 在EVT-Bench上，VLingNav在单目标跟踪和分心跟踪任务中均达到SOTA性能，尤其在复杂混乱场景中优势明显。
- 在Image Goal Navigation上，VLingNav的成功率和导航效率显著高于UniGoal，表明其先进的推理和规划能力。
- 在真实世界机器人平台上实现了零样本迁移，成功执行了未见过的导航任务，展示了强大的真实世界泛化和实用性。

**局限性**
- 当前模型主要依赖单目自我中心观测，这限制了其感知能力。未来工作可以探索多视角观测以提高导航效率。
- 模型采用单系统架构，限制了预测频率，可能影响在高度动态环境中的快速决策和障碍物处理。未来可升级为双系统结构以支持高频动作输出。
- 当前方法仅使用基于MPC的路点控制器，缺乏更灵活的运动模型，未来可集成更多运动能力。



---
## 16. Slow4fast-VLN (2026)
——General Vision-Language Navigation via Fast-Slow Interactive Reasoning

📄 **Paper**: https://arxiv.org/abs/2601.09111v1

### 精华 
这篇论文的核心借鉴价值在于其提出的动态交互式快慢脑（Fast-Slow Interactive Reasoning）导航框架。它通过模拟人类的“快思考”（直觉决策）和“慢思考”（深度反思），实现导航策略的持续优化。其亮点在于，慢脑系统能够从历史经验中提炼出可泛化的“导航知识”，并用其“赋能”快脑，从而有效提升了智能体在未知环境（OOD场景）中的泛化能力和决策效率，解决了传统方法中快慢系统割裂、经验无法沉淀的问题。

### 1. 研究背景/问题 
传统的视觉-语言导航（VLN）方法在封闭环境下表现良好，但在面对环境和指令风格多变的开放世界时，其泛化能力严重不足。GSA-VLN任务通过引入多样化的场景和指令，对模型的场景适应性提出了更高要求。当前方法的主要挑战在于如何让智能体在导航过程中动态生成可泛化的策略，以应对前所未见的场景和指令。

### 2. 主要方法/创新点 (Core content, most detailed)
论文提出了一个名为 **slow4fast-VLN** 的动态交互式快慢推理框架，以应对开放环境下的视觉语言导航挑战。该框架包含快慢两个核心模块：

*   **快推理模块 (Fast Reasoning)**：这是一个端到端的策略网络（基于DUET），负责根据实时的视觉和指令输入，快速生成导航动作。同时，它会记录导航过程中的所有执行记录（如观测、动作、度量等），形成历史记忆（History Repository）。

*   **慢推理模块 (Slow Reasoning)**：该模块是整个框架的核心创新点。它利用大语言模型（LLM）对快推理模块产生的历史记忆进行深度“反思”（Reflection），从中提取出结构化、可泛化的导航经验（Structured Experience），并存入一个经验库（Experience Library）。这些经验包含了场景类型、空间上下文、空间规则、导航策略等关键信息。

*   **快慢交互机制 (Interaction)**：这是区别于以往工作的关键。在导航决策时，快推理模块会从经验库中检索与当前场景最相关的经验，并将这些经验特征与实时视觉特征进行融合（通过Attention机制），从而“赋能”快脑，使其做出更精准、更泛化的决策。这种交互使得慢脑提炼的经验能够持续优化快脑的性能。

*   **指令风格转换 (Instruction Style Conversion)**：为了应对多样的指令风格（如场景化、用户个性化），论文还设计了一个基于LLM的指令转换模块，通过CoT提示工程，将不同风格的指令实时转换为统一的“基础风格”指令，降低了模型对指令变化的敏感度。

<div align="center">
  <img src="/images/slow4fast-VLN-framework.png" width="100%" />
<figcaption>
图1: slow4fast-VLN 框架概览，展示了快慢推理模块如何通过历史记忆和泛化经验进行交互，以适应不同环境。
</figcaption>
</div>

<div align="center">
  <img src="/images/slow4fast-VLN-overview.png" width="100%" />
<figcaption>
图2: 方法概览。策略网络（快推理）处理实时输入并存储历史，LLM（慢推理）反思历史并生成经验，这些经验反过来指导策略网络。
</figcaption>
</div>

### 3. 核心结果/发现 (Key findings)
*   **环境适应性**：在GSA-R2R数据集上，使用基础指令进行测试时，slow4fast-VLN在住宅（ID）和非住宅（OOD）场景中的成功率（SR）分别比基线方法GR-DUET提升了1.5%和2.2%，证明了快慢交互框架对于提升场景泛化能力的有效性。
*   **指令适应性**：在面对用户个性化指令和场景化指令时，该方法同样全面优于基线。例如，在用户指令测试中，其SR和SPL指标在多种角色（如Child, Keith, Moira等）下均达到SOTA水平。这得益于其指令风格转换模块和动态经验反馈循环。
*   **消融实验**：实验证明，快慢推理（FSR）框架和指令风格转换（ISC）模块都是有效的。当两者协同工作时，模型在最具挑战的Test-N-Scene任务上达到了最佳性能。
*   **案例研究**：通过可视化导航轨迹，论文展示了在引入慢脑反思后，智能体能够修正初始的错误路径，并基于经验（如“寻找蓝色画作”作为线索）更高效、更准确地完成导航任务，避免了不必要的探索。

<div align="center">
  <img src="/images/slow4fast-VLN-casestudy.png" width="100%" />
<figcaption>
图3: 案例研究。左图为仅使用快推理的轨迹，右图为经过慢推理优化后的轨迹，显示出路径更优，定位更准。
</figcaption>
</div>

### 4. 局限性 (Brief, 1-2 sentences)
论文指出的一个局限是，慢脑推理产生的知识是隐式地编码在策略网络的权重中，这种“黑盒”形式使得学习到的经验难以解释和直接干预。未来的一个研究方向是让慢脑生成显式的、结构化的知识库（如语义地图或知识图谱），以供快脑在导航时直接查询。


---
## 17. FantasyVLN (2026)
———统一多模态Chain-of-Thought推理用于视觉-语言导航

📄 **Paper**: https://arxiv.org/abs/2601.13976

**精华**
这篇论文展示了如何通过统一框架整合文本、视觉和多模态CoT推理模式,值得借鉴的点包括:(1) 训练时使用CoT监督、推理时直接预测的隐式推理范式,避免了显式CoT的token膨胀问题;(2) 使用预训练VAR模型将想象的视觉观测压缩到紧凑潜在空间,大幅降低序列长度;(3) 通过跨模态对齐约束统一不同推理模式,学习模态不变的推理表示;(4) 门控机制实现单一模型灵活切换多种推理模式。这种设计在保持推理能力的同时实现了实时导航,为具身智能任务提供了实用的解决方案。

**研究背景/问题**
现有VLN方法面临关键挑战:纯文本CoT缺乏空间理解且容易过拟合稀疏标注;多模态CoT通过生成想象的视觉观测引入严重的token膨胀,导致推理延迟增加数个数量级,无法实现实时导航。这在长时域、多阶段导航场景中尤为突出。

<div align="center">
  <img src="/images/FantasyVLN-overview.png" width="100%" />
<figcaption>
FantasyVLN系统概览:整合文本和视觉CoT推理模式,联合建模语义规划和空间理解
</figcaption>
</div>

**主要方法/创新点**

FantasyVLN提出了统一的隐式推理框架,核心创新包括:

**1. Compact Visual CoT (CompV-CoT)**
- 使用预训练的Visual AutoRegressor (VAR)模型将想象的视觉观测编码到紧凑潜在空间
- VAR采用next-scale预测范式,256×256图像仅需30个视觉token即可精确重建,压缩比达1/2185
- 训练时VLM直接生成VAR潜在表示,推理时无需显式VAR解码,大幅提升效率

**2. 统一多模态CoT (UM-CoT)框架**
- 通过二元门控信号 gT 和 gV 控制文本和视觉推理的激活
- 四种推理模式:(a) Non-CoT (gT=0, gV=0) 直接预测动作;(b) T-CoT (gT=1, gV=0) 生成文本推理步骤;(c) V-CoT (gT=0, gV=1) 生成压缩视觉想象;(d) MM-CoT (gT=1, gV=1) 联合生成文本-视觉推理
- 单一模型共享参数,通过数据混合实现端到端联合训练

<div align="center">
  <img src="/images/FantasyVLN-architecture.png" width="100%" />
<figcaption>
统一多模态CoT推理框架:支持四种推理模式,训练时使用CoT监督,推理时直接动作预测
</figcaption>
</div>

**3. 跨模态对齐约束 (Cross-Mode Alignment)**
- 将Non-CoT模式的动作预测作为软监督信号,对齐所有CoT变体的动作输出
- 交替优化Non-CoT目标和跨模态对齐的联合目标,嵌入多样化推理模式到统一潜在策略
- 防止不同推理模式间的冲突,学习一致的模态不变表示

**4. 隐式推理机制**
- 训练时:联合学习文本、视觉和多模态CoT模式
- 推理时:采用Non-CoT模式直接指令到动作映射,无需生成显式CoT序列
- 借鉴Aux-Think的"train-with-CoT, infer-without-CoT"范式,模型隐式保留推理感知表示

**训练细节**
- 基础模型:Qwen2.5-VL (7B参数)
- 数据:LH-VLN训练集18,554个导航轨迹切片(每5步一个切片)
- T-CoT标注:使用Qwen-VL-Max生成,包含语义规划、视觉描述、动作规划和视觉想象四部分
- 优化:LoRA微调,AdamW优化器,学习率1e-4,64×H20 GPUs,DeepSpeed ZeRO-2

<div align="center">
  <img src="/images/FantasyVLN-VAR-scale-comparison.png" width="100%" />
<figcaption>
不同VAR scale对ISR性能的影响:scale 4达到最佳平衡
</figcaption>
</div>

<div align="center">
  <img src="/images/FantasyVLN-VAR-reconstruction.png" width="100%" />
<figcaption>
VAR模型在不同scale下的图像重建质量对比:scale越高,重建质量越好,但token数量也越多
</figcaption>
</div>

**核心结果/发现**

**导航精度 (LH-VLN benchmark)**
- SR (成功率): 2.44% (所有基线中最佳)
- ISR (独立成功率): 11.01% (显著优于所有方法)
- CSR (条件成功率): 9.64%
- CGT (加权CSR): 8.99%
- 显著超越次优方法Aux-Think (仅T-CoT): SR提升3.75×,ISR提升3.5×

**推理效率**
- APS (每秒动作数): 1.03,与WorldVLA (1.02)和Aux-Think (0.97)相当
- 比显式CoT方法CoT-VLA (0.19 APS)快5.4×,推理延迟降低一个数量级
- 隐式推理每次预测仅解码单个token,而显式CoT需生成3k-5k个token

**训练效率**
- FantasyVLN在few thousand迭代内快速收敛,token预测准确率达到1.0
- WorldVLA (像素级V-CoT)需10k+迭代才能达到0.5准确率,且训练不稳定
- CompV-CoT通过潜在空间推理提供更强梯度信号和更稳定的学习动态

<div align="center">
  <img src="/images/FantasyVLN-training-efficiency.png" width="100%" />
<figcaption>
FantasyVLN与WorldVLA的训练效率对比:CompV-CoT快速收敛,像素级V-CoT训练缓慢且不稳定
</figcaption>
</div>

**消融实验**
- 各推理模式贡献:结合任何CoT模式与Non-CoT都能提升性能,四模式联合训练效果最佳
- VAR scale选择:scale 4最优(ISR 7.41%),更小scale信息不足,更大scale冗余
- 跨模态对齐:关键组件,移除后SR从2.44%降至0,ISR从11.01%降至2.39%
- 显式vs隐式推理:隐式推理在多模态设置下表现最佳(MM-CoT隐式:SR 2.44 vs 显式0.98)

**局限性**
该方法在LH-VLN这种小规模数据集(18k轨迹切片)上训练,显式CoT容易过拟合并产生累积误差;在更大规模数据集上的表现有待验证。此外,绝对成功率仍较低(SR 2.44%),表明长时域多阶段导航仍是极具挑战性的任务。


---
## 18. VL-Nav (2025)
——实时零样本 Vision-Language 导航系统，融合像素级视觉-语言特征与启发式空间推理

📄 **Paper**: https://arxiv.org/abs/2502.00931

**精华**

这篇论文展示了如何将像素级 vision-language 特征与启发式探索策略结合，实现高效的零样本导航。值得借鉴的核心思想包括：(1) 使用 Gaussian 混合模型将像素级 VL 特征转换为空间分布，而非依赖单一图像级相似度分数；(2) 引入 instance-based target points 模拟人类搜索行为，允许机器人接近并验证潜在目标；(3) 通过 rolling occupancy grid 和 partial frontier detection 优化计算开销，使系统能在低功耗平台上实时运行；(4) 结合 distance weighting 和 unknown-area heuristic 避免反复移动，提升大规模环境中的导航效率；(5) 证明了模块化方法在真实世界中的泛化能力优于端到端学习方法。

**研究背景/问题**

当前的 vision-language navigation 系统面临三大挑战：难以解释像素级 vision-language 特征、在不同环境中泛化能力差、无法在低功耗平台上实时运行。现有方法如 VLFM 依赖计算密集型模型且仅使用单一图像级相似度分数进行目标选择，限制了其利用细粒度 vision-language 线索的能力。

**主要方法/创新点**

<div align="center">
  <img src="/images/VL-Nav-system-overview.png" width="100%" />
<figcaption>
VL-Nav 系统架构总览：整合了 VL 模块、地图模块和 HVL 空间推理
</figcaption>
</div>

VL-Nav 提出了一个针对低功耗机器人优化的 vision-language navigation 框架，在 Jetson Orin NX 上实现 30 Hz 实时性能。核心创新在于 **Heuristic-Vision-Language (HVL) 空间推理**，将像素级 vision-language 特征与启发式探索策略相结合。

**Rolling Occupancy Map**：系统维护一个动态 2D 占用栅格地图，每个单元格标记为 free (0)、unknown (-1) 或 occupied (100)。与传统固定大小全局栅格不同，VL-Nav 采用 rolling grid，仅在新传感器数据需要时动态扩展，降低内存使用和 BFS/cluster 计算开销。更新过程包括：(1) 根据需要扩展地图；(2) 清除前向 FOV 内的过时障碍物；(3) 膨胀新障碍物；(4) 使用 raycasting 将 unknown cells 标记为 free。

**Frontier-based 与 Instance-based Target Points**：系统生成两类候选目标点。Frontier-based points 通过 partial frontier detection 在前向楔形区域内识别，仅测试满足角度和距离约束的单元格，并使用 BFS 聚类。Instance-based target points (IBTP) 来自 vision-language 检测器周期性报告的候选实例中心，保留置信度高于阈值 τdet 的检测结果。IBTP 模拟人类搜索行为：看到可能匹配的目标时会靠近确认，而非忽略中间检测结果。

<div align="center">
  <img src="/images/VL-Nav-spatial-reasoning.png" width="100%" />
<figcaption>
VL Scoring 示意图：像素级开放词汇检测结果通过 Gaussian 混合模型和 FOV 加权转换为空间分布
</figcaption>
</div>

**HVL 空间推理**：这是 VL-Nav 的核心创新。对每个候选目标 g，系统计算 HVL score。VL Score 使用 Gaussian 混合模型将像素级 vision-language 特征转换为机器人水平 FOV 上的分布。假设开放词汇检测模型识别出 K 个可能方向，每个由 (μk, σk, αk) 参数化，其中 μk 表示 FOV 内的平均偏移角度，σk 编码检测的角度不确定性（固定为 0.1），αk 是基于置信度的权重。VL score 计算为：

S_VL(g) = Σ(k=1 to K) αk * exp(-1/2 * ((Δθ - μk)/σk)²) * C(Δθ)

其中 C(Δθ) = cos²(Δθ/(θ_fov/2) * π/2) 是视野置信度项，降低大角度偏移检测的权重。

Heuristic Cues 包括两个启发式项：(1) Distance Weighting: S_dist(g) = 1/(1+d(xr,g))，使较近目标获得更高分数，减少能量消耗和不必要的徘徊；(2) Unknown-Area Weighting: S_unknown(g) = 1 - exp(-k*ratio(g))，其中 ratio(g) 是局部 BFS 中 unknown cells 与可达 cells 的比率，鼓励探索可能揭示大量未知空间的目标。

最终 HVL score 为：S_HVL(g) = w_dist * S_dist(g) + w_VL * S_VL(g) * S_unknown(g)。系统优先选择 instance-based goals（基于 VL score），若无则选择得分最高的 frontier goal（基于 HVL score）。

**Path Planning**：选定 HVL goal 后，系统使用 FAR Planner 进行 point-goal 路径规划，以多边形表示障碍物并实时更新可见性图，支持部分未知环境中的高效重规划。局部规划器将 FAR Planner 的路径点细化为短时域速度命令，确保对新障碍物的快速反应。
<div align="center">
  <img src="/images/VL-Nav-experiment-environments.png" width="50%" />
<figcaption>
四种不同规模和语义复杂度的真实世界实验环境
</figcaption>
</div>

**核心结果/发现**

<div align="center">
  <img src="/images/VL-Nav-trajectory-comparison.png" width="100%" />
<figcaption>
不同环境中的轨迹对比和检测结果，展示 VL-Nav 相比 Classical 和 VLFM 方法的优势
</figcaption>
</div>

VL-Nav 在四个真实世界环境（Hallway、Office、Apartment、Outdoor）上进行了全面评估，每个环境具有不同的语义复杂度（High、Medium、Low）和规模（Big、Mid、Small）。主要发现包括：

- **整体性能**：VL-Nav 达到 86.3% 的总体成功率 (SR)，比先前方法提升 44.15%。在所有四个环境中，VL-Nav 的 SR 和 SPL（Success weighted by Path Length）均为最高。
- **Instance-based Target Points 的影响**：去除 IBTP 后性能显著下降，特别是在复杂环境（Apartment 和 Office）中，证明了允许机器人接近并验证潜在检测结果的重要性。
- **Heuristics 的贡献**：去除启发式项后 SR 和 SPL 均下降，特别是在大规模环境中，表明 distance weighting 和 unknown-area heuristic 对提升效率至关重要。
- **相比 VLFM**：VL-Nav 在所有环境中均超越 VLFM，特别是在语义复杂（Apartment）和开放区域（Outdoor）环境中，优势更加明显，证明了像素级 VL 特征和 HVL 空间推理的有效性。
- **环境规模影响**：经典 Frontier Exploration 在大规模环境中性能急剧下降（Big 环境中 SR 仅 36.7%），而 VL-Nav 保持鲁棒（82.3% SR），证明了其在各种规模环境中的适应能力。
- **语义复杂度影响**：所有方法在语义更丰富的环境中表现更好，因为结构化室内空间提供了更强的检测和分割线索。VL-Nav 能够充分利用语义上下文，在高复杂度环境中获得更显著的优势。
- **实时性能**：VL-Nav 在 Jetson Orin NX 上以 30 Hz 运行，通过选择高效的 YOLO-World 模型变体（256×320 输入，标准 GPU runtime）和 rolling occupancy grid 实现了真实世界部署的可行性。

**局限性**

系统在处理包含隐藏对象引用和特定文本注释的复杂语言描述时存在困难。此外，系统依赖于手动定义的阈值（如光照条件等），这些阈值可能无法在不同环境和场景中很好地泛化，需要进一步研究自适应或基于学习的阈值调整方法。

---
## 19. StreamVLN (2025)

——— 通过慢-快上下文建模实现流式视觉-语言导航

📄 **Paper**: https://arxiv.org/abs/2507.05240

**精华**

这篇论文提出了一个适用于真实世界部署的流式 VLN 框架，值得借鉴的核心思想包括：(1) 采用慢-快双通道上下文建模策略，平衡全局场景理解和实时响应能力；(2) 利用 3D 几何信息进行智能 token 剪枝，在保持性能的同时显著降低计算开销；(3) 通过 KV cache 复用机制利用时间连贯性，支持长视频流的高效推理；(4) 将上下文大小和推理成本控制在有界范围内，为 embodied AI 的实际部署提供了可行方案；(5) 采用多源数据联合训练策略（VLA数据+通用VL数据+DAgger数据），同时保持通用推理能力和导航专业性能。这些设计思路可以迁移到其他需要处理长序列多模态输入的具身智能任务中。

**研究背景/问题**

现有的 Vision-and-Language Navigation 方法在处理真实世界连续环境时面临关键挑战：如何在长视频流中高效进行多模态推理，同时保持低延迟以支持实时交互。现有 Video-LLM 基于的 VLN 方法往往在细粒度视觉理解、长期上下文建模和计算效率之间存在权衡。本文旨在设计一个既能捕捉全局场景理解，又能快速响应的流式导航框架。

<div align="center">
  <img src="/images/StreamVLN-framework-overview.png" width="100%" />
<figcaption>
StreamVLN 整体框架：输入包括语言指令和 RGB 图像流，每个导航episode 被建模为多轮对话，智能体持续查询下一步动作。采用固定大小的滑动窗口保留最近的对话历史，通过 token 剪枝更新非活跃窗口的上下文以减少内存开销。
</figcaption>
</div>

**主要方法/创新点**

论文提出了 StreamVLN，一个基于慢-快上下文建模的流式视觉-语言导航框架，将 Video-LLM 扩展为交错的 vision-language-action 模型。


**1. 连续多轮自回归生成**

VLN 的多轮对话会话由一系列交错的观测和动作组成。在每个对话 $d_i = (o_i, a_i)$ 中，VLN 模型接收新观测 $o_i$ 并生成动作响应 $a_i$，条件于当前输入和对话历史。完整输入序列构造为：$o_1a_1o_2a_2...o_{i-1}a_{i-1}$。Transformer 基于 LLM 首先执行 **prefill 阶段**（预填充阶段）编码输入 token 并缓存 key/value（KV）状态，然后在 **decoding 阶段**使用缓存的 KV 对生成新 token。

**2. 快速流式对话上下文 (Fast-Streaming Dialogue Context)**

虽然跨轮次复用 KV cache 可以消除超过 99% 的 prefilling 时间，但会引入巨大的内存开销。随着对话数量增加，KV cache 呈线性增长（例如 2K token 可能消耗约 5GB 内存），使长会话变得不切实际。此外，现有 Video-LLM 在处理过长上下文时推理性能会下降。

StreamVLN 采用 **滑动窗口 KV cache** 管理对话上下文，保留固定数量 $N$ 的最近对话在活跃窗口中：$W_j = [o_{(i-N+1)}a_{(i-N+1)}...o_ia_i]$。当窗口达到容量时，key/value 状态从 LLM 中卸载，非观测对话 token（如提示词和生成的动作）的状态立即丢弃。对于新的滑动窗口，来自过去窗口的 token 状态被处理为记忆 token 状态 $\{M_0, ..., M_j\}$。

<div align="center">
  <img src="/images/StreamVLN-training-data-recipe.png" width="50%" />
<figcaption>
StreamVLN 的联合训练数据配方：67% VLA 导航数据（包括 MP3D 31%、HM3D 20%、DAgger 16%）+ 33% 通用多模态数据（VQA 17% + MMC4 16%），确保在保持导航性能的同时维持通用视觉-语言推理能力。
</figcaption>
</div>

**3. 慢速更新记忆上下文 (Slow-Updating Memory Context)**

在有限的上下文长度内平衡时间分辨率和细粒度空间感知仍然是 Video-LLM 的关键挑战。StreamVLN 不在特征层面压缩视频 token（如通过平均池化），而是保留高图像分辨率的同时选择性地丢弃空间和时间冗余 token，以更好地保持 Video-LLM 的可迁移性。

- **时间采样**: 采用简单的固定数量采样策略，避免不同长度的记忆 token 引入时间持续偏差
- **体素化空间剪枝 (Voxel-based Spatial Pruning)**: 使用深度信息将视频流中的 2D 图像patches 反投影到共享 3D 空间，离散化为均匀体素。通过跟踪 patch token 在时间上的体素索引，如果给定时长内的多个 token 投影到同一体素，仅保留最新观测的 token。该剪枝掩码用于选择保留的 token 状态（详见 Algorithm 1）。

**4. 多源数据联合训练**

- **Vision-Language Action (VLA) 数据**:
  - 使用 Habitat 模拟器收集 450K 样本（来自 60 个 Matterport3D 环境的 R2R、R2R-EnvDrop 和 RxR 数据集）
  - 额外 300K 样本来自 ScaleVLN（涵盖 700 个 HM3D 场景）以提高场景多样性
  - 采用 DAgger 算法收集 240K 纠正示范样本以增强鲁棒性和错误恢复能力

- **通用Vision-Language数据**: 为保持预训练 Video-LLM 的通用推理能力，引入：
  - 248K 视频基础 VQA 样本（来自 LLaVA-Video-178K 和 ScanQA）
  - 230K 交错图像-文本样本（来自 MMC4）以增强多轮视觉-语言交互能力

**主要创新点**：
- 首次提出针对实时 VLN 的慢-快上下文建模策略
- 设计了基于 3D 几何的智能 token 剪枝方法，优于通用的均匀剪枝
- 实现了低延迟、可扩展的流式多模态推理框架，支持 KV cache 高效复用
- 通过交错 vision-language-action 建模支持连贯的多轮对话
- 有界的上下文大小和推理成本，适合长视频流处理

**核心结果/发现**

<div align="center">
  <img src="/images/StreamVLN-visual-reasoning-transfer.png" width="100%" />
<figcaption>
StreamVLN 的视觉推理能力迁移：模型能够通过 VQA 对话正确识别画面内容（如蒙娜丽莎画像），并将这种推理能力迁移到理解导航指令中，展示了强大的跨模态理解能力。
</figcaption>
</div>

- **VLN-CE 基准测试上取得 state-of-the-art 性能**：
  - R2R Val-Unseen: SR 56.9%, SPL 51.9%（无额外数据）
  - RxR Val-Unseen: SR 52.9%, SPL 46.0%, nDTW 61.9%
  - 性能与 ETPNav 相当，但不依赖全景视图或航点监督

- **ScanQA 3D 问答基准测试**：超越 NaVILA 和 NaviLLM，Exact Match达到 28.8%

- **真实世界部署验证**：
  - 在 Unitree Go2 机器狗上成功部署
  - 平均推理延迟 0.27s（4个动作）+ 通信延迟 0.2s（室内）/ 1.0s（室外）
  - 支持实时物理部署

<div align="center">
  <img src="/images/StreamVLN-real-world-qualitative.png" width="100%" />
<figcaption>
StreamVLN 在多个真实世界环境中的定性结果（从上到下：Home、Workspace、Mall、Outdoor）。模型能够准确遵循包含多个地标的复杂指令，并处理真实世界中的干扰和变化。
</figcaption>
</div>

- **关键消融实验发现**：
  - KV cache 复用在多轮对话中消除超过 99% 的 prefilling 时间
  - 滑动窗口大小为 8 个对话轮次时实现最佳平衡
  - 记忆上下文大小从 2×196 增加到 8×196 tokens 时，SR 从 37.3% 提升到 45.5%
  - 体素化空间剪枝减少约 20% 的输入 token，同时提升性能（R2R +1.2% SR，RxR +1.1% SR）
  - DAgger 数据对性能提升至关重要（+5.5% SR / +3.8% SPL）
  - 通用 VL 数据（VideoQA + MMC4）的联合训练带来显著增益（+7.3% SR / +5.6% SPL）

<div align="center">
  <img src="/images/StreamVLN-KV-cache-latency.png" width="70%" />
<figcaption>
KV cache 复用对多轮对话解码延迟的影响：全轮次 KV cache 保持最低延迟；滑动窗口 KV cache 在窗口切换时有轻微延迟增加；单轮 KV cache（先前工作）的延迟随轮次线性增长。
</figcaption>
</div>

**局限性**

1. 直接从原始视觉观测生成低级动作对视点和遮挡变化的鲁棒性较弱，在真实世界环境中可能导致次优控制
2. 当前的混合上下文建模策略在更长视野的导航场景中仍然面临挑战，保持扩展序列上的一致推理较为困难
3. 依赖显式动作历史作为对话上下文的一部分，为异步推理和部署带来额外复杂性，需要同步过去的动作以保持对话连贯性


---


## 20. NavFoM (2025)
——Embodied Navigation Foundation Model

📄 **Paper**: https://arxiv.org/abs/2509.12129

**精华**

这篇论文展示了如何构建跨任务、跨具身体的导航基础模型,值得借鉴的核心思想包括:(1) 引入 Temporal-Viewpoint Indicator (TVI) tokens 来统一编码不同相机配置和时间信息,使模型能够处理多视角输入;(2) 提出 Budget-Aware Temporal Sampling (BATS) 策略,通过遗忘曲线动态采样历史帧,平衡性能和推理速度;(3) 在 8.02M 导航样本(包括四足机器人、无人机、轮式机器人、汽车等多种具身体)上联合训练,展示了大规模多任务训练对泛化能力的提升;(4) 采用视觉特征缓存机制加速训练 2.9 倍;(5) 证明了无需针对特定任务微调即可在多个基准测试上达到 SOTA 或竞争性能。

**研究背景/问题**

当前导航系统主要聚焦于特定任务设定和具身体架构,缺乏跨任务和跨具身体的泛化能力。现有 VLM 虽然在零样本任务上表现出色,但导航任务仍然局限于狭窄的任务领域、固定的相机配置和特定的具身体平台。本文旨在构建一个统一的导航基础模型,能够处理来自不同具身体(四足机器人、无人机、轮式机器人、汽车)的多视角输入,并跨越多个导航任务(VLN、目标搜索、目标追踪、自动驾驶)。

**主要方法/创新点**

<div align="center">
  <img src="/images/NavFoM-pipeline-overview.png" width="100%" />
<figcaption>
NavFoM 整体架构:统一框架处理 Image QA、Video QA 和导航任务
</figcaption>
</div>

NavFoM 基于 Vision-Language Model 架构,扩展为双分支系统:一个用于导航,一个用于问答。核心创新包括:

**1. Temporal-Viewpoint Indicator (TVI) Tokens**
- 引入特殊 indicator tokens 来编码相机视角和时间信息,每个 TVI token 由三部分组成:
  - **可学习的 base embedding** (E_Base)
  - **时间编码** (Time PE): 使用正弦位置编码标识帧的时间顺序
  - **视角编码** (Angle PE): 使用正弦/余弦编码保持方位角的循环连续性
- 对于导航任务,使用: E_TVI = E_Base + Time PE + Angle PE
- 对于 Video QA,仅使用时间信息;对于 Image QA,仅使用 base embedding
- TVI tokens 使 LLM 能够区分不同时间步和不同视角的 tokens,实现多视角导航

**2. Budget-Aware Temporal Sampling (BATS)**
- 解决在线导航时视觉 tokens 数量激增的问题
- 基于遗忘曲线(exponential decay)的采样概率: P(t) = (1 - ε)e^(k(t-T)/T) + ε
- 动态调整历史帧采样,越近的帧采样概率越高
- 在 token budget 约束下,平衡短期上下文和长期历史信息
- 相比 Uniform Sampling,BATS 在保持性能的同时显著降低推理时间

**3. 观测编码**
- 使用预训练 vision encoders (DINOv2, SigLIP) 提取视觉特征
- 采用 Grid Average Pooling 策略生成两种分辨率的视觉 tokens:
  - **Fine-grained** (64×C): 用于当前最新观测和 Image QA
  - **Coarse-grained** (4×C): 用于导航历史和 Video QA
- 通过 cross-modality projector 将视觉特征映射到 LLM latent space

**4. Token 组织策略**
- 不同任务采用不同的 token 组织方式:
  - **Image QA**: fine-grained visual tokens + base TVI embedding
  - **Video QA**: coarse-grained visual tokens + base + time embedding
  - **Navigation**: coarse-grained + fine-grained tokens + base + time + angle embedding
- 这种设计实现了导航和 QA 数据的联合训练

**5. 轨迹预测**
- 使用三层 MLP 作为 planning head 从 LLM 隐藏状态预测轨迹
- 轨迹归一化到 [-1, 1] 分布,针对不同具身体(室内导航 vs 户外驾驶)采用不同的 scaling factor
- 对于室内机器人,预测 8 个航点;对于汽车和无人机,预测更长的轨迹

**6. 数据规模与来源**
- **导航数据** (8.02M): VLN-CE R2R/RxR (2.94M), OpenUAV (429K), 目标导航 (1.02M), 主动视觉追踪 (897K), 自动驾驶 (681K), Web 导航伪标签 (2.03M)
- **QA 数据** (4.76M): Image QA (3.15M) + Video QA (1.61M)
- 总计 12.7M 训练样本,覆盖四足机器人、无人机、轮式机器人、汽车等多种具身体

**7. 训练优化**
- 视觉特征缓存: 预先计算并缓存 coarse-grained visual tokens,训练加速 2.9 倍,GPU 内存减少 1.8 倍
- 使用 Qwen2-7B 作为 LLM backbone
- 单次训练所有参数(仅 designated trainable parameters),无需多阶段训练

**核心结果/发现**

**VLN 性能**:
- **VLN-CE R2R** (single-view): SR 5.01% → 64.9%, SPL 56.2%,无需任务特定微调即达到 SOTA
- **VLN-CE RxR** (four-view): SR 5.51% → 57.4%, SPL 49.4%,超越所有基线方法
- **OpenUAV** (四视角,UM split): SR 6.38% → 14.05%, OSRL 5.68% → 18.65%,显著优于 TravelUAV

**目标搜索**:
- **HM3D-OVON** (zero-shot): VAL SEEN SR 55.0%, VAL UNSEEN SR 45.2%,超越 MTU3D baseline

**主动视觉追踪**:
- **EVT-Bench** (four-view, zero-shot): Single Target SR 85.1%/TR 80.5%, Distracted Target SR 62.0%/TR 67.9%

**自动驾驶**:
- **NAVSIM** (eight-view): PDMS 84.3%, 与 SOTA 方法竞争性能
- **nuScenes** (six-view): CR 93%, 接近 SOTA

**Ablation 研究**:
- 多任务训练带来显著增益:联合训练使 VLN SR 从 57.3% 提升到 64.4%
- 相机数量对性能的影响:从单视角到四视角,SR 从 58.3% 提升到 65.8%,但增加到六视角略有下降
- BATS 相比 Uniform Sampling,在 RxR 上 nDTW 仅下降 1.4%,但保持稳定推理速度
- TVI tokens 相比其他替代方案(learned special tokens, handcraft tokens)显著提升性能

**实际部署**:
- 在 110 个真实世界测试场景(50 VLN + 30 搜索 + 30 追踪)中验证,成功率达到 72%~93%
- 支持跨具身体部署:四足机器人(Unitree Go2)、类人机器人、无人机、轮式机器人
- 0.5 秒内生成 8 航点轨迹(1600 token budget)

**局限性**

该方法在训练时需要大量计算资源(56 NVIDIA H100 GPUs,72 小时)。尽管引入了视觉特征缓存等优化策略,大规模训练仍然是资源密集型任务。此外,在需要遍历 300 米复杂邻域的 Unseen-Map 场景中表现较差,表明模型在大规模环境探索和长距离规划方面仍有改进空间。作者也指出 NavFoM 只是一个起点,未来需要更高质量的数据、更先进的技术以及新一代基准测试来推动泛化导航研究的发展。


---
## 21. DGNav (2026)
———动态拓扑感知：打破视觉-语言导航中的粒度刚性

📄 **Paper**: https://arxiv.org/abs/2601.21751

### 精华

这篇论文解决了 VLN-CE 中的"粒度刚性"问题，值得借鉴的核心思想：

1. **自适应结构调整**：不仅调整模型参数，还动态调整数据结构本身（拓扑图的节点密度），实现"简单场景保效率、复杂场景保安全"的自适应平衡，这一思路可迁移到其他需要精度/效率权衡的规划任务（如 SLAM、点云处理）。
2. **条件干预设计**：引入"稳定性门槛"（中位数离散度 σ_med），只在高不确定性场景触发动态调整，而非全局自适应，有效避免了在简单场景引入不必要的噪声——这是一种极具工程实用性的设计思想。
3. **多模态软硬约束融合**：将几何硬约束（物理可达性）与视觉语义和语言指令软约束通过可学习权重动态融合，使图连接从"物理近邻关系"升级为"语义近邻关系"，为多约束优化提供了优雅解法。
4. **线性映射的理论优越性**：基于信息论论证了线性映射是保持最大熵属性的最优一阶近似，既优于 Sigmoid 的梯度饱和，又优于 Exponential 的保守偏差——理论驱动设计的典范。
5. **结构与训练解耦**：Scene-Aware Adaptive Strategy 仅在推理阶段激活，训练阶段使用固定阈值，实现了稳定的特征学习与灵活的测试时推理之间的解耦。

---

### 1. 研究背景/问题

VLN-CE（连续环境中的视觉-语言导航）中，现有拓扑规划方法（如 ETPNav）依赖固定的图构建阈值 γ 和静态欧式距离边权重，导致"粒度刚性"问题：在简单低不确定性区域产生大量冗余节点，在复杂高不确定性区域图过于稀疏导致导航失败。更严重的是，纯几何边权重使智能体优先连接物理距离近但语义无关的节点（"导航性近视" Navigational Myopia），无法遵从指令中的语义意图。

**主要方法/创新点**

论文提出 **DGNav (Dynamic Graph Navigation)** 框架，包含两大核心模块：

<div align="center">
  <img src="/images/DGNav-overall-framework.png" width="100%" />
<!-- RENAME: figure_01.png -> DGNav-overall-framework.png -->
<figcaption>
DGNav 整体框架。根据估计的场景复杂度 σ 动态调整导航策略：高复杂度场景构建更密集的拓扑图，简单环境则采用更稀疏的表示。图合并阈值 γ 控制图粒度，与 σ 呈反相关，实现导航安全性与效率的自适应权衡。
</figcaption>
</div>

**1. 场景感知自适应策略 (Scene-Aware Adaptive Strategy)**

针对物理结构层面的粒度刚性问题，提出动态调整图构建阈值的方法：

- **场景复杂度度量**：通过分析预测路径点的角度离散度 (angular dispersion) σ 来量化局部场景复杂度：
  ```
  σ_t = sqrt(1/N_c * Σ(θ_i - θ̄)²)
  ```
  其中 θ_i 是候选节点相对于智能体朝向的角度。高 σ 表示复杂决策边界（如交叉路口），低 σ 表示简单几何结构（如走廊）。

- **条件线性映射控制律**：基于统计校准的高斯分布特性，采用线性映射动态调整合并阈值 γ：
  ```
  γ_t = γ_fix                                        if σ_t ≤ σ_med
  γ_t = γ_fix - (σ_t - σ_med)/(σ_max - σ_med) * (γ_fix - γ_min)   if σ_t > σ_med
  ```
  
<div align="center">
  <img src="/images/DGNav-adaptive-strategy.png" width="100%" />
<!-- RENAME: figure_02.png -> DGNav-adaptive-strategy.png -->
<figcaption>
场景感知自适应策略示意图。从深度图生成候选路径点后，根据候选节点的角度离散度 (σ) 动态调整合并阈值 γ。在简单环境中 (低 σ)，较大的 γ 产生稀疏图以提升效率；在复杂环境中 (高 σ)，较小的 γ 产生密集图以确保安全。
</figcaption>
</div>

- **理论依据**：选择线性映射而非 Sigmoid/指数映射的原因是线性变换保持高斯源分布的最大熵特性。非线性映射会在分布尾部引入饱和区域（梯度消失），导致高不确定状态的信息丢失。条件映射策略仅在 σ > σ_med 时激活自适应机制，在稳定场景中保持拓扑稳定性。

**2. 动态图 Transformer (Dynamic Graph Transformer)**

针对语义逻辑层面的导航近视问题，融合多模态线索动态重构图连接性：

<div align="center">
  <img src="/images/DGNav-dynamic-edge-fusion.png" width="100%" />
<!-- RENAME: figure_03.png -> DGNav-dynamic-edge-fusion.png -->
<figcaption>
多模态编码和动态边融合架构。视觉编码器和指令编码器分别提取节点特征 (V) 和词特征 (W)。动态边融合模块通过融合几何地图 (E_geo)、成对视觉相似度 (E_sem) 和指令相关性 (E_inst) 构建图连接性。生成的动态邻接矩阵 E_dynamic 指导 Graph Transformer 执行上下文感知的路径规划。
</figcaption>
</div>

#### 1.Scene-Aware Adaptive Strategy（场景感知自适应策略）

通过计算当前时刻候选节点的**角度离散度** $\sigma_t$ 来量化场景复杂度：

$$\sigma_t = \sqrt{\frac{1}{N_c} \sum_{i=1}^{N_c} (\theta_i - \bar{\theta})^2}$$

基于 $\sigma_t$，采用**条件线性映射**动态调整图合并阈值 $\gamma_t$：

$$\gamma_t = \begin{cases} \gamma_{fix} & \text{if } \sigma_t \leq \sigma_{med} \\ \gamma_{fix} - \dfrac{\sigma_t - \sigma_{med}}{\sigma_{max} - \sigma_{med}}(\gamma_{fix} - \gamma_{min}) & \text{if } \sigma_t > \sigma_{med} \end{cases}$$

- 简单场景（$\sigma_t \leq \sigma_{med}$）：$\gamma_t = \gamma_{fix} = 0.5\text{m}$，保持稀疏效率
- 复杂场景（$\sigma_t > \sigma_{med}$）：线性降低 $\gamma_t$（最低至 $\gamma_{min} = 0.1\text{m}$），生成密集拓扑

$\sigma_{med}$ 和 $\sigma_{max}$ 通过在 ETPNav 基线模型上统计推断得到（数据驱动校准），线性函数的选择基于信息论证明其是保最大熵的最优一阶近似。

#### 2.Dynamic Graph Transformer（动态图 Transformer）

**Dynamic Edge Fusion**：融合三种信息流构造动态邻接矩阵：

$$\mathbf{E}_{dynamic} = \mathbf{E}_{geo} + \omega_1 \cdot \mathbf{E}_{sem} + \omega_2 \cdot \mathbf{E}_{inst}$$

- $\mathbf{E}_{geo}$：归一化欧式距离（物理可达性硬约束）
- $\mathbf{E}_{sem}$：CLIP-ViT 提取的视觉特征通过 MLP 计算的成对相似度
- $\mathbf{E}_{inst}$：节点特征与全局指令 token $\mathbf{W}_L$ 的外积相关性分数，即 $w_i = \text{MLP}([v_i; \mathbf{W}_L])$，$E_{inst}^{(i,j)} = w_i \cdot w_j$

**Graph-Aware Self-Attention (GASA)**：

$$\text{GASA}(\mathbf{H}^l, \mathbf{E}_{dynamic}) = \text{Softmax}\!\left(\frac{(\mathbf{H}^l \mathbf{W}_Q)(\mathbf{H}^l \mathbf{W}_K)^\top}{\sqrt{d_k}} + \mathbf{E}_{dynamic}\right)\!(\mathbf{H}^l \mathbf{W}_V)$$

将 $\mathbf{E}_{dynamic}$ 直接叠加到注意力分数上，强制模型关注语义相关（$\omega_1 \cdot \mathbf{E}_{sem}$）且指令对齐（$\omega_2 \cdot \mathbf{E}_{inst}$）的节点，同时 $\mathbf{E}_{geo}$ 保证物理约束不被完全忽略，实现从纯几何到语义驱动的平滑过渡。

**训练策略**：采用两阶段训练，Adaptive Strategy 仅在推理阶段激活，训练阶段固定 $\gamma = 0.5\text{m}$ 确保稳定的特征学习。

---

### 3. 核心结果/发现

**R2R-CE 数据集**：
- Val-Unseen：SR **64.82%**，SPL **50.08%**，超越 ETPNav 基线（+4.66% SR，+2.21% SPL）
- Test-Unseen：SR 64%（+1% vs ETPNav），SPL 47%，NE 下降 0.2m
- 超越所有 End-to-End 方法和显式地图方法（含 GridMM, Safe-VLN, OVL-MAP）

**RxR-CE 数据集**（多语言，更长路径）：
- Val-Unseen：SR **53.78%**，nDTW **62.04%**（+0.55%），SDTW **44.49%**（+0.57%）
- 路径保真度指标全面超越 ETPNav，证明在长时域细粒度指令遵从上的优越性

**消融实验关键发现**：
- 条件线性映射 vs 全局线性映射：SR +1.52%（稳定性门槛机制的贡献）
- 动态 $\gamma$ vs 固定 $\gamma$（0.25/0.40/0.50m）：SR 最高提升 +1.63%，且计算开销仅增加 0.4 个节点
- 完整 $\mathbf{E}_{dynamic}$ vs 仅几何：SR 大幅提升，验证语义软约束的关键作用
- 定性分析（Fig.9）：在"绕过木质围栏"场景中，仅几何模型因物理距离过近而提前错误转向，DGNav 正确识别指令语义并忽略了几何干扰，成功到达目标

---

### 4. 局限性

自适应策略的核心参数（$\gamma_{fix}, \gamma_{min}, \sigma_{med}, \sigma_{max}$）通过在 R2R-CE 训练集上进行统计校准获得，在分布外场景（如户外环境、高度动态场景）中的泛化能力尚未验证；同时，随着导航轨迹增长，拓扑图规模持续膨胀，论文未讨论图压缩和历史节点管理策略，在超长路径任务中可能面临内存和计算的挑战。

---
## 22. MapNav (2025)
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


---

# 基石论文

---
## 1. Diffusion Policy (2023)
——通过动作扩散实现视觉运动策略学习

📄 **Paper**: https://arxiv.org/abs/2303.04137

**精华**

这篇论文展示了如何将扩散模型应用于机器人策略学习,值得借鉴的核心思想包括:1)利用扩散模型的梯度场表示能力来优雅地处理多模态动作分布;2)通过预测高维动作序列而非单步动作来确保时间一致性;3)将 receding horizon control 与扩散模型结合实现闭环规划;4)位置控制相比速度控制在扩散策略中表现更优;5)训练稳定性显著优于基于能量的隐式策略。

**研究背景/问题**

现有的机器人模仿学习方法在处理多模态动作分布、高维动作空间和训练稳定性方面面临挑战。显式策略(如 LSTM-GMM)难以表达复杂的多模态分布,而隐式策略(如 IBC)虽然理论上可以建模任意分布,但由于需要负采样估计归一化常数,导致训练不稳定。

<div align="center">
  <img src="/images/DiffusionPolicy-policy-representations.png" width="100%" />
<figcaption>
三种策略表示方法对比:显式策略、隐式策略和 Diffusion Policy
</figcaption>
</div>

**主要方法/创新点**

Diffusion Policy 将机器人的视觉运动策略表示为条件去噪扩散过程。核心思想是通过迭代去噪过程从噪声中生成动作序列,而不是直接预测动作。

**核心技术贡献:**

1. **闭环动作序列预测**: 结合 receding horizon control,策略预测 Tp 步未来动作,但只执行 Ta 步,之后重新规划。这在长期规划和响应性之间取得平衡。

2. **视觉条件化**: 将视觉观察作为条件而非联合分布的一部分,使得视觉特征只需提取一次,显著降低计算成本,实现实时控制。

3. **时间序列扩散 Transformer**: 提出基于 Transformer 的扩散网络,减少 CNN 模型的过度平滑效应,在需要高频动作变化和速度控制的任务上达到最优性能。

4. **位置控制的协同效应**: 发现位置控制相比速度控制更适合扩散策略,因为位置空间的多模态性更明显,且扩散策略能更好地利用位置控制的优势。

**训练过程:**
- 随机采样演示数据 A⁰ₜ 和去噪迭代 k
- 添加噪声 εₖ 得到 A⁰ₜ + εₖ
- 训练噪声预测网络 εθ 最小化 MSE(εₖ, εθ(Oₜ, A⁰ₜ + εₖ, k))

**推理过程:**
- 从高斯噪声 Aᴷₜ 开始
- 迭代 K 次去噪: Aᵏ⁻¹ₜ = α(Aᵏₜ - γεθ(Oₜ, Aᵏₜ, k) + N(0, σ²I))
- 使用 DDIM 加速推理,将迭代次数从 100 降至 10-16

<div align="center">
  <img src="/images/DiffusionPolicy-overview.png" width="100%" />
<figcaption>
Diffusion Policy
</figcaption>
</div>


**核心结果/发现**

论文在 4 个基准测试的 15 个任务上进行了系统评估,包括模拟和真实环境:

1. **显著性能提升**: 相比现有最优方法平均提升 46.9% 的成功率
   - RoboMimic 基准:在所有 9 个变体上均显著优于 LSTM-GMM、IBC 和 BET
   - Push-T 任务:95% 成功率(IBC 0%, LSTM-GMM 0-20%)
   - Kitchen 环境:p4 指标提升 213%

2. **多模态建模能力**: 能够准确表达短期和长期多模态行为,并在执行时稳定地选择单一模式

3. **训练稳定性**: 无需负采样,训练过程稳定,评估性能曲线平滑,易于选择检查点

4. **真实世界验证**: 在 7 个真实机器人任务上验证,包括:
   - 6 自由度倒酱汁和涂抹任务(接近人类水平)
   - 双臂协作任务(打蛋器、展开垫子、叠衬衫)

5. **延迟鲁棒性**: 使用位置控制的 Diffusion Policy 对最多 4 步延迟具有鲁棒性

**局限性**

继承了行为克隆的局限性,在演示数据不足时性能次优。推理延迟高于简单方法(如 LSTM-GMM),尽管动作序列预测部分缓解了这一问题。未来工作可以探索将扩散策略应用于强化学习,以及利用最新的扩散模型加速技术(如一致性模型)进一步降低推理迭代次数。

---
## 2. Lumina-Next (2024)
——通过 Next-DiT 让 Lumina-T2X 更强更快

📄 **Paper**: https://arxiv.org/abs/2406.xxxxx (NeurIPS 2024)

**精华**

这篇论文展示了如何通过架构优化和推理技术改进 diffusion transformer 的关键方法,值得借鉴的核心思想包括:(1) 使用 3D RoPE 和 sandwich normalization 控制网络激活量级,实现更稳定的训练和更好的分辨率外推能力;(2) 提出 Frequency- 和 Time-Aware Scaled RoPE,针对 diffusion 模型的时间感知特性设计位置编码外推策略;(3) 通过优化时间离散化调度和 Time-Aware Context Drop 显著提升推理效率;(4) 采用 flow matching 框架和 transformer 架构构建统一的多模态生成框架;(5) 使用更强大的 decoder-only LLM 作为文本编码器实现零样本多语言生成能力。

**研究背景/问题**

Lumina-T2X 作为基于 Flow-based Large Diffusion Transformer (Flag-DiT) 的生成模型家族,虽然在多模态生成方面展现出潜力,但仍然面临训练不稳定、推理速度慢和分辨率外推时出现伪影等挑战。现有的 diffusion transformer 架构在扩展到大规模模型和长序列时存在激活量级难以控制的问题,且缺乏针对视觉生成任务优化的位置编码外推策略。

**主要方法/创新点**

<div align="center">
  <img src="/images/Lumina-Next-architecture-comparison.png" width="100%" />
<figcaption>
Flag-DiT 和 Next-DiT 的架构对比,展示了主要改进点
</figcaption>
</div>

**Next-DiT 架构改进:**

1. **3D RoPE 替代 1D RoPE**: 原始 Flag-DiT 使用 1D RoPE 将图像 token 序列化处理,丢失了空间和时间关系。Next-DiT 采用 3D RoPE,将嵌入维度分为三个独立部分,分别编码 x、y、z 轴的位置信息,提供统一且精确的时空位置表示,无需额外的可学习标识符如 [nextline] 和 [nextframe]。

2. **Sandwich Normalization**: 通过在每个 attention 和 MLP 层前后都添加 RMSNorm,有效控制网络激活量级的增长。实验表明,Flag-DiT 中存在的长信号路径会导致激活值在深层累积,特别是在分辨率外推时,早期层的小误差会在后续层被放大。Sandwich normalization 配合 tanh gating 机制,稳定了训练和采样过程。

<!-- <div align="center">
  <img src="/images/Lumina-Next-rope-comparison.png" width="100%" />
<figcaption>
1D RoPE 与 2D RoPE 的注意力模式对比,以及不同外推策略在 2K 生成中的效果
</figcaption>
</div> -->

3. **Grouped-Query Attention**: 采用 GQA 将 32 个 query heads 分成 8 组,共享 8 个 key-value heads,在保持性能的同时降低参数量和计算复杂度,特别适合高分辨率图像生成。

**Frequency- 和 Time-Aware Scaled RoPE:**

针对 3D RoPE 的分辨率外推,论文系统比较了现有方法并提出创新策略:

1. **Frequency-Aware Scaled RoPE**: 识别波长等于训练序列长度的维度 d_target,通过 b' = b · s^(d_head/d_target) 调整频率基,使该维度等效于位置插值,有效减少内容重复问题。

2. **Time-Aware Scaled RoPE**: 考虑 diffusion 模型先生成全局概念再生成局部细节的特性,设计时间相关系数 d_t = (d_head - 1)·t + 1,在去噪早期使用位置插值确保全局结构,在后期逐渐转向 NTK-Aware Scaled RoPE 保留局部细节。

<!-- <div align="center">
  <img src="/images/Lumina-Next-convergence.png" width="100%" />
<figcaption>
Next-DiT 在 ImageNet-256 基准上的收敛速度对比
</figcaption>
</div> -->

**优化采样效率:**

1. **Sigmoid 时间调度**: 分析发现 Flow ODE 的离散化误差在纯噪声(t=0)附近最大,在接近干净数据(t=1)时相对较大。提出 sigmoid 分段函数确保采样开始和结束阶段的步长大于中间步骤,配合高阶 ODE solver(如 midpoint method)实现 5-10 步高质量生成。

2. **Time-Aware Context Drop**: 对 keys 和 values 进行空间平均池化合并相似 token,减少注意力计算冗余。与 Token Merging 不同,该方法只下采样 KV 而保留完整的视觉内容,并且引入时间感知机制,在 t=0 时最大程度 drop token 提升效率,在 t=1 时不 drop 以保持视觉质量,在 1K 分辨率生成中实现 2× 推理加速。

<!-- <div align="center">
  <img src="/images/Lumina-Next-resolution-extrapolation.png" width="100%" />
<figcaption>
4× 分辨率外推对比,Lumina-Next 相比其他方法展现出更好的全局一致性和局部细节
</figcaption>
</div> -->

**统一多模态框架:**

论文将 Lumina-Next 扩展到多种模态:
- **多视图生成**: 引入相对姿态控制和图像条件,支持任意视图数量的灵活推理
- **音频/音乐生成**: 使用 1D 卷积 VAE 编码 mel-spectrogram,配合 CLAP+FLAN-T5 双文本编码器
- **点云生成**: 提出 Time-aware Scaled Fourier feature,实现密度不变的点云生成,可以在 256 点训练后生成 8192 点的高密度点云
- **任意分辨率识别**: 通过 dynamic partitioning 和 masked attention 实现图像识别任务的分辨率外推

**训练数据改进:**

采用 Mixture-of-Captioners (MoC) 策略,使用多个预训练 VLM(BLIP2, LLaVA, SPHINX, Share-GPT4V)生成互补的图像描述,并使用 GPT-4V 对 10 万张高分辨率图像进行多方面整体描述。训练时从 caption pool 随机采样,增强模型对不同风格用户输入的鲁棒性。数据集从 14M 扩展到 20M 高质量图像-文本对。

**核心结果/发现**

1. **架构性能**: Next-DiT 在 ImageNet-256 上训练 100 epochs 即达到 81.6% Top-1 准确率,接近 DeiT-base 训练 300 epochs 的结果。训练 300 epochs 后达到 82.3%,显著超越 DeiT-base 的 81.6%。

2. **文生图质量**: 2B 参数的 Next-DiT 配合 Gemma-2B 文本编码器,在生成质量上超越了 5B Flag-DiT 配合 LLaMA-7B 的 Lumina-T2X,同时显著降低训练和推理成本。

3. **分辨率外推**: 支持训练时未见过的任意分辨率和宽高比生成,在 2K 和全景图生成任务上展现出全局一致性和丰富的局部细节,优于 MultiDiffusion、DemoFusion 和 ScaleCrafter 等方法。

4. **少步采样**: 使用 midpoint solver 配合 sigmoid schedule,在 10-20 个函数评估步数内生成高质量图像,性能持续优于使用 DPM-Solver 的 PixArt-α 和 SDXL。

5. **多语言能力**: 使用 decoder-only LLM 作为文本编码器,实现零样本多语言文生图能力,在 15 种语言测试中展现出比 SDXL 和 PixArt-α 更好的文本理解和文化细微差别捕捉能力。

6. **推理效率**: Time-Aware Context Drop 在 1K 分辨率下实现 2× 加速,且在配合 Flash Attention 时对超高分辨率图像生成的加速更显著。

7. **多模态扩展**:
   - 音乐生成: FAD 3.75, MOS-Q 83.56, MOS-F 85.69,优于 MusicLDM 和 AudioLDM 2
   - 音频生成: FAD 1.03, MOS-Q 77.53, MOS-F 76.52,优于 Make-An-Audio 2 和 AudioLDM 2
   - 多视图生成: 512×512 分辨率支持 1-8 视图灵活推理
   - 点云生成: 在 airplane 和 chair 类别上达到与 PDiffusion 相当的 MMD 和 COV 指标

8. **任意分辨率识别**: 经过 any-resolution fine-tuning 后,在 ImageNet-1K 上达到 84.2% Top-1 准确率,在 1024×1024 分辨率上显著优于 DeiT-base。

**局限性**

尽管 Lumina-Next 在多个方面取得显著改进,但在文本-图像对齐和视觉美学方面仍落后于 Midjourney 和 DALLE 3 等闭源模型。主要差距在于多阶段训练使用的文本-图像对数量:论文将数据集扩展到 2000 万,但仍远小于闭源模型使用的数据规模。此外,使用人类偏好数据进行微调(如 Direct Preference Optimization)对提升图像质量也很重要,这是未来研究的方向。

---

## 3. Qwen3-VL (2025)
——最强视觉-语言模型系列,原生支持256K上下文

📄 **Paper**: https://arxiv.org/abs/2511.21631v2

**精华**

这篇论文展示了如何构建一个全面的视觉-语言模型系列,值得借鉴的核心思想包括:
1. **平衡文本和多模态能力**:通过square-root reweighting确保多模态训练不损害文本能力,甚至在某些文本任务上超越纯文本模型
2. **渐进式上下文扩展**:采用四阶段预训练(8K→32K→256K),逐步扩展上下文窗口,而不是一步到位
3. **架构优化的实用主义**:Interleaved MRoPE、DeepStack、文本时间戳等创新都针对实际问题(长视频理解、视觉-语言对齐、时序定位)
4. **分层式后训练**:区分non-thinking和thinking变体,针对不同应用场景优化
5. **全栈式能力整合**:将感知(grounding)、推理(reasoning)和行动(agentic)能力统一到单一模型框架中

**研究背景/问题**

现有的视觉-语言模型在发展过程中面临几个关键挑战:一是多模态训练往往会损害底层LLM的语言能力;二是长上下文支持不足,难以处理长文档和长视频;三是在STEM推理、文档理解、视频理解等专业任务上性能参差不齐;四是缺乏统一的框架整合感知、推理和决策能力。Qwen3-VL旨在系统性地解决这些问题。

**主要方法/创新点**

<div align="center">
  <img src="/images/Qwen3-VL-architecture.png" width="100%" />
<figcaption>
Qwen3-VL整体架构:集成视觉编码器和语言模型解码器处理文本、图像和视频等多模态输入。视觉编码器支持动态原生分辨率,通过DeepStack机制将多层视觉特征注入到LLM的对应层中。采用Interleaved MRoPE编码位置信息,并引入文本时间戳标记捕获视频的时序结构
</figcaption>
</div>

Qwen3-VL提出了一个完整的视觉-语言模型系列,包括4个dense模型(2B/4B/8B/32B)和2个MoE模型(30B-A3B/235B-A22B),均原生支持256K token的交错式上下文。

**架构创新**:

1. **Interleaved MRoPE** - 针对Qwen2.5-VL中MRoPE频谱不平衡的问题,将时间(t)、水平(h)、垂直(w)维度交错分布在低频和高频频段,显著改善长视频理解能力

2. **DeepStack跨层融合** - 从ViT的多个层提取视觉特征,通过轻量级残差连接路由到LLM的对应层,增强多层次视觉-语言对齐,不增加额外上下文长度

3. **显式视频时间戳** - 用文本token(如`<3.0 seconds>`)标记视频帧组,替代Qwen2.5-VL中的绝对时间位置编码,提供更简单直接的时序表示,支持seconds和HMS两种格式

**训练策略**:

**预训练**分为四个阶段:
- S0 (67B tokens, 8K): 仅训练merger层进行视觉-语言对齐
- S1 (~1T tokens, 8K): 全参数多模态预训练,混合VL数据和文本数据
- S2 (~1T tokens, 32K): 长上下文预训练,增加文本数据比例和视频/agent数据
- S3 (100B tokens, 256K): 超长上下文适应,聚焦长视频和长文档理解

**后训练**包含三个阶段:
1. SFT - 分为32K和256K两个阶段,提供non-thinking和thinking两个变体
2. Strong-to-Weak蒸馏 - 用text-only数据微调LLM backbone,显著提升推理能力
3. 强化学习 - 分为Reasoning RL(数学、代码、逻辑推理等)和General RL(指令遵循、格式控制等),使用SAPO算法

**数据优化**:

- **高质量caption** - 使用Qwen2.5-VL-32B对web图像重新标注,基于视觉embedding聚类增强稀疏概念覆盖
- **交错文本-图像** - 收集多模态文档,用domain classifier过滤低质量内容,构建256K长序列
- **知识数据** - 覆盖12+语义类别(动物、植物、地标等),采用importance-based采样平衡长尾分布
- **OCR扩展** - 从10种语言扩展到39种语言,合成3000万高质量样本
- **Grounding归一化** - 统一采用[0, 1000]归一化坐标系统,支持2D/3D grounding和counting
- **视频数据** - 密集caption合成(short-to-long策略)和时空grounding数据
- **STEM数据** - 6M图表caption + 60M+ K-12/本科习题 + 12M长CoT推理样本
- **Agent数据** - GUI感知(描述、grounding)+ 自进化轨迹生成框架

**优化技巧**:

- **Square-root reweighting** - 对per-token loss进行平方根归一化,平衡文本和多模态数据贡献
- **分层式训练** - 预训练阶段逐步扩展上下文,后训练阶段区分thinking/non-thinking模式

**核心结果/发现**

**综合性能**:
- 在多模态reasoning任务上(MMMU、MathVista、MathVision等),Qwen3-VL-235B-A22B-Thinking达到SOTA水平
- 在文本任务上超越或持平纯文本模型(如DeepSeek V3、Qwen3-235B),证明多模态训练未损害语言能力
- 小模型(2B/4B/8B)表现出色,8B模型在很多任务上接近Qwen2.5-VL-72B

**长上下文能力**:
- Needle-in-a-Haystack评估:256K token(30分钟视频)内100%准确率,外推到1M token(2小时视频)仍保持99.5%准确率
- MMLongBench-Doc: 57.0%准确率,SOTA表现

**领域专项**:
- **OCR/文档**: OCRBench 920分,支持39种语言,32/39语言准确率>70%
- **2D/3D Grounding**: RefCOCO 91.9%,ODinW-13 48.6 mAP,3D grounding在SUNRGBD上超越Gemini-2.5-Pro 5.2点
- **视频理解**: VideoMME 79.2%,MLVU 84.3%,在长视频理解上超越Gemini-2.5-Pro
- **GUI Agent**: ScreenSpot Pro 62.0%,OSWorld 38.1%,AndroidWorld 63.7%
- **Fine-grained Perception**: 使用工具后V* 93.7%,HRBench4K 85.4%
- **STEM推理**: MathVista 85.8%(thinking),MathVision 74.6%,MMMU 80.6%

**思考模式收益**:
- Thinking模式在推理密集型任务上带来显著提升(如AIME-25: 89.7% vs 74.7%, HMMT-25: 77.4% vs 57.4%)
- 在某些任务上instruct模式反而更好(如RealWorldQA),说明需要针对应用场景选择模式

**局限性**

论文未明确指出局限性,但从架构和实验设计可推断:训练成本较高(四阶段预训练+三阶段后训练),对于资源受限的场景可能难以复现;虽然支持256K上下文,但在超长序列(>256K)上仍需YaRN外推;thinking模式虽然提升推理能力,但会增加推理延迟和成本。

---
## 4. VLN-CE (2020)
——Beyond the Nav-Graph: 在连续环境中的视觉-语言导航

📄 **Paper**: https://arxiv.org/abs/2004.02857

**精华**

这篇论文通过将 VLN 任务从离散导航图迁移到连续 3D 环境,揭示了基于导航图的设定中隐含的强假设对性能的巨大影响。值得借鉴的核心思想包括:批判性地审视任务设定中的隐含假设、通过消除不现实的简化来提高任务的实际应用价值、深度信息在具身导航中的关键作用、以及端到端学习与低层控制结合的必要性。这种"去简化"的研究思路对构建更接近真实机器人应用的 AI 系统具有重要指导意义。

**研究背景/问题**

现有的 Vision-and-Language Navigation (VLN) 任务基于导航图 (nav-graph) 表示,引入了三个不现实的假设:已知环境拓扑、短距离 oracle 导航、以及完美的智能体定位。这些假设使得任务本质上退化为视觉引导的图搜索问题,与真实机器人导航场景存在巨大差距,限制了向实际机器人平台迁移的可能性。

**主要方法/创新点**

<div align="center">
  <img src="/images/VLN-CE-comparison.png" width="100%" />
<figcaption>
VLN 与 VLN-CE 的对比:VLN 基于固定拓扑的全景图节点(左),而 VLN-CE 在连续环境中使用低层动作(右)
</figcaption>
</div>

论文提出了 Vision-and-Language Navigation in Continuous Environments (VLN-CE) 任务,在 Habitat 模拟器中实例化连续的 Matterport3D 环境。主要创新包括:

1. **连续环境设定**:智能体通过低层动作(前进 0.25m、左转/右转 15°、停止)在连续 3D 空间中自由导航,而非在固定节点间传送。

2. **轨迹迁移方法**:设计了将 Room-to-Room (R2R) 数据集的导航图轨迹转换为连续环境路径的算法。通过向下投射射线找到最近的可导航点,并使用 A* 算法验证路径可达性,成功转换了 77% 的 R2R 轨迹(4475 条)。

3. **模型架构**:
   - **Seq2Seq Baseline**: 使用 GRU 处理 RGB 和 Depth 观察的均值池化特征以及 LSTM 编码的指令
   - **Cross-Modal Attention Model**: 采用双 GRU 架构,一个处理视觉观察,另一个基于注意力机制融合指令和视觉特征进行决策。使用预训练的 ResNet50 (ImageNet) 提取 RGB 特征,使用预训练的 ResNet50 (Point-Goal Navigation) 提取深度特征。

4. **训练策略**:
   - 基础模仿学习 with inflection weighting
   - DAgger 应对 exposure bias
   - Progress Monitor 辅助损失
   - Speaker 模型生成的合成数据增强(~150k 条轨迹)

**核心结果/发现**

1. **任务难度显著增加**:VLN-CE 中平均轨迹长度为 55.88 个动作,而 VLN 仅需 4-6 个节点跳转。最佳模型在 val-unseen 上达到 32% 成功率 (SR) 和 0.30 SPL,显著低于 VLN 中的表现。

2. **深度信息至关重要**:移除深度输入导致模型性能崩溃(成功率 ≤1%),而移除 RGB 或指令的影响相对较小。深度使智能体能够快速学会有效遍历环境(避免碰撞),是引导学习的关键信号。

3. **训练技术的混合效果**:Cross-Modal Attention 优于 Seq2Seq;DAgger 带来 3-5% SPL 提升;但 Progress Monitor 和数据增强单独使用时效果不佳,需要组合使用(预训练 + DAgger 微调)才能达到最佳性能。

4. **导航图的强先验**:将 VLN-CE 训练的智能体路径转换回导航图并在 VLN 测试集上评估,SPL 为 0.21,远低于利用导航图训练的 SOTA 方法(0.47 SPL)。这表明现有 VLN 结果可能因导航图的强先验而被高估。

5. **单模态消融**:无指令模型达到 17% SR,无图像模型也达到 17% SR,表明轨迹存在共同的规律性;但完整多模态模型(20% SR)仍明显优于单模态基线。

**局限性**

约 23% 的 R2R 轨迹无法在连续环境中导航(环境重建的不连续性、物体移动等)。当前端到端方法的绝对性能仍较低,未来需要探索模块化方法,如将学习到的智能体与运动控制器集成。论文未详细探索所有可能改善 VLN-CE 性能的技术(如更多应对 exposure bias 和数据稀疏性的方法)。

---
## 5. VLN-PE (2025)
———重新思考视觉-语言导航中的具身化差距:物理和视觉差异的全面研究

📄 **Paper**: https://arxiv.org/abs/2507.13019v2

**精华**

这篇论文通过构建物理真实的VLN平台,系统性地揭示了理想化仿真与物理部署之间的巨大差距。核心启示包括:(1) 跨具身数据融合训练可以显著提升模型泛化能力,为统一的跨机器人导航模型奠定基础;(2) 多模态感知(RGB+Depth)比单一RGB更鲁棒,尤其在光照变化环境下;(3) 物理控制器的引入对于腿足机器人至关重要,训练和评估阶段的控制器一致性直接影响性能;(4) 现有MP3D风格数据集的泛化能力有限,小规模域内数据微调即可超越大模型零样本性能;(5) diffusion policy作为连续路径点预测的新范式在VLN任务中展现潜力。

**研究背景/问题**

现有的VLN方法在理想化仿真环境中表现优异,但在部署到真实物理机器人时面临巨大挑战。主要问题包括:当前VLN平台忽视了机器人的物理具身特性(如视点高度、运动动力学、碰撞和跌倒等),并且缺乏对不同机器人类型(轮式、人形、四足)的跨具身支持。研究核心问题是:物理具身约束和视觉环境变化对现有VLN方法的性能影响究竟有多大?

**主要方法/创新点**

<div align="center">
  <img src="/images/VLN-PE-evolution.png" width="100%" />
<figcaption>
VLN任务的演进:从oracle-based导航(2018)到VLN-CE连续导航(2020),再到VLN-PE物理真实导航(2025)
</figcaption>
</div>

论文提出了**VLN-PE平台**,一个基于GRUTopia构建的物理真实VLN基准测试平台,具有以下核心特性:

1. **跨具身支持**:支持人形机器人(Unitree H1, G1)、四足机器人(Unitree Aliengo)和轮式机器人(Jetbot),并提供基于RL的物理控制器API,实现真实的运动动力学模拟

2. **场景多样性**:除了90个MP3D场景外,新增10个高质量合成家居场景(GRScenes)和3DGS在线渲染实验室场景,支持无缝集成更多环境

<div align="center">
  <img src="/images/VLN-PE-platform-overview.png" width="100%" />
<figcaption>
VLN-PE平台概览:支持多种机器人具身、场景类型、光照条件和控制器模式
</figcaption>
</div>

3. **系统性评估框架**:评估三类ego-centric VLN方法
   - **单步端到端方法**:Seq2Seq、CMA(约36M参数)和NaVid(7B参数的视频MLLM)
   - **多步端到端方法**:首次提出RDP(Recurrent Diffusion Policy),使用transformer-based diffusion模块预测连续轨迹路径点
   - **地图基零样本方法**:改进的VLMaps,结合LLM和语义地图进行路径规划

<div align="center">
  <img src="/images/VLN-PE-RDP-framework.png" width="100%" />
<figcaption>
RDP(循环扩散策略)框架:使用GRU维护历史信息,交叉注意力融合视觉-语言特征,Transformer扩散模块预测连续动作序列
</figcaption>
</div>

4. **新数据集**:
   - **R2R-filtered**:过滤楼梯场景后保留8,679/658/1,347个训练/val-seen/val-unseen episodes
   - **GRU-VLN10**:10个合成场景,441/111/1,287个episodes
   - **3DGS-Lab-VLN**:3DGS渲染实验室环境,160训练/640评估episodes

5. **新评估指标**:除了传统的TL、NE、SR、OS、SPL外,新增Fall Rate (FR)和Stuck Rate (StR)来衡量物理真实性挑战

**核心结果/发现**

<div align="center">
  <img src="/images/VLN-PE-main-results.png" width="100%" />
<figcaption>
使用人形机器人Unitree H1在R2R数据集上的主要实验结果对比
</figcaption>
</div>

**零样本迁移性能大幅下降**:
- VLN-CE模型直接迁移到VLN-PE时,SR相对下降约34%
- Seq2Seq-Full、CMA-Full和NaVid的SR分别下降10%、16%和18%
- 这表明现有模型严重过拟合特定仿真平台

**域内微调显著提升**:
- 在VLN-PE上从头训练的CMA(无数据增强)超越了使用175K增强数据训练的CMA-Full
- 小模型CMA+经过微调后,在val-seen上达到SR 28.72,SPL 24.24,超越NaVid的零样本性能

**跨具身敏感性**:
- 四足机器人(相机高度约0.5m)在迁移时几乎完全失败
- 调整相机高度到1.8m可改善人形机器人的迁移性能
- 跨具身联合训练使单一模型在所有机器人类型上达到SoTA性能

**物理控制器的重要性**:
- 训练和评估使用相同控制器时性能最佳
- 使用物理控制器收集数据可降低Fall Rate和Stuck Rate

**多模态鲁棒性**:
- 仅RGB的NaVid在低光照下SR下降12.47%
- RGB+Depth的CMA和RDP受光照影响较小(下降约1-2%)

**MP3D数据集泛化能力有限**:
- 在GRU-VLN10上,RDP用6M参数仅441个训练样本,零样本超越NaVid大模型
- 在3DGS-Lab-VLN上,NaVid完全失败(SR仅5.81),可能是3DGS渲染噪声导致

**扩散策略的潜力**:
- RDP作为首个VLN扩散策略基线,在从头训练时优于Seq2Seq和CMA
- 预测连续密集路径点,可与MPC等控制理论方法结合

**真机实验验证**:
- 使用Unitree Go2机器人进行14个室内场景测试
- VLN-PE微调模型在真实环境中OS达到57.14,SR达到28.57,显著优于VLN-CE训练模型

**局限性**

当前RL-based运动控制器无法可靠处理复杂环境中的楼梯导航,需要过滤相关场景。论文主要聚焦ego-centric视角,未评估panoramic VLN方法。MLLM在精确目标识别和停止决策上仍存在挑战。3DGS渲染引入的像素级噪声可能干扰纯RGB模型,需要进一步研究图像扰动的鲁棒性。

---
## 6. InternVLA-A1 (2026)
——Unifying Understanding, Generation and Action for Robotic Manipulation

📄 **Paper**: https://arxiv.org/abs/2601.02456

---

**精华**

InternVLA-A1 的核心创新在于将语义理解、视觉预见（visual foresight）与动作执行统一到单一 Mixture-of-Transformers (MoT) 框架中，用"想象未来"来指导当前动作，特别适合动态场景。其层级数据金字塔（合成仿真数据 + 真实数据混合预训练）有效弥合了 sim-to-real gap，值得 VLA 研究者借鉴。Generation Expert 的引入通过联合训练视觉预测和动作预测目标，使模型内化了动作与环境动力学之间的因果关系，是提升动态鲁棒性的关键设计。Flow Matching 作为动作解码器既保留了 MLLM 语义理解能力，又获得了对多模态动作分布的精细建模。

---

**研究背景/问题**

主流 VLA 模型（如 π₀、GR00T N1.5）基于 MLLM 构建，具有强大的语义理解能力，但本质上缺乏对物理世界动态的推理能力——它们执行的是反应式感知到动作映射，而非预判状态将如何演变。现有引入 World Model 的视频预测方法（如 VPP、Genie Envisioner）虽然能预测未来观测，但语义接地弱且对预测误差敏感。本文的目标是构建一个能同时紧密耦合语义理解与动态预测的统一架构。

---

**主要方法/创新点**

<div align="center">
  <img src="/images/InternVLA-A1-overview.png" width="100%" />
<figcaption>
InternVLA-A1 整体框架：理解专家、生成专家、动作专家三者协同工作，将语义推理与动力学预测融合以指导动作执行
</figcaption>
</div>

InternVLA-A1 采用 **Mixture-of-Transformers (MoT)** 架构，协调三个专家模块共同工作：

**（1）Understanding Expert（理解专家）**
直接复用现有 MLLM 架构（InternVL3-1B 或 Qwen3-VL-2.13B），通过 ViT 视觉编码器处理多视角观测 `o_t`，通过文本 Tokenizer 处理语言指令 `l`，将二者拼接为 prefix tokens `h_und`，为下游专家提供语义上下文。

**（2）Generation Expert（生成专家）**
受 Janus Pro 启发，采用**解耦视觉编码**策略——理解用 ViT（高层语义），生成用 VAE（像素级保真）。具体使用 Cosmos CI8×8 连续 VAE tokenizer 将输入图像编码为 latent features `z_t`，再经卷积层压缩空间维度至 4×4（每帧仅 16 个 tokens），对齐 Transformer 隐维度后送入生成专家。生成专家在历史帧 `z_{t-m}` 和当前帧 `z_t` 基础上，以 `h_und` 为条件，预测未来帧的 latent `ẑ_{t+m}`，最终经反卷积和 Cosmos decoder 重建预测图像。

<div align="center">
  <img src="/images/InternVLA-A1-architecture.png" width="100%" />
<figcaption>
InternVLA-A1 架构详图：三专家通过 Unified Masked Self-Attention 交互，理解专家输出语义上下文，生成专家预测未来视觉状态，动作专家基于两者产生控制指令
</figcaption>
</div>

**（3）Action Expert（动作专家）**
以语言目标 `l`、当前观测（经 `h_und`）、本体感知 `q_t` 和生成专家的预测 latent `ẑ_{t+m}` 为条件，使用 **Flow Matching** 目标预测动作块 `â_{t:t+k}`。采样时从高斯噪声出发，通过 Euler 迭代法解 ODE 得到目标动作。

**（4）Unified Masked Self-Attention**
实现三专家间信息流的分块注意力掩码：累积分段掩码确保信息流单向传递（理解 → 生成 → 动作）；前缀块（视觉+语言）完全双向；生成块完全双向且仅接收 Cosmos latent tokens；动作块分为状态 token（只关注自身和更早块）和动作 tokens（相互关注）。

**（5）优化目标**
联合优化两个目标：
- **视觉预见生成**：$\mathcal{L}_{\text{gen}} = \mathbb{E}\left[\|f_{\text{gen}}(z_{t-m}, z_t; h_{\text{und}}) - \text{sg}[z_{t+m}]\|^2\right]$
- **Flow Matching 动作预测**：$\mathcal{L}_{\text{action}} = \mathbb{E}\left[\|v_\theta(l, \{o_i\}_{i=t-m}^t, q_t, a_{t:t+k}^\tau) - (a_{t:t+k} - \epsilon)\|^2\right]$
- **总损失**：$\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{action}}$，其中 $\lambda = 0.01$

**（6）层级数据金字塔**

<div align="center">
  <img src="/images/InternVLA-A1-data-pyramid.png" width="100%" />
<figcaption>
层级数据金字塔：底层为大规模开源示范数据（AgiBot-World），中层为仿真合成数据（InternData-A1），顶层为专项真实数据
</figcaption>
</div>

预训练数据混合配方（共 533M+ 帧）：
- InternData-A1（ARX Lift-2）：96M 帧（18%）
- InternData-A1（AgileX）：122.5M 帧（23%）
- InternData-A1（Franka）：90.5M 帧（17%）
- InternData-A1（Genie-1）：16M 帧（3%）
- AgiBot-World（Beta）：208M 帧（39%）

预训练后，使用少量专项真实数据进行 post-training 微调，适配目标部署环境。

**（7）模型规模**
- InternVLA-A1（2B）：Understanding=InternVL3（0.94B）+ Gen/Act=Qwen2.5（各 0.36B），共 1.8B
- InternVLA-A1（3B）：Understanding=Qwen3-VL（2.13B）+ Gen/Act=Qwen3（各 0.44B），共 3.2B
- 推理速度：两者均约 13 Hz（NVIDIA RTX 4090）

---

**核心结果/发现**

**通用任务（10 个真实任务，Table 4）**：
- InternVLA-A1（3B）平均成功率 **75.1%**，比 π₀（3.3B）的 60.6% 提升 **14.5%**
- InternVLA-A1（2B）以 64.7% 超越更大的 π₀（3.3B）模型，凸显架构与数据质量优势
- 在精细操作任务（Make Sandwich: 93.3% vs 66.7%；Operate Oven: 86.7% vs 73.3%）表现尤为突出

**动态场景专项任务（Figure 6）**：

<div align="center">
  <img src="/images/InternVLA-A1-dynamic-results.png" width="100%" />
<figcaption>
Express Sorting 和 In-motion Ingredient Picking 任务的成功率对比：InternVLA-A1（3B）以 80% 和 93.3% 大幅领先基线
</figcaption>
</div>

- Express Sorting：π₀ 仅 36.7%，GR00T N1.5 仅 40.0%，InternVLA-A1（3B）达 **80.0%**（+40%以上）
- In-motion Ingredient Picking：基线均仅 20.0%，InternVLA-A1（3B）达 **93.3%**（+73.3%）

**仿真基准（RoboTwin 2.0, 50 任务）**：InternVLA-A1（3B）Easy/Hard 分别为 65.0%/25.4%，超越 π₀ 的 54.5%/19.8%（+10.5%/+5.6%）

**消融实验**：
- 去除预训练：平均成功率从 77.0% 降至 25.4%（↓51.6%）
- 去除 Generation Expert：平均成功率从 77.0% 降至 57.6%（↓19.4%），11/12 个任务均退化

---

**局限性**

理解专家缺乏与多模态 VQA 数据集的联合训练，导致通用语义推理和复杂指令跟随能力有所退化；视觉预见模块为保证实时推理效率而牺牲了图像预测的保真度，生成未来帧的粒度有限。


---
## 7. InternData-A1 (2025)
——Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy

📄 **Paper**: https://arxiv.org/abs/2511.16651

---

**精华**

本文首次证明纯合成数据预训练的 VLA 模型可以匹配甚至超越使用真实机器人数据的最强基线（π-dataset），打破了"仿真数据无法替代真实数据"的固有认知。数据合成 pipeline 完全解耦（环境构建、技能组合、Domain Randomization、轨迹生成独立模块化），极大降低人工成本（每条 episode 低于 0.003 美元）。消融实验揭示**轨迹多样性**（articulation + long-horizon tasks）而非单一规模是有效 VLA 预训练的核心驱动力，这对数据采集策略有重要指导意义。大规模 domain randomization 使仿真与真实的视觉 gap 缩小到约 1:8 的仿真对真实数据等效比例，强调了渲染保真度和随机化的重要性。开源数据集和生成 pipeline 为 embodied AI 社区提供了可复现的大规模数据基础设施。

---

**研究背景/问题**

现有 VLA 模型已证明大规模真实机器人数据预训练的有效性，但合成数据单独能否达到相同效果尚未被系统验证。真实数据采集代价高昂，需要专业遥操作员、特殊硬件和大量人力，大多数研究机构难以复现；现有仿真数据集覆盖的技能集窄（主要是 pick-and-place）、仅涉及 rigid 物体，且未在大规模 VLA 预训练中验证有效性。

---

**主要方法/创新点**

InternData-A1 是一个包含 630k 轨迹、7,433 小时、覆盖 4 种机器人体态（AgiBot Genie-1、Franka Emika Panda、AgileX Split Aloha、ARX Lift-2）、18 种技能、70 个任务、227 个室内场景的大规模高保真合成数据集。

<div align="center">
  <img src="/images/InternData-A1-data-statistics.png" width="100%" />
<figcaption>
InternData-A1 数据统计概览：4 种体态、70 个任务、3185 个 rigid 物体、321 个 articulation 物体、20 件服装，共 630k episodes、401.4M 帧、7433.9 小时
</figcaption>
</div>

### 数据合成 Pipeline（4 阶段全自动）

<div align="center">
  <img src="/images/InternData-A1-pipeline.png" width="100%" />
<figcaption>
InternData-A1 数据合成 pipeline，包含环境构建、技能组合、Domain Randomization 和轨迹生成与存储四个阶段
</figcaption>
</div>

**1. Environment Construction（环境构建）**
- **Embodiment**: 支持 4 种体态，均以 USD 格式定义，经过碰撞动力学验证
- **Scene Library**: 227 个室内场景（厨房、书房、餐厅、客厅）来自 GRScenes-100，每个场景标注了详细的操作区域元数据
- **Object Library**: 覆盖 rigid（3185 个，含自动 grasp pose 标注）、articulated（321 个，含关节轴和物理参数）、deformable（20 件真实扫描服装，用 Vertex Block Descent 模拟）、fluid（粒子系统 + isosurface 渲染）四类物体

**2. Skill Composition（技能组合）**
- 每个技能是模块化脚本策略，输入：物体状态、机器人状态、用户约束；输出：waypoints 序列（end-effector 6D pose）
- 包含 Pick、Place、Push 等 18 种原子技能，通过简单配置文件组合成完整任务
- 支持双臂并行和顺序执行，无需额外代码即可扩展到新物体、场景、体态
- 18 个 long-horizon 任务（每个涉及至少 3 个连续技能），共 124,789 条轨迹

**3. Domain Randomization（域随机化）**
- **视觉多样性**: 相机视角 ±5° 旋转、±5cm 平移；174 个环境光照图（随机光温和强度）；目标物体可从同类资产中替换
- **轨迹多样性**: 物体位姿在任务特定空间范围内随机采样；AnyGrasp 生成数百万 grasp 候选，最终随机选取 top-40 之一；articulated 和 deformable 物体的接触区域扩展为邻域

**4. Generation & Storage（生成与存储）**
- 使用 **CuRobo** 运动规划器在 waypoints 间插值密集关节空间动作
- 仅存储成功完成的轨迹（Isaac Sim 物理验证），转换为 **LeRobot** 格式
- 记录：物体元数据、语言指令、多视角 RGB、相机参数、机器人本体感知状态和动作标签

**5. Framework Optimization（框架优化）**
- **Stage Decoupling**: 轨迹规划（CPU-bound）与视觉渲染（GPU-bound）解耦为 pipeline 架构，规划失败不触发冗余渲染
- **Dynamic Resource Scheduling**: Planner 和 Renderer 内部均采用并行批处理策略 + 动态调度算法
- **Stack Render**: 堆叠渲染技术进一步提升 GPU 利用率
- **Cluster Stability**: Balancer 模块负载均衡 + Supervisor 模块监控，整体吞吐量提升 **2–3×**，生产成本低于 **$0.003/episode**

---

**核心结果/发现**

**与 π-dataset 对比（49 个仿真任务）**
- π₀(InternData-A1) vs 官方 π₀：Easy 模式 **60.0% vs 55.0%**（+5%），Hard 模式 **26.5% vs 20.0%**（+6.5%）
- 在 Hard 模式下的提升说明 InternData-A1 的大规模 domain randomization 提供的鲁棒性在下游 fine-tuning 中持续保留

**与 π-dataset 对比（9 个真实世界任务）**
<div align="center">
  <img src="/images/InternData-A1-realworld-comparison.png" width="100%" />
<figcaption>
InternData-A1 在 9 个真实世界任务上的性能对比，包括 5 个常规任务和 4 个灵巧任务，平均超越 π-dataset 6.2%
</figcaption>
</div>

- 在 5 个常规任务上平均超越 π-dataset **6.2%**（包括 Place Markpen、Pass Bottle、Heat Sandwich、Sort Rubbish、Sweep Trash）
- 在 4 个灵巧任务（Sort Parts、Unscrew Cap、Fold Clothes、Zip Bag）上性能与 π-dataset 相当，使用了全新体态 ARX AC One（训练数据中未见过）

**与开源数据集对比（49 个仿真 + 2 个真实任务）**
- InternData-A1 大幅领先：Easy **60.0%** vs OXE 32.5% / Agibot World 52.5% / RoboCasa 50.0%
- 真实任务 Sort Rubbish：**90.0%** vs OXE 40.0%；Pass Bottle：**60.0%** vs RoboCasa 13.3%

**Sim-to-Real 迁移**
<div align="center">
  <img src="/images/InternData-A1-sim2real-results.png" width="100%" />
<figcaption>
6 个 sim-to-real 任务仅使用 500 条仿真 episodes 即可实现超过 50% 的成功率
</figcaption>
</div>

- 10 个任务中直接零样本迁移成功率超过 50%；仅需 500 条仿真数据即达到高成功率
- 对于基础技能任务（Sort Rubbish、Wipe Stain），200 条仿真 episodes ≈ 200 条真实数据
- 对于复杂任务（Flip Package、Instructional Pick），仿真对真实等效比约为 **8:1**

**消融实验（数据组成分析）**
- 去除 Base 或 Long-horizon 任务的性能下降 > 去除 PnP 任务，说明任务多样性比单一任务规模更重要
- 去除 Articulation 任务（仅 11.67%）导致显著下降，说明 articulated 操作能扩展 action space 多样性
- 核心结论：**轨迹多样性（Trajectory Diversity）是有效预训练的核心驱动**

---

**局限性**

由于物理仿真器的局限，目前难以模拟高度灵巧的操作任务（如系鞋带、穿针引线等精细接触任务）；未来工作将扩展任务多样性和灵巧度，进一步确立大规模仿真数据作为 VLA 模型发展基石的地位。

---

