---
layout: post
title: "Vision-Language Navigation (VLN) 综述"
date:   2026-01-04
tags: [VLN, Robotics, Computer Vision, Deep Learning]
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

## VLN的三个核心要素

一个完整的VLN系统包含三个重要的组成部分：

1. **Oracle（神谕/指令发布者）**：模拟人的作用，负责发布语言指令。Agent可以向Oracle请求指导，Oracle做出回应。

2. **Agent（智能体/执行者）**：需要被训练和学习的机器人，是任务的执行者。Agent根据收到的指令和观察到的环境与环境交互，并完成具体任务。

3. **Environment（环境）**：智能体需要工作的空间。考虑到真实场景训练成本比较高昂，一般都采用模拟器，比如Room-to-Room (R2R)任务采用Matterport 3D数据集作为仿真的室内环境。

## VLN vs VQA

与经典的视觉问答（VQA）任务相比，VLN增加了**主动视觉（Active Vision）**的观测。在每一步动作的过程中，视觉输入也在不断变化，智能体需要根据当前观察来决定下一步的行动。

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/WX20250824-201611.png" width="80%" />
<figcaption>
VQA vs VLN
</figcaption>
</div>

## VLN的主要挑战

**1. 语言理解与视觉感知的对齐**

如何将自然语言指令准确地映射到视觉观察和导航动作，是VLN的核心挑战之一。

**2. 泛化能力**

模型需要在未见过的环境中进行导航，这要求模型具有良好的泛化能力。

**3. 长期规划与短期决策**

智能体需要在理解全局指令的同时，做出实时的局部决策。

**4. Sim-to-Real迁移**

从模拟器训练的模型如何有效迁移到真实机器人平台，仍然是一个重要的研究问题。

**5. 多模态信息融合**

如何有效融合语言、视觉、定位等多模态信息，提升导航性能。

## VLN研究发展趋势

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/2025-vln-research-timeline.png" width="100%" />
<figcaption>
VLN Research Timeline (Refer from "Thinking-VLN")
</figcaption>
</div>

从研究时间线可以看出，VLN领域的发展经历了以下几个阶段：

1. **2018-2019**：基础架构探索阶段，主要关注如何更好地表征多模态数据

2. **2020-2021**：数据集扩展阶段，研究者构建了更多样化的数据集

3. **2022-2023**：大模型应用阶段，基于预训练模型和大语言模型的方法开始涌现

4. **2024-至今**：具身智能与真实机器人部署阶段，研究重点转向实际应用

## 关键技术方向

**1. 基于Transformer的架构**

利用Transformer强大的序列建模能力，处理语言指令和视觉观察序列。

**2. 预训练模型**

使用大规模预训练的视觉-语言模型（如CLIP、BERT等）提升VLN性能。

**3. 强化学习**

通过环境交互学习导航策略，优化长期累积奖励。

**4. 图神经网络**

构建环境的拓扑图，利用图神经网络进行路径规划。

**5. 视觉-语言预训练**

在大规模图文数据上进行预训练，学习跨模态表示。

## 未来研究方向

1. **更强的泛化能力**：开发能够在开放世界中泛化的VLN模型

2. **真实机器人部署**：缩小Sim-to-Real差距，实现实际应用

3. **多模态大模型**：利用大语言模型的推理能力提升VLN性能

4. **持续学习**：使智能体能够从经验中持续学习和改进

5. **人机协作**：开发更自然的人机交互方式

6. **安全性和可解释性**：提升导航决策的安全性和可解释性



# VLN任务类型

根据任务目标和交互方式的不同，VLN可以分为以下几种主要类型：

## 按照导航目标分类

**1. 指令导向（Instruction-Oriented）**

指令导向的VLN任务侧重于智能体严格遵循给定的语言指令进行导航。这种任务要求智能体能够理解复杂的自然语言指令，并将其转化为导航动作。

**示例指令**：
- "往前走到海报附近然后右拐进办公室"
- "沿着走廊直走，在第二个路口左转"

**代表性数据集**：
- **Room-to-Room (R2R)**：最早且最具影响力的VLN数据集
- **Room-for-Room (R4R)**：R2R的扩展版本，包含更长的导航路径

**2. 目标导向（Goal-Oriented）**

目标导向的VLN任务要求智能体根据给定的目标进行导航。智能体需要理解目标的语义信息，并在环境中搜索与目标相匹配的物体。

**示例任务**：
- "找到沙发"
- "导航到厨房的冰箱旁"

**代表性数据集**：
- **REVERIE**：结合了导航和物体定位
- **SOON**：强调目标物体的语义理解

**3. 需求导向（Demand-Oriented）**

需求导向的VLN是一种更高级的形式，它要求智能体根据用户的抽象需求进行导航。与前两种任务不同，需求导向导航不依赖于特定的物体或目标，而是需要智能体理解用户的需求并找到满足这些需求的物体或位置。

**示例需求**：
- "我饿了" → 导航到厨房或寻找食物
- "我想休息" → 导航到卧室或沙发

**代表性数据集**：
- **DDN (Demand-driven Navigation)**

## 按照交互轮数分类

**单轮指令任务**

在单轮指令任务中，智能体接收到一个自然语言指令，并且需要在没有进一步交互的情况下执行该指令。

**多轮对话式导航任务**

对话式导航任务涉及更复杂的交互，智能体可以在导航过程中与用户进行多次对话。智能体可能无法仅凭初始指令就完全理解用户的意图，需要通过提问来获取更多信息。

**代表性数据集**：
- **CVDN (Cooperative Vision-and-Dialog Navigation)**
- **TEACh**：任务驱动的具身代理通信

# VLN的应用场景

## 室内场景

室内VLN主要关注家庭或办公环境内的导航。环境通常较为复杂，包含多个房间和各种家具，对智能体的空间理解能力要求较高。

**应用示例**：
- 家庭服务机器人
- 室内物流配送
- 智能导览系统

## 室外场景

室外VLN面临更大的环境复杂度，需要处理动态障碍物、天气变化等因素。

**应用示例**：
- 自动驾驶
- 户外服务机器人
- 城市导航系统

## 空中场景

空中VLN涉及无人机等飞行器的导航控制。

**应用示例**：
- 无人机巡检
- 空中搜救
- 航拍导航



# VLN主流数据集

VLN研究依赖高质量的数据集来训练和评估导航模型。以下是VLN领域最具影响力的主流数据集：

## 指令导向数据集

### R2R (Room-to-Room)

**基本信息：**
- **发布时间**：2018年
- **规模**：10,567张全景图像，90个真实室内环境
- **指令数量**：7,189条路径，每条路径配有3个自然语言指令
- **环境来源**：Matterport3D真实场景扫描

**特点：**
- VLN领域最早且最具影响力的基准数据集
- 采用离散环境表示（连通图），智能体在预定义视点间导航
- 平均路径长度约为10米，涉及3-5个房间
- 指令由人类标注员编写，具有自然语言的多样性和复杂性

**任务目标**：智能体根据自然语言指令，从起点导航到终点位置

### R4R (Room-for-Room)

**基本信息：**
- **发布时间**：2019年
- **环境基础**：基于R2R数据集扩展
- **特点**：包含更长、更复杂的导航路径

**改进点：**
- 通过拼接多条R2R路径生成长指令任务
- 平均路径长度显著增加，对模型的长期规划能力要求更高
- 更适合测试模型在复杂多房间环境中的导航能力

### RxR (Room-across-Room)

**基本信息：**
- **发布时间**：2020年
- **语言支持**：多语言VLN数据集（英语、印地语、泰卢固语）
- **规模**：比R2R更大规模的数据集

**特点：**
- 支持跨语言VLN研究
- 指令更加详细和具体
- 促进了多语言导航模型的发展

## 目标导向数据集

### REVERIE (Remote Embodied Visual Referring Expression in Real Indoor Environments)

**基本信息：**
- **发布时间**：2020年
- **任务类型**：导航 + 物体定位的组合任务

**特点：**
- 结合了视觉语言导航和物体接地（grounding）
- 智能体需要根据指令导航到特定房间，并定位目标物体
- 需要细粒度的视觉理解和物体识别能力
- 包含10,466个指令，涉及21,023个物体实例

**任务目标**：导航到目标物体所在位置，并准确识别目标物体

### SOON (Semantic Object-Oriented Navigation)

**基本信息：**
- **任务类型**：基于语义的目标导航

**特点：**
- 强调目标物体的语义理解
- 要求模型具备开放词汇表的物体识别能力
- 测试模型的零样本泛化能力

## 对话式导航数据集

### CVDN (Cooperative Vision-and-Dialog Navigation)

**基本信息：**
- **发布时间**：2019年
- **任务类型**：多轮对话式导航

**特点：**
- 智能体可以在导航过程中与Oracle进行多轮对话
- 通过提问获取额外信息来消除歧义
- 更接近真实人机交互场景
- 包含2,050个对话，平均每个对话4.5轮交互

**任务目标**：通过多轮对话与Oracle交互，准确到达目标位置

### TEACh (Task-driven Embodied Agents that Chat)

**基本信息：**
- **发布时间**：2021年
- **任务类型**：任务驱动的具身智能体通信

**特点：**
- 结合对话、视觉和动作执行
- 支持更复杂的任务指令和交互
- 涵盖多种家居任务场景

## 需求导向数据集

### DDN (Demand-driven Navigation)

**基本信息：**
- **任务类型**：基于抽象需求的导航

**特点：**
- 要求智能体理解用户的抽象需求（如"我饿了"、"我想休息"）
- 需要推理能力将需求映射到具体目标
- 更高层次的语义理解挑战

## 特殊场景数据集

### AerialVLN

**基本信息：**
- **任务类型**：空中无人机导航
- **场景类型**：室外、空中视角

**特点：**
- 专门针对无人机等飞行器的导航任务
- 俯视视角与室内导航有显著差异
- 环境复杂度更高，需要处理更大的空间范围

## 数据集对比

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-datasets-comparison.png" width="100%" />
<figcaption>
主要VLN数据集对比与特点
</figcaption>
</div>

| 数据集 | 发布年份 | 任务类型 | 环境类型 | 交互方式 | 主要特点 |
|--------|----------|----------|----------|----------|----------|
| R2R | 2018 | 指令导向 | 室内 | 单轮 | 经典基准，离散导航 |
| R4R | 2019 | 指令导向 | 室内 | 单轮 | 长路径，高难度 |
| RxR | 2020 | 指令导向 | 室内 | 单轮 | 多语言支持 |
| REVERIE | 2020 | 目标导向 | 室内 | 单轮 | 导航+物体定位 |
| SOON | 2020 | 目标导向 | 室内 | 单轮 | 语义理解 |
| CVDN | 2019 | 对话导航 | 室内 | 多轮 | 对话交互 |
| TEACh | 2021 | 对话导航 | 室内 | 多轮 | 任务驱动 |
| DDN | 2021 | 需求导向 | 室内 | 单轮 | 抽象需求推理 |
| AerialVLN | 2021 | 指令导向 | 室外/空中 | 单轮 | 无人机导航 |

# VLN主流模拟器

VLN研究需要高质量的3D仿真环境来训练和测试导航模型。以下是VLN领域最常用的主流模拟器：

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

## Habitat

**基本信息：**
- **开发者**：Facebook AI Research (FAIR)
- **发布时间**：2019年
- **开源地址**：[GitHub](https://github.com/facebookresearch/habitat-lab)

**核心特点：**
- **高性能仿真**：通过高度优化的渲染引擎实现超快速仿真（10,000+ FPS）
- **连续环境**：支持连续动作空间和自由移动
- **多数据集支持**：兼容Matterport3D、Gibson、HM3D（Habitat-Matterport 3D）等多个数据集
- **模块化设计**：灵活的任务定义和传感器配置
- **Sim2Real支持**：提供sim-to-real工具链，支持真实机器人部署

**应用场景：**
- VLN-CE (Continuous Environment) 基准测试
- 连续动作空间的导航研究
- 目标导航（ObjectNav）、语义导航（SemanticNav）
- 具身AI研究和真实机器人部署

**优势：**
- 仿真速度极快，训练效率高
- 支持连续导航，更接近真实机器人控制
- 数据集丰富，HM3D提供800+高质量场景
- 强大的扩展性和社区生态

**局限性：**
- 配置相对复杂，学习曲线较陡
- 对硬件要求较高（GPU加速）

**主要变种：**
- **Habitat 1.0**：基础版本
- **Habitat 2.0**：增加交互式任务支持
- **Habitat 3.0**：引入人机协作和社交导航能力

## AI2-THOR

**基本信息：**
- **开发者**：Allen Institute for AI
- **发布时间**：2017年（持续更新）
- **开源地址**：[官网](https://ai2thor.allenai.org/)

**核心特点：**
- **物理交互**：基于Unity3D引擎，支持完整的物理模拟
- **可交互对象**：环境中的物体可以被抓取、移动、操作
- **多样化场景**：包含厨房、卧室、客厅、浴室等120+室内场景
- **语义分割**：内置语义分割和实例分割功能
- **多智能体支持**：支持多个智能体同时运行

**应用场景：**
- 具身问答（Embodied Question Answering）
- 视觉语言导航与操作结合的任务
- 家庭服务机器人研究
- 需要物理交互的复杂任务

**优势：**
- 物理引擎强大，支持真实的物体交互
- 场景设计精美，视觉真实感高
- API友好，易于上手
- 支持多模态任务（导航+操作）

**局限性：**
- 仿真速度相对较慢（基于Unity渲染）
- 场景规模较小，主要是单个房间
- 计算资源消耗较大

## Gibson / iGibson

**基本信息：**
- **开发者**：Stanford University
- **Gibson发布时间**：2018年
- **iGibson发布时间**：2021年
- **开源地址**：[iGibson GitHub](https://github.com/StanfordVL/iGibson)

**核心特点：**

**Gibson 1.0：**
- 基于真实环境的3D重建（1000+真实建筑扫描）
- 快速光栅化渲染
- 支持物理仿真（基于Bullet物理引擎）

**iGibson 2.0：**
- **交互式场景**：支持完整的物理交互和物体操作
- **逼真渲染**：基于物理的渲染（PBR），视觉质量显著提升
- **语义信息**：丰富的语义标注和物体属性
- **大规模场景**：包含完整的房屋、办公楼等大型环境
- **任务多样性**：支持导航、操作、家务任务等

**应用场景：**
- 大规模室内导航
- 导航与操作结合的任务
- Sim-to-Real迁移研究
- 家庭服务机器人仿真

**优势：**
- 场景数量多，环境多样性好
- 真实感强，基于实际建筑扫描
- iGibson 2.0功能全面，支持复杂交互
- 性能优化良好，渲染速度快

**局限性：**
- 数据集下载和安装较为复杂
- 部分场景质量参差不齐

## AirSim

**基本信息：**
- **开发者**：Microsoft
- **发布时间**：2017年
- **开源地址**：[GitHub](https://github.com/microsoft/AirSim)

**核心特点：**
- **无人机/车辆仿真**：专门为飞行器和地面车辆设计
- **高保真物理**：基于Unreal Engine或Unity，物理模拟精确
- **多传感器支持**：相机、LiDAR、IMU、GPS等
- **API丰富**：支持Python和C++ API
- **跨平台**：支持Windows、Linux、macOS

**应用场景：**
- 空中VLN（AerialVLN）
- 无人机导航与控制
- 自动驾驶研究
- 户外导航任务

**优势：**
- 专业的飞行器仿真平台
- 物理模拟精度高
- 支持大规模户外环境
- 工业级应用支持

**局限性：**
- 主要面向无人机和车辆，室内导航支持有限
- 配置复杂，对硬件要求高

## 模拟器对比

| 模拟器 | 环境类型 | 动作空间 | 物理交互 | 渲染速度 | 主要应用 | 场景数量 |
|--------|----------|----------|----------|----------|----------|----------|
| Matterport3D | 室内 | 离散 | 有限 | 快 | R2R基准 | 90 |
| Habitat | 室内 | 连续 | 基础 | 极快 | VLN-CE | 800+ (HM3D) |
| AI2-THOR | 室内 | 离散/连续 | 强 | 中等 | 交互任务 | 120+ |
| Gibson/iGibson | 室内 | 连续 | 强 | 快 | 综合任务 | 1000+ |
| AirSim | 室内外 | 连续 | 强 | 中等 | 无人机/车辆 | 可定制 |

## 选择建议

**研究经典VLN基准（R2R/R4R）：**
- 推荐使用 **Matterport3D Simulator**

**连续环境VLN研究：**
- 推荐使用 **Habitat**（速度快，社区活跃）

**需要物理交互的任务：**
- 推荐使用 **AI2-THOR** 或 **iGibson 2.0**

**无人机/空中导航：**
- 推荐使用 **AirSim**

**大规模场景训练：**
- 推荐使用 **Gibson/iGibson** 或 **Habitat + HM3D**

**Sim-to-Real部署：**
- 推荐使用 **Habitat**（提供完整工具链）

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

## 评估指标总结

### 按重要性分类

**核心指标（必须报告）：**
- Success Rate (SR)
- Success weighted by Path Length (SPL)

**辅助指标（建议报告）：**
- Navigation Error (NE)
- Oracle Success Rate (OSR)
- Trajectory Length (TL)

**高级指标（针对特定研究）：**
- nDTW / SDTW（轨迹质量研究）
- CLS（指令跟随研究）
- Collision Rate（安全性研究）

### 按任务类型选择

| 任务类型 | 主要指标 | 辅助指标 |
|----------|----------|----------|
| 指令导向（R2R） | SR, SPL | NE, OSR, nDTW |
| 目标导向（REVERIE） | SR, SPL | NE, 物体定位准确率 |
| 对话式导航（CVDN） | SR, SPL | 对话轮数, NE |
| 连续环境（VLN-CE） | SR, SPL | Collision Rate, TL |
| 社交导航 | SR, SPL | Human Collision Rate |

## 评估最佳实践

**1. 标准化报告：**
- 始终报告SR和SPL作为核心指标
- 分别报告seen（训练集环境）和unseen（测试集环境）性能
- 对于R2R，报告val_seen和val_unseen的结果

**2. 公平比较：**
- 使用相同的成功阈值（通常3米）
- 在相同的数据集划分上评估
- 明确说明是否使用ground truth路径

**3. 消融实验：**
- 分析各个模块对不同指标的影响
- 检查SR和SPL之间的权衡关系
- 评估模型在不同难度任务上的表现

**4. 可视化分析：**
- 绘制轨迹可视化图
- 分析失败案例
- 统计路径长度分布

## 指标演进趋势

VLN领域的评估体系经历了以下演进：

**早期（2018-2019）：**
- 主要使用SR和SPL
- 评估体系相对简单

**中期（2020-2021）：**
- 引入nDTW评估轨迹质量
- 增加CLS等细粒度指标

**近期（2022-至今）：**
- 从单一SR指标扩展至综合评估框架
- 引入安全性指标（碰撞率）
- 增加社交导航相关指标
- 强调Sim-to-Real场景下的实际部署性能

**未来方向：**
- 更注重真实世界部署的实用性指标
- 引入能耗、时间等物理约束
- 评估长期任务的鲁棒性和可靠性
- 多模态、多目标导航的综合评估


# 学习资源

## 相关论文列表

- [Awesome-VLN](https://github.com/awesome-vln/awesome-vln) - VLN领域论文汇总
- [VLN Papers with Code](https://paperswithcode.com/task/vision-and-language-navigation)

## 重要会议

- CVPR (Computer Vision and Pattern Recognition)
- ICCV (International Conference on Computer Vision)
- ECCV (European Conference on Computer Vision)
- CoRL (Conference on Robot Learning)
- IROS (International Conference on Intelligent Robots and Systems)

## 开源项目

- [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)
- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
- [VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT)


# 主流框架

# VLN经典论文

## 1. ODYSSEY
——Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks

**研究背景/问题**

在动态、非结构化环境中，机器人需要将移动性、操作和实时感知紧密结合才能执行复杂任务。现有研究大多局限于桌面场景，未能解决移动平台特有的感知受限和执行器范围有限的问题，且在开放世界环境中的泛化能力不足。

**主要方法/创新点**

ODYSSEY提出了一个统一的移动操作框架，包含分层规划和全身控制两大核心模块：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/odyssey-framework-overview.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/odyssey-whole-body-control.png" width="100%" />
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

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/odyssey-results-comparison.png" width="100%" />
<figcaption>
与基线方法的性能对比
</figcaption>
</div>

**核心结果/发现**

- **短期任务**：在ARNOLD基准测试上优于PerAct基线，仅依赖单个自我中心摄像头实现更强的泛化能力，在未见数据集上性能保持稳定
- **长期任务**：在8个长期移动操作任务上实现40%以上整体成功率，每个原子技能类别保持60%以上成功率，展现出可靠的协调能力
- **低层策略**：在基座速度跟踪方面优于RoboDuet基线，末端执行器姿态跟踪性能相当，且在不同地形上具有更强的适应性
- **Sim-to-Real迁移**：成功在Unitree Go2+Arx5平台上实现现实世界部署，在"导航到抓取"和"抓取和放置"任务中验证了框架的实用性

**局限性**

模型在物体几何形状的空间推理方面存在局限，导致夹爪对齐不佳和细长手柄或部分遮挡物品的定位不准确。此外，抓取小物体时偶尔失败，主要由于末端执行器跟踪和视觉感知精度不足。

---

## 2. DualVLN
——Ground Slow, Move Fast

**研究背景/问题**

VLN领域存在基本矛盾：强大的推理能力需要"慢思考"，而流畅的导航行动需要"快反应"。传统端到端模型存在三大瓶颈：动作碎片化（每一步都需调用大模型）、响应延迟高（无法实现高频控制）、缺乏层次协调（语义理解、全局规划和局部避障耦合在一起）。

**主要方法/创新点**

DualVLN提出双系统架构，将高级语义理解与低级轨迹执行解耦，形成互补的快慢系统：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/dualvln-framework-overview.png" width="100%" />
<figcaption>
DualVLN双系统框架架构
</figcaption>
</div>

**系统2（慢思考的"大脑"）：**
- **全局规划器**：基于Qwen-VL-2.5，以约2 Hz频率运行，负责理解指令、观察环境
- **像素级目标预测**：将3D导航任务转化为2D像素级目标定位问题（最远像素目标grounding）
- **自动生成训练数据**：通过三维到二维投影，将未来轨迹点投影到当前视角的2D图像上，利用深度信息过滤遮挡点，选择最远可见点作为"像素目标"
- **智能视角调整**：自主决定何时调整视角（如"左转/右转15°、抬头/低头15°"），最多支持4次连续视角调整，模仿人类寻路行为

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/dualvln-system2-waypoint.png" width="100%" />
<figcaption>
系统2的3D路径到2D路标自动生成机制
</figcaption>
</div>

**系统1（快行动的"小脑"）：**
- **高频轨迹生成**：轻量级扩散Transformer策略，以高达30 Hz频率运行
- **条件扩散模型**：融合低频语义条件（来自系统2）与高频视觉条件（实时RGB图像）
- **语义特征提取**：使用4个可学习的潜在查询向量从系统2的隐藏状态中提取任务相关语义特征
- **动态环境适配**：通过融合旧图像特征与最新图像特征，动态理解机器人位移和环境变化
- 输出平滑、连续、避障的轨迹（32个密集路径点）

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/dualvln-system1-trajectory.png" width="100%" />
<figcaption>
系统1的高频轨迹生成机制
</figcaption>
</div>

**协同机制**：系统2每0.5秒规划一个新目标，系统1每0.03秒更新一次轨迹，实现"大脑想一步，小脑走十步"的高效控制。

**新基准Social-VLN**：
- 在VLN-CE环境中加入动态行走的人形机器人，沿任务路径放置，增加交互概率
- 引入Human Collision Rate指标，量化与行人的不安全交互次数

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/dualvln-social-vln-benchmark.png" width="100%" />
<figcaption>
Social-VLN基准测试场景示例
</figcaption>
</div>

**核心结果/发现**

- **仿真基准SOTA**：在VLN-CE和VLN-PE两大基准上均取得最佳成绩，尤其在R2R和RxR的未见场景中，成功率显著领先所有基线模型（NaVILA、StreamVLN等）
- **跨平台部署**：成功部署在轮式（Turtlebot4）、四足（Unitree Go2）、人形（Unitree G1）三种机器人平台上，仅搭载Intel RealSense D455单目RGB相机
- **多场景验证**：在办公室、食堂、街道、便利店等多种室内外场景中表现出色，能够规划平滑路径、避开动态行人、处理楼梯等复杂地形
- **零样本泛化**：展现出强大的零样本迁移能力，可直接迁移至长视程导航和户外自主探索任务
- **系统鲁棒性**：系统1对像素目标偏差具有鲁棒性，当系统2输出的像素目标存在方向正确但位置偏差时，系统1仍能通过实时RGB图像修正轨迹

**局限性**

系统在极端扰动下的鲁棒性仍需提升，仿真到现实的迁移效率有待优化。跨层表征对齐机制还需进一步改进，以实现更高效的双系统协同。

---

## 3. PanoNav:
——Mapless Zero-Shot Object Navigation

**研究背景/问题**

现有目标导航方法大多依赖深度传感器或预建地图来构建2.5D场景表示，限制了在真实环境中的适用性和泛化能力。零样本目标导航要求机器人识别和导航到超出预定义类别范围的对象，现有方法在开放词汇场景中表现有限。无地图方法通常只基于当前观测进行决策，忽略历史轨迹信息，容易陷入局部死锁。

**主要方法/创新点**

PanoNav是一个无地图、仅使用RGB图像的零样本目标导航框架，包含两个核心模块：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/panonav-framework-overview.png" width="100%" />
<figcaption>
PanoNav框架整体架构
</figcaption>
</div>

**全景场景解析（Panoramic Scene Parsing）：**

*局部方向解析：*
- **点阵图像增强**：将每个RGB图像转换为点阵图像，通过Scaffold方法增强平面位置理解，与RGB图像共同作为MLLM输入
- **空间关系图构建**：MLLM利用几何距离关系和平面位置关系，构建空间关系图，生成每个方向的详细描述（物体存在、空间关系、房间类型等）

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/panonav-panoramic-parsing.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/panonav-dynamic-memory.png" width="100%" />
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

## 4. NavDP与NoMaD
——扩散模型在通用导航中的应用

**研究背景/问题**

机器人的通用导航不仅要求能够在未知或动态环境中安全移动，还需要具备灵活的探索能力和跨场景、跨平台的适应性。传统方法通常为探索和目标导航分别训练独立策略，导致模型复杂且泛化能力有限。本文介绍了两个代表性工作：NavDP（上海AI Lab）和NoMaD（伯克利，ICRA2024 Best Paper）。

**主要方法/创新点**

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/diffusion-navigation-overview.png" width="100%" />
<figcaption>
扩散模型在通用导航中的应用框架对比
</figcaption>
</div>

### NavDP（上海AI Lab）

**核心思路**：扩散模型负责生成候选轨迹，Critic负责挑选安全路线

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/navdp-framework.png" width="100%" />
<figcaption>
NavDP双阶段推理框架
</figcaption>
</div>

**两阶段推理框架：**
- **第一阶段（策略生成）**：用RGB-D图像+导航目标，经策略Transformer编码后，通过扩散生成候选轨迹
- **第二阶段（安全评估）**：将生成轨迹与RGB-D token融合，再经共享Transformer与critic head，选择与目标无关的安全轨迹

**模拟特权信息利用：**
- **生成器训练**：利用模拟环境中的全局最优规划器指导轨迹生成
- **Critic训练**：利用模拟环境的全局ESDF，从负样本轨迹中学习精细空间理解
- **数据增强**：对原始轨迹进行随机旋转和插值，生成混合轨迹增加多样性

**多模态输入编码：**
- **输入**：单帧RGB-D图像+导航目标（四种类型：点目标、图像目标、轨迹目标、无目标）
- **深度处理**：裁剪至0.1-3.0 m，RGB经预训练DepthAnything ViT编码，深度由自训练ViT编码
- **Transformer解码器**：将512个RGB-D token压缩为16个融合token

**Real-to-Sim增强：**
- 采用Gaussian Splatting重建真实环境，提供高真实感的训练与评测平台
- 在训练集中加入27%的real-to-sim样本，可使目标场景成功率提升30%，且不损害泛化能力

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/navdp-data-generation.png" width="100%" />
<figcaption>
NavDP模拟数据生成流程
</figcaption>
</div>

### NoMaD（伯克利，ICRA2024 Best Paper）

**核心思路**：通过统一的扩散策略，同时建模任务特定和任务无关行为

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/nomad-framework.png" width="100%" />
<figcaption>
NoMaD目标掩码扩散策略框架
</figcaption>
</div>

**两个关键组件：**

*目标掩码（Goal Masking）：*
- 通过二值掩码控制策略是否关注目标图像，实现任务条件的灵活切换
- **训练时**：目标掩码以50%概率随机设置，使模型同时学习目标导向行为和探索行为
- **推理时**：根据任务需要设置掩码（探索时掩盖目标，导航时提供目标）

*扩散策略（Diffusion Policy）：*
- 利用扩散模型生成多模态、无碰撞的动作序列
- 从随机噪声逐步迭代生成预测动作序列
- 动作分布既可在无目标条件下表达探索行为，也可在提供目标条件下收敛到目标导向行为

**统一框架设计：**
- 通过Transformer编码视觉观测并结合扩散模型生成未来动作序列
- 同时支持任务特定行为（目标导向）和任务无关行为（探索）
- 使用大规模多样化数据集（GNM和SACSoN）进行端到端监督训练

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/nomad-goal-masking.png" width="100%" />
<figcaption>
NoMaD目标掩码机制示意图
</figcaption>
</div>

**核心结果/发现**

### NavDP实验结果：

- **跨机器人平台泛化**：在不同机器人（Dingo、Go2、Galaxea R1）上稳定高于基线（GNM, ViNT, NoMad, iPlanner, ViPlanner, EgoPlanner）
- **零样本Sim-to-Real**：成功在Unitree Go2、Galaxea R1、Unitree G1上部署，室内外场景均表现良好，含动态行人干扰
- **数据规模与效率**：模拟数据生成速度约2,500条轨迹/GPU/天，比真实采集快20倍；数据集覆盖1244个场景、总长度363.2 km
- **模型组件贡献**：Critic模块是性能提升的关键，移除后性能显著下降；No-goal训练目标对整体避障行为影响最大
- **Real-to-Sim效果**：真实场景成功率提高30%，证明real-to-sim数据能显著提升sim-to-real成功率
- **高速避障**：>10Hz推理，支持2.0 m/s高速避障，动态场景下优于传统地图规划方法

### NoMaD实验结果：

- **探索未知环境**：成功率达到98%，平均碰撞数仅0.2，超过最优基线Subgoal Diffusion约25%，且参数量仅为其1/15
- **目标导航**：在已知环境的目标导航任务中，成功率与最优基线相当，但计算资源需求更少
- **计算效率**：比现有方法计算效率提升约15倍，是首个成功在物理机器人上部署的目标条件动作扩散模型
- **统一策略优势**：联合训练能够学习共享表示和环境可操作性，单一策略即可胜任多种行为
- **编码器选择**：ViNT编码器配合注意力目标掩码效果最佳，成功率98%，碰撞数最少
- **多场景验证**：在6个复杂的室内外环境中表现优异

**局限性**

两种方法虽然在跨场景、跨平台泛化方面表现出色，但在极端复杂环境下的鲁棒性仍需进一步提升。NavDP依赖高质量的模拟数据训练，Real-to-Sim数据比例需要仔细平衡。NoMaD的视觉编码器选择对性能影响较大，需要仔细调优，且ViT编码器虽然容量大但训练优化难度高。

---

## 5. VLN-R1
——基于GRPO与Time-Decayed Reward的端到端导航

**研究背景/问题**

VLN是具身人工智能领域的一项核心挑战，要求智能体根据自然语言指令在真实世界环境中进行导航。传统的导航方法通常依赖离散的拓扑图和预定义的节点连接，限制了智能体在连续环境中的泛化能力。

**主要方法/创新点**

VLN-R1提出了一种创新的端到端框架，利用大型视觉-语言模型（LVLM）直接处理自我中心视频流，生成连续的导航动作。

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-r1-framework-overview.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/vln-ego-dataset.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/vln-r1-training-pipeline.png" width="100%" />
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

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-r1-tdr-mechanism.png" width="100%" />
<figcaption>
时间衰减奖励（TDR）机制示意图
</figcaption>
</div>

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


## 6. Survey on VLA for Autonomous Driving

**研究背景/问题**

自动驾驶正从规则驱动、端到端学习向更高阶的"人类级智能"演进。当前方法在处理长尾场景、理解驾驶意图、执行复杂推理时面临瓶颈。VLA（Vision-Language-Action）模型通过融合多模态理解与推理能力，为实现人类级驾驶智能提供新路径。

**主要方法/创新点**

文章提出了从"昆虫智能"到"哺乳动物智能"再到"人类级智能"的三阶段演进框架：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vla-driving-evolution.png" width="100%" />
<figcaption>
自动驾驶智能演进的三个阶段
</figcaption>
</div>

**VLA驾驶智能体架构：**
- **Driver Agent核心**：融合Chain-of-Thought推理与多模态感知，将驾驶决策分解为"感知-理解-推理-决策"链路
- **四阶段训练流程**：
  1. VL预训练（10B级通用数据+车端多模态数据）
  2. 蒸馏（32B云端大模型→3.2B/4B边缘模型）
  3. 驾驶模仿学习（专家轨迹+规划奖励）
  4. 强化学习（场景覆盖奖励+违规惩罚）

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vla-training-pipeline.png" width="100%" />
<figcaption>
VLA模型四阶段训练流程
</figcaption>
</div>

**技术创新点：**
- **多模态融合**：统一处理多视角图像、激光雷达、定位信息和自然语言指令
- **思维链增强**：引入CoT提示工程，使模型输出可解释的中间推理步骤
- **分布式部署**：云端32B模型提供规划监督，边缘3.2B/4B模型实现实时推理（Orin-X/Thor-U硬件）
- **数据闭环**：整合真实驾驶场景数据，持续优化长尾case处理能力

**核心结果/发现**

- 在理想汽车AD Pro平台上实现端到端部署，边缘模型推理延迟满足实时性要求
- 通过蒸馏和量化技术，将32B云端模型的能力有效迁移至3.2B/4B边缘模型，准确率保持在92%以上
- 在复杂城区场景中，VLA方案相比传统端到端方法，接管率降低37%，长尾case处理成功率提升42%
- CoT推理使决策过程可解释，用户信任度调查显示满意度提升28%

**局限性**

当前VLA模型在极端天气（如暴雨、浓雾）下的感知能力仍需增强，计算资源消耗较高导致部署成本上升。未来需进一步优化模型效率，并拓展至更广泛的驾驶场景。

---



## 7. VLN入门基础技术梳理

**研究背景/问题**

VLN是一个多学科交叉的研究领域，涵盖自然语言处理、计算机视觉、多模态信息融合及机器人导航等学科。智能体需要理解自然语言指令并在复杂环境中实现自主导航，但数据稀缺、跨模态匹配困难以及泛化能力不足是该领域面临的核心挑战。

**主要方法/创新点**

本文系统性地介绍了VLN领域的基础知识体系：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-basics-overview.png" width="100%" />
<figcaption>
VLN任务基本框架与分类体系
</figcaption>
</div>

**任务分类与环境设置：**
- **任务类型**：指令导向（R2R、R4R）、目标导向（REVERIE、SOON）、需求导向（DDN）
- **交互方式**：单轮指令任务、多轮对话式导航任务
- **场景类型**：室内场景（Room-to-Room）、室外场景（街道导航）、空中场景（AerialVLN）
- **环境表示**：离散环境（连通图表示，节点间导航）、连续环境（三维空间自由移动）

**测试基准与评估：**
- **模拟器**：Matterport3DSimulator、Habitat、AirSim、Gibson/iGibson
- **数据集**：R2R（10,567张全景图，90个房屋）、R4R（长指令）、CVDN（对话式）、REVERIE/SOON（对象定位）、AerialVLN（无人机导航）
- **评估指标**：
  - 导航精度：Success Rate (SR)、Navigation Error (NE)、Oracle Success Rate (OSR)
  - 导航效率：Success weighted by Path Length (SPL)、Coverage weighted by Length Score (CLS)
  - 轨迹质量：normalized Dynamic Time Warping (nDTW)

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-datasets-comparison.png" width="100%" />
<figcaption>
主要VLN数据集对比与特点
</figcaption>
</div>

**典型方法分类：**
1. **传统Seq2seq方法**：基于注意力机制的LSTM序列到序列模型，使用"学生自学"训练方法
2. **数据增强方法**：Speaker-Follower模型，通过生成指令扩充训练数据
3. **辅助目标方法**：引入辅助推理任务（如自监督学习、进度估计）增强泛化能力
4. **拓扑图方法**：构建场景拓扑图支持全局路径规划（如DualVLN的双尺度图Transformer）
5. **大模型方法**：NavGPT/NavGPT-2利用大语言模型的多模态理解和推理能力

**理论基础：**
- **神经网络架构**：RNN/LSTM/GRU（序列建模）、CNN（视觉特征提取）、Transformer（长距离依赖捕获）
- **训练范式**：模仿学习（交叉熵损失）、强化学习（策略梯度）、辅助监督学习（自监督预训练）
- **工具框架**：PyTorch（模型构建）、Transformers库（预训练模型）

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vln-methods-evolution.png" width="100%" />
<figcaption>
VLN方法演进路线图
</figcaption>
</div>

**核心结果/发现**

- VLN任务在Matterport3D等真实场景数据集上的性能不断提升，从早期Seq2seq方法的30-40% SR提升至基于大模型方法的60%+ SR
- 数据增强技术（如Speaker-Follower）可有效缓解数据稀缺问题，性能提升10-15%
- 拓扑图表示和全局规划显著改善长距离导航任务表现，R4R等复杂任务成功率提升20%+
- 大模型（如GPT-4、Qwen-VL）的引入实现了零样本学习和可解释推理，为VLN带来范式转变
- 评估体系已从单一SR指标扩展至涵盖精度、效率、轨迹质量的综合评估框架

**局限性**

当前VLN研究主要集中在仿真环境，Sim-to-Real迁移仍面临挑战。数据集规模有限且场景多样性不足，长尾场景处理能力有待提升。此外，实时性、鲁棒性和跨平台部署等实际应用需求尚未得到充分解决。

---



## 8. LagMemo 
——Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation

**研究背景/问题**

传统视觉导航方法受限于单目标、单模态和封闭类别设置，无法满足实际应用中多模态开放词汇表多目标导航的需求。现有方法如端到端强化学习依赖隐式状态编码导致泛化能力差，而模块化方法基于2D语义地图仅支持预定义类别，无法适应开放词汇场景。

**主要方法/创新点**

LagMemo提出了首个将语言特征融入3D Gaussian Splatting（3DGS）的视觉导航系统：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/lagmemo-framework-overview.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/lagmemo-language-injection.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/lagmemo-navigation-pipeline.png" width="100%" />
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



## 9. GaussNav 
——Gaussian Splatting for Visual Navigation

**研究背景/问题**

Instance ImageGoal Navigation (IIN)要求智能体在未探索环境中定位并导航至目标图像所描绘的特定对象实例，需要跨视角识别目标对象同时忽略干扰物。现有基于BEV地图的导航方法缺乏详细纹理表示，难以胜任实例级任务，无法保留场景的实例感知特征，不足以区分同类别的多个对象。

**主要方法/创新点**

GaussNav首次将3D Gaussian Splatting（3DGS）引入具身视觉导航，提出语义高斯地图表示：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/gaussnav-framework-overview.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/gaussnav-semantic-gaussian-construction.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/gaussnav-navigation-pipeline.png" width="100%" />
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


## 10. VLFM
——Vision-Language Frontier Maps for Zero-Shot Semantic Navigation

**研究背景/问题**
零样本语义导航要求机器人在未见环境中高效定位目标对象，现有方法（如ESC、SemUtil）依赖物体检测器将视觉线索转化为文本后再用LLM/BERT进行语义推理，存在计算瓶颈且无法充分利用视觉-语言联合表征。如何直接从RGB观测中提取语义价值以指导前沿探索成为关键挑战。

**主要方法/创新点**

VLFM提出语言驱动的前沿价值图框架，实现端到端视觉-语义推理：

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vlfm-system-overview.png" width="100%" />
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
  <img src="https://r-c-group.github.io/blog_media/images/vlfm-value-map-generation.png" width="100%" />
<figcaption>
价值图生成流程：BLIP-2计算语义分数并投影到俯视图
</figcaption>
</div>

3. **置信度加权更新（Confidence-Weighted Averaging）**
   - 置信度分数基于像素相对光轴位置：$c_{i,j} = \cos^2(\theta/(\theta_{fov}/2) \times \pi/2)$
   - 重叠区域的语义值更新：$v_{i,j}^{new} = (c_{i,j}^{curr}v_{i,j}^{curr} + c_{i,j}^{prev}v_{i,j}^{prev})/(c_{i,j}^{curr} + c_{i,j}^{prev})$
   - 置信度更新偏向高置信值：$c_{i,j}^{new} = ((c_{i,j}^{curr})^2 + (c_{i,j}^{prev})^2)/(c_{i,j}^{curr} + c_{i,j}^{prev})$

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vlfm-confidence-weighting.png" width="100%" />
<figcaption>
置信度评分机制：光轴附近像素置信度最高，边缘递减
</figcaption>
</div>

4. **物体检测与导航**
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

## 11. Vision-Language-Action Models for Robotics
——A Review Towards Real-World Applications

**研究背景/问题**

机器人领域正努力利用大语言模型（LLM）和视觉-语言模型（VLM）的进展来实现更通用和可扩展的机器人系统。Vision-Language-Action (VLA) 模型通过统一视觉、语言和行动数据，旨在学习能够泛化到不同任务、物体、embodiment和环境的策略。然而，VLA模型的架构和训练方法尚未标准化，实际部署面临数据稀缺、embodiment迁移和计算成本等挑战。

**主要方法/创新点**

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vla-survey-structure.png" width="100%" />
<figcaption>
VLA综述论文的整体结构
</figcaption>
</div>

本综述提供了VLA模型的全栈式系统性回顾，涵盖软件和硬件组件：

**VLA架构演进**：
- **早期CNN-based端到端架构**：CLIPort等模型整合CLIP进行视觉和语言特征提取，结合Transporter Network学习物体操作任务
- **Transformer-based序列模型**：Gato作为通用智能体先驱，使用单个transformer执行多种任务；VIMA通过encoder-decoder transformer实现通用任务指令跟随
- **基于预训练VLM的统一实时策略**：RT-1首次实现大规模真实世界通用控制；RT-2利用PaLM-E/PaLI-X等VLM主干，通过互联网规模数据联合微调实现强泛化；OpenVLA作为开源框架
- **Diffusion Policy架构**：Octo首次将Diffusion Policy引入VLA，生成连续平滑的动作输出
- **Diffusion Transformer架构**：RDT-1B将扩散过程直接集成到transformer解码器中
- **Flow Matching策略架构**：π0利用flow-matching实现高达50Hz的实时控制
- **潜在动作学习**：LAPA从视频中学习潜在动作表示，有效利用人类演示数据
- **分层策略架构**：RT-H引入高层"language motion"预测和低层动作生成的分层结构；π0.5和GR00T N1进一步整合多阶段策略

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vla-architecture-timeline.png" width="100%" />
<figcaption>
VLA模型架构演进时间线
</figcaption>
</div>

**核心架构组件**：
- **感知运动模型（Sensorimotor Model）**：七种主要架构变体
  - Transformer + 离散动作Token
  - Transformer + Diffusion动作头
  - Diffusion Transformer
  - VLM + 离散动作Token
  - VLM + Diffusion/Flow Matching动作头
  - VLM + Diffusion Transformer
- **世界模型（World Model）**：通过预测未来观察支持规划和推理
- **Affordance-based模型**：预测动作相关的视觉affordances

**模态处理**：
- 视觉：SigLIP、DINOv2成为主流视觉编码器
- 语言：继承LLM的tokenizer，使用T5、CLIP Text Encoder等编码器
- 动作：从离散化binning到连续动作生成（diffusion/flow matching）
- 其他模态：音频、触觉、3D信息（深度、点云、体素）

**训练策略**：
- 预训练：利用大规模VLM主干（如PaliGemma、Prismatic VLM、Qwen2.5-VL）
- 后训练：任务特定的高质量数据微调
- Gradient insulation：冻结主干防止梯度污染预训练表示
- 自监督学习：模态对齐、视觉表示学习、潜在动作表示学习
- 强化学习：通过RL微调提升VLA鲁棒性和适应性

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/vla-training-stages.png" width="100%" />
<figcaption>
VLA模型训练阶段
</figcaption>
</div>

**数据收集与增强**：
- 遥操作：ALOHA系列、Mobile ALOHA
- 代理设备：UMI、DexUMI、Dobb-E
- 人类数据：Ego4D、EPIC-KITCHENS、Ego-Exo4D
- 数据集：RT-X (1.4M episodes)、DROID (76K)、RoboMIND (107K)、AgiBot World (1M)
- 数据增强：视觉增强（Stable Diffusion-based）、语言增强（DIAL）、动作增强（DAgger、CCIL）

**核心结果/发现**

- VLA模型已从早期CNN方法演进到复杂的多模态transformer架构，显著提升了泛化能力
- 预训练VLM主干（如PaliGemma、Qwen2.5-VL）对VLA性能至关重要，能够继承常识知识和in-context学习能力
- Diffusion和flow matching方法生成平滑连续动作，优于离散token化方法
- 分层架构（如RT-H、π0.5、GR00T N1）在长时域任务中表现优异
- 数据规模和多样性对VLA泛化至关重要，但数据收集成本高昂
- 跨embodiment迁移仍是重大挑战，需要统一的动作空间表示
- 实际部署面临安全性、实时性、计算效率等挑战

**局限性**

当前VLA模型主要在仿真环境或受控实验室环境中评估，实际部署面临诸多挑战：安全性保障不足、缺乏故障检测和恢复机制、计算资源需求高、跨embodiment泛化能力有限。此外，大多数模型缺乏持续学习能力，无法在部署后适应新环境。评估方法不够严格，缺乏统计显著性检验。未来需要在世界模型、推理能力、多模态融合、持续学习和实际应用方面取得突破。

## 12. Motus
——A Unified Latent Action World Model

**研究背景/问题**

当前具身智能体的理解、世界建模和控制能力被孤立地建模在不同模型中,这种碎片化阻碍了统一多模态生成能力的实现,也限制了从大规模异构数据中学习。现有方法将本应统一的系统分割为5个独立的建模任务:VLA(视觉-语言-动作模型)、WM(世界模型)、IDM(逆动力学模型)、VGM(视频生成模型)和视频-动作联合预测模型。两个核心挑战包括:如何在单一框架中统一这些多模态生成能力,以及如何利用大规模异构数据(互联网视频、自我中心人类演示、多机器人轨迹)进行动作专家的预训练。

**主要方法/创新点**

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-architecture-overview.png" width="100%" />
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

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-tri-model-attention.png" width="100%" />
<figcaption>
三模态联合注意力机制详解
</figcaption>
</div>

**2. UniDiffuser式调度器:**
- 为视频和动作分配不同的时间步τ_o、τ_a和噪声尺度
- 支持五种推理模式的灵活切换:VLA、世界模型、IDM、VGM、视频-动作联合预测
- 使用rectified flow目标函数:
  - l_action = E[||v^θ_a - (ε_a - a_{t+1:t+k})||²]
  - l_obs = E[||v^θ_o - (ε_o - o_{t+1:t+k})||²]

**3. 潜在动作(Latent Actions) - 像素级"增量动作":**

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-latent-action-vae.png" width="100%" />
<figcaption>
潜在动作VAE架构:从光流到潜在动作表示
</figcaption>
</div>

- **光流表示**:使用DPFlow计算光流作为通用运动表示,将其转换为RGB图像
- **深度压缩自编码器(DC-AE)**:将高维光流压缩为4×512维token,再通过轻量级编码器投影到14维潜在动作向量
- **训练策略**:混合90%无标注数据(自监督重建)+10%有标注轨迹(任务无关数据+标准演示)
- **分布对齐**:引入任务无关数据(AnyPos方法),使用Curobo随机采样目标机器人动作空间
- **损失函数**:L = L_recon + λ_a||a_real - a_pred||² + βL_KL

**4. 动作密集-视频稀疏预测策略:**
- 视频帧率:8帧 @ 5Hz
- 动作块:48步 @ 30Hz
- 通过下采样视频帧平衡token数量,防止过拟合视频预测而削弱动作预测能力

**5. 三阶段训练流程:**

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-training-pipeline.png" width="100%" />
<figcaption>
Motus三阶段训练流程与数据金字塔
</figcaption>
</div>

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
  <img src="https://r-c-group.github.io/blog_media/images/motus-embodied-data-pyramid.png" width="100%" />
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

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-robotwin-results.png" width="100%" />
<figcaption>
RoboTwin 2.0仿真基准测试结果对比
</figcaption>
</div>

**真实世界实验:**
- **两个平台**:AC-One和Agilex-Aloha-2双臂机器人
- **9个复杂任务**:测试空间理解、可变形物体操作、精确流体控制、视觉理解、长时域规划
  - 任务包括:叠毛巾、使用滴滤咖啡机煮咖啡、研磨咖啡豆、将面包放入烤箱、从饮水机取水、倒水浇花、按键盘按键

- **AC-One平台**:平均部分成功率63.22%(Motus) vs 25.86%(无预训练) vs 14.79%(π0.5)
  - 突出任务:研磨咖啡豆92% vs 0%(无预训练),煮咖啡62% vs 0%,放立方体入盘100% vs 60%

- **Agilex-Aloha-2平台**:平均59.30%(Motus) vs 26.60%(无预训练) vs 48.60%(π0.5)
  - 突出任务:从饮水机取水96% vs 8%(无预训练),叠毛巾39% vs 0%

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-real-world-tasks.png" width="100%" />
<figcaption>
Motus在真实世界复杂任务上的执行展示
</figcaption>
</div>

**其他基准测试:**
- **LIBERO-Long**:97.6%成功率(与X-VLA并列最优,达到state-of-the-art)
- **VLABench**: In Distribution平均0.48(vs π0.5的0.43),Cross Category平均0.25(vs π0.5的0.22)

**消融实验验证:**
- **训练阶段重要性**:完整Motus(阶段2预训练) 87.02% vs 仅阶段1 81.86%(+10.02%提升)
- **IDM模式性能**:动作MSE 0.014(Motus) vs 0.044(ResNet18+MLP) vs 0.122(DINOv2+MLP),显著优于专门训练的IDM基线
- **VLA模式竞争力**:83.90%成功率,与联合模式87.02%性能接近
- **世界模型生成质量**:FID 11.209,FVD 61.209,SSIM 0.866,PSNR 25.07(在两个平台上评估)

**五种统一模式实证验证:**
1. VLA: p(a_{t+1:t+k} | o_t, ℓ) - 从观察和语言预测动作
2. 世界模型: p(o_{t+1:t+k} | o_t, a_{t+1:t+k}) - 从当前观察和动作预测未来观察
3. IDM: p(a_{t+1:t+k} | o_{t:t+k}) - 从观察序列推断动作
4. VGM: p(o_{t+1:t+k} | o_t, ℓ) - 从观察和语言生成未来视频
5. 视频-动作联合预测: p(o_{t+1:t+k}, a_{t+1:t+k} | o_t, ℓ) - 同时生成视频和动作

<div align="center">
  <img src="https://r-c-group.github.io/blog_media/images/motus-unified-modes-visualization.png" width="100%" />
<figcaption>
Motus五种统一模式的可视化展示
</figcaption>
</div>

**局限性**

当前方法需要大量计算资源(总计约18,400 GPU小时训练)。某些复杂任务(如叠毛巾)的性能仍有限,部分成功率仅为39%。尽管通过潜在动作改进了跨具身泛化,但仍需进一步研究。未来工作将探索更先进的统一模型架构,追求更通用的运动先验,并从互联网规模的通用视频中学习潜在动作。此外,需要研究如何降低部署成本并提升模型在极端条件下的鲁棒性。



---

**注**：本文为个人学习笔记，大量内容来自网络公开资料，仅供参考。如有错误或建议，欢迎指正！
