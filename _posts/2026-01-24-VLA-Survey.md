---
layout: post
title: "Vision-Language-Action (VLA) 综述"
date:   2026-01-24
tags: [VLA, VLM, Robotics, Manipulation, Deep Learning]
comments: true
author: Tingde Liu
toc: true
excerpt: "视觉-语言-行动（VLA）模型是机器人领域的前沿研究方向，统一了感知、推理和控制能力，使机器人能够基于自然语言指令执行复杂操作任务。本文对VLA的基本概念、技术架构、主要挑战和最新进展进行全面综述。"
---

# 引言

视觉-语言-行动（Vision-Language-Action, VLA）模型代表了机器人技术的范式转变，将视觉感知、自然语言理解和动作控制统一在单一的端到端学习框架中。VLA模型使机器人能够直接从自然语言指令和视觉观察中学习操作策略，无需显式的中间表示或分离的感知-规划-控制模块。

自2023年Google DeepMind推出RT-2以来，VLA模型在机器人操作任务中展现出强大的泛化能力和零样本学习潜力。这一技术路线继承了大规模视觉-语言模型（VLM）在跨模态理解方面的优势，并将其扩展到具身智能领域，使机器人能够像理解图像和文本一样"理解"物理世界中的交互任务。

VLA模型在服务机器人、工业自动化、智能制造等领域有着广泛的应用前景。本文旨在系统梳理VLA领域的研究进展，为学习和研究VLA提供全面参考。

# VLA基本概述

## 什么是VLA？

VLA（Vision-Language-Action）模型是一类端到端的多模态学习模型，能够直接从视觉输入（相机图像）和语言指令中预测机器人的控制动作。与传统的模块化机器人系统不同，VLA将感知、推理和控制统一在单一的神经网络架构中。

<div align="center">
  <img src="https://openvla.github.io/assets/openvla_approach.png" width="80%" />
<figcaption>
VLA模型架构示意图（来源：OpenVLA）
</figcaption>
</div>

### VLA的核心特征

**1. 端到端学习**：直接从原始传感器数据到低层控制信号的映射，无需手工设计的中间表示。

**2. 多模态融合**：统一处理视觉、语言和动作三种模态，实现跨模态的语义对齐。

**3. 指令驱动**：通过自然语言指令指定任务目标，支持灵活的任务切换和零样本泛化。

**4. 大规模预训练**：继承视觉-语言模型的预训练知识，并在机器人演示数据上微调。

### VLA的发展里程碑

- **2023年7月**：Google DeepMind发布RT-2，首次将VLM成功转化为VLA模型，开创了这一研究方向
- **2024年6月**：斯坦福大学发布OpenVLA，首个7B参数的开源VLA模型，在多项操作任务上超越RT-2
- **2025年**：出现多个轻量化VLA模型（如SmolVLA 450M参数），使VLA模型能够在边缘设备上实时运行
- **2026年**：VLA研究进入快速发展期，ICLR 2026收录164篇VLA相关论文，研究重点转向推理能力、跨机器人泛化和实际部署

## VLA的三个核心要素

一个完整的VLA系统包含三个核心组成部分：

### 1. 视觉编码器（Vision Encoder）

**功能**：从相机图像中提取视觉特征表示。

**主流架构**：
- **DINOv2**：自监督学习的视觉编码器，提供强大的几何和空间先验
- **CLIP/SigLIP**：对比学习的视觉-语言编码器，实现视觉与语言的语义对齐
- **混合编码器**：结合多个视觉编码器的优势（如OpenVLA同时使用DINOv2和CLIP）

**发展趋势**：从单一视觉骨干网络转向多视角、多分辨率的视觉编码，以提升对复杂场景的感知能力。

### 2. 语言-行动骨干网络（Language-Action Backbone）

**功能**：作为VLA模型的核心"大脑"，融合视觉和语言信息，并进行高层推理和决策。

**主流架构**：
- **Transformer-based LLM**：如Llama-2、Phi-3等预训练语言模型
- **跨模态融合机制**：通过交叉注意力机制融合视觉和语言特征
- **世界模型**：隐式学习环境动态和物理规律

**关键能力**：
- 理解复杂的自然语言指令
- 进行常识推理和任务分解
- 维护长时序的任务上下文

### 3. 动作解码器（Action Decoder）

**功能**：将模型的高层决策转化为具体的机器人控制信号。

**主流范式**：

**离散动作建模**：
- 将连续的机器人动作量化为离散token
- 使用自回归或分类头预测动作序列
- 优势：可直接利用语言模型的生成能力

**连续动作建模（扩散模型）**：
- 使用扩散过程建模动作分布
- 能够捕获多模态的动作可能性
- 在精细操作任务中表现更优

**动作分块（Action Chunking）**：
- 一次预测多个时间步的动作序列
- 提升长时序任务的执行流畅性
- 2025年的FAST tokenizer将动作块压缩，实现15倍推理加速

## VLA系统的基本组成

VLA系统通常由以下模块构成，呈现出明显的**端到端统一建模**趋势：

### 1. 感知模块（Perception Module）—— 多模态视觉编码

**功能**：从环境中提取视觉和语言的观察信息。

**架构演进**：
- **早期**：单一RGB图像编码
- **现在**：多视角、深度图像、点云等多模态输入
- **趋势**：引入触觉、力传感等非视觉模态

**关键技术**：
- 视觉-语言对齐预训练（如CLIP）
- 自监督视觉表示学习（如DINOv2）
- 时序信息编码（处理视频输入）

### 2. 推理模块（Reasoning Module）—— 语言引导的任务规划

**功能**：理解任务指令，进行高层次的任务分解和规划。

**实现方式**：
- **隐式推理**：直接在VLA模型内部进行端到端推理
- **显式推理**：利用大语言模型进行链式思考（CoT）和子目标分解

**推理能力**：
- 常识推理：理解物理世界的基本规律
- 因果推理：预测动作的后果
- 组合泛化：处理未见过的物体组合和任务

### 3. 控制模块（Control Module）—— 精确的动作执行

**功能**：将高层决策转化为底层电机控制信号。

**控制范式**：
- **开环控制**：直接预测动作序列
- **闭环控制**：基于视觉反馈持续调整动作
- **分层控制**：高层规划 + 低层控制器（如MPC）

**控制频率**：
- 传统VLA：10Hz左右
- 优化后的VLA（如OFT）：30Hz+，支持双臂高频控制

## VLA与传统机器人控制的区别

VLA代表了机器人控制范式的根本性转变：

### 从模块化到端到端

**传统方法**：
- 感知模块：物体检测、位姿估计
- 规划模块：轨迹规划、路径优化
- 控制模块：PID控制、力控制
- 各模块独立设计和优化

**VLA方法**：
- 单一神经网络直接从像素到控制信号
- 联合优化所有组件
- 减少了模块间的信息损失

### 从任务特定到通用能力

**传统方法**：
- 为每个任务设计专门的控制策略
- 难以泛化到新任务和新环境
- 需要大量的工程调试

**VLA方法**：
- 通过自然语言指令指定任务
- 零样本或少样本泛化到新任务
- 利用预训练知识处理未见场景

### 从显式建模到隐式学习

**传统方法**：
- 需要精确的环境模型和物体模型
- 依赖手工设计的特征和规则
- 对模型误差敏感

**VLA方法**：
- 从数据中隐式学习环境动力学
- 自动发现任务相关的特征
- 对感知噪声和模型不确定性更鲁棒

### 从工程驱动到数据驱动

**传统方法**：
- 依赖专家知识和人工调参
- 开发周期长，部署成本高
- 难以处理长尾场景

**VLA方法**：
- 基于大规模演示数据学习
- 通过数据增强提升泛化能力
- 持续学习和在线适应

---

## VLA的主要挑战

本节采用**挑战驱动分类法**（Challenge-Centric Taxonomy），围绕VLA研究的五大核心瓶颈进行组织，而非传统的架构或任务分类。这种分类方式更好地反映了当前VLA领域亟需突破的关键问题。

### 挑战1：多模态对齐与物理世界建模

**核心问题**：弥合语义感知与物理交互之间的鸿沟，实现从2D图像到时空表示的跨越，并发展动态预测世界模型。

VLA模型需要将视觉-语言模型的语义理解能力转化为对物理世界的准确建模，这涉及三个层次的挑战：

#### 1.1 语义到物理的接地（Semantic-to-Physical Grounding）

**问题描述**：如何将抽象的语言描述（如"红色的杯子"）映射到物理世界中的具体对象和可执行动作。

**关键技术**：
- **对比学习**：通过对比学习将视频的视觉潜在空间与可执行的机器人动作空间对齐（如CLAP）
- **显式视觉线索**：通过边界框等明确视觉提示解决指代歧义（如Point-VLA）
- **3D可供性图**：生成三维可供性图作为强中间表示（如VoxPoser）

**代表性工作**：
- **CLAP**：对齐视频视觉潜在空间与机器人动作空间
- **Point-VLA**：通过显式视觉提示解决指代模糊
- **VoxPoser**：生成3D可供性图作为中间表示

#### 1.2 2D到3D的空间表示（2D-to-3D Spatial Representations）

**问题描述**：从2D图像输入推断3D空间结构，理解深度、遮挡和空间关系。

**关键技术**：
- **3D占据网络**：学习场景的体素化占据表示（如OccLLaMA）
- **轨迹追踪**：通过追踪3D空间中的轨迹理解运动（如TraceVLA）
- **多视角融合**：整合多个相机视角构建完整的3D场景理解

**代表性工作**：
- **OccLLaMA**：基于占据网络的3D场景理解
- **TraceVLA**：3D轨迹追踪与预测

#### 1.3 动态世界建模（Dynamic World Modeling）

**问题描述**：预测物体交互的动态变化，理解物理规律和因果关系。

**关键技术**：
- **视频预测模型**：通过视频数据学习物理动力学（如mimic-video）
- **动作条件世界模型**：建模动作对环境的影响（如VideoVLA）
- **隐式物理引擎**：在神经网络中隐式编码物理规律

**代表性工作**：
- **mimic-video**：从视频中学习物理动力学
- **VideoVLA**：视频条件的世界模型

---

### 挑战2：指令跟随、规划与鲁棒实时执行

**核心问题**：解析复杂指令，进行分层任务分解，实现错误检测与自主恢复，并保证实时计算效率。

#### 2.1 复杂指令解析与理解

**问题描述**：理解多样化、组合式的自然语言指令，处理模糊性和不完整性。

**关键技术**：
- **大语言模型推理**：利用LLM进行语义理解和常识推理
- **上下文学习**：通过少样本示例快速适应新指令模式
- **多轮对话**：支持交互式指令澄清

#### 2.2 分层任务分解与规划

**问题描述**：将长时序复杂任务分解为可执行的子任务序列，并进行高层规划。

**关键技术**：
- **具身数字孪生**：使用基于物理的交互式数字孪生作为具身世界模型（如EToT）
- **专家混合（MoE）**：通过专家混合层将长时序任务分解为专门技能（如MoE-DP）
- **分层强化学习**：在不同抽象层次上学习策略

**代表性工作**：
- **EToT**：利用物理交互式数字孪生
- **MoE-DP**：混合专家进行任务分解
- **WALL-OSS**：大规模分层规划系统

#### 2.3 错误检测与自主恢复

**问题描述**：实时检测执行失败，并自主生成恢复策略。

**关键技术**：
- **失败预测**：基于视觉反馈预测潜在失败
- **重规划机制**：动态调整执行计划
- **链式思考推理**：通过CoT进行故障诊断（如CoT-VLA）

**代表性工作**：
- **Fast-ThinkAct**：实时推理与失败恢复
- **CoT-VLA**：链式思考增强的VLA

#### 2.4 实时性与计算效率

**问题描述**：满足高频控制的实时性要求（30Hz+），同时保证推理准确性。

**关键技术**：
- **模型压缩**：量化、剪枝、蒸馏
- **推理优化**：动作token压缩、批处理优化
- **边缘部署**：轻量化架构设计

---

### 挑战3：从泛化到持续适应

**核心问题**：实现开放世界泛化，支持持续学习与增量技能获取，完成sim-to-real迁移，并启用在线强化学习。

#### 3.1 开放世界泛化（Open-World Generalization）

**问题描述**：在未见环境、未见物体和未见任务组合上的零样本或少样本泛化。

**关键技术**：
- **空间接地预训练**：结合空间引导的动作后训练（如InternVLA-M1）
- **大规模多样化数据**：在海量异构数据上预训练
- **元学习**：学习快速适应新任务的能力

**代表性工作**：
- **InternVLA-M1**：空间接地预训练 + 空间引导动作后训练
- **ProphRL**：统一的动作条件世界模型

#### 3.2 持续学习与增量技能获取

**问题描述**：在不遗忘旧技能的前提下持续学习新技能。

**关键技术**：
- **经验回放**：保留旧任务数据防止遗忘
- **弹性权重固化**：保护重要参数
- **模块化技能库**：独立学习和组合技能模块

#### 3.3 Sim-to-Real迁移

**问题描述**：缩小仿真训练与真实部署之间的性能差距。

**关键技术**：
- **域随机化**：增强模拟环境的多样性
- **高效适应**：快速在真实环境中微调（如AdaWorld）
- **鲁棒视觉表示**：学习对外观变化不敏感的特征

**代表性工作**：
- **AdaWorld**：高效适应与动作迁移到新环境

#### 3.4 在线强化学习

**问题描述**：通过与环境交互自主学习和改进策略。

**关键技术**：
- **离线到在线**：先离线预训练再在线微调
- **安全探索**：在保证安全的前提下探索
- **奖励学习**：从人类反馈中学习奖励函数

---

### 挑战4：安全性、可解释性与可靠交互

**核心问题**：确保可靠性保证，提升可解释性，实现可信的人机交互，并满足安全约束。

#### 4.1 可靠性保证

**问题描述**：保证VLA系统在各种情况下的稳定性和可预测性。

**关键技术**：
- **形式化验证**：数学证明系统满足安全属性
- **冗余机制**：多层次的安全保障
- **失效安全**：检测异常并进入安全状态

#### 4.2 可解释性与可信赖性

**问题描述**：使VLA的决策过程透明可理解，增强用户信任。

**关键技术**：
- **可编辑推理链**：用户可通过语言修正的逐步推理（如ECoT）
- **注意力可视化**：展示模型关注的视觉区域
- **反事实解释**：说明为何选择某个动作而非其他

**代表性工作**：
- **ECoT**：可编辑的链式思考推理
- **RationalVLA**：可学习的拒绝token

#### 4.3 安全约束与主动拒绝

**问题描述**：识别并拒绝不安全或无效的指令，主动避免危险行为。

**关键技术**：
- **控制屏障函数**：插件式安全约束层（如VLSA-AEGIS）
- **可学习拒绝**：训练模型识别并拒绝危险指令（如RationalVLA）
- **碰撞避免**：实时预测和避免碰撞

**代表性工作**：
- **VLSA-AEGIS**：基于控制屏障函数的即插即用安全层
- **RationalVLA**：学习拒绝不安全/无效指令

#### 4.4 人机协作交互

**问题描述**：与人类自然、高效地协作完成任务。

**关键技术**：
- **意图识别**：理解人类的隐含意图
- **主动询问**：在不确定时向人类请求帮助
- **从反馈学习**：从人类纠正中快速学习

---

### 挑战5：数据构建与基准测试标准

**核心问题**：管理多源异构数据整合，建立标准化评测基准。

#### 5.1 多源异构数据整合

**问题描述**：有效整合来自不同机器人平台、不同任务、不同采集方式的数据。

**关键技术**：
- **统一数据格式**：设计通用的数据表示标准
- **大规模数据聚合**：整合海量机器人轨迹（如WALL-OSS超过10000小时）
- **数据质量控制**：筛选和清洗低质量数据

**代表性工作**：
- **WALL-OSS**：聚合超过10000小时的自收集机器人轨迹
- **Open X-Embodiment**：跨机器人数据集标准

#### 5.2 评测基准与标准化指标

**问题描述**：建立公平、全面、标准化的VLA评测体系。

**关键技术**：
- **终身学习基准**：首个终身机器人基准与标准化指标（如LIBERO）
- **真实场景转仿真**：通过神经重建将真实视频转化为交互仿真环境（如PolaRiS）
- **多维度评估**：成功率、泛化能力、鲁棒性、效率等

**代表性工作**：
- **LIBERO**：首个终身机器人学习基准
- **PolaRiS**：将视频扫描转化为交互式仿真环境
- **SIMPLER**：标准化VLA评测平台

#### 5.3 数据高效学习

**问题描述**：在有限数据下实现高性能，降低数据收集成本。

**关键技术**：
- **数据增强**：自动生成多样化训练样本
- **主动学习**：选择最有价值的数据进行标注
- **迁移学习**：利用预训练知识减少对任务特定数据的需求

## VLA研究发展趋势

<div align="center">
  <img src="https://vla-survey.github.io/static/images/timeline.png" width="100%" />
<figcaption>
VLA研究时间线（参考自VLA Survey）
</figcaption>
</div>

从整体发展脉络来看，VLA研究经历了从模块化系统到端到端模型的重要转变。

### 1. 早期阶段：基于VLM的机器人控制探索（2022-2023初）

该阶段的研究主要探索如何将预训练的视觉-语言模型应用于机器人任务，通常采用VLM作为高层规划器，结合传统控制器执行动作。

**代表工作**：
- PaLM-E、SayCan等工作探索语言模型在机器人规划中的作用
- 主要采用模块化架构，VLM用于语义理解和任务分解

### 2. 突破阶段：端到端VLA模型的诞生（2023中-2024）

2023年Google DeepMind推出RT-2，标志着VLA作为独立研究方向的确立。RT-2证明了VLM可以通过微调直接输出机器人动作，实现端到端的感知-控制。

**关键里程碑**：
- **RT-2**（2023.7）：首个将VLM转化为VLA的工作
- **RT-X**（2023.10）：跨机器人数据集和模型
- **OpenVLA**（2024.6）：首个开源大规模VLA模型

### 3. 快速发展阶段：架构创新与性能优化（2024-2025）

这一阶段出现了大量VLA架构和训练方法的创新，研究重点从证明可行性转向提升性能和效率。

**主要进展**：
- 扩散模型作为动作解码器成为主流
- 分层VLA架构（高层语言规划 + 低层视觉控制）
- 模型压缩和加速（SmolVLA、OFT等）
- 多机器人数据集和基准测试

### 4. 当前阶段：走向通用具身智能（2025-2026）

最新研究趋势表明，VLA正从单一操作任务向通用具身智能体演进，研究重点包括推理能力、全身协调、人机协作等。

**前沿方向**：
- 基于推理的VLA（Reasoning VLA）
- 多任务、多机器人的统一VLA
- VLA与世界模型的结合
- 从模仿学习到强化学习的转变

## 关键技术方向

### 1. 统一的视觉-语言-行动架构

VLA模型的核心是将视觉编码器、语言模型和动作解码器统一在单一架构中，实现端到端的多模态学习。

**主流架构模式**：
- **VLM + Action Head**：在预训练VLM基础上添加动作解码头（如RT-2）
- **Modular Fusion**：独立训练视觉和语言编码器，通过交叉注意力融合（如OpenVLA）
- **Diffusion-based VLA**：使用扩散模型作为动作解码器，建模连续动作分布

**架构优化**：
- 多视角视觉输入的融合
- 时序信息的编码（视频而非单帧）
- 分层表示学习（高层语义 + 低层几何）

### 2. 大规模数据与预训练

VLA模型的性能高度依赖于大规模、高质量的训练数据。当前研究重点在于如何有效利用多源异构数据。

**数据来源**：
- **互联网视觉-语言数据**：用于预训练视觉和语言编码器
- **机器人演示数据**：真实机器人遥操作轨迹（如Open X-Embodiment包含970k轨迹）
- **模拟器数据**：在仿真环境中生成大规模数据
- **合成数据**：利用生成模型创建训练数据

**预训练策略**：
- **两阶段预训练**：先在VLM任务上预训练，再在机器人数据上微调
- **多任务联合训练**：同时学习多个机器人任务
- **跨机器人预训练**：在多种机器人平台上训练通用模型

### 3. 基于扩散模型的动作生成

扩散模型在VLA的动作解码中展现出优越性，能够建模复杂的多模态动作分布。

**优势**：
- 捕获动作的多模态性（同一状态可能有多种合理动作）
- 生成平滑、连续的动作轨迹
- 优于自回归模型的跨域迁移能力

**实现方式**：
- **DDPM**：去噪扩散概率模型
- **Discrete Diffusion**：离散动作空间的扩散建模
- **条件扩散**：以视觉和语言为条件生成动作

### 4. 分层控制与任务分解

分层架构将VLA系统分为高层规划和低层控制，提升了长时序任务的执行能力。

**分层方式**：
- **语言分解**：使用LLM将复杂指令分解为子任务序列
- **时间抽象**：在不同时间尺度上进行规划和控制
- **技能组合**：学习可复用的基本技能，通过组合完成复杂任务

**实现架构**：
- 高层：语言模型进行任务规划和子目标生成
- 低层：VLA模型执行具体的操作技能
- 中间层：目标条件化策略或选项框架

### 5. 跨机器人泛化

训练能够在不同机器人平台上部署的通用VLA模型是当前的重要研究方向。

**关键技术**：
- **统一动作表示**：设计平台无关的动作空间（如末端执行器速度）
- **迁移学习**：从一个机器人迁移到另一个机器人
- **元学习**：学习快速适应新机器人的能力

**代表工作**：
- RT-X：在22个不同机器人上训练统一模型
- Cross-Embodiment Transfer：跨具身形态的策略迁移

### 6. 推理增强的VLA

将显式推理能力集成到VLA模型中，使其能够进行链式思考和因果推理。

**方法**：
- **思维链（CoT）**：生成中间推理步骤
- **世界模型**：预测动作的长期后果
- **反事实推理**：评估不同动作的可能结果

**应用**：
- 复杂任务的分解
- 失败检测和恢复
- 零样本任务泛化

### 7. 高效VLA与模型压缩

为了在实际机器人上实时运行，需要设计轻量化的VLA模型。

**优化方向**：
- **模型量化**：降低参数精度
- **知识蒸馏**：将大模型的知识迁移到小模型
- **架构搜索**：自动设计高效的网络结构

**代表工作**：
- SmolVLA：450M参数的轻量化模型
- OFT：优化微调方法，实现25-50倍加速
- FAST tokenizer：动作压缩，实现15倍推理加速

## 未来研究方向

### 1. 通用具身基础模型

未来的VLA模型将不局限于机械臂操作，而是支持全身协调、移动操作、人形机器人等多种具身形态。

**研究问题**：
- 如何设计统一的观察和动作空间
- 如何处理不同形态的运动学和动力学
- 如何实现跨任务、跨场景的泛化

### 2. 基于世界模型的VLA

结合世界模型使VLA能够进行想象推理和长期规划。

**研究方向**：
- 学习物理世界的因果模型
- 基于模型的强化学习
- 利用想象轨迹进行规划

### 3. 人机协作与交互学习

VLA系统需要与人类自然协作，并从人类反馈中持续学习。

**关键技术**：
- 意图理解和协作规划
- 从人类反馈中学习（RLHF）
- 主动学习和查询策略

### 4. 安全性与可解释性

提升VLA系统的安全性和决策透明度对于实际部署至关重要。

**研究问题**：
- 形式化验证VLA的安全性
- 可解释的决策过程
- 失效检测和安全恢复机制

### 5. 开放世界中的持续学习

VLA系统需要在开放世界中不断遇到新对象、新任务，并持续学习新技能。

**研究方向**：
- 在线学习和模型更新
- 灾难性遗忘的避免
- 主动探索和自主数据收集

### 6. 多模态感知整合

除了视觉和语言，整合触觉、听觉、力/扭矩传感等多种模态。

**技术挑战**：
- 异构模态的融合
- 触觉-视觉对齐
- 多模态预训练

---

# VLA任务类型

VLA模型可以应用于多种类型的机器人任务，根据任务特性和复杂度可以划分为以下类别：

## 按任务复杂度划分

### 1. 单步操作任务（Single-Step Manipulation）

**任务特征**：
- 单一动作即可完成目标
- 不需要长期规划
- 主要考察感知和控制精度

**典型任务**：
- 抓取指定物体
- 按下按钮
- 打开抽屉

**代表数据集**：
- RLBench中的简单任务
- 物体抓取基准测试

---

### 2. 多步骤操作任务（Multi-Step Manipulation）

**任务特征**：
- 需要执行一系列有序动作
- 涉及子目标分解
- 需要维护任务进度状态

**典型任务**：
- 物体重排列
- 多物体组装
- 厨房任务（如准备食物）

**代表数据集**：
- CALVIN
- LIBERO
- RLBench长时序任务

---

### 3. 长时程任务（Long-Horizon Tasks）

**任务特征**：
- 包含数十到上百个动作步骤
- 需要分层规划能力
- 需要错误检测和恢复机制

**典型任务**：
- 完整的烹饪流程
- 家具组装
- 房间整理

**代表数据集**：
- ALFRED
- TEACh
- Habitat 2.0

---

## 按交互对象划分

### 1. 刚体操作（Rigid Object Manipulation）

专注于操作刚性物体，如抓取、放置、推拉等基本操作。

**代表工作**：
- RT-2在刚体操作上的应用
- OpenVLA的基础操作任务

---

### 2. 可变形物体操作（Deformable Object Manipulation）

涉及衣物折叠、绳索整理等可变形物体的操作。

**挑战**：
- 物体状态的高维表示
- 复杂的物理交互建模
- 难以在模拟器中准确建模

---

### 3. 流体操作（Liquid Manipulation）

包括倒水、搅拌等涉及流体的操作任务。

**挑战**：
- 流体动力学的建模
- 状态估计的不确定性
- 传感器的限制

---

### 4. 工具使用（Tool Use）

要求机器人使用工具完成任务，如使用勺子、刀具等。

**关键能力**：
- 工具的功能理解
- 工具-物体交互建模
- 灵巧的操作技能

---

## 按控制模式划分

### 1. 位置控制（Position Control）

**特点**：
- 直接输出末端执行器的目标位置
- 适用于精确定位任务
- 常用于刚体操作

**动作空间**：
- 7自由度机械臂位置（xyz + 四元数）
- 夹爪开合状态

---

### 2. 速度控制（Velocity Control）

**特点**：
- 输出末端执行器的移动速度
- 适用于连续跟踪任务
- 更平滑的运动轨迹

**动作空间**：
- 笛卡尔空间速度（dx, dy, dz, droll, dpitch, dyaw）
- 夹爪速度

---

### 3. 力/阻抗控制（Force/Impedance Control）

**特点**：
- 控制接触力而非位置
- 适用于接触丰富的任务
- 需要力/扭矩传感器

**应用场景**：
- 组装任务
- 人机协作
- 精细操作

---

## 按环境类型划分

### 1. 桌面操作（Tabletop Manipulation）

**环境特征**：
- 相对简单的场景
- 固定的工作台
- 有限的物体种类

**应用**：
- 工业分拣
- 实验室自动化
- 教学演示

---

### 2. 厨房环境（Kitchen Tasks）

**环境特征**：
- 结构化但复杂的场景
- 多样的物体和工具
- 涉及多种交互类型

**应用**：
- 服务机器人
- 家庭辅助
- 食品制备

---

### 3. 移动操作（Mobile Manipulation）

**环境特征**：
- 结合移动和操作
- 大范围的工作空间
- 需要导航和操作的协同

**应用**：
- 仓库物流
- 家庭服务
- 建筑工地

---

# VLA的应用场景

## 工业制造

VLA模型在工业自动化中展现出巨大潜力，能够处理多样化的生产任务。

<div align="center">
  <img src="https://example.com/industrial_vla.png" width="80%" />
<figcaption>
工业场景中的VLA应用
</figcaption>
</div>

**应用示例**：
- 智能装配线：基于语言指令调整装配流程
- 质量检测：结合视觉检测和操作反馈
- 柔性制造：快速适应产品变化

**优势**：
- 减少编程和调试时间
- 提高生产线灵活性
- 降低对专业技术人员的依赖

---

## 服务机器人

在家庭和商业服务场景中，VLA使机器人能够理解和执行多样化的用户指令。

<div align="center">
  <img src="https://example.com/service_robot_vla.png" width="80%" />
<figcaption>
服务机器人应用场景
</figcaption>
</div>

**应用示例**：
- 家庭助理：帮助整理物品、准备食物
- 酒店服务：客房清理、物品配送
- 医疗辅助：协助护理、物品递送

**优势**：
- 自然的人机交互
- 适应个性化需求
- 持续学习用户偏好

---

## 仓储物流

VLA模型使仓储机器人能够处理更复杂、更灵活的物流任务。

<div align="center">
  <img src="https://example.com/warehouse_vla.png" width="80%" />
<figcaption>
仓储物流中的VLA应用
</figcaption>
</div>

**应用示例**：
- 智能分拣：理解包裹描述进行分类
- 货架整理：根据指令调整货物摆放
- 订单拣选：执行复杂的拣选指令

**优势**：
- 提高操作效率
- 减少人工依赖
- 适应多样化的SKU

---

## 农业自动化

在农业领域，VLA模型使机器人能够进行精准的农业操作。

**应用示例**：
- 精准采摘：识别成熟果实并采摘
- 植物护理：根据植物状态进行修剪、施肥
- 自动化收获：执行复杂的收获流程

**优势**：
- 减少劳动力需求
- 提高作业精度
- 降低农产品损伤

---

## 建筑施工

VLA模型在建筑机器人中的应用正在兴起。

**应用示例**：
- 自动化砌砖
- 焊接作业
- 建筑材料搬运

**优势**：
- 提高施工效率
- 改善工作安全性
- 标准化施工质量

---

# VLA主流数据集

VLA模型的性能高度依赖于高质量的训练数据。以下是VLA领域最具影响力的数据集：

## 大规模跨机器人数据集

### Open X-Embodiment Dataset

**基本信息：**
- **发布时间**：2023年10月
- **数据规模**：970k条真实机器人轨迹
- **机器人平台**：22种不同的机器人
- **任务类型**：多样化的操作任务

**数据特点**：
- 跨机器人、跨任务的统一格式
- 包含视觉观察、语言指令和动作轨迹
- 支持跨具身形态的泛化研究

**应用模型**：
- RT-X
- OpenVLA
- 多数最新的VLA模型

**获取方式**：https://robotics-transformer-x.github.io/

---

### RT-1 Dataset

**基本信息：**
- **发布时间**：2022年12月
- **数据规模**：130k条真实机器人轨迹
- **机器人平台**：定制的移动操作机器人
- **任务类型**：700+种日常操作任务

**数据特点**：
- 高质量的专家演示
- 真实办公环境数据
- 丰富的语言指令多样性

**应用模型**：
- RT-1
- RT-2

---

## 模拟器数据集

### CALVIN (Composing Actions from Language and Vision)

**基本信息：**
- **发布时间**：2021年
- **环境**：PyBullet仿真器
- **任务类型**：长时序组合任务

**数据特点**：
- 多步骤任务链
- 语言条件的任务执行
- 评估长期规划能力

**评测指标**：
- 连续成功任务数
- 零样本泛化能力

**官网**：http://calvin.cs.uni-freiburg.de/

---

### LIBERO (Lifelong Benchmark for Robot Manipulation)

**基本信息：**
- **发布时间**：2023年
- **环境**：MuJoCo仿真器
- **任务数量**：130个多样化任务

**数据特点**：
- 4个任务套件，难度递增
- 评估持续学习和泛化能力
- 标准化的评测协议

**任务套件**：
1. LIBERO-Spatial：空间推理
2. LIBERO-Object：物体泛化
3. LIBERO-Goal：目标泛化
4. LIBERO-Long：长时序任务

**官网**：https://libero-project.github.io/

---

### RLBench

**基本信息：**
- **发布时间**：2019年
- **环境**：CoppeliaSim（V-REP）
- **任务数量**：100+任务

**数据特点**：
- 涵盖多种操作技能
- 提供视觉观察和状态信息
- 支持多种机器人平台

**任务类别**：
- 抓取和放置
- 工具使用
- 组装任务

**官网**：https://sites.google.com/view/rlbench

---

## 真实世界数据集

### Bridge Dataset

**基本信息：**
- **发布时间**：2023年
- **数据规模**：60k条真实机器人轨迹
- **机器人平台**：WidowX 250机械臂

**数据特点**：
- 多样化的桌面操作任务
- 真实办公和家庭场景
- 人类遥操作演示

**应用**：
- 模仿学习研究
- Sim-to-Real转移

---

### Language-Table

**基本信息：**
- **发布时间**：2022年
- **数据类型**：真实桌面操作
- **任务类型**：语言条件的物体重排列

**数据特点**：
- 简化的二维操作任务
- 清晰的语言-动作对应
- 便于快速原型开发

**应用**：
- 语言理解研究
- 策略学习方法验证

---

## 人机交互数据集

### Ego4D

**基本信息：**
- **发布时间**：2022年
- **数据规模**：3,600小时第一人称视频
- **场景类型**：日常生活活动

**数据特点**：
- 人类操作演示
- 丰富的语言标注
- 多样化的交互场景

**VLA应用**：
- 从人类视频中学习操作策略
- 理解人类意图和目标

---

## 评测基准

### SIMPLER (Simulation Platform for Embodied Learning and Evaluation Research)

**基本信息**：
- **发布时间**：2025年
- **目标**：标准化的VLA评测框架

**评测内容**：
- 跨任务泛化
- 跨环境泛化
- 鲁棒性测试

**应用**：
- ICLR 2026等会议广泛使用
- 比较不同VLA模型的性能

---

# 经典论文

本节将VLA领域的代表性论文分为“奠基性工作”和“最新突破”两类，以展示该领域的核心进展和前沿动态。

---

## 奠基性工作

奠基性工作通常指那些开创了新方向、提出了核心模型架构或关键技术范式，并对后续研究产生深远影响的论文。

### π₀ (Pi-Zero): A Universal Policy for Embodied AI (2024)

**核心贡献**: Physical Intelligence 公司提出的通用具身AI策略，旨在提供一种能够解决各种机器人任务的通用解决方案。它强调通过大规模预训练和持续学习，使机器人具备泛化能力和对未知环境的适应性。




### RT-1: Robotics Transformer for Real-World Control at Scale (2022)

**核心贡献**：首个大规模真实世界机器人Transformer模型，在130k真实轨迹上训练，证明了Transformer架构在端到端机器人控制中的巨大潜力。

---

### RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control (2023)

**核心贡献**：VLA领域的里程碑，首次将预训练的VLM成功转化为VLA模型，实现了从网络知识到机器人控制的有效迁移，正式开创了VLA这一研究方向。

---

### Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (2023)

**核心贡献**：首次将扩散模型（Diffusion Model）应用于机器人模仿学习，能够有效建模多模态的动作分布，在精细操作任务中表现出色，成为主流动作解码范式之一。

---

### OpenVLA: An Open-Source Vision-Language-Action Model (2024)

**核心贡献**：首个开源的7B参数大规模VLA模型，在970k的Open X-Embodiment数据集上训练，性能超越RT-2，极大地推动了VLA领域的开放研究。

---

### RT-X / Open X-Embodiment Dataset (2023)

**核心贡献**：通过构建包含22种不同机器人的大规模跨平台数据集（Open X-Embodiment），并训练了统一模型RT-X，证明了跨具身形态（cross-embodiment）策略迁移的可行性。

---

## 最新突破

最新突破涵盖了近年来在VLA领域的关键技术创新和性能飞跃，代表了当前研究的前沿方向，如更强的推理能力、更高的执行效率、更可靠的安全性以及对物理世界更深入的理解。

### 规划与推理增强

- **CoT-VLA**: 引入链式思考（Chain-of-Thought）增强VLA的规划与失败恢复能力。
- **EToT (Embodied Thought)**: 利用基于物理的交互式数字孪生作为具身世界模型，进行更可靠的规划。
- **WALL-OSS**: 聚合超过10000小时的机器人数据，实现大规模分层规划系统。
- **MoE-DP**: 集成混合专家层（Mixture of Experts）将长时序任务分解为专门技能。
- **ECoT (Editable CoT)**: 提供可编辑的逐步推理链，允许用户通过语言修正模型的决策过程。

### 多模态对齐与世界模型

- **VoxPoser**: 生成3D可供性图（Affordance Maps）作为连接语言和动作的强中间表示。
- **OccLLaMA**: 利用3D占据网络（Occupancy Networks）增强VLA对三维空间的理解能力。
- **VideoVLA**: 使用视频条件的世界模型学习物理动力学，预测动作对环境的影响。
- **InternVLA-M1**: 结合空间接地预训练和空间引导的动作后训练，提升泛化能力。
- **ProphRL**: 在多样化机器人数据上预训练统一的动作条件世界模型。
- **Point-VLA**: 通过边界框等显式视觉线索解决指代歧义，实现更精确的物体接地。

### 轻量化与高效执行

- **SmolVLA**: 450M参数的轻量化VLA，为在资源受限设备上实时推理提供了可能。
- **OFT**: 优化的微调方法，在不牺牲性能的前提下实现25-50倍的推理加速。
- **FAST Action Tokenizer**: 通过压缩动作块，将动作序列编码为更少的token，实现15倍推理加速。
- **Octo**: 开源的通用机器人策略模型，支持多任务和跨机器人泛化，具有灵活的微调接口。

### 安全性与可靠性

- **VLSA-AEGIS**: 基于控制屏障函数（CBF）实现即插即用的安全约束层，提供形式化安全保证。
- **RationalVLA**: 训练模型学习一个可拒绝的token，使其能够主动识别并拒绝不安全或无效的指令。

### 数据与基准

- **LIBERO**: 首个为终身机器人学习（Lifelong Learning）设计的基准，包含标准化评测指标。
- **PolaRiS**: 通过神经重建技术将真实视频扫描转化为可交互的仿真环境，极大地丰富了仿真数据的来源。
- **SIMPLER**: 标准化的VLA评测平台，支持对模型的泛化能力、鲁棒性等多维度进行评估。

---
**注**：由于VLA领域发展迅速，更多最新论文请参考相关会议论文集（ICLR、CoRL、RSS、ICRA等）以及[VLA-Survey-Anatomy](https://github.com/SuyuZ1/VLA-Survey-Anatomy)等开源项目。

---

# 参考资源

## 综述与调研

### 学术综述

- [VLA-Survey-Anatomy](https://github.com/SuyuZ1/VLA-Survey-Anatomy) - **挑战驱动的VLA分类法**，本文的主要参考资源之一
- [Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications](https://vla-survey.github.io/) - 面向真实世界应用的系统性综述
- [Multimodal fusion with vision-language-action models for robotic manipulation: A systematic review](https://www.sciencedirect.com/science/article/pii/S1566253525011248) - ScienceDirect 2025年发表的系统综述
- [10 Open Challenges Steering the Future of Vision-Language-Action Models](https://arxiv.org/abs/2511.05936) - 未来十大开放挑战

### 会议分析

- [State of VLA Research at ICLR 2026](https://mbreuss.github.io/blog_post_iclr_26_vla.html) - 分析了164篇ICLR 2026的VLA论文
- [Muhayyuddin's VLA Repository](https://muhayyuddin.github.io/VLAs/) - 整合102个模型、26个数据集和12个仿真平台

## 开源模型与工具

### 主要VLA模型

- [OpenVLA](https://github.com/openvla/openvla) - 7B参数开源VLA模型（斯坦福）
- [Large VLM-based VLA for Robotic Manipulation](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation) - 大规模VLM-based VLA列表
- [LeRobot](https://learnopencv.com/vision-language-action-models-lerobot-policy/) - Hugging Face的机器人学习库

### 工具与框架

- [RT-1 & RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) - Google DeepMind的VLA模型
- [Octo](https://octo-models.github.io/) - 开源通用机器人策略
- [π₀ (Pi-Zero)](https://www.physicalintelligence.company/blog/pi0) - Physical Intelligence的通用策略

## 数据集与基准

### 大规模数据集

- [Open X-Embodiment](https://robotics-transformer-x.github.io/) - 970k真实机器人轨迹，22种机器人
- [RT-1 Dataset](https://sites.google.com/view/rt1-robot) - 130k轨迹，700+任务
- [Bridge Dataset](https://rail-berkeley.github.io/bridgedata/) - 60k桌面操作轨迹

### 仿真基准

- [LIBERO](https://libero-project.github.io/) - 终身机器人学习基准
- [CALVIN](http://calvin.cs.uni-freiburg.de/) - 长时序组合任务
- [RLBench](https://sites.google.com/view/rlbench) - 100+仿真操作任务
- [SIMPLER](https://simpler-env.github.io/) - 标准化VLA评测平台

## 学习资源

### 教程与博客

- [Vision Language Action Models & Policies - LearnOpenCV](https://learnopencv.com/vision-language-action-models-lerobot-policy/)
- [VLA Models: The AI Brain Behind Next-Gen Robots](https://medium.com/@raktims2210/vision-language-action-vla-models-the-ai-brain-behind-the-next-generation-of-robots-physical-bced48e8ae94)
- [What are Vision Language Action Models? Complete Guide](https://www.articsledge.com/post/vision-language-action-vla-models)

### 视频课程

- [机器人学习课程](https://rail.eecs.berkeley.edu/deeprlcourse/) - UC Berkeley的深度强化学习课程
- [具身AI讲座系列](https://embodied-ai.org/) - 具身智能相关讲座

## 会议与研讨会

### 主要会议

- **CoRL** (Conference on Robot Learning) - 机器人学习顶会
- **RSS** (Robotics: Science and Systems) - 机器人系统顶会
- **ICRA** (International Conference on Robotics and Automation) - 机器人与自动化国际会议
- **IROS** (International Conference on Intelligent Robots and Systems) - 智能机器人与系统国际会议

### 专题研讨会

- [ICLR 2026 VLA Workshop](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- NeurIPS/CVPR/ICCV 的具身AI workshop

## 实验室与研究组

### 领先研究团队

- **Google DeepMind** - RT-1, RT-2, RT-X等
- **Stanford Vision and Learning Lab** - OpenVLA, LIBERO等
- **UC Berkeley RAIL** - Bridge, Octo等
- **Physical Intelligence** - π₀等商业化产品
- **Toyota Research Institute** - 具身AI研究

## 相关领域

### 扩展阅读

- **Vision-Language Models (VLM)**: CLIP, Flamingo, GPT-4V等
- **Diffusion Models for Robotics**: Diffusion Policy, 3D Diffuser Actor等
- **World Models**: Dreamer, Genie, VideoGPT等
- **Embodied AI**: Habitat, AI2-THOR, iGibson等平台

---

# 总结

Vision-Language-Action (VLA) 模型代表了机器人技术的重要范式转变，通过统一视觉、语言和行动三个模态，实现了从自然语言指令到机器人控制的端到端学习。自2023年RT-2开创这一方向以来，VLA研究取得了快速进展，在模型架构、训练方法、数据集和评测基准等方面都有重要突破。

当前VLA研究正朝着以下方向发展：
1. **通用化**：从单一任务向多任务、跨机器人的通用模型演进
2. **高效化**：通过模型压缩、架构优化实现实时部署
3. **智能化**：集成推理能力，处理更复杂的长时序任务
4. **实用化**：缩小sim-to-real差距，走向真实世界应用

随着大规模预训练模型、高质量机器人数据集和先进训练方法的发展，VLA模型有望成为下一代智能机器人的"大脑"，推动具身智能进入新的发展阶段。

---

**声明**：本文旨在为VLA领域的研究者和学习者提供全面的技术综述。由于该领域发展迅速，部分内容可能随时间推移而更新。欢迎读者在评论区讨论和补充最新进展。

---

## Sources

本文参考了以下资料：

- [State of Vision-Language-Action (VLA) Research at ICLR 2026 – Moritz Reuss](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
- [Multimodal fusion with vision-language-action models for robotic manipulation: A systematic review](https://www.sciencedirect.com/science/article/pii/S1566253525011248)
- [Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications](https://vla-survey.github.io/)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/)
- [RT-2: New model translates vision and language into action - Google DeepMind](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)
- [Vision-Language-Action (VLA) Models: The AI Brain Behind the Next Generation of Robots & Physical AI](https://medium.com/@raktims2210/vision-language-action-vla-models-the-ai-brain-behind-the-next-generation-of-robots-physical-bced48e8ae94)
- [Vision Language Action Models (VLA) & Policies for Robots - LearnOpenCV](https://learnopencv.com/vision-language-action-models-lerobot-policy/)
