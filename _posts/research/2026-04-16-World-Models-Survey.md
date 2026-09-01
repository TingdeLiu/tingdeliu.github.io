---
layout: post
title: "世界模型（World Models）全景综述：从认知理论、四大技术范式到物理基础模型与前沿评测"
date:  2026-09-01
tags: [VLA, World Models, Robotics, Embodied AI, Survey]
categories: research
comments: true
author: Tingde Liu
toc: true
excerpt: "本文系统梳理具身智能世界模型（World Models）的完整发展图景：从认知科学与经典有模型强化学习（World Models 2018、DreamerV3、TD-MPC2）的理论奠基，到大模型时代的四大技术范式（世界规划器、世界动作模型、世界合成器、世界模拟器）与三维正交分类轴，深度解析 Cosmos、Wan2.1、SANA-WM、Motus、AIM 等基础模型与前沿里程碑，全面总结评测基准体系并展望六大未来关键挑战。"
---

# 1. 引言

具身智能（Embodied AI）的终极目标是开发能够像人类一样在复杂、连续且不可逆的物理世界中进行感知、推理并执行任务的通用智能体（Generalist Embodied Agents）。近年来，视觉-语言-动作（Vision-Language-Action, VLA）模型的崛起，标志着具身智能向通用化迈出了关键一步。VLA 模型通过微调多模态大语言模型（LLMs/VLMs）的语义常识与视觉推理能力，将开放式高层自然语言指令直接映射到底层机器人控制 Token。

然而，单纯依靠反应式映射的 VLA 智能体在真实物理场景部署中面临着难以逾越的**四大根本瓶颈**：

1. **物理幻觉与无常识（Physical Hallucination）**：现有的 VLA 模型本质上是模式匹配器，生成的动作往往缺乏对刚体约束、重力、摩擦力、流体动力学等物理规律的显式认知，容易做出穿模、抓空或破坏环境的危险行为。
2. **计划前瞻与因果验证缺失（Lack of Foresight & Plan Verification）**：反应式策略无法在脑海中预演动作执行后的物理后果，难以在行动前进行假设性验证（Counterfactual Reasoning），导致在长程或不可逆任务中一旦出错即彻底失败。
3. **真实交互数据稀缺与分布长尾（Extreme Data Scarcity）**：高质量的机器人真机遥操作数据获取成本高昂、危险度大，真实世界的长尾场景与失败分布极难被有限的离线数据集所覆盖。
4. **不可逆破坏与安全性风险（Safety & Irreversibility）**：在物理世界中盲目试错伴随着昂贵的硬件损耗甚至人身安全风险，智能体必须在物理执行前拥有自我保护与风险预判能力。

为了攻克上述挑战，**世界模型（World Models）** 被全面引入具身智能领域，作为智能体的**“认知大脑”与“内部物理仿真器”**。通过对环境时空演化动力学的深度建模，世界模型赋予了智能体**感知物理常识、脑内推演未来、合成海量数据与在想象中自优化**的核心能力。

<div align="center">
  <img src="/images/wm/Survey-Fig1-Overview.webp" width="75%" />
<figcaption>图：具身智能世界模型全景总览。世界模型（交互性、未来预测、物理接地）与 VLA（通用策略、开放指令、多模态推理）的深度融合，赋予智能体前瞻规划、虚拟仿真、联合决策和数据飞轮四大核心能力。（图源：Tan et al., 2026）</figcaption>
</div>

本文立足于国际最新学术前沿与工业级落地实践（涵盖 Tan et al., 2026; Li et al., 2025/2026 等权威综述），系统梳理具身智能世界模型的理论根基、四大核心技术范式、物理基础模型平台、经典里程碑代表作、评测基准与指标体系，以及未来的关键挑战与演进方向。

---

# 2. 具身智能世界模型理论根基与架构演进

## 2.1 什么是具身智能世界模型？认知科学起源与数学形式化

### 认知科学与预测编码理论起源

“世界模型”这一概念并非源于深度学习，而是植根于认知科学与控制论的深厚土壤：
- **心理模型假说（Mental Models）**：早在 1943 年，认知科学家 Kenneth Craik 在 *The Nature of Explanation* 中提出，人类大脑在头脑内部运行着一个微型“心理模型”，能够在实际采取危险行动之前，在脑中模拟替代方案的后果。
- **自由能原理与预测编码（Predictive Coding）**：Karl Friston 等人指出，生物大脑本质上是一个层级化的预测机器，通过不断最小化“内部感知预测”与“外部真实输入”之间的预测误差（Prediction Error）来理解世界并指导行动。
- **自主机器智能架构（Yann LeCun, 2022）**：Yann LeCun 在其机器智能愿景中强调，世界模型是自主智能体不可或缺的核心模块，负责根据当前状态与候选动作预测世界可能出现的演化。

<div align="center">
  <img src="/images/wm/wmandvla.png" width="90%" />
<figcaption>图：物理 AI 世界模型（Cosmos WM）与具身策略模型（OpenVLA）的功能定位对比：前者负责环境演化与物理仿真，后者负责动作决策与执行。</figcaption>
</div>

### 数学形式化

在具身智能语境下，真实物理世界通常被建模为一个部分可观测马尔可夫决策过程（POMDP），由元组 $$(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{E}, \mathcal{R}, \gamma)$$ 描述。由于真实状态 $$s_t \in \mathcal{S}$$ 无法直接获取，智能体只能接收高维视觉观测 $$o_t \in \mathcal{O}$$（如 RGB-D 图像、点云）。

**世界模型** $$\mathcal{W}_\phi$$ 的核心任务是通过学习参数 $$\phi$$，在紧凑潜空间或像素空间中近似环境的联合前向转移分布：

$$
P_\phi(s_{t+1}, o_{t+1}, r_t \mid s_{\le t}, a_t, o_{\le t})
$$

其中：
- **状态转移模型（Dynamics / Transition Model）**：$$s_{t+1} \sim P_\phi(s_{t+1} \mid s_t, a_t)$$，刻画动作引发的环境内在物理状态跃迁；
- **观测解码/预测模型（Observation Predictor）**：$$o_{t+1} \sim P_\phi(o_{t+1} \mid s_{t+1})$$ 或直接在像素层建模 $$o_{t+1} \sim \mathcal{W}_\phi(o_{t+1} \mid o_{\le t}, a_t)$$；
- **奖励与终止评估模型（Reward & Termination Model）**：$$r_t \sim P_\phi(r_t \mid s_t, a_t), \; c_t \sim P_\phi(c_t \mid s_t)$$，用于在想象中评估状态价值与任务进度。

---

## 2.2 理论演进脉络：三大奠基性里程碑（从经典有模型 RL 到大模型时代）

> 💡 **姊妹篇导读**：关于经典强化学习数学基础、Bellman 算子推导、无模型与有模型 RL（DreamerV3 / TD-MPC2）的系统性算法剖析，详见专题博文 [强化学习（RL）全景综述：从马尔可夫决策过程、价值/策略迭代到前沿具身控制](/Reinforcement-Learning-Survey/)。

在现代视频扩散基础模型爆发之前，世界模型在有模型强化学习（Model-Based Reinforcement Learning, MBRL）领域经历了数代关键演化，奠定了整个领域的数学理论与算法基石。

### 里程碑 1：World Models（Ha & Schmidhuber, NeurIPS 2018）——潜空间“预知梦”奠基

David Ha 与 Jürgen Schmidhuber 提出的 **World Models** 首次在深度学习框架下完整实现了认知科学中的 **V-M-C 三位一体架构**：

<div align="center">
  <img src="/images/wm/WorldModels-vmc-overview.webp" width="90%" />
<figcaption>图：World Models (2018) 的 V-M-C 核心架构：视觉感知 V（VAE）、记忆动力学 M（MDN-RNN）与轻量控制器 C。（图源：Ha & Schmidhuber, 2018）</figcaption>
</div>

1. **V 模型（Vision Model / VAE）**：将高维输入图像帧 $$o_t$$ 压缩为 32 维连续高斯潜向量 $$z_t \sim \mathcal{N}(\mu, \sigma^2)$$，过滤与控制无关的高频视觉噪声；
2. **M 模型（Memory Model / MDN-RNN）**：基于带有混合高斯输出层（MDN）的 LSTM，自回归预测下一时刻潜状态的多分支概率分布：
   
   $$
   P(z_{t+1} \mid a_t, z_t, h_t) = \sum_{k=1}^K \pi_k(h_t) \mathcal{N}\left( z_{t+1};\; \mu_k(h_t), \Sigma_k(h_t) \right)
   $$

3. **C 模型（Controller）**：仅包含千余参数的超轻量前馈网络，直接将 $$z_t$$ 与循环隐状态 $$h_t$$ 映射为控制动作 $$a_t = W_c [z_t \; h_t] + b_c$$。

<div align="center">
  <img src="/images/wm/WorldModels-flow-diagram.webp" width="90%" />
<figcaption>图：World Models 完整数据流：离线收集经验训练 V 与 M，随后在完全脱离真实环境的 RNN 梦境中训练控制器 C。（图源：Ha & Schmidhuber, 2018）</figcaption>
</div>

> **历史性突破**：World Models 首次实现了**“完全在虚拟梦境中训练策略（Training inside the Dream）”**——拔掉真实仿真器，让 M 模型在内存中自回归展开虚拟赛道（CarRacing-v0）与虚拟射击场景（VizDoom），控制器通过进化策略（CMA-ES）在数分钟内即可在纯梦境中学会极限漂移与火球闪避，并在真机/真实游戏中无缝迁移。

---

### 里程碑 2：Dreamer 系列与 RSSM（Hafner et al., Nature 2025 / 2023）——通用离散状态空间与 Minecraft 破局

Danijar Hafner 等人开创的 **Dreamer 系列**（DreamerV1 $\to$ V2 $\to$ V3，技术成果于 2025 年正式发表于 *Nature*，题目为 *Mastering Diverse Domains through World Models*）将有模型 RL 推向了通用化的高峰。

<div align="center">
  <img src="/images/wm/DreamerV3-training-architecture.webp" width="95%" />
<figcaption>图：DreamerV3 训练流水线：(a) 从真实经验中自监督学习 RSSM 世界模型；(b) 在潜空间自回归展开轨迹；(c) 在想象中优化 Actor-Critic 策略。（图源：Hafner et al., Nature 2025）</figcaption>
</div>

Dreamer 解决了连续世界模型长期存在的表示坍缩与数值不稳定性问题：
1. **循环状态空间模型（Recurrent State Space Model, RSSM）**：将潜状态解耦为确定性时序特征 $$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$$ 与离散随机变量 $$z_t$$（采用 32 个 32 类别的 Categorical 潜变量）。离散 Categorical 分布彻底避免了连续高斯分布在面对非线性突变（如门开/关、物体碎裂）时的模糊与坍缩问题；
2. **Symlog 变换与无量纲化设计**：提出对称对数变换 $$\mathrm{symlog}(x) = \mathrm{sign}(x)\ln(|x|+1)$$ 统一缩放特征与回归目标，配合自适应百分位数价值归一化，解决了跨任务奖励尺度横跨 7 个数量级导致的梯度弥散与数值发散；
3. **历史性里程碑**：DreamerV3 是人类历史上**首个使用完全相同的超参数与模型架构**，在 Atari、Crafter、DMC、Procgen 以及高难度 3D 沙盒 **Minecraft** 中通用的算法。在零专家演示、纯潜空间梦境训练的严苛设定下，首次自主攻克了从伐木、造台、采矿、冶炼到**合成钻石的 14 步深度依赖链**。

<div align="center">
  <img src="/images/wm/DreamerV3-benchmark-summary.webp" width="90%" />
<figcaption>图：DreamerV3 在 7 大异构领域（2D 像素、连续控制、3D 长程沙盒）中的标准化性能对比。（图源：Hafner et al., Nature 2025）</figcaption>
</div>

---

### 里程碑 3：TD-MPC 与 TD-MPC2（Hansen et al., ICLR 2024 Oral）——无需像素重构的任务驱动隐式规划

以往的世界模型（如 World Models、Dreamer）大多依赖逐像素的图像重建损失，大量网络算力被浪费在与下游控制无关的背景视觉细节上。Nicklas Hansen 等人提出的 **TD-MPC 系列** 实现了关键的技术转向：

<div align="center">
  <img src="/images/wm/TD-MPC2-architecture.webp" width="90%" />
<figcaption>图：TD-MPC2 整体架构：无需逐像素解码重建，在紧凑潜空间中深度融合 MPPI 在线采样规划与时序差分（TD）长程价值学习。（图源：Hansen et al., ICLR 2024）</figcaption>
</div>

1. **纯潜空间任务驱动（Task-Driven Latent Dynamics）**：彻底摒弃像素解码器，直接通过时序差分（TD-Target）、即时奖励与策略先验联合约束潜状态 $$z_t$$ 的动力学演化；
2. **MPPI 毫秒级在线并行规划**：在推理时利用模型预测路径积分（MPPI）在 512 维的潜空间中并行推演数千条轨迹，耗时不到 5ms，兼顾了高频闭环控制与长程价值视野；

---

### 从经典有模型 RL 到大模型时代具身世界模型的技术跃迁

下表总结了世界模型在过去数年间的核心技术范式演进：

| 演进维度 | 经典奠基期 (2018, World Models) | 离散 RSSM 时代 (2023–2025, DreamerV3) | 隐式 MPC 时代 (2024, TD-MPC2) | 生成式基础模型时代 (2025–2026, Cosmos / WAM / SANA) |
|:---|:---|:---|:---|:---|
| **状态表征** | 连续高斯潜变量 (32D VAE) | 离散 Categorical 潜变量 (32×32) | 紧凑任务潜向量 (512D MLP/Trans.) | 时空 Latent 网格 / 3DGS 显式场 / 统一 Canvas |
| **动力学骨干** | LSTM / MDN-RNN | RSSM (GRU + Categorical) | MLP / 密集 Transformer | Diffusion Transformer (DiT) / Flow Matching / MoT |
| **重建机制** | 逐像素 2D 解码 (64×64) | 逐像素 2D 解码 (64×64) | **无像素重建**（纯潜空间任务信号） | 高压缩时空 VAE (4×8×8) / 单步生成 / 隐空间对齐 |
| **与策略关系** | 解耦：梦境中离线进化 C | 解耦：潜空间 Actor-Critic | 耦合：在线 MPPI 轨迹优化 | 四大范式并存（规划器 / WAM / 合成器 / 模拟器） |
| **动作生成频率** | 离线优化后真机部署 | 离线优化后真机部署 | 毫秒级在线 MPPI (~50Hz) | 5Hz–15Hz 联合去噪 (WAM) 或 500Hz 混合控制 |
| **多任务与规模** | 单任务小模型 (<1M) | 跨领域单模型 (~200M) | 104 种具身多任务 (1M $\to$ 317M) | 互联网级图文视频预训练 + 领域微调 (2B $\to$ 64B) |

---

## 2.3 核心要素与系统架构：System 1 与 System 2 双系统协同

进入大模型与具身智能深度交织的时代，学术界与工业界逐渐收敛于一种**双系统（Dual-System）认知架构**：

```mermaid
flowchart TD
    Env["真实物理环境 / 机器人本体"] -->|"多视角观测 o_t, 本体感觉 s_t"| Perception["多模态感知编码器\n(ViT / DINOv3 / SigLIP)"]
    Instruct["开放式语言任务指令"] --> Perception

    subgraph System2["慢系统 System 2：认知推理与世界模拟 (1Hz – 5Hz)"]
        direction TB
        WM["生成式世界模型 W_φ\n(DiT / Flow Matching 动力学)"]
        VLM["多模态大语言模型 (MLLM)\n(高层意图分解、因果推理、常识验证)"]
        ValueMap["空间价值图 / 交互意图预测 (ASVM)"]
        WM <--> VLM
        WM --> ValueMap
    end

    Perception --> System2
    Perception --> System1

    subgraph System1["快系统 System 1：反应式高频运动控制 (50Hz – 500Hz)"]
        direction TB
        Policy["动作解码器 / 扩散策略 (Policy Head)\n(Diffusion Policy / Action Flow Matching)"]
        LowCtrl["低级关节控制器\n(EtherCAT / 阻抗控制 / 扭矩输出)"]
        Policy --> LowCtrl
    end

    System2 -->|"前瞻潜引导 z_{t+1} / 空间价值约束 / 子目标"| Policy
    LowCtrl -->|"控制扭矩 / 关节速度 a_t"| Env

    style System2 fill:#fff4e6,stroke:#d68910,stroke-width:2px
    style System1 fill:#e8f4fd,stroke:#2c7fb8,stroke-width:2px
```

1. **慢系统（System 2，认知推理与世界模拟，1Hz–5Hz）**：
   - 由世界模型（WM）与多模态大模型（VLM）组成，负责长时程任务规划、环境动态推演、物理常识校验、意图分析以及危险评估；
   - 通过生成未来潜特征 $$z_{t+1}$$ 或空间价值热图，为底层提供物理接地的条件引导。
2. **快系统（System 1，反应式高频动作执行，50Hz–500Hz）**：
   - 由轻量级策略网络（如 Diffusion Policy、Action Flow Matching）与底层控制器构成，负责根据当前状态与 System 2 提供的物理先验，以极低延迟实时生成平滑、精确的电机扭矩或关节轨迹。

---

## 2.4 具身世界模型的多维分类学 (Multidimensional Taxonomy)

根据 Tan et al. (2026) 与 Li et al. (arXiv:2510.16732) 的最新理论，具身世界模型可从**四大应用技术范式**与**三大正交建模轴**进行多维解构：

### 四大应用技术范式（The Four Paradigms）
1. **世界规划器（World Planner）**：世界模型作为前向动力学引擎，预测显式未来帧或隐式潜嵌入，为下游策略提供前瞻性条件引导；
2. **世界动作模型（World Action Model, WAM）**：将世界状态演化与机器人控制动作纳入统一网络，联合建模观测与控制的联合分布；
3. **世界合成器（World Synthesizer）**：作为高保真数据生成飞轮，合成海量带标注的多视角、长程交互轨迹，支持大规模模仿学习；
4. **世界模拟器（World Simulator）**：将世界模型作为虚拟物理沙盒，结合强化学习（RL）算法在想象空间中优化策略参数。

### 三大正交建模轴（Three-Axis Modeling Taxonomy）
- **功能耦合度（Functionality Coupling）**：
  - *决策解耦（Decision-Decoupled / General Purpose）*：世界模型独立于特定动作空间预训练（如纯视频生成），下游通过逆动力学或特征微调适配；
  - *决策耦合（Decision-Coupled / Policy-Integrated）*：世界模型与动作头深度交织，动作作为原生 Token 或条件通道共同优化。
- **时序建模方式（Temporal Modeling）**：
  - *序列自回归/自回归扩散（Sequential Simulation & Rollout）*：逐步展开未来状态 $$s_{t+1}, s_{t+2}, \dots$$，适合长程交互与连续物理演变；
  - *全局差分/跳步预测（Global Difference & Jump-Step Prediction）*：直接预测关键子目标帧或最终转移差分 $$\Delta s$$，跳过中间无关微动态。
- **空间与状态表征（Spatial & State Representation）**：
  - *全局潜向量（Global Latent Vectors）*：如 RSSM、V-JEPA 2，高度抽象，计算极快，但缺少细粒度空间几何；
  - *空间潜在网格（Spatial Latent Grids）*：如 DiT Latent Patches、VAE 特征图，平衡了感知保真度与计算效率；
  - *显式几何场（Explicit 3D Fields）*：如 3DGS、点云（Point Clouds）、神经占据栅格（Occupancy Grids），具备天然的 3D 空间一致性与度量物理约束；
  - *统一多模态画布（Unified Latent Canvas）*：如 NavWAM、Cosmos 3，将视觉、动作、状态、价值多模态拼装为时空画布。

---

## 2.5 演进时间线（2018–2026）

<div align="center">
  <img src="/images/wm/Survey-Fig2-Timeline.webp" width="100%" />
<figcaption>图：具身智能世界模型演化时间线。从 2018 年潜空间梦境训练奠基，到 2023 年视频生成驱动规划，再到 2025–2026 年世界动作模型（WAM）、全模态基础模型（Cosmos 3）与可探索 3D 宇宙（Lyra 2.0 / Marble）的爆发。（图源：Tan et al., 2026）</figcaption>
</div>

**关键演进脉络**：
- **2018–2022年（MBRL 奠基期）**：World Models 提出 V-M-C 梦境训练；PlaNet 引入 RSSM；DreamerV1/V2 建立潜空间 Actor-Critic；
- **2023–2024年（视频先验与 3D 萌芽期）**：UniPi、SuSIE 探索利用视频扩散模型进行文字引导规划；GR-1 开创自回归视频动作预训练；3D-VLA 引入 3D 几何先验；DreamerV3 发表并通关 Minecraft；TD-MPC2 统一 104 种具身控制；
- **2025年（四大范式爆发期）**：WorldVLA、UniVLA 统一自回归序列建模；DreamGen、GigaWorld-0 构建数据合成飞轮；VLA-RFT、WoVR 开启世界模型内部强化学习；Cosmos 平台建立工业级数据与模型体系；
- **2026年（全模态收敛与空间智能时代）**：Cosmos 3 以 MoT 架构统一理解-生成-动作；SANA-WM 实现分钟级高效生成；Lyra 2.0 与 Marble 实现 3DGS 可探索持久世界；Motus、AIM、NavWAM、Qwen-RobotWorld 推动世界动作模型成为具身控制新标准。

---

# 3. 四大技术范式详解


<div align="center">
  <img src="/images/wm/Survey-Fig3-Paradigms.webp" width="95%" />
<figcaption>图：VLA 世界模型的四大技术范式。(a) 世界规划器：世界模型生成潜表示 z 引导 VLA；(b) 世界动作模型：将观察与动作联合建模；(c) 世界合成器：通过模仿学习（IL）构建合成数据集；(d) 世界模拟器：通过强化学习（RL）优化策略并获取外部奖励。（图源：Tan et al., 2026）</figcaption>
</div>

## 3.1 世界规划器 (World Planner)

> 💡 **姊妹篇导读**：关于具身策略模型（OpenVLA、$\pi_0$、Octo、RoboCat 等）的动作 Tokenization、跨本体预训练与端到端控制架构，详见专题博文 [视觉-语言-动作（VLA）全景综述：从大模型底座、数据引擎到物理落地](/VLA-Survey/)。

<div align="center">
  <img src="/images/wm/World_Planner.webp" width="80%" />
<figcaption>图：InternVLA·N1 的端到端双系统架构：前向动力学规划器提供未来潜特征引导策略执行。（图源：Intern Robotics）</figcaption>
</div>

**定义**：该范式采用世界模型 $$\mathcal{W}_\phi$$ 作为前向动力学模型，以显式未来观测帧 $$\hat{o}_{t+1}$$ 或隐式潜特征 $$z_{t+1}$$ 的形式合成前瞻引导信号，为下游策略 $$\pi_\theta$$ 提供强语义与物理条件：

$$
\max_\theta \mathbb{E}_{z_{t+1} \sim \mathcal{W}_\phi(\cdot|o_t)} \left[ \sum_t \log \pi_\theta(a_{t+1} | o_t, z_{t+1}) \right]
$$

世界规划器的核心哲学是**“预测先于行动”（Predict before Act）**：先由世界模型预见未来（显式图像或隐式潜向量），再将该前瞻信号作为输入喂给策略网络，使决策具备物理接地的未来感知与因果预判。两种主流路径的信息流如下：

```mermaid
flowchart LR
    O_t["当前观测 o_t"] --> WM

    subgraph WM["世界模型 W_φ（前向动力学）"]
        direction TB
        ExP["显式路径（Explicit）\n扩散/流匹配生成未来像素帧\nUniPi / SuSIE / FLIP"]
        ImP["隐式路径（Implicit）\n自监督时空潜空间预测\nV-JEPA 2 / PIVOT-R"]
    end

    ExP -->|"预测未来帧 o_{t+1}"| IDM["逆动力学模型 (IDM)\n从帧差推断动作 a"]
    ExP -->|"潜特征嵌入 z_{t+1}"| Policy
    ImP -->|"高维动力学潜嵌入 z_{t+1}"| Policy

    IDM --> Policy
    O_t --> Policy
    Policy["策略网络 π_θ(a | o_t, z_{t+1})"] --> Robot["执行器 / 关节控制器"]
```

**显式规划 vs 隐式规划深度对比**：

| 维度 | 显式像素规划（Explicit Pixel Planning） | 隐式潜空间规划（Implicit Latent Planning） |
|:---|:---|:---|
| **引导信号** | 像素级未来图像/视频帧 $$\hat{o}_{t+1}$$ | 紧凑时空特征嵌入 $$z_{t+1}$$（如 V-JEPA 2 特征） |
| **代表方法** | UniPi, SuSIE, GR-MG, Vidar, 3D-VLA, FLIP | V-JEPA 2, PIVOT-R, VPP, MinD, TriVLA, MoWM |
| **主要优势** | 人类直观可解释、视觉调试方便、可直接接入通用 VLM | 过滤光照/纹理等与控制无关的视觉噪声，计算速度快，不易产生像素级伪影 |
| **主要劣势** | 扩散反向去噪采样耗时大、易在细小接触面出现像素级变形 | 缺乏直观可解释性、下游策略需与特定潜空间强对齐 |
| **动作推导机制** | 逆动力学模型（IDM）从帧间变化解算动作，或由策略网络条件读取 | 策略网络直接在潜空间 cross-attend 读取前瞻潜特征 |

**演进路径**：早期工作（UniPi、SuSIE、GR-MG、Vidar、3D-VLA、FLIP）将规划视为高保真条件视频生成任务，通过视频扩散模型合成像素级未来状态，再经逆动力学模型导出动作。然而，像素级生成面临严重的推理延迟与细微物理接触模糊。近期 V-JEPA 2、PIVOT-R 和 TriVLA 转向隐式规划，直接在自监督潜空间预测未来特征，彻底摆脱了动力学无关背景细节的干扰，大幅提升了引导信号的信噪比与计算吞吐量。MoWM 则融合多模态动力学先验形成混合方案，进一步降低动作推导误差。

### 工业落地案例：GENE-26.5（Genesis AI, 2026）

[Genesis AI 于 2026 年发布的 GENE-26.5](https://www.genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level) 是世界规划器范式在工业级灵巧操作领域的标杆落地。它最核心的技术突破在于：**新任务仅需 < 1 小时（约 200 episodes、< 20 秒技能）的真机数据即可完成微调**，而支撑这一极高样本效率的，正是“以世界模型为低层动作策略注入物理常识先验”的设计哲学。

**三个功能角色，一个统一模型**

从系统分工看，GENE-26.5 呈现出三个功能层：
- **语义感知层（VLM）**：编码自然语言指令与场景语义，负责高层逻辑任务链分解；
- **物理预测层（World Model）**：动作条件视频生成模型，从海量无标注视频中预先习得“未来几秒内物体如何受力、形变、断裂与滑动”，提供强物理常识先验；
- **执行转化层（Action Model）**：高频底层控制，将语义+物理条件直接翻译为连续关节扭矩。

在工程实现上，**GENE-26.5 并非三个孤立串联的模块**，而是采用 **Flow Matching** 统一建模 language、vision、proprioception、tactile 和 action 的**联合多模态分布**。VLM 与 World Model 是被吸收进来的预训练组件，下游通过**条件查询（Conditional Queries）**从同一联合分布中无缝采样出 control、generative simulation、state estimation、IDM 或 value estimation 等不同子任务。

```mermaid
flowchart LR
    Lang["语言指令\n(intent)"] --> VLM
    Img["第一/第三人称视频\n本体感觉 / 触觉"] --> VLM

    VLM["VLM\n语义编码"] -->|"语义条件"| Joint
    VLM -->|"语义条件"| WM

    subgraph Joint["联合分布模型 (Flow Matching)\n language ⊕ vision ⊕ proprioception ⊕ tactile ⊕ action"]
        direction TB
        WM["World Model\n动作条件视频生成\n→ 物理先验 z_phys"]
        Cond["条件查询接口\n control / sim / state est. / IDM / value"]
    end

    WM -->|"潜空间物理引导"| Cond
    Cond -->|"action sample"| Ctrl["500Hz 控制栈\nEtherCAT, 3ms 端到端\n~2mm 追踪误差"]
    Ctrl --> Hand["Genesis Hand 1.0\n20-DoF 仿人手"]
```

**训练范式与硬件协同**：
- **异构多模态预训练（> 200,000 小时）**：覆盖手套捕捉数据（轨迹+触觉）、第一人称自然交互视频、第三人称互联网物理交互视频以及图文语料。模型直接从非完美对齐的异构数据中习得“感知-物理-动作”耦合先验，因而下游任务只需 20–30 分钟数据微调；
- **500Hz / 3ms 超低延迟硬件控制栈**：世界规划器的物理轨迹在下发到真机时，往往因控制栈延迟导致误差放大。GENE-26.5 自研 EtherCAT 中间件，将端到端通信延迟压缩至 3ms，结合 20 个主动可反驱（Back-drivable）自由度的 Genesis Hand 1.0，将跟踪误差压至 ~2mm，确保了世界模型预测的物理轨迹在物理执行端得到高保真复现。

---

## 3.2 世界动作模型 (World Action Model, WAM)

**定义**：该范式采用生成式序列模型或扩散模型，将未来观测状态与控制动作纳入统一网络，直接建模观测与控制的联合时空分布：

$$
\max_\phi \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_t \log \mathcal{W}_\phi(o_{t+1}, a_{t+1} \mid o_{\le t}, a_{< t}) \right]
$$

与世界规划器“前瞻预测与策略解码前后串联”不同，世界动作模型将两者**统一在同一个骨干网络中联合优化**：模型既要预测未来帧（自监督物理演化目标），又要直接解码控制动作（策略执行目标）。

```mermaid
flowchart LR
    subgraph IN["多模态输入"]
        O_t["当前观测 o_t"]
        H["历史观测/动作<br/>o_{t-k:t-1}, a_{t-k:t-1}"]
        L["语言指令 / 目标 g"]
    end
    subgraph WM["世界动作模型 W_φ（联合动力学骨干）"]
        direction TB
        Enc["多模态编码器 / Tokenizer"] --> Joint["自回归 Transformer / 扩散 DiT / MoT\n(联合序列建模 / 潜在画布 Latent Canvas)"]
    end
    subgraph OUT["统一联合输出"]
        O1["预测未来帧/潜变量\n(视觉自监督/锚定约束)"]
        A1["控制动作 Chunk a_{t:t+H}\n(5Hz–15Hz 闭环高频输出)"]
    end
    O_t --> Enc
    H --> Enc
    L --> Enc
    Joint --> O1
    Joint --> A1
    A1 --> Robot["机器人执行器"]
    O1 -.->|"共享注意力梯度反传\n约束动作符合物理常识"| Joint
```

### 为什么世界动作模型（WAM）正在成为具身控制新标准？

传统的世界模型在推理时往往需要依赖外挂的轨迹搜索算法（如交叉熵方法 CEM、蒙特卡洛树搜索 MCTS、模型预测路径积分 MPPI），在成百上千条随机候选动作序列中逐一推演评分，导致单步决策耗时高达数秒，无法满足高频动态交互。

**WAM 的颠覆性突破在于彻底消除了测试时在线规划的巨大计算开销**：
1. **单次前向直接生成动作**：在测试阶段，WAM 可以直接以 Policy 模式单次前向去噪输出可执行动作块（Action Chunk），控制频率可达 **5Hz–15Hz**，计算量较传统 CEM 规划降低上千倍；
2. **未来视觉预测作为强正则化与路标锚定**：在训练阶段，模型被迫在预测未来画面的同时生成动作。由于视觉去噪包含密集的像素级自监督信号，动作头获得了深度的物理动力学约束，有效缓解了反应式策略在长程控制中的“策略漂移（Policy Drift）”；
3. **架构的统一与灵活性**：通过灵活的掩码机制（Masking）或加噪调度，同一个 WAM 权重即可自由切换为前向动力学模拟器、逆动力学标注器、纯策略控制器或跨模态编辑工具。

**细粒度分类**（根据建模范式和实现机制）：

| 建模范式 | 核心机制 | 代表性方法 | 核心技术亮点 |
|:---|:---|:---|:---|
| **自回归（AR）** | 视频预训练 | GR-1, HMA, UniVLA, GR-2 | 大规模视频先验转化为端到端动作预测 |
| **自回归（AR）** | 统一序列建模 | WorldVLA, RynnVLA-002, UP-VLA | 将图像、动作、文本离散化为统一 Token 流 |
| **自回归（AR）** | 前瞻与思维链推理 | Seer, FlowVLA, CoT-VLA, DreamVLA | 引入多模态思维链与未来光流引导结构化决策 |
| **扩散 / 流匹配** | 混合专家（MoT） | Motus, Cosmos 3 | 共享自注意力 + 解耦 FFN，UniDiffuser 多模式切换 |
| **扩散 / 流匹配** | 空间价值接口 | AIM | 空间价值图（ASVM）显式解耦意图，自蒸馏 RL 优化 |
| **扩散 / 流匹配** | 潜在画布（Canvas） | NavWAM | 9帧世界-动作潜在画布，消除 CEM 实现 5Hz 实时导航 |
| **扩散 / 流匹配** | 非对称视界 | WAM-Nav | 动作长视界（24步）+ 视觉短视界（1步），防止视角剧变漂移 |
| **扩散 / 流匹配** | 语言统一动作接口 | Qwen-RobotWorld | 冻结 MLLM 将 20+ 本体、500+ 动作映射为自然语言指令 |

---

## 3.3 世界合成器 (World Synthesizer)

**定义**：该范式将世界模型构建为可无限扩展的**生成式数据飞轮（Data Engine）**，通过联合生成器 $$\mathcal{G}_{\theta,\phi}$$ 自主合成包含交错观测与控制标注的大规模轨迹数据集 $$\mathcal{D}_{syn}$$，用以支撑大规模模仿学习（IL）：

$$
\mathcal{D}_{syn} \triangleq \left\{ \tilde{\tau} \sim p(o_0) \prod_t \mathcal{G}_{\theta,\phi}(\hat{o}_{t+1}, a_{t+1} \mid \hat{o}_t, \text{instruction}) \right\}
$$

世界合成器充当解决机器人领域“数据长尾与昂贵采集”的关键钥匙。根据是否依赖真实动作标注，演化出两条核心合成路径：

```mermaid
flowchart LR
    Init["初始环境观测 o_0"] --> G
    Cmd["任务语言指令"] --> G

    subgraph G["世界合成器 G_{θ,φ}"]
        direction TB
        PathA["路径 A：动作条件生成 (Action-Conditioned)\n给定动作序列展开未来高保真视频\nCtrl-World / Genie Envisioner / WristWorld"]
        PathB["路径 B：无动作视觉合成 + IDM 标注 (Action-Free)\n先利用视频大模型合成视觉动作流，再经逆动力学反推控制\nDreamGen / GigaWorld-0 / Image2Sim"]
        IDM["高精度逆动力学模型 (IDM)\n从相邻生成帧差精确提取动作 â_t"]
        PathB --> IDM
    end

    PathA -->|"(ô_{t+1}, a_{t+1})"| Dsyn
    IDM  -->|"(ô_{t+1}, â_{t+1})"| Dsyn
    Dsyn["亿级大规模合成数据集 D_syn\n(覆盖多视角、新物体、干扰背景)"] -->|"大规模模仿学习 (IL)"| Policy["下游通用 VLA 策略 π_θ"]
```

**细粒度分类与核心路径**：

1. **动作条件生成路径（Action-Conditioned Rollouts）**：
   - *代表方法*：Ctrl-World, Genie Envisioner, WristWorld；
   - *机制*：基于真实采集的轨迹动作序列作为输入条件，通过世界模型生成对应视角的未来演变视频（如 WristWorld 生成 4D 手腕视角动态），实现对现有数据的视角扩充与背景泛化；
2. **无动作合成与逆动力学标注路径（Action-Free Video Synthesis + IDM）**：
   - *代表方法*：DreamGen, GigaWorld-0, Image2Sim；
   - *机制*：彻底摆脱对真机动作标签的依赖！直接利用大规模视频生成底座（如 Wan2.1、Sora、Image2Sim）在给定任务指令下合成视觉上物理合理的交互视频轨迹，随后通过高精度逆动力学模型（IDM）从生成的视频帧差中反推出机器人关节控制动作 $$\hat{a}_t$$。该路径能够无缝吸收互联网规模的视频知识，是突破机器人长尾数据瓶颈的最具潜力的方向。

---

## 3.4 世界模拟器 (World Simulator)

**定义**：该范式将动作条件世界模型 $$\mathcal{W}_\phi$$ 作为**神经虚拟物理仿真器**，智能体在世界模型展开的“想象空间”中执行交互试错，并结合外部奖励评估器 $$\mathcal{R}_{ext}$$，利用强化学习（RL）算法端到端优化策略参数：

$$
\max_\theta \mathbb{E}_{\substack{a \sim \pi_\theta(\cdot|o) \\ \hat{o} \sim \mathcal{W}_\phi(\cdot|o,a)}} \left[ \mathcal{R}_{ext}(\hat{o}, a) \right]
$$

世界模拟器实现了“脱离昂贵真机与传统物理引擎，在神经仿真器中直接进行大规模强化学习”的闭环：

```mermaid
flowchart TB
    O["真实/初始观测 o"] --> Policy
    O --> WS

    Policy["策略网络 π_θ(a|o)"] -->|"采样动作 a"| WS["世界模拟器 W_φ\n（动作条件视频生成/潜空间动力学）"]

    WS -->|"生成想象下一状态 ô"| Reward["外部验证奖励评估器\nR_ext(ô, a)\n(VLM 验证 / 空间价值图 / 稠密进度)"]
    WS -->|"下一状态 ô"| Policy

    Reward -->|"标量奖励 r_t"| RL["强化学习优化器\n(PPO / GRPO / WMPO / PACE)"]
    RL -->|"梯度更新策略参数 θ"| Policy

    style WS fill:#fff4e6,stroke:#d68910,stroke-width:2px
    style RL fill:#fde9e9,stroke:#c0392b,stroke-width:2px
```

### 世界模拟器面临的两大根本挑战与最新破局方案

将生成式世界模型用作 RL 模拟器时，存在两个核心理论难题：
1. **物理幻觉累积（Hallucination Accumulation）**：自回归生成的多步误差随时间累积，出现物体凭空消失、重力失效等虚假动态，RL 策略容易利用模型的物理漏洞获得虚假高分（Adversarial Exploitation）；
2. **策略演化与环境动力学的分布漂移（Distribution Shift）**：随着策略 $$\pi_\theta$$ 持续更新，其探索出的动作序列逐渐脱离世界模型预训练时的数据分布，导致世界模型对新动作的预测精度急剧下降。

**最新破局技术方案**：
- **关键帧初始化回放（Keyframe-Initialized Rollouts, KIR，如 WoVR）**：从专家演示的关键帧（如抓取前夕、对准瞬间）附近初始化短程探索，将预测时域限制在 3–5 步内，从根源上截断误差累积；
- **策略对齐协同演化（Policy-Aligned Co-Evolution, PACE，如 WoVR）**：在 RL 策略演化过程中，定期收集当前策略生成的动作轨迹，对世界模型进行在线增量微调，动态保持模拟器与策略动作分布的同步对齐；
- **基于 MLLM 的可验证与稠密进度奖励（Verified & Dense Progress Rewards，如 VLA-RFT, PRBench, SRPO）**：利用经过专门物理推理训练的 VLM（如 Cosmos-Reason1）或空间价值图（ASVM）提供每一步的稠密进度奖励，而非易受欺骗的简单图像相似度；
- **测试时适应（Test-Time Adaptation, TTA，如 VLA-Reasoner, AdaPower）**：在真实部署测试阶段，允许策略根据环境反馈动态微调世界模型参数，实现在线即时校准。

---

# 4. Cosmos：物理 AI 世界基础模型平台
———World Simulation with Video Foundation Models for Physical AI

📄 **Cosmos-Predict1 (2025)**: [arxiv.org/abs/2501.03575](https://arxiv.org/abs/2501.03575)  
📄 **Cosmos-Predict2.5 (2025/2026)**: [arxiv.org/abs/2511.00062](https://arxiv.org/abs/2511.00062)  
🔗 **代码**: [nvidia-cosmos](https://github.com/nvidia-cosmos)

Cosmos 是 NVIDIA 发布的**物理 AI 世界基础模型平台**。其核心目标是用生成式视频模型替代昂贵的真实世界数据采集与物理仿真器，为机器人、自动驾驶、具身智能等物理 AI 系统提供高质量、大规模、可控的"世界模拟"能力。

与单一视频生成模型不同，Cosmos 是一套**分层平台**：从数据策展基础设施，到多种预训练模型系列，再到面向具体场景的后训练（Post-training）工作流，构成一条完整的工具链。

<div align="center">
  <img src="/images/wm/Cosmos-Platform-Components.webp" width="100%" />
<figcaption>图：Cosmos WFM 平台核心组件：视频数据策展流水线、多模态 Tokenizer、预训练 WFM 与后训练应用样例。</figcaption>
</div>

---

## 4.1 数据基础设施：Cosmos Video Curator

物理 AI 世界模型的训练瓶颈首先是**数据质量**，而非模型能力。Cosmos 开发了名为 **Cosmos Video Curator** 的大规模视频处理流水线，分七个阶段将原始视频转化为高质量训练数据：

<div align="center">
  <img src="/images/wm/Cosmos-Predict2.5-VideoCurationPipeline.webp" width="100%" />
<figcaption>图：Cosmos Video Curator 流水线：原始多领域视频经切分、转码、裁剪、多级过滤、字幕生成、语义去重、结构化分片七个阶段，输出可直接用于大规模预训练的高质量数据集。（图源：Cosmos-Predict2.5）</figcaption>
</div>

1. **镜头感知切分（Shot-Aware Splitting）**：用高精度边界检测模型将长视频分段，剔除镜头切换片段；
2. **GPU 加速转码（Transcoding）**：GPU 加速转码、裁剪黑边，丢弃5秒以下片段；
3. **视频裁剪（Cropping）**：标准化分辨率与画幅比；
4. **多级过滤（Filtering）**：依次经过美学质量、运动检测、OCR 文字、感知质量（DOVER）、语义伪影（VTSS）、VLM 精筛六道过滤，最终仅约 **4%** 的片段通过；
5. **多粒度字幕（Captioning）**：每个片段切为5秒窗口，用 Qwen2.5-VL-7B 生成短/中/长三种粒度的事实性字幕；
6. **语义去重（Deduplication）**：基于嵌入相似度聚类，保留最高分辨率版本，支持增量在线去重；
7. **结构化分片（Sharding）**：按内容类型（26类）、分辨率、宽高比、长度四维度分片，支持课程学习（Curriculum Learning）与细粒度域平衡采样。

**规模**：Cosmos-Predict2.5 流水线处理了数亿条原始视频片段，经严格多阶段精筛（约 4% 通过率）与去重后保留数千万条高质量训练片段（Predict1 时代为 1000 万条）。底层基础设施支持 PB 级处理（Delta Lake 数据湖 + Milvus 向量库），具备 CPU/GPU 动态自动扩缩容能力。

**领域专属数据**：在通用数据之外，Cosmos 针对五个物理 AI 核心领域构建了专属数据集：

| 领域 | 特点 |
| --- | --- |
| 机器人操作 | 汇聚 AgiBot-Beta、GR00T、DROID、OpenX 等主流数据集，统一标注动作类型、运动部位与相机视角 |
| 自动驾驶 | ~310 万条 20 秒环视视频，7路相机同步（前宽/前长/左/右/后/后左/后右） |
| 智能空间 | 工厂、仓库、建筑工地等工业场景（~4 万条），VLM 语义验证后保留 |
| 人类动力学 | YOLOX 人体检测 + RTMPose 姿态估计过滤，聚焦人体动态行为 |
| 物理现象 | 覆盖经典力学、流体力学等可观测物理现象，系统化构建物理接地数据 |

---

## 4.2 模型体系：三大产品线

Cosmos 平台由三条功能互补的模型产品线构成，共同覆盖物理 AI 世界模拟的完整能力谱：

| 模型 | 核心能力 | 典型输入 | 典型输出 |
| --- | --- | --- | --- |
| **Cosmos-Predict** | 未来世界状态预测 | Text / Image / 历史视频 | 未来多秒视频 |
| **Cosmos-Transfer** | 结构化世界翻译（Sim2Real） | 边缘/深度/分割图 | 照片级真实视频 |
| **Cosmos-Reason** | 物理推理 VLM | 视频 + 文本问题 | 带 CoT 自然语言回答 |

### Cosmos-Predict：核心预测引擎

Cosmos-Predict 是平台的**世界前向动力学模型**——给定文本或图像/视频条件，生成未来的世界演化视频。其发展经历了两代：

**Cosmos-Predict1（2025）**：同时提供两种并行架构：

- **扩散模型（Diffusion WFM）**：基于 DiT + Elucidated Diffusion Model（EDM）、T5 文本编码器，擅长生成高视觉质量、3D 空间一致性强的视频。
<div align="center">
  <img src="/images/wm/Cosmos-Predict1-Diffusion-Architecture.webp" width="100%" />
<figcaption>图：Cosmos-Predict1 扩散模型架构：DiT 主干 + T5 文本编码器 + 3D RoPE 位置编码。</figcaption>
</div>

- **自回归模型（Autoregressive WFM）**：将视频视为离散 Token 序列，通过因果 Transformer 进行 Token 预测，适合长序列、交互式展开。
<div align="center">
  <img src="/images/wm/Cosmos-Predict1-Autoregressive-Architecture.webp" width="100%" />
<figcaption>图：Cosmos-Predict1 自回归模型架构：因果 Transformer Token 预测。</figcaption>
</div>

两种模型共用 **Cosmos Tokenizer**——采用小波变换 + 因果 3D 卷积的编解码结构，支持连续表示（供扩散模型使用）和离散表示（供自回归模型使用），在极高压缩比下优于同期 SOTA（如 Video-MAGVIT2）。

<div align="center">
  <img src="/images/wm/Cosmos-Tokenizer-Architecture.webp" width="100%" />
<figcaption>图：Cosmos Tokenizer：基于小波变换的编解码结构，通过因果 3D 卷积捕获时间相关性，同时输出连续 token（供扩散模型）与离散 token（供自回归模型）。</figcaption>
</div>

**Cosmos-Predict2.5（2025/2026）**：全面升级，核心改进包括：

- 将 Diffusion 和 Autoregressive 两条路线**统一为单一 Flow Matching 模型**（Text2World + Image2World + Video2World 三模共用同一套权重）；
- 视觉 Tokenizer 换用 **WAN2.1 VAE**（时间×高×宽方向 4×8×8 压缩，每次生成 93 帧约 5.8 秒）；
- 文本编码器从 T5 升级为 **Cosmos-Reason1**（跨多层激活拼接、投影至 1024 维），提供更细粒度语言接地；
- 移除绝对位置编码、保留相对 3D RoPE，增强对训练外分辨率与序列长度的泛化能力；
- 引入 **RL Post-training**（VideoAlign 奖励 + GRPO 算法），显著提升生成质量与指令对齐。

<div align="center">
  <img src="/images/wm/Cosmos-Predict2.5-Architecture.webp" width="100%" />
<figcaption>图：Cosmos-Predict2.5 整体架构：右侧为 DiT 主干，在潜空间中以"自注意力 → 交叉注意力 → 前馈 MLP"堆叠的 Block 预测去噪速度场，时间步以 AdaLN-LoRA 注入；左侧为 Cosmos-Reason1 文本编码器，跨多层激活拼接后投影为 1024 维文本嵌入，通过交叉注意力层引导视频生成。（图源：Cosmos-Predict2.5）</figcaption>
</div>

可用权重规模：**2B**（轻量部署）与 **14B**（最优质量），均提供 pre-trained / post-trained 两种检查点，以及针对机器人操纵、自动驾驶等场景的领域专属微调版本。

### Cosmos-Transfer：结构化世界翻译

Cosmos-Transfer 不是"凭空预测未来"，而是将**结构化世界表示**翻译成**感官真实的视频**——典型用途是将仿真器（Isaac Sim、CARLA 等）的几何/语义输出提升为照片级真实画面（Sim2Real）。

📄 **Cosmos-Transfer1 (2025)**: [arXiv:2503.14492](https://arxiv.org/abs/2503.14492)

**Cosmos-Transfer1** 在 Cosmos-Predict1（7B 扩散模型）基础上后训练而来，核心创新是**自适应多模态 ControlNet** 架构：

- **多分支 ControlNet**：为每种控制模态独立设置一条控制分支（各含 3 个 Transformer Block，权重从主分支继承初始化）；各分支独立训练，推理时融合——新模态可随时添加而无需重训全模型；
- **时空控制图（Spatiotemporal Control Map）**：引入 $N \times X \times Y \times T$ 维权重张量 $\mathbf{w}$，对每个模态在每个时空位置分配不同影响权重，使模型能在不同区域动态偏向最有用的控制信号——例如前景使用边缘图保持细节，背景允许自由生成。

<div align="center">
  <img src="/images/wm/CosmosTransfer1-AdaptiveControlNet.webp" width="100%" />
<figcaption>图：Cosmos-Transfer1 自适应多模态 ControlNet 架构：每种控制模态对应一条独立控制分支，通过时空控制图 w 加权后注入主 DiT 生成分支，实现位置自适应的多模态融合。（图源：Cosmos-Transfer1）</figcaption>
</div>

**支持的控制模态**（通用版 Transfer1-7B）：

| 模态 | 提取方法 | 保留特性 |
| --- | --- | --- |
| Blur/Vis（模糊视觉） | 双边滤波 | 颜色与大体形状，允许纹理改变 |
| Edge（边缘图） | Canny 算子 | 场景结构与物体边界 |
| Depth（深度图） | DepthAnything2 | 三维几何布局 |
| Segmentation（分割图） | GroundingDino + SAM2 | 语义类别布局 |

自动驾驶专用版（Transfer1-7B-Sample-AV）额外支持 **HDMap**（高精地图，控制车道/路面布局与相对轨迹）和 **LiDAR**（点云投影，保留场景语义细节及光照条件控制）两种模态；另有 **4KUpscaler** ControlNet 可将 720p 生成视频无缝超分至 4K。

<div align="center">
  <img src="/images/wm/CosmosTransfer1-ModalityComparison.webp" width="100%" />
<figcaption>图：Cosmos-Transfer1-7B 四种模态的输入→输出对比（相同文本 prompt）：Vis 保持颜色构图、Edge 保持物体边界、Depth 保持三维几何、Seg 保持语义类别布局，同一张图中不同模态对生成内容的控制侧重各异。（图源：Cosmos-Transfer1）</figcaption>
</div>

**评测**（TransferBench：600 例，含 200 机器人操作 / 200 自动驾驶 / 200 第一人称日常场景）：三项指标——控制信号遵循度（Adherence）、生成多样性（Diversity）、整体质量（Quality）——Transfer1-7B [Seg] 与 [Depth] 在各场景均达到最优控制精度。

**Cosmos-Transfer2.5** 继承全部模态能力，模型尺寸缩小 **3.5×**（7B → ~2B），PAIBench-Transfer 整体质量评分从 6.56 提升至 9.75，并新增 **RNDS** 指标衡量长视频质量退化。

<div align="center">
  <img src="/images/wm/Cosmos-Predict2.5-TransferComparison.webp" width="100%" />
<figcaption>图：Cosmos-Transfer2.5 vs Transfer1-7B 效果对比：Transfer2.5 在 Prompt 对齐、错误累计、控制信号遵循三项均优于前代，且模型尺寸缩小 3.5×。（图源：Cosmos-Predict2.5）</figcaption>
</div>

### Cosmos-Reason：物理推理 VLM

Cosmos-Reason 是一个专为物理 AI 强化推理能力的视觉语言模型，输出带 **Chain-of-Thought** 的自然语言回答。

📄 **Cosmos-Reason1 (2025)**: [arXiv:2503.15558](https://arxiv.org/abs/2503.15558)

#### 两套本体论（Ontology）

Cosmos-Reason1 从两个维度定义物理 AI 推理能力：

**① 物理常识本体（Physical Common Sense Ontology）**：三大类、16 细分子类：

- **Space（空间）**：Relationship（空间关系）、Plausibility（合理性）、Affordance（可操作性）、Environment（环境理解）
- **Time（时间）**：Actions（动作）、Order（顺序）、Causality（因果）、Camera（镜头运动）、Planning（规划）
- **Fundamental Physics（基础物理）**：Attributes（属性）、States（状态变化）、Object Permanence（物体恒存性）、Mechanics（力学）、Electromagnetism（电磁学）、Thermodynamics（热力学）、Anti-Physics（反物理违例）

<div align="center">
  <img src="/images/wm/CosmosReason1-CommonSenseOntology.webp" width="80%" />
<figcaption>图：Cosmos-Reason1 物理常识本体：三大类（Space、Time、Fundamental Physics）划分为 16 个细粒度子类，定义 Physical AI 模型应具备的感知与推理能力边界。（图源：Cosmos-Reason1）</figcaption>
</div>

**② 具身推理本体（Embodied Reasoning Ontology）**：4 种能力 × 2 类智能体（自然智能体 / 机器人系统）的二维矩阵，重点考察三项核心能力：**任务完成验证**（Task-Completion Verification）、**动作可操作性**（Action Affordance）、**下一步动作预测**（Next Plausible Action Prediction）。

#### 模型架构

Cosmos-Reason1 为 decoder-only 多模态 LLM：视频帧经 Vision Encoder 提取视觉 token，Projector 下采样后与文本 token 拼接输入 LLM 主干，输出带 `<think>...</think>` 标签的长链推理回答。

| 配置 | Cosmos-Reason1-7B | Cosmos-Reason1-56B |
| --- | --- | --- |
| Vision Encoder | ViT-676M（动态分辨率） | ViT-300M（固定 448×448） |
| LLM 架构 | 密集 Transformer（28 层） | Mamba-MLP-Transformer 混合（118 层） |
| LLM 预训练底座 | Qwen2.5-VL | Nemotron-H |

56B 模型采用 **Mamba-MLP-Transformer 混合架构**：交替 Mamba-MLP 模块（线性时间复杂度，高效处理长视频序列）+ 稀疏 Transformer 层（负责长程细节捕获），兼顾效率与能力。

<div align="center">
  <img src="/images/wm/CosmosReason1-Architecture.webp" width="100%" />
<figcaption>图：Cosmos-Reason1 多模态 LLM 架构：视频输入经 Vision Encoder 和 Projector 投影为视觉 token，与文本 token 拼接后输入密集 Transformer（7B）或混合 Mamba-MLP-Transformer（56B）主干，输出带长链推理过程的自然语言回答。（图源：Cosmos-Reason1）</figcaption>
</div>

#### 训练范式：Physical AI SFT + Physical AI RL

**Physical AI SFT** — 精选约 4M 视频-文本标注对，分两类：

- **理解型**（Understanding）：VQA 自由问答 + 多项选择题，覆盖物理常识与具身推理所有子类（约 1.81M 条）；
- **推理型**（Reasoning）：带完整 CoT 推理链的标注，由 DeepSeek-R1 蒸馏生成，训练模型输出 step-by-step 思考过程（约 1.93M 条）。

**Physical AI RL** — 在 SFT 基础上采用 **GRPO** 算法进一步强化物理推理：

- **奖励信号**：多项选择题规则验证奖励（可自动校验，免人工标注）；
- **自监督 MCQ**：从视频数据结构自动生成——如打乱时空块要求模型还原（Spatiotemporal Puzzle）、判断视频播放方向（Time Arrow），完全免标注且与物理 AI 能力高度相关；
- **异步训练框架**：Dispatcher–Actor Rollout–Policy Training 三节点异步设计，通信效率较主流共置框架提升约 **160%**，并支持节点故障动态恢复与弹性扩缩容。

**在平台内的多重角色**：

- **裁判**：评估 Predict / Transfer 输出的物理合理性（自动化评估流水线）；
- **文本编码器**：Predict2.5 直接将 Reason1 用作多层激活拼接的文本编码器，替代 T5，提供更细粒度语言接地；
- **规划器**：作为机器人 / VLA 的高层任务分解与 Affordance 推理模块；
- **数据策展**：VLA 合成数据自动打标与质量过滤。

---

## 4.3 训练范式：预训练 + 三阶段后训练

<div align="center">
  <img src="/images/wm/Cosmos-Training-Paradigm.webp" width="100%" />
<figcaption>图：Cosmos 训练范式：通用物理知识大规模预训练 → 领域 SFT → 模型融合 → RL 后训练，最终微调适配各类下游 Physical AI 任务。</figcaption>
</div>

Cosmos-Predict2.5 采用**预训练 → SFT → 模型融合 → 强化学习**四阶段渐进范式：

**① 大规模预训练（Pre-training）** — 课程学习，逐步提升任务难度与分辨率：

| 阶段 | 任务 | 分辨率 | 帧数 |
| --- | --- | --- | --- |
| 1 | Text2Image | 256p（320×192） | 1 |
| 2 | Text2Image + Image/Video2World | 256p | 1 / 93 |
| 3 | Text2Image + Image/Video2World | 480p → 720p | 1 / 93 |
| 4 | 全部（含 Text2World） | 720p（1280×704） | 1 / 93 |

**② 领域 SFT（Supervised Fine-tuning）** — 在高质量领域数据上独立训练专域模型（每域 30K 步，batch size 256）：

| 领域 | 视频规模 |
| --- | --- |
| 物体持久性（Object Permanence） | 10.4M |
| 高动态（High Motion） | 1.0M |
| 复杂场景（Complex Scenes） | 1.6M |
| 驾驶（Driving） | 3.1M |
| 机器人操纵（Robotic Manipulation） | 730K |

**③ 模型融合（Model Merging）** — 将多个专域 SFT 模型通过参数插值（Model Soup、TIES、DARE-TIES 等）融合为单一模型，在保留专域能力的同时维持通用生成质量。人类偏好评测中，融合后模型在所有领域均优于任一单独 SFT 模型。

**④ 强化学习（RL Post-training）** — 以 **VideoAlign**（VLM-based 奖励，三维度：文本对齐 + 运动质量 + 视觉质量）为奖励信号，使用 **GRPO** 算法进行 RL 训练（256 步，batch size 32）。人类评测中 RL 后生成视频胜率较 RL 前提升约 20 个百分点。

**加速推理（Timestep Distillation）**：采用 rCM 混合蒸馏框架将推理步数压缩至 **4 步**，PAI-Bench 总分损失小于 0.005。

**训练规模**：4096 台 NVIDIA H100 GPU，2B 模型 MFU 约 36.5%，14B 模型约 33.1%；采用 FSDP2 混合并行 + Ulysses 上下文并行 + 选择性激活检查点（SAC）等多项系统优化。

---

## 4.4 典型应用场景

Cosmos 平台在六个物理 AI 场景上展示了多样的适用性：

**① 机器人策略视觉增强**：用 Cosmos-Transfer2.5 对机器人演示视频进行外观多样化（替换背景、更改物体颜色、添加干扰物），以合成数据增强策略训练。在对抗性视觉扰动（不可见背景、物体变化）评测中，Cosmos 增强数据训练的策略成功率显著高于仅用真实数据训练的基准。

**② 自动驾驶多视角仿真**：以世界场景图（HD map + 语义信息）为条件，驱动 Cosmos-Transfer2.5 生成 7 路时空同步的环视视频，覆盖多样天气、光照、交通密度，用于驾驶策略闭环训练与测试。

**③ 相机可控多视角生成**：对 Cosmos-Predict2.5 进行相机位姿条件化后训练，支持任意相机外参组合下的跨视角一致生成。

**④ VLA 合成训练数据**：以单张初始帧 + 动作条件驱动 Cosmos-Predict2.5 生成机器人操纵视频序列，配合 Cosmos-Reason1 自动打标与质量过滤，低成本构建大规模 VLA 训练数据集。

**⑤ 动作条件世界生成（Action-conditioned World Generation）**：对 Cosmos-Predict2.5 进行低级动作（关节角度 / 末端轨迹）条件化后训练，实现动作驱动的未来视频预测，可直接用于策略评估的闭环模拟。

<div align="center">
  <img src="/images/wm/Cosmos-Predict2.5-PredictionSamples.webp" width="100%" />
<figcaption>图：Cosmos-Predict2.5-2B post-trained 模型在 PAI-Bench 上的生成样本：覆盖自动驾驶（上两行）、工业机器人操纵（中三行）、人类动力学（下行）等多个物理 AI 场景，展示了模型在时序一致性与物理合理性上的能力。（图源：Cosmos-Predict2.5）</figcaption>
</div>

<div align="center">
  <img src="/images/wm/Cosmos-Physics-Alignment.webp" width="100%" />
<figcaption>图：受控物理场景仿真对比（Cosmos-Predict1）：通过重力、碰撞等受控实验验证 Cosmos WFM 对牛顿力学的遵循程度，预测结果接近专用物理引擎。</figcaption>
</div>

---

## 4.5 开源生态：Cosmos Cookbook

官方仓库 [nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) 提供端到端、可直接跑通的 recipes 集合：

- **推理脚本**：Predict（Text2World / Image2World / Video2World）、Transfer（多模态控制信号混合）、Reason 三产品线的最小可运行示例；
- **后训练模板**：相机控制、机器人操纵、自动驾驶下游任务的标准微调配置（含 LoRA / 全量微调策略与超参）；
- **数据策展**：Cosmos Video Curator 自定义数据集接入流程；
- **安全护栏（Guardrail）**：输入 prompt 到输出内容全链路安全检测调用示例；
- **部署工具链**：与 NVIDIA NeMo、Isaac Sim、TensorRT-LLM 的集成示例。

<div align="center">
  <img src="/images/wm/Cosmos-Guardrail-Overview.webp" width="100%" />
<figcaption>图：Cosmos Guardrail 架构：涵盖从输入 prompt 到输出内容的完整安全检测流程。</figcaption>
</div>

对具身 AI 研究者而言，Cosmos 的实用价值在于：**无需从头训练，可直接调用 Predict 做滚动仿真、Transfer 做大规模 Sim2Real 数据增强、Reason 做自动化物理合理性评估**，将科研原型到规模化应用的门槛大幅降低。

---

## 4.6 Cosmos 3：全模态统一世界模型（2026）
———Omnimodal World Models for Physical AI

📄 **Cosmos 3 (2026)**: [arxiv.org/abs/2606.02800](https://arxiv.org/abs/2606.02800)  
🔗 **代码/权重**: [github.com/nvidia/cosmos](https://github.com/nvidia/cosmos) · [huggingface.co/collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)（OpenMDW-1.1 License）  
💡 **专题详解**: 关于 Mixture-of-Transformers (MoT) 架构的详细数学拆解与分析，可参考我的专题博客 [Mixture-of-Transformers (MoT) 架构详解](/mixture-of-transformers/)。

如果说 §4.1–§4.5 的 Cosmos 平台是用**一条工具链串起多个专用模型**（Predict 预测、Transfer 翻译、Reason 推理各司其职），那么 2026 年 6 月 NVIDIA 发布的 **Cosmos 3** 则把这条路线推到了终点：**用单一网络架构同时完成理解与生成、并原生覆盖语言、图像、视频、音频、动作五大模态**。它直接把本文 §3 的四大范式——视觉语言模型（VLM）、视频生成 / 前向动力学模型（世界合成器/模拟器）、世界动作模型（WAM/VLA）——**吸收进同一个模型**，是"单模型、多范式角色"趋势（参见 §3.1 的 GENE-26.5 讨论与 §9.3）最彻底的一次工程落地。

<div align="center">
  <img src="/images/wm/Cosmos3-Fig1-UnifiedBackbone.webp" width="100%" />
<figcaption>图：Cosmos 3 作为 Physical AI 的通用骨干。仅通过改变输入-输出配置，同一套权重即可化身视觉语言模型、图像生成模型、音视频生成模型、策略/世界动作模型、前向动力学模型、逆动力学模型，无需任何结构改动。（图源：Cosmos 3）</figcaption>
</div>

### 核心动机：终结"范式割裂"

论文的出发点是一个尖锐的判断：**理解与生成被人为割裂是根本性的局限**。以"晚餐后清理餐桌"的家用机器人为例，当前范式需要拼装一条割裂的流水线——VLM 定位餐具并生成计划、VLA/WAM 生成动作序列、前向动力学模型（"世界模型"）仿真并评估未来状态。这种碎片化架构既不优雅也浪费算力。Cosmos 3 的主张是：理解本就需要推理"世界如何演化、动作有何后果"，而生成本就依赖"对世界与行为的紧凑结构化表示"——两者应当统一在**一个可扩展框架**里。

### 架构：双塔 Mixture-of-Transformers（MoT）

Cosmos 3 的核心是一个 **MoT 双塔**结构：把一条 token 序列切成两段——前段是**自回归（AR）子序列**负责理解推理，后段是**扩散（DM）子序列**负责生成。每个 Transformer 解码层内部都并行持有**两套独立参数**（Reasoner 塔 + Generator 塔），二者均从预训练 VLM 权重初始化，从而继承强语言/视觉推理能力。

<div align="center">
  <img src="/images/wm/Cosmos3-Fig5-MoTArchitecture.webp" width="100%" />
<figcaption>图：Cosmos 3 的 MoT 架构。同一条序列由 AR 子序列（语言 + ViT 视觉 token，以 EOS/BOG 收尾）与 DM 子序列（VAE 视觉、音频、动作 token，训练时加噪）拼接而成；层内 AR 与 DM token 各用独立 LayerNorm 与 MLP（均由预训练 VLM 共同初始化），仅在共享自注意力处交汇。右图为注意力掩码：AR 为因果三角、DM 为全注意力。（图源：Cosmos 3）</figcaption>
</div>

两塔虽参数独立，却通过**双流联合注意力（Dual-Stream Joint Attention）**耦合：

- **AR 子序列**使用**因果自注意力**，只能看到自身前序 token——完整保留了从 VLM 继承的自回归文本生成能力（语言走 next-token prediction）；
- **DM 子序列**使用**全双向注意力**，以 AR 与 DM token 的并集为 Key/Value，使每个扩散 token 都能自由"读取"文本提示与所有条件帧（生成走迭代去噪，Flow Matching 预测速度场）；
- **关键约束**：AR token 永远不会被 DM token 更新——保证了条件通路的因果完整性。

这种设计的精妙之处在于：**理解（AR）为生成（DM）提供语义条件，而生成不污染理解**，二者在同一张注意力图里完成协作，却互不破坏各自的归纳偏置。

**编码器**：视觉理解用与语言对齐预训练的 **ViT**（随骨干联合训练），视觉生成用 **Wan2.2-TI2V-5B 的视频 VAE**（冻结，时间 4×、空间 32×32 压缩）；音频用冻结的音频 VAE（48kHz 立体声，25 token/秒）；动作用域感知投影层。位置编码采用带**绝对时间调制的 3D MRoPE**，把不同帧率/采样率的视频、音频、动作 token 对齐到同一条物理时间轴上。

### 把"动作"当作一等模态

与多数工作把动作当作附属输出不同，Cosmos 3 显式引入一类**动作 token**，作为连接物理世界与语言推理、视频建模的桥梁。它用一套**统一动作表示**容纳异构本体（自动驾驶、相机运动、第一人称人体、单臂/双臂/人形机器人）：自我位姿（Ego Pose 9D）与执行器位姿（Effector Pose 9D）以"3D 平移 + 6D 旋转"的相对位姿伪动作表示，抓取状态（Grasp State）直接编码当前操作状态。各本体用**域感知的输入/输出投影**适配不同维度，同时共享 MoT 骨干。动作 token $a_t$ 表示从视频状态 $v_{t-1}$ 到 $v_t$ 的转移。

<div align="center">
  <img src="/images/wm/Cosmos3-Fig3-UnifiedActionRepresentation.webp" width="100%" />
<figcaption>图：统一动作表示。异构本体的控制被映射为由共享几何分量构成的紧凑动作向量——Ego/Effector 运动编码为相对位姿伪动作（3D 平移 + 6D 旋转），抓取状态直接编码指尖位置或夹爪开合。（图源：Cosmos 3）</figcaption>
</div>

正因为动作与视频被纳入同一序列模型，Cosmos 3 仅靠"哪些 token 干净、哪些 token 加噪"的不同配置，就统一了三种动作生成模式：

<div align="center">
  <img src="/images/wm/Cosmos3-Fig4-ActionModes.webp" width="95%" />
<figcaption>图：三种动作模式由 token 加噪配置决定。前向动力学（给定干净动作去噪视频）、逆动力学（给定干净视频去噪动作）、策略（同时去噪视频与动作）。（图源：Cosmos 3）</figcaption>
</div>

- **前向动力学（Forward Dynamics）**：以观测上下文 + 干净动作为条件，预测未来视觉状态——即 §3.4 的世界模拟器；
- **逆动力学（Inverse Dynamics）**：从观测到的视觉转移反推动作——即 §3.3 世界合成器中常用的 IDM 标注器；
- **策略（Policy）**：同时预测动作与视频，既给出"干预"又给出"预期视觉后果"——即 §3.2 的世界动作模型。

加上作为 VLM 的纯语言理解、Text2Image、Text2Video（可联合生成音频）、Image/Video2Video、Video Transfer 等生成模式，**四大范式在 Cosmos 3 中第一次由同一套权重原生支持**。

### 模型变体：Edge / Nano / Super

三个尺度覆盖从端侧部署到数据中心推理。注意总参数约为稠密 Transformer 的 2 倍——这正是双塔（Reasoner + Generator 各持一套参数）的代价：

| 变体 | 总参 / 稠密骨干 | 层数 | 隐藏维 | 注意力头 | KV 头 | FFN 维 | 初始化 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Cosmos3-Edge | 4B / 2B | 28 | 2,048 | 16 | 8 | 9,216 | 从零训练（类 Qwen3-1.7B） |
| Cosmos3-Nano | 16B / 8B | 36 | 4,096 | 32 | 8 | 12,288 | Qwen3-VL 8B |
| Cosmos3-Super | 64B / 32B | 64 | 5,120 | 64 | 8 | 25,600 | Qwen3-VL 32B |

本次发布 Nano 与 Super，Edge 留待后续。Reasoner 在约 **24.2M** 样本（22.0M 预训练 + 2.2M SFT）的图文/视频-文本对上训练；Generator 在大规模图像/视频/音频/动作语料上以重建目标（Flow Matching）训练，经历预训练 → 中训练（mid-training）→ T2I 后训练 → I2V 后训练 → 机器人策略后训练的多阶段课程。

### 训练范式与 Physical AI 角色

<div align="center">
  <img src="/images/wm/Cosmos3-Fig2-TrainingParadigm.webp" width="92%" />
<figcaption>图：Cosmos 3 是训练 Physical AI 智能体的强起点。预训练 + 中训练得到通用底座后，可在目标数据上无结构改动地后训练，分别服务于合成数据生成、任务专域特化、闭环训练环境三类用途。（图源：Cosmos 3）</figcaption>
</div>

Cosmos 3 把自己定位为破解"数据与环境扩展瓶颈"的三重起点：(i) **合成数据生成**——后训练为更强的 T2I / I2V 生成器，低成本合成高保真多样视觉数据；(ii) **任务专域特化**——在共享底座上做本体/任务特定微调，保留统一世界表示；(iii) **训练环境**——长期目标是生成高质量、可交互的复杂环境用于闭环训练。论文同时开源了 5 个合成数据集（SDG-PhyxSim / RobotSim / DriveSim / SynHuman / Warehouse）与评测基准 **Cosmos-HUE**。

### 核心结果

撰写技术报告时，Cosmos 3 的后训练变体取得多项 SOTA：

- **Cosmos3-Super-Text2Image**：Artificial Analysis 文生图榜单**开源权重第 1**（含闭源模型计第 4，日期 2026-05-28）；
- **Cosmos3-Super-Image2Video**：Artificial Analysis 图生视频榜单**开源权重第 1**，整体优于 Veo-3.1 等强闭源模型；
- **Cosmos3-Nano-Policy-DROID**：在 RoboLab 与 **RoboArena** 真机策略评测中**均排名第 1**（从中训练 Nano 续训，在 DROID 76k 轨迹上后训练，15Hz 联合输出动作与未来视频帧）；
- 在机器人、智能空间、自动驾驶领域的推理任务上同时超越开源与闭源模型（机器人仅略逊 Gemini 3.1 Pro），且两代视频生成均显著优于前代 Cosmos-Predict2.5。

### 与四大范式的关系

Cosmos 3 是本文叙事的一个**收敛点**。GENE-26.5（§3.1）已展示"联合分布 + 条件查询"如何让单模型兼任多角色，而 Cosmos 3 把这一思路推广到全模态、并以双塔 MoT 给出更清晰的工程边界：**Reasoner 塔承担世界规划器的高层语义推理，Generator 塔以前向动力学/视频生成承担世界合成器与世界模拟器，Policy 模式则是世界动作模型**。§2.5 时间线中 2025–2026 年世界合成器/模拟器的爆发，最终汇流为"理解-生成-动作一体化"的全模态世界模型——这与 §9.3"从想象到验证再到规划"的判断完全一致，也指向 §8 中长航程前瞻、4D 感知、物理一致性等方向的统一载体。

---

# 5. 经典代表性工作

本章节系统剖析具身智能世界模型演进过程中的 16 项里程碑式前沿研究。为了便于读者快速查阅与横向对比，下表汇总了各代表性工作的发表时间、所属技术范式、核心技术机制、关键基准指标及正文跳转导航：

### 具身世界模型 16 项代表性工作速查表

| 论文 / 模型 | 年份 | 所属技术范式 | 核心技术机制 | 关键性能 / 评测表现 | 跳转锚点 |
|:---|:---:|:---|:---|:---|:---:|
| **Lyra 2.0** | 2026 | 世界模拟器 / 空间智能 | 几何记忆解耦 + 空间记忆检索路由 + 自增强去漂移 | 800 帧以上长程生成仍保持几何一致，3DGS 重建近乎无浮点伪影 | [查看详情](#sec-5-1-lyra) |
| **Genie** | 2024 | 世界合成器 / 基础环境生成 | 潜在动作模型 (LAM) + ST-Transformer + MaskGIT 动力学（11B） | 无动作标注即挖掘出 8 维可控动作空间，任意图像转可玩环境（160×90） | [查看详情](#sec-5-2-genie) |
| **VLA-World** | 2026 | 世界规划器 / 端到端自动驾驶 | 单帧未来生成 + 反思推理（Think with Generated future）+ GRPO | nuScenes 碰撞率 1.09% → 0.94%，FID 更优 | [查看详情](#sec-5-3-vla-world) |
| **WorldVLA** | 2025 | 世界动作模型 (WAM) | 统一自回归骨干 + 因果注意力掩码 + 视频预测预训练 | LIBERO Avg 81.8%，多视角时空一致生成 | [查看详情](#sec-5-4-worldvla) |
| **WoVR** | 2026 | 世界模拟器 (RL) | 梦境内部强化学习 + 关键帧初始化 (KIR) + 策略协同演化 (PACE) | LIBERO 平均 SR 39.95% → 69.2%；真机 61.7% → 91.7%；生成 23 FPS | [查看详情](#sec-5-5-wovr) |
| **Janus-Pro** | 2025 | 多模态统一骨干 | 理解与生成解耦表征 + 纯自回归 Transformer | MMBench 79.2、GenEval 0.80、DPG-Bench 84.19（7B） | [查看详情](#sec-5-6-janus-pro) |
| **VideoGen Survey** | 2026 | 机器人视频生成体系 | 视频生成在具身控制中的应用分类体系与评测全景 | 系统梳理数据、模型架构与下游策略评测基准 | [查看详情](#sec-5-7-videogen) |
| **AIM** | 2026 | 世界动作模型 (WAM) | 空间价值图 (ASVM) + 意图因果注意力 + 价值自蒸馏 RL | RoboTwin 2.0 Avg SR 93.1% (SOTA) | [查看详情](#sec-5-8-aim) |
| **LingBot-World** | 2026 | 世界模拟器 / 交互生成 | 开源实时交互 + 分层语义数据引擎 + 60s 长程记忆 | 16 fps 实时推演（亚秒级延迟），60s 场景重访结构高度一致 | [查看详情](#sec-5-9-lingbot-world) |
| **Marble (World Labs)** | 2025/2026 | 大型世界模型 (LWM) / 空间智能 | 3DGS 空间持久宇宙 + Chisel 结构解耦 + World API | Forbes AI 50，单图/文本生成无限可探索 3D 世界 | [查看详情](#sec-5-10-marble) |
| **SANA-WM** | 2026 | 世界模拟器 / 高效分钟级生成 | 混合线性 GDN/Softmax + 双分支相机控制 (UCPE+Plücker) + 两阶段精化 | VBench Overall 80.62/81.89；RTX 5090 上 34s 生成 60s 720p | [查看详情](#sec-5-11-sana-wm) |
| **Qwen-RobotWorld** | 2026 | 基础世界模型 / 空间智能 | 双流 MMDiT + 空间-物理-时序三位一体训练 + 跨本体接地 | EWMBench 4.60 (#1)，WorldModelBench 8.99 | [查看详情](#sec-5-12-qwen-robotworld) |
| **Wan2.1** | 2025 | 视频扩散基础模型 | Spatio-Temporal VAE (4x8x8) + AdaLN 共享 + VACE 框架 | 开源 T2V/I2V 基础底座，消费级 4090 可跑 | [查看详情](#sec-5-13-wan) |
| **Motus** | 2025/2026 | 世界动作模型 (WAM) | MoT 三专家解耦 FFN + 光流潜动作 VAE + 六层数据金字塔 | RoboTwin 2.0 Avg 87.8%，单步 80ms 极速闭环 | [查看详情](#sec-5-14-motus) |
| **NavWAM & WAM-Nav** | 2026 | 世界动作模型 / 具身导航 | 统一潜时空画布 + 零 CEM 在线搜索 + 双流特征融合 | 推理 205.7 ms（5Hz），较 NWM 快 1100×；真机成功率 79.2% / 85% | [查看详情](#sec-5-15-navwam) |
| **Image2Sim** | 2026 | 世界模拟器 / 空间智能 | 前馈 3D 特征高斯锚定 + 单步 Pixel Flow (MeanFlow) 渲染 | 全景 RGB-D 45.6 FPS；自动构建 2 万个环境，R2R-CE 零样本 70.3% | [查看详情](#sec-5-16-image2sim) |

---

## 5.1 Lyra 2.0 (2026) {#sec-5-1-lyra}
———Explorable Generative 3D Worlds at Scale

📄 **Paper**: [https://arxiv.org/abs/2604.13036](https://arxiv.org/abs/2604.13036)

### 精华

NVIDIA 推出的 Lyra 2.0 解决了长程（Long-horizon）3D 一致性场景生成的两大核心痛点，值得借鉴的点包括：
1. **解耦几何与外观（Decoupled Memory）**：将显式 3D 几何（点云缓存）仅用于信息路由和建立像素级对应关系，而将外观合成交给 Diffusion Model 的强生成先验，有效避免了渲染伪影的传播。
2. **空间记忆路由（Anti-forgetting）**：通过几何感知检索机制，即便在长距离移动或重新访问（Revisit）区域时，也能通过 3D 投影检索最相关的历史帧，克服了 Transformer 有限上下文导致的"空间遗忘"。
3. **自增强训练（Self-augmentation）**：在训练阶段引入带有自身预测偏差的损坏数据，使模型学会纠正自回归生成的漂移（Temporal Drifting），而非让误差无限累积。
4. **生成式重建（Generative Reconstruction）**：展示了如何通过视频生成模型合成高一致性的多视角序列，进而驱动 Feed-forward 3DGS 模型快速重建高质量 3D 场景资产。

---

### 1. 研究背景/问题

当前的视频生成模型在生成长视频时极易出现**空间遗忘（Spatial Forgetting）**和**时间漂移（Temporal Drifting）**。当相机移动超出模型的有限上下文窗口时，模型会丢失对早先场景的记忆，导致回看时场景结构崩溃；同时，自回归生成的微小误差会随时间累积，造成颜色偏移和几何扭曲。这限制了生成式 3D 场景重建向大规模、可探索环境的扩展。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/wm/Lyra2-Overview.webp" width="100%" />
<figcaption>图：Lyra 2.0 能够从单张图像出发，支持长程、3D 一致的场景生成与探索，并能导出为高质量 3D 资产。（图源：Lyra 2.0）</figcaption>
</div>

Lyra 2.0 的核心是一个基于"检索-生成-更新"的自回归循环：

1. **抗遗忘机制（Anti-Forgetting）**：
<div align="center">
  <img src="/images/wm/Lyra2-Method-Overview.webp" width="100%" />
<figcaption>图：方法概览：左侧为交互式探索循环，右侧展示了如何从空间记忆中检索历史帧并注入到 DiT 注意力机制中。（图源：Lyra 2.0, 2026）</figcaption>
</div>
系统维护一个 3D 缓存（3D Cache），存储每帧的深度图和点云。在生成下一段视频时，系统会根据当前相机视角，通过投影计算可见度（Visibility Score），检索出最相关的历史帧。

2. **几何引导的上下文注入**：
检索到的历史帧不会直接作为 RGB 图像输入，而是通过**正则化坐标映射（Canonical Coordinate Warping）**建立像素级对应关系。这种方式将几何约束与外观生成分离，允许视频模型在不引入渲染噪声的前提下保持空间一致性。

3. **抗漂移训练（Anti-Drifting）**：
采用了**自增强训练策略（Self-augmentation Training）**。在训练时，模型不仅在完美的高清图像上训练，还会随机在自己生成的"损坏"潜变量（Latent）上进行去噪。这教导模型在推理过程中识别并修正微小的漂移误差，而非放大它们。

4. **实时交互与 3D 导出**：
<div align="center">
  <img src="/images/wm/Lyra2-Applications.webp" width="100%" />
<figcaption>图：Lyra 2.0 应用：交互式 GUI 允许用户自定义轨迹，生成的场景可直接导入 NVIDIA Isaac Sim 进行具身智能仿真。（图源：Lyra 2.0, 2026）</figcaption>
</div>

---

### 3. 核心结果/发现

- **长程一致性**：实验表明，Lyra 2.0 在 800 帧以上的生成序列中仍能保持极其稳定的几何结构和风格一致性，显著优于 GEN3C 和 SPMem 等基线方法。
<div align="center">
  <img src="/images/wm/Lyra2-Video-Comparison.webp" width="100%" />
<figcaption>图：视频生成对比：Lyra 2.0 在长程探索中展现了更强的真实感和更少的几何畸变。（图源：Lyra 2.0, 2026）</figcaption>
</div>

- **高质量 3D 重建**：生成的视频序列通过微调后的 feed-forward 3DGS 流程，可以生成几乎无伪影（Floater-free）的高质量 3D 高斯泼溅模型。
<div align="center">
  <img src="/images/wm/Lyra2-3DGS-Comparison.webp" width="100%" />
<figcaption>图：3DGS 重建对比：Lyra 2.0 生成的视频驱动的重建结果在保真度和一致性上大幅领先。（图源：Lyra 2.0, 2026）</figcaption>
</div>

- **具身智能赋能**：
<div align="center">
  <img src="/images/wm/Lyra2-In-the-Wild.webp" width="100%" />
<figcaption>图：野外场景生成：模型展现了极强的泛化能力，能够处理从室内书房到室外街道、沙漠和古建筑等多样化环境。（图源：Lyra 2.0, 2026）</figcaption>
</div>

---

### 4. 局限性
目前 Lyra 2.0 主要聚焦于静态场景的生成，尚未显式建模动态物体（如行人和车辆）。此外，模型生成的质量仍然受限于训练数据（如 DL3DV）中的光照变化和曝光差异。

---

## 5.2 Genie (2024) {#sec-5-2-genie}
———Generative Interactive Environments

📄 **Paper**: [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)

### 精华

Genie 是首个仅通过无标注视频学习而成的生成式交互环境（Foundation World Model），其核心贡献在于：1) **无监督动作挖掘**：通过潜动作模型（LAM）从纯视频中自动挖掘可控动作空间，解决了世界模型对真实动作标签的依赖；2) **高效时空架构**：设计了基于 ST-Transformer 的计算架构，使显存占用随帧数线性增长，支持长序列视频生成；3) **具身智能底座**：不仅能将任意图像（素描、照片等）转化为可玩的游戏世界，还展现了在机器人操作和智能体训练方面的巨大潜力，为“通向通用智能体的路径”提供了海量仿真数据。

---

### 1. 研究背景/问题

当前的生成式 AI（如 ChatGPT, DALL-E）在文本和图像领域取得了巨大成功，但视频生成模型（如 Video Diffusion）大多缺乏细粒度的交互控制能力。传统的“世界模型”通常需要大量带有真实动作标签（Action Labels）的数据进行训练，这在互联网海量视频面前成了瓶颈。Genie 旨在通过 20+ 万小时无标注互联网视频（经过滤清洗后构建了约 3 万小时 / 680 万段 16 秒高质量 2D 平台游戏片段训练集），学习一个能实时响应用户操作、具有物理常识且能无限生成的交互式环境。

---

### 2. 主要方法/创新点

Genie 是一个参数量达 110 亿的基础模型，其架构由三个深度集成的组件构成，全部基于改进的 **ST-Transformer**。

<div align="center">
  <img src="/images/wm/Genie-components.webp" width="100%" />
<figcaption>图：Genie 整体训练框架：包含视频分词器、潜动作模型 (LAM) 和动力学模型。（图源：Genie, 2024）</figcaption>
</div>

#### 2.1 潜动作模型 (Latent Action Model, LAM)
这是 Genie 的灵魂所在。为了在没有动作标签的情况下实现控制，LAM 采用 VQ-VAE 结构：
- **编码器**：同时接收当前帧和下一帧，输出一个离散的潜动作 $$\mathbf{a}_t$$（通常限制在 8 个离散值以内，以模拟控制器按键）。
- **瓶颈机制**：由于解码器只能通过历史帧和 $$\mathbf{a}_t$$ 来预测下一帧，模型被迫将视频中最具语义一致性的变化（如人物的左右移动、跳跃）编码进这 8 个 Token 中。
- **一致性**：实验证明，即使在不同游戏中，相同的潜动作 Token 往往对应相同的物理语义（如 Action 0 始终代表左移）。

<div align="center">
  <img src="/images/wm/Genie-LAM.webp" width="70%" />
<figcaption>图：潜动作模型 (LAM)：通过重构目标实现无监督动作挖掘。（图源：Genie, 2024）</figcaption>
</div>

#### 2.2 视频分词器 (Video Tokenizer)
Genie 提出了 **ST-ViViT** 架构：
- **时空压缩**：不同于常规只在空间维度压缩的分词器，ST-ViViT 在编码和解码时都引入了时间轴。
- **效率优化**：通过交替使用空间注意力和时间注意力，模型避免了计算量随时间呈平方级增长的问题，保证了在大规模数据集上的训练可行性。

#### 2.3 动力学模型 (Dynamics Model)
基于 **MaskGIT** 的掩码自回归模型：
- **输入**：接收当前视觉 Token 和用户选择的潜动作。
- **预测**：模型在隐空间内预测下一帧的 Token。通过海量数据的“喂养”，模型学习到了复杂的 2D 平台游戏规则，如碰撞、重力、敌人交互和屏幕卷轴滚动。

<div align="center">
  <img src="/images/wm/Genie-ST-transformer.webp" width="55%" />
<figcaption>图：ST-transformer：交替进行空间与时间层计算，实现线性复杂度。（图源：Genie, 2024）</figcaption>
</div>

---

### 3. 核心结果/发现

*   **“化腐朽为神奇”的生成能力**：用户可以上传一张手绘草图、真实的自然景观照片，甚至是通过文生图模型（如 Imagen）生成的图片，Genie 都能立即将其转化为一个可以“玩”的横版过关游戏环境。
*   **语义一致的操控感**：在 Platformers 数据集上，潜动作展现了极强的泛化性。用户点击对应的潜动作，角色会做出连贯的位移或跳跃，且这种操控在视觉风格迥异的环境中依然有效。
*   **机器人领域的潜力**：研究人员在 RT1 机器人数据集上验证了 Genie。模型不仅学会了控制机械臂，还学会了模拟复杂物体的物理形变（如挤压面包袋），这证明 Genie 能够捕捉真实的物理世界动态。
*   **作为强化学习的“母体”**：在 Genie 内部训练的智能体，可以极快地迁移到真实环境中。相比于从零开始训练，使用潜动作预训练的智能体在样本效率上提升了数倍。

<div align="center">
  <img src="/images/wm/Genie-robotics-actions.webp" width="100%" />
<figcaption>图：在机器人操作数据上学习到的具有语义意义的潜动作。（图源：Genie, 2024）</figcaption>
</div>

---

### 4. 局限性

*   **分辨率瓶颈**：受限于目前的计算资源，Genie 生成的视频分辨率较低（160x90），离高清沉浸式体验仍有距离。
*   **自回归发散**：由于是自回归生成，随着步数增加，视频内容可能会逐渐偏离物理真实或出现伪影。
*   **动作映射**：虽然挖掘出了潜动作，但将这些离散 Token 精确映射到人类直觉的复杂多级控制（如手柄的线性摇杆）仍需进一步研究。

---

## 5.3 VLA-World (2026) {#sec-5-3-vla-world}
———Learning Vision-Language-Action World Models for Autonomous Driving

📄 **Paper**: [https://vlaworld.github.io](https://vlaworld.github.io)

### 精华

VLA-World 的核心思想在于通过在单帧未来预测的基础上进行反思性推理，将世界模型的生成能力与 VLA 模型的推理能力相结合。最值得借鉴的设计是其“分步走”的流程：首先根据预测的动作生成一张未来图，再让模型去观察这张自己生成的图，从而识别潜在的碰撞风险并修正动作。这种“脑内模拟后二次评估”的机制（Think with Generated future）极大地增强了端到端驾驶系统的安全性和可解释性。

---

### 1. 研究背景/问题

现有的端到端自动驾驶模型（如 VLA）通常缺乏显式的时空建模，难以预测环境中其他交通参与者的演变。而纯世界模型虽然能生成连贯的未来场景，却往往缺乏推理能力，难以评估所生成未来的安全性或优劣。VLA-World 通过统一预测性想象与反思性推理，提升了驾驶前瞻性。

---

### 2. 主要方法/创新点

VLA-World 提出了一个结合了感知、动作衍生预测、图像生成、反思推理和规划的完整流程。

<div align="center">
  <img src="/images/wm/VLA-World-overview.webp" width="100%" />
<figcaption>图：VLA-World 三阶段训练与性能概览。（图源：VLA-World, 2026）</figcaption>
</div>

#### 三阶段训练策略
1. **阶段 1：视觉预训练**：在大规模图像-指令数据集上激活图像生成知识。
2. **阶段 2：监督微调 (SFT)**：通过 nuScenes-GR-20K 混合任务数据集，建立感知、未来生成与规划的逻辑链接。
3. **阶段 3：强化学习 (RL)**：利用 GRPO 算法探索类人推理，使模型能更深入地反思生成的未来是否安全。

<div align="center">
  <img src="/images/wm/VLA-World-paradigm-comparison.webp" width="100%" />
<figcaption>图：VLA、世界模型与 VLA-World 三种范式的对比。（图源：VLA-World, 2026）</figcaption>
</div>

#### 反思推理机制 (Think with Generated future)
模型首先输出一个 0.5 秒内的轨迹预测，并据此生成对应的未来图。随后，模型再次“审阅”这张自生成的图，识别重要物体和潜在风险，最终修正决策，输出最终的长程轨迹。这种机制类似于人类驾驶员遇到突发状况时的二次反思过程。

---

### 3. 核心结果/发现

- **性能表现**: 在 nuScenes 等基准测试中，VLA-World 达到了比现有 VLA 和世界模型更低的碰撞率（Collision Rate 从 1.09% 降至 0.94%）和更高的 FID 视频生成质量。
- **可解释性**: 通过让模型写下对“自己生成的未来”的推理过程（如识别某卡车的碰撞风险），系统的决策过程变得更加透明。

<div align="center">
  <img src="/images/wm/VLA-World-visualization.webp" width="100%" />
<figcaption>图：VLA-World 在复杂场景下的多视角图像预测可视化。（图源：VLA-World, 2026）</figcaption>
</div>

---

### 4. 局限性

由于模型需要先生成图像再进行推理，系统的端到端延迟仍然是一个挑战。未来研究将聚焦于提高实时推理速度。

---

## 5.4 WorldVLA (2025) {#sec-5-4-worldvla}
———Towards Autoregressive Action World Model

📄 **Paper**: [https://arxiv.org/abs/2506.21539](https://arxiv.org/abs/2506.21539)

### 精华

这篇论文的核心亮点在于将 Vision-Language-Action (VLA) 模型与世界模型（World Model）统一在单个自回归框架中。值得借鉴的思想包括：利用世界模型预测未来图像的能力来学习环境底层物理规律，从而增强动作生成的准确性；反之，动作模型也辅助视觉理解，提升了图像生成的质量。此外，针对自回归动作序列生成中的误差累积问题，提出的动作注意力掩码策略（Action Attention Masking）能够显著提升动作块（Action Chunk）的生成性能。

---

### 1. 研究背景/问题

当前的 VLA 模型主要关注从图像和文本生成动作，但往往缺乏对动作深层次的理解，因为动作仅作为输出而未作为输入。相比之下，世界模型能够通过预测未来视觉状态来理解物理动力学，但通常无法直接生成动作。WorldVLA 旨在打破这一界限，通过统一架构实现动作与图像的协同理解与生成。

---

### 2. 主要方法/创新点

WorldVLA 采用自回归架构，集成了图像、文本和动作三种模态的 Tokenizer。

<div align="center">
  <img src="/images/wm/WorldVLA-overview.webp" width="100%" />
<figcaption>图：WorldVLA 与传统动作模型、世界模型的对比。（图源：WorldVLA, 2025）</figcaption>
</div>

#### 统一架构
模型初始化自 Chameleon，一个统一的图像理解与生成模型。它包含：
- **图像 Tokenizer**: VQ-GAN 模型，将图像离散化为 Token。
- **动作 Tokenizer**: 将 7 维机器人动作（位置、角度、夹具状态）离散化为 256 个 Bin 的 Token。
- **文本 Tokenizer**: 标准的 BPE Tokenizer。

<div align="center">
  <img src="/images/wm/WorldVLA-architecture.webp" width="100%" />
<figcaption>图：WorldVLA 整体架构图。（图源：WorldVLA, 2025）</figcaption>
</div>

#### 训练策略
训练过程混合了动作模型数据和世界模型数据：
1. **动作预测 ($L_{action}$)**: 给定指令和多帧图像，预测后续动作。
2. **未来预测 ($L_{world}$)**: 给定当前观察和动作，预测下一帧图像。

#### 动作注意力掩码 (Action Attention Masking)
论文发现，由于预训练模型在动作域的泛化能力有限，传统的因果掩码会导致前一动作的错误迅速传播。为此，WorldVLA 设计了一种特殊的掩码：在生成当前动作块时，遮蔽之前的动作，使动作生成仅依赖于视觉和文本输入，从而支持并行生成动作块并减少误差累积。

<div align="center">
  <img src="/images/wm/WorldVLA-attention-mask.webp" width="100%" />
<figcaption>图：WorldVLA 的注意力掩码机制。（图源：WorldVLA, 2025）</figcaption>
</div>

---

### 3. 核心结果/发现

- **LIBERO 基准测试**: WorldVLA 在 256x256 和 512x512 分辨率下均显著优于 OpenVLA。
- **协同效应**: 加入世界模型数据后，动作生成的成功率（SR）有明显提升（例如在 LIBERO-Goal 上从 67.3% 提升至 73.1%）；同时，动作模型也帮助降低了视频生成的 FVD 值。
- **动作块生成**: 采用新掩码策略后，动作块生成的鲁棒性大幅增强。

<div align="center">
  <img src="/images/wm/WorldVLA-visualization-action.webp" width="100%" />
<figcaption>图：动作模型可视化：WorldVLA 能在失败后多次尝试抓取。（图源：WorldVLA, 2025）</figcaption>
</div>

<div align="center">
  <img src="/images/wm/WorldVLA-visualization-world.webp" width="100%" />
<figcaption>图：世界模型可视化：WorldVLA 生成的未来图像更符合物理逻辑。（图源：WorldVLA, 2025）</figcaption>
</div>

---

### 4. 局限性

目前使用的离散图像 Tokenizer 在感知表现力上仍有局限。未来工作将探索更大规模的数据和模型，以及设计能够更平衡理解与生成的统一 Tokenizer。

---

## 5.5 WoVR (2026) {#sec-5-5-wovr}
———World Models as Reliable Simulators for Post-Training VLA Policies with RL

📄 **Paper**: [https://arxiv.org/abs/2602.13977](https://arxiv.org/abs/2602.13977)

### 精华

WoVR 提出了一种基于世界模型的机器人强化学习（RL）框架，核心贡献在于解决了世界模型中的“幻觉（Hallucination）”问题对 RL 优化信号的干扰。值得借鉴的三个机制包括：**稳定的动作调节视频模型**（Stabilized Action-conditioned Video World Model）通过双通道动作注入提升稳定性；**关键帧初始化回放（Keyframe-Initialized Rollouts, KIR）**通过在任务关键点附近初始化轨迹，缩短了有效预测深度并限制误差累积；以及**世界模型与策略的协同演化策略（PACE）**，通过迭代精调世界模型来恢复策略更新带来的分布漂移，确保了在想象空间中 RL 训练的可靠性。

---

### 1. 研究背景/问题

利用学习到的世界模型作为仿真器进行强化学习是机器人领域的热门方向，但闭环想象中的“幻觉”——即模型生成的视觉序列与真实物理规律不符——会误导 RL 优化，使其利用模型的错误而非真实的任务进度。随着策略演化，动作分布发生漂移，进一步加剧了幻觉问题。

---

### 2. 主要方法/创新点

WoVR 并不假设世界模型是完美的，而是通过三个层面显式地调节 RL 与不完美模拟器的交互。

<div align="center">
  <img src="/images/wm/WoVR-hallucination-overview.webp" width="100%" />
<figcaption>图：世界模型中的幻觉问题及其对 RL 的干扰。（图源：WoVR, 2026）</figcaption>
</div>

#### 稳定的世界模型架构
WoVR 引入了一种增强型 DiT（Diffusion Transformer）世界模型，通过双通道动作注入机制实现更稳定的动作控制，减少了长程漂移和结构崩溃。

#### 关键帧初始化回放 (KIR)
为了防止自回归生成的误差随时间累加，WoVR 采用了 Keyframe-Initialized Rollouts。它利用人类演示中的关键帧作为起始点，在这些状态附近进行短程想象探索。这种做法大大限制了有效预测深度，抑制了幻觉的积累。

<div align="center">
  <img src="/images/wm/WoVR-pipeline.webp" width="100%" />
<figcaption>图：WoVR 核心三步走架构：稳定模型、关键帧初始化、协同演化。（图源：WoVR, 2026）</figcaption>
</div>

#### 策略对齐协同演化 (PACE)
为了应对策略更新导致的动作分布漂移（Distribution Shift），PACE 策略会定期在当前演化策略生成的动作轨迹上对世界模型进行微调。这种协同演化机制使模拟器能够动态适应新的动作分布，保持了策略与模拟器的对齐。

---

### 3. 核心结果/发现

- **LIBERO 基准测试**: WoVR 将 LIBERO 的平均成功率从 39.95% 提升至 69.2%（+29.3个百分点）。
- **真机验证**: 在真实机器人操作任务中，成功率从 61.7% 提升至 91.7%。
- **生成效率**: WoVR 达到了 23 FPS 的生成速度，使其成为一种高效的训练模拟器。

<div align="center">
  <img src="/images/wm/WoVR-visualization-results.webp" width="100%" />
<figcaption>图：WoVR 在 LIBERO 任务上的想象生成与策略执行可视化。（图源：WoVR, 2026）</figcaption>
</div>

---

### 4. 局限性

虽然 WoVR 缓解了幻觉，但对于极其复杂的多步长程任务，其稳定性仍有待提升。此外，协同演化过程中的计算开销也是一个需要优化的方向。

---

## 5.6 Janus-Pro (2025) {#sec-5-6-janus-pro}
———Unified Multimodal Understanding and Generation with Data and Model Scaling

📄 **Paper**: https://arxiv.org/abs/2501.17811

### 精华

Janus-Pro 最值得借鉴的核心思想是**解耦视觉编码**：理解任务与生成任务对视觉表征的需求本质不同，强行共享编码器会造成任务冲突，解耦后两路可独立优化。此外，训练策略的精细化同样重要——Stage I 充分训练像素依赖建模、Stage II 去除低效的 ImageNet 预热、Stage III 调整多模态数据比例，每一步都针对已知痛点而非盲目堆量。合成数据（1:1 比例）对生成质量的稳定性提升至关重要，是解决真实数据噪声问题的实用路径。模型规模从 1.5B 扩展到 7B 验证了解耦编码方法的强可扩展性，为统一理解与生成框架的规模化提供了实证支撑。

---

### 1. 研究背景/问题

当前统一多模态理解与生成的模型通常共享同一视觉编码器处理两类任务，但理解与生成对视觉表征的需求存在本质冲突，导致多模态理解性能受损。前代模型 Janus 虽通过解耦视觉编码验证了该思路，但受限于训练数据量少和模型容量小，在短提示图像生成质量和生成稳定性上表现欠佳。

---

### 2. 主要方法/创新点

Janus-Pro 从三个维度对 Janus 进行系统性增强：训练策略优化、数据扩展和模型规模扩展。

**架构**（与 Janus 相同，解耦视觉编码）：

<div align="center">
  <img src="/images/vlm/Janus-Pro-architecture.webp" width="100%" />
<figcaption>图：Janus-Pro 整体架构：理解侧使用 SigLIP Understanding Encoder，生成侧使用 VQ Generation Encoder，共享同一个 Auto-Regressive Transformer。（图源：Janus-Pro, 2025）</figcaption>
</div>

为便于理解，下图是我对 Janus-Pro 架构的手绘版本整理（核心：自回归统一框架，图像侧解耦为理解与生成两条编码路径）：

```mermaid
flowchart TB
    TextIn["文本输入"] --> TextTok["Text Tokenizer"]
    ImgIn["图像输入"] --> UndEnc["理解 Encoder<br/>(SigLIP)"]
    ImgIn --> GenEnc["生成 Encoder<br/>(VQ-Tokenizer)"]

    UndEnc --> UndFeat["理解特征"]
    GenEnc --> GenFeat["生成特征<br/>(视觉 Token 词表)"]

    UndFeat --> UndAdapt["Understanding Adaptor<br/>(MLP, 理解时使用)"]
    GenFeat --> GenAdapt["Generation Adaptor<br/>(MLP)"]

    TextTok --> LLM
    UndAdapt --> LLM
    GenAdapt --> LLM

    LLM["LLM / 自回归 Transformer"] -->|自回归| TextOut["文本输出"]
    LLM -->|预测视觉 Tokens 16×16| VisTok["多模态视觉 Tokens"]
    VisTok --> VQDec["VQ-Decoder"]
    VQDec --> ImgOut["图像输出"]

    classDef input fill:#e8f4fd,stroke:#2c7fb8,stroke-width:1px;
    classDef enc fill:#fff4e6,stroke:#d68910,stroke-width:1px;
    classDef core fill:#fde9e9,stroke:#c0392b,stroke-width:2px;
    classDef output fill:#e8f8e8,stroke:#27ae60,stroke-width:1px;

    class TextIn,ImgIn input;
    class TextTok,UndEnc,GenEnc,UndFeat,GenFeat,UndAdapt,GenAdapt,VisTok,VQDec enc;
    class LLM core;
    class TextOut,ImgOut output;
```

整体框架基于统一的自回归 Transformer。对于多模态理解任务，使用 SigLIP-Large-Patch16-384 编码器提取高维语义特征，经 Understanding Adaptor（两层 MLP）映射到 LLM 输入空间；对于视觉生成任务，使用来自 **LlamaGen** 的 VQ tokenizer 将图像离散化为 ID 序列，经 Generation Adaptor 映射 codebook embedding 输入 LLM，最终通过 Image Decoder 输出 $384 \times 384$ 图像。

**三阶段训练流程**：

Janus 与 Janus-Pro 均采用三阶段训练范式，下图（取自原 Janus 论文）展示了每个阶段中各模块的冻结（❄️）与可训练（🔥）状态：

<div align="center">
  <img src="/images/wm/janus-training.webp" width="100%" />
<figcaption>图：Janus / JanusFlow 三阶段训练流程图：火焰标记代表可训练模块，雪花标记代表冻结模块。Janus-Pro 沿用该流程，但在 Stage 1 和 Stage 2 做出关键调整。（图源：Janus, 2024）</figcaption>
</div>

- **Stage 1 — Adaptation（适配）**：目标是让新引入的模块与预训练组件协同工作。此阶段冻结 **LLM** 与 **图像理解编码器（Und. Enc.）**，仅训练将图像编码映射到 LLM 输入空间的 **Linear 映射层** 和 **图像生成头（Gen. Dec.）**。训练数据为 ImageNet（基于类别名提示生成图像）。**Janus-Pro 的改动：显著增加 Stage 1 的训练步数**，让模型在 LLM 参数固定的情况下更充分地建模像素依赖。

- **Stage 2 — Unified Pre-Training（统一预训练）**：在继续训练新模块的基础上，**解冻 LLM 及其文本预测头（Text De-Token）**，使其能够处理多模态嵌入序列。训练样本包括多模态理解、图像生成与纯文本数据三类。**Janus-Pro 的改动：完全移除 ImageNet 数据**，直接使用密集描述的真实文生图数据——原版 Janus 在此阶段以 ImageNet 开始并逐步提升文生图数据比例，Janus-Pro 则跳过该预热阶段，训练效率显著提升。此外，图像编码器的表征会与图像生成潜在输出做对齐，以增强生成过程的语义一致性。

- **Stage 3 — Supervised Fine-Tuning（监督微调）**：在指令微调数据（对话 + 高质量文生图样本）上进行 SFT。此阶段**图像理解编码器（Und. Enc.）也加入训练**，即除 VAE 编码器外的全部模块都被解冻。Janus-Pro 在此阶段与原版 Janus 流程一致。

**Stage 3 数据比例调整**：将多模态理解数据、纯文本数据、文生图数据的比例从原版 Janus 的 7:3:10 调整为 5:1:4，在保持生成能力的同时提升多模态理解性能。

**数据扩展**：

- **多模态理解**：参考 DeepSeek-VL2，增加约 9000 万样本（图像描述、表格、图表、文档理解等），Stage III 额外加入 MEME 理解、中文对话等数据；
- **视觉生成**：引入约 7200 万合成图像样本，将真实与合成数据比例调整为 1:1，有效解决原始真实数据噪声大、生成不稳定的问题。

**模型扩展**：

将基础 LLM 从 1.5B 扩展至 7B（使用 DeepSeek-LLM），形成 Janus-Pro-1B 和 Janus-Pro-7B 两个版本。实验表明更大规模 LLM 使两类任务的 loss 收敛速度均显著加快。

<div align="center">
  <img src="/images/vlm/Janus-Pro-performance.webp" width="100%" />
<figcaption>图：Janus-Pro 在多模态理解（左，四个基准平均分 vs LLM 参数量）和文生图指令跟随（右，GenEval 和 DPG-Bench）上的性能对比，Janus-Pro-7B 在两类任务上均达到最优。（图源：Janus-Pro, 2025）</figcaption>
</div>

---

### 3. 核心结果/发现

**多模态理解**（Table 3）：
- Janus-Pro-7B 在 MMBench 上达到 79.2，超越同类统一模型 Janus（69.4）、TokenFlow-XL（68.9，13B）、MetaMorph（75.2，8B）
- MMMU 得分 50.0，GQA 62.0，全面领先统一理解+生成类模型

**文生图生成**（Table 4 & 5）：
- GenEval 整体得分 0.80，超越 Janus（0.61）、DALL-E 3（0.67）、SD3-Medium（0.74）
- DPG-Bench 得分 84.19，超越所有对比方法（含生成专用模型）

**定性结果**：

<div align="center">
  <img src="/images/vlm/Janus-Pro-qualitative.webp" width="100%" />
<figcaption>图：Janus-Pro-7B 的多模态理解（图像描述、地标识别、通识问答、文字识别）和文生图生成定性结果，生成分辨率为 384×384。（图源：Janus-Pro, 2025）</figcaption>
</div>

---

### 4. 局限性

多模态理解输入分辨率限制在 $384 \times 384$，影响 OCR 等细粒度任务性能；VQ tokenizer 的重建损失导致生成图像中小面部区域等细节欠缺，提升分辨率是解决上述两个问题的主要方向。

---

## 5.7 Video Generation Models in Robotics (2026) {#sec-5-7-videogen}
———Applications, Research Challenges, Future Directions

📄 **Paper**: [arXiv:2601.07823](https://arxiv.org/abs/2601.07823)

### 精华

1. **核心价值**：视频生成模型作为**高保真物理世界模拟器**，能克服物理仿真器的简化假设，为机器人提供精细的交互感知。
2. **具身世界模型**：视频模型不仅是视觉输出工具，更是能够预测时空演变的"具身世界模型"，支持策略学习与视觉规划。
3. **关键应用**：涵盖模仿学习（数据增强）、强化学习（动力学建模）、策略评估（免真实环境部署）和视觉规划。
4. **主要挑战**：包括违反物理规律的幻觉（Hallucinations）、指令遵循能力弱、长视频生成的连贯性以及极高的推理成本。
5. **未来方向**：整合物理先验（物理引擎作为约束）、不确定性量化、更高效的推理架构（如 DiT）以及长序列生成。

---

### 1. 研究背景/问题

传统的机器人研究依赖物理仿真器进行策略验证和训练，但仿真器通常需要复杂的参数调整且难以模拟柔性体或精细物理交互。与此同时，仅依赖语言抽象的大模型（LLMs）缺乏对物理世界细粒度时空动态的理解。视频生成模型（Video Generation Models）凭借其在互联网规模数据上学习到的丰富视觉和动作知识，展现出作为**具身世界模型（Embodied World Models）**的巨大潜力。

<div align="center">
  <img src="/images/vln/Robot-Video-Gen-Overview.webp" width="100%" />
<figcaption>图：视频生成模型在机器人领域的应用框架，包括策略学习、视觉规划和策略评估。（图源：Robot-Video-Gen, 2026）</figcaption>
</div>

---

### 2. 主要方法/创新点

论文系统地梳理了视频生成模型在机器人中的架构分类、应用范式及评估体系。

#### 核心分类学 (Taxonomy)
视频生成模型在机器人中的角色主要分为：
- **模仿学习中的数据生成器**：合成多样化的专家演示，缓解数据稀缺问题。
- **强化学习中的动力学/奖励模型**：预测未来状态并提供视觉反馈。
- **视觉规划器**：通过合成未来视频序列来辅助机器人进行任务分解和搜索。

<div align="center">
  <img src="/images/vln/Robot-Video-Gen-Taxonomy.webp" width="100%" />
<figcaption>图：论文的组织架构，展示了背景、应用、评估及开放挑战的分类体系。（图源：Robot-Video-Gen, 2026）</figcaption>
</div>

#### 模型架构演进
从传统的基于 RNN/CNN 的预测模型演进到如今主流的基于 **Diffusion** 和 **Flow-matching** 的架构。
- **扩散模型 (Diffusion Models)**：利用逐步去噪过程合成高质量视频帧，结合 Transformer (DiT) 或 U-Net 实现条件控制。
- **联合嵌入预测架构 (JEPA)**：通过学习隐藏特征空间中的动态，实现更鲁棒的非像素级世界建模。

<div align="center">
  <img src="/images/vln/Diffusion-Video-Architecture.webp" width="100%" />
<figcaption>图：基于扩散的视频模型架构示意图，展示了条件输入（文本、图像、动作）如何指导合成。（图源：Robot-Video-Gen, 2026）</figcaption>
</div>

#### 显式与隐式世界模型
- **隐式模型**：通过视觉像素或潜空间表示世界状态。
- **显式模型**：输出如点云（Point Cloud）、体素网格（Voxel Map）或 3D 高斯泼溅（3DGS）等显式 3D 表示，以增强物理一致性。

<div align="center">
  <img src="/images/vln/Implicit-vs-Explicit-Models.webp" width="100%" />
<figcaption>图：具身世界模型的两种表示形式：隐式表示（如视频潜空间）与显式表示（如点云、3DGS）。（图源：Robot-Video-Gen, 2026）</figcaption>
</div>

---

### 3. 核心结果/发现

- **性能评估标准**：除了传统的视觉指标（PSNR, SSIM, FVD），机器人领域更关注物理一致性（Physics-IQ）、指令遵循度（VBench）和策略部署后的成功率。
- **跨模态优势**：视频模型能整合文本指令、参考图像和动作序列，生成的视频轨迹可直接用于训练 VLA（Vision-Language-Action）策略。
- **成本效益**：通过视频生成进行大规模策略评估，可减少对真实物理站点的依赖，降低硬件损耗和人工成本。

---

### 4. 局限性

- **Hallucinations**：生成的视频常出现物体凭空消失或违反重力等现象，限制了其在安全敏感场景的应用。
- **长序列漂移**：随着生成步数增加，视频的物理真实度和连贯性会迅速下降。
- **实时性瓶颈**：扩散模型的采样过程极其耗时，难以满足机器人闭环控制的需求。

---

## 5.8 AIM (2026) {#sec-5-8-aim}
———Intent-Aware Unified World Action Modeling with Spatial Value Maps

📄 **Paper**: https://arxiv.org/abs/2604.11135

### 精华

这篇论文最值得借鉴的是**用显式的空间价值图（Spatial Value Map, ASVM）作为世界模型（World Model）和动作头（Action Head）之间的中间接口**，把“未来视觉预测”与“动作解码”之间缺失的“在哪里交互、为什么交互”这一操作意图（Manipulation Intent）补齐，从而避免动作头从稠密 RGB 像素未来中隐式反推逆动力学。具体可迁移的设计包括：(1) **意图因果注意力（Intent-Causal Attention）**——通过显式注意力掩码（Attention Mask）强制动作分支只能经由空间价值图访问未来信息，而不能直接读取未来 RGB Tokens，形成结构化的信息瓶颈；(2) **混合专家 Transformer（Mixture-of-Transformers, MoT）**的共享自注意力 + 分支 FFN 让视频、价值图、动作三个数据流既紧密耦合又各自保留专有特征空间；(3) **自蒸馏强化学习后训练（Self-Distillation RL Post-Training）**——冻结视频生成与价值图分支，仅利用投影空间价值图响应所产生的稠密奖励训练动作头，相当于让预训练的价值头自监督指导动作头，无需额外人工标签。这种“把语义意图落地为空间热图”的抽象在具身 VLA 控制领域具有极强的通用迁移价值。

---

### 1. 研究背景/问题

预训练视频生成模型为机器人控制提供了强大的视觉先验，但已有的统一世界动作模型（Unified World Action Models）在不做海量机器人专有数据微调的情况下难以解码出高精度动作。作者指出这并非纯粹的数据统计问题，而是**结构性失配（Structural Mismatch）**：视频生成模型捕获的是“物理场景如何演化”，而动作生成还需要显式推理“在哪里交互（Where）”以及“背后的操作意图（Intent）”；直接从未来 RGB 潜特征解码动作，会迫使模型从一个并非为控制优化的视觉表征中隐式恢复操作意图。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/wm/AIM-typical-vs-ours.webp" width="100%" />
<figcaption>图：典型统一世界动作模型（左）直接从共享未来视觉表征解码动作；AIM（右）在世界模型与动作头间引入空间价值图接口并通过自蒸馏优化。（图源：AIM, 2026）</figcaption>
</div>

**核心思路：显式空间接口（Explicit Spatial Interface）。** AIM 不直接从未来视觉特征解码动作，而是联合预测未来视频帧 $X^+$ 与与之空间几何对齐的动作空间价值图（Action-aligned Spatial Value Map）$M^+ \in [0,1]^{H \times W \times 3}$；空间价值图高亮任务相关的交互区域（例如抓取任务的抓取可操作性 Grasp Affordance 区域、放置任务的放置接触 Placement Contact 区域），作为操作意图的控制抽象。条件联合分布被分解为：

$$p(X^+, M^+, A^+ \mid \mathcal H_t) = p(X^+, M^+ \mid \mathcal H_t)\, p(A^+ \mid \mathcal H_t, M^+).$$

动作生成**仅通过预测出的空间价值图**获取未来信息，而不直接访问未来 RGB Tokens。

<div align="center">
  <img src="/images/wm/AIM-framework.webp" width="100%" />
<figcaption>图：AIM 整体框架：Stage I 联合训练未来帧、空间价值图与动作；意图因果注意力将任务意图传递至动作分支；Stage II 冻结视频与价值分支，通过 GRPO 强化学习优化动作头。（图源：AIM, 2026）</figcaption>
</div>

**架构设计（Architecture）。** 基于预训练视频生成模型 **Wan2.2-TI2V-5B** 初始化视频分支，加入一个与之同深度但隐藏层宽度更紧凑的动作解码头。采用 **Mixture-of-Transformers (MoT)**：视频、价值图、动作三个流在每个 Transformer Block 共享自注意力子层，但各自拥有独立的 $W_{Q,s}^\ell, W_{K,s}^\ell, W_{V,s}^\ell$ 投影与独立的前馈网络（FFN）。T5 编码的自然语言指令仅通过交叉注意力注入视频分支，保证动作头仅经由共享的世界表征接收任务语义。Token 化时将三个视角（头顶俯视 / 左腕 / 右腕）拼接为 T-pose Canvas 画布，并复用 Wan2.2 VAE 同时编码 RGB 观测 $z_t^o$ 与空间价值图 $z_t^m$，使价值 Tokens 与视觉 Tokens 天然保持几何对齐。

**意图因果自注意力（Intent-Causal Self-Attention）。** 这是 AIM 的关键结构创新，通过对共享自注意力的可见性掩码（Visibility Mask）实现：

$$\mathcal V_x = [z_t^o,\, z_{t-k:t-1}^o,\, z_{t-k:t-1}^a,\, z^\ell,\, z^x],$$

$$\mathcal V_m = [z_t^o,\, z_{t-k:t-1}^o,\, z^x,\, z^m],$$

$$\mathcal V_a = [z_t^o,\, z_{t-k:t-1}^a,\, z^o,\, z^a].$$

语义上：未来视频 Tokens 能看到当前观测、指令、历史观测动作，从而预测任务条件下的未来物理演化；未来价值 Tokens 能看到当前/历史观测与**未来视频 Tokens**，从而将空间价值预测锚定到推演出的未来状态；**动作 Tokens 只能看到当前观测、历史动作和未来价值 Tokens，而看不到未来 RGB Tokens**——这一掩码机制的作用是将任务语义先经过 T5 交叉注意力进入视频流 → 再凝聚沉淀到空间价值流 → 最后才被动作解码头读取，形成严密的“视频 $\to$ 价值 $\to$ 动作”因果信息瓶颈。

**训练目标。** 整体损失函数是 RGB 流匹配（Flow Matching）、空间价值图流匹配与动作逆动力学损失的加权和：

$$\mathcal L = \mathcal L_{rgb} + \lambda_m \mathcal L_{map} + \lambda_a \mathcal L_{act}.$$

未来 RGB 与未来价值图 Tokens 由视频生成主干沿同一条流匹配轨迹联合去噪，动作 Tokens 由动作头去噪为连续双臂控制向量 $\hat A^+$。推理时 AIM 采用自回归 Chunk-wise Rollout 并利用 KV 缓存复用历史 Tokens，显著提升长时程推理效率。

**自蒸馏强化学习后训练（Self-Distillation RL Post-Training）。** 监督学习（SFT）只能模仿动作分布而无法直接优化闭环任务成功率。因此引入第二阶段：**冻结视频生成主干与空间价值图预测头，仅用 GRPO 算法更新动作头**。单步奖励函数由稠密与稀疏两部分构成：

$$r_t = \lambda_d r_t^{dense} + \lambda_s r_t^{sparse},\qquad r_t^{dense} = M_t(\Pi(p_t)),$$

其中 $r_t^{sparse}$ 为任务完成的稀疏奖励信号，$p_t$ 为预测动作的落点或末端执行器目标位置，$\Pi(\cdot)$ 为相机几何投影矩阵，$M_t$ 为冻结价值头预测的空间价值图。直观而言，动作头因将动作精确投射到高价值交互区域而获得正向奖励——这是一种**利用模型自身的空间价值先验作为稠密奖励的自蒸馏机制**，免除了繁重的人工标注。GRPO 优化目标：

$$\mathcal L_{GRPO}(\phi) = \mathbb E_t\left[\min\!\Big(\rho_t(\phi)\hat A_t,\, \mathrm{clip}(\rho_t(\phi), 1-\epsilon, 1+\epsilon)\hat A_t\Big)\right].$$

**空间价值图标注方案。** 针对抓取（Pick）任务，在夹爪与目标物体建立有效物理接触时记录接触表面点云，经相机投影矩阵映射至图像平面并施加高斯平滑，构建抓取可操作性区域（Grasp Affordance Region）；高斯核宽度根据相机内参及深度距离动态调整，保证不同视角与距离下的几何尺度一致。针对放置（Place）任务，检测被操作物体达到静止构型时的接触区域，生成放置接触区域（Placement Contact Region）。作者在 RoboTwin 2.0 仿真平台构建了 30K 轨迹的大规模数据集，包含同步多视角视频、精确动作序列及逐帧空间价值图标注。

---

### 3. 核心结果/发现

在 RoboTwin 2.0 的 50 个双臂精细操作任务上进行评测，Easy / Hard 难度均以任务成功率（SR %）为主指标：

- **平均成功率：AIM 达到 94.0% / 92.1%（Easy / Hard），综合平均成功率达 93.1%**，全面领先 $\pi_0$ (62.2%)、X-VLA (72.8%)、$\pi_{0.5}$ (79.8%)、GigaWorld-0 (86.0%)、Motus (87.8%)、Fast-WAM (91.8%)、LingBot-VA (92.2%) 等主流基线模型。
- **强化学习后训练增益显著**：Stage 1（SFT 监督微调）已达到 93.0% / 92.0%（平均 92.5%）；Stage 2 RL 阶段进一步带来 +0.6 个百分点的平均提升（达到 94.0% / 92.1%，平均 93.1%），在 *Place Mouse Pad* (97%/95%)、*Scan Object* (100%/98%)、*Turn Switch* (100%/98%) 等接触敏感和阶段依赖任务上增益尤为突出。
- **大幅领先同类方法**：相对同类统一世界动作模型 Motus，AIM 在 Easy / Hard 难度分别提升 **+4.8 / +5.7 个百分点**（平均提升 +5.3%）；相对 $\pi_{0.5}$ 提升 **+12.7 / +13.9 个百分点**（平均提升 +13.3%）。这证明将“空间交互意图”显式化建模带来的增益，显著超越了单纯扩大动作模型或视频模型参数量。
- **定性可视化验证**：未来帧预测与操作阶段时序高度对齐，空间价值图精准聚焦于具有明确物理交互语义的区域（而非一般的视觉显著性 Saliency），机械臂投影动作落点严密落在高价值区域内，表明性能突破确实源于所设计的“空间桥梁”机制。

<div align="center">
  <img src="/images/wm/AIM-task-execution.webp" width="100%" />
<figcaption>图：RoboTwin 2.0 代表性任务执行过程（置鼠标垫/压订书机/扫物体/扳开关/开笔记本），左列为 Easy 设置，右列为 Hard 设置。（图源：AIM, 2026）</figcaption>
</div>

---

### 4. 局限性

目前该工作主要在 RoboTwin 2.0 仿真环境中构建数据并评测，尚需在真实双臂机器人平台上做进一步闭环迁移验证；此外，空间价值图的自动标注依赖仿真器的碰撞检测 API 与接触物理状态，在真实无标注视频中如何以无监督方式高效挖掘同等精度的空间接触标签仍是开放课题。

---

## 5.9 LingBot-World (2026) {#sec-5-9-lingbot-world}
———首个开源、支持实时交互的长程世界模型

📄 **Paper**: [arXiv:2601.20540](https://arxiv.org/abs/2601.20540)

### 精华
1. **LingBot-World** 是一个开源的实时交互世界模型，支持分钟级的长程生成一致性。
2. 提出了包含分层语义的数据引擎，通过叙事、静态场景和密集时间描述解决了交互数据稀缺问题。
3. 采用了三阶段进化训练策略：预训练（通用视频先验）、中训练（知识注入与 MoE 架构）和后训练（因果适配与蒸馏）。
4. 实现了亚秒级（<1s）的推理延迟，支持 16 fps 的实时生成。
5. 展示了在可控世界事件编辑、具身智能 Action Agent 和 3D 重建等领域的广泛应用潜力。

### 1. 研究背景/问题
当前的视频生成模型虽能生成高质量短片，但本质上是“梦想家”而非“模拟器”，缺乏对物理规律（如因果性、物体恒久性）的理解，且难以实现实时交互。此外，高质量交互数据的匮乏、长程一致性的维持以及扩散模型高昂的计算开销，也是阻碍世界模型发展的核心瓶颈。

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/vla/LingBot-World-overview.webp" width="100%" />
<figcaption>图：LingBot-World 交互式世界模拟概览：支持在多种场景（写实、科学、卡通等）下通过键盘操作进行实时交互。（图源：LingBot-World, 2026）</figcaption>
</div>

#### ① 数据引擎与分层描述
为了解决高质量交互数据稀缺的问题，LingBot-World 构建了一个混合数据引擎，结合了真实世界视频、游戏录像和 Unreal Engine (UE) 合成数据。关键创新在于**分层描述策略**：
- **叙事描述 (Narrative Caption)**：描述整体环境和摄像机轨迹，作为全局语义提示。
- **静态场景描述 (Scene-Static Caption)**：仅聚焦环境，实现动作与场景的解耦。
- **密集时间描述 (Dense Temporal Caption)**：对视频事件进行细粒度的时间对齐描述。

#### ② 三阶段进化训练管线
模型采用了从视频生成器向交互式模拟器进化的三阶段策略：

<div align="center">
  <img src="/images/vla/LingBot-World-training-pipeline.webp" width="100%" />
<figcaption>图：LingBot-World 训练管线：从预训练的视频先验出发，经过中训练注入知识，最后通过后训练实现实时交互能力。（图源：LingBot-World, 2026）</figcaption>
</div>

- **Stage I: 预训练**：利用 14B 参数的 Wan2.2 扩散模型建立强大的时空相干性和视觉先验。
- **Stage II: 中训练 (MoE 知识注入)**：引入 Mixture-of-Experts (MoE) 架构（总参数 28B，激活 14B），通过 progressive curriculum 策略将训练时长从 5 秒扩展到 60 秒，并注入 Plücker 编码的动作信号。
- **Stage III: 后训练 (实时化)**：将双向扩散模型适配为因果自回归系统，并结合分布匹配蒸馏 (DMD) 和对抗优化，将推理延迟降低至亚秒级。

#### ③ 模型架构与动作注入

<div align="center">
  <img src="/images/vla/LingBot-World-architecture.webp" width="100%" />
<figcaption>图：LingBot-World 模型架构：基于 DiT 块，通过 Plücker Encoder 注入动作信号，并利用 AdaLN 进行调制。（图源：LingBot-World, 2026）</figcaption>
</div>

LingBot-World 基于 DiT (Diffusion Transformer) 架构。动作信号（离散键盘输入和连续摄像机旋转）通过 **Plücker Encoder** 投影为嵌入向量，再通过 Adaptive Layer Normalization (AdaLN) 注入到 DiT 块中，实现对视频生成的精确控制。

### 3. 核心结果/发现

<div align="center">
  <img src="/images/vla/LingBot-World-memory-capability.webp" width="100%" />
<figcaption>图：涌现的记忆能力：模型能够记住视野外的静态地标（如巨石阵），并在 60 秒后返回时保持结构一致，且能模拟视野外物体的动态演化。（图源：LingBot-World, 2026）</figcaption>
</div>

- **长程一致性**：模型表现出显著的涌现记忆能力，即使物体长时间离开视野，返回时仍能保持结构完整。
- **实时性与质量平衡**：LingBot-World-Fast 在单 GPU 节点上实现了 16 fps 的吞吐量，同时保持了与教师模型相当的视觉质量。
- **可控编辑**：支持通过文本指令（如“Firework”、“Fish”）对生成的世界进行实时干预。

<div align="center">
  <img src="/images/vla/LingBot-World-promptable-events.webp" width="100%" />
<figcaption>图：可控世界事件示例：通过文本提示词实时改变天气、风格或在场景中注入特定动态元素。（图源：LingBot-World, 2026）</figcaption>
</div>

### 4. 局限性
- **记忆稳定性**：长程一致性仍是基于上下文窗口的涌现能力，缺乏显式的存储模块。
- **交互精度**：对细粒度物体操作（如抓取特定物体）的支持尚不足。
- **算力成本**：推理仍需企业级 GPU 支持。

---

## 5.10 Marble & World Labs (2025–2026) {#sec-5-10-marble}
———大型世界模型 (LWM) 与空间智能

> [!TIP]
> 💡 **姊妹篇导读**：关于 3D 几何表征（如 3D Gaussian Splatting、NeRF、点云）与具身感知的深度融合，可进一步参阅本站的 **[空间智能全景综述：从 3D 几何表征、多模态时空推演到具身物理世界交互](/Spatial-Intelligence-Survey/)**。

🔗 **产品**: [marble.worldlabs.ai](https://marble.worldlabs.ai)  
🔗 **技术博客**: [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model)  
🔗 **API**: [World Labs API](https://www.worldlabs.ai)

### 精华

World Labs 代表了一种与主流 LLM 路线截然不同的 AGI 追求路径——**空间智能（Spatial Intelligence）**。其旗舰模型 Marble 是首个面向大众商用的**大型世界模型（Large World Model, LWM）**，能够从单张图片、一段视频或文本提示生成可探索、可编辑的持久 3D 世界。值得借鉴的核心思想包括：

1. **超越 2D 的多模态建模**：传统视频生成模型将世界压平为像素序列，LWM 直接在 3D 空间中建模——输入多模态，输出空间一致的 3D 环境，而非平面视频帧。
2. **3D 高斯泼溅作为世界表示（3DGS）**：Marble 采用 3D 高斯粒子作为最高保真度的世界表示，支持精确相机控制、交互式场景编辑，并能跨平台（手机到 VR 头显）实时渲染——这是 §8.2 时空 4D 感知方向的首个大规模工业落地。
3. **无限可探索的持久世界**：与生成单段视频不同，Marble 生成的 3D 世界没有时间限制，可以持续探索和扩展，支持导出为 Gaussian splats、传统网格（mesh）或视频。
4. **空间智能 × 具身 AI 评测**：通过 World API（2026 年初开放）和与光轮智能的战略合作，World Labs 正将 LWM 推向具身智能评测基础设施，解决"可规模化评估"这一机器人 AI 的核心瓶颈。

---

### 1. 研究背景/问题

李飞飞在 ImageNet 时代奠定了计算机视觉的数据基础，World Labs 的创立反映了她对 AI 下一阶段的判断：**当前 AI 缺失的核心能力是空间智能——即理解、生成和推理三维物理世界的能力**。

现有 LLM / VLM 的局限在于：它们本质上是"语言生物"，将世界压缩为 token 序列，缺乏在连续 3D 空间中感知和行动的能力。视频生成模型虽能生成逼真的 2D 视频序列，但无法支持视角自由移动、交互式编辑或精确的 3D 几何建模。

World Labs 提出的解决方案是构建 LWM：以多模态输入为条件，直接生成空间一致、高保真、持久化的 3D 环境。这与具身世界模型的终极目标高度契合——世界模型不应仅仅预测下一帧像素，而应理解和生成完整的三维物理世界。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/wm/MarbleWorldModelV4.svg" width="90%" />
<figcaption>图：Marble 大型世界模型（LWM）整体架构：多模态输入经过空间推理与 3D 世界生成，输出为可实时渲染、自由探索的 3D 高斯泼溅世界。（图源：World Labs）</figcaption>
</div>

**Marble 多模态输入体系**

Marble 支持四类输入模态，真正实现了多模态 → 3D 世界的生成：

| 输入类型 | 说明 |
|:---|:---|
| 文本提示（Text） | 直接描述目标世界的外观、风格和内容 |
| 单张图像（Image） | 将单张室内照片、风景图或艺术插画外推为可探索 3D 世界 |
| 视频片段（Video） | 从短视频或 360° 全景视频中重建空间结构 |
| 粗糙 3D 布局（Coarse 3D Layout） | 通过 Chisel 工具手绘草图或导入 3D 资产作为结构框架 |

<div align="center">
  <img src="/images/wm/i2w-living-room.webp" width="48%" />
  <img src="/images/wm/i2w-mushroom-forest.webp" width="48%" />
<figcaption>图：Image-to-World 示例：单张室内照片（左）和蘑菇森林插画（右）被 Marble 外推为完整可探索的 3D 世界。（图源：World Labs）</figcaption>
</div>

**3DGS 作为世界表示：选型的核心逻辑**

Marble 的核心技术选型为 **3D 高斯泼溅（3D Gaussian Splatting, 3DGS）**。3DGS 将 3D 场景表示为一组半透明粒子集合，在世界模型场景下具有显著优势：

<div align="center">
  <img src="/images/wm/streaming-3dgs-worlds.webp" width="90%" />
<figcaption>图：Marble 的流式 3DGS 渲染：生成的 3D 世界以高斯粒子表示，支持跨平台（手机到 VR 头显）实时渲染与自由视角探索。（图源：World Labs）</figcaption>
</div>

| 特性 | NeRF | 3DGS（Marble）|
|:---|:---:|:---:|
| 实时渲染 | ✗ | ✓ |
| 精确相机控制 | 有限 | ✓ |
| 交互式编辑 | ✗ | ✓ |
| 跨设备兼容 | ✗ | ✓（手机→VR）|
| 物理引擎集成 | 困难 | 自然兼容 |

**四大核心功能模块**

<div align="center">
  <img src="/images/wm/3d-as-code.webp" width="90%" />
<figcaption>图：Chisel 工具：用户通过盒子、平面等基础 3D 形状或导入现有 3D 资产确定世界结构，文本 prompt 控制整体风格，实现结构与风格的解耦。（图源：World Labs）</figcaption>
</div>

① **Chisel（AI 原生 3D 雕刻）**：实验性的 AI-native 3D 建模工具，允许用户在 3D 空间中直接用粗糙几何体（盒子、平面）或导入现有 3D 资产布置世界结构框架。核心设计原则是**结构与风格解耦**——粗糙 3D 场景决定世界的空间结构，文本 prompt 控制整体视觉风格，二者独立可控。

② **World Expansion（世界扩展）**：一键扩展已生成的世界边界，用户选定需要扩展的区域，Marble 自动生成更多连续一致的内容填充选定区域，支持无限延伸。

③ **Composition Mode（合成模式）**：将任意数量的独立世界组合为超大规模空间。各子世界的位置和衔接完全由用户控制，适用于游戏场景、VFX 大型布景或机器人仿真测试场的构建。

④ **Video Enhancement（视频增强）**：对生成的 3D 世界渲染输出进行后处理，去除伪影、添加动态元素（如人物、粒子效果），同时保持像素级精确的相机控制和 3D 结构一致性。

**多格式导出与生态集成**

| 导出格式 | 用途 |
|:---|:---|
| Gaussian Splats | 最高保真度，用于实时渲染、VR/AR |
| Collider Mesh | 低精度碰撞网格，用于物理引擎仿真 |
| High-Quality Mesh | 高精度三角网格，用于 CG 生产管线 |
| Video | 固定路径视频导出，用于内容创作 |

**持久世界 vs. 视频流：设计哲学的差异**

与本文 §5.1 Lyra 2.0 的视频流路线不同，Marble 追求的是**空间中持久存在的 3D 世界**：用户可以自由导航、探索任意角度，世界不会因为"走远了"而消失。这种"持久性（Persistency）"设计原则，使 Marble 更接近传统游戏引擎的能力边界，而非视频生成模型。

---

### 3. 核心结果/发现

**具身智能 × 评测基础设施：与光轮智能的战略合作（2026 年 1 月）**

2026 年 1 月，World Labs 与国内仿真合成数据公司光轮智能联手，目标是系统性解决具身智能的规模化评测难题。行业三大困境被明确点名：

- 学术级基准已跟不上模型进化速度（对应 §7.1 中 LIBERO/CALVIN 性能饱和现象）；
- 真机测试成本高、周期长（对应 §8.5 失效感知动力学的数据瓶颈）；
- 传统仿真评测停留在理想化场景（对应 §9.3 研判中提到的真实世界泛化鸿沟）。

两者的分工互补，形成完整的评测驱动闭环：

```mermaid
flowchart LR
    WL["World Labs\n（世界从哪来）\nLWM 生成多样化 3D 虚拟环境\n作为机器人策略测试场"] --> Eval
    GR["光轮智能\n（进步如何被衡量）\n可扩展评测框架\n多维度、自动化基准"] --> Eval
    Eval["具身智能评测基础设施\n可规模化 · 自动化 · 接近真实"] --> Robot["机器人 VLA 策略\n持续迭代优化"]
    Robot --> WL
```

**商业规模与行业认可**

| 指标 | 数据 |
|:---|:---|
| Marble 发布时间 | 2025 年 11 月 |
| 2026 年 2 月融资 | $10 亿美元（总融资 $12.3 亿） |
| 公司估值 | ~$50 亿（较创立时 $10 亿增长 5×） |
| 主要投资方 | NVIDIA、AMD、Autodesk 等 |
| 行业认可 | Forbes AI 50 2026 |
| 世界模型赛道融资变化 | $14 亿（2024）→ $69 亿（2025）|

---

### 4. 局限性

- **技术细节未公开**：World Labs 目前尚未发表技术论文，Marble 的具体模型架构、训练数据规模和方法细节尚未披露，学术社区难以复现与跟进。
- **物理动力学有限**：当前 Marble 更侧重于静态/准静态场景的高保真生成，对复杂动力学（流体、碰撞、软体形变）的仿真能力尚不明确，与 Cosmos（§4）在物理精度上的定位有所差异。
- **机器人专用验证缺失**：公开应用案例主要集中在游戏、VFX、设计领域，机器人操控任务上的量化评测尚未公开。
- **实时交互延迟**：复杂场景下 3DGS 渲染的计算开销仍是规模化部署的瓶颈，尤其在低端设备上。

---

## 5.11 SANA-WM (2026) {#sec-5-11-sana-wm}
———Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer

📄 **Paper**: [arXiv:2605.15178](https://arxiv.org/abs/2605.15178)  
🔗 **项目主页**: [nvlabs.github.io/Sana/WM](https://nvlabs.github.io/Sana/WM/)

### 精华

SANA-WM 最值得借鉴的核心思想是**以效率为第一设计目标的世界模型**：用 2.6B 参数、64 块 H100、15 天训练，在单 GPU 上生成分钟级 720p 视频，达到与 14B+14B 工业级模型相当的视觉质量。具体可迁移的设计包括：(1) **混合线性-Softmax Attention（Hybrid GDN/Softmax）**——以帧粒度 Gated DeltaNet 替代大多数 Softmax 层，使 KV 状态保持 $D \times D$ 常数，内存不随序列长度增长，Softmax 层仅保留 5/20 块用于长程精确回忆，巧妙平衡效率与质量；(2) **双分支相机控制（Dual-Branch Camera Control）**——粗分支 UCPE 在潜帧率上捕获全局 6-DoF 轨迹结构，细分支 Plücker 在原始帧率上补偿 VAE 压缩丢失的帧内运动细节，两者协同实现高精度连续轨迹跟随；(3) **两阶段视觉精化（Two-Stage Refiner）**——第一阶段生成结构正确但质量稍逊的视频，第二阶段用截断 Flow Matching 的 17B LoRA 精化器无缝修复细节，整体吞吐仍达 22 个 60s 视频/小时；(4) **鲁棒度量标注管线**——用 VIPE+Pi3X+MoGe-2 从公开视频恢复度量尺度 6-DoF 姿态，无需昂贵专有数据，仅 213K 片段即完成训练。

---

### 1. 研究背景/问题

现有开源分钟级世界模型（LingBot-World 14B+14B、HY-WorldPlay 8B）普遍需要大模型参数量、海量专有数据、多 GPU 推理，对学术界和小团队门槛极高。另一种替代——用短视频生成器蒸馏长程模型——因短程教师对分钟级场景持久性和轨迹跟随的监督信号不足而效果有限。SANA-WM 的目标是：**在严格效率约束下原生训练一个高保真、可相机控制的分钟级世界模型**，使其在单 GPU 上可推理、在 64 块 H100 上 15 天可收敛。

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/wm/SANA-WM-teaser.webp" width="100%" />
<figcaption>图：SANA-WM 概览。从单张图像和动作轨迹出发，生成分钟级 720p 世界，支持精确相机控制、64-GPU 训练与单 GPU 推理。（图源：SANA-WM，arXiv:2605.15178）</figcaption>
</div>

**整体框架**：SANA-WM 由四个核心组件构成：① 混合线性 DiT 骨干（Hybrid GDN/Softmax）负责高效长程上下文建模；② 双分支相机控制（UCPE + Plücker）负责精确 6-DoF 轨迹注入；③ 第二阶段视觉精化器（LTX-2 LoRA）负责提升最终帧质量；④ 鲁棒度量标注管线（VIPE+Pi3X+MoGe-2）负责从公开视频提取高质量训练数据。

<div align="center">
  <img src="/images/wm/SANA-WM-architecture.webp" width="100%" />
<figcaption>图：SANA-WM 架构。文本、视频和姿态 Token 交替经过 GDN 块和 Softmax 块；UCPE Attention 和 Plücker Mixing 提供几何感知的相机条件；第二阶段精化器进一步提升视觉质量。（图源：SANA-WM）</figcaption>
</div>

#### ① 混合线性-Softmax Attention（Hybrid GDN/Softmax）

**输入**：时序潜帧序列（LTX2 VAE 编码，时间×高×宽压缩比远高于 Wan 2.1-VAE，尺寸缩小 8×）。

**处理**：SANA-WM 共 20 个 Transformer Block，其中 15 个为**帧粒度 Gated DeltaNet（GDN）块**，5 个（位于层 3/7/11/15/19）为标准 Softmax Attention 块。GDN 块的关键是将 token 级递推（每步一个 token）升级为**帧级递推**（每步消费一个潜帧的全部 $S$ 个空间 token），状态矩阵 $S_t \in \mathbb R^{D \times D}$ 通过衰减门 $\gamma_t$ 和 delta-rule 修正实现"遗忘旧信息、精准更新当前帧"：

$$S_t = S_{t-1} M_t + U_t, \quad M_t = \gamma_t(I - \hat K_t \beta_t \hat K_t^\top), \quad U_t = V_t \beta_t \hat K_t^\top$$

为防止空间 token 数 $S$ 导致转移矩阵 $M_t$ 膨胀，对 key 施加 $1/\sqrt{DS}$ 缩放（代替 token 级的 $1/\sqrt{D}$），确保 $\lVert M_t \rVert_2 \le \gamma_t \le 1$，训练稳定不出现 NaN。Softmax 块负责精确长程回忆，在 60s 序列中引入局部注意力窗口和 attention sink，使推理时 Softmax 内存保持常数。

**设计动机**：60s 720p 视频展开为约 961 个潜帧；纯 Softmax 的 KV Cache 随序列长度平方增长，60s 时直接 OOM；纯线性注意力（如 SANA-Video 的累积线性注意力）缺乏衰减机制，旧特征与新特征等权累积，导致分钟级建模出现漂移。混合设计兼顾了"大多数时间步高效更新 + 关键时刻精确回忆"的需求。

#### ② 双分支相机控制（Dual-Branch Camera Control）

**粗分支（Coarse - UCPE）**：在**潜帧率**上建模全局 6-DoF 轨迹。对每个潜帧 $t$ 和空间格 $s$，由相机外参计算世界空间射线，构建射线局部坐标系变换 $D_{t,s} \in \mathbb R^{4 \times 4}$，将 QKV 的几何通道经 $D^\top / D^{-1}$ 旋转，其余通道保留 RoPE——本质是将相机位姿编码进注意力位置编码。该分支有独立 QKV 投影，但与主分支共享 GDN 门，通过零初始化投影叠加到主注意力输出上。

**细分支（Fine - Plücker Mixing）**：弥补粗分支因 VAE 将 8 个原始帧压缩为 1 个潜帧而丢失的帧内运动细节。对每个原始帧 $r$ 和像素 $p$，计算 Plücker 射线 $\rho_{r,p} = (d_{r,p},\, o_r \times d_{r,p}) \in \mathbb R^6$，将 VAE 步内 8 帧的 Plücker 图堆叠为 48 通道张量，经零初始化 3D Patch Embedder 处理后逐块叠加至自注意力输出后。

**消融验证**：UCPE+Plücker 组合在 OmniWorld 验证集上 CamMC 达 0.2047，优于单独 UCPE（0.2453）和单独 Plücker（0.4742），FVD 也更低。

#### ③ 两阶段视觉精化

第一阶段（SANA-WM 主模型）生成结构正确的 60s 视频。第二阶段精化器基于 LTX-2 17B 模型，只训练秩 384 LoRA（附于 Q/K/V/O 和 FFN），用截断-$\sigma$ Flow Matching 对第一阶段噪化潜变量进行精化（3 步 Euler，推理时与主模型完全解耦）。精化后 VBench Overall 从 79.29 提升至 80.62（Simple Trajectory），同时将晚期质量退化 $\Delta IQ$ 从 3.79 压缩至 1.17。

#### ④ 鲁棒度量标注管线与数据

<div align="center">
  <img src="/images/wm/SANA-WM-data-pipeline.webp" width="100%" />
<figcaption>图：SANA-WM 数据构建管线。收集开源视频与静态 3D 资源，标注度量尺度相机姿态，用 3DGS 渲染增强 DL3DV，过滤/字幕处理后得到 213K 片段训练语料。（图源：SANA-WM）</figcaption>
</div>

标注引擎基于 VIPE，将深度估计后端替换为 Pi3X（多帧一致结构）+ MoGe-2（度量尺度锚点），支持公开视频的鲁棒度量尺度 6-DoF 姿态提取。对 DL3DV 等静态 3D 数据集，用 FCGS 拟合 3DGS 重建后渲染多样化一分钟相机路径，再经 DiFix3D 精化减少拼接伪影，生成 14,881 条合成 60s 片段。最终语料共 212,975 条片段，涵盖室内、室外、游戏、合成场景。

**渐进式训练策略**（4 阶段，共约 15 天 64× H100）：

| 阶段 | 目标 | 序列长度 | 训练步数 |
|:---|:---|:---|:---|
| Stage 1 | VAE 适配（LTX2 空间对齐） | 5s | 50K（VAE）+ 30K（DiT） |
| Stage 2 | 混合架构适配（GDN/Softmax） | 5s | 30K |
| Stage 3 | 分钟级扩展 + 相机控制 | 60s | 31K |
| Stage 4 | Chunk-Causal 微调 + 4步蒸馏 | 60s | 10K |

---

### 3. 核心结果/发现

<div align="center">
  <img src="/images/wm/SANA-WM-qualitative-comparison.webp" width="100%" />
<figcaption>图：四种方法在 Hard Trajectory 60s 视频上的定性对比。绿色边框为 SANA-WM，左下角为动作轨迹叠加。SANA-WM 在复杂轨迹下仍保持场景一致性，基线方法则出现模糊、布局漂移或结构崩溃。（图源：SANA-WM）</figcaption>
</div>

**相机控制精度**（↓ 越低越好）：SANA-WM+精化器在 60s 基准上取得最优 RotErr（4.50°/8.34°，Simple/Hard），CamMC 1.41/1.44，全面优于 LingBot-World（14B+14B，RotErr 10.47°/18.99°）、Matrix-Game 3.0（5B）和 HY-WorldPlay（8B）。

**视觉质量**：精化后 VBench Overall 80.62/81.89（Simple/Hard），与 LingBot-World（81.82/81.89）相当，但 LingBot-World 需要 8 块 H100（454.1 GB 显存），SANA-WM 单 GPU 仅需 74.7 GB。

<div align="center">
  <img src="/images/wm/SANA-WM-efficiency-ablation.webp" width="100%" />
<figcaption>图：效率消融与扩展性分析。(a) 60s 单 GPU 推理延迟分解：经蒸馏+attention sink+NVFP4 量化，RTX 5090 上 34s 生成一条 60s 720p 视频。(b) H100 延迟与显存随视频时长的变化：混合 GDN/Softmax 线性增长，纯 Softmax 在 60s 时 OOM。（图源：SANA-WM）</figcaption>
</div>

**推理效率**：SANA-WM 生成吞吐 24.1 视频/小时（8× H100），比最快 480p 基线 Infinite-World 快 4.1×；4步蒸馏+NVFP4 量化后，单 RTX 5090 仅需 34s 生成一条完整 60s 720p 视频（**36× 高于 LingBot-World 的吞吐**）。

**渐进训练消融**（VBench-I2V）：

| 配置 | VBench Total ↑ | 峰值内存 (GB) ↓ | 推理速度 (steps/s) ↑ |
|:---|:---:|:---:|:---:|
| SANA-Video（原始） | 0.838 | 8.90 | 0.79 |
| + LTX2 VAE | 0.839 | 5.40 | 2.69 |
| + Hybrid GDN/Softmax | **0.853** | 5.68 | 2.31 |

---

### 4. 局限性

SANA-WM 受限于规模（2.6B 参数与 213K 片段），在动态场景、罕见视角或超长序列中仍可能产生漂移，且缺乏显式 3D 场景记忆（无法像 Lyra 2.0 那样精确"重访"旧区域）。未来工作需要扩大模型与数据规模、引入机器人动作或点跟踪控制、强化持久场景记忆，以及开发鲁棒的实时或流式精化器。

---

## 5.12 Qwen-RobotWorld (2026) {#sec-5-12-qwen-robotworld}
———用自然语言统一具身世界模型：机械臂操作、自动驾驶、室内导航和人到机器人迁移

📄 **Paper**: [arXiv:2606.17030](https://arxiv.org/abs/2606.17030)

### 精华

把"语言指令"当成唯一的统一动作接口，不同具身领域（操作、驾驶、导航、人到机器人迁移）的视频生成任务就能被改写成同一个 $s_{t+1} = f(s_t, a_t)$ 问题，从而联合训练而不互相冲突。用冻结的 MLLM（Qwen2.5-VL）做动作编码器，比 T5/CLIP 更能利用其内部世界知识（刚体约束、关节限制）隐式约束生成的物理合理性。双流 MMDiT 通过逐层联合注意力，让语言条件与视觉隐变量在每一层都双向融合，而不是只在输入端拼接一次。Scene2Robot 用"分段拼接 + 仅对生成段计损失"的方式，在不改架构的前提下把同一个 TI2V 骨干复用成跨具身视频编辑工具。数据侧的核心是把 20+ 机器人本体、500+ 动作类别全部映射到统一的自然语言描述，这比堆数据量本身更关键。

---

### 1. 研究背景/问题

通用视频生成模型（Sora2、Veo3 等）从互联网数据学到了丰富的视觉先验，但不懂接触动力学、刚体约束等具身物理规律；而 Cosmos、LVP 等领域专用具身世界模型虽然懂物理，却依赖关节角、路径点等机器人专属动作表示，无法跨本体、跨任务泛化。Qwen-RobotWorld 希望用自然语言作为统一动作接口，把操作、驾驶、导航、人到机器人迁移这四类互补的物理知识在同一个骨干网络里联合训练，相互增强而不是各自为战。

---

### 2. 主要方法/创新点

Qwen-RobotWorld 由三部分构成：**架构**（双流 MMDiT + MLLM 动作编码）、**数据**（EWK 具身世界知识数据集）、**训练**（通用先验+专家能力的渐进式课程）。这三者紧密耦合：数据提供统一语言接口下的多领域监督信号，架构保证语言语义和视觉状态在每一层都能融合，训练策略则决定先学什么再学什么。

<div align="center">
  <img src="/images/wm/Qwen-RobotWorld-EWK-dataset-overview.webp" width="100%" />
<figcaption>图：EWK 训练语料概览：通用世界数据提供外观、几何、动力学先验；结构化具身数据沿 Multi-Embodiment、Multi-Task、Multi-Scenario、Multi-View 四个维度组织，共同支撑语言条件下的动作理解和未来状态生成。</figcaption>
</div>

**① 整体框架概述**

模型核心是一个 60 层的双流 Multimodal Diffusion Transformer（MMDiT）：理解流（understanding stream）处理冻结 Qwen2.5-VL 抽取的语言语义特征，代表动作 $a_t$；生成流（generation stream）处理视频 VAE 编码出的视觉隐变量，代表状态 $s_t$。两条流在每一个 block 都通过联合注意力交互，而不是只在输入层做一次拼接，这样去噪的每一步都能让视觉隐变量同时关注语义动作信号。

<div align="center">
  <img src="/images/wm/Qwen-RobotWorld-architecture.webp" width="100%" />
<figcaption>图：双流 MMDiT 架构：冻结的 Qwen2.5-VL 编码语言动作，VAE 编码视频观测/预测帧的隐变量，二者在每层 MMDiT block 中联合注意力交互。</figcaption>
</div>

**② 逐模块讲解**

- **MLLM 动作编码器**：输入是一句自然语言指令（如"用右手拿起粉色瓶子，把水倒在花上"），通过冻结的 Qwen2.5-VL 提取末层隐藏状态 $h = \phi(S)$ 作为条件信号。用 MLLM 而非轻量编码器（T5、CLIP）的原因有两点：(1) 深层语言理解能把复杂的组合式指令精确解析为细粒度的状态转移条件；(2) MLLM 内部沉淀的世界知识（例如机械臂是刚体、有固定连杆长度和关节约束）能隐式约束生成空间中物理合理的状态转移，配合 T2I 联合训练可以防止视频帧间的物体形变，这是缺乏语义接地的模型常见的失败模式。
- **VAE 状态编码/解码器**：采用 Wan-VAE 架构，把视频帧编码为隐变量 $z = \mathcal{E}(x)$，并把预测的隐变量解码回视觉观测，同时支持图像和视频两种模态。
- **MMDiT 转移函数**：双流设计中，理解流接收经可训练 connector 投影后的 MLLM 编码，生成流接收 VAE 输出的带噪状态隐变量。骨干共 60 个双流 block，24 个注意力头（每头维度 128），隐藏维度 3072，patch size 2×2；总参数量 MLLM 7B、VAE 127M（编码器 54M+解码器 73M）、MMDiT 20B，最长支持 48,360 个视频 token。
- **3D RoPE 位置编码**：时间、空间高、空间宽三个维度独立编码，采用非对称划分（pe_axes_dim = [16, 56, 56]）——时间轴维度少是因为相邻帧强相关，空间轴维度多是为了捕捉更丰富的物体位置和场景布局差异；同时配合 Scalable RoPE 支持推理时泛化到不同分辨率和时长。

**③ Scene2Robot：跨具身视频编辑**

人到机器人迁移本质是一个视频编辑问题：模型需要同时参考场景上下文（背景、物体布局、光照）和目标机器人的运动轨迹。Scene2Robot 在不改架构的前提下，把输入组织成三个连续分段：场景条件段（人类示范视频，人手已被遮罩处理，F 帧）、机器人参考段（MuJoCo 渲染的仿真机器人执行，F 帧）、生成段（待去噪的噪声隐变量，F 帧）。前两段都被赋予时间步 $t=0$ 并排除在去噪损失之外，只有生成段参与梯度更新；3D RoPE 给每个分段分配各自的时间索引范围。每一层 MMDiT 的联合注意力让生成段同时关注场景外观、机器人运动轨迹和语言动作语义，从而合成既保留场景上下文又遵循指令操作的逼真机器人执行视频。

<div align="center">
  <img src="/images/wm/Qwen-RobotWorld-Scene2Robot.webp" width="100%" />
<figcaption>图：Scene2Robot 多分段条件机制：场景条件段、机器人参考段仅提供条件（赋时间步 0、不计损失），生成段通过联合注意力同时关注场景外观与机器人运动轨迹，实现跨具身视频合成。</figcaption>
</div>

**④ 数据：EWK 数据集与动作-语言映射**

核心数据贡献是**动作-语言映射框架**：把 20+ 机器人本体类型、500+ 动作类别统一投射到自然语言空间，使 Franka 夹爪、自动驾驶车辆、室内导航 agent 的视频都变成"同一种语言条件视频生成任务"的实例。最终构成约 860 万视频-文本对、超过 2 亿观测帧的 EWK 数据集：操作领域约 590 万样本（20+ 机器人形态、1300+ 技能）为核心，自动驾驶约 20 万样本（Waymo、NVIDIA PhysicalAI-AD、Bench2Drive、Sekai），室内导航 6000+ 语言引导轨迹（VLNVerse），以及通过 MANO 重建+逆动力学渲染自动生成的人到机器人迁移数据（覆盖 14 种机器人形态）。

标注上采用**五层分层标注框架**：任务目标层（要发生什么状态转移）→ 动作细节层（拆解为时空轨迹、微动作、速度力度，并显式声明视角：第一人称主视角/手腕视角/外部视角/多视角拼接）→ 物理反馈层（物体位移、形变、接触状态等可视化验证的后果）→ 综合描述（50-100词）和简要描述（15-30词）两种粒度，训练时按 50%/50% 概率采样，让模型既能执行详细轨迹指令也能响应简短高层命令。

**⑤ 训练目标**

采用 flow matching 目标，输入视频经 VAE 编码到隐空间，噪声采样自标准正态分布；时间步采用基于视频序列长度自适应偏移的对数正态分布采样；TI2V 任务中首帧时间步固定为 0 以确保生成过程以给定观测帧为条件。训练分两阶段：**预训练阶段**联合训练 T2I/T2V/TI2V 任务建立通用视觉先验（T2I 锚定几何正确的物体形态，迁移到视频生成防止形变）；**SFT 阶段**按四阶段课程逐步注入具身数据（70%具身/30%通用混合）：单视角操作 → 多视角扩展 → 多视角拼接生成 → 复杂任务与跨域数据，具身部分中操作任务占约 90% 采样权重保证物理理解深度，多视角拼接和导航/驾驶各占约 5% 保证广度。

---

### 3. 核心结果/发现

在四个基准上评测：**EWMBench** 综合得分 4.60 排名第一（领先第二名 LVP 的 4.05 达 +0.55），其中运动保真度 HSD 达 0.566，比第二名高 33%。**DreamGen Bench**（GR1 机器人三个子集）总分 4.952 排名第一，物体级组合泛化能力（GR1-Object IF 0.878）最强。**PBench** 总分 0.804，超过所有开源模型，领域理解 0.857 排第 3，运动平滑度 0.990 在开源模型中排第 2。**WorldModelBench** 总分 8.99，超过所有开源模型（仅次于闭源 Wan2.6、Veo3），物理符合性（牛顿定律、质量守恒、流体动力学、重力）四项均满分。

定性结果显示该模型支持细粒度语言接地（仅改变指令中的一个关键词即可产生不同的操作视频）、跨本体泛化（同一指令驱动单臂夹爪、双臂系统、人形机器人、灵巧手等四种形态而无需专门适配）、多视角一致性，以及人到机器人迁移、自动驾驶场景合成、室内导航生成等跨域能力。在 RoboTwin-IF 零样本基准上，尽管训练时只混入了少量 RoboTwin 开源数据，仍展现出较强的零样本指令跟随和多视角一致性，优于 LVP 和 Cosmos2.5-14B 两个强基线。

---

### 4. 局限性

由于模型专为具身任务设计且输出分辨率低于通用视频生成器，PBench 上的美学质量（0.455）和成像质量（0.649）相对较低；WorldModelBench 上的常识维度（帧/时序质量）也因分辨率原因落后于通用模型。DreamGen Bench 的长时程行为泛化（GR1-Behavior IF 0.832）略逊于 LVP 和 GigaWorld，仍有提升空间。

---

## 5.13 Wan2.1 (2025) {#sec-5-13-wan}
———阿里巴巴开源的高效视频生成基础模型家族

📄 **Paper**: https://arxiv.org/abs/2503.20314

### 精华

- 提出了 **Wan2.1** 视频生成模型家族，采用主流的 Diffusion Transformer (DiT) 架构，包含 1.3B 和 14B 参数两个版本，开源了全部代码与权重。
- 引入了创新的 **Spatio-Temporal VAE (Wan-VAE)**，能够将视频在时空维度上压缩 4x8x8 倍，并引入 RMSNorm 和特征缓存机制以支持任意长度的长视频流式重建与低内存推理。
- 针对 DiT 训练，优化了特征调制（AdaLN）参数共享设计，不仅使模型参数量减少约 25%，还显著加快了收敛速度并提升了指令遵循能力。
- 采用 **2D 上下文并行（Ulysses + Ring Attention）** 和 FSDP 混合分布式并行策略，解决了超长序列（达 1M 级别 tokens）所带来的显存和计算瓶颈。
- 构建了统一的视频控制与编辑框架 **VACE**，通过对掩码区域和非掩码区域的“概念解耦”时空编码，实现了高质量的局部视频编辑、视频外扩等下游任务。

---

### 1. 研究背景/问题

- 现有的视频生成模型在生成大幅度动作、高保真画面、超长视频以及复杂的文本提示词理解上仍面临巨大挑战。
- 同时，大模型的高显存消耗和计算复杂度使得它们难以在消费级显卡（如 RTX 4090）上运行，极大地限制了开源社区的二次开发与应用。
- 此外，时空自编码器（VAE）往往缺乏良好的时空因果性保证，且在流式长视频生成中面临内存溢出和边界不连续等缺陷。

---

### 2. 主要方法/创新点

#### Wan-T2V (Text-to-Video) 整体架构

<div align="center">
  <img src="/images/wm/Wan-T2V-architecture.webp" width="100%" />
<figcaption>图：Wan 文本到视频生成（T2V）的整体架构图。（图源：Wan2.1, 2025）</figcaption>
</div>

**① 整体框架概述**
Wan2.1 整体架构基于 Diffusion Transformer (DiT) 范式，包含三个核心模块：用于将视频/图像从像素空间压缩到低维潜空间的 **Wan-VAE**、执行流匹配去噪过程的 **Diffusion Transformer (DiT)** 以及用于文本理解的 **umT5 文本编码器**。

**② 逐模块讲解**

- **Wan-VAE (Spatio-Temporal VAE)**：
  - **输入**：大小为 $(1+T) \times H \times W \times 3$ 的高维原始视频。
  - **处理**：采用 3D 因果卷积结构，其中第一帧仅进行空间压缩（以保留图像先验），其余帧进行时空联合压缩。模型将所有 GroupNorm 替换为 RMSNorm 以保持严格的临时因果性，并支持特征缓存机制（Feature Cache Mechanism）。在空间上采样层中，将输入特征通道减半，以降低 33% 的推理显存。
  - **输出**：时空维度压缩了 $4 \times 8 \times 8$ 倍、通道数为 16 的低维潜空间表征 $x \in \mathbb{R}^{(1+T/4) \times H/8 \times W/8 \times 16}$。
  - **特征缓存推理**：在处理超长视频时，将视频按 Latent 帧拆分为 Chunks（每块最多 4 帧），在块与块之间传递和复用前一阶段的最后两帧特征缓存，确保在受限的显存内实现连续、无缝的流式重建。

<div align="center">
  <img src="/images/wm/Wan-VAE-framework.webp" width="100%" />
<figcaption>图：Wan-VAE 时空压缩自编码器架构图。（图源：Wan2.1, 2025）</figcaption>
</div>

- **umT5 文本编码器**：
  - **输入**：用户输入的自然语言提示词（支持中英双语以及复杂的排版描述）。
  - **处理**：利用双向注意力机制编码，相比于单向注意力 LLM 更加注重全局语义表示与空间排版。
  - **输出**：长度为 512 的语义 Token 序列 $ctxt \in \mathbb{R}^{512 \times D_{text}}$。

- **Diffusion Transformer (DiT)**：
  - **输入**：经由 3D 卷积（Patchify，核大小为 $(1, 2, 2)$，步长为 $(1, 2, 2)$）打块并展平后的潜空间序列 $x_{flat} \in \mathbb{R}^{B \times L \times D}$，以及文本 Token 和时间步 $t$。
  - **处理**：由 $N$ 层堆叠的 Wan Transformer Block 构成。在 Block 内部，通过自注意力（Self-Attention）机制捕获时空关系，通过交叉注意力（Cross-Attention）将文本 Token 注入到图像 Token 中。时间步 $t$ 编码经由一个全局共享的 MLP (Linear + SiLU) 映射为调制参数，以调节各 LayerNorm 的尺度与偏置。
  - **输出**：预测的去噪速度向量 $v_t$。

<div align="center">
  <img src="/images/wm/Wan-transformer-block.webp" width="80%" />
<figcaption>图：Wan Transformer Block 结构细节。（图源：Wan2.1, 2025）</figcaption>
</div>

**③ 端到端数据流**
训练时，原始视频经 Wan-VAE 编码为 Latent 状态，与高斯噪声进行 Flow Matching（流匹配）线性插值得到 $x_t$，通过 Patchify 模块转换为 1D Token 序列；同时文本经 umT5 编码为文本 Embedding。在 DiT Blocks 中，文本与时空 Token 通过 Cross-Attention 进行交互。最后，利用预测的 Velocity $v_t$ 引导 ODE 求解去噪，生成的 Latent 再由 Wan-VAE Decoder 恢复出清晰的视频画面。

**④ 训练目标 / 损失函数**
基于 Rectified Flows (RFs) 框架，中间潜空间 $x_t$ 通过对干净视频潜特征 $x_1$ 和高斯噪声 $x_0 \sim \mathcal{N}(0, I)$ 实施线性插值获得：
$$x_t = tx_1 + (1-t)x_0$$
真值变化速率为 $v_t = x_1 - x_0$。模型学习参数 $\theta$ 以拟合这个变化率 $u(x_t, ctxt, t; \theta)$，损失函数采用均方误差 (MSE)：
$$\mathcal{L} = \mathbb{E}_{x_0, x_1, ctxt, t} \left[ \lVert u(x_t, ctxt, t; \theta) - v_t \rVert^2 \right]$$

#### Wan-I2V (Image-to-Video) 架构与控制框架

<div align="center">
  <img src="/images/wm/Wan-I2V-architecture.webp" width="100%" />
<figcaption>图：Wan-I2V 图生视频模型框架。（图源：Wan2.1, 2025）</figcaption>
</div>

**① 整体框架概述**
为了兼容图片生成视频（I2V）、视频续写（Video Continuation）以及首尾帧过渡（First-Last Frame Transition）等多种下游任务，Wan 引入了掩码（Mask）机制和双编码器联合调节策略。

**② 模块与数据流详解**
- **双图像编码器**：同时输入第一帧像素，一方面经由 **Wan-Encoder** 编码为 Latent，作为与噪声等维度的掩码提示，并和掩码矩阵 $M$ 一起与嘈杂的潜空间特征 $x_t$ 进行 Channel-wise Concatenation（通道级拼接）作为 DiT 的主轴输入；另一方面，通过 **CLIP Image Encoder** 提取全局语义特征，在 DiT 的 **Decoupled Cross-Attention（解耦交叉注意力）** 中与 umT5 文本 Embedding 一起分别与时空特征进行交互，提供高保真视觉细节与空间语义。
- **掩码通道设计**：对第一帧画面（或已知参考帧）赋予值为 0 的掩码（代表需要被重建的区域），对其余生成帧赋予值为 1 的掩码（代表生成区域）。该设计使用户能自由指定参考帧的空间与时间排布。

#### VACE：统一的可控生成与编辑框架

<div align="center">
  <img src="/images/wm/Wan-VACE-editing-framework.webp" width="100%" />
<figcaption>图：VACE 可控生成与编辑模型框架与概念解耦机制。（图源：Wan2.1, 2025）</figcaption>
</div>

**① 整体框架概述**
**VACE (Video Condition Unit)** 旨在将局部重绘（Repainting）、Canny 边缘提取、深度估计（Depth）、姿态引导（Pose）以及线稿引导（Scribble）等多种编辑和生成条件统一到同一种输入范式中。

**② 数据流与概念解耦 (Concept Decoupling) 详解**
- **概念解耦策略**：为保证在各种不同控制任务下模型能够平稳收敛，VACE 将输入视频 $F$ 和掩码 $M$ 解耦为两个相同尺寸的序列：**活性帧** $F_c = F \times M$（包含所有需要被修改的像素）与 **惰性帧** $F_k = F \times (1-M)$（保留所有需要保持原样的像素）。
- **编码与注入**：$F_c$ 和 $F_k$ 分别通过同一个冻结的 Wan-VAE Encoder 映射到潜空间，并在通道维度与噪声拼接后输入到 DiT 中。VACE 提供两种训练模式：**Fully Fine-tuning**（全参数微调）以及 **Context Adapter Tuning**（通过外挂的 Context Block 以残差形式集成到原 DiT block 中，支持无损基础权重插拔）。

---

### 3. 核心结果/发现

- **性能优异**：14B 模型在大规模图像与视频数据集上训练，在各项内部和外部基准测试中超越了当时的主流开源模型（如 CogVideoX、Hunyuan Video 等）及闭源商业模型。
- **高压缩比与高质量**：Wan-VAE 的时空压缩比达到 $4 \times 8 \times 8$，潜表征维度为 16 维。在 720p 分辨率及 25 帧的视频重建测试中，重建质量（PSNR）与 Hunyuan Video 相当甚至更好，同时重建速度快了 **2.5 倍**。
- **极低的计算硬件门槛**：1.3B 模型专门为消费级 GPU（如 RTX 4090）设计，开启 int8 甚至 TensorRT 量化后，推理时仅需 **8.19 GB** 显存，却在 T2V 任务上能产生媲美更大模型的流畅度和一致性。
- **首创双语字符生成**：在视频中实现了中英双语的高清、正确字符排版生成能力（如生成包含 "Wan2.1" 和中文牌匾的视频）。

---

### 4. 局限性

- 模型在处理极其复杂的极速物理交互（如破碎、流体变化等细微碰撞细节）时，依然会出现一定程度的幻觉或时空扭曲。
- 尽管 1.3B 模型实现了消费级显卡部署，但 14B 参数模型在单卡推理时仍具有较高的计算延迟，在大规模生产部署中仍然需要多卡 Context Parallel 协同。

---

## 5.14 Motus (2025/2026) {#sec-5-14-motus}
———统一隐动作世界模型：Mixture-of-Transformers 与光流动作金字塔

📄 **Paper**: [arXiv:2512.18876](https://arxiv.org/abs/2512.18876) · [Project Page](https://motubrain.com) · [Code](https://github.com/PKU-YuanGroup/Motus) · 清华大学 & 生数科技

### 精华

Motus 是首个将**混合专家 Transformer（Mixture-of-Transformers, MoT）**与**统一扩散生成调度（UniDiffuser）**引入机器人双臂操作的统一世界动作模型（WAM）。最值得借鉴的核心设计包括：
1. **解耦 FFN 的三专家 MoT 架构**：在单个 DiT 骨干中集成多模态理解专家、视频生成专家与连续动作专家。三类 token 在每层共享自注意力实现深层信息互通，但在前馈网络（FFN）处解耦，既杜绝了模态特征干扰，又赋予模型在世界模型、VLA 策略、逆动力学和联合仿真之间任意切换的能力；
2. **基于光流的潜动作模型（Latent Action VAE）**：彻底摆脱对真机机械臂特定关节角（Joint Angles）标签的强依赖！提出直接从视频像素光流（Optical Flow）中自监督提取潜动作向量，使互联网海量无标注人类交互视频能够无缝转化为机器人可用的预训练数据；
3. **六层数据金字塔**：构建了从通用视频、带相机位姿视频、人类手部交互视频到真机遥操作的逐级蒸馏数据金字塔，利用像素级“差分动作（Delta Action）”实现跨本体知识迁移；
4. **RoboTwin 2.0 双臂仿真与真机 SOTA**：在极具挑战性的细粒度双臂操作基准 RoboTwin 2.0 上，Motus 平均成功率 87.8%，较 X-VLA（72.8%）高出 15.0 个百分点、较 $$\pi_{0.5}$$（79.8%）高出 8.0 个百分点，其后续演进版 Motubrain 更实现了 10× 推理加速。

---

### 1. 研究背景/问题

具身智能系统长期面临严重的技术割裂：VLA 策略模型（如 OpenVLA、$$\pi_0$$）仅关注从图像映射到动作，缺乏对物理世界演变的前瞻理解；而视频世界模型（如 Sora、Wan2.1）擅长视频生成，却不懂如何输出精确的力控动作；两者的简单串联又存在高延迟与误差累积。

核心问题在于：**如何构建一个统一的生成式骨干，既能吸收互联网无标注视频的通用物理先验，又能精确生成机器人连续双臂动作，并在毫秒级内完成闭环推理？**

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/vln/motus-architecture-overview.webp" width="100%" />
<figcaption>图：Motus 整体架构图：基于 MoT（Mixture-of-Transformers）三专家网络，统一建模多模态理解、视频生成与机器人连续动作控制。（图源：Motus, 2025）</figcaption>
</div>

#### ① 三专家 Mixture-of-Transformers（MoT）架构

Motus 在每个 Transformer Block 内部引入了分流设计：
- **Token 组织**：输入序列由视觉潜 Token $$z_v$$（来自 Wan-VAE）、文本 Token $$z_t$$（来自 T5）与连续动作 Token $$z_a$$（来自 Latent Action VAE）拼接而成；
- **共享自注意力（Shared Self-Attention）**：视觉、文本与动作三种异构 Token 共享同一组注意力权重，允许动作 Token 自由查询场景未来的物理演化趋势，同时让未来视觉生成受到预定操作意图的约束；
- **解耦专家 FFN（Decoupled FFNs）**：在注意力交互之后，视觉 Token 路由至 Video FFN，文本 Token 路由至 Text FFN，动作 Token 路由至 Action FFN。这种解耦避免了视频生成的大梯度破坏精细动作的数值敏感性。

<div align="center">
  <img src="/images/vln/motus-latent-action-vae.webp" width="75%" />
<figcaption>图：Motus 的 Latent Action VAE：通过自编码光流场自监督提取连续潜动作空间。（图源：Motus, 2025）</figcaption>
</div>

#### ② 光流动作提取与 UniDiffuser 联合去噪调度

为了利用互联网海量无控制标签的视频，Motus 设计了 **Latent Action VAE**：
- 输入相邻两帧的稠密光流场 $$F_{t \to t+1}$$，经编码器压缩为低维连续潜动作向量 $$a_t \in \mathbb{R}^{d_a}$$；
- 在训练阶段，采用类似 UniDiffuser 的联合去噪调度器，对视频潜变量 $$z_v$$ 与动作变量 $$a_t$$ 分配独立的加噪时间步；
- 通过在训练中动态 Mask 掉某些通道，模型天然支持四种运行模式：
  1. **世界模型模式**：给定观测 $$o_t$$ 与动作 $$a_t$$，去噪生成未来帧 $$o_{t+1}$$；
  2. **策略（Policy）模式**：给定观测 $$o_t$$ 与指令，单次前向直接去噪生成执行动作 $$a_t$$；
  3. **逆动力学（IDM）模式**：给定前后两帧 $$o_t, o_{t+1}$$，反推执行动作 $$a_t$$；
  4. **联合仿真模式**：同时去噪动作与未来画面，既给出决策又呈现预想后果。

---

### 3. 核心结果/发现

<div align="center">
  <img src="/images/vln/motus-robotwin-results.webp" width="100%" />
<figcaption>图：Motus 在双臂操作仿真基准 RoboTwin 2.0 上的性能对比，在多种高精度接触任务中大幅领先基线。（图源：Motus, 2025）</figcaption>
</div>

1. **RoboTwin 2.0 刷新记录**：在双臂协调操作基准测试中，Motus 的平均任务成功率达到 **87.8%**，大幅超越 X-VLA（72.8%）和 $$\pi_{0.5}$$（79.8%）；
2. **真机跨本体泛化**：在单臂 Franka 与双臂移动机器人真机上执行 20 余项日常复杂技能（如开箱、倒水、折叠毛巾），平均成功率突破 **85%**；
3. **消除在线规划延迟**：Policy 模式下单步动作生成延迟仅需 **80ms**，实现了真正可部署的高频闭环伺服。

<div align="center">
  <img src="/images/vln/motus-real-world-tasks.webp" width="100%" />
<figcaption>图：Motus 真机多任务操控执行轨迹。（图源：Motus, 2025）</figcaption>
</div>

---

### 4. 局限性

- 隐动作空间从光流中提取，虽然摆脱了硬件标签依赖，但在面对手部严重自遮挡或极速运动导致的光流伪影时，潜动作可能产生短暂失真；
- 模型参数量较大，端侧部署依赖 TensorRT 量化与显存优化。

---

## 5.15 NavWAM & WAM-Nav (2026) {#sec-5-15-navwam}
———导航世界动作模型：潜在画布（Latent Canvas）与非对称时空视界

📄 **NavWAM (2026)**: [arXiv:2606.13494](https://arxiv.org/abs/2606.13494) · [Project Page](https://dachii-azm.github.io/navwam/)  
📄 **WAM-Nav (2026)**: [arXiv:2606.04907](https://arxiv.org/abs/2606.04907) · WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation

### 精华

在长程视觉导航（Visual Navigation）领域，传统的导航世界模型（Navigation World Models, NWM）长期受困于测试时极高昂的在线规划开销（使用 CEM 算法采样数千条候选轨迹，单步耗时甚至高达数分钟）。2026 年最新涌现的 **NavWAM** 与 **WAM-Nav** 彻底颠覆了这一范式，其最核心的技术突破包括：
1. **统一潜在画布（9-Frame Latent Canvas，NavWAM）**：将当前状态、目标图像、当前视觉观测、未来动作 Chunk、未来状态预测、未来视觉前瞻与目标进度价值（Value）全部打包为固定 9 帧的时空潜在画布，通过联合去噪一次性完成动作生成与物理预测，彻底消除了推理时 CEM 搜索开销，控制频率可达 **5Hz**，计算量降低数千倍；
2. **非对称时空视界（Asymmetric Horizon，WAM-Nav）**：深刻揭示了导航与操作的根本差异——导航中机器人的自我中心视角变换剧烈，长程视觉展开极易导致累积误差爆炸！WAM-Nav 创造性地采用**“动作长视界（$$H_{act}=24$$ 步保轨迹平滑）+ 视觉短视界（$$H_{vis}=1$$ 步提供可靠近未来几何锚定）”**的非对称设计；
3. **纯隐空间前瞻与零解码自监督**：视觉预测全部在预训练 VAE 隐空间进行，无需昂贵的逐像素渲染，通过隐空间速度匹配损失惩罚动作与物理几何的不一致；
4. **真实物理轮式/双足机器人验证**：在 Diablo 轮足机器人和宇树 Unitree G1 人形机器人上完成多场景闭环实测，成功率达 **79.2%–85%**。

---

### 1. 研究背景/问题

在视觉目标导航（Image-Goal / Point-Goal Navigation）中，环境通常是高度复杂且局部可观测的。传统的规划式世界模型（NWM）仅充当前向预测器，在执行每一步动作前，必须在内存中通过交叉熵方法（CEM）闭环展开上千条视觉轨迹并逐一评分，导致**单步推理延迟高达数十秒至数百秒**（FLOPs 超过 14,000 TF），根本无法用于移动机器人避障。

核心问题在于：**能否在单个生成式网络中，将未来视觉预测、价值评估与连续动作决策深度熔合，实现既有物理前瞻视野又具备实时高频闭环的统一导航模型？**

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/vln/NavWAM-concept-comparison.webp" width="100%" />
<figcaption>图：传统规划式导航世界模型（NWM，左）与导航世界动作模型（NavWAM，右）的决策流对比：NavWAM 彻底剔除繁重的在线 CEM 采样优化，实现高频闭环。（图源：NavWAM, 2026）</figcaption>
</div>

#### ① NavWAM 的 9 帧世界-动作潜在画布（Latent Canvas）

NavWAM 基于预训练 Cosmos-Predict2（2B）底座，构建了一个统一的 9 帧时空潜在画布：

<div align="center">
  <img src="/images/vln/NavWAM-architecture-overview.webp" width="100%" />
<figcaption>图：NavWAM 的 9 帧潜在画布布局：将状态、目标、当前观测、动作 Chunk、未来状态、未来图像及进度价值统一排布并联合去噪。（图源：NavWAM, 2026）</figcaption>
</div>

- **画布排布**：
  - *帧 0–3（已观测条件）*：时空 VAE 边界 Pad、标准化机器人位姿 $$s_t$$、目标图像 $$g$$ 与第一人称当前观测 $$o_t$$；
  - *帧 4–8（待预测输出）*：可执行动作 Chunk $$a_{t:t+H-1}$$（$$H=4$$ 局部航向增量）、未来状态 $$s_{t+H}$$、未来连续观测预测 $$o_{t+H-1}, o_{t+H}$$，以及反映局部到终点进度的归一化距离价值 $$v_{t+H} \in [0, 1]$$。
- **空间广播与平均**：标量/向量特征（动作、状态、价值）经归一化后广播填充为整张特征图，解码时通过空间平均池化恢复，完美复用标准视频 DiT 架构；
- **Policy 模式单步直出**：推理时输入帧 0–3，单次去噪前向即可同时输出高精度的未来动作 Chunk 与预期视觉画面，推理延迟仅 **205.7 ms**（5Hz 控制），较 NWM 的 233.8 秒快了 **1100 倍**！

#### ② WAM-Nav 的非对称视界与双流上下文融合（DSCC）

针对剧烈旋转下的长程漂移，WAM-Nav 提出了两大核心创新：

<div align="center">
  <img src="/images/vln/WAM-Nav-architecture.webp" width="100%" />
<figcaption>图：WAM-Nav 架构：统一目标对齐解耦为视觉查询 gV 与几何查询 gG，DSCC 双流融合历史观测与运动动量，共享 DiT 联合去噪动作与近未来隐特征。（图源：WAM-Nav, 2026）</figcaption>
</div>

1. **非对称视界设计（Asymmetric Horizon）**：
   - 动作时域设定为长程（$$H_{act}=24$$ 步），保证机器人运动学轨迹平滑且具有足够前瞻；
   - 视觉时域设定为极短程（$$H_{vis}=1$$ 步），在 Stable Diffusion VAE 的潜空间中预测近未来特征 $$z_{t+1}$$，为动作去噪提供立竿见影的近场几何障碍碰撞约束，同时彻底避免了长程自回归视频生成带来的发散伪影；
2. **双流上下文条件（DSCC）**：
   - 目标调制视觉流：用视觉查询 $$g_V$$ 残差强化 DINOv2 提取的视觉空间记忆；
   - 相对运动历史流：将历史轨迹转换为坐标无关的相对位移 $$(\Delta x_i, \Delta y_i, \Delta \theta_i)$$，保证运动动量平滑。

---

### 3. 核心结果/发现

1. **离线基准与未来视觉一致性**：在 GO STANFORD 测试集上，NavWAM 在无需 CEM 动作搜索的前提下，轨迹误差 ATE 仅为 0.192，显著优于传统 NWM（0.453），未来视觉预测一致性（0.668）大幅领先；
2. **实机机器人部署突破**：
   - **NavWAM**：在 Diablo 轮足机器人室内多场景（办公室、仓库、会议室、大厅）24 次盲测中，取得 **79.2%** 的高成功率，远超 OmniVLA（58.3%）与 NWM（16.7%）；
   - **WAM-Nav**：在 Unitree G1 人形机器人真机实测中取得 **85%** 的平均导航成功率，展示出极强的 Sim2Real 零样本泛化能力。

<div align="center">
  <img src="/images/vln/NavWAM-real-world-rollouts.webp" width="100%" />
<figcaption>图：Diablo 机器人实机运行期间的实测相机画面与预测未来画面对比。（图源：NavWAM, 2026）</figcaption>
</div>

---

### 4. 局限性

- 当前主要聚焦于静态/准静态室内环境，对于穿梭行人等动态障碍物的复杂物理交互仍需引入动态流场建模；
- 目标输入主要针对图像目标导航（Image-Goal），未来需与全模态语言指令进一步深度融合。

---

## 5.16 Image2Sim (2026) {#sec-5-16-image2sim}
———解耦 3D 空间锚定与单步像素流的实时神经仿真引擎

📄 **Paper**: [arXiv:2607.05765](https://arxiv.org/abs/2607.05765) · [Project Page](https://github.com/MrZihan/Image2Sim) · 清华大学 & 智源研究院

### 精华

构建大规模、高保真且具备物理接地的交互式仿真环境是世界模型赋能具身智能的核心使命。Image2Sim 提出了“3D 空间锚定”与“超真实图像合成”解耦的神经仿真新范式：
1. **打破几何与合成的博弈（Decoupled Geometry & Generation）**：利用前馈 3D 特征高斯（Feature Gaussian）提供显式度量几何约束，再由单步像素流（Pixel Flow）生成模型在 3D 几何 Alpha 掩码引导下补全未观测视野，彻底克服了自回归生成式世界模型的空间遗忘与几何崩溃；
2. **45.6 FPS 极速闭环仿真**：采用连续时间 MeanFlow 单步速度估计与动量自蒸馏，将传统扩散/流匹配的多步迭代采样压缩为单步前向映射，在全景 RGB-D 渲染上达到 **45.6 FPS**，首次满足具身智能在线闭环交互与大规模强化学习/DAgger 训练的实时性要求；
3. **自动化具身数据飞轮**：直接从无标注 RGB-D 视频/图像构建近 **2 万个**交互式神经环境，并自动合成了超过 **1000 万条**跨视角高保真导航轨迹与多模态指令；
4. **零样本 Sim2Real 跨域泛化**：基于纯 Image2Sim 神经环境训练的导航策略 Image2Nav，跨模拟器 zero-shot 泛化至真实 Habitat（R2R-CE 成功率 70.3%）与真实 Hello Robot Stretch 3 物理机器人上。

---

### 1. 研究背景/问题

传统具身智能策略训练严重依赖人工手工建模的物理仿真环境（如 Matterport3D、HM3D、ProcTHOR）：
- 真实扫描环境成本极高，环境多样性受限（仅数百个场景）；
- 程序化合成环境存在严重的 Sim-to-Real 视觉与物理保真度鸿沟；
- 传统生成式视频世界模型虽然画面逼真，但缺乏显式持久的 3D 空间结构和度量坐标系，机器人走远后“回看即崩溃”，无法支持长时间自由导航闭环。

Image2Sim 旨在回答：**能否直接从现实世界采集的无约束视频中，秒级构建出兼具毫米级 3D 空间一致性、照片级视觉保真度与高帧率闭环交互的神经物理世界？**

---

### 2. 主要方法/创新点

<div align="center">
  <img src="/images/vln/Image2Sim-pipeline-comparison.webp" width="100%" />
<figcaption>图：传统导航数据流水线（需昂贵人工 3D 重建）与 Image2Sim 神经仿真框架（自动从无约束数据构建 2 万个场景与千万轨迹）对比。（图源：Image2Sim, 2026）</figcaption>
</div>

Image2Sim 将神经环境仿真解耦为两级级联结构：

<div align="center">
  <img src="/images/vln/Image2Sim-architecture.webp" width="100%" />
<figcaption>图：Image2Sim 架构：前馈 3D 特征高斯编码器（左）与几何感知单步 Pixel Flow 渲染器（右）。（图源：Image2Sim, 2026）</figcaption>
</div>

1. **前馈 3D 特征高斯几何构建（Feed-Forward 3D Gaussian Encoder）**：
   - 摒弃传统 3DGS 逐场景优化耗时数小时的弊端，采用双流编码器（DINOv3 高层语义流 + 几何细节流），单次前向直接预测场景的 3D 特征高斯集合 $$\mathcal{G} = \{g_j\}_{j=1}^M$$；
   - 在任意查询视角位姿 $$p$$ 下，通过可微光栅化毫秒级投影出全景几何、深度图与不透明度图 $$\tilde{\mathbf{A}}_p$$，提供不可动摇的 3D 几何锚定；
2. **几何感知单步 Pixel Flow 渲染器（Geometry-Aware One-Step Pixel Flow）**：
   - 当机器人漫游至未被扫描的死角盲区时，3DGS 投影会出现空洞与撕裂。Image2Sim 利用不透明度图 $$\tilde{\mathbf{A}}_p$$ 构造 Alpha 门控源状态：
     
     $$
     \mathbf{z}_{\mathrm{src}} = \tilde{\mathbf{A}}_p \odot \tilde{\mathbf{X}}_p + \Sigma(\tilde{\mathbf{A}}_p) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
     $$

   - 在高置信度几何区域保持原始投影，在盲区由基于 Flow Matching 的生成网络智能生成物理合乎逻辑的纹理细节；
   - 配合动量自蒸馏算法，反向 ODE 积分被蒸馏为单步前向预测，实现 45.6 FPS 极速运行。

---

### 3. 核心结果/发现

1. **渲染速度与保真度双赢**：在 20,000 个场景上，全景 RGB-D 渲染速度达到 **45.6 FPS**，显著快于同类扩散世界模型（通常 < 1 FPS），同时 PSNR 与 LPIPS 均达到一流视觉水准；
2. **纯神经仿真训练的真机策略泛化**：智能体完全在由 Image2Sim 生成的神经环境中进行大规模交互学习，零样本部署至真实世界机器人 Hello Robot Stretch 3，在包含复杂家具布局的未知房间中顺利完成多目标寻物导航，证明了神经世界模拟器替代传统仿真引擎的可行性。

---

### 4. 局限性
- 当前主要处理室内静态刚体场景，对于可形变柔性物体及大范围流体交互的度量高斯建模尚处于探索阶段。

---

# 6. 基础模型生态库 (Foundation Models Zoo)

具身智能世界模型的飞速发展，高度仰赖于底层多模态生成、表征学习与空间几何基础模型的支撑。根据功能定位，可划分为四大基础模型支柱：

## 6.1 图像/视频生成模型（Video Generation Foundations）

作为世界模型的“想象引擎”，负责在自然语言、历史图像或动作条件控制下，高保真生成连续的时空未来视频，参数规模从轻量级 0.6B 到工业级 17B：

| 模型 | 参数规模 | 建模骨干 | 典型应用与具身角色 |
|:---|:---:|:---|:---|
| **Wan2.1** | 1.3B / 14B | DiT + Flow Matching | 开源标杆底座；WristWorld, DreamGen, Motus, AIM |
| **Cosmos-Predict2.5** | 2B / 14B | DiT + Flow Matching | 物理 AI 专用底座；NavWAM, AdaPower, Prophet |
| **SANA-WM** | 2.6B | Hybrid GDN/Softmax | 分钟级 720p 极速生成，单卡低显存交互仿真 |
| **LingBot-World** | 14B+14B MoE | MoE DiT | 分钟级实时交互世界模拟器，支持事件编辑与指令干预 |
| **LTX-Video / LTX-2** | 2B / 17B | DiT + Flow Matching | SANA-WM 两阶段精化器底座，超高帧率视频生成 |
| **HunyuanVideo** | 13B | 双流 DiT | 高视觉保真度与精细动作先验建模 |
| **Stable Video Diffusion** | 1.5B | UNet 扩散模型 | Ctrl-World, MoWM, HMA, VPP 等早期探索 |
| **iVideoGPT / NOVA** | 0.6B | 自回归 Transformer | VLA-RFT, WMPO 等轻量级仿真评估 |

## 6.2 统一理解与生成模型（Unified Multimodal Foundations）

打破感知理解（VLM）与图像/视频生成（Diffusion）的人为割裂，在单一自回归或混合专家网络中同时支持指令理解、物理推理与动作/图像生成：

| 模型 | 参数规模 | 核心架构 | 典型应用与具身角色 |
|:---|:---:|:---|:---|
| **Cosmos 3** | 4B / 16B / 64B | 双塔 MoT (AR + DM) | 全模态统一 Physical AI 骨干；兼任 VLM、WAM、模拟器与标注器 |
| **Motus** | ~3B | MoT + UniDiffuser | 统一双臂操作 WAM，支持策略生成、正向模拟与逆动力学 |
| **Janus-Pro** | 1B / 7B | 解耦视觉编码 AR | 理解与生成解耦编码，多模态物理常识问答与规划 |
| **Chameleon** | 7B | 早期融合全自回归 | WorldVLA, RynnVLA-002 的原生多模态 Tokenizer 底座 |
| **Emu3** | 8.5B | 纯自回归序列预测 | FlowVLA, UniVLA, UD-VLA 端到端 Token 化策略 |
| **Show-o / VILA-U** | 1.3B / 7B | 统一 Transformer | UP-VLA, CoT-VLA 视觉思维链与前瞻推理 |

## 6.3 具身表示学习模型（Embodied Representation Foundations）

将连续的高维感觉输入抽象编码为紧凑、具备动力学不变性与物理因果性的潜空间表征，而非直接生成易受高频噪声干扰的像素：

| 模型 | 参数规模 | 预训练目标 | 典型应用与具身角色 |
|:---|:---:|:---|:---|
| **V-JEPA 2** | 1B | 联合嵌入预测 (JEPA) | NORA-1.5, MoWM, SRPO 隐式潜空间规划与稠密奖励提取 |
| **DINOv2 / DINOv3** | 300M / 1B | 自监督视觉特征 | WAM-Nav, Image2Sim 空间几何与物体语义记忆检索 |
| **SigLIP / SigLIP-2** | 400M / 1B | Sigmoid 对比学习 | Janus-Pro, OpenVLA 多模态高层指令对齐与场景语义解析 |

## 6.4 空间几何与 3D 基础模型（3D Spatial Foundations）

为世界模型提供不可动摇的三维度量坐标系、深度几何与空间持久性约束，是实现“可探索空间智能（Spatial Intelligence）”的基石：

| 模型 / 技术 | 空间表示 | 核心能力 | 典型应用与具身角色 |
|:---|:---|:---|:---|
| **3D Gaussian Splatting (3DGS)** | 显式高斯粒子 | 毫秒级可微渲染、跨设备漫游 | Lyra 2.0, Marble, Image2Sim 场景持久化资产与碰撞检测 |
| **Depth Anything 2 / 3** | 单目深度 / 点云 | 极高精度的度量几何估计 | Cosmos-Transfer1, SANA-WM 几何条件图与相机姿态恢复 |
| **VGGT / MapAnything** | 3D 几何拓扑 | 大范围度量地图与 3D 场景重建 | 长程具身导航地图构建与物理边界约束 |

---

# 7. 评测基准与指标体系

具身智能世界模型的评估已从单纯的“看图说话”（像素视频质量）全面进阶为“知行合一”（物理规律符合性、空间一致性与下游机器人任务闭环控制性能）。

## 7.1 评测基准概览

评测环境分为**仿真交互基准**与**真实世界多任务数据集**两大类：

### 仿真交互基准（Simulation Benchmarks）

| 基准 | 场景类型 | 任务特点 | 机器人本体 | 轨迹数 | 任务数 | 适用评估范式 |
|:---|:---|:---:|:---|---:|---:|:---|
| **RoboTwin 2.0** | 桌面/台面 | 双臂协调、接触密集、空间价值热图 | 双臂 Franka / 移动底座 | 30k+ | 50 | 世界动作模型 (WAM)、空间意图评估 |
| **LIBERO** | 桌面 | 空间、目标、长程多任务知识迁移 | Franka Panda | 6.5k | 130 | 策略规划器、自回归 WAM |
| **CALVIN** | 桌面 | 连续 5 步子任务链、开环/闭环测试 | Franka Panda | 24k | 34 | 长程前瞻与思维链推理 |
| **WorldArena 2.0** | 多场景 | 物理常识与牛顿定律符合性 | 多种实体 | — | 100+ | 物理一致性与因果逻辑审计 |
| **RoboCasa** | 厨房/室内 | 大规模日常复杂家务、移动操控 | Franka（移动） | 100k+ | 100 | 长航程任务分解与策略泛化 |
| **SimplerEnv** | 真实渲染 | 逼真 Sim2Real 评估环境 | Google Robot, WidowX | — | 8 | 真机部署策略前置验证 |

### 真实世界数据集与竞技场（Real-World Datasets & Arenas）

| 数据集 / 竞技场 | 场景与形态 | 长航程 | 规模 | 适用评估范式 |
|:---|:---|:---:|---:|:---|
| **RoboArena / RoboLab** | 真实机械臂多任务盲测竞技场 | ✓ | 持续评测 | 真机闭环策略横向对比（Cosmos 3 等） |
| **DROID** | 室内多元真实场景（双臂/单臂） | ✓ | 76k 轨迹 | 策略预训练与真机微调评估 |
| **Open X-Embodiment (OXE)** | 跨 22 种机器人形态混合数据 | ✓ | 1M+ 轨迹 | 通用具身预训练表征评测 |
| **RT-1 / BridgeData V2** | 厨房、桌面真实操作轨迹 | ✓ | 130k / 60k | 基础动作模仿与泛化测试 |

---

## 7.2 核心基准性能综合对比

### 1. 双臂精细操作基准：RoboTwin 2.0 评测（成功率 SR %，越高越好）

RoboTwin 2.0 是当前评估世界动作模型（WAM）双臂协调与物理接触精度的权威基准：

| 模型 / 方法 | 核心技术架构 | Easy 难度 SR | Hard 难度 SR | **平均成功率 Avg. SR ↑** |
|:---|:---|:---:|:---:|:---:|
| $\pi_0$ | 反应式流匹配策略 | 64.5 | 59.8 | 62.2 |
| X-VLA | 跨模态动作大模型 | 75.2 | 70.4 | 72.8 |
| $\pi_{0.5}$ | 增强型 VLA 策略 | 81.3 | 78.2 | 79.8 |
| GigaWorld-0 | 世界合成器数据增强 | 88.0 | 84.0 | 86.0 |
| Motus | 混合专家 WAM + 光流潜动作 | 89.2 | 86.4 | 87.8 |
| Fast-WAM | 极速流匹配 WAM | 93.0 | 90.6 | 91.8 |
| LingBot-VA | 交互式视听动作模型 | 93.5 | 90.9 | 92.2 |
| **AIM (Stage 1 SFT)** | 空间价值图 (ASVM) + 意图因果掩码 | 93.0 | 92.0 | 92.5 |
| **AIM (Stage 2 RL)** | **价值自蒸馏强化学习后训练** | **94.0** | **92.1** | **93.1** |

### 2. 经典操作基准：LIBERO（成功率 %，越高越好）

| 方法 | 范式类型 | Spatial | Object | Goal | Long | **Avg. ↑** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| World-Env | 世界模拟器 (RL) | 87.6 | 86.6 | 86.4 | 57.8 | 79.6 |
| VLA-Reasoner | 规划器 (TTA) | 91.2 | 90.6 | 82.4 | 59.8 | 81.0 |
| WorldVLA | 自回归 WAM (因果掩码) | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| CoT-VLA | 自回归 WAM (思维链) | 87.5 | 91.6 | 87.6 | 69.0 | 83.9 |
| TriVLA | 规划器 (隐式潜引导) | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |
| FlowVLA | 自回归 WAM (流感知) | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| VLA-RFT | 世界模拟器 (稠密奖励 RL) | 94.4 | 94.4 | 95.4 | 80.2 | 91.1 |
| DreamVLA | 自回归 WAM (世界梦境) | 97.5 | 94.0 | 89.5 | 85.2 | 91.6 |
| UD-VLA | 扩散 WAM (离散扩散) | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| UniVLA | 自回归 WAM (潜动作) | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| dVLA | 扩散 WAM | 97.4 | 97.9 | 98.2 | 92.2 | 96.4 |
| RynnVLA-002 | 统一序列 WAM | **99.0** | 99.8 | 96.4 | 94.4 | 97.4 |
| **SRPO（在线）** | **世界模拟器 (脚手架 RL)** | 98.8 | **100.0** | **99.4** | **98.6** | **99.2** |

### 3. 长程序列操作基准：CALVIN ABC→D（连续任务成功率 %，Avg. Len. 越高越好）

| 方法 | 范式类型 | 任务 1 | 任务 2 | 任务 3 | 任务 4 | 任务 5 | **Avg. Len. ↑** |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GR-1 | 早期自回归 WAM | 85.4 | 71.2 | 59.6 | 49.7 | 40.1 | 3.06 |
| GR-MG | 规划器 (显式像素) | 96.8 | 89.3 | 81.5 | 72.7 | 64.4 | 4.04 |
| MoWM | 混合规划器 | 94.3 | 87.3 | 81.2 | 75.0 | 67.5 | 4.05 |
| UP-VLA | 自回归 WAM | 92.8 | 86.5 | 81.5 | 76.9 | 69.9 | 4.08 |
| Seer | 预测逆动力学 | 96.3 | 91.6 | 86.1 | 80.3 | 74.0 | 4.28 |
| VPP | 规划器 (隐式潜表示) | 95.7 | 91.2 | 86.3 | 81.0 | 75.0 | 4.29 |
| UniVLA | 统一自回归 WAM | **98.9** | **94.8** | 89.0 | 82.8 | 75.1 | 4.41 |
| TriVLA | 规划器 | 96.8 | 92.4 | 86.8 | 83.2 | **81.8** | 4.41 |
| **DreamVLA** | **世界梦境增强 WAM** | 98.2 | 94.6 | **89.5** | **83.4** | 78.1 | **4.44** |

### 4. 世界模型自身生成质量：各工作报告的关键指标

与前三张下游策略表不同，世界模型"生成侧"目前**没有统一的横向榜单**——各工作的评测集、分辨率、视频时长与指标口径都不一致。下表只汇总各论文自己报告的数字，便于定位来源，**不构成横向可比排名**：

| 模型 / 架构 | 生成质量指标（论文口径） | 相机 / 几何控制精度 | 效率与部署 | 出处 |
|:---|:---|:---|:---|:---:|
| **Cosmos-Predict2.5** (2B/14B) | RL 后训练人类偏好胜率较 RL 前 +约 20 个百分点 | — | rCM 蒸馏至 4 步，PAI-Bench 总分损失 < 0.005；4096×H100 训练 | §4.3 |
| **Cosmos-Transfer2.5** (~2B) | PAIBench-Transfer 整体质量 6.56 → **9.75** | 控制信号遵循度优于 Transfer1-7B | 模型尺寸缩小 3.5×（7B → ~2B） | §4.2 |
| **Cosmos 3** (Nano/Super) | Artificial Analysis T2I / I2V 榜单**开源权重第 1**（T2I 含闭源计第 4） | — | Nano-Policy 15Hz 联合输出动作与未来帧；RoboArena / RoboLab 真机第 1 | §4.6 |
| **SANA-WM** (2.6B) | VBench Overall **80.62 / 81.89**（Simple / Hard 轨迹） | RotErr **4.50° / 8.34°**，CamMC **1.41 / 1.44**（↓ 越低越好） | 单 GPU 74.7 GB；RTX 5090 上 34s 生成 60s 720p | §5.11 |
| **LingBot-World** (14B+14B MoE) | VBench Overall 81.82 / 81.89 | RotErr 10.47° / 18.99° | 8×H100（454.1 GB）；16 fps，亚秒级延迟 | §5.9 / §5.11 |
| **Qwen-RobotWorld** (20B MMDiT) | EWMBench **4.60（第 1）**、WorldModelBench **8.99**、PBench 0.804、DreamGen Bench 4.952 | 物理符合性四项满分（牛顿定律 / 质量守恒 / 流体 / 重力） | 美学与成像质量因分辨率偏低（0.455 / 0.649） | §5.12 |
| **Wan2.1** (1.3B/14B) | 720p·25 帧重建 PSNR 与 HunyuanVideo 相当，重建速度 **2.5×** | — | 1.3B int8 量化后 **8.19 GB**，RTX 4090 可跑 | §5.13 |
| **Image2Sim** | 全景 RGB-D 渲染 **45.6 FPS**（同类扩散世界模型通常 < 1 FPS） | 前馈 3D 特征高斯提供显式度量几何锚定 | 自动构建 2 万个交互式神经环境；R2R-CE 零样本 70.3% | §5.16 |
| **Marble** (World Labs) | 未发表技术论文，无公开量化指标 | 3DGS 显式几何，支持精确相机控制 | 跨设备（手机→VR）实时流式渲染 | §5.10 |

---

## 7.3 多维评估指标体系

现代世界模型评估体系由**视觉保真度**、**物理几何一致性**与**闭环控制表现**三维交织构成：

```mermaid
mindmap
  root((世界模型评估指标体系))
    视觉保真度与生成质量
      PSNR 峰值信噪比 ↑
      SSIM 结构相似性 ↑
      LPIPS 感知特征距离 ↓
      FID / FVD 图像与视频分布距离 ↓
    物理与空间几何一致性
      Physics-IQ 物理常识评分 ↑
      Physics Compliance 牛顿力学/重力遵循度 ↑
      CamMC 相机轨迹运动一致性 ↓
      RotErr 旋转误差与姿态漂移 ↓
      Depth L1 深度几何度量误差 ↓
    下游具身闭环控制
      SR 任务成功率 ↑
      SPL 路径长度加权成功率 ↑
      ATE 绝对轨迹误差 ↓
      ATP 长程平均任务进度 ↑
      Collision Rate 碰撞率 ↓
```

**专项综合基准体系**：

| 综合基准 | 主要评估维度与考察重点 | 代表评测模型 |
|:---|:---|:---|
| **WorldModelBench** | 物理规律遵循度（牛顿定律、质量守恒、流体动力学、重力常识） | Qwen-RobotWorld, Wan2.6, Veo3 |
| **EWMBench** | 复杂具身操作物理仿真、多视角几何一致性与运动真实度 | Genie Envisioner, Qwen-RobotWorld |
| **DreamGen Bench** | 复杂指令遵循（Instruction Following）与跨物体长时程泛化 | DreamGen, GigaWorld-0 |
| **PAI-Bench (PBench)** | 文本到物理世界生成的质量分（Quality）与领域分（Domain） | Cosmos-Predict2.5, GigaWorld-0 |
| **PRBench (进度奖励基准)** | 阶段进度单调性对齐（SC/Mono）与目标判别灵敏度（MMD/JS） | SRPO, NORA-1.5 |
| **TransferBench** | Sim2Real 翻译控制遵循度（Adherence）、生成多样性与视觉质量 | Cosmos-Transfer1 / 2.5 |

---

# 8. 未来六大关键挑战与前沿研究方向

尽管具身智能世界模型在 2025–2026 年取得了突破性进展，要真正实现工业级通用物理智能体，仍面临以下关键挑战：

## 8.1 物理一致性与神经符号物理引擎（Differentiable & Symbolic Physics）

当前生成式世界模型在微观运动上逼真，但在中宏观物理规律（如刚体碰撞冲量守恒、弹性/塑性形变、流体飞溅与不可穿模约束）上仍依赖纯统计拟合，极易产生物理幻觉。
- **前沿方向**：将显式**可微物理方程（Differentiable Physics Simulators）**嵌入去噪扩散过程，以物理残差作为 Loss 正则项；
- **因果推理与反事实推演**：融合神经因果图（Causal Discovery），让世界模型能够严密回答“如果机械臂多施加 2N 的侧向力，杯子是否会倾倒”等假设性问题。

## 8.2 时空 4D 感知与大型世界模型（Large World Models, LWM）

传统 2D 像素流无法持久保留 3D 空间结构，智能体在大范围移动漫游时容易发生“空间遗忘”与几何失真。
- **前沿方向**：以 **3D 高斯泼溅（3DGS）**、**持久点跟踪（Persistent Point Tracking）**与**连续神经占据场（Occupancy Fields）**为原生状态表示；
- **工业范式跃迁**：李飞飞团队创立的 World Labs（Marble 模型）与 Image2Sim 展示了从“2D 视频预测”走向“可无限探索、可持久交互的 3D 物理宇宙”的必然趋势，使世界模型成为具备度量几何的完整数字孪生空间。

## 8.3 闭环安全性、不确定性量化与幻觉审计（Safety & Conformal Uncertainty）

世界模型在用于高危真机控制前，必须具备对自身预测能力的自知之明（Self-Awareness）。
- **前沿方向**：引入**共形预测（Conformal Prediction）**与**星形不确定性量化（Epistemic Uncertainty Quantification）**，当环境处于未见过的极端 OOD 分布或碰撞风险过高时，主动发出安全警报并切换至人类遥操作；
- **自动化物理合理性审计**：利用专用物理推理 VLM（如 Cosmos-Reason1）充当实时“物理裁判”，在生成帧下发策略前完成几何干涉与力学合理性校验。

## 8.4 长时程推演与常数内存架构（Long-Horizon Foresight & Linear Attention）

标准 Softmax 自注意力的显存与计算开销随时间步呈二次方爆炸，使得超长时程（数分钟）的前瞻推演难以在端侧部署。
- **前沿方向**：发展**混合线性-门控注意力架构（如 SANA-WM 的 Gated DeltaNet）**与 **Attention Sink 机制**，使 KV 缓存内存保持常数级别；
- **分层动力学抽象（Hierarchical Dynamics）**：在高层以低频跳步预测任务里程碑（Subgoals），在底层以高频展开精细力控轨迹，实现分钟级任务的稳定推演。

## 8.5 失效感知动力学与真实世界泛化（Failure-Aware Dynamics & Sim2Real）

当前大部分机器人数据集仅包含人类专家成功的正向演示，导致模型对“失败状态”产生认知盲区。
- **前沿方向**：**主动失败模式生成**——利用世界模型定向合成打翻、滑脱、卡死等次优与失败轨迹，训练具备自我纠偏与错误恢复能力（Self-Correction）的鲁棒策略；
- **自适应 Sim2Real 域桥接**：利用结构化世界翻译器（如 Cosmos-Transfer）将低成本仿真渲染无缝升格为照片级真实视界，彻底抹平仿真与现实的感知差距。

## 8.6 统一基础模型的全模态架构收敛（All-in-One Foundation Convergence）

理解、生成、预测与控制的割裂拼装正在走向历史终点。
- **前沿方向**：以 **Mixture-of-Transformers（MoT）**、**全模态流匹配（Omnimodal Flow Matching）**为核心骨干（如 Cosmos 3、Motus），将自然语言、第一/第三人称视觉、本体感觉、高频触觉、度量几何与电机扭矩融为统一的 Token 流，实现“感知即理解，生成即规划，推演即控制”的完全统一通用智能体。

---

# 9. 总结与实践启示

## 9.1 核心全景回顾

具身智能世界模型在 2018 至 2026 年间，完成了从**潜空间理论奠基**到**全模态基础模型生态**的宏伟技术跨越：

```mermaid
graph TD
    subgraph S1["奠基阶段 (2018–2024)"]
        WM0["World Models 2018\n(V-M-C 架构, 潜空间梦境)"] --> D3["DreamerV3 Nature 2025\n(离散 RSSM, 通用跨领域)"]
        D3 --> TD2["TD-MPC2 ICLR 2024\n(无重构隐式 MPC, 104 任务)"]
    end

    subgraph S2["四大范式探索与爆发 (2023–2025)"]
        P1["① 世界规划器\n(UniPi / SuSIE / GENE-26.5)"]
        P2["② 世界动作模型 WAM\n(GR-1 / WorldVLA / Motus)"]
        P3["③ 世界合成器\n(Genie / DreamGen / Image2Sim)"]
        P4["④ 世界模拟器\n(VLA-RFT / WoVR / SRPO)"]
    end

    subgraph S3["全模态收敛与空间智能 (2025–2026)"]
        C3["Cosmos 3 / Motus\n(MoT 双塔全模态统一骨干)"]
        SANA["SANA-WM / LingBot-World\n(分钟级高效交互世界生成)"]
        L3D["Lyra 2.0 / Marble (World Labs)\n(3DGS 可探索持久物理宇宙)"]
    end

    S1 --> S2
    S2 --> S3
```

## 9.2 给具身 AI 实践工作者的五大黄金法则

对于致力于具身智能、VLA 与机器人开发的科研与工程团队，本综述提炼出以下五项核心准则：

1. **“数据质量与策展”重于“盲目扩大参数”**：Cosmos Video Curator 与 EWK 数据集的经验表明，严格的镜头切分、多级物理过滤、多视角空间与时间分层描述对物理世界模型的贡献，远大于单纯堆叠 DiT 参数；
2. **优先拥抱世界动作模型（WAM）以消除在线搜索开销**：在对实时性要求严苛的闭环控制中，应优先采用 WAM 或 Latent Canvas 架构，将未来视觉预测作为自监督锚定，实现单步前向 5Hz–15Hz 高频输出，规避昂贵的 CEM 采样；
3. **重视隐空间规划与非对称时空视界**：在视角剧烈变化的移动导航或复杂操作中，切忌无节制拉长自回归像素生成；采用“长动作视界 + 短隐空间视觉前瞻”能以极低成本提供最可靠的几何约束；
4. **警惕强化学习模拟器中的“分布漂移与幻觉漏洞”**：在虚拟世界模型中做 RL 后训练时，必须配合关键帧初始化（KIR）、策略协同演化（PACE）以及基于物理推理 VLM 的可验证稠密奖励，防止策略过拟合于生成器的物理 bug；
5. **布局 3D 显式表示与空间智能基础**：2D 像素是 3D 物理世界的降维投影，长期来看，深度融合 3DGS、点云及度量几何的 Large World Models（LWM）将成为破解空间泛化鸿沟的终极路径。


## 9.3 从想象到验证再到规划：技术路线演进与综合研判 {#sec-9-3-future-roadmap}

具身智能世界模型的发展历程，折射出 AI 对物理世界认知能力的根本跃迁。纵观 2018 至 2026 年的技术演化，我们可以清晰地梳理出三条交织演进的技术主线：

```mermaid
flowchart TD
    subgraph Track1["主线一：生成表征维度"]
        T1_1["1D/2D 潜状态<br/>(World Models, Dreamer)"] --> T1_2["2D 像素视频<br/>(Genie, Wan2.1, Sora)"]
        T1_2 --> T1_3["3D/4D 显式时空<br/>(Marble, 3DGS, LWM)"]
    end

    subgraph Track2["主线二：控制与决策机制"]
        T2_1["开环像素规划<br/>(UniPi, SuSIE, CEM)"] --> T2_2["统一世界动作模型<br/>(WorldVLA, Motus, AIM)"]
        T2_2 --> T2_3["双系统想象闭环<br/>(System 1 反应 + System 2 前瞻)"]
    end

    subgraph Track3["主线三：训练与可信仿真"]
        T3_1["纯监督行为克隆 SFT"] --> T3_2["世界模拟器强化学习<br/>(WoVR, VLA-RFT, SRPO)"]
        T3_2 --> T3_3["自监督价值蒸馏与对齐<br/>(AIM, GRPO, 神经符号物理)"]
    end

    Track1 --> Convergence["全模态通用物理大模型<br/>(Cosmos 3, Physical AGI)"]
    Track2 --> Convergence
    Track3 --> Convergence

    classDef stage fill:#f0f4f8,stroke:#2b6cb0,stroke-width:1px;
    classDef target fill:#e6fffa,stroke:#319795,stroke-width:2px;
    class T1_1,T1_2,T1_3,T2_1,T2_2,T2_3,T3_1,T3_2,T3_3 stage;
    class Convergence target;
```

### 1. 演进规律：四大范式的深度融合与优势互补

回顾 **§3 具身世界模型的四大主流范式** 与 **§5 十六项前沿工作**，四大范式并非互相替代的竞争关系，而是正加速走向深层互补：

| 范式定位 | 典型代表 | 核心优势 | 核心瓶颈 | 未来融合方向 |
|:---|:---|:---|:---|:---|
| **① 世界规划器** (Planner) | UniPi, SuSIE, GENE-26.5 | 目标驱动、可解释性强、灵活泛化 | 采样延迟高 (CEM 达秒级)、不适合高频控制 | 隐空间极速梯度规划、扩散蒸馏采样 |
| **② 世界动作模型** (WAM) | WorldVLA, Motus, AIM, NavWAM | 5–15Hz 高频闭环、无缝融合前瞻与控制 | 多视角时空对齐难、长程轨迹漂移 | MoT 解耦架构、空间价值图 (ASVM) 中介 |
| **③ 世界合成器** (Synthesizer) | Genie, DreamGen, Image2Sim | 无限扩充边缘工况长尾数据、泛化性强 | 仿真与真实域差 (Sim2Real)、物理幻觉 | 4D 动态流生成、光流差分动作迁移 |
| **④ 世界模拟器** (Simulator) | WoVR, VLA-RFT, SRPO | 免真机损耗的低成本强化学习母体 | 策略过拟合于生成器幻觉漏洞 | 关键帧回放 (KIR)、策略协同演化 (PACE) |

### 2. 核心研判：破解物理世界交互的三大突破口

1. **从“像素直推动作”到“空间意图显式化（Spatial Intent as Bridge）”**：AIM 与 NavWAM 的成功表明，像素与动作之间存在天然的物理鸿沟。引入显式的 3D 几何、点云接触面或 2D 空间价值图作为结构化中间层，是消除反向动力学（Inverse Dynamics）学习困难的关键抓手；
2. **从“2D 纯视频梦境”到“3DGS 空间智能宇宙（3D Spatial Grounding）”**：以 Marble 和 Lyra 2.0 为代表的 3D 显式世界模型彻底解决了 2D 视频生成中的视角不一致与物体恒久性丧失问题，使世界模型兼具“可微分生成”与“物理引擎级几何持久性”；
3. **从“单向开环生成”到“慢思考与快反应双系统（System 1 & System 2 Co-Design）”**：高频电机控制（100Hz–1kHz）由轻量化策略或 WAM 动作头负责（System 1），而宏观场景推演、危险审计与长程任务拆解由大型世界模型在后台异步运行（System 2），构成了当前最稳健的具身智能工程落地范式。

---

# 10. 参考资料与延伸阅读 {#sec-10-references}

1. Tan, Z., et al. (2026). *Towards Generalist Embodied AI: A Survey on World Models for VLA Agents*. TechRxiv. [arXiv/TechRxiv 链接](https://www.techrxiv.org/)
2. Li, X., et al. (2025/2026). *A Comprehensive Survey on World Models for Embodied AI*. [arXiv:2510.16732](https://arxiv.org/abs/2510.16732) · [AwesomeWorldModels](https://github.com/Li-Zn-H/AwesomeWorldModels)
3. Ha, D., & Schmidhuber, J. (2018). *World Models*. NeurIPS 2018. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122) · [Project Page](https://worldmodels.github.io/)
4. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). *Mastering Diverse Control Tasks through World Models (DreamerV3)*. Nature 640, 647–653. [Nature 论文](https://www.nature.com/articles/s41586-025-08744-2) · [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
5. Hansen, N., et al. (2024). *TD-MPC2: Scalable, Robust World Models for Continuous Control*. ICLR 2024 (Oral). [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) · [Project Page](https://tdmpc2.github.io/)
6. NVIDIA. (2025). *Cosmos World Foundation Model Platform for Physical AI*. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
7. NVIDIA. (2025). *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control*. [arXiv:2503.14492](https://arxiv.org/abs/2503.14492)
8. NVIDIA. (2025). *Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning*. [arXiv:2503.15558](https://arxiv.org/abs/2503.15558)
9. NVIDIA. (2026). *Cosmos 3: Omnimodal World Models for Physical AI*. [arXiv:2606.02800](https://arxiv.org/abs/2606.02800) · [GitHub](https://github.com/nvidia-cosmos)
10. NVIDIA. (2026). *Lyra 2.0: Explorable Generative 3D Worlds at Scale*. [arXiv:2604.13036](https://arxiv.org/abs/2604.13036)
11. Bruce, J., et al. (2024). *Genie: Generative Interactive Environments*. Google DeepMind. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
12. *Motus: A Unified Latent Action World Model for Robotic Manipulation* (2025/2026). 清华大学 & 生数科技. [arXiv:2512.18876](https://arxiv.org/abs/2512.18876) · [Code](https://github.com/PKU-YuanGroup/Motus)
13. *AIM: Intent-Aware Unified World Action Modeling with Spatial Value Maps* (2026). [arXiv:2604.11135](https://arxiv.org/abs/2604.11135)
14. *NavWAM: Navigation World Action Models for Autonomous Embodied Agents* (2026). [arXiv:2606.13494](https://arxiv.org/abs/2606.13494) · [Project Page](https://dachii-azm.github.io/navwam/)
15. *WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation* (2026). [arXiv:2606.04907](https://arxiv.org/abs/2606.04907)
16. *Image2Sim: Decoupled 3D Gaussian Geometry and One-Step Pixel Flow for Real-Time Neural Simulation* (2026). [arXiv:2607.05765](https://arxiv.org/abs/2607.05765)
17. Zhu, H., et al. (2026). *SANA-WM: Efficient Minute-Scale World Modeling with Hybrid Linear Diffusion Transformer*. [arXiv:2605.15178](https://arxiv.org/abs/2605.15178) · [Project Page](https://nvlabs.github.io/Sana/WM/)
18. *LingBot-World: Open-Source Minute-Scale Interactive World Model* (2026). [arXiv:2601.20540](https://arxiv.org/abs/2601.20540)
19. Alibaba. (2026). *Qwen-RobotWorld: Unifying Embodied World Models via Natural Language*. [arXiv:2606.17030](https://arxiv.org/abs/2606.17030)
20. Alibaba. (2025). *Wan: Open and Advanced Large-Scale Video Generative Models*. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
21. *VLA-World: Learning Vision-Language-Action World Models for Autonomous Driving* (2026). [项目主页](https://vlaworld.github.io)
22. Cen, J., et al. (2025). *WorldVLA: Towards Autoregressive Action World Model*. [arXiv:2506.21539](https://arxiv.org/abs/2506.21539)
23. *WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL* (2026). [arXiv:2602.13977](https://arxiv.org/abs/2602.13977)
24. Chen, X., et al. (2025). *Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling*. [arXiv:2501.17811](https://arxiv.org/abs/2501.17811)
25. World Labs. (2025/2026). *Marble: A Multimodal Large World Model*. [worldlabs.ai](https://www.worldlabs.ai/blog/marble-world-model)
26. *Video Generation Models in Robotics: Applications, Research Challenges, Future Directions* (2026). [arXiv:2601.07823](https://arxiv.org/abs/2601.07823)
27. Du, Y., et al. (2023). *Learning Universal Policies via Text-Guided Video Generation (UniPi)*. NeurIPS 2023.
28. Black, K., et al. (2024). *Zero-Shot Robotic Manipulation with Pre-trained Image-Editing Diffusion Models (SuSIE)*. ICLR 2024.
29. Wu, H., et al. (2023). *Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation (GR-1)*. ICLR 2024.
30. Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. Meta AI.
31. Zhen, H., et al. (2024). *3D-VLA: A 3D Vision-Language-Action Generative World Model*. ICML 2024.
32. Gao, C., et al. (2024). *PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation*.
33. Liu, J., et al. (2025). *CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models*.
34. Zhang, W., et al. (2025). *DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge*.
35. Zhang, Z., et al. (2025). *FlowVLA: Thinking in Flow for Vision-Language-Action Models*.
36. Bu, Q., et al. (2025). *UniVLA: Learning to Act Anywhere with Task-centric Latent Actions*.
37. Zhao, J., et al. (2025). *DreamGen: Unlocking Generalization in Robot Learning through Neural Trajectories*. NVIDIA.
38. *GigaWorld-0: World Models as Data Engine to Empower VLA Models* (2025).
39. *WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation* (2025).
40. *Ctrl-World: A Controllable Generative Framework for Robotic Manipulation* (2025).
41. *VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators* (2025).
42. *WMPO: World Model-based Policy Optimization for Vision-Language-Action Models* (2025).
43. *SRPO: Scaffolded Reinforcement Policy Optimization for Robotic Manipulation* (2025).
44. *NORA-1.5: A Small Open Vision-Language-Action Model for Embodied Tasks with Flow-Matching Action Expert* (2025).
45. *VLA-Reasoner: Empowering Vision-Language-Action Models for Complex Tasks with Future Imagination* (2025).
46. *AdaPower: Adaptive Test-Time Scaling for Vision-Language-Action Models* (2025).
47. Liu, B., et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. [libero-project.github.io](https://libero-project.github.io/)
48. Mees, O., et al. (2022). *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*. [github.com/mees/calvin](https://github.com/mees/calvin)
49. *RoboTwin 2.0: Dual-Arm Benchmark for Scalable Embodied Manipulation* (2025). Tsinghua University.
50. Kerbl, B., et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023.
51. Huang, Z., et al. (2024). *VBench: Comprehensive Benchmark Suite for Video Generative Models*.
52. Genesis AI. (2026). *GENE-26.5: Advancing Robotic Manipulation to Human-Level*. [genesis.ai](https://www.genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level)
53. NVIDIA Cosmos Cookbook — [github.com/nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook).

---

本文所有示意图、架构图、实验对比图均来自上述公开论文或对应官方项目主页，版权归原作者所有，仅用于学术交流与学习整理。
