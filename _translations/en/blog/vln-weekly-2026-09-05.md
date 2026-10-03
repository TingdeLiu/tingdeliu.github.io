---
layout: post
lang: en
translation_id: vln-weekly-2026-09-05
permalink: /en/vln-weekly-2026-09-05/
source_path: _posts/weekly-reports/2026-09-05-VLN-Weekly.md
source_url: /vln-weekly-2026-09-05/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-08-26 to 2026-09-03)"
date: 2026-09-05
period_start: 2026-08-26
period_end: 2026-09-03
issue_number: 3
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "All 45 new entries come from arXiv; the WeChat source is absent after its upstream relay failed with the shutdown of Deno Deploy Classic. Only 2 works report R2R-CE: open-source LookStep reports 49.7% Val-Unseen SR, while Revisiting Topological Graphs makes closed-loop RL trainable through a hierarchical MDP but claims SOTA without numbers. With the same Qwen3-VL-8B backbone on GOAT-Bench, CGFM-Nav raises SR from 53.2% to 63.0% and SPL from 30.0% to 39.6%, providing this issue's strongest navigation evidence. Verification is becoming a shared component across tasks; 13 of the 45 works primarily use LIBERO."
---
## 1. Key conclusions
{: id="一本期结论"}

- The training paradigm of **continuous environment VLN shows signs of converging towards reinforcement learning.** Only 2 of the 45 articles in this issue report on R2R-CE. Among them, *Revisiting Topological Graphs* reconstructs VLN-CE into a hierarchical MDP, using the frontier node of the topological graph as the macro action space and the training-free low-level controller as the state transition, thereby compressing the decision horizon and making closed-loop RL solvable. The paper states that two old problems in imitation learning under closed-loop (distribution shift of behavioral clones, expert action ambiguity after DAgger deviates from the trajectory) are the motivations of RL. This digest's assessment: This route puts the existing conclusion that "RL sample efficiency is too low on VLN-CE" back to the table, and deserves priority tracking.
- **Another parallel route is to reduce costs rather than increase the upper limit.** *LookStep* achieved an R2R-CE Val-Unseen success rate of 49.7% with less data and smaller memory overhead under the same training settings by abandoning cognitive maps/historical frame stacking/external 3D tools, instead using language tags to generate coarse-grained navigation progress and future states, and deciding independently whether to write observations into bounded rolling memory. This digest's assessment: This is in contrast to the previous one - while raising the upper limit, lowering the deployment cost, and the evaluation protocol of both is R2R-CE, which can be directly compared horizontally.
- **universal navigation model begins to cover multi-tasks and multi-embodiments with a single VLM backbone.** *LightNav-0* replaces the task-specific prediction head with a unified token interface (dual-channel pointing to express spatial intention independent of task/scenario/embodiment + residual vector quantization action tokenizer mapped to specific embodiment trajectories). The paper claims to have achieved a monocular success rate of state-of-the-art on all 10 public navigation simulation settings, and generalized it to zero-shot cross-embodiment on real robots. This digest's assessment: The corpus size of this work (2K+ scenes, 4K+ hours) constitutes the main reproduction threshold.
- **The debate on memory representation has shifted from "capacity" to "update discipline".** *CGFM-Nav* uses multi-modal scene graph + target conditional semantic frontier field to increase the success rate from 53.2% to 63.0% and SPL from 30.0% to 39.6% on GOAT-Bench with the same Qwen3-VL-8B backbone; *AGM* on the robotic arm side It clearly advocates that reliable embodied memory relies more on disciplined state updates than memory capacity, and only advances the progress pointer after the sub-goal is verified by physical evidence. This digest's assessment: The two works provide evidence in the same direction from different tasks, which has direct reference value for sub-goal tracking of long-term navigation.
- **The source of WeChat public account in this issue is absent, and all 45 entries are from arXiv.** The transfer service that the original wewe-rss relied on has been permanently invalidated when Deno Deploy Classic went offline on 2026-07-20; it has been migrated to we-mp-rss (using the WeChat public platform interface), but the new authorized account triggered WeChat rate limiting, and the four rounds of collection were all 0. Therefore, this issue lacks an interpretation perspective of WeChat public account, cross-source merger has not occurred, and "45 items" means "45 independent works."

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [Revisiting Topological Graphs](https://arxiv.org/abs/2609.03906v1)** · VLN-CE (Continuous Environment)
  - Contribution: hierarchical MDP + topology graph macro action + graph-based PPO supported by action-aware value head.
  - Evidence: The abstract states that state-of-the-art is achieved on R2R-CE and RxR-CE, **does not provide a verifiable number**.
  - Reason: Making closed-loop RL trainable on VLN-CE is a route-level change.
- **A2 · [LookStep](https://arxiv.org/abs/2609.02350v1)** · VLN-CE (Continuous Environment)
  - Contribution: Language-centric future state modeling + event-driven bounded rolling memory, removing cognitive maps and external 3D tools.
  - Evidence: R2R-CE Val-Unseen has a success rate of 49.7%; the paper states that it is better than existing methods under the same training settings, and has higher memory efficiency and uses less data.
  - Reason: This issue is the only work that provides verifiable R2R-CE numbers, and [has open source code ](https://github.com/kunyang-YU/LookStep)
- **A3 · [LightNav-0](https://arxiv.org/abs/2608.30935v1)** · Universal embodied navigation (command following / open vocabulary ObjectNav / visual tracking)
  - Contribution: Unified token interface: dual-channel pointing + residual VQ action tokenizer, no task-specific header.
  - Evidence: The paper states that the monocular success rates of 10 public navigation simulation settings are all SOTA, and LightNav-ER has the highest complete set average on 8 embodied reasoning benchmarks. The summary of specific values for each setting of **is not listed for**.
  - Reason: A single VLM backbone covers a complete solution for multiple tasks and multiple embodiments.
- **A4 · [CGFM-Nav](https://arxiv.org/abs/2608.29114v1)** · Lifelong multimodal embodied navigation
  - Contribution: Multimodal scene graph (explicit relational memory) + target conditional semantic frontier field (continuous exploration guidance) form a closed loop.
  - Evidence: GOAT-Bench: Same as Qwen3-VL-8B backbone, success rate 53.2% → 63.0%, SPL 30.0% → 39.6% (the paper states that it is a preliminary experiment).
  - Reason: The specific design of coupling memory and exploration, and the numerical caliber is clear.
- **A5 · [VerNav](https://arxiv.org/abs/2609.00920v1)** · **Discrete R2R** (non-CE)
  - Contribution: verifier-first: Batch action verification replaces stepwise autoregressive generation, calling the generator only for uncertain decisions.
  - Evidence: The single-step LLM delay in the decision phase on the R2R benchmark is lower than that of the autoregressive method **10 times or more** , navigation performance is called competitive.
  - Reason: Latency is a hard constraint for the implementation of LLM-based VLN; note that its evaluation is in discrete R2R, **cannot be directly compared with A1/A2’s R2R-CE**.
- **A6 · [CanonNav](https://arxiv.org/abs/2608.30242v1)** · Cross-platform visual navigation (real robot related)
  - Contributions: Camera geometry normalization, decoupling navigation behavior from platform camera geometry; providing safety and local progress supervision with pseudo-labels for offline traversability estimators.
  - Evidence: The paper claims that using only RGB reasoning consistently outperforms the RGB baseline and exceeds the RGB-D method in difficult scenarios. **does not provide the dataset and value**.
  - Reason: The mechanism layer contribution of cross-platform data reuse is useful for multi-robot data aggregation.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. Revisiting Topological Graphs: shortening the VLN-CE decision horizon for RL
{: id="1-revisiting-topological-graphs把-vln-ce-的决策视界压到-rl-能训练的尺度"}

**Problem.** The paper points out two specific bottlenecks in imitation learning under closed-loop VLN-CE: behavioral clones encounter distribution shifts; DAgger expert actions become ambiguous after the agent deviates from the trajectory. However, doing RL directly on the micro-action space has extremely low sample efficiency due to sparse rewards.

**Method.** is refactored into a hierarchical MDP to explicitly decouple high-level planning and low-level control: the environment is abstracted into a topological graph, high-level strategies are decided on the macro-action space composed of frontier nodes, and training-free low-level controllers act as state transfers. In order to evaluate the state value in the dynamic frontier action space, an action-aware value head is proposed to support graph-based PPO.

**Evidence.** The abstract states that it has reached state-of-the-art on R2R-CE and RxR-CE, **but does not give any numerical value, comparison method or sub-indicator**.

**Value.** If the conclusion is true, this is a feasible engineering path to introduce closed-loop RL into VLN-CE; the decomposition method of macro actions + training-free low-level controllers is naturally aligned with the existing navigation stack (topology map + local planner) of ground robots.

**Limitations.** All the evidence relies on the wording SOTA, and the magnitude cannot be verified; the quality of topology construction and the sensitivity of the frontier selection strategy are not stated in the abstract; the training-free nature of the low-level controller means that its upper limit of capabilities directly constrains the overall performance.

**Recommendation.** Prioritize reading the value head design and PPO training details of the text, focusing on checking the specific SR/SPL values and comparison baselines of R2R-CE Val-Unseen.

### 2. LookStep: language tags reduce VLN-CE memory and data costs
{: id="2-lookstep用语言标签替代认知地图压低-vln-ce-的记忆与数据成本"}

**Problem.** The paper points out that existing MLLM-driven VLN follows the next action prediction paradigm, only supervises expert actions, and requires large training data; it also relies on cognitive maps, accumulated historical frames, or external 3D tools to maintain status, resulting in high computational and memory overhead.

**Method.** End-to-end unified framework, including two components: Language Centric Future State Modeling uses language tags to generate coarse-grained navigation progress and future states for each candidate action; Event Driven Rolling Memory independently decides whether to write each frame of observation into bounded rolling memory with a certain semantic role.

**Evidence.** R2R-CE Val-Unseen success rate **49.7%**; the paper claims that it is better than existing methods under the same training settings, while having better memory efficiency and less data usage. The **abstract does not give other indicators such as SPL and nDTW, nor does it list the specific values of the control method.**

**Value.** is directly oriented to continuous environments, and the optimization goal is deployment cost (memory, data volume), which is complementary to the route of pursuing the upper limit; the write decision-making mechanism of bounded memory can be independently transplanted to other navigation frameworks.

**Limitations.** only has a single indicator (SR) and only Val-Unseen, so it is impossible to judge whether it is at the expense of path efficiency; the definition of equivalent training settings needs to be confirmed in the text; the absolute level of 49.7% needs to be compared with the R2R-CE list of the same period to be meaningful.

**Recommendation.** The work that deserves priority to be reproduced in this issue - there is [open source code ](https://github.com/kunyang-YU/LookStep) with clear goals. Focus on verifying the performance of the bounded rolling memory writing strategy on long instructions.

### 3. LightNav-0: one VLM backbone for multiple tasks and embodiments
{: id="3-lightnav-0单一-vlm-骨干覆盖多任务与多本体"}

**Problem.** The paper points out that existing navigation systems rely on task-specific or embodiment-specific components, which separate perception, reasoning and action, and have limited generalization; while VLM has encoded spatial priors such as visual positioning, spatial reasoning, and pointing, but is rarely used directly for robot control.

**Method.** Unified token interface: Dual-channel pointing expresses task/scene/embodiment-independent spatial intent; the residual vector quantization action tokenizer maps the intent into an accurate trajectory of a specific embodiment. Visual history compression, ER mid-term training, supervised fine-tuning and reinforcement learning with temporal awareness. The training corpus covers 2K+ scenes and 4K+ hours of embodied navigation data.

**Evidence.** The paper states that LightNav-ER achieves the highest complete set average on 8 embodied reasoning benchmarks; LightNav-0 achieves monocular success rate state-of-the-art on all 10 public navigation simulation settings; real robot evaluation shows cross-embodiment, cross-scenario and zero-shot generalization of static/dynamic targets. The **summary does not list specific values for either setting.**

**Value.** The interface design that decouples spatial intention and embodiment action is a feasible abstraction for deploying a model to a variety of ground platforms; it is of direct value to teams that maintain multiple robots at the same time.

**Limitations.** The data scale constitutes the main barrier to reproduction; the 10 settings of full SOTA lack numerical support, and it is impossible to judge whether the improvement of each task is balanced; the real robot conclusion can only be described qualitatively.

**Recommendation.** is an intensive reading of the architecture reference of the general navigation model, focusing on the specific definition of dual-channel pointing and the embodiment adaptation method of the action tokenizer; it is not included in the recurrence plan for the time being.

### 4. CGFM-Nav: coupling explicit semantic memory with semantic-guided exploration
{: id="4-cgfm-nav显式语义记忆与语义引导探索的闭环耦合"}

**Problem.** The paper points out that it is difficult for existing environmental representations to simultaneously support explicit semantic memory and continuous exploration guidance—either retrieving seen targets or guiding exploration of unseen areas.

**Method.** CGFM is a persistent multi-modal scene representation: it organizes objects, spatial relationships and visual observations into multi-modal scene graphs to support target retrieval and cross-task long-term reasoning; when there is no reliable target matching, the graph evidence is projected into a semantic frontier field of target conditions, guiding exploration to semantically promising frontiers and areas. CGFM-Nav superimposes task-related subgraph selection, VLM inference and verification feedback on it to form a closed-loop decision-making.

**Evidence.** On GOAT-Bench, **has the same Qwen3-VL-8B backbone** with an overall success rate of 53.2%→63.0% and SPL 30.0%→39.6%. The paper describes itself as a preliminary experiment.

**Value.** controls the control of backbone variables, so that the gains of +9.8 percentage points SR and +9.6 percentage points SPL can be attributed to the representation and exploration mechanism itself, rather than the model size; this is one of the navigation results with the highest quality of evidence in this issue.

**Limitations.** The author's preliminary statement; only GOAT-Bench single benchmark, not verified on R2R-CE / RxR-CE; computational overhead and error accumulation of scene graph construction are not discussed in the abstract; no real robot results.

**Recommendation.** tracks its official version; the graph evidence projection part of the semantic frontier field can be extracted separately and connected to the existing ObjectNav exploration strategy for ablation.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Robotic Arm / Freeze VLA: [AGM](https://arxiv.org/abs/2608.29537v1)**
  - Mechanism: Achievement-anchored memory: tasks are represented as a sequence of sub-goals with progress pointers, **Only after the current subgoal is verified by physical evidence** before advancing the pointer; proprioceptive interaction cues determine when to verify, point tracking and language condition cross-view comparison (2.43M parameter verification header) determine what to verify.
  - Access position: The sub-goal completion determination of long-sequence VLN replaces the optimistic update of "the action is deemed completed".
  - Prerequisites and risks: The PickXTimes / BinFill values of RoboMME Counting in the abstract are missing in the original text. **Unable to verify the increase** ; The verification signal relies on the contact clues of the manipulation task, and navigation requires another source of physical evidence.
- **Air-ground coordination VLN: [AGC-VLN](https://arxiv.org/abs/2609.03483v1)**
  - Mechanism: Sharing a bird's-eye view as a collaboration interface: the drone renders the ground vehicle's reported pose and VLM anchored target as CAR/GOAL markers and attaches distance labels on the global bird's eye view. Based on this, the ground vehicle obtains the global spatial context that cannot be provided by the first-person perspective, uses the frozen VLM to plan the path along the road and executes it in a closed loop.
  - Integration point: External overhead information injection when the ground vehicle lacks global context (not limited to drones, but also from fixed cameras or pre-built maps).
  - Prerequisites and risks: CARLA-Air Town10HD has 100 closed-loop episodes, and the joint success rate is 77.0%, which is 27.0 points higher than the weaker individual (UAV 50.0%) and 24.0 points higher than the strongest published single agent baseline (Travel UAV 53.0%); the conclusion is limited to simulated urban road scenes, and indoor transferability has not been verified.
- **long-term agent framework: [EmbodiedSkills](https://arxiv.org/abs/2609.01281v1)**
  - Mechanism: combine perception, planning, execution, **Progress verification and recovery** Unified orchestration framework.
  - Integration point: Failure recovery and re-planning layer of the VLN system.
  - Prerequisites and risks: The evidence is all in the manipulation tasks (RoboTwin 2.0 50 task average 86.20%, four LIBERO suites 97.40%), there is no verification on the navigation side.
- **failed correction: [Training-Free Action Correction](https://arxiv.org/abs/2608.29967v1)**
  - Mechanism: Use verbal feedback to correct VLA deployment failure without retraining.
  - Integration point: Navigation strategies are corrected online to avoid retraining for each type of failure.
  - Prerequisites and risks: The evidence is the manipulation task on LIBERO; the failure mode of navigation (wrong bifurcation, semantic misbinding) is different from the operation.
- **evaluation method: [R2S-Eval](https://arxiv.org/abs/2609.03276v1)**
  - Mechanism: Use VLM for real-to-sim calibration and then evaluate the real robot strategy in simulation.
  - Integration point: Reduce the labor cost and instability of navigation policy real robot evaluation.
  - Prerequisites and risks: Operation scenario calibration; navigation requires higher scene scale and dynamics.
- **evaluation method: [GeoAgent](https://geoagent-benchmark.github.io)**
  - Mechanism: Transform the static image task into an embodied navigation evaluation that requires active exploration; the paper states that agentic navigation significantly improves accuracy compared to the static image baseline.
  - Integration point: Evaluation design idea: Supplement the embodied link of "explore first and then judge" for perceptual abilities.
  - Prerequisites and risks: The task is geographical positioning rather than goal navigation; the paper also reports significant deviations of the model in developed/developing areas, as well as poor self-improvement ability when a priori errors are made.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **Revisiting Topological Graphs** (VLN-CE + closed loop RL · A): See key analysis 1. [Source](https://arxiv.org/abs/2609.03906v1)
- **LookStep** (VLN-CE efficiency · A): See key analysis 2. [Source](https://arxiv.org/abs/2609.02350v1)
- **LightNav-0** (general embodied navigation · A): See key analysis 3. [Source](https://arxiv.org/abs/2608.30935v1)
- **CGFM-Nav** (Lifelong Multi-modal Navigation · A): See Key Analysis 4. [Source](https://arxiv.org/abs/2608.29114v1)
- **VerNav** (Discrete R2R Low Latency · A): verifier-first replaces stepwise autoregression with batch action verification, reducing single-step LLM latency by more than 10 times in the decision-making phase. [Source](https://arxiv.org/abs/2609.00920v1)
- **CanonNav** (Cross-platform visual navigation · A): Camera geometry normalization decouples navigation behavior from platform geometry, with RGB-only inference outperforming RGB-D methods. [Source](https://arxiv.org/abs/2608.30242v1)
- **AGC-VLN** (Air-Ground Collaboration VLN · B): Sharing aerial view as a training-free collaboration interface, CARLA-Air joint success rate 77.0%. [Source](https://arxiv.org/abs/2609.03483v1)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **AGM** (Embodied Memory · B): Sub-goal progress pointer advances only after physical evidence verification; advocates that memory reliability depends on update discipline rather than capacity. [Source](https://arxiv.org/abs/2608.29537v1)
- **EmbodiedSkills** (Agent Orchestration·B): VLA framework for unified orchestration awareness/planning/execution/progress verification/recovery. [Source](https://arxiv.org/abs/2609.01281v1)
- **Training-Free Action Correction** (Failure Correction · B): Use verbal feedback to correct VLA deployment failure without retraining. [Source](https://arxiv.org/abs/2608.29967v1)
- **R2S-Eval** (Evaluation · B): VLM-driven real-to-sim calibration evaluation process. [Source](https://arxiv.org/abs/2609.03276v1)
- **GeoAgent** (Evaluation · B): Street View embodied navigation geolocation benchmark; reports developed/developing regional bias. [Source](https://geoagent-benchmark.github.io)
- **Drive the Thoughts** (Runtime Monitoring · C): Monitors VLA inference chain and trajectory consistency, and analysis shows that 33.3% of CoTs are unreliable. [Source](https://arxiv.org/abs/2608.29583v1)
- **LAVLA** (Interpretability · C): Layer-by-layer latent space clustering analysis of the GR00T N1.5 action decoder. [Source](https://arxiv.org/abs/2609.02634v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **MINERVA** (small model · C): The 0.54M parameter strategy averages 95.1% over 2,000 rollouts of the four standard LIBERO suites, just 2.4 points below the reported LeRobot π0.5 result. [Source](https://arxiv.org/abs/2609.03715v1)
- **DriftingVLA** (one-step generation · C): Native one-step action generation for dimension-by-dimensional time drift, LIBERO 98.32%, RoboTwin 2.0 81.09%, real robot six tasks 77.67%. [Source](https://arxiv.org/abs/2608.29749v1)
- **Temporal Forcing** (4D representation · C): Temporal representation alignment alleviates observation confusion, LIBERO 98.8% (+2.2 points compared to the pedestal model). [Source](https://arxiv.org/abs/2608.30643v1)
- **SMILE** (Action Smoothing · C): Predicted B-spline coefficients suppress motion block jitter, LIBERO 98.0% and 1.1x speedup. [Source](https://arxiv.org/abs/2608.29432v1)
- **GIFT** (Intermediate Feature Supervision · C): Action-oriented structured supervision, zero-shot transfer LIBERO-Plus up to 79.6% / 72.6% / 87.8%. [Source](https://arxiv.org/abs/2609.04193v1)
- **VLAct (Beyond Data Scaling)** (continuous pre-training · C): representation-centered continued pre-training, LIBERO-Plus 82.6%, RoboTwin 2.0 92.5%. [Source](https://arxiv.org/abs/2608.27550v1)
- **PHR-VLA** (Planning Horizon · C): Contact center potential dynamics supervision from wrist camera, LIBERO 84.1%→88.4%, real robot disassembly task 63.3%→82.5%. [Source](https://arxiv.org/abs/2608.27609v1)
- **PredVLA** (small model · C): Predictive sensorimotor modeling, LIBERO short-time three-suite 86.9%, four-suite 75.4%. [Source](https://arxiv.org/abs/2608.26673v2)
- **AdaVLA** (Inference Acceleration · C): Adaptive step flow matching, training-free acceleration. [Source](https://arxiv.org/abs/2608.29208v1)
- **Knowing When to Stop** (action chunking · C): Use internal cross-attention to dynamically adaptively determine the action chunk length. [Source](https://arxiv.org/abs/2609.00908v1)
- **REFACTOR-VLA** (Skill Library · C): Unsupervised learning of typed motion libraries; average pairwise NMI across providers 0.705 (95% confidence interval [0.683, 0.729]). [Source](https://arxiv.org/abs/2609.01215v1)
- **WISE** (post-training efficiency · C): World model-guided imagination scheduling, GPU computing time on π0/π0.5 is reduced by about 80% compared to full imagination. [Source](https://arxiv.org/abs/2609.03681v1)
- **PAVE** (Representation Alignment · C): Trajectory relative multi-horizon transfer alignment (25 / 50 / 75 / 100% of remaining episodes). [Source](https://arxiv.org/abs/2608.30378v2)
- **CometVLA** (collaborative training · C): Embodied data pyramid collaborative training to supplement physical knowledge. [Source](https://arxiv.org/abs/2608.30289v1)
- **SymVD** (Distillation · C): Symmetric visual language action distillation, reducing the data and retraining costs of task migration. [Source](https://arxiv.org/abs/2608.29828v1)
- **DREAM** (data generation · C): Deployment period real-to-sim generates demo data to adapt to the new workspace. [Source](https://arxiv.org/abs/2608.29078v1)
- **GRAFT** (Online RL · C): Online enhanced adaptation for delicate biomedical operations. [Source](https://arxiv.org/abs/2608.27079v2)
- **DeicticVLA** (command mode · C): Unifies the two command modes of language and indicative gestures; both modes are 100% successful on unseen categories, and the joint training LI baseline is 16.7%. [Source](https://arxiv.org/abs/2608.28108v1)
- **HINT** (long-term intent · C): Infer human intent from simple overall instructions and continuously adapt with visual observations. [Source](https://arxiv.org/abs/2609.02653v1)
- **Evidence-Gated Regularization** (Multimodal Robust · C): Alleviating modal entanglement, SR 12.5%→16.4% (all modes), 9.4%→16.5% (invalid sensor), 2.8%→6.1% (single sensor fallback). [Source](https://arxiv.org/abs/2609.03142v1)
- **ZETA** (cross-embodiment transfer · C): Controlled studies show that adding 5% target embodiment data to pre-training can improve the average progress of the target embodiment by 13.4 percentage points. [Source](https://arxiv.org/abs/2609.02546v1)
- **FWBC-VLA** (Force Sensing · C): Full-body force compensation for contact-intensive loco-manipulation. [Source](https://arxiv.org/abs/2609.03889v1)
- **Scaling Bimanual Household Manipulation** (Dataset · C): Released 1,500 hours of dual-arm household manipulation demonstration and made in-policy corrections. [Source](https://arxiv.org/abs/2609.03591v1)

### 5.4 UAVs, autonomous driving, and less relevant topics
{: id="54-无人机自动驾驶与其他低相关方向"}

- **MulDP** (Quadruped parkour navigation · C): The multi-modal diffusion policy generates navigation speed instructions and publishes the quadruped parkour navigation dataset QPND; no language instruction component. [Source](https://arxiv.org/abs/2609.03984v1)
- **LaPla** (autonomous driving · C): latent space alignment planning, the long horizon L2 error on nuScenes is 15.52% lower than the SOTA VLA method. [Source](https://arxiv.org/abs/2609.04070v1)
- **Rethinking Language's Role** (autonomous driving · C): Discuss the delay and memory cost of language modules in vehicle scenarios. [Source](https://arxiv.org/abs/2608.30144v1)
- **Aligning Multi-Trajectory Supervision** (autonomous driving · C): Trajectory selection problem when multi-trajectory simulation is combined with GRPO. [Source](https://arxiv.org/abs/2608.30122v1)
- **Towards Zero-Shot Transfer for Driving VLAs** (Autonomous Driving · C): Cross-embodiment zero-shot transfer for Driving VLAs. [Source](https://arxiv.org/abs/2609.02341v1)
- **Degradation-Tolerance Benchmark** (autonomous driving · C): Tolerance benchmark for pure camera end-to-end driving under blur/noise/low light/weather/dropped frames/memory failures. [Source](https://arxiv.org/abs/2608.29005v1)
- **MLLMs as Drone VLA Agents** (Drone · C): Evaluation MLLM directly into the drone control loop (command/approach/track/search). [Source](https://arxiv.org/abs/2609.01404v1)
- **Taxonomy of Construction Task Activities** (Domain Analysis · C): Taxonomy of construction worker activities and list of required capabilities for robots. [Source](https://arxiv.org/abs/2608.25395v2)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

There are no informational items in this issue. The WeChat public account source did not produce an entry due to an upstream service failure. For the reason, see the final note under "Key conclusions".

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- The method competition for **VLN-CE is shifting from “better representation” to “better training signal”.** Revisiting Topological Graphs uses macro actions to compress the decision horizon to make closed-loop RL trainable, and LookStep uses language labels to construct intermediate supervision (navigation progress, future state) instead of only supervising expert actions - both are bypassing the supervision sparse problem of pure imitation learning, but in different ways.
- **"Verification" is becoming a common component across tasks.** VerNav uses verifier instead of gradual generation to reduce latency, AGM uses verification headers to determine whether the memory pointer is advanced, CGFM-Nav incorporates verification feedback into closed-loop decision-making, and EmbodiedSkills lists progress verification as a first-level component of the framework. This digest's assessment: The design of validators (what evidence to use, when to trigger) is becoming an independent object of study.
- **inference cost is regarded as the first-class indicator.** LookStep (memory efficiency, data volume), VerNav (latency more than 10 times), MINERVA (0.54M parameters up to 95.1% of LIBERO), AdaVLA and DriftingVLA (number of inference steps) respectively reduce costs from the four dimensions of memory, delay, parameter volume, and sampling steps. This digest's assessment: This type of work is more valuable to real robot deployment than ranking on the list.
- **The number of robotic arm operations (13 articles in the LIBERO series) in this issue far exceeds that of ground navigation work.** This digest's assessment: This is the field distribution fact of arXiv during this period, and it is not appropriate to adjust the navigation direction input accordingly.

### Research gaps
{: id="研究空白"}

- Verifiable numbers for **R2R-CE are scarce.** Only 2 of the 45 articles in this issue are reported to R2R-CE, and 1 of them (Revisiting Topological Graphs) only claims SOTA without giving a numerical value, so direct comparison cannot be completed.
- The **navigation side lacks a source of physical evidence like AGM.** The manipulation task can be completed with the sub-goal of proprioceptive contact cue determination. What is the equivalent evidence in navigation (arrival determination, semantic matching confidence, traversability change) has not been systematically studied.
- **cross-platform data reuse only has a mechanism and no scale verification.** CanonNav proposed camera geometry normalization, but did not provide a dataset and numerical values; it is still blank how much gain multi-robot data aggregation can bring.

### Suggested actions
{: id="建议动作"}

**High priority**

- **reproduces**: LookStep: There is open source code, R2R-CE Val-Unseen SR 49.7%, the caliber is clear, focusing on verifying the writing strategy of bounded rolling memory.
- **Read closely**: Revisiting Topological Graphs: Check the actual values of R2R-CE / RxR-CE in the text and the comparison baseline to judge the quality claimed by SOTA.
- **Read closely**: CGFM-Nav: The GOAT-Bench comparison controlling backbone variables is one of the navigation results with the highest evidence quality in this issue. Pay attention to its official version.

**Medium priority**

- **Adapt**: AGM’s “progress pointer will be advanced after verification” update discipline to design the physical evidence determination of navigation sub-goals.
- **architecture reference**: LightNav-0’s dual-channel pointing and residual VQ action tokenizer; defer reproduction because of the required corpus scale.
- **Track**: VerNav's verifier-first delay optimization; note that it is discrete R2R and needs to be re-verified when migrating to CE.

**Low priority**

- **Defer**: the 13 LIBERO-family manipulation works and driving/UAV research; continue tracking benchmark developments only.
