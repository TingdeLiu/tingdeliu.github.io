---
layout: post
lang: en
translation_id: vln-weekly-2026-09-05
permalink: /en/vln-weekly-2026-09-05/
source_path: _posts/weekly-reports/2026-09-05-VLN-Weekly.md
source_url: /vln-weekly-2026-09-05/
source_revision_date: 2026-10-24
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly · Issue 3"
date: 2026-09-05
last_modified_at: 2026-10-24
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

- **The training paradigm for VLN in continuous environments shows signs of converging on reinforcement learning.** Only 2 of the 45 works in this issue report on R2R-CE. Among them, *Revisiting Topological Graphs* recasts VLN-CE as a hierarchical MDP, using the frontier nodes of the topological graph as the macro-action space and a training-free low-level controller as the state transition, which compresses the decision horizon and makes closed-loop RL tractable. The paper cites two old problems of closed-loop imitation learning (distribution shift in behavior cloning, and expert-action ambiguity once DAgger deviates from the trajectory) as its motivation for RL. This digest's assessment: this route reopens the established conclusion that "RL is too sample-inefficient on VLN-CE" and deserves priority tracking.
- **A parallel route is to lower cost rather than raise the ceiling.** By dropping cognitive maps, stacked history frames, and external 3D tools, *LookStep* reaches an R2R-CE Val-Unseen success rate of 49.7% with less data and less memory overhead under the same training settings. It instead uses language labels to generate coarse-grained navigation progress and future states, and decides on its own whether to write each observation into a bounded rolling memory. This digest's assessment: this contrasts with the previous route, which raises the ceiling, while this one lowers deployment cost; both are evaluated on R2R-CE and can be compared directly.
- **General navigation models are starting to cover multiple tasks and embodiments with a single VLM backbone.** *LightNav-0* replaces task-specific prediction heads with a unified token interface (dual-channel pointing that expresses spatial intent independent of task, scene, and embodiment, plus a residual vector quantization action tokenizer that maps it to the trajectory of a specific embodiment). The paper claims state-of-the-art single-goal success rates on all 10 public navigation simulation settings and zero-shot cross-embodiment generalization on real robots. This digest's assessment: the corpus size of this work (2K+ scenes, 4K+ hours) is the main barrier to reproduction.
- **The debate on memory representation has shifted from "capacity" to "update discipline".** *CGFM-Nav* uses a multimodal scene graph plus a goal-conditioned semantic frontier field to raise success from 53.2% to 63.0% and SPL from 30.0% to 39.6% on GOAT-Bench with the same Qwen3-VL-8B backbone; on the robotic-arm side, *AGM* argues explicitly that reliable embodied memory depends more on disciplined state updates than on memory capacity, advancing its progress pointer only after a sub-goal is verified by physical evidence. This digest's assessment: the two works give evidence pointing the same way from different tasks, with direct reference value for sub-goal tracking in long-horizon navigation.
- **This issue has no WeChat source; all 45 entries come from arXiv.** The relay service that the original wewe-rss relied on was permanently shut down when Deno Deploy Classic went offline on 2026-07-20. Collection was migrated to we-mp-rss (which uses the WeChat Official Accounts Platform interface), but the newly authorized account triggered WeChat rate limiting and all four collection rounds returned 0. As a result, this issue has no WeChat interpretation perspective, no cross-source merging took place, and "45 entries" means 45 independent works.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [Revisiting Topological Graphs](https://arxiv.org/abs/2609.03906v1)** · VLN-CE (continuous environments)
  - Contribution: A hierarchical MDP with topological-graph macro actions, and graph-based PPO supported by an action-aware value head.
  - Evidence: The abstract states state-of-the-art results on R2R-CE and RxR-CE, **but gives no verifiable numbers**.
  - Why it matters: Making closed-loop RL trainable on VLN-CE is a route-level change.
- **A2 · [LookStep](https://arxiv.org/abs/2609.02350v1)** · VLN-CE (continuous environments)
  - Contribution: Language-centric future-state modeling plus event-driven bounded rolling memory, removing cognitive maps and external 3D tools.
  - Evidence: R2R-CE Val-Unseen success rate of 49.7%; the paper states it beats existing methods under the same training settings, with higher memory efficiency and less data.
  - Why it matters: The only work in this issue that provides verifiable R2R-CE numbers, and it [has open-source code](https://github.com/kunyang-YU/LookStep)
- **A3 · [LightNav-0](https://arxiv.org/abs/2608.30935v1)** · General embodied navigation (instruction following / open-vocabulary ObjectNav / visual tracking)
  - Contribution: A unified token interface: dual-channel pointing plus a residual VQ action tokenizer, with no task-specific heads.
  - Evidence: The paper states that single-goal success rates on all 10 public navigation simulation settings are SOTA, and that LightNav-ER has the highest overall average on 8 embodied reasoning benchmarks. **The summary does not list specific values for each setting.**
  - Why it matters: A complete solution in which one VLM backbone covers multiple tasks and embodiments.
- **A4 · [CGFM-Nav](https://arxiv.org/abs/2608.29114v1)** · Lifelong multimodal embodied navigation
  - Contribution: A multimodal scene graph (explicit relational memory) and a goal-conditioned semantic frontier field (continuous exploration guidance) form a closed loop.
  - Evidence: On GOAT-Bench, with the same Qwen3-VL-8B backbone, success rate goes 53.2% → 63.0% and SPL 30.0% → 39.6% (the paper calls these preliminary experiments).
  - Why it matters: A concrete design for coupling memory with exploration, with a clearly defined metric protocol.
- **A5 · [VerNav](https://arxiv.org/abs/2609.00920v1)** · **Discrete R2R** (non-CE)
  - Contribution: Verifier-first: batch action verification replaces step-by-step autoregressive generation, and the generator is called only for uncertain decisions.
  - Evidence: On the R2R benchmark, per-step LLM latency in the decision stage is **10 times or more** lower than for autoregressive methods, with navigation performance described as competitive.
  - Why it matters: Latency is a hard constraint for deploying LLM-based VLN. Note that its evaluation is on discrete R2R and **cannot be compared directly with A1/A2's R2R-CE**.
- **A6 · [CanonNav](https://arxiv.org/abs/2608.30242v1)** · Cross-platform visual navigation (real-robot related)
  - Contribution: Camera-geometry normalization that decouples navigation behavior from platform camera geometry; pseudo-labels from an offline traversability estimator provide safety and local-progress supervision.
  - Evidence: The paper claims that RGB-only inference consistently beats RGB baselines and surpasses RGB-D methods in difficult scenarios. **No datasets or values are given.**
  - Why it matters: A mechanism-level contribution to cross-platform data reuse, useful for aggregating data from multiple robots.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. Revisiting Topological Graphs
{: id="1-revisiting-topological-graphs把-vln-ce-的决策视界压到-rl-能训练的尺度"}

**Shortening the VLN-CE decision horizon for RL.**

**Problem.** The paper identifies two specific bottlenecks of closed-loop imitation learning for VLN-CE: behavior cloning runs into distribution shift, and DAgger expert actions become ambiguous once the agent deviates from the trajectory. Yet running RL directly in the primitive-action space has very low sample efficiency because rewards are sparse.

**Method.** It is reformulated as a hierarchical MDP that explicitly decouples high-level planning from low-level control: the environment is abstracted into a topological graph, the high-level policy decides over a macro-action space made of frontier nodes, and a training-free low-level controller acts as the state transition. To evaluate state values over the dynamic frontier action space, an action-aware value head is proposed to support graph-based PPO.

**Evidence.** The abstract states state-of-the-art results on R2R-CE and RxR-CE, **but gives no numerical values, comparison methods, or sub-metrics**.

**Value.** If the conclusion holds, this is a workable engineering path for bringing closed-loop RL into VLN-CE; the decomposition into macro actions plus a training-free low-level controller aligns naturally with the existing navigation stack (topological map plus local planner) of ground robots.

**Limitations.** All the evidence rests on the word "SOTA", so the margin cannot be verified; the abstract does not describe the quality of topology construction or the sensitivity to the frontier-selection strategy; and because the low-level controller is training-free, its capability ceiling directly bounds overall performance.

**Recommendation.** Read the value-head design and PPO training details in the full text first, and check the specific SR/SPL values and comparison baselines on R2R-CE Val-Unseen.

### 2. LookStep
{: id="2-lookstep用语言标签替代认知地图压低-vln-ce-的记忆与数据成本"}

**Language labels reduce the memory and data cost of VLN-CE.**

**Problem.** The paper points out that existing MLLM-driven VLN follows the next-action prediction paradigm, supervises only expert actions, and needs large amounts of training data; it also relies on cognitive maps, accumulated history frames, or external 3D tools to maintain state, which causes high computation and memory overhead.

**Method.** An end-to-end unified framework with two components. Language-Centric Future State Modeling uses language labels to generate coarse-grained navigation progress and future states for each candidate action; Event-Driven Rolling Memory decides independently whether to write each frame's observation, with a given semantic role, into a bounded rolling memory.

**Evidence.** R2R-CE Val-Unseen success rate **49.7%**; the paper claims it beats existing methods under the same training settings while using memory more efficiently and less data. **The abstract gives no other metrics such as SPL and nDTW, and no specific values for the compared methods.**

**Value.** It targets continuous environments directly and optimizes deployment cost (memory, data volume), complementing the route of chasing the ceiling; the write-decision mechanism of the bounded memory can be ported independently to other navigation frameworks.

**Limitations.** Only a single metric (SR) and only Val-Unseen are reported, so it is impossible to tell whether it comes at the expense of path efficiency; the definition of "the same training settings" must be confirmed in the full text; and the absolute 49.7% only means something once compared against contemporaneous R2R-CE leaderboards.

**Recommendation.** The work in this issue most worth reproducing first: there is [open-source code](https://github.com/kunyang-YU/LookStep) and the goals are clear. Focus on verifying how the bounded rolling-memory write strategy performs on long instructions.

### 3. LightNav-0
{: id="3-lightnav-0单一-vlm-骨干覆盖多任务与多本体"}

**One VLM backbone for multiple tasks and embodiments.**

**Problem.** The paper points out that existing navigation systems rely on task-specific or embodiment-specific components that separate perception, reasoning, and action, which limits generalization; VLMs already encode spatial priors such as visual grounding, spatial reasoning, and pointing, but are rarely used directly for robot control.

**Method.** A unified token interface: dual-channel pointing expresses spatial intent independent of task, scene, and embodiment, and a residual vector quantization action tokenizer maps that intent to an accurate trajectory for a specific embodiment. The recipe includes visual history compression, ER mid-training, supervised fine-tuning, and temporally aware reinforcement learning. The training corpus covers 2K+ scenes and 4K+ hours of embodied navigation data.

**Evidence.** The paper states that LightNav-ER achieves the highest overall average on 8 embodied reasoning benchmarks; LightNav-0 achieves state-of-the-art single-goal success rates on all 10 public navigation simulation settings; and real-robot evaluation shows cross-embodiment, cross-scene, and zero-shot generalization to static and dynamic goals. **The summary lists no specific values for either.**

**Value.** The interface design that decouples spatial intent from embodiment-specific actions is a workable abstraction for deploying one model across many ground platforms, and it is of direct value to teams that maintain several robots at once.

**Limitations.** The data scale is the main barrier to reproduction; the claim of SOTA on all 10 settings lacks numerical support, so it is impossible to judge whether the gains are balanced across tasks; and the real-robot conclusions are only qualitative.

**Recommendation.** Read closely as an architecture reference for general navigation models, focusing on the specific definition of dual-channel pointing and on how the action tokenizer adapts to an embodiment; leave it out of the reproduction plan for now.

### 4. CGFM-Nav
{: id="4-cgfm-nav显式语义记忆与语义引导探索的闭环耦合"}

**Coupling explicit semantic memory with semantics-guided exploration.**

**Problem.** The paper points out that existing environment representations struggle to support both explicit semantic memory and continuous exploration guidance: they either retrieve targets already seen or guide exploration into unseen areas, but not both.

**Method.** CGFM is a persistent multimodal scene representation: it organizes objects, spatial relations, and visual observations into a multimodal scene graph to support goal retrieval and cross-task long-term reasoning; when there is no reliable goal match, the graph evidence is projected into a goal-conditioned semantic frontier field that guides exploration toward semantically promising frontiers and regions. CGFM-Nav layers task-relevant subgraph selection, VLM inference, and verification feedback on top to form a closed decision loop.

**Evidence.** On GOAT-Bench, **with the same Qwen3-VL-8B backbone**, overall success rate goes 53.2%→63.0% and SPL 30.0%→39.6%. The paper describes these as preliminary experiments.

**Value.** Holding the backbone fixed means the gains of +9.8 percentage points SR and +9.6 percentage points SPL can be attributed to the representation and exploration mechanism itself rather than model size; this is one of the navigation results with the highest evidence quality in this issue.

**Limitations.** The authors call it preliminary; only the single GOAT-Bench benchmark, with no validation on R2R-CE / RxR-CE; the abstract does not discuss the computational overhead or error accumulation of scene-graph construction; no real-robot results.

**Recommendation.** Track its official version; the graph-evidence projection into the semantic frontier field can be extracted on its own and attached to an existing ObjectNav exploration policy for ablation.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Robotic arm / frozen VLA: [AGM](https://arxiv.org/abs/2608.29537v1)**
  - Mechanism: Achievement-anchored memory: a task is represented as a sequence of sub-goals with a progress pointer, and the pointer advances **only after the current sub-goal is verified by physical evidence**; proprioceptive interaction cues decide when to verify, and point tracking plus language-conditioned cross-view comparison (a 2.43M-parameter verification head) decides what to verify.
  - Integration point: Sub-goal completion judgment in long-horizon VLN, replacing the optimistic update of "assume the action completed".
  - Prerequisites and risks: The PickXTimes / BinFill values for RoboMME Counting in the abstract are missing from the source, so **the improvement cannot be verified**; the verification signal relies on contact cues in manipulation tasks, and navigation needs a different source of physical evidence.
- **Air-ground collaborative VLN: [AGC-VLN](https://arxiv.org/abs/2609.03483v1)**
  - Mechanism: A shared bird's-eye view serves as the collaboration interface: the drone renders the ground vehicle's reported pose and the VLM-anchored goal as CAR/GOAL markers with distance labels on a global bird's-eye view. From this the ground vehicle obtains global spatial context that a first-person view cannot provide, plans a route along the road with a frozen VLM, and executes it in a closed loop.
  - Integration point: Injecting external overhead information when a ground vehicle lacks global context (not limited to drones; it could also come from fixed cameras or prebuilt maps).
  - Prerequisites and risks: On CARLA-Air Town10HD, over 100 closed-loop episodes, the joint success rate is 77.0%, 27.0 points above the weaker individual agent (UAV 50.0%) and 24.0 points above the strongest published single-agent baseline (Travel UAV 53.0%); the conclusions are limited to simulated urban road scenes, and transfer to indoor settings is unverified.
- **Long-horizon agent framework: [EmbodiedSkills](https://arxiv.org/abs/2609.01281v1)**
  - Mechanism: A unified orchestration framework combining perception, planning, execution, and **progress verification and recovery**.
  - Integration point: The failure-recovery and replanning layer of a VLN system.
  - Prerequisites and risks: All evidence is on manipulation tasks (RoboTwin 2.0 50-task average of 86.20%, four LIBERO suites 97.40%); nothing is validated on the navigation side.
- **Failure correction: [Training-Free Action Correction](https://arxiv.org/abs/2608.29967v1)**
  - Mechanism: Uses verbal feedback to correct VLA deployment failures without retraining.
  - Integration point: Online correction of navigation policies, avoiding retraining for each type of failure.
  - Prerequisites and risks: The evidence is manipulation tasks on LIBERO; navigation failure modes (wrong forks, semantic misbinding) differ from manipulation.
- **Evaluation method: [R2S-Eval](https://arxiv.org/abs/2609.03276v1)**
  - Mechanism: Uses a VLM for real-to-sim calibration, then evaluates real-robot policies in simulation.
  - Integration point: Reducing the labor cost and instability of real-robot evaluation of navigation policies.
  - Prerequisites and risks: Calibrated for manipulation scenes; navigation demands larger scene scale and richer dynamics.
- **Evaluation method: [GeoAgent](https://geoagent-benchmark.github.io)**
  - Mechanism: Turns static-image tasks into embodied navigation evaluations that require active exploration; the paper states that agentic navigation improves accuracy significantly over the static-image baseline.
  - Integration point: An evaluation design idea: add an embodied "explore first, then judge" stage to perception evaluations.
  - Prerequisites and risks: The task is geolocation, not goal navigation; the paper also reports significant model bias between developed and developing regions, and weak self-improvement when the prior is wrong.

## 5. Research roundup by category
{: id="五分类速览"}

Relevance labels: A means directly relevant to ground navigation research, B means the mechanism is transferable, and C is for field observation only.

### 5.1 VLN and ObjectNav
{: id="51-vln-与-objectnav"}

- **Revisiting Topological Graphs** (VLN-CE + closed-loop RL · A): See key analysis 1. [Source](https://arxiv.org/abs/2609.03906v1)
- **LookStep** (VLN-CE efficiency · A): See key analysis 2. [Source](https://arxiv.org/abs/2609.02350v1)
- **LightNav-0** (general embodied navigation · A): See key analysis 3. [Source](https://arxiv.org/abs/2608.30935v1)
- **CGFM-Nav** (lifelong multimodal navigation · A): See key analysis 4. [Source](https://arxiv.org/abs/2608.29114v1)
- **VerNav** (low-latency discrete R2R · A): Verifier-first: replaces step-by-step autoregression with batch action verification, cutting per-step LLM latency in the decision stage by more than 10 times. [Source](https://arxiv.org/abs/2609.00920v1)
- **CanonNav** (cross-platform visual navigation · A): Camera-geometry normalization decouples navigation behavior from platform geometry; RGB-only inference outperforms RGB-D methods. [Source](https://arxiv.org/abs/2608.30242v1)
- **AGC-VLN** (air-ground collaborative VLN · B): A shared aerial view serves as a training-free collaboration interface; joint success rate on CARLA-Air is 77.0%. [Source](https://arxiv.org/abs/2609.03483v1)

### 5.2 Memory, planning and evaluation
{: id="52-记忆规划与评测"}

- **AGM** (embodied memory · B): The sub-goal progress pointer advances only after verification by physical evidence; argues that memory reliability depends on update discipline rather than capacity. [Source](https://arxiv.org/abs/2608.29537v1)
- **EmbodiedSkills** (agent orchestration · B): A VLA framework that orchestrates perception, planning, execution, progress verification, and recovery in a unified way. [Source](https://arxiv.org/abs/2609.01281v1)
- **Training-Free Action Correction** (failure correction · B): Uses verbal feedback to correct VLA deployment failures without retraining. [Source](https://arxiv.org/abs/2608.29967v1)
- **R2S-Eval** (evaluation · B): A VLM-driven real-to-sim calibration and evaluation pipeline. [Source](https://arxiv.org/abs/2609.03276v1)
- **GeoAgent** (evaluation · B): A Street View embodied-navigation geolocation benchmark; reports bias between developed and developing regions. [Source](https://geoagent-benchmark.github.io)
- **Drive the Thoughts** (runtime monitoring · C): Monitors the consistency between the VLA reasoning chain and the trajectory; analysis finds 33.3% of CoTs unreliable. [Source](https://arxiv.org/abs/2608.29583v1)
- **LAVLA** (interpretability · C): Layer-by-layer latent-space clustering analysis of the GR00T N1.5 action decoder. [Source](https://arxiv.org/abs/2609.02634v1)

### 5.3 VLA and manipulation
{: id="53-具身-vla--移动操作"}

- **MINERVA** (small model · C): A 0.54M-parameter policy averages 95.1% over 2,000 rollouts of the four standard LIBERO suites, just 2.4 points below the reported LeRobot π0.5 result. [Source](https://arxiv.org/abs/2609.03715v1)
- **DriftingVLA** (one-step generation · C): Native one-step action generation using per-dimension temporal drift; LIBERO 98.32%, RoboTwin 2.0 81.09%, and 77.67% on six real-robot tasks. [Source](https://arxiv.org/abs/2608.29749v1)
- **Temporal Forcing** (4D representation · C): Temporal representation alignment alleviates observation aliasing; LIBERO 98.8% (+2.2 points over the base model). [Source](https://arxiv.org/abs/2608.30643v1)
- **SMILE** (action smoothing · C): Predicting B-spline coefficients suppresses jitter within action chunks; LIBERO 98.0% and a 1.1x speedup. [Source](https://arxiv.org/abs/2608.29432v1)
- **GIFT** (intermediate feature supervision · C): Action-oriented structured supervision; zero-shot transfer on LIBERO-Plus reaches up to 79.6% / 72.6% / 87.8%. [Source](https://arxiv.org/abs/2609.04193v1)
- **VLAct (Beyond Data Scaling)** (continued pre-training · C): Representation-centered continued pre-training; LIBERO-Plus 82.6%, RoboTwin 2.0 92.5%. [Source](https://arxiv.org/abs/2608.27550v1)
- **PHR-VLA** (planning horizon · C): Contact-centric latent dynamics supervision from the wrist camera; LIBERO 84.1%→88.4%, real-robot disassembly task 63.3%→82.5%. [Source](https://arxiv.org/abs/2608.27609v1)
- **PredVLA** (small model · C): Predictive sensorimotor modeling; 86.9% on three LIBERO suites and 75.4% on all four. [Source](https://arxiv.org/abs/2608.26673v2)
- **AdaVLA** (inference acceleration · C): Adaptive-step flow matching, a training-free speedup. [Source](https://arxiv.org/abs/2608.29208v1)
- **Knowing When to Stop** (action chunking · C): Uses internal cross-attention to decide the action chunk length adaptively. [Source](https://arxiv.org/abs/2609.00908v1)
- **REFACTOR-VLA** (skill library · C): Unsupervised learning of typed motion libraries; average pairwise NMI across providers 0.705 (95% confidence interval [0.683, 0.729]). [Source](https://arxiv.org/abs/2609.01215v1)
- **WISE** (post-training efficiency · C): World-model-guided scheduling of imagination; GPU compute on π0/π0.5 drops by about 80% compared with full imagination. [Source](https://arxiv.org/abs/2609.03681v1)
- **PAVE** (representation alignment · C): Multi-horizon alignment relative to the trajectory (25 / 50 / 75 / 100% of the remaining episode). [Source](https://arxiv.org/abs/2608.30378v2)
- **CometVLA** (co-training · C): Co-training over an embodied data pyramid to supplement physical knowledge. [Source](https://arxiv.org/abs/2608.30289v1)
- **SymVD** (distillation · C): Symmetric vision-language-action distillation, reducing the data and retraining costs of task transfer. [Source](https://arxiv.org/abs/2608.29828v1)
- **DREAM** (data generation · C): Real-to-sim generation of demonstration data at deployment time to adapt to a new workspace. [Source](https://arxiv.org/abs/2608.29078v1)
- **GRAFT** (online RL · C): Online reinforcement-based adaptation for delicate biomedical manipulation. [Source](https://arxiv.org/abs/2608.27079v2)
- **DeicticVLA** (instruction modality · C): Unifies two instruction modes, language and deictic gestures; both modes reach 100% success on unseen categories, versus 16.7% for the jointly trained LI baseline. [Source](https://arxiv.org/abs/2608.28108v1)
- **HINT** (long-horizon intent · C): Infers human intent from a simple overall instruction and keeps adapting it with visual observations. [Source](https://arxiv.org/abs/2609.02653v1)
- **Evidence-Gated Regularization** (multimodal robustness · C): Alleviates modality entanglement; SR 12.5%→16.4% (all modalities), 9.4%→16.5% (failed sensor), 2.8%→6.1% (single-sensor fallback). [Source](https://arxiv.org/abs/2609.03142v1)
- **ZETA** (cross-embodiment transfer · C): A controlled study shows that adding 5% target-embodiment data to pre-training improves the target embodiment's average progress by 13.4 percentage points. [Source](https://arxiv.org/abs/2609.02546v1)
- **FWBC-VLA** (force sensing · C): Whole-body force compensation for contact-rich loco-manipulation. [Source](https://arxiv.org/abs/2609.03889v1)
- **Scaling Bimanual Household Manipulation** (dataset · C): Releases 1,500 hours of bimanual household manipulation demonstrations along with on-policy corrections. [Source](https://arxiv.org/abs/2609.03591v1)

### 5.4 UAVs and other
{: id="54-无人机与其他"}

- **MulDP** (quadruped parkour navigation · C): A multimodal diffusion policy generates navigation velocity commands, and the quadruped parkour navigation dataset QPND is released; no language-instruction component. [Source](https://arxiv.org/abs/2609.03984v1)
- **LaPla** (autonomous driving · C): Latent-space alignment planning; long-horizon L2 error on nuScenes is 15.52% lower than the SOTA VLA method. [Source](https://arxiv.org/abs/2609.04070v1)
- **Rethinking Language's Role** (autonomous driving · C): Discusses the latency and memory cost of language modules in in-vehicle scenarios. [Source](https://arxiv.org/abs/2608.30144v1)
- **Aligning Multi-Trajectory Supervision** (autonomous driving · C): The trajectory-selection problem when multi-trajectory simulation is combined with GRPO. [Source](https://arxiv.org/abs/2608.30122v1)
- **Towards Zero-Shot Transfer for Driving VLAs** (autonomous driving · C): Zero-shot cross-embodiment transfer for driving VLAs. [Source](https://arxiv.org/abs/2609.02341v1)
- **Degradation-Tolerance Benchmark** (autonomous driving · C): A tolerance benchmark for camera-only end-to-end driving under blur, noise, low light, weather, dropped frames, and memory failures. [Source](https://arxiv.org/abs/2608.29005v1)
- **MLLMs as Drone VLA Agents** (UAV · C): Evaluates MLLMs placed directly in the drone control loop (instruction following / approach / tracking / search). [Source](https://arxiv.org/abs/2609.01404v1)
- **Taxonomy of Construction Task Activities** (domain analysis · C): A taxonomy of construction worker activities and a list of the capabilities robots would need. [Source](https://arxiv.org/abs/2608.25395v2)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

There are no news items in this issue. The WeChat source produced no entries because of an upstream service failure; see the final note under "Key conclusions".

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **The method competition for VLN-CE is shifting from "better representations" to "better training signals".** Revisiting Topological Graphs compresses the decision horizon with macro actions to make closed-loop RL trainable, while LookStep uses language labels to build intermediate supervision (navigation progress, future state) rather than supervising only expert actions. Both sidestep the sparse supervision of pure imitation learning, in different ways.
- **"Verification" is becoming a component shared across tasks.** VerNav uses a verifier instead of step-by-step generation to cut latency, AGM uses a verification head to decide whether the memory pointer advances, CGFM-Nav folds verification feedback into its closed-loop decisions, and EmbodiedSkills lists progress verification as a first-class framework component. This digest's assessment: the design of verifiers (what evidence to use and when to trigger them) is becoming a research subject in its own right.
- **Inference cost is being treated as a first-class metric.** LookStep (memory efficiency, data volume), VerNav (latency more than 10 times lower), MINERVA (0.54M parameters reaching 95.1% on LIBERO), and AdaVLA and DriftingVLA (number of inference steps) cut cost along four dimensions: memory, latency, parameter count, and sampling steps. This digest's assessment: such work is more valuable for real-robot deployment than a leaderboard rank.
- **Manipulation works (13 papers in the LIBERO family) far outnumber ground navigation works in this issue.** This digest's assessment: this is simply how arXiv was distributed during this period, and it would not be appropriate to shift navigation research investment in response.

### Research gaps
{: id="研究空白"}

- **Verifiable R2R-CE numbers are scarce.** Only 2 of the 45 works in this issue report on R2R-CE, and 1 of them (Revisiting Topological Graphs) only claims SOTA without giving a value, so direct comparison cannot be completed.
- **The navigation side lacks a source of physical evidence like AGM's.** In manipulation, sub-goal completion can be judged from proprioceptive contact cues. What the equivalent evidence in navigation would be (arrival judgments, semantic-match confidence, traversability changes) has not been studied systematically.
- **Cross-platform data reuse has only a mechanism and no validation at scale.** CanonNav proposes camera-geometry normalization but gives no datasets or values; how much gain aggregating multi-robot data can bring is still unknown.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Reproduce**: LookStep: open-source code is available, R2R-CE Val-Unseen SR is 49.7% with a clear protocol; focus on verifying the write strategy of the bounded rolling memory.
- **Read closely**: Revisiting Topological Graphs: check the actual R2R-CE / RxR-CE values and comparison baselines in the full text to judge the quality of its SOTA claim.
- **Read closely**: CGFM-Nav: its GOAT-Bench comparison with a controlled backbone is one of the navigation results with the highest evidence quality in this issue; watch for its official version.

**Medium priority**

- **Adapt**: AGM's update discipline of "advance the progress pointer only after verification" to design physical-evidence judgments for navigation sub-goals.
- **Architecture reference**: LightNav-0's dual-channel pointing and residual VQ action tokenizer; defer reproduction because of the required corpus scale.
- **Track**: VerNav's verifier-first latency optimization; note that it is on discrete R2R and must be re-verified when moved to CE.

**Low priority**

- **Defer**: the 13 LIBERO-family manipulation works and the driving/UAV works; only keep following benchmark developments.
