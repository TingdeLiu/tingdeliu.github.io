---
layout: post
lang: en
translation_id: vln-weekly-2026-08-22
permalink: /en/vln-weekly-2026-08-22/
source_path: _posts/weekly-reports/2026-08-22-VLN-Weekly.md
source_url: /vln-weekly-2026-08-22/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-08-13 to 2026-08-20)"
date: 2026-08-22
period_start: 2026-08-13
period_end: 2026-08-20
issue_number: 1
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "Only 5 of this issue's 46 new works primarily concern navigation, and R2R/RxR results are becoming scarce. Embodied-Navigator is the only work reporting R2R-CE numbers (66.2% SR). CondVLN uses conditional instructions to expose how SR/SPL can overestimate instruction understanding. Pluggable memory and planning modules such as RP1 and Remember Smarter offer more transferable value for long-horizon ground navigation than any individual manipulation SOTA."
---
## 1. Key conclusions
{: id="一本期结论"}

- **Among the 46 new works in this issue, only 1 reports on R2R series indicators and 0 reports on RxR. The only matching work is** [Embodied-Navigator/TAMP-Nav](https://arxiv.org/abs/2608.17512v1), which the abstract states achieves 66.2% SR on R2R-CE. The rest of the navigation work uses AI2-THOR, RoboTHOR, Matterport3D, Gibson or self-built real robot scenes, and none of them uses the R2R/RxR rankings as the main battlefield. This digest's assessment: The supply of papers with R2R/RxR list gain as the core selling point is becoming thinner. Comparable numbers on standard benchmarks need to be actively checked on arXiv full text.
- **The proportion of navigation work is very low: 5 of the 46 articles are mainly navigation, and 23 primarily concern manipulation.** This issue's arXiv hit set is dominated by VLA manipulation (especially the LIBERO family), and new methods for ground-based VLN are in short supply. This digest's assessment: This week is not suitable for the topic of "direct comparison of VLN methods". It is more suitable for the topic of evaluation methods or transferable mechanisms.
- **VLN The focus of the evaluation has shifted from "did you get there" to "whether you followed the correct conditions".** [CondVLN](https://arxiv.org/abs/2608.17318v1) points out that an agent can take the wrong logical branch while the path looks reasonable, and that standard success rates and path lengths cannot expose this failure. Impact on ground-based VLN: The existing evaluation protocol mainly based on SR/SPL may overestimate the instruction understanding ability, and it is worth adding conditional branch instructions in the own evaluation.
- **Several works have made "memory compression" and "learnable planning" into pluggable modules**, and at least one ([RP1](https://arxiv.org/abs/2608.18669v1)) clearly includes visual navigation experiments in the abstract. This digest's assessment: This type of module is the most valuable part of this issue for direct migration to ground long-range navigation, and has a higher priority than any single-article manipulation SOTA.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [Embodied-Navigator / TAMP-Nav](https://arxiv.org/abs/2608.17512v1)** · Continuous Environment VLN
  - Contributions: pixel-to-3D action representation + selective reasoning memory + GRPO two-level alignment.
  - Evidence: The abstract claims R2R-CE 66.2% SR, trained with 90k trajectories.
  - Reason: This issue is the only one reporting R2R indicators, and the sample efficiency has clear figures.
- **A2 · [CondVLN（If, Then, Otherwise）](https://arxiv.org/abs/2608.17318v1)** · VLN conditional branch evaluation
  - Contribution: conditional instruction benchmark for scene graph grounding + branch-specific indicators.
  - Evidence: Abstract claims 11,500+ instructions, covering AI2-THOR/MP3D/Gibson/ReplicaCAD, evaluating VLN-Zero, NaVid, NaVILA, Open-Nav; neural symbolic branch selector improved by 2x.
  - Reason: Direct impact on the existing SR/SPL evaluation protocol, and can be reused as its own diagnostic set.
- **A3 · [Graph-MambaNav](https://arxiv.org/abs/2608.13723v1)** · ObjectNav
  - Contribution: Target correlation driven node ordering + spatiotemporal Graph-Mamba.
  - Evidence: The abstract states that navigation performance has been improved on AI2-THOR and RoboTHOR, and has been verified by real robot deployment; no verifiable value is given.
  - Reason: Specific usage of graph structure + state space model on ObjectNav.
- **A4 · [DevGRU](https://arxiv.org/abs/2608.18470v1)** · Point target visual navigation (indoor obstacle avoidance)
  - Contribution: Deep guidance + collision-aware trajectory prediction.
  - Evidence: The abstract states that comparing ViNT/NoMaD/NavDP in 9 scenarios, the model size is 1/7 of NavDP and the inference time is 1/17; no standard benchmark value is given.
  - Reason: Worth reading when paying attention to real robot narrow channel obstacle avoidance and reasoning overhead.
- **A5 · [RP1 (Reinforced Planning with Latent World Models) ](https://arxiv.org/abs/2608.18669v1)** · Visual Navigation / Robotic Arm / Operation
  - Contribution: A planner for completely offline learning "how to improve multi-step planning", which can be connected to any pre-trained latent world model.
  - Evidence: The abstract states that the model backbone across two worlds is better than the manual search algorithm, and the world model rollout usage is reduced by 1000 times and is 67 times faster; the values are not split by tasks.
  - Reason: It contains navigation experiments and the modules are pluggable. It is the most valuable planning work for this migration period.
- **B1 · [Remember Smarter](https://arxiv.org/abs/2608.15269v1)** · Long-range robot memory
  - Contribution: visual history compression + hyperbolic experience space, dual-branch plug-and-play.
  - Evidence: The abstract states that the total success rate of LIBERO-Plus after accessing pi0 is 53.6% → 70.6%.
  - Reason: A direct reference structure for long-range navigation history compression.
- **B2 · [Calibrated Predictive Safety](https://arxiv.org/abs/2608.17496v1)** · Heterogeneous robot safe execution
  - Contributions: Action Conditions JEPA Predictive Risk and Schedule + Deterministic Safety Shield.
  - Evidence: The abstract claims superiority to the shield-only baseline in the LIBERO-Long 600-round configuration and reduced collision misses under match recall; real robot experiments remain future work.
  - Reason: Navigation and obstacle avoidance can learn from the division of labor of "learning sequencing + deterministic assurance".
- **B3 · [The Embodiment Gap in Robot Foundation Models](https://arxiv.org/abs/2608.18433v1)** · cross-embodiment Summary
  - Contribution: Proposed a two-axis graph of embodiment gaps and an adaptation workload reporting framework.
  - Evidence: review type, no experimental values.
  - Reason: A framework reference for evaluating reuse costs when changing robot platforms.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. Embodied-Navigator / TAMP-Nav: the only R2R-CE result this week, using VLMs for 2D point selection
{: id="1-embodied-navigator--tamp-nav本期唯一给出-r2r-ce-数字的工作路线是让-vlm-只做它擅长的-2d-选点"}

**Problem.** The abstract states that existing methods stuff VLM into an unnatural action space, are misaligned with its 2D pre-training priors, and suffer from stiff inference scheduling and inefficient memory management.

**Method.** Three-stage mechanism: Point recapitulates navigation as 2D visual cues, VLM only selects pixels, and then projects them into 3D coordinates and passes them to the underlying SLAM controller; Think/Memorize only triggers the thinking chain at key nodes and retains high-fidelity memory, and compresses the rest of the trajectories into lightweight space-time indicators; Align uses GRPO to superimpose global result rewards and fine-grained process rewards for two-level alignment.

**Evidence.** abstract claims 66.2% SR on R2R-CE, requiring only 90k trajectories for training. Note that this is R2R-CE (continuous environment) rather than discrete R2R, and is not directly comparable to the discrete R2R rankings. The abstract does not give other indicators such as SPL and NE, nor does it list the specific values of the control method, so the statement state-of-the-art cannot be independently verified within the entry.

**Value.** has direct value to ground-based VLN: the division of pixel point selection + SLAM controller is highly consistent with the existing navigation stack of the real ground platform, and does not require VLM to directly output low-level actions. The sample efficiency number (90k trajectories) is useful for estimating replication costs.

**Limitations.** relies on the quality of the underlying SLAM controller. The abstract does not explain its performance under odometry drift or dynamic obstacles; no real robot results are given; the title name (Embodied-Navigator) and the method name (TAMP-Nav) are inconsistent, so please pay attention when retrieving and citing.

**Recommendation.** Prioritize intensive reading, focusing on the R2R-CE comparison table and the specific design of the GRPO process rewards; if you want to compare with your own results, be sure to confirm that the other party uses R2R-CE division rather than discrete R2R.

### 2. CondVLN: exposing wrong-branch successes in navigation evaluation
{: id="2-condvln把选错分支从成功率里拆出来是本期对评测口径冲击最大的工作"}

**Problem.** The abstract states that most of the existing VLN evaluations are route-based instructions toward a fixed target, and real instructions often have conditions (if the condition is true, go to A, otherwise go to B); the existing evaluations lack control over branch execution, and cannot distinguish whether the failure is caused by sensing, grounding, navigation, or logical decision-making.

**Method.** programmatically generates conditional instructions, grounding branch conditions on verifiable 3D scene graph predicates, and making controlled changes to branch depth, dependency chain length, spatial composition, evidence observability, and instruction duration. In addition to the standard VLN indicators, two new diagnostic indicators, Branch Selection Accuracy and Conditional Success Rate, are added. We also propose a lightweight neural symbolic branch selection model that separates conditional grounding from navigation execution.

**Evidence.** The summary of states that it contains 11,500+ generation instructions, covering AI2-THOR, Matterport3D, Gibson, ReplicaCAD; four methods, VLN-Zero, NaVid, NaVILA, and Open-Nav are evaluated; the conclusion is that the possible paths of the agent seem reasonable but the branches are inconsistent with the observed scene conditions; the branch selection model improves performance by 2 times. The abstract does not give specific values for each of the four tested methods, and 2x does not indicate which indicator it is based on.

**Value.** The direct value is on the evaluation side: This digest's assessment. If your own system only reports SR/SPL, it is likely to cover up the failure of condition understanding. This benchmark covers MP3D and Gibson, is compatible with common ground-based VLN setups, and can be used as a complementary set to existing benchmarks.

**Limitations.** The instructions are programmatically generated, and the distribution difference from the natural human condition instructions is not stated in the abstract; the four tested methods are all newer VLM-style navigators and do not cover the classic R2R fine-tuning model, so it is unknown whether the conclusion can be extrapolated to the traditional VLN model.

**Recommendation.** It is recommended to read it carefully and consider including it in your own benchmark tracking; focus on whether the generation template of conditional instructions and the definition of Branch Selection Accuracy can be directly transplanted.

### 3. Graph-MambaNav: bringing Graph-Mamba node ordering to ObjectNav
{: id="3-graph-mambanav把-graph-mamba-的节点排序机制引入-objectnav"}

**Problem.** Abstract: The existing graph-based ObjectNav method introduces target awareness at the feature or attention layer, but is still permutation-invariant and lacks a mechanism to explicitly control the order of information propagation, which limits the modeling of target-related importance and long-range dependencies.

**Method.** performs heuristic sorting based on the correlation between objects and targets, allowing more informative objects to be processed later in the sequence to aggregate richer context; node order and edge weights are initialized by common sense object relationships derived from LLM. The spatial module combines local message passing with GraphMamba global selective scanning, and the temporal module performs Mamba sequence modeling on object-level temporal order.

**Evidence.** The abstract states that the navigation performance and generalization on AI2-THOR and RoboTHOR have been improved, and it has been verified by real robot deployment. The abstract does not give any SR/SPL values or comparison methods, so validity cannot be verified within the entry.

**Value.** has reference value for ground ObjectNav: use LLM common sense to initialize the graph prior, and then use sorting to control the propagation order. This combination can be migrated to its own semantic map module.

**Limitations.** There is no verifiable number; AI2-THOR/RoboTHOR is a synthetic indoor scene, which is quite different from the real ground platform; whether the heuristic ranking is true for open vocabulary targets outside the target category is not stated in the abstract.

**Recommendation.** Medium priority tracking, give priority to confirming the absolute values and baseline selection in the experimental table when reading the full text.

### 4. DevGRU: lightweight point-goal navigation for collision avoidance in narrow passages
{: id="4-devgru面向窄通道碰撞问题的轻量点目标导航模型"}

**Problem.** Abstract: The existing visual navigation foundation model is prone to collisions in complex indoor environments (especially structured layouts and narrow passages).

**Method.** A navigation system in which depth maps and point goals are jointly conditioned. The action predictor generates a collision-aware future trajectory. In conjunction with the collision predictor, the action predictor also compensates for the cumulative error in target pose estimation and actively suppresses future offsets.

**Evidence.** The abstract states that comparing the four variants of ViNT, NoMaD, NavDP and ViNT/NoMaD on 9 scenes, the navigation performance is significantly better than ViNT and NoMaD; the model size is 1/7 of NavDP, and the inference time is 1/17. No specific success rate or collision rate values were given, nor was it stated whether the 9 scenarios were simulations or real robots.

**Value.** What is valuable for a real ground platform is the efficiency number: 1/17 of the inference time means a significant increase in control frequency when computing power is limited at the edge.

**Limitations.** relies on depth input and point goals, and does not involve language instructions, so it cannot directly replace the VLN module; the lack of standard benchmark values makes direct comparison difficult; for NavDP, it only reports efficiency advantages but not performance advantages. You need to confirm whether there is any performance loss in the full text.

**Recommendation.** If you pay attention to real robot obstacle avoidance and reasoning overhead, you can read it carefully; as a comparison object for VLN work, its relevance is limited.

## 4. Transferable methods
{: id="四可迁移方法"}

- **latent world model planning: [RP1](https://arxiv.org/abs/2608.18669v1)**
  - Mechanism: Completely offline learning from imaginary rollout to improve the optimizer + evaluator of multi-step plans, which can be hooked to any pre-trained latent world model.
  - Integration point: A high-level path planner for long-range navigation, replacing manual searches.
  - Prerequisites and risks: Although the abstract contains visual navigation experiments, it does not break down the task values; it requires its own latent world model.
- **long-range memory: [Remember Smarter](https://arxiv.org/abs/2608.15269v1)**
  - Mechanism: Two-way space Mamba + causal time Mamba compresses multi-view history, accesses the action-side hidden state through residual cross-attention, and does not change the VLM visual token stream.
  - Integration point: VLN’s observation history encoding to alleviate context expansion under long instructions.
  - Prerequisites and risks: The evidence comes from LIBERO-Plus manipulation tasks (53.6% → 70.6%), and the navigation scenario is not verified.
- **Execution Safety: [Calibrated Predictive Safety](https://arxiv.org/abs/2608.17496v1)**
  - Mechanism: Learning sorting only rearranges feasible candidates, and mandatory guarantees are handed over to deterministic safety shields and fallback ladders.
  - Integration point: Navigation Local obstacle avoidance: VLM proposed path + geometric shield filtering.
  - Prerequisites and risks: Only LIBERO-Long simulation verification, real robot experiment is listed as future work by the author.
- **process reward: [Robo-Dopamine 2.0](https://arxiv.org/abs/2608.15680v1)**
  - Mechanism: History-conditioned pairwise process rewards + signed progress space distinguishing valid progress/robustness/failure/recovery.
  - Access locations: Navigating RL's dense reward design to mitigate sparse success signals.
  - Prerequisites and risks: The values come from manipulation tasks (RoboTwin 86.8%, real robot insertion 71/80), and navigation needs to reconstruct reward semantics.
- **Enhanced during testing: [Reuse Before You Retrieve](https://arxiv.org/abs/2608.17484v1)**
  - Mechanism: The two measurable factors of recoverable margin and retrieval complementarity are used to decide whether to resample or retrieval.
  - Integration point: Freeze deployment-period enhancement decisions for navigation policies.
  - Prerequisites and risks: The evidence is on LIBERO (maximum +21.0 success rate points), the retry semantics of navigation are different, and the fallback cost is higher.
- **cross-embodiment reuse: [The Embodiment Gap](https://arxiv.org/abs/2608.18433v1)**
  - Mechanism: The two-axis graph distinguishes the shareable structure types and the stages that require adaptation, and provides an adaptation workload reporting framework.
  - Integration point: Evaluate the cost of reuse and the parts that need to be redone when replacing the ground platform.
  - Prerequisites and risks: Review nature, no experimental evidence, only an analytical framework.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **Embodied-Navigator / TAMP-Nav** (Continuous Environment VLN · A): See key analysis. [Source](https://arxiv.org/abs/2608.17512v1)
- **CondVLN** (VLN conditional branch evaluation · A): See the key analysis. [Source](https://arxiv.org/abs/2608.17318v1)
- **Graph-MambaNav** (ObjectNav · A): See key analysis. [Source](https://arxiv.org/abs/2608.13723v1)
- **DevGRU** (point target visual navigation · A): See key analysis. [Source](https://arxiv.org/abs/2608.18470v1)
- **Exposing the Long-tail in Embodied Urban Navigation** (Urban Point Target Navigation · A): Automatically annotate metric trajectories and navigation semantics from first-view videos in the wild. Training can explain VLA planning strategies and systematically expose long-tail failure modes. [Source](https://arxiv.org/abs/2608.16476v1)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **RP1** (Learnable Planner · B): See Transferable Methods. [Source](https://arxiv.org/abs/2608.18669v1)
- **Remember Smarter** (Long Range Memory Compression · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.15269v1)
- **Calibrated Predictive Safety** (Predictive Safety · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.17496v1)
- **Robo-Dopamine 2.0** (Process Reward Modeling · B): See Transferable Methods. [Source](https://arxiv.org/abs/2608.15680v1)
- **Reuse Before You Retrieve** (Enhanced Diagnostics While Testing · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.17484v1)
- **The Embodiment Gap** (cross-embodiment overview · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.18433v1)
- **LIBERO-VIFO** (Visual cue following evaluation · C): A benchmark for VLA visual cue following ability and security, based on scene instantiation experiments. [Source](https://arxiv.org/abs/2608.17600v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **GigaBrain-0.7** (Base Model · C): The three-system architecture extends the embodied base model to obtain emergent capabilities. [Source](https://arxiv.org/abs/2608.15875v1)
- **τ_0-VLA** (layered base model · C): Test-time calculation guided by the world model. [Source](https://arxiv.org/abs/2608.16885v1)
- **Teach and Grow** (General Robot Learning · C): Agent-centric architecture. [Source](https://arxiv.org/abs/2608.17209v1)
- **EXIMO** (Policy Exploration · C): Use VLM to guide the exploration of VLA policies. [Source](https://arxiv.org/abs/2608.19891v1)
- **StructRL** (RL·C for streaming VLA): Structured action space exploration. [Source](https://arxiv.org/abs/2608.15139v1)
- **Prism-GRPO** (Policy Optimization Efficiency · C): Split identical result groups to speed up VLA policy optimization. [Source](https://arxiv.org/abs/2608.17423v1)
- **PACE** (Long-range credit allocation · C): Stage progress-aware credit allocation. [Source](https://arxiv.org/abs/2608.15026v1)
- **Q-Learning With World Models** (World Model RL · C): Combined with Q learning while reducing model bias. [Source](https://arxiv.org/abs/2608.17163v1)
- **Don't Drop the BATON** (Long Range Operation · C): Intelligent Asana Subtask Exploration + Transfer Perceptual Memory. [Source](https://arxiv.org/abs/2608.16889v1)
- **Imagining Recovery** (error correction during inference · C): Counterfactual realignment realizes recovery during inference. [Source](https://arxiv.org/abs/2608.14822v1)
- **FabriMAE** (Self-evaluation · C): Self-evaluation of VLA action generation with Markov attention entropy. [Source](https://arxiv.org/abs/2608.16697v1)
- **SparkVLA** (Long Range Operation · C): Stop-aware hierarchical VLA with adaptive action chunking. [Source](https://arxiv.org/abs/2608.16172v1)
- **NebulaVLA** (Dual-band architecture · C): Dual-band VLA introducing boot action. [Source](https://arxiv.org/abs/2608.16503v1)
- **Reflex** (real-time · C): Fast predictive VLA for reaction-critical missions. [Source](https://arxiv.org/abs/2608.14379v1)
- **PhaseLoRA** (Parameter Efficient Fine-tuning · C): Low-rank adaptation of control mechanism conditionalization. [Source](https://arxiv.org/abs/2608.15285v1)
- **Role-Conditioned Sub-Token Routing** (reasoning efficiency · C): Role-conditioned sub-token routing. [Source](https://arxiv.org/abs/2608.18410v1)
- **Algorithm-Architecture Co-Design** (Inference Acceleration · C): Algorithm architecture co-design for speculative reasoning and verification. [Source](https://arxiv.org/abs/2608.15636v1)
- **EcoVLA** (device-edge collaboration · C): Energy-efficient device-edge collaborative reasoning under real-time constraints. [Source](https://arxiv.org/abs/2608.15502v1)
- **GS-VLA** (View Robustness · C): Use Gaussian splashing for view normalization without changing the freezing strategy. [Source](https://arxiv.org/abs/2608.19066v1)
- **EATR-Stereo** (Humanoid Vision · C): Proprioceptive token routing for pairwise stereo evidence. [Source](https://arxiv.org/abs/2608.17453v3)
- **HAF** (Humanoid Full Body Manipulation · C): Hierarchical Action Flow + Spectral Latent Space RL Adaptation to Universal VLA. [Source](https://arxiv.org/abs/2608.16837v1)
- **ViTaR** (Optohaptic·C): Optohaptic residual adaptation for basic VLA manipulation. [Source](https://arxiv.org/abs/2608.15816v1)
- **AdvDex** (dexterous manipulation · C): joint-aligned actions + adversarial learning from human demonstrations. [Source](https://arxiv.org/abs/2608.14028v1)
- **CompCPZ** (Language Guided Operation · C): Preserve multimodal intent. [Source](https://arxiv.org/abs/2608.17717v1)
- **Fine-Tuning VLAs with Self-Demonstrated Generative Control** (Multi-task fine-tuning · C): Self-Demonstrated Generative Control. [Source](https://arxiv.org/abs/2608.19490v1)
- **OrthoSkillVLA** (Continuous Learning · C): Gradient-guided skill subspace adaptation. [Source](https://arxiv.org/abs/2608.19589v1)
- **Bit-Flip Attacks on VLA** (Security · C): Action decoding architecture affects vulnerability to bit-flip attacks. [Source](https://arxiv.org/abs/2608.15475v1)

### 5.4 UAVs, autonomous driving, and less relevant topics
{: id="54-无人机自动驾驶与其他低相关方向"}

- **BrainWAM** (autonomous driving · C): Action space coordination of semantic priors and predictive dynamics. [Source](https://arxiv.org/abs/2608.12854v2)
- **Planning-Oriented End-to-End Autonomous Driving** (Autonomous Driving Review·C): Architecture, evaluation and emerging paradigm review. [Source](https://arxiv.org/abs/2608.20111v1)
- **Plug-and-Play Traffic Element Awareness** (autonomous driving · C): Plug-and-play traffic element awareness. [Source](https://arxiv.org/abs/2608.18035v1)
- **Inference-Time Attention Steering** (autonomous driving VLA · C): Attention guidance during the inference period. [Source](https://arxiv.org/abs/2608.17095v1)
- **SSP** (autonomous driving evaluation · C): Syn2Sim2Phy cross-domain evaluation framework for event matching. [Source](https://arxiv.org/abs/2608.14024v1)
- **ForceU-VLA** (Medical Ultrasound · C): Force-aware ultrasound scanning VLA. [Source](https://arxiv.org/abs/2608.15009v1)
- **US-VLA** (Medical Ultrasound · C): Ultrasound VLA for abdominal scans. [Source](https://arxiv.org/abs/2608.16074v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

- **— · The WeChat public account source was not captured in this issue, only the paper entry** (—):—.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **VLN Evaluation is moving from endpoint success rate to process correctness.** CondVLN introduces branch selection accuracy and conditional success rate, Robo-Dopamine 2.0 constructs a signed progress space to distinguish effective progress/robustness/failure/recovery, and Reuse Before You Retrieve proposes recoverable margin as a measurable diagnostic factor. The three directions are consistent: a single endpoint indicator is not enough to characterize the quality of the strategy.
- **frozen large model + pluggable lightweight module has become the mainstream engineering form.** Remember Smarter, GS-VLA, Reuse Before You Retrieve, and RP1 all emphasize plug-and-play without changing the trunk. This digest's assessment: This reduces the cost of doing incremental experiments on your own navigation stack, and is the most worthy engineering paradigm to follow in this issue.
- **embodiment gap is explicitly problematized.** The Embodiment Gap proposes a cross-embodiment reuse analysis framework. EATR-Stereo and HAF adapt to the humanoid respectively. Calibrated Predictive Safety uses embodiment to embed the conditional world model. The common sign is that the implicit workload of cross-platform migration is beginning to require explicit reporting.

### Research gaps
{: id="研究空白"}

- **Continuous Environment VLN lacks unified and comparable public figures.** The only R2R-CE result in this issue (66.2% SR) is not supported by SPL/NE. The rest of the navigation work uses different scene sets, so direct comparison cannot be made.
- **The evidence for memory compression and process reward comes almost exclusively from manipulation tasks. The values of** Remember Smarter and Robo-Dopamine 2.0 are all obtained on LIBERO/RoboTwin, and the characteristics of long navigation time series, partial observability, and high rollback cost have not been verified.
- **There is no corresponding trained method for understanding conditional and logical instructions.** CondVLN exposed the problem and gave a lightweight branch selector, but there is no article in this issue that solves the conditional instruction grounding from the training side system.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read the full text**: Embodied-Navigator / TAMP-Nav: Check whether the R2R-CE comparison table and SPL/NE are given, and confirm the GRPO process reward design.
- **carefully read and evaluated for inclusion in the review**: CondVLN: Confirm whether the conditional instruction generation template and two branch indicators can be transplanted to the own MP3D/Gibson review.
- **intensively read and evaluated the reproduction**: RP1: Confirm the actual settings and values of the visual navigation subtask, and determine whether it can hook up its own latent world model.

**Medium priority**

- **borrows from the structure**: Remember Smarter’s dual Mamba history compression branch, and performs navigation scenario verification on its own VLN model.
- **Track**: Graph-MambaNav: wait or check the full text value before judging the actual gain on ObjectNav.

**Low priority**

- **Suspended**: The operation, autonomous driving and medical ultrasound work in 5.3 and 5.4 has no direct correlation with ground navigation in this issue.
