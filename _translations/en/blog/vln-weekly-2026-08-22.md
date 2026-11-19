---
layout: post
lang: en
translation_id: vln-weekly-2026-08-22
permalink: /en/vln-weekly-2026-08-22/
source_path: _posts/weekly-reports/2026-08-22-VLN-Weekly.md
source_url: /vln-weekly-2026-08-22/
source_revision_date: 2026-10-24
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly · Issue 1"
date: 2026-08-22
last_modified_at: 2026-10-24
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

- **Among the 46 new works in this issue, only 1 reports R2R-series metrics and 0 report RxR. The only such work is** [Embodied-Navigator/TAMP-Nav](https://arxiv.org/abs/2608.17512v1), whose abstract states 66.2% SR on R2R-CE. The other navigation works use AI2-THOR, RoboTHOR, Matterport3D, Gibson, or self-built real-robot scenes, and none treats the R2R/RxR leaderboards as its main battleground. This digest's assessment: papers whose core selling point is gains on the R2R/RxR leaderboards are becoming scarcer, so comparable numbers on standard benchmarks have to be checked proactively in the arXiv full texts.
- **Navigation makes up a very small share: 5 of the 46 works are mainly about navigation, and 23 primarily concern manipulation.** The arXiv hits this issue are dominated by VLA manipulation (especially the LIBERO family), and new methods for ground-based VLN are in short supply. This digest's assessment: this week is not suited to "direct comparison of VLN methods"; it fits better with evaluation methods and transferable mechanisms.
- **The focus of VLN evaluation is shifting from "did the agent arrive" to "did it follow the right conditions".** [CondVLN](https://arxiv.org/abs/2608.17318v1) points out that an agent can take the wrong logical branch while its path looks reasonable, a failure that standard success rate and path length cannot expose. Implication for ground-based VLN: evaluation protocols built mainly on SR/SPL may overestimate instruction understanding, so it is worth adding conditional-branch instructions to your own evaluation.
- **Several works turn "memory compression" and "learnable planning" into pluggable modules**, and at least one ([RP1](https://arxiv.org/abs/2608.18669v1)) explicitly includes visual navigation experiments in its abstract. This digest's assessment: such modules are the part of this issue most directly transferable to long-horizon ground navigation, and rank above any single manipulation SOTA.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [Embodied-Navigator / TAMP-Nav](https://arxiv.org/abs/2608.17512v1)** · VLN in continuous environments
  - Contribution: Pixel-to-3D action representation, selective reasoning memory, and two-level GRPO alignment.
  - Evidence: The abstract claims 66.2% SR on R2R-CE, trained with 90k trajectories.
  - Why it matters: The only work in this issue reporting R2R metrics, and its sample efficiency comes with clear figures.
- **A2 · [CondVLN（If, Then, Otherwise）](https://arxiv.org/abs/2608.17318v1)** · Evaluation of conditional branching in VLN
  - Contribution: A conditional-instruction benchmark grounded in scene graphs, with branch-specific metrics.
  - Evidence: The abstract claims 11,500+ instructions covering AI2-THOR/MP3D/Gibson/ReplicaCAD, evaluating VLN-Zero, NaVid, NaVILA, and Open-Nav; a neuro-symbolic branch selector improves performance by 2x.
  - Why it matters: It bears directly on existing SR/SPL evaluation protocols and can be reused as your own diagnostic set.
- **A3 · [Graph-MambaNav](https://arxiv.org/abs/2608.13723v1)** · ObjectNav
  - Contribution: Node ordering driven by goal relevance, plus a spatiotemporal Graph-Mamba.
  - Evidence: The abstract states improved navigation performance on AI2-THOR and RoboTHOR, validated by real-robot deployment; no verifiable values are given.
  - Why it matters: A concrete use of graph structure plus a state-space model for ObjectNav.
- **A4 · [DevGRU](https://arxiv.org/abs/2608.18470v1)** · Point-goal visual navigation (indoor obstacle avoidance)
  - Contribution: Depth guidance plus collision-aware trajectory prediction.
  - Evidence: The abstract states that, compared against ViNT/NoMaD/NavDP in 9 scenarios, the model is 1/7 the size of NavDP with 1/17 of its inference time; no standard benchmark values are given.
  - Why it matters: Worth reading if you care about obstacle avoidance in narrow passages on real robots and about inference overhead.
- **A5 · [RP1 (Reinforced Planning with Latent World Models) ](https://arxiv.org/abs/2608.18669v1)** · Visual navigation / robotic arm / manipulation
  - Contribution: A planner that learns, fully offline, "how to improve multi-step planning", and can be attached to any pretrained latent world model.
  - Evidence: The abstract states that across two worlds the model backbone outperforms hand-crafted search algorithms, with world-model rollout usage reduced by 1000 times and a 67 times speedup; values are not broken down by task.
  - Why it matters: It includes navigation experiments and its modules are pluggable, making it the most valuable planning work in this issue to borrow from.
- **B1 · [Remember Smarter](https://arxiv.org/abs/2608.15269v1)** · Long-horizon robot memory
  - Contribution: Visual history compression plus a hyperbolic experience space, as a dual-branch plug-and-play module.
  - Evidence: The abstract states that after being attached to pi0, the overall success rate on LIBERO-Plus goes from 53.6% to 70.6%.
  - Why it matters: A direct structural reference for compressing history in long-horizon navigation.
- **B2 · [Calibrated Predictive Safety](https://arxiv.org/abs/2608.17496v1)** · Safe execution across heterogeneous robots
  - Contribution: Action-conditioned JEPA predicts risk and ranks candidates, with a deterministic safety shield.
  - Evidence: The abstract claims to beat a shield-only baseline in a LIBERO-Long 600-episode setting and to reduce missed collisions at matched recall; real-robot experiments remain future work.
  - Why it matters: Navigation and obstacle avoidance can borrow the division of labor of "learned ranking plus deterministic guarantees".
- **B3 · [The Embodiment Gap in Robot Foundation Models](https://arxiv.org/abs/2608.18433v1)** · Cross-embodiment survey
  - Contribution: Proposes a two-axis graph of embodiment gaps and a framework for reporting adaptation workload.
  - Evidence: A survey, with no experimental values.
  - Why it matters: A framework reference for assessing reuse cost when switching robot platforms.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. Embodied-Navigator / TAMP-Nav
{: id="1-embodied-navigator--tamp-nav本期唯一给出-r2r-ce-数字的工作路线是让-vlm-只做它擅长的-2d-选点"}

**The only R2R-CE result this week, using a VLM for 2D point selection.**

**Problem.** The abstract states that existing methods force the VLM into an unnatural action space that is misaligned with its 2D pretraining priors, and suffer from rigid inference scheduling and inefficient memory management.

**Method.** A three-stage mechanism. Point recasts navigation as 2D visual cues: the VLM only selects pixels, which are then projected into 3D coordinates and handed to the underlying SLAM controller. Think/Memorize triggers chain-of-thought only at key nodes and keeps high-fidelity memory for them, compressing the rest of the trajectory into lightweight spatiotemporal indicators. Align uses GRPO to combine a global outcome reward with fine-grained process rewards for two-level alignment.

**Evidence.** The abstract claims 66.2% SR on R2R-CE with only 90k training trajectories. Note that this is R2R-CE (continuous environments), not discrete R2R, so it is not directly comparable to discrete R2R leaderboards. The abstract gives no other metrics such as SPL and NE and no specific numbers for the compared methods, so the state-of-the-art claim cannot be verified independently from the entry.

**Value.** Direct value for ground-based VLN: the division of labor between pixel selection and a SLAM controller closely matches the existing navigation stack on real ground platforms and does not require the VLM to output low-level actions. The sample-efficiency figure (90k trajectories) is useful for estimating replication cost.

**Limitations.** It depends on the quality of the underlying SLAM controller. The abstract does not describe behavior under odometry drift or dynamic obstacles, and gives no real-robot results. The title name (Embodied-Navigator) and the method name (TAMP-Nav) differ, so take care when searching for and citing it.

**Recommendation.** Read closely, focusing on the R2R-CE comparison table and the specific design of the GRPO process rewards. If you want to compare against your own results, make sure it uses the R2R-CE split rather than discrete R2R.

### 2. CondVLN
{: id="2-condvln把选错分支从成功率里拆出来是本期对评测口径冲击最大的工作"}

**Exposing wrong-branch successes in navigation evaluation.**

**Problem.** The abstract states that most existing VLN evaluations use route-style instructions toward a fixed goal, whereas real instructions often contain conditions (if the condition holds, go to A; otherwise go to B). Existing evaluations lack control over branch execution and cannot tell whether a failure comes from perception, grounding, navigation, or logical decision-making.

**Method.** It generates conditional instructions programmatically, grounds branch conditions in verifiable 3D scene-graph predicates, and varies branch depth, dependency-chain length, spatial composition, evidence observability, and instruction length in a controlled way. Besides standard VLN metrics, it adds two diagnostic metrics, Branch Selection Accuracy and Conditional Success Rate. It also proposes a lightweight neuro-symbolic branch selection model that separates condition grounding from navigation execution.

**Evidence.** The abstract states 11,500+ generated instructions covering AI2-THOR, Matterport3D, Gibson, and ReplicaCAD, and evaluates four methods: VLN-Zero, NaVid, NaVILA, and Open-Nav. Its conclusion is that agents' paths may look reasonable while the branches conflict with the observed scene conditions; the branch selection model improves performance by 2 times. The abstract gives no specific values for the four tested methods, and does not say which metric the 2x refers to.

**Value.** The direct value is on the evaluation side. This digest's assessment: if your own system reports only SR/SPL, it may well be hiding failures of conditional understanding. The benchmark covers MP3D and Gibson, is compatible with common ground-based VLN setups, and can serve as a complementary set to existing benchmarks.

**Limitations.** The instructions are generated programmatically, and the abstract does not say how they differ in distribution from natural human conditional instructions. The four tested methods are all recent VLM-style navigators and do not cover classic fine-tuned R2R models, so it is unknown whether the conclusions extend to traditional VLN models.

**Recommendation.** Read closely and consider adding it to your own benchmark tracking; focus on whether the generation templates for conditional instructions and the definition of Branch Selection Accuracy can be ported directly.

### 3. Graph-MambaNav
{: id="3-graph-mambanav把-graph-mamba-的节点排序机制引入-objectnav"}

**Bringing Graph-Mamba node ordering to ObjectNav.**

**Problem.** From the abstract: existing graph-based ObjectNav methods introduce goal awareness at the feature or attention level but remain permutation-invariant and lack a mechanism for explicitly controlling the order of information propagation, which limits the modeling of goal-related importance and long-range dependencies.

**Method.** Objects are sorted heuristically by their relevance to the goal, so that more informative objects are processed later in the sequence and aggregate richer context; node order and edge weights are initialized from commonsense object relations derived from an LLM. The spatial module combines local message passing with GraphMamba global selective scanning, and the temporal module applies Mamba sequence modeling to the object-level temporal order.

**Evidence.** The abstract states improved navigation performance and generalization on AI2-THOR and RoboTHOR, validated by real-robot deployment. It gives no SR/SPL values or comparison methods, so its validity cannot be checked from the entry.

**Value.** Useful for ground ObjectNav: initialize the graph prior with LLM commonsense, then control the propagation order through sorting. This combination can be ported to your own semantic map module.

**Limitations.** No verifiable numbers; AI2-THOR/RoboTHOR are synthetic indoor scenes that differ considerably from real ground platforms; the abstract does not say whether the heuristic ordering holds for open-vocabulary targets outside the training categories.

**Recommendation.** Track at medium priority; when reading the full text, first confirm the absolute values and baseline choices in the experiment tables.

### 4. DevGRU
{: id="4-devgru面向窄通道碰撞问题的轻量点目标导航模型"}

**Lightweight point-goal navigation with collision avoidance in narrow passages.**

**Problem.** From the abstract: existing visual navigation foundation models are prone to collisions in complex indoor environments, especially structured layouts and narrow passages.

**Method.** A navigation system jointly conditioned on depth maps and point goals. The action predictor generates collision-aware future trajectories. Together with the collision predictor, it also compensates for the accumulated error in goal-pose estimation and actively suppresses future drift.

**Evidence.** The abstract states that, compared against ViNT, NoMaD, NavDP, and variants of ViNT/NoMaD across 9 scenes, navigation performance is significantly better than ViNT and NoMaD; the model is 1/7 the size of NavDP with 1/17 of its inference time. No specific success rate or collision rate values are given, nor whether the 9 scenes are simulated or real.

**Value.** For real ground platforms the efficiency figure is what counts: 1/17 of the inference time means a markedly higher control frequency when edge compute is limited.

**Limitations.** It relies on depth input and point goals and involves no language instructions, so it cannot directly replace a VLN module. The lack of standard benchmark values makes direct comparison hard, and relative to NavDP only an efficiency advantage is reported, not a performance advantage; check the full text for any performance loss.

**Recommendation.** Read closely if you care about obstacle avoidance on real robots and inference overhead; as a comparison for VLN work its relevance is limited.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Latent world model planning: [RP1](https://arxiv.org/abs/2608.18669v1)**
  - Mechanism: Learns, fully offline from imagined rollouts, an optimizer plus evaluator that improves multi-step plans, and can be attached to any pretrained latent world model.
  - Integration point: A high-level path planner for long-horizon navigation, replacing hand-written search.
  - Prerequisites and risks: Although the abstract includes visual navigation experiments, it does not break down task-level values; you need your own latent world model.
- **Long-horizon memory: [Remember Smarter](https://arxiv.org/abs/2608.15269v1)**
  - Mechanism: A bidirectional spatial Mamba plus a causal temporal Mamba compress multi-view history, which enters the action-side hidden state through residual cross-attention without changing the VLM's visual token stream.
  - Integration point: Encoding VLN observation history, to alleviate context growth under long instructions.
  - Prerequisites and risks: The evidence comes from LIBERO-Plus manipulation tasks (53.6% → 70.6%); navigation scenarios are unverified.
- **Execution safety: [Calibrated Predictive Safety](https://arxiv.org/abs/2608.17496v1)**
  - Mechanism: Learned ranking only reorders feasible candidates, while hard guarantees are left to a deterministic safety shield and a fallback ladder.
  - Integration point: Local obstacle avoidance in navigation: paths proposed by a VLM, filtered by a geometric shield.
  - Prerequisites and risks: Verified only in LIBERO-Long simulation; the authors list real-robot experiments as future work.
- **Process reward: [Robo-Dopamine 2.0](https://arxiv.org/abs/2608.15680v1)**
  - Mechanism: History-conditioned pairwise process rewards plus a signed progress space that distinguishes effective progress, robustness, failure, and recovery.
  - Integration point: Dense reward design for navigation RL, to mitigate sparse success signals.
  - Prerequisites and risks: The values come from manipulation tasks (RoboTwin 86.8%, real-robot insertion 71/80); navigation needs reward semantics to be rebuilt.
- **Test-time enhancement: [Reuse Before You Retrieve](https://arxiv.org/abs/2608.17484v1)**
  - Mechanism: Two measurable factors, recoverable margin and retrieval complementarity, decide whether to resample or to retrieve.
  - Integration point: Deployment-time enhancement decisions for a frozen navigation policy.
  - Prerequisites and risks: The evidence is on LIBERO (up to +21.0 success-rate points); retry semantics differ in navigation, and fallback costs are higher.
- **Cross-embodiment reuse: [The Embodiment Gap](https://arxiv.org/abs/2608.18433v1)**
  - Mechanism: A two-axis graph separates the structure types that can be shared from the stages that need adaptation, and provides a framework for reporting adaptation workload.
  - Integration point: Assessing reuse cost and the parts that must be redone when swapping ground platforms.
  - Prerequisites and risks: A survey with no experimental evidence, only an analytical framework.

## 5. Research roundup by category
{: id="五分类速览"}

Relevance labels: A means directly relevant to ground navigation research, B means the mechanism is transferable, and C is for field observation only.

### 5.1 VLN and ObjectNav
{: id="51-vln-与-objectnav"}

- **Embodied-Navigator / TAMP-Nav** (VLN in continuous environments · A): See key analysis. [Source](https://arxiv.org/abs/2608.17512v1)
- **CondVLN** (conditional-branch VLN evaluation · A): See the key analysis. [Source](https://arxiv.org/abs/2608.17318v1)
- **Graph-MambaNav** (ObjectNav · A): See key analysis. [Source](https://arxiv.org/abs/2608.13723v1)
- **DevGRU** (point-goal visual navigation · A): See key analysis. [Source](https://arxiv.org/abs/2608.18470v1)
- **Exposing the Long-tail in Embodied Urban Navigation** (urban point-goal navigation · A): Automatically annotates metric trajectories and navigation semantics from in-the-wild first-person videos; training on them yields interpretable VLA planning strategies and systematically exposes long-tail failure modes. [Source](https://arxiv.org/abs/2608.16476v1)

### 5.2 Memory, planning and evaluation
{: id="52-记忆规划与评测"}

- **RP1** (learnable planner · B): See Transferable Methods. [Source](https://arxiv.org/abs/2608.18669v1)
- **Remember Smarter** (long-horizon memory compression · B): See transferable methods. [Source](https://arxiv.org/abs/2608.15269v1)
- **Calibrated Predictive Safety** (Predictive Safety · B): See transferable methods. [Source](https://arxiv.org/abs/2608.17496v1)
- **Robo-Dopamine 2.0** (process reward modeling · B): See Transferable Methods. [Source](https://arxiv.org/abs/2608.15680v1)
- **Reuse Before You Retrieve** (test-time enhancement diagnostics · B): See transferable methods. [Source](https://arxiv.org/abs/2608.17484v1)
- **The Embodiment Gap** (cross-embodiment overview · B): See transferable methods. [Source](https://arxiv.org/abs/2608.18433v1)
- **LIBERO-VIFO** (visual-cue following evaluation · C): A benchmark of VLA visual-cue following ability and safety, based on scene-instantiated experiments. [Source](https://arxiv.org/abs/2608.17600v1)

### 5.3 VLA and manipulation
{: id="53-具身-vla--移动操作"}

- **GigaBrain-0.7** (foundation model · C): A three-system architecture that extends an embodied foundation model to obtain emergent capabilities. [Source](https://arxiv.org/abs/2608.15875v1)
- **τ_0-VLA** (hierarchical foundation model · C): Test-time computation guided by a world model. [Source](https://arxiv.org/abs/2608.16885v1)
- **Teach and Grow** (general robot learning · C): An agent-centric architecture. [Source](https://arxiv.org/abs/2608.17209v1)
- **EXIMO** (policy exploration · C): Uses a VLM to guide the exploration of VLA policies. [Source](https://arxiv.org/abs/2608.19891v1)
- **StructRL** (RL for streaming VLA · C): Structured exploration of the action space. [Source](https://arxiv.org/abs/2608.15139v1)
- **Prism-GRPO** (policy optimization efficiency · C): Splits groups with identical outcomes to speed up VLA policy optimization. [Source](https://arxiv.org/abs/2608.17423v1)
- **PACE** (long-horizon credit assignment · C): Credit assignment aware of stage progress. [Source](https://arxiv.org/abs/2608.15026v1)
- **Q-Learning With World Models** (world-model RL · C): Combines world models with Q-learning while reducing model bias. [Source](https://arxiv.org/abs/2608.17163v1)
- **Don't Drop the BATON** (long-horizon manipulation · C): Intelligent subtask exploration plus transferable perceptual memory. [Source](https://arxiv.org/abs/2608.16889v1)
- **Imagining Recovery** (inference-time error correction · C): Counterfactual realignment enables recovery at inference time. [Source](https://arxiv.org/abs/2608.14822v1)
- **FabriMAE** (self-evaluation · C): Self-evaluates VLA action generation using Markov attention entropy. [Source](https://arxiv.org/abs/2608.16697v1)
- **SparkVLA** (long-horizon manipulation · C): A stop-aware hierarchical VLA with adaptive action chunking. [Source](https://arxiv.org/abs/2608.16172v1)
- **NebulaVLA** (dual-band architecture · C): A dual-band VLA that introduces bootstrap actions. [Source](https://arxiv.org/abs/2608.16503v1)
- **Reflex** (real-time · C): A fast predictive VLA for reaction-critical tasks. [Source](https://arxiv.org/abs/2608.14379v1)
- **PhaseLoRA** (parameter-efficient fine-tuning · C): Low-rank adaptation conditioned on the control mechanism. [Source](https://arxiv.org/abs/2608.15285v1)
- **Role-Conditioned Sub-Token Routing** (inference efficiency · C): Role-conditioned sub-token routing. [Source](https://arxiv.org/abs/2608.18410v1)
- **Algorithm-Architecture Co-Design** (inference acceleration · C): Algorithm-architecture co-design for speculative inference and verification. [Source](https://arxiv.org/abs/2608.15636v1)
- **EcoVLA** (device-edge collaboration · C): Energy-efficient device-edge collaborative inference under real-time constraints. [Source](https://arxiv.org/abs/2608.15502v1)
- **GS-VLA** (view robustness · C): Uses Gaussian splatting for view normalization without changing the frozen policy. [Source](https://arxiv.org/abs/2608.19066v1)
- **EATR-Stereo** (humanoid vision · C): Proprioceptive token routing for stereo-pair evidence. [Source](https://arxiv.org/abs/2608.17453v3)
- **HAF** (humanoid whole-body manipulation · C): Hierarchical action flow plus RL in a spectral latent space to adapt a general-purpose VLA. [Source](https://arxiv.org/abs/2608.16837v1)
- **ViTaR** (visuo-tactile · C): Visuo-tactile residual adaptation for foundation-VLA manipulation. [Source](https://arxiv.org/abs/2608.15816v1)
- **AdvDex** (dexterous manipulation · C): Joint-aligned actions plus adversarial learning from human demonstrations. [Source](https://arxiv.org/abs/2608.14028v1)
- **CompCPZ** (language-guided manipulation · C): Preserves multimodal intent. [Source](https://arxiv.org/abs/2608.17717v1)
- **Fine-Tuning VLAs with Self-Demonstrated Generative Control** (multi-task fine-tuning · C): Self-Demonstrated Generative Control. [Source](https://arxiv.org/abs/2608.19490v1)
- **OrthoSkillVLA** (continual learning · C): Gradient-guided adaptation of skill subspaces. [Source](https://arxiv.org/abs/2608.19589v1)
- **Bit-Flip Attacks on VLA** (security · C): The action decoding architecture affects vulnerability to bit-flip attacks. [Source](https://arxiv.org/abs/2608.15475v1)

### 5.4 UAVs and other
{: id="54-无人机与其他"}

- **BrainWAM** (autonomous driving · C): Coordinates semantic priors and predictive dynamics in the action space. [Source](https://arxiv.org/abs/2608.12854v2)
- **Planning-Oriented End-to-End Autonomous Driving** (autonomous driving survey · C): A survey of architectures, evaluation, and emerging paradigms. [Source](https://arxiv.org/abs/2608.20111v1)
- **Plug-and-Play Traffic Element Awareness** (autonomous driving · C): Plug-and-play traffic element awareness. [Source](https://arxiv.org/abs/2608.18035v1)
- **Inference-Time Attention Steering** (autonomous driving VLA · C): Attention steering at inference time. [Source](https://arxiv.org/abs/2608.17095v1)
- **SSP** (autonomous driving evaluation · C): A Syn2Sim2Phy cross-domain evaluation framework based on event matching. [Source](https://arxiv.org/abs/2608.14024v1)
- **ForceU-VLA** (medical ultrasound · C): A force-aware VLA for ultrasound scanning. [Source](https://arxiv.org/abs/2608.15009v1)
- **US-VLA** (medical ultrasound · C): An ultrasound VLA for abdominal scans. [Source](https://arxiv.org/abs/2608.16074v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

- **— · No WeChat source was captured in this issue, only paper entries** (—):—.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **VLN evaluation is moving from endpoint success rate to process correctness.** CondVLN introduces branch selection accuracy and conditional success rate, Robo-Dopamine 2.0 builds a signed progress space that distinguishes effective progress, robustness, failure, and recovery, and Reuse Before You Retrieve proposes recoverable margin as a measurable diagnostic factor. The three point the same way: a single endpoint metric is not enough to characterize the quality of a policy.
- **Frozen large model plus pluggable lightweight module has become the mainstream engineering form.** Remember Smarter, GS-VLA, Reuse Before You Retrieve, and RP1 all emphasize plug-and-play use without changing the backbone. This digest's assessment: this lowers the cost of running incremental experiments on your own navigation stack, and is the engineering paradigm most worth following in this issue.
- **The embodiment gap is being made an explicit problem.** The Embodiment Gap proposes an analysis framework for cross-embodiment reuse. EATR-Stereo and HAF each adapt to humanoids. Calibrated Predictive Safety embeds embodiment into a conditional world model. The common sign is that the implicit workload of cross-platform transfer is starting to require explicit reporting.

### Research gaps
{: id="研究空白"}

- **VLN in continuous environments lacks unified, comparable public figures.** The only R2R-CE result in this issue (66.2% SR) comes without SPL/NE. The other navigation works use different scene sets, so no direct comparison is possible.
- **Evidence for memory compression and process rewards comes almost exclusively from manipulation tasks.** The values for Remember Smarter and Robo-Dopamine 2.0 are all obtained on LIBERO/RoboTwin; the properties of long navigation sequences, partial observability, and costly rollbacks remain unverified.
- **There is no trained method for understanding conditional and logical instructions.** CondVLN exposes the problem and provides a lightweight branch selector, but no work in this issue tackles conditional-instruction grounding systematically from the training side.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read the full text**: Embodied-Navigator / TAMP-Nav: check whether the R2R-CE comparison table and SPL/NE are given, and confirm the design of the GRPO process rewards.
- **Read closely and assess for inclusion in your evaluation**: CondVLN: confirm whether the conditional-instruction generation templates and the two branch metrics can be ported to your own MP3D/Gibson evaluation.
- **Read closely and assess for reproduction**: RP1: confirm the actual settings and values of the visual navigation subtask, and determine whether it can be attached to your own latent world model.

**Medium priority**

- **Borrow the structure**: Remember Smarter's dual-Mamba history compression branch, and validate it in navigation scenarios on your own VLN model.
- **Track**: Graph-MambaNav: wait for or check the full-text values before judging its actual gain on ObjectNav.

**Low priority**

- **Paused**: The manipulation, autonomous driving, and medical ultrasound works in 5.3 and 5.4 have no direct relevance to ground navigation in this issue.
