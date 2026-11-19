---
layout: post
lang: en
translation_id: vln-weekly-2026-08-15
permalink: /en/vln-weekly-2026-08-15/
source_path: _posts/weekly-reports/2026-08-15-VLN-Weekly.md
source_url: /vln-weekly-2026-08-15/
source_revision_date: 2026-09-27
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly (2026-08-01 to 2026-08-15)"
date: 2026-08-15
period_start: 2026-08-01
period_end: 2026-08-15
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
published: false
excerpt: "Ground-navigation memory is moving from maps within a single episode to persistent structures reused across episodes: SSTG-Nav, LifelongCrossNav, and SAIN independently point in this direction. SC²-WM, WNM-3D, and Route2Step move VLN-CE toward closed-loop execution with explicit correction points. For cross-embodiment navigation, the key design question is which intermediate interface connects semantics to the robot embodiment."
---

* Contents
{:toc}

## 1. Key conclusions
{: id="一本期结论"}

- **Memory for ground navigation is shifting from a "map within a single episode" to a "persistent structure reused across episodes".** [SSTG-Nav](https://arxiv.org/abs/2608.00527v1) (a metric semantic topological map built once and reused many times), [LifelongCrossNav](https://arxiv.org/abs/2608.07079v1) (shared sparse 3D semantic voxel memory for multi-goal, cross-floor navigation) and [SAIN](https://arxiv.org/abs/2608.09196v1) (compiling dialogue answers into persistent spatial and object states) are independent of one another, yet all solve the same problem: keeping what was learned in one navigation available for the next query. This digest's assessment: this is the clearest directional signal for ground-based VLN in this issue: evaluation protocols for long-term resident robots (homes, hospitals, office buildings) are expanding from "single-instruction success rate" to "information reuse rate under sequential queries".
- **The execution paradigm of VLN-CE is changing from open-loop to closed-loop, with error-correction points explicitly separated.** [SC²-WM](https://arxiv.org/abs/2608.07548v1) uses world-model lookahead to make state-level plan corrections before actions are executed, [WNM-3D](https://arxiv.org/abs/2608.07267v1) conditions the joint generation of future views and actions on geometry-aware representations, [Route2Step](https://arxiv.org/abs/2608.03143v1) splits "progress-tracking errors" and "execution errors" into two separately supervised modules. The three share one premise: next-action supervision alone cannot tell whether the agent executed a step wrongly or misunderstood it.
- **"Which intermediate interface should connect semantics to the embodiment" has become the core design variable of cross-embodiment navigation.** [CrossTracer](https://arxiv.org/abs/2608.06688v1) uses normalized image-plane waypoints as a unified interface and then applies embodiment-conditioned residual correction; [HumanoidVLN](https://arxiv.org/abs/2608.12860v1) points out, from the evaluation side, that wheeled benchmarks hide the constraints of bipedal locomotion and the camera shake it causes. This digest's assessment: cross-embodiment is one of the few sub-directions in this issue with progress on both the "method" and "benchmark" sides.
- Much of the work on the **VLA side consists of runtime monitoring and memory mechanisms that can be transferred to navigation, not manipulation-specific techniques.** [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1) shows that task progress can be read from the residual stream with a linear probe and used for label-free OOD detection, [Continue or Replan?](https://arxiv.org/abs/2608.03483v1) turns replanning opportunities into learnable decisions, [SAFECAST](https://arxiv.org/abs/2608.04246v1) / [GUARD](https://arxiv.org/abs/2608.04510v1) provide failure detection at deployment time. Plugging these mechanisms into VLN-CE costs less than retraining navigation policies.
- **Coverage boundaries:** New arXiv entries in this issue hit the crawl limit of 100, and the earliest ones only reach back to 2026-08-01; submissions on 07-31 and earlier appear in the previous report and are not repeated here. Of the 14 WeChat entries, 2 are non-paper items and the text of 1 failed to be fetched. There are 114 original entries in total; after cross-source merging into 5 groups, there are 109 independent works (107 papers + 2 news items).

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [SSTG-Nav](https://arxiv.org/abs/2608.00527v1)** · Reusable indoor ObjectNav, including real-robot deployment
  - Contribution: Untargeted mapping builds a metrically grounded semantic topological map; subsequent queries only need retrieval and lightweight planning.
  - Evidence: The WeChat interpretation reports a 97.5% success rate on a simulation benchmark and completed ROS2 deployment on a physical robot; the entry does not name the benchmark or the comparison methods.
  - Why it matters: The only ground navigation work that proposes a "reuse paradigm + evaluation protocol + real-robot implementation" together.
- **A2 · [LifelongCrossNav](https://arxiv.org/abs/2608.07079v1)** · Multi-goal sequential ObjectNav in unknown multi-floor indoor environments
  - Contribution: Shared sparse 3D semantic voxel memory plus stair-specific perception and cross-floor traversal; proposes the HM3D-MFMON benchmark.
  - Evidence: The abstract is truncated at the baseline description and gives no verifiable values.
  - Why it matters: Merges the "persistent memory" and "cross-floor" lines, previously handled separately, and comes with its own benchmark.
- **A3 · [WNM-3D](https://arxiv.org/abs/2608.07267v1)** · VLN in continuous environments (VLN-CE)
  - Contribution: A frozen feed-forward geometry encoder plus a 3D Scene-to-Token Adapter converts monocular RGB history into a fixed-length prefix that conditions a world-action DiT.
  - Evidence: The abstract is truncated at the method description and gives no verifiable values.
  - Why it matters: Among the world-model works in this issue, this has the most complete mechanism description and is the closest counterpart to ground-based VLN-CE.
- **A4 · [Route2Step](https://arxiv.org/abs/2608.03143v1)** · Instruction-following VLN
  - Contribution: An explicit step-level interface decouples semantic progress tracking from action generation; E-SPA can supervise progress status without manual temporal annotation.
  - Evidence: The abstract is truncated at the description of the supervision process and gives no verifiable values.
  - Why it matters: Targets the persistent evaluation and training problem of corrective action labels masking progress errors.
- **A5 · [SAP-Nav](https://arxiv.org/abs/2608.12707v1)** · Hierarchical open-vocabulary object navigation (OVON)
  - Contribution: Builds a queryable spatial-semantic representation online with active viewpoint verification; zero-shot, with no precomputed scene graph.
  - Evidence: The abstract is truncated in the experimental section and gives no verifiable values.
  - Why it matters: The active perception loop of "if the evidence is insufficient, change viewpoint and confirm again" can be grafted directly onto existing ObjectNav stacks.
- **A6 · [HumanoidVLN](https://arxiv.org/abs/2608.12860v1)** · Humanoid VLN simulation and benchmark
  - Contribution: A physically grounded platform on Isaac Sim, with an RL locomotion policy and replaceable PD/MPC path tracking.
  - Evidence: 4 robots (Unitree G1, H1, Internal-A/B), 10–12 DoF in the lower limbs, heights of 1.17–1.80 m; declared compatible with NaVILA, DualVLN, StreamVLN, and JanusVLN; scenes require a navigable area of more than 100 m².
  - Why it matters: For evaluating the cross-embodiment robustness of in-house policies, this is the only directly usable physically grounded platform in this issue.
- **A7 · [SAIN](https://arxiv.org/abs/2608.09196v1)** · Interactive Instance Goal Navigation (IIGN)
  - Contribution: Compiles oracle answers into goal evidence, corridor memory, and candidate object labels, and stores them in structured memory for a unified policy to consume.
  - Evidence: On the VL-LN IIGN benchmark, SR goes 20.2 → 25.4 and SPL 13.07 → 14.17 (compared with the "strongest reported method"; the abstract does not name it).
  - Why it matters: One of the few ground navigation works in this issue with complete, verifiable numbers; the low absolute success rate also shows the task is far from solved.
- **A8 · [Embodied Agents Take Control](https://arxiv.org/abs/2607.26148)** · Zero-shot VLN-CE
  - Contribution: Directly reuses a general-purpose coding-agent framework with only monocular RGB and 4 basic actions, and no map, depth, panorama, or waypoint modules.
  - Evidence: The WeChat interpretation reports zero-shot R2R-CE success rates of 68.3%–78% and claims to surpass benchmark policies trained at scale; the entry gives no item-by-item comparison table.
  - Why it matters: If the conclusion holds, it changes the judgment on "whether continued investment in dedicated navigation training is worthwhile", so it deserves to be falsified first.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. SSTG-Nav: map once, search repeatedly
{: id="1-sstg-nav把-objectnav-从每次重探索改写为一次测绘多次检索"}

**Problem.** Mainstream ObjectNav explores in a single pass: every instruction requires rescanning the environment, and for long-term stationed robots, historical observations are discarded entirely. There is also a failure mode of "the object is recognized but cannot be reached": the camera's observation point is not the same as a legal stopping point for the robot.

**Method.** According to the WeChat interpretation, it has three parts: metric grounding (back-projecting 2D detections into 3D and offsetting by 0.8 m to generate a safe docking area), source-aware 3D soft fusion (aggregating cross-view evidence and filtering false detections by confidence), and multi-candidate fault self-healing. The query phase performs only retrieval and lightweight path planning, with no repeated mapping.

**Evidence.** The WeChat interpretation reports a 97.5% success rate on a simulation benchmark and completed ROS2 real-robot deployment. **This digest's assessment: this number cannot be used for direct comparison yet.** The entry gives no benchmark name, scene split, or baseline methods. The authors also introduce 3 evaluation protocols that isolate information boundaries to separate the effects of "map geometric coverage" and "semantic recognition errors"; this protocol design deserves more attention than the success rate.

**Value.** It speaks directly to the real working conditions of long-term home and office robots. The geometric conversion from "observation point → reachable stopping point" is a commonly missing link in existing semantic-map approaches (VLMaps, ConceptFusion, etc.).

**Limitations.** It relies on a static environment, on the premise that layout and object positions do not change frequently. The entry does not explain map invalidation or update strategies after objects move or the scene changes; the very high 97.5% also suggests the benchmark may be biased.

**Recommendation.** Read its **evaluation protocol** closely first, rather than the success rate; [Project page](https://daojiepeng.github.io/SSTG-Nav) and [Code](https://github.com/DaojiePENG/sstg-nav-bench) are provided, and the 0.8 m docking offset can be ablated independently on your own ObjectNav stack.

### 2. LifelongCrossNav: combining persistent memory with cross-floor navigation
{: id="2-lifelongcrossnav持久记忆与跨楼层被合并成同一个问题"}

**Problem.** ObjectNav's persistent memory (sequential multi-goal queries) and cross-floor navigation used to be handled as two separate lines of work, and the single-goal, single-floor assumption does not match real homes and office buildings.

**Method.** In each episode the agent receives an ordered sequence of object-goal queries and continuously maintains a shared sparse 3D semantic voxel memory, incrementally accumulating geometric structure, traversability, and vision-language features; subsequent queries retrieve directly without rebuilding the map. The cross-floor part consists of support-surface-aware 3D traversability mapping, stair-specific perception, and direction-aware stair traversal. A unified policy coordinates frontier exploration, real-time and historical POI retrieval, stair navigation, and goal approach on the same floor.

**Evidence.** It proposes HM3D-MFMON, a benchmark for sequential multi-floor multi-goal navigation built on HM3D scenes. **The abstract is truncated at the benchmark description, and this issue's entry gives no verifiable success rate or SPL values.**

**Value.** "Whether memory can reduce the exploration cost of subsequent queries" is a directly quantifiable metric, closer to long-term deployment than single-goal success rate; stair perception is a necessary module for carrying simulation conclusions over to real homes.

**Limitations.** The abstract does not cover memory growth and long-term drift of sparse voxels + VL features; cross-floor traversability depends heavily on depth and geometry quality, and the risk of transfer to real robots is unknown.

**Recommendation.** Track whether HM3D-MFMON is released; if it is, compare it with SSTG-Nav's reuse protocol to determine whether "in-episode memory" and "cross-episode memory" metrics are interchangeable.

### 3. WNM-3D: adding geometric conditioning to a world-action model
{: id="3-wnm-3d给世界-动作模型补上几何条件"}

**Problem.** VLN systems increasingly convert pretrained VLMs into VLAs that output actions directly. They have strong semantics but do not explicitly model "how observations should evolve under predicted actions"; existing continuous-VLN world-action models (WAM) do not condition on geometry-aware representations inferred from history when jointly generating future views and actions.

**Method.** A frozen feed-forward geometry encoder extracts geometry-aware representations from the monocular first-person RGB history, and a trainable 3D Scene-to-Token Adapter converts them into a fixed-length prefix in the token space of the world-action Diffusion Transformer. Through block-causal attention, this prefix conditions each future "video-action" block and provides shared geometric context.

**Evidence.** **The abstract is truncated here. This issue's entry provides no dataset, metric, or comparison values.** The text fetched for the WeChat interpretation of the same paper (2026-08-10) is empty, so cross-validation is not possible.

**Value.** Monocular RGB input plus a frozen geometry encoder has the lowest sensor requirements for a ground platform; injecting persistent scene context as a fixed-length prefix is more computationally efficient than frame-by-frame geometry fusion.

**Limitations.** No public values yet, so it can only be treated as a candidate mechanism, not a verified solution; the entry does not address the tension between the inference overhead of a diffusion world-action model and closed-loop control frequency.

**Recommendation.** Review the experiments once the full text is available. At the mechanism level, the interface design of "frozen geometry encoder + lightweight Adapter producing a token prefix" is worth borrowing first, as it is orthogonal to existing VLN-CE policies.

### 4. Route2Step: supervising execution errors and progress errors separately
{: id="4-route2step把走错了和理解错了分开监督"}

**Problem.** VLM-based navigators usually supervise both "progress tracking" and "step execution" with next-action prediction alone. When the agent strays from the route, a corrective action label can resume the next move, but it cannot indicate whether the agent picked the wrong sub-instruction or failed to execute the right one, so the agent keeps deciding from a wrong progress state.

**Method.** The instruction analysis module M_IA predicts step-level status from the global instruction and visual history; the action generation module M_AG generates local action chunks from that status and recent observations; the E-SPA step alignment procedure supervises progress status without manual temporal annotation.

**Evidence.** **The abstract is truncated at the description of the supervision process and provides no verifiable values.**

**Value.** This is the work in this issue closest to engineering diagnosis needs: the explicit progress state is itself an observable quantity that can be used directly for failure attribution, early stopping, and triggering manual takeover, so its value is not limited to success rate.

**Limitations.** The explicit interface introduces error-propagation risk: M_IA mistakes will systematically contaminate downstream actions. The quality of the label-free alignment (E-SPA) sets the overall upper bound, and the entry gives no alignment accuracy.

**Recommendation.** Prioritize its **interface definition and E-SPA alignment procedure**; even without adopting the full framework, an auxiliary head that "outputs the current sub-instruction index" can be added to an existing policy.

### 5. SAP-Nav: changing viewpoint to confirm uncertain observations
{: id="5-sap-nav把看不清就换个角度再确认写进策略"}

**Problem.** Hierarchical OVON requires following free-form instructions that may specify targets via scene-level, room-level, zone-level, or instance-level cues. Under partial observation there are two conflicting requirements: spatial grounding needs persistent environment-level evidence, while target verification needs clear, discriminable views of candidates.

**Method.** It incrementally builds a queryable spatial-semantic representation from actively acquired room views, so spatial-semantic queries can be issued from any explored location. Active Viewpoint Verification assesses whether the current observation is sufficient evidence; if not, the agent first moves to a more informative viewpoint and then verifies candidates against category and attribute constraints.

**Evidence.** It is declared fully online and zero-shot, requiring no task-specific training or precomputed scene maps, and supports both hierarchical and standard category-level OVON. **The abstract is truncated in the experimental part and provides no verifiable values.**

**Value.** Active viewpoint verification targets a failure type that accounts for a high share of ObjectNav errors: long-range misjudgments that lead to early termination. The module can be inserted into an existing zero-shot navigation pipeline as an independent component.

**Limitations.** Changing viewpoints adds path overhead, and the abstract does not explain the cost in terms of SPL; the criterion for "whether the evidence is sufficient" relies on VLM confidence, which may itself be an unreliable signal.

**Recommendation.** Read it alongside SSTG-Nav's multi-view evidence fusion: both deal with "unreliable single-shot recognition", SSTG-Nav through offline multi-view fusion and SAP-Nav through online active repositioning. The trade-off is worth quantifying.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Manipulation VLA interpretability: [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1)**
  - Mechanism: Task progress (normalized remaining time) can be read out linearly from the π0.5 residual stream with a single linear probe, and used as a label-free OOD detector to identify progress stalls.
  - Integration point: Runtime progress monitoring and stuck detection for navigation policies, replacing hand-set timeout thresholds.
  - Prerequisites and risks: The report states that the probe cannot effectively steer the policy and can only read, not control; the definition of "progress" for navigation (path completion vs. sub-instruction index) differs from manipulation and needs to be re-verified.
- **Manipulation VLA execution scheduling: [Continue or Replan? (BCP)](https://arxiv.org/abs/2608.03483v1)**
  - Mechanism: Replaces the fixed execution horizon with a sequence of "continue / replan" Bernoulli decisions; the base policy stays frozen, making it plug-and-play.
  - Integration point: Replanning timing for action chunks in VLN-CE; most current implementations replan after a fixed number of steps, regardless of key turning points.
  - Prerequisites and risks: The optimal horizon cannot be observed directly, so its supervision must be rebuilt on navigation data.
- **Safety at deployment: [SAFECAST](https://arxiv.org/abs/2608.04246v1), [GUARD](https://arxiv.org/abs/2608.04510v1)**
  - Mechanism: The former uses contrastive-set perturbation to improve the training and calibration of latent-state risk probes; the latter measures how strongly actions are grounded in vision-language evidence by ablating KV cache entries.
  - Integration point: Navigation failure detection and takeover triggering; GUARD does not modify the pretrained policy, so the integration cost is low.
  - Prerequisites and risks: Both are validated on manipulation benchmarks (LIBERO / DROID / SimplerEnv) and do not cover navigation failure modes (circling, floor confusion, early stopping).
- **Long-term memory: [AtlasVLA](https://arxiv.org/abs/2608.06729v1)、[Skills in Weights, Memory in Code (HyMeS)](https://arxiv.org/abs/2608.09410v1)**
  - Mechanism: The former uses a persistent 4D voxel-hashed world state plus an egocentric working-state dual memory to address "objects are forgotten once they leave view"; the latter lets a coding agent take over memory management through an executable heuristic system, while the underlying policy remains Markov.
  - Integration point: Corresponds to "looking back and not finding the object I just passed" in navigation; HyMeS's division of labor suits keeping map and memory logic on the auditable code side.
  - Prerequisites and risks: AtlasVLA targets wrist-camera manipulation scenes, and how voxel hashing performs as memory at room scale is unknown; HyMeS relies on the agent's iterative feedback loop, and its real-time performance is questionable.
- **3D vision-language efficiency: [HiSC](https://arxiv.org/abs/2608.04610v1), [CoverPrune](https://arxiv.org/abs/2608.13226v1), [3DZip](https://arxiv.org/abs/2608.01185v1)**
  - Mechanism: Three complementary 3D token compression ideas: hierarchical clustering with spatial graph merging (training-free), maintaining evidence coverage in an optimal-transport sense, and voxelization plus feature-diversity anchor selection.
  - Integration point: The computational bottleneck of semantic-map queries and scene-level VLM inference, especially when the persistent map grows with exploration.
  - Prerequisites and risks: Both are validated on 3D QA and scene understanding. Navigation needs traversability and goal localization rather than QA accuracy, so the compression selection criteria may differ.
- **Spatial reasoning grounding: [Chain of Spatial Thoughts / Space Tokens](https://arxiv.org/abs/2608.10278v1)**
  - Mechanism: Distills scene-level 3D geometry and object-level spatial attributes into continuous latent tokens that enter CoT reasoning directly, with no extra spatial encoder.
  - Integration point: Instruction parsing that requires spatial-relation judgments ("walk around the table to the chair by the window").
  - Prerequisites and risks: The entry title (Chain of Spatial Thoughts) is inconsistent with the method name in the abstract (Space Tokens), so **attribution** still needs to be verified; no navigation experiments have been seen.
- **Data quality audit: [Auditing Instruction-Trajectory Mismatches (MMPF)](https://arxiv.org/abs/2608.07895v1)**
  - Mechanism: Training-free multimodal probabilistic fusion that detects samples with "correct trajectory but mismatched language instruction" and corrects the labels.
  - Integration point: Auditing instruction-trajectory pairing in VLN demonstration data, especially automatically generated or crowdsourced instructions.
  - Prerequisites and risks: Validated on LIBERO with injected mismatches and on noisy real-robot data; VLN instructions are longer-grained, so whether the local-neighborhood consistency assumption holds needs to be verified.

## 5. Research roundup by category
{: id="五分类速览"}

Relevance labels: A means directly relevant to ground navigation research, B means the mechanism is transferable, and C is for field observation only.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **SSTG-Nav** (reusable ObjectNav · A): See key analysis. [Source](https://arxiv.org/abs/2608.00527v1)
- **LifelongCrossNav** (Cross-floor persistent memory · A): See key analysis. [Source](https://arxiv.org/abs/2608.07079v1)
- **WNM-3D** (VLN-CE World Model · A): See key analysis. [Source](https://arxiv.org/abs/2608.07267v1)
- **Route2Step** (decoupling progress and execution · A): See key analysis. [Source](https://arxiv.org/abs/2608.03143v1)
- **SAP-Nav** (hierarchical OVON · A): See key analysis. [Source](https://arxiv.org/abs/2608.12707v1)
- **SC²-WM** (closed-loop VLN-CE · A): Uses world-model lookahead to make state-level plan corrections. When feedback shows the model is insufficient, the world model is selectively updated at test time; [Code](https://github.com/sunrise-ikun/SC2_WM) is released. [Source](https://arxiv.org/abs/2608.07548v1)
- **CompactNav** (VLN-CE representation · A): The WeChat interpretation says it introduces the "minimal sufficient representation" for the first time, chaining text-instruction-guided image filtering, a low-rank cross-modal information bottleneck, and a compressed world model. [Source](https://arxiv.org/abs/2607.23181)
- **Embodied Agents Take Control** (zero-shot VLN-CE · A): A general-purpose coding agent directly takes over the closed navigation loop using only monocular RGB and 4 actions. [Source](https://arxiv.org/abs/2607.26148)
- **SAIN** (interactive instance goal navigation · A): Compiles dialogue answers into persistent structured memory instead of one-off text prompts. [Source](https://arxiv.org/abs/2608.09196v1)
- **HumanoidVLN** (humanoid VLN benchmark · A): A multi-morphology humanoid VLN simulation platform with physical grounding, built on Isaac Sim. [Source](https://arxiv.org/abs/2608.12860v1)
- **CrossTracer** (cross-embodiment navigation · A): Normalized image-plane waypoints serve as a unified interface, a CE-Adapter predicts embodiment-conditioned residual corrections, and CE-RRT* generates training annotations automatically. [Source](https://arxiv.org/abs/2608.06688v1)
- **ULVN** (unordered image-goal navigation · A): RGB only, with no temporal or odometry priors; builds a 2D topological graph from an unordered image collection and localizes by graph-based confidence propagation. [Source](https://arxiv.org/abs/2608.06833v2)
- **UniNav** (image-goal navigation · A): Denoises visual tokens and continuous waypoints jointly in a single diffusion process, and can be trained on pure video data without waypoint annotations. [Source](https://arxiv.org/abs/2608.03244v1)
- **Latent World Models with Monotone Planning Costs** (image-goal navigation planning · A): Points out that errors in the planning-cost ordering mislead CEM sampling planners, and proposes a monotonic cost-ordering loss. [Source](https://arxiv.org/abs/2608.09073v1)
- **SpikingNav** (robust embodied navigation · A): A spiking-aware encoder plus a spiking policy network, aimed at resource-constrained platforms and degraded visual conditions. [Source](https://arxiv.org/abs/2608.05078v1)
- **360CityArena** (urban navigation benchmark · A): 602 360° videos of Akihabara, 85 streets, and 175 manually written tasks; evaluations show that the spatial reasoning of mainstream LMM agents falls below that of human experts. [Source](https://arxiv.org/abs/2608.08814v1)
- **Can VLMs Assess Proxemic Risk** (navigation safety assessment · B): Three open-source VLMs perform four-level hazard classification on first-person robot images, with limited gains after fine-tuning; correct classification does not mean the people are localized correctly in space. [Source](https://arxiv.org/abs/2608.12515v1)
- **Embodied Multimodal Grounding via Semantic-3DGS** (open-vocabulary mobile manipulation · B): Active multi-view semantic 3DGS with reachability-aware base pose selection; 3D semantic cues are injected only into the later layers of the action expert. [Source](https://arxiv.org/abs/2608.10756v1)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **AtlasVLA** (persistent state memory · B): Dual memory: a 4D voxel-hashed world state plus an egocentric working state. [Source](https://arxiv.org/abs/2608.06729v1)
- **ChainVLA** (cross-query execution state · B): A recurrent working state plus sparse event memory carries task progress, and unexecuted actions are carried over into the next generation. [Source](https://arxiv.org/abs/2608.02326v2)
- **Explicit Language Memory** (long-horizon planning · B): Converts discrete temporal observations into text memory sequences with temporal logic. [Source](https://arxiv.org/abs/2608.04765v1)
- **SkillMemo** (skill memory · B): MoE-guided trajectory segmentation plus skill-level dynamic episodic memory. [Source](https://arxiv.org/abs/2608.05970v1)
- **Skills in Weights, Memory in Code (HyMeS)** (hybrid memory · B): Low-level skills are learned by imitation, while high-level memory management is handed to the executable heuristics of a coding agent. [Source](https://arxiv.org/abs/2608.09410v1)
- **BridgeVLA++** (3D manipulation memory · B): A unified spatiotemporal memory architecture that models persistent spatial context and temporal interactions. [Source](https://arxiv.org/abs/2608.05042v1)
- **Continue or Replan? (BCP)** (adaptive execution horizon · B): See transferable methods. [Source](https://arxiv.org/abs/2608.03483v1)
- **RTCF** (training-free test-time correction · B): Retrieves successful trajectories through progressive memory alignment, and corrects and fuses them in the frequency domain rather than the time domain. [Source](https://arxiv.org/abs/2608.04527v2)
- **VANE** (test-time training · B): Candidate updates are isolated from the online policy and committed only after verification against subsequent observations, making adaptation optional and reversible. [Source](https://arxiv.org/abs/2608.09448v2)
- **SAFECAST** (failure detection · B): See transferable methods. [Source](https://arxiv.org/abs/2608.04246v1)
- **GUARD** (failure detection · B): See transferable methods. [Source](https://arxiv.org/abs/2608.04510v1)
- **ValueFormer** (per-frame value signal · B): A frozen causal transformer on DINOv3 outputs a smooth value signal and a binary correction signal in a single forward pass. [Source](https://arxiv.org/abs/2608.02958v1)
- **Decoding Task Progress** (Progress Interpretability · B): See transferable methods. [Source](https://arxiv.org/abs/2608.13474v1)
- **Auditing ITM (MMPF)** (Data Auditing · B): See transferable methods. [Source](https://arxiv.org/abs/2608.07895v1)
- **From Recovery to Drop-off** (VLA representation degradation · B): Action post-training lowers the depth decodability of every layer, with an additional collapse in the last layer that can be traced to interference from the last-layer MLP. [Source](https://arxiv.org/abs/2608.08904v1)
- **Positional Blind Spots** (spatial blind spots · B): Merely moving a task-irrelevant distractor can sharply raise the failure rate in localized regions; a localization and LoRA mitigation procedure is proposed. [Source](https://arxiv.org/abs/2608.01573v1)
- **Suppression Sticks, Locality Is Fragile** (model-editing audit · C): Task-vector subtraction on the ten LIBERO-Goal skills shows three regimes (clean removal, resistance, or global collapse), so its locality is unreliable. [Source](https://arxiv.org/abs/2608.04692v1)
- **3DZip** (3D token compression · B): Voxelization to remove redundancy, plus feature-diversity anchor selection. [Source](https://arxiv.org/abs/2608.01185v1)
- **HiSC** (3D token compression · B): Training-free hierarchical spatial clustering that lifts compression from the token level to the cluster level. [Source](https://arxiv.org/abs/2608.04610v1)
- **CoverPrune** (3D token pruning · B): Formalizes "maintaining evidence coverage" through optimal transport, replacing diversity maximization. [Source](https://arxiv.org/abs/2608.13226v1)
- **Chain of Spatial Thoughts / Space Tokens** (spatial grounding · B): See transferable methods; the title and method name are inconsistent, so attribution needs to be verified. [Source](https://arxiv.org/abs/2608.10278v1)
- **World Tokens** (training-time world modeling · B): A World Adapter converts VLM features into fixed-length world tokens that connect to a future-video denoiser during training and are discarded at deployment. [Source](https://arxiv.org/abs/2608.09730v1)
- **GWM-VLA** (geometry-aware world model · B): VGGT-Ω aggregates multiple views into a geometry-aware state and predicts the next patch token of the target view. [Source](https://arxiv.org/abs/2608.07619v1)
- **SLIM-0.5B** (compact latent interaction model · B): 0.5B parameters; predictive latent variables learn action grounding through self-supervised masked trajectory prediction. [Source](https://arxiv.org/abs/2608.09771v1)
- **Weights or Skills?** (survey · B): Organizes robot learning along the axis of "frozen-weight policies vs. self-written executable skills", ranking code-as-policy methods by their degree of self-improvement. [Source](https://arxiv.org/abs/2608.01851v1)
- **StellaVLA** (in-context adaptation · B): Offline, converts raw trajectories into structured demonstrations containing a task plan, sub-goal descriptions, and verbalized 3D motion; at test time a single retrieved demonstration serves as the condition. [Source](https://arxiv.org/abs/2608.11671v1)
- **In-Context VLA** (language consumption · B): Argues that free-form text CoT harms low-level control, and advocates that VLAs consume language rather than generate it. [Source](https://arxiv.org/abs/2608.05738v1)
- **PhyAI** (inference engine · B): A single runtime unifies VLA and WAM inference across onboard, edge, and cloud settings, reporting a 1.40×–4.65× speedup over the official implementations. [Source](https://arxiv.org/abs/2608.03682v2)
- **EMS (Fast and Accurate)** (dual-system decoupling · B): Context-aware model selection that switches between two fully decoupled large and small systems, with no end-to-end joint training. [Source](https://arxiv.org/abs/2608.06434v1)
- **WA-SpecDec** (speculative decoding · B): Injects physical scene awareness derived from the world model into the prefill so that the acceptance threshold varies with scene risk. [Source](https://arxiv.org/abs/2608.08725v1)
- **Temporal GRPO** (RL credit assignment · B): Builds detectable task stages and compares only rollouts that reach the same stage, alleviating trajectory-level credit aliasing. [Source](https://arxiv.org/abs/2608.13026v1)
- **TEMPO** (RL post-training · B): Dual-timescale optimization: the VLM backbone is frozen, while the semantic projection layer and action expert are updated at different rates. [Source](https://arxiv.org/abs/2608.07314v1)
- **HiRoC** (hierarchical post-training · B): The planner decomposes sub-goals, and the executor keeps improving sub-goal-conditioned actions during online interaction. [Source](https://arxiv.org/abs/2608.05999v1)
- **DyPES-VLA** (cross-embodiment · B): A shared dynamics prior (a future-prediction target) plus embodiment-specific control, reducing manual alignment of action formats. [Source](https://arxiv.org/abs/2608.06374v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **G0.5** (unified autoregressive VLA · B): A single transformer decoder outputs reasoning and action tokens under one objective, with a cross-embodiment action tokenizer and a visual memory module. [Source](https://arxiv.org/abs/2608.11739v1)
- **Ego2Robot** (data synthesis · C): Converts first-person human video into robot training data, reporting 18,561 hours across 15 embodiments. [Source](https://arxiv.org/abs/2608.02580v1)
- **DreamTrajectory** (mobile manipulation · B): Plans task-space motion jointly, regenerates it as whole-body motion, and verifies whether the predicted motion achieves the intended motion. [Source](https://arxiv.org/abs/2608.01381v1)
- **Panorama-Aware VLA** (mobile manipulation · B): A whole-body teleoperation system plus a panoramic-perception policy, collecting 5.5 hours of multimodal wheeled-arm demonstrations. [Source](https://arxiv.org/abs/2608.02257v1)
- **MVUCF** (multi-camera representation · C): Depth and cross-view correspondence targets are injected only during training, and the auxiliary head is removed at deployment. [Source](https://arxiv.org/abs/2608.01826v1)
- **Mind-VLA** (instruction-aware spatial alignment · B): Aligns the three-view geometry of only the language-specified target object, not the entire scene. [Source](https://arxiv.org/abs/2608.04633v1)
- **CofactVLA** (causal deconfounding · B): Builds a language-masked counterfactual branch within a single forward pass to suppress the causal confounding in which "vision overpowers language". [Source](https://arxiv.org/abs/2608.04396v1)
- **Grounded Semantic Re-Binding** (instruction generalization · B): Argues that the performance collapse under rephrased instructions stems from the architecture (feature drift caused by jointly encoding vision and text) rather than a lack of semantic understanding. [Source](https://arxiv.org/abs/2608.02497v1)
- **SALT** (action tokenizer · B): Requires a frozen VLM to recover the instruction from the quantized action latents; reports an average SimplerEnv success rate of 71.9% vs. 42.7% for a reconstruction baseline. [Source](https://arxiv.org/abs/2608.10484v1)
- **LIRA** (Inter-layer information routing · C): Each fusion block is aligned with a local depth window centered on the corresponding VLM layer. [Source](https://arxiv.org/abs/2608.07596v1)
- **How Should VLAs Use Proprioceptive State** (ablation study · B): Fixes the backbone and data, and compares five ways of feeding in the embodiment state as well as the effect of history length. [Source](https://arxiv.org/abs/2608.03052v1)
- **Cross-View Action Consistency** (view robustness · B): Regularizes the action flow velocity field, building supervision from view pairs rendered from the same MuJoCo state. [Source](https://arxiv.org/abs/2608.06965v1)
- **Track4Action** (world-centric distillation · C): Distills the realized transfer of a frozen 3D tracker into a policy that uses only the current observation, so deployment needs no tracker. [Source](https://arxiv.org/abs/2608.03727v1)
- **World-to-Wrist** (fine-grained manipulation · C): Predicts future wrist latents, conditioned on the task, as context for action prediction. [Source](https://arxiv.org/abs/2608.05369v1)
- **ReTouch** (tactile VLA · C): Tactile patch encoding preserves finger identity and local contact structure, and tactile prediction is refined online. [Source](https://arxiv.org/abs/2608.01824v1)
- **FACT (Demystifying VLA Failures)** (contact-rich failure analysis · C): Distinguishes accuracy failures (flow-matching training mismatch) from force failures (structure of the force signal), reporting 66% vs. 41% for the best baseline across five tasks. [Source](https://arxiv.org/abs/2608.01402v1)
- **SpaceVLA** (user-annotated anchors · C): An XR interface lets users mark grasp and placement regions, which are rendered as image overlays; closed-loop grasp success in Unity is 91.25%. [Source](https://arxiv.org/abs/2608.05730v1)
- **RoboSynChallenge** (competition benchmark · C): A unified competition setup with training on synthetic data and evaluation in real environments. [Source](https://arxiv.org/abs/2608.12416v1)
- **Policy-Induced Hand Priors** (humanoid arms · C): Quantifies dependence on the initial pose and hand-selection bias across 17 initial configurations. [Source](https://arxiv.org/abs/2608.11769v1)
- **RL Bootstrapping of OpenVLA-OFT** (demonstration-free embodiment alignment · C): A two-stage PPO + GRPO adaptation to a cable-driven parallel robot with no demonstrations on that embodiment. [Source](https://arxiv.org/abs/2608.01013v1)
- **Trajectory Divergence Horizon** (surgical arm · C): Formalizes surgical VLA deployment as an adaptive execution-horizon decision problem. [Source](https://arxiv.org/abs/2608.09125v1)
- **Deltoris** (inference acceleration · C): Algorithm-hardware co-design combining bit-level sparsity and speculative inference, enabling 50–200 Hz control for diffusion VLAs. [Source](https://arxiv.org/abs/2608.04428v1)
- **Neural Introspection Gating** (KV cache reuse · C): Uses the logit margin between the top-2 action tokens as a zero-cost confidence signal to trigger cache invalidation. [Source](https://arxiv.org/abs/2608.10824v1)
- **The Gate, Not the Cache** (acceleration reliability · B): When the gate signal comes from the accelerated forward pass itself, success on LIBERO-Object at a 0.9 skip rate drops to 0.68 (reuse) / 0.31 (deletion), and action-level detectors cannot catch it. [Source](https://arxiv.org/abs/2608.00391v1)
- **CloudEdgeVLA** (cloud-edge collaboration · B): Treats timing mismatch as a representation-learning problem, combining slow cloud-side features with the latest edge-side vision. [Source](https://arxiv.org/abs/2608.00569v1)
- **Hermite Curves as Trajectory Priors** (action chunk structure · C): Parameterizes the action chunk with piecewise cubic Hermite curves, enforcing smoothness and continuity at the endpoints. [Source](https://arxiv.org/abs/2608.01265v2)

### 5.4 UAVs, autonomous driving, and less relevant topics
{: id="54-无人机自动驾驶与其他低相关方向"}

- **FreqNav** (UAV VLN · C): Routes visual tokens between low-frequency global structure and high-frequency target detail according to flight phase. [Source](https://arxiv.org/abs/2608.00970v1)
- **AeroDPO** (UAV VLN · C): Argues that perception quality matters more than language reasoning capacity: a 2B model with high-fidelity vision matches a 7B baseline. [Source](https://arxiv.org/abs/2608.07557v1)
- **CoNav-UAV** (UAV collaboration · C): Dual-altitude, two-drone collaboration explicitly modeled via Stackelberg learning. [Source](https://arxiv.org/abs/2608.01802v1)
- **DBFly** (UAV VLN · C): Explicit spatial reasoning before waypoint generation; the WeChat interpretation reports an average success rate gain of 25.07 percentage points. [Source](https://arxiv.org/abs/2608.04825)
- **FlowPilot** (UAV obstacle avoidance · C): A dual-flow world-action model with a 7th-order Bernstein polynomial action representation; the WeChat interpretation reports inference within 18 ms on a Jetson Orin NX and a measured 5.5 m/s. [Source](https://arxiv.org/abs/2608.00635)
- **RecoverFly** (UAV RL·C): Failure-aware token-level RL post-training. [Source](https://arxiv.org/abs/2608.09467v1)
- **AirForesight** (UAV VLN · C): The current map representation is supervised by both current-frame reconstruction and future trajectory prediction. [Source](https://arxiv.org/abs/2608.12835v1)
- **ARIES-Mission2** (UAV mission generation · C): After zero-shot target localization, converts pixel positions to GPS waypoints and optimizes the visiting order with a TSP solver. [Source](https://arxiv.org/abs/2608.12763v1)
- **DreamFly** (UAV VLN · C): Causally aligned history memory plus receding-horizon diffusion planning. [Source](https://arxiv.org/abs/2608.12308v1)
- **GRASP** (UAV cross-modal · C): Region-focused alignment plus semantic prototypes to handle background interference and visual homogeneity in overhead views. [Source](https://arxiv.org/abs/2608.09270v1)
- **Semantic grounding to a unified framework for decision optimization** (UAV VLN · C): Semantic enhancement of instruction grounding plus correlation-aware dynamic aggregation of history. [Source](https://arxiv.org/abs/2608.09564v1)
- **SkyAnchor** (aerial streaming segmentation · C): Semantic token routing plus a two-tier memory bank, with the DroneEyes pixel-level streaming dataset. [Source](https://arxiv.org/abs/2607.19857)
- **DisasterBench** (UAV disaster reasoning · C): 5330 aerial images and 29300 multiple-choice reasoning samples, together with a 2B on-device model. [Source](https://arxiv.org/abs/2606.06217v1)
- **DaViNCi** (Outdoor VLN dataset · C): The first outdoor VLN dataset with both continuous motion and dynamic elements: 6 maps and 6933 trajectories. [Source](https://arxiv.org/abs/2608.11901v1)
- **WAM-Diff2** (driving VLA·C): Distills a pretrained autoregressive generalist into a multi-task discrete diffusion model. [Source](https://arxiv.org/abs/2608.01035v2)
- **Deferred Exposure of Future Trajectories** (Driving CoT·C): Points out that exposing ground-truth future trajectories during annotation induces trajectory anchoring bias. [Source](https://arxiv.org/abs/2608.01755v2)
- **XCoT-VLA** (Driving CoT · C): Replaces natural-language reasoning with compact executable CoT tokens. [Source](https://arxiv.org/abs/2608.10976v1)
- **BrainWAM** (Driving Planning · C): Points out that semantic shortcuts suppress predictive dynamics in shared attention, and replaces them with coordination in action space. [Source](https://arxiv.org/abs/2608.12854v1)
- **FlashDrive** (driving inference acceleration · C): Targets four bottlenecks at once: visual encoding, prefill, serial generation of reasoning tokens, and denoising. [Source](https://arxiv.org/abs/2608.12932v1)
- **FIRE-VLA** (DRIVING RL·C): Groups with low reward and low diversity trigger self-distillation, turning unresolved failures into privileged supervision. [Source](https://arxiv.org/abs/2608.13395v1)
- **DriveVLA-M0** (Driving Memory · C): A latent memory pool of failure cases plus a retrieval model that decouples static road structure from dynamic interaction. [Source](https://arxiv.org/abs/2608.10413v1)
- **CMU-Drive / V2V-VLA** (Cooperative Driving · C): A closed-loop benchmark for connected multi-vehicle collaboration, with actions, waypoints, reasoning, and communication policies generated jointly in a single forward pass. [Source](https://arxiv.org/abs/2608.07621v1)
- **Depth-Wise Probing of Planning Token** (Driving Interpretability · C): Navigation commands can be linearly decoded from the first layer onward (97.7%), but compatibility with the native planner peaks only at the last layer. [Source](https://arxiv.org/abs/2608.07361v1)
- **VLAGuard** (Physical Attack Defense · C): Attention-protection fine-tuning reduces OpenVLA's failure rate in LIBERO simulation from 100.0% to 25.9%. [Source](https://arxiv.org/abs/2608.01028v1)
- **SARF** (Physical Attack Defense · C): Structure-aware robust fine-tuning with zero inference overhead. [Source](https://arxiv.org/abs/2608.03231v1)
- **DRIFT** (Adversarial Attack · C): Attacking only the first denoising step of a flow-matching VLA is stronger and cheaper than attacking a wider window. [Source](https://arxiv.org/abs/2608.03207v1)
- **DURA** (Adversarial Attack · C): Generates visually natural adversarial patches with diffusion, supporting black-box settings. [Source](https://arxiv.org/abs/2608.10393v1)
- **UniTexture** (Adversarial Attack · C): A single textured 3D object induces target drift across tasks. [Source](https://arxiv.org/abs/2608.13453v1)
- **Text-Guided Glioma Segmentation** (medical imaging · —): Unrelated to embodied navigation; a noisy keyword hit (vision-language). [Source](https://arxiv.org/abs/2608.05389v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

- **2026-08-07 · Deep Blue Academy and the Qintong team at Shanghai Jiao Tong University are recruiting for an offline "Quadruped Robot VLN Training Camp", which claims to cover motion control, SLAM, zero-shot goal navigation, and the TravExplorer framework, using Unitree Go2 + Mid-360 + RealSense + Orin NX** (training enrollment): Commercial course information, not a research result. Worth watching whether papers or code for the mentioned TravExplorer navigation framework become public. [Source](https://mp.weixin.qq.com/s/yXNDgROEEe0A7xBuXnzesg)
- **2026-08-09 · WeChat public account article "From understanding the world to entering the scene - What step is left for ROBOT"** (opinion/review): **the text could not be fetched (the text container was not found), so the content is unknown**; no judgment is made in this issue. [Source](https://mp.weixin.qq.com/s/l7oCKh7jQJkbS6mGf0aCgQ)

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **"Memory" is moving from implicit state inside the model to explicit, inspectable structure.** AtlasVLA's voxel-hashed world state, LifelongCrossNav's sparse semantic voxels, SSTG-Nav's topological map, SAIN's structured dialogue memory, and HyMeS's coded heuristic system. The five works come from the manipulation and navigation communities, yet all choose to keep memory outside the network. This digest's assessment: this choice stems partly from the need for engineering diagnosability rather than purely from performance considerations, since implicit memory cannot answer "why did the robot forget the chair just now?"
- **Failure is no longer handled only at training time; it is moving into runtime detection and correction.** At least 8 works in this issue focus on reliability at deployment: SC²-WM (navigation state drift), VANE (reversible test-time adaptation), SAFECAST / GUARD (failure detection), ValueFormer / Decoding Task Progress (per-frame progress signals), RTCF (training-free test-time correction), and BCP (replanning timing). Their shared assumption is that the base policy is frozen.
- **Efficiency work is starting to expose reliability costs instead of only reporting speedups.** The Gate, Not the Cache shows that token skipping collapses in closed loop at high skip rates and cannot be detected by action-level detectors; Suppression Sticks shows that the locality of task-vector subtraction is unreliable; From Recovery to Drop-off shows that action post-training systematically weakens the deep decodability of the VLM. This digest's assessment: for navigation, such "cost audits" are worth tracking more than new acceleration methods, because navigation failures tend to accumulate slowly rather than crash in a single step.
- **UAVs and autonomous driving take up a large share of this issue (29 entries in Section 5.4, about 27% of independent works), but few of their mechanisms transfer to ground-based VLN.** They focus mostly on flight dynamics, frequency-domain token allocation, driving trajectory representation, and reasoning acceleration, and share no bottleneck with the semantic navigation problem of ground platforms. This digest's assessment: the current search keywords are noisy in the UAV direction.

### Research gaps
{: id="研究空白"}

- **There is no update strategy for when maps or memory become invalid.** SSTG-Nav and LifelongCrossNav both assume that "the environment structure is relatively stable", but neither entry explains how to detect and locally update the map after objects move or rooms are rearranged. This is the first problem exposed by long-term field deployment.
- **The cost of active sensing is not captured by any metric.** SAP-Nav's active viewpoint verification and SAIN's active questioning both increase path length and the number of interaction rounds. Existing SPL metrics cannot reflect the trade-off of "asking one more question saves ten meters of travel", nor the cost of disturbing people with questions.
- **Public numbers are seriously lacking.** Among the A-level works in this issue, only SAIN and HumanoidVLN provide verifiable quantitative information; the rest mostly have truncated abstracts or rely on WeChat interpretations. Cross-method comparison is largely infeasible in this issue.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read closely + reproduce the evaluation protocol**: SSTG-Nav's three evaluation protocols that isolate information boundaries (separating the effects of map geometric coverage and semantic recognition errors) are more worth porting to your own ObjectNav stack than its 97.5% success rate.
- **Read closely + partially reproduce**: Route2Step's step-level interface and E-SPA alignment procedure; first run a low-cost check in the form of a "current sub-instruction index" auxiliary head.
- **Join and track the benchmarks**: HumanoidVLN (clearest availability) and LifelongCrossNav's HM3D-MFMON (whether it is released still needs to be confirmed).

**Medium priority**

- **Borrow the mechanism**: attach the linear progress probe from Decoding Task Progress to your own navigation policy as a stuck/circling detector, and compare it with the existing timeout threshold.
- **Falsify first**: the 68.3%–78% zero-shot R2R-CE claimed by Embodied Agents Take Control. If true, the investment in dedicated navigation training needs to be reassessed; if not, the source of the gap needs to be clarified.
- **Review once the full text is available**: the complete experimental sections of WNM-3D, SAP-Nav, and LifelongCrossNav (their abstracts in this issue were truncated).
- **Adjust the crawl configuration**: Search keywords are noisy in the UAV direction (Section 5.4 accounts for about 27% of this issue), and the arXiv limit of 100 per run was hit within the 15-day window. Shortening the crawl interval or raising the limit is recommended.

**Low priority**

- **Paused**: The manipulation-specific work in Section 5.3 (tactile, dexterous, contact-rich tasks) and the driving-inference acceleration direction in Section 5.4. This issue shows no clear transfer path to ground navigation.

---

**Disclaimer**: This report is based on public abstracts captured through RSS and on WeChat interpretations. Most arXiv abstracts were truncated during crawling, and the experimental conclusions have not been verified against the original text. All numbers marked "WeChat interpretation" come from third-party interpretations rather than the paper itself; please check the original document before citing.
