---
layout: post
lang: en
translation_id: vln-weekly-2026-08-15
permalink: /en/vln-weekly-2026-08-15/
source_path: _posts/weekly-reports/2026-08-15-VLN-Weekly.md
source_url: /vln-weekly-2026-08-15/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
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

- **The memory of ground navigation begins to shift from "map within a single episode" to "persistent structure reused across episodes".** [SSTG-Nav](https://arxiv.org/abs/2608.00527v1) (metric semantic topology map for once mapping and multiple reuse), [LifelongCrossNav](https://arxiv.org/abs/2608.07079v1) (cross-floor multi-objective shared sparse 3D semantic voxel memory) and [SAIN](https://arxiv.org/abs/2608.09196v1) (compiling dialogue answers into resident space/object states) are unrelated to each other, but they are all solving the same thing: making the information obtained from the last navigation still available for the next query. This digest's assessment: this is the most directional signal for ground-based VLN in this issue - the evaluation protocol for long-term resident robots (homes, hospitals, office buildings) is expanding from "single instruction success rate" to "information reuse rate under sequence query".
- The execution paradigm of **VLN-CE is changing from open-loop to closed-loop, and error correction points are explicitly separated.** [SC²-WM](https://arxiv.org/abs/2608.07548v1) uses world model look-ahead to make state-level plan corrections before action execution, [WNM-3D](https://arxiv.org/abs/2608.07267v1) uses geometry perception to represent the joint generation of conditional future views and actions, [Route2Step](https://arxiv.org/abs/2608.03143v1) separates "progress tracking errors" and "execution errors" into two modules that can be supervised separately. The common premise among the three is that next-action supervision alone cannot distinguish whether the agent has made a wrong move or understood it wrong.
- **"What intermediate interface should be used to connect semantics and embodiment" has become the core design variable of cross-embodiment navigation.** [CrossTracer](https://arxiv.org/abs/2608.06688v1) selects the normalized image plane waypoint as the unified interface, and then uses the embodiment condition residual correction; [HumanoidVLN](https://arxiv.org/abs/2608.12860v1) pointed out from the evaluation side that the wheeled benchmark masks the bipedal motion constraints and camera shake caused by motion. In this digest's assessment, cross-embodiment is one of the few sub-directions in this issue that has progress on both sides of "methods" and "benchmarks".
- A lot of work on the **VLA side is actually the runtime monitoring and memory mechanism that can be migrated to navigation, rather than being exclusive to operations.** [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1) Prove that task progress can be read from the residual stream by a linear probe and used as label-free OOD detection, [Continue or Replan?](https://arxiv.org/abs/2608.03483v1) Turn replanning opportunities into learnable decisions, [SAFECAST](https://arxiv.org/abs/2608.04246v1) / [GUARD](https://arxiv.org/abs/2608.04510v1) Provides failure detection during deployment. The cost of these mechanisms to access VLN-CE is lower than retraining navigation strategies.
- **coverage boundary description:** The new entries on the arXiv side in this issue have reached the crawl limit of 100, and the earliest ones are only backtracked to 2026-08-01; submissions on 07-31 and earlier are shown in the previous report, and are not covered again in this issue. Among the 14 items on the WeChat public account side, 2 items are non-paper information, and the text of 1 item failed to be captured. There are a total of 114 original entries this time, and after cross-source merging into 5 groups, there are 109 independent works (107 papers + 2 pieces of information).

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [SSTG-Nav](https://arxiv.org/abs/2608.00527v1)** · Indoor reusable ObjectNav, including real robot
  - Contribution: An untargeted mapping establishes a semantic topology map that measures grounding, and subsequent queries only perform retrieval and lightweight planning.
  - Evidence: WeChat public account interpreted that the simulation benchmark success rate was 97.5%, and the ROS2 physical robot deployment was completed; the benchmark name and comparison method were not given in the entry.
  - Reason: The only ground navigation work that simultaneously proposes "reuse paradigm + evaluation protocol + real robot implementation".
- **A2 · [LifelongCrossNav](https://arxiv.org/abs/2608.07079v1)** · Unknown multi-floor indoor sequence multi-target ObjectNav
  - Contributions: Shared sparse 3D semantic voxel memory + stair-specific perception and cross-layer traversal; proposed HM3D-MFMON benchmark.
  - Evidence: The abstract is truncated at the baseline description and no verifiable value is provided.
  - Reason: Merge the two lines of "persistent memory" and "cross-floor" that were previously handled separately, and have their own benchmarks.
- **A3 · [WNM-3D](https://arxiv.org/abs/2608.07267v1)** · Continuous environment VLN (VLN-CE)
  - Contribution: Freeze feedforward geometry encoder + 3D Scene-to-Token Adapter, converting monocular RGB history into fixed-length prefixed conditional world-action DiT.
  - Evidence: The abstract is truncated at the method description and no verifiable values are provided.
  - Reason: This is the article with the most complete mechanism description and the most direct counterpart to ground-based VLN-CE among the world model works in this issue.
- **A4 · [Route2Step](https://arxiv.org/abs/2608.03143v1)** · VLN command follow
  - Contribution: Use an explicit step-level interface to decouple semantic progress tracking and action generation; E-SPA can monitor progress status without manual time annotation.
  - Evidence: The abstract was truncated at the oversight process and no verifiable value was provided.
  - Reason: Directly targeted at the persistent evaluation and training problem of "correction action labels covering up progress errors".
- **A5 · [SAP-Nav](https://arxiv.org/abs/2608.12707v1)** · Hierarchical Open Vocabulary Object Navigation (OVON)
  - Contribution: Online construction of queryable space-semantic representation + active viewpoint verification, zero-shot, no need to precompute the scene graph.
  - Evidence: The abstract is truncated in the experimental section and no verifiable values are provided.
  - Reason: The active perception loop of "if there is insufficient evidence, change the viewpoint and confirm again" can be directly grafted onto the existing ObjectNav stack.
- **A6 · [HumanoidVLN](https://arxiv.org/abs/2608.12860v1)** · Humanoid robot VLN simulation and benchmark
  - Contributions: Physics grounding platform on Isaac Sim, RL motion strategy + replaceable PD/MPC path tracking.
  - Evidence: 4 robots (Unitree G1, H1, Internal-A/B), lower limbs 10–12 DoF, height 1.17–1.80 m; declared compatible with NaVILA, DualVLN, StreamVLN, JanusVLN; scene requires navigable area >100 m².
  - Reason: To evaluate the cross-modality robustness of self-developed strategies, this is the only directly available physical grounding platform in this issue.
- **A7 · [SAIN](https://arxiv.org/abs/2608.09196v1)** · Interactive Instance Goal Navigation (IIGN)
  - Contribution: Compile oracle answers into target evidence, corridor memory and candidate object labels, and store them in structured memory for unified strategy consumption.
  - Evidence: VL-LN IIGN Benchmark: SR 20.2 → 25.4, SPL 13.07 → 14.17 (compared to "Strongest reported method", the method name is not listed in the abstract).
  - Reason: There are only a few ground navigation tasks in this issue that provide complete and verifiable values; the low absolute success rate also shows that the task is far from being solved.
- **A8 · [Embodied Agents Take Control](https://arxiv.org/abs/2607.26148)** ·  zero-shot  VLN-CE
  - Contribution: Directly reuse the common code Agent framework, only monocular RGB + 4 basic actions, no map/depth/panorama/waypoint modules.
  - Evidence: WeChat public account interprets that the zero-shot R2R-CE success rate range is 68.3%–78%, and claims that it is a benchmark large-scale training strategy; the entry does not provide an item-by-item comparison table.
  - Reason: If the conclusion is established, it will change the judgment of "whether it is worth continuing to invest in special navigation training" and deserves priority to be falsified.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. SSTG-Nav: map once, search repeatedly
{: id="1-sstg-nav把-objectnav-从每次重探索改写为一次测绘多次检索"}

**Problem.** Mainstream ObjectNav adopts a single exploration mode. Each command requires a rescan of the environment. Historical observations in long-term station scenarios are completely discarded; at the same time, there is a fault of "objects are recognized but cannot be reached" - the camera observation point is not equal to the legal stopping point of the robot.

**Method.** WeChat public account interprets that it consists of three parts: metric grounding (back-projecting 2D detection to 3D and offset 0.8 m to generate a safe docking area), source-aware 3D soft fusion (cross-view evidence aggregation + confidence filtering misdetection), multi-candidate fault self-healing; the query phase only performs retrieval and lightweight path planning, without repeated mapping

**Evidence.** WeChat public account interpreted that the simulation benchmark success rate was 97.5%, and the ROS2 real robot deployment was completed. **This digest's assessment: this number cannot yet be used for direct comparison.**: The entry does not give the benchmark name, scene division and baseline method. The authors also introduce 3 evaluation protocols to isolate information boundaries to separate the respective effects of "map geometric coverage" and "semantic recognition error" - this protocol design deserves more attention than the success rate number.

**Value.** directly corresponds to the real working conditions of long-term home/office robots; the geometric conversion of "observation point → accessible stop point" is a commonly missing link in existing semantic map solutions (VLMaps, ConceptFusion, etc.)

**Limitations.** relies on the static nature of the environment - the premise of a survey is that the layout and object positions do not change frequently. The entry does not explain the map invalidation and update strategy after the object moves or the scene changes; the absolute high value of 97.5% also indicates that the benchmark may be biased.

**Recommendation.** Prioritize the intensive reading of its **evaluation protocol part** rather than the success rate; [Project page ](https://daojiepeng.github.io/SSTG-Nav) and [Code ](https://github.com/DaojiePENG/sstg-nav-bench)] have been given, and the specific mechanism of 0.8 m docking offset can be independently ablated on its own ObjectNav stack

### 2. LifelongCrossNav: combining persistent memory with cross-floor navigation
{: id="2-lifelongcrossnav持久记忆与跨楼层被合并成同一个问题"}

**Problem.** ObjectNav's persistent memory (multi-target sequence query) and cross-floor navigation were previously two separate processing lines; the monocular single-floor assumption is inconsistent with real residential/office buildings

**Method.** In each episode, the agent receives an ordered sequence of object target queries, continuously maintains shared sparse 3D semantic voxel memory, incrementally accumulates geometric structures, traversable states, and visual-linguistic features, and subsequent queries directly retrieve without reconstructing the map; the cross-layer part consists of support surface-aware 3D traversability mapping, stair-specific perception, and direction-aware stair traversal; a unified strategy coordinates frontier exploration, real-time/historical POI retrieval, stair navigation, and target proximity on the same floor

**Evidence.** proposed HM3D-MFMON (a benchmark for sequential multi-floor multi-goal navigation based on HM3D scenes). **Abstract truncated at benchmark description, entry in this issue does not provide any verifiable success rate or SPL value**

**Value.** "Whether memory can reduce the exploration cost of subsequent queries" is a directly quantifiable indicator, which is closer to long-term deployment than monocular marking success rate; stair perception is a necessary module to push simulation conclusions to real residences

**Limitations.** The memory growth and long-term drift of sparse voxels + VL features are not covered in the abstract; cross-layer traversability is highly dependent on depth/geometric quality, and real robot migration risk is unknown

**Recommendation.** tracks whether HM3D-MFMON is open; if it is open, it can be compared with the multiplexing protocol of SSTG-Nav to determine whether the indicators of "in-episode memory" and "cross-episode memory" are interchangeable.

### 3. WNM-3D: adding geometric conditioning to a world-action model
{: id="3-wnm-3d给世界-动作模型补上几何条件"}

**Problem.** VLN systems increasingly transform pre-trained VLMs into VLAs that directly output actions. They have strong semantic capabilities but do not explicitly model "how observations should evolve under predicted actions"; the existing continuous VLN world-action model (WAM) is not conditioned on geometric perceptual representations inferred from history when jointly generating future views and actions.

**Method.** A frozen feed-forward geometry encoder extracts geometry-aware representations from monocular first-view RGB history, which a trainable 3D Scene-to-Token Adapter converts into a fixed-length prefix in world-action Diffusion Transformer token space; via block-causal attention, this prefix conditions each future "video-action" block, providing shared geometric context

**Evidence.** **The abstract is truncated here. No dataset, indicator or comparison value is provided for this issue's entry.** WeChat public account Interpretation of the same question (2026-08-10) The text captured is empty and cross-validation cannot be performed

**Value.** The combination of monocular RGB input + frozen geometry encoder has the lowest sensor requirements on the ground platform; "persistent scene context injection with fixed-length prefix" is a more computationally efficient method than frame-by-frame geometry fusion

**Limitations.** does not have any public value and can currently only be regarded as a mechanism candidate rather than a verified solution; the contradiction between the reasoning overhead of the diffuse world-action model and the closed-loop control frequency is not explained in the entry

**Recommendation.** Review the experiment after the text is available; at the mechanism level, you can first learn from the interface design of "frozen geometry encoder + lightweight Adapter to token prefix", which is orthogonal to the existing VLN-CE strategy

### 4. Route2Step: supervising execution errors and progress errors separately
{: id="4-route2step把走错了和理解错了分开监督"}

**Problem.** VLM-based navigators usually only use next-action prediction to supervise both "progress tracking" and "step execution" capabilities. When the agent deviates from the route, a corrective action tag can resume the next move, but it cannot indicate whether it selected the wrong sub-instruction or failed to execute the correct sub-instruction - so the agent continues to make decisions from the wrong progress state

**Method.** Instruction analysis module M_IA predicts step-level status from global instructions and visual history; action generation module M_AG generates local action chunks based on the status and recent observations; E-SPA step alignment process supervises progress status without manual time annotation

**Evidence.** **The abstract is truncated at the description of the supervision process, and no verifiable value** is provided

**Value.** This is the article in this issue that is closest to the needs of engineering diagnosis: the explicit progress status itself is an observable measure, which can be directly used for failure attribution, early stop and manual takeover triggering, and its value is not limited to the success rate.

**Limitations.** Explicit interfaces introduce the risk of error propagation - M_IA judgment errors will systematically contaminate downstream actions; the quality of unlabeled alignment (E-SPA) determines the overall upper limit, and the alignment accuracy is not given in the entry

**Recommendation.** gives priority to its **interface definition and E-SPA alignment process**; even if the complete framework is not used, "output current subcommand number" can be added to the existing strategy as a lightweight auxiliary header

### 5. SAP-Nav: changing viewpoint to confirm uncertain observations
{: id="5-sap-nav把看不清就换个角度再确认写进策略"}

**Problem.** Hierarchical OVON requires following free-form instructions that may specify targets via scene-level, room-level, zone-level, instance-level cues. There is a pair of contradictory requirements under partial observation: spatial grounding requires environmental-level persistent evidence, while target verification requires clear and discriminable candidate views.

**Method.** incrementally builds a queryable spatial-semantic representation from actively acquired room views, so that spatial semantic queries can be initiated at any explored location; Active Viewpoint Verification evaluates whether the current observation evidence is sufficient. If it is insufficient, the agent is first moved to a more informative viewpoint, and then the candidates are verified according to category and attribute constraints.

**Evidence.** is declared to be fully online, zero-shot, requiring no task-specific training or precomputed scene maps, and supports both hierarchical and standard class-level OVON. The abstract of **was truncated in the experimental part and no verifiable value** was provided

**Value.** Active viewpoint verification targets a type of failure that accounts for a high proportion of ObjectNav - long-range misjudgments leading to early termination; this module can be inserted into the existing zero-shot navigation pipeline as an independent component

**Limitations.** Changing viewpoints brings additional path overhead, and the summary does not explain the cost at the SPL level; the criterion of "whether the evidence is sufficient" relies on VLM confidence, which itself may be an unreliable signal

**Recommendation. Comparative reading of multi-perspective evidence fusion between** and SSTG-Nav - both are dealing with "untrustworthy single identification". The former relies on offline multi-perspective fusion, and the latter relies on online active relocation. The trade-off is worth quantifying

## 4. Transferable methods
{: id="四可迁移方法"}

- **Operation VLA Interpretability: [Decoding Task Progress](https://arxiv.org/abs/2608.13474v1)**
  - Mechanism: Task progress (normalized remaining time) can be linearly read out from the π0.5 residual stream by a single linear probe and used as a label-free OOD detector to identify progress stalls.
  - Integration point: The runtime progress monitoring and stuck detection of the navigation policy can replace the manually set timeout threshold.
  - Prerequisites and risks: The report states that the probe cannot effectively guide the strategy and can only read but not control; the "progress" definition of the navigation task (path completion vs. sub-command serial number) is different from the operation and needs to be re-verified.
- **operation VLA execution schedule: [Continue or Replan? (BCP)](https://arxiv.org/abs/2608.03483v1)**
  - Mechanism: Replace the fixed execution horizon with a series of "continue/replan" Bernoulli decisions, and the base strategy is frozen and plug-and-play.
  - Access position: re-planning timing of action chunks in VLN-CE; most current implementations re-plan based on a fixed number of steps, regardless of key turning points.
  - Prerequisites and risks: The optimal horizon cannot be directly observed, and its supervision construction method needs to be redone on navigation data.
- **Security during deployment: [SAFECAST](https://arxiv.org/abs/2608.04246v1), [GUARD](https://arxiv.org/abs/2608.04510v1)**
  - Mechanism: The former uses contrast set perturbation to improve the training and calibration of latent state risk probes, and the latter measures the grounding degree of actions on visual-linguistic evidence through ablation KV cache entries.
  - Integration point: navigation failure detection and takeover triggering; GUARD does not modify the pre-training strategy and has low access cost.
  - Prerequisites and risks: Both are verified in the operational benchmark (LIBERO / DROID / SimplerEnv classes), and the failure modes of navigation (circling, staggered levels, early stopping) are not covered.
- **long-term memory: [AtlasVLA](https://arxiv.org/abs/2608.06729v1)、[Skills in Weights, Memory in Code (HyMeS)](https://arxiv.org/abs/2608.09410v1)**
  - Mechanism: The former uses 4D voxel hashing persistent world state + ego working state dual memory to solve the problem of "objects are forgotten when they move out of sight"; the latter allows the encoding agent to assume memory management with an executable heuristic system, and the underlying strategy remains Markov.
  - Integration point: Corresponds to "Looking back and can't find the object I just passed" in navigation; HyMeS's division of labor is suitable for leaving the map/memory logic on the reviewable code side.
  - Prerequisites and risks: AtlasVLA is oriented to single-camera manipulation scenarios on the wrist, and the memory performance of voxel hashing at the room scale is unknown; HyMeS relies on the Agent's iterative feedback loop, and its real-time performance is questionable.
- **3D visual-language efficiency: [HiSC](https://arxiv.org/abs/2608.04610v1), [CoverPrune](https://arxiv.org/abs/2608.13226v1), [3DZip](https://arxiv.org/abs/2608.01185v1)**
  - Mechanism: Three complementary 3D token compression ideas: hierarchical clustering of spatial graph merging (training-free), evidence coverage maintenance in the sense of optimal transmission, voxelization + feature diversity anchor point selection.
  - Integration point: The computational bottleneck of semantic map query and scene-level VLM inference, especially in scenarios where the size of the persistent map increases with exploration.
  - Prerequisites and risks: Both are verified on 3D QA/scene understanding. What is required for navigation is "traversability + target positioning" rather than question and answer accuracy. The compression selection criteria may be different.
- **space reasoning grounding: [Chain of Spatial Thoughts / Space Tokens](https://arxiv.org/abs/2608.10278v1)**
  - Mechanism: Distill scene-level 3D geometry and object-level spatial attributes into continuous latent tokens, which directly enter CoT reasoning without the need for additional spatial encoders.
  - Integration point: Instruction parsing that requires spatial relationship judgment ("walk around the table to the chair by the window").
  - Prerequisites and risks: The entry title (Chain of Spatial Thoughts) is inconsistent with the method name (Space Tokens) in the abstract, and **belongs to** to be verified; no navigation mission experiment has been seen.
- **Data quality audit: [Auditing Instruction-Trajectory Mismatches (MMPF)](https://arxiv.org/abs/2608.07895v1)**
  - Mechanism: Training-free multi-modal probabilistic fusion, detects samples with "correct trajectories but mismatched language instructions" and corrects the labels.
  - Where to access: Instruction-trajectory pairing auditing of VLN demo data, especially automatically generated or crowdsourced instructions.
  - Prerequisites and risks: Verified on LIBERO injection mismatch and real robot noise data; the granularity of VLN instructions is longer, and whether the local neighborhood consistency assumption is established needs to be verified.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **SSTG-Nav** (reusable ObjectNav · A): See key analysis. [Source](https://arxiv.org/abs/2608.00527v1)
- **LifelongCrossNav** (Cross-floor persistent memory · A): See key analysis. [Source](https://arxiv.org/abs/2608.07079v1)
- **WNM-3D** (VLN-CE World Model · A): See key analysis. [Source](https://arxiv.org/abs/2608.07267v1)
- **Route2Step** (decoupling progress and execution · A): See key analysis. [Source](https://arxiv.org/abs/2608.03143v1)
- **SAP-Nav** (layered OVON · A): See key analysis. [Source](https://arxiv.org/abs/2608.12707v1)
- **SC²-WM** (VLN-CE closed loop · A): The world model is forward-looking to make state-level plan corrections. When feedback shows that the model capability is insufficient, the world model is selectively updated during the test period; [Code ](https://github.com/sunrise-ikun/SC2_WM) has been made public. [Source](https://arxiv.org/abs/2608.07548v1)
- **CompactNav** (VLN-CE representation · A): WeChat public account explains that it introduces "minimum sufficient representation" for the first time, using text instructions as a priori to filter images, low-rank cross-modal information bottlenecks, and compressed world models in series. [Source](https://arxiv.org/abs/2607.23181)
- **Embodied Agents Take Control** (zero-shot VLN-CE · A): The general code Agent directly takes over the navigation closed loop, only monocular RGB + 4 actions. [Source](https://arxiv.org/abs/2607.26148)
- **SAIN** (Interactive Instance Target Navigation · A): Compile conversation answers into resident structured memory instead of one-time text prompts. [Source](https://arxiv.org/abs/2608.09196v1)
- **HumanoidVLN** (Humanoid VLN Benchmark · A): A polymorphic humanoid VLN simulation platform based on Isaac Sim physics grounding. [Source](https://arxiv.org/abs/2608.12860v1)
- **CrossTracer** (cross-embodiment navigation · A): Normalized image plane waypoints as a unified interface, CE-Adapter predicts embodiment conditional residual correction; CE-RRT* automatically generates training annotations. [Source](https://arxiv.org/abs/2608.06688v1)
- **ULVN** (Unordered Image Target Navigation · A): RGB only, no timing and mileage priors, constructing a 2D topological graph from an unordered image collection + graph-based confidence propagation positioning. [Source](https://arxiv.org/abs/2608.06833v2)
- **UniNav** (Image Target Navigation · A): Combines denoising visual tokens and continuous waypoints in a single diffusion process, and can be trained using pure video data without waypoint annotations. [Source](https://arxiv.org/abs/2608.03244v1)
- **Latent World Models with Monotone Planning Costs** (Image Target Navigation Planning · A): It is pointed out that planning cost ordering errors will mislead the CEM sampling planner, and a monotonic cost ordering loss is proposed. [Source](https://arxiv.org/abs/2608.09073v1)
- **SpikingNav** (robust embodied navigation · A): pulse-aware encoder + pulse strategy network, oriented to resource-constrained platforms and visual degradation conditions. [Source](https://arxiv.org/abs/2608.05078v1)
- **360CityArena** (City Navigation Benchmark·A): Akihabara 602 360° videos, 85 streets, 175 manual tasks; evaluations indicate that mainstream LMM agent spatial reasoning is lower than human experts. [Source](https://arxiv.org/abs/2608.08814v1)
- **Can VLMs Assess Proxemic Risk** (Navigation Safety Assessment · B): Three open source VLMs perform four-level hazard classification on first-view robot images, with limited improvement after fine-tuning; correct classification does not equal correct spatial positioning of characters. [Source](https://arxiv.org/abs/2608.12515v1)
- **Embodied Multimodal Grounding via Semantic-3DGS** (Open Vocabulary Mobile Operation · B): Active multi-view semantic 3DGS + accessibility-aware chassis pose selection, 3D semantic cues are only injected into the motion expert back segment. [Source](https://arxiv.org/abs/2608.10756v1)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **AtlasVLA** (persistent state memory · B): 4D voxel hash world state + ego working state dual memory. [Source](https://arxiv.org/abs/2608.06729v1)
- **ChainVLA** (cross-query execution status · B): recursive working status + sparse event memory carries task progress, and unexecuted actions continue to the next generation. [Source](https://arxiv.org/abs/2608.02326v2)
- **Explicit Language Memory** (long time series planning · B): Convert discrete time series observations into text memory sequences with time logic. [Source](https://arxiv.org/abs/2608.04765v1)
- **SkillMemo** (Skill Memory · B): MoE-guided trajectory segmentation + skill-level dynamic plot memory. [Source](https://arxiv.org/abs/2608.05970v1)
- **Skills in Weights, Memory in Code (HyMeS)** (Hybrid Memory · B): The underlying skills are learned through imitation, and the high-level memory management is handed over to the executable heuristics of the coding agent. [Source](https://arxiv.org/abs/2608.09410v1)
- **BridgeVLA++** (3D Operational Memory · B): A unified spatiotemporal memory architecture modeling persistent spatial context and temporal interactions. [Source](https://arxiv.org/abs/2608.05042v1)
- **Continue or Replan? (BCP)** (adaptive execution horizon · B): See migration method. [Source](https://arxiv.org/abs/2608.03483v1)
- **RTCF** (training-free test period correction · B): Progressive memory alignment to retrieve successful trajectories, correction and fusion in the frequency domain rather than the time domain. [Source](https://arxiv.org/abs/2608.04527v2)
- **VANE** (training at test time · B): Candidate updates are isolated from the online strategy, and are submitted after verification with subsequent observations, making adaptation optional and reversible. [Source](https://arxiv.org/abs/2608.09448v2)
- **SAFECAST** (failed detection · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.04246v1)
- **GUARD** (failed detection · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.04510v1)
- **ValueFormer** (frame-by-frame value signal · B): freezes the causal transformer on DINOv3, and outputs smooth value and binary error correction signals simultaneously in one forward direction. [Source](https://arxiv.org/abs/2608.02958v1)
- **Decoding Task Progress** (Progress Interpretability · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.13474v1)
- **Auditing ITM (MMPF)** (Data Auditing · B): See Migration Methods. [Source](https://arxiv.org/abs/2608.07895v1)
- **From Recovery to Drop-off** (VLA representation degradation · B): Post-action training causes the depth decodability to decrease at each layer, and additional collapse occurs in the last layer, which can be localized to the MLP interference in the last layer. [Source](https://arxiv.org/abs/2608.08904v1)
- **Positional Blind Spots** (Spatial Capability Blind Spot · B): Only moving distractors unrelated to the task can significantly increase the failure rate in local areas, and a positioning and LoRA mitigation process is proposed. [Source](https://arxiv.org/abs/2608.01573v1)
- **Suppression Sticks, Locality Is Fragile** (Model Edit Audit · C): Task vector subtraction presents three states of separation/resistance/global collapse on the ten LIBERO-Goal skills, and the locality is unreliable. [Source](https://arxiv.org/abs/2608.04692v1)
- **3DZip** (3D token compression · B): voxelization de-redundancy + feature diversity anchor point selection. [Source](https://arxiv.org/abs/2608.01185v1)
- **HiSC** (3D token compression · B): training-free hierarchical spatial clustering, lifting compression from token level to cluster level. [Source](https://arxiv.org/abs/2608.04610v1)
- **CoverPrune** (3D token pruning · B): Formalize "maintain evidence coverage" with optimal transmission, replacing maximizing diversity. [Source](https://arxiv.org/abs/2608.13226v1)
- **Chain of Spatial Thoughts / Space Tokens** (space grounding · B): See the migration method; the title and method name are inconsistent, and the ownership needs to be verified. [Source](https://arxiv.org/abs/2608.10278v1)
- **World Tokens** (World Modeling during Training · B): World Adapter converts VLM features into fixed-length world tokens, which are connected to the future video denoiser during the training period and discarded during the deployment period. [Source](https://arxiv.org/abs/2608.09730v1)
- **GWM-VLA** (Geometry-aware world model · B): VGGT-Ω aggregates multiple perspectives to construct a geometry-aware state and predicts the next patch token of the target view. [Source](https://arxiv.org/abs/2608.07619v1)
- **SLIM-0.5B** (Compact Latent Interaction Model · B): 0.5B parameters, predictive latent variables for self-supervised mask trajectory prediction learning action grounding. [Source](https://arxiv.org/abs/2608.09771v1)
- **Weights or Skills?** (Review · B): Organize robot learning along the axis of "frozen weight strategy vs self-written executable skills", ranking code-as-policy methods according to the degree of self-improvement. [Source](https://arxiv.org/abs/2608.01851v1)
- **StellaVLA** (Context Adaptation·B): Offline convert the original trajectory into a structured demonstration containing task plan, sub-goal description and oral 3D movement, and retrieve a single demonstration during the test period to make conditions. [Source](https://arxiv.org/abs/2608.11671v1)
- **In-Context VLA** (Language Consumption Capacity · B): Argues that free-form text CoT compromises underlying control, advocating for VLAs to consume rather than generate language. [Source](https://arxiv.org/abs/2608.05738v1)
- **PhyAI** (Inference Engine · B): A single runtime unifies VLA and WAM for in-vehicle/edge/cloud inference, reporting a 1.40×–4.65× acceleration compared to the official implementation. [Source](https://arxiv.org/abs/2608.03682v2)
- **EMS (Fast and Accurate)** (Dual-system decoupling · B): Context-aware model selection, switching between two fully decoupled large and small systems without end-to-end joint training. [Source](https://arxiv.org/abs/2608.06434v1)
- **WA-SpecDec** (Speculative Decoding · B): Inject the physical scene perception derived from the world model into the prefill, so that the acceptance threshold changes with the scene risk. [Source](https://arxiv.org/abs/2608.08725v1)
- **Temporal GRPO** (RL credit allocation · B): Construct detectable task stages and only compare rollouts entering the same stage to alleviate trajectory-level credit aliasing. [Source](https://arxiv.org/abs/2608.13026v1)
- **TEMPO** (RL post-training · B): Dual time scale optimization with frozen VLM backbone, semantic projection layer and action experts updated at different rates. [Source](https://arxiv.org/abs/2608.07314v1)
- **HiRoC** (post-layered training · B): The planner decomposes sub-goals, and the executor continuously improves sub-goal conditional actions during online interaction. [Source](https://arxiv.org/abs/2608.05999v1)
- **DyPES-VLA** (cross-embodiment · B): shared dynamics prior (future prediction target) + embodiment-specific control, reducing manual action format alignment. [Source](https://arxiv.org/abs/2608.06374v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **G0.5** (Unified Autoregressive VLA · B): A single transformer decoder outputs reasoning and action tokens under the same goal, including a cross-embodiment action tokenizer and visual memory module. [Source](https://arxiv.org/abs/2608.11739v1)
- **Ego2Robot** (data synthesis · C): First-person human video-to-robot training data, reporting 18,561 hours, 15 forms. [Source](https://arxiv.org/abs/2608.02580v1)
- **DreamTrajectory** (Mobile Operation · B): Joint planning task space motion is regenerated into whole-body motion, and it is verified whether the predicted motion can achieve the intended motion. [Source](https://arxiv.org/abs/2608.01381v1)
- **Panorama-Aware VLA** (Mobile Operation · B): Full-body teleoperating system + panoramic sensing strategy, capturing 5.5 hours of wheeled-arm multi-modal demonstration. [Source](https://arxiv.org/abs/2608.02257v1)
- **MVUCF** (Multi-camera representation · C): Only depth and cross-view corresponding targets are injected during the training period, and the auxiliary head is removed during deployment. [Source](https://arxiv.org/abs/2608.01826v1)
- **Mind-VLA** (command-aware spatial alignment · B): Only aligns the three-view geometry of the language-specified target object, not the entire scene. [Source](https://arxiv.org/abs/2608.04633v1)
- **CofactVLA** (causal deobfuscation · B): Construct the language mask counterfactual branch in a single forward-inward manner to suppress the causal confusion of "visual overpowering language". [Source](https://arxiv.org/abs/2608.04396v1)
- **Grounded Semantic Re-Binding** (Instruction Generalization · B): Points out that the performance collapse caused by rewriting instructions stems from the architecture (feature drift caused by joint encoding of visual and text) rather than the lack of semantic understanding. [Source](https://arxiv.org/abs/2608.02497v1)
- **SALT** (action tokenizer · B): Requires frozen VLM to restore instructions from quantized action latent variables, reporting an average success rate of 71.9% for SimplerEnv vs 42.7% for reconstructed. [Source](https://arxiv.org/abs/2608.10484v1)
- **LIRA** (Inter-layer information routing · C): Each fusion block is aligned with a local depth window centered on the corresponding VLM layer. [Source](https://arxiv.org/abs/2608.07596v1)
- **How Should VLAs Use Proprioceptive State** (ablation study · B): Fixed the backbone and data, and compared the effects of five embodiment state access methods and history length. [Source](https://arxiv.org/abs/2608.03052v1)
- **Cross-View Action Consistency** (View Robust · B): Regularize the action flow velocity field, and use the same MuJoCo state rendering perspective to supervise the construction. [Source](https://arxiv.org/abs/2608.06965v1)
- **Track4Action** (World Center Distillation · C): Distill the implemented transfer of a frozen 3D tracker into the current observation strategy, deploying without a tracker. [Source](https://arxiv.org/abs/2608.03727v1)
- **World-to-Wrist** (fine-grained operation · C): Predict future wrist latent variables as action prediction context under task conditions. [Source](https://arxiv.org/abs/2608.05369v1)
- **ReTouch** (haptic VLA · C): tactile patch encoding preserves finger identity and local contact structure, and refines tactile prediction online. [Source](https://arxiv.org/abs/2608.01824v1)
- **FACT (Demystifying VLA Failures)** (Contact Dense Failure Analysis · C): Distinguishes accuracy failure (flow matching training mismatch) from force failure (force signal structure), reporting 66% for five tasks vs. 41% for the best baseline. [Source](https://arxiv.org/abs/2608.01402v1)
- **SpaceVLA** (user annotation anchor point · C): The XR interface allows users to mark the grab and place areas and render them into image overlays. The closed-loop Unity grab success rate is 91.25%. [Source](https://arxiv.org/abs/2608.05730v1)
- **RoboSynChallenge** (Competition Benchmark · C): Unified competition setup for synthetic data training + real environment evaluation. [Source](https://arxiv.org/abs/2608.12416v1)
- **Policy-Induced Hand Priors** (humanoid arms · C): Quantify initial pose dependence and hand selection bias under 17 initial configurations. [Source](https://arxiv.org/abs/2608.11769v1)
- **RL Bootstrapping of OpenVLA-OFT** (zero demonstration embodiment alignment · C): PPO + GRPO two-stage adaptive rope-driven parallel robot without embodiment demonstration. [Source](https://arxiv.org/abs/2608.01013v1)
- **Trajectory Divergence Horizon** (Surgical Arm·C): Formalizing surgical VLA deployment as an adaptive execution horizon decision problem. [Source](https://arxiv.org/abs/2608.09125v1)
- **Deltoris** (Inference Acceleration · C): Algorithm-hardware synergy for bit-level sparse + speculative inference, 50–200 Hz control for diffuse VLAs. [Source](https://arxiv.org/abs/2608.04428v1)
- **Neural Introspection Gating** (KV cache multiplexing · C): Use the logit margin of the top-2 action token as a zero-cost confidence signal to trigger cache invalidation. [Source](https://arxiv.org/abs/2608.10824v1)
- **The Gate, Not the Cache** (Acceleration Reliability · B): When the gate signal comes from its own acceleration forward, the success rate drops to 0.68 (reuse)/0.31 (deletion) under LIBERO-Object 0.9 skip rate, and the action-level detector cannot detect it. [Source](https://arxiv.org/abs/2608.00391v1)
- **CloudEdgeVLA** (Cloud-Edge Collaboration · B): Treating timing mismatch as a representation learning problem, combining slow features on the cloud with the latest vision on the edge. [Source](https://arxiv.org/abs/2608.00569v1)
- **Hermite Curves as Trajectory Priors** (Action Block Structure · C): Parameterize the action chunk with piecewise cubic Hermite curves to force smoothness to be continuous with the endpoints. [Source](https://arxiv.org/abs/2608.01265v2)

### 5.4 UAVs, autonomous driving, and less relevant topics
{: id="54-无人机自动驾驶与其他低相关方向"}

- **FreqNav** (UAV VLN · C): Routing visual tokens between low-frequency global structure and high-frequency target details by flight phase. [Source](https://arxiv.org/abs/2608.00970v1)
- **AeroDPO** (Drone VLN · C): Demonstrating perceptual quality over verbal reasoning capacity, 2B + high-fidelity visual matching 7B baseline. [Source](https://arxiv.org/abs/2608.07557v1)
- **CoNav-UAV** (UAV Collaboration·C): Dual-altitude dual-drone collaboration via Stackelberg learning for explicit modeling. [Source](https://arxiv.org/abs/2608.01802v1)
- **DBFly** (UAV VLN · C): explicit spatial reasoning before waypoint generation, WeChat public account interpreted that the average success rate increased by 25.07 percentage points. [Source](https://arxiv.org/abs/2608.04825)
- **FlowPilot** (UAV obstacle avoidance · C): dual-flow world-action model, 7th-order Bernstein polynomial action representation; WeChat public account interprets that Jetson Orin NX can reason within 18 ms and measured 5.5 m/s. [Source](https://arxiv.org/abs/2608.00635)
- **RecoverFly** (UAV RL·C): Failure-aware token-level RL post-training. [Source](https://arxiv.org/abs/2608.09467v1)
- **AirForesight** (UAV VLN · C): Current map representation is supervised by both current reconstruction and future trajectory prediction. [Source](https://arxiv.org/abs/2608.12835v1)
- **ARIES-Mission2** (UAV mission generation · C): After zero-shot target positioning, convert the pixel position to GPS waypoint, and use TSP to optimize the access sequence. [Source](https://arxiv.org/abs/2608.12763v1)
- **DreamFly** (UAV VLN · C): causally aligned history memory + rolling time-domain diffusion planning. [Source](https://arxiv.org/abs/2608.12308v1)
- **GRASP** (UAV cross-modal · C): regional focus alignment + semantic prototype to deal with background interference and visual isomorphism in the overhead perspective. [Source](https://arxiv.org/abs/2608.09270v1)
- **Semantic grounding to a unified framework for decision optimization** (UAV VLN · C): Instruction grounding semantic enhancement + correlation-aware historical dynamic aggregation. [Source](https://arxiv.org/abs/2608.09564v1)
- **SkyAnchor** (Aerial Streaming Segmentation · C): Semantic token routing + double-layer memory library, supporting DroneEyes pixel-level streaming dataset. [Source](https://arxiv.org/abs/2607.19857)
- **DisasterBench** (UAV disaster inference·C): 5330 aerial photos, 29300 multi-select inference samples, equipped with 2B end-side model. [Source](https://arxiv.org/abs/2606.06217v1)
- **DaViNCi** (Outdoor VLN dataset · C): The first outdoor VLN dataset that contains both continuous motion and dynamic elements, 6 maps, 6933 trajectories. [Source](https://arxiv.org/abs/2608.11901v1)
- **WAM-Diff2** (driving VLA·C): Distilling pretrained autoregressive generalists into multi-task discrete diffusion models. [Source](https://arxiv.org/abs/2608.01035v2)
- **Deferred Exposure of Future Trajectories** (Driving CoT·C): Points out that exposing ground truth future trajectories when annotating can induce trajectory anchoring bias. [Source](https://arxiv.org/abs/2608.01755v2)
- **XCoT-VLA** (Driving CoT · C): Replace natural language inference with compact executable CoT tokens. [Source](https://arxiv.org/abs/2608.10976v1)
- **BrainWAM** (Driving Planning · C): Point out that semantic shortcuts suppress predictive dynamics in shared attention, replacing action space coordination. [Source](https://arxiv.org/abs/2608.12854v1)
- **FlashDrive** (driving inference acceleration · C): simultaneously targets the four-level bottlenecks of visual encoding, prefill, inference token serialization, and denoising. [Source](https://arxiv.org/abs/2608.12932v1)
- **FIRE-VLA** (DRIVING RL·C): Low-reward low-diversity group triggers self-distillation, turning unresolved failures into privileged supervision. [Source](https://arxiv.org/abs/2608.13395v1)
- **DriveVLA-M0** (Driving Memory · C): Failure case latent memory pool + retrieval model that decouples static road structure and dynamic interaction. [Source](https://arxiv.org/abs/2608.10413v1)
- **CMU-Drive / V2V-VLA** (Cooperative Driving · C): Multi-networked vehicle closed-loop collaborative benchmark and single forward joint generation of actions, waypoints, reasoning and communication strategies. [Source](https://arxiv.org/abs/2608.07621v1)
- **Depth-Wise Probing of Planning Token** (Driving Interpretability · C): Navigation commands can be linearly decoded after the first layer (97.7%), but compatibility with the native planner is optimal until the last layer. [Source](https://arxiv.org/abs/2608.07361v1)
- **VLAGuard** (Physical Attack Defense · C): Attention protection fine-tuning reduces OpenVLA failure rate in LIBERO simulation from 100.0% to 25.9%. [Source](https://arxiv.org/abs/2608.01028v1)
- **SARF** (Physical Attack Defense · C): Structure-aware robust fine-tuning with zero inference overhead. [Source](https://arxiv.org/abs/2608.03231v1)
- **DRIFT** (Adversarial Attack · C): Attacking only the first step of flow-matching VLA denoising is stronger and less expensive than attacking a wider window. [Source](https://arxiv.org/abs/2608.03207v1)
- **DURA** (Adversarial Attack · C): Generates visually natural adversarial patches based on diffusion, supporting black box settings. [Source](https://arxiv.org/abs/2608.10393v1)
- **UniTexture** (Adversarial Attack · C): A single textured 3D object induces target drift across tasks. [Source](https://arxiv.org/abs/2608.13453v1)
- **Text-Guided Glioma Segmentation** (Medical Imaging · —): Not related to embodied navigation, but keyword hit noise (related to vision-language). [Source](https://arxiv.org/abs/2608.05389v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

- **2026-08-07 · Deep Blue Academy and the Shanghai Jiao Tong University Qintong team have recruited students for the "Quadruped Robot VLN Offline Training Camp", claiming to cover motion control, SLAM, zero-shot goal navigation and TravExplorer framework, using Yushu Go2 + Mid-360 + RealSense + Orin NX** (Training Enrollment): Business course information, not research results. You can pay attention to whether there are public papers or codes for the TravExplorer navigation framework mentioned. [Source](https://mp.weixin.qq.com/s/yXNDgROEEe0A7xBuXnzesg)
- **2026-08-09 · WeChat public account article "From understanding the world to entering the scene - What step is left for ROBOT"** (opinion/review): **failed to crawl the text (the text container was not found), the content is unknown**, no judgment will be made in this issue. [Source](https://mp.weixin.qq.com/s/l7oCKh7jQJkbS6mGf0aCgQ)

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **"memory" is moved from the implicit state inside the model to an explicit structure that can be checked.** AtlasVLA's voxel hashed world state, LifelongCrossNav's sparse semantic voxels, SSTG-Nav's topological map, SAIN's structured dialogue memory, and HyMeS's coded heuristic system. The five works come from the two communities of operation and navigation, but they all choose to put memory outside the network. This digest's assessment This approach stems in part from engineering diagnosability requirements rather than pure performance considerations - implicit memory cannot answer "Why did the robot forget the chair just now?"
- **failure is no longer only handled during the training period, but is moved to runtime detection and correction.** At least 8 tasks in this issue focus on reliability during the deployment period: SC²-WM (navigation state drift), VANE (reversible test period adaptation), SAFECAST / GUARD (failure detection), ValueFormer / Decoding Task Progress (frame-by-frame progress signal), RTCF (training-free test period correction), BCP (replanning timing). Their common assumption is that the base strategy is frozen.
- **efficiency work begins to expose reliability costs instead of just reporting speedups.** The Gate, Not the Cache shows that token skipping will collapse in a closed loop under high skip rates and is undetectable by action-level detectors; Suppression Sticks shows that the locality of task vector subtraction is unreliable; From Recovery to Drop-off shows that post-action training systematically weakens the deep decodability of VLM. This digest's assessment, for navigation, is that these kinds of "cost audits" are worth tracking more than new acceleration methods - navigation failures tend to accumulate slowly rather than in single-step crashes.
- **UAVs and autonomous driving account for the majority of entries in this issue (29 entries in Section 5.4, accounting for approximately 27% of independent work), but the mechanisms that can be migrated to ground-based VLN are limited.** mostly focuses on flight dynamics, frequency domain token allocation, driving trajectory representation and reasoning acceleration, and does not share bottlenecks with the semantic navigation problem of ground platforms. This digest's assessment, the current configuration of the search keyword has a high noise ratio in the direction of the drone.

### Research gaps
{: id="研究空白"}

- **The update strategy after map/memory failure is missing.** SSTG-Nav and LifelongCrossNav are both based on the premise that "the environment structure is relatively stable", but neither entry explains how to detect and locally update the map after objects are moved and rooms are rearranged. This is the first problem exposed by long-term field deployment.
- The cost of **active sensing is not included in the indicator.** SAP-Nav's active viewpoint verification and SAIN's active questioning will both increase the path length and the number of interaction rounds. However, the existing SPL indicators cannot reflect the trade-off of "asking more questions saves ten meters of travel", nor can they reflect the cost of disturbing people by asking questions.
- The public value of **is seriously missing.** In this issue of A-level work, only SAIN and HumanoidVLN provide verifiable quantitative information, and the rest are mostly abstract truncation or interpretation from WeChat public account. Cross-method comparisons are largely unfeasible in this issue.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read closely + recurrence evaluation protocol**: SSTG-Nav’s three sets of isolated information boundary evaluation protocols (separating the impact of map geometric coverage and semantic recognition errors) are more worthy of porting to their own ObjectNav stack than their 97.5% success rate.
- **Read closely + partial reproduction of**: Route2Step's step-level interface and E-SPA alignment process; first perform low-cost verification in the form of "current sub-command number" auxiliary header.
- **joins the benchmark to track**: HumanoidVLN (the clearest availability) and LifelongCrossNav's HM3D-MFMON (to be confirmed whether it is open).

**medium priority**

- The **mechanism borrows from**: connect the linear progress probe of Decoding Task Progress to its own navigation policy as a stuck/circling detector, and compare it with the existing timeout threshold.
- **gives priority to falsification of**: the zero-shot R2R-CE claimed by Embodied Agents Take Control is 68.3%–78%; if it is true, the investment in special navigation training needs to be reassessed. If it is not true, the source of the gap needs to be clarified.
- **will review** after the main text is available: the complete experimental parts of WNM-3D, SAP-Nav, and LifelongCrossNav (the abstracts of this issue have been truncated).
- **Adjust the crawling configuration**: The noise ratio of search keywords in the direction of drones is relatively high (section 5.4 accounts for about 27% of this issue); at the same time, the arXiv single limit of 100 items has been filled under the 15-day window. It is recommended to shorten the crawling interval or increase the upper limit.

**low priority**

- **Suspended**: The operation-specific work (tactile, dexterous, contact-intensive tasks) in Section 5.3 and the acceleration direction of driving reasoning in Section 5.4. There is no clear migration path to ground navigation in this issue.

---

**Disclaimer**: This report is based on the public abstracts captured by RSS and the interpretation of WeChat public accounts. Most arXiv abstracts were truncated during the crawling, and the experimental conclusions have not been verified by the original text. All numbers marked "WeChat public account interpretation" are from third-party interpretations rather than the original text of the paper. Please check the original document before citing.
