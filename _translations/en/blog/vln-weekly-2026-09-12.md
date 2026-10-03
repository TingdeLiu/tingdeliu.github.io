---
layout: post
lang: en
translation_id: vln-weekly-2026-09-12
permalink: /en/vln-weekly-2026-09-12/
source_path: _posts/weekly-reports/2026-09-12-VLN-Weekly.md
source_url: /vln-weekly-2026-09-12/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-09-02 to 2026-09-10)"
date: 2026-09-12
period_start: 2026-09-02
period_end: 2026-09-10
issue_number: 4
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "All 46 new entries come from arXiv; WeChat rate limiting leaves both subscribed accounts with 0 collected entries. Only O2C-Nav reports R2R-CE, reducing large-model calls to one per decision step but giving no verifiable performance figures in the abstract. The strongest navigation evidence comes from OmniNav (real-robot pick-and-place: 53.3% to 71.7%) and OVMAN (11.4% zero-shot ObjectNav success on change-defined goals; 0.000 / 0.025 at locations where objects have left). Evaluation assumptions are being challenged, while 10 of the 46 works primarily use LIBERO."
---
## 1. Key conclusions
{: id="一本期结论"}

- **Among the 46 new entries in this issue, only 1 reports R2R-CE/RxR-CE results** (O2C-Nav), and another 1 reports VLN-CE results (MobileVLA-R1 2.0). The number of papers on command-following continuous environment navigation is at a low level, while there are 3 new navigation evaluation/benchmark works (NavArena, OVMAN, and manipulation-oriented MEMOBench). This digest's assessment: The increase in ground language navigation in this issue comes more from "how to evaluate" rather than "how to do".
- The inference cost of **zero-shot VLN-CE is addressed head-on.** O2C-Nav compresses large model calls per decision step into one, the path is training-free structured waypoint generation, and candidate waypoints are drawn directly on the RGB image as visual markers, which are executed by MLLM point selection and FMM planner. The paper claims that it outperforms existing zero-shot methods in R2R-CE and RxR-CE, but the abstract does not give verifiable values. For real robot deployments, the number of calls rather than pure success rate is the current practical constraint.
- **navigation evaluation is shifting from "static scene + final success rate" to "scenes will change and memory will expire".** OVMAN constructs goals defined by the change itself such as "go to the removed chair" and "go to the original position of the vase". It reports that the zero-shot object target navigator is only 11.4% successful and is close to failure at the position where the object has left (former 0.000, removed 0.025); OmniNav uses incrementally updateable 3D Object-scene memory plus Bayesian belief correction addresses the same problem. This digest's assessment: Static semantic maps are the most vulnerable link in the current ground navigation system. These two works point to it from both the evaluation and method ends.
- **LIBERO's ability to distinguish methods is being openly questioned.** LIBERO-RECOVER points out that the SOTA method is close to 100% success rate on LIBERO, but success under ideal conditions does not equal true robustness. 10 of the 46 articles in this issue use LIBERO as the main benchmark. This digest's assessment: In the subsequent screening, the marginal information content of the work that only reports LIBERO numbers has dropped significantly, and priority should be given to whether it comes with evidence of failure recovery or real robot.
- **humanoid whole-machine navigation appears in the form of full-body VLA.** TANGO directly predicts 29-degree-of-freedom joint space motion, rewriting navigation from 2D path planning to continuous geometric adaptation that requires arm placement, torso adjustment, and gait adjustment. After pure simulation training, it is deployed in zero-shot on Unitree G1. This changes the definition of "travelability" and is not directly applicable to wheeled platforms, but it has reference value for modeling the accessibility of narrow and messy scenes.

> There are 0 new entries in the WeChat public account source (we-mp-rss) in this issue. The reason is the WeChat public platform rate limiting: both subscription accounts (vision-language navigation, embodied intelligence navigation) have completely entered the collection process, and the log reports `rate limit` (translated log message) after retrying, `frequencey control, stop at 0`, with a total of 0 updates. Not caused by keyword filtering. All 46 articles in this issue are from arXiv. They are all independent work and there is no cross-source duplication.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [O2C-Nav](https://arxiv.org/abs/2609.06476v1)** · zero-shot VLN-CE (continuous environment)
  - Contribution: Only one MLLM call per step, training-free waypoint generation + waypoints as visual markers for images.
  - Evidence: R2R-CE and RxR-CE are superior to existing zero-shot methods; **abstract does not provide a verifiable number**.
  - Reason: This issue is the only work reported on R2R-CE/RxR-CE, and it directly addresses the deployment bottleneck of inference delay; the code address has been given.
- **A2 · [OmniNav](https://arxiv.org/abs/2609.08159v1)** · Long-term goal navigation in dynamic environment
  - Contribution: continuous inference of the combination of scene validity, target belief, and interactive accessibility.
  - Evidence: Semantic ObjectNav and fine-grained instance navigation have the highest success rate (no numerical value is given); real robot pick-and-place increased from 53.3% to 71.7% (compared to the modified open-loop baseline).
  - Reason: It is rare to write "memory will expire" and "search failure is also evidence" into the navigation work of the method, and there is a real robot number.
- **A3 · [OVMAN](https://arxiv.org/abs/2609.06424v1)** · Open vocabulary navigation with two visits, change definition goals
  - Contribution: New tasks and benchmarks: target specified by the scene change itself, including "object has left" gap query.
  - Evidence: 219 two-interview plots, all of which have been verified three times by the oracle and can be solved; zero-shot object target navigator 11.4%, former 0.000, removed 0.025; two-interview reference agent 45.2%, embodied oracle 99.5%.
  - Reason: Quantify the failure mode of the current method to nearly zero, and the error decomposition points to the implementation of open vocabulary instances rather than geometry.
- **A4 · [TANGO](https://arxiv.org/abs/2609.09158v1)** · Humanoid full body in messy indoor space vision-language navigation
  - Contribution: Direct prediction of 29-DoF joint motions from verbal commands and first-person RGB.
  - Evidence: In the simulation, it is said that the vision-language navigation is optimal and better than the modular baseline (no value is given); Unitree G1 zero-shot real robot does not use any real robot navigation data for training.
  - Reasons: Extend traversability from 2D occupancy grids to full body geometry; synthetic data pipeline from simulation to real robot worth tearing apart.
- **A5 · [MobileVLA-R1 2.0](https://arxiv.org/abs/2609.06251v1)** · Language-guided navigation + quadruped/humanoid mobile operation
  - Contribution: Supervised thinking chain alignment reinforcement learning, equipped with inference conditional action decoder.
  - Evidence: The average SR on VLN-CE is increased by 1.6 points (compared to MobileVLA-R1); the success rate of the real robot G1 mobile manipulation task is increased by 10.0 points.
  - Reason: This issue is the second work involving VLN-CE, and also provides a comparison between simulation and real robot.
- **A6 · [NavArena](https://arxiv.org/abs/2609.04602v1)** · Automatically constructed goal-oriented navigation datums reconstructed from 3DGS
  - Contributions: Frozen 3DGS rendering + Gaussian density/height derived occupancy cost maps + Semantic targets for multi-view open vocabulary mask boosting.
  - Evidence: Covers more than 2,000 scenarios and generates 22.2 million expert trajectories; tools, evaluation protocols and derivative assets are said to be made public.
  - Reason: Solve the problem of "reconstruction but no traversability and closed-loop agreement", and may become the infrastructure for self-built scenario evaluation.
- **B1 · [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178v1)** · Failure recovery evaluation of operating model
  - Contribution: Pointing out the misleading nature of LIBERO's near-perfect score, turning to failure recovery.
  - Evidence: The abstract states that SOTA has a near 100% success rate on LIBERO.
  - Reason: It affects the subsequent selection criteria for "LIBERO only" work in this database.
- **B2 · [No Free Checker](https://arxiv.org/abs/2609.09250v1)** · Robot Strategy Verifier Review
  - Contribution: About 150 validators organized along two axes: availability and trustworthiness.
  - Evidence: Review Conclusion: Credibility decreases with availability among the four types of validators.
  - Reason: Provide a selection framework for navigation system selection success determination/runtime monitoring.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. O2C-Nav: one large-model call per zero-shot VLN-CE decision step
{: id="1-o2c-nav把零样本-vln-ce-的每步大模型调用压到一次"}

**Problem.** The abstract states that existing zero-shot VLN-CE methods either rely on pre-trained waypoint predictors or require multiple queries of large models at each step, resulting in unbearable inference latency and computational overhead.

**Method.** Three-stage: The training-free structured waypoint generator generates candidate waypoints that are sparse and have historical information; candidate waypoints are directly projected onto RGB images as visual markers as abstract representations; MLLM selects a waypoint or outputs a complete target bounding box in a single call, and then converts it into a collision-free executable path by the fast marching method (FMM) planner

**Evidence.** was evaluated on R2R-CE and RxR-CE, and the paper stated that it exceeded the current optimal zero-shot method. The abstract of **does not give any values such as SR, SPL, NE, etc., nor does it list the control method name**, so it is impossible to enter the direct comparison. Code address https://github.com/kkpsq/O2C-Nav-Code

**Value.** Regarding the real-time constraints of ground robots: reducing "multiple queries" to "one query" is a gain that can be directly converted into control frequency. Draw candidate waypoints on the image as visual markers, transforming the spatial selection problem into a visual identification problem that MLLM is better at, while explicitly carrying historical memory

**Limitations.** There is no verifiable number; the recall rate of the training-free waypoint generator in cluttered or narrow spaces is unknown; FMM planning relies on more reliable depth and occupancy estimates, and the performance summary of the real robot under noise is not covered; the trigger frequency and failure mode of the fallback bounding-box mechanism are not explained

**Recommendation.** **gives priority to intensive reading and reproduction of**. This is the only R2R-CE/RxR-CE job in this issue that has code. When reproducing, first confirm the R2R-CE value and control settings in the main text of the paper (note that the R2R-CE continuous environment and discrete R2R are distinguished, the two are not directly comparable), and then measure the single-step end-to-end delay

### 2. OmniNav: accounting for outdated memory in navigation state inference
{: id="2-omninav把记忆会过期写进导航状态推断"}

**Problem.** The three types of states in long-term goal navigation are only conditionally valid: the scene representation becomes stale after the object moves or disappears; search failure will change the belief of the target position; the geometrically convenient end point may not be feasible for subsequent operations.

**Method.** formalizes long-term navigation as continuous inference of factorized task state posteriors, combining scene validity, goal belief and interaction feasibility. In terms of representation, an updateable 3D object scene memory is incrementally constructed; in terms of exploration, evidence-aware Bayesian belief modification is introduced, a perception-dependent regional prior is derived from the semantic context, and failed searches are incorporated into the posterior as negative evidence to guide frontier point selection; in terms of interaction, operational accessibility and collision constraints are incorporated into navigation endpoint selection, and execution feedback is propagated through hierarchical closed-loop recovery

**Evidence.** Semantic ObjectNav and fine-grained instance navigation benchmarks claim that the success rate is the highest among the comparison methods, **, but the abstract does not give specific values and the name of the comparison method**; it is said to remain robust to target relocation; the real-world pick-and-place success rate increased from 53.3% to 71.7%, and the comparison is the modified open-loop baseline. Project page https://omni-nav.github.io/

**Value.** "Treat unsuccessful searches as negative evidence" is a directly portable mechanism, in contrast to most current systems that only do positive semantic frontier selection. Incorporating operational accessibility into navigation endpoint selection is a practical engineering benefit to the mobile operating platform.

**Limitations.** No numerical results are provided for the simulation benchmark and it is impossible to judge the extent of improvement; the comparison of real robot 53.3%→71.7% is the "modified open-loop baseline" rather than similar closed-loop methods. The advantage may partly come from the closed-loop itself rather than Bayesian belief modification; the update of 3D object memory relies on reliable object-level detection and association

**Recommendation.** **Read the method part intensively, focusing on the specific form of belief modification**. Consider ablating the negative evidence mechanism separately and then connecting it to the existing frontier exploration module.

### 3. OVMAN: scene changes as navigation goals
{: id="3-ovman把场景变化本身作为导航目标"}

**Problem.** The goal of the existing navigation benchmark is defined in the world currently seen by the agent. The existing two-visit benchmark evaluates recall or rearrangement, not navigation. The abstract states that there is no basis for expressing "go to the chair that was removed" or "go to the original position of the vase"

**Method.** Define a new task: the agent first patrols the scene and returns after a scripted change occurs, needing to navigate to the target specified by the change itself. The answer for two of the six change relationships is that the object has moved away from a position where there are no more detectable objects. Published 219 two-interview episodes, each completed three times by the oracle agent to verify solvability

**Evidence.** The zero-shot object target navigator has a rate of 11.4% reaching the change definition target, and basically fails at the vacant position (former 0.000, removed 0.025); the rate of self-maintained open vocabulary map answering past tense queries is about one-third of that when the same map is read in two visits, even if landing information is provided; the success rate of simple two-visit reference agent navigation is 45.2%, and the embodied oracle is 99.5%. Error decomposition attributes the remaining difficulty to the implementation of the open word list instance, rather than the answer choice after the geometry or position is known

**Value.** gives a set of clear failure scenarios where existing methods approach zero scores, and the errors have been decomposed into specific links. For home service ground robots, "things being moved" is the norm rather than an edge case.

**Limitations.** changes are scripted and may not be consistent with the distribution of changes in real families; the scale of 219 plots is limited; the gap between the reference agent 45.2% and the oracle 99.5% shows that the task itself still has a large unsolved space and should not be used as the main indicator in the short term

**Recommendation.** **is added to the benchmark tracking list**. There is no need to invest in reproduction in the near future, but when evaluating the map aging problem of the self-developed navigation system, the structure method of its "past tense query" can be borrowed

### 4. TANGO: from 2D paths to full-body geometric adaptation
{: id="4-tango把人形导航从-2d-路径规划改写为全身几何适配"}

**Problem.** Humanoid traffic in a cluttered indoor environment is not a 2D path planning problem. It requires continuous, geometry-aware whole-body adaptation, including arm placement, torso adjustment and gait adjustment, in order to move without collision in a complex three-dimensional space.

**Method.** Given natural language instructions and first-person RGB observations, directly predicts 29 degrees of freedom joint space motion for downstream full-body control. The training is completely conducted in simulation. Through global path planning, kinematic whole-body motion generation, obstacle-aware motion editing and reinforcement learning-based tracking, a variety of collision-free traffic behaviors are synthesized, providing dynamically feasible action supervision for language-conditioned whole-body strategies.

**Evidence.** is said to have achieved optimal performance in vision-language navigation in a large number of simulation experiments, and is better than the strong modular baseline in difficult scenes that need to avoid obstacles. The abstract of **does not give the numerical value and the specific benchmark name**; it is deployed in zero-shot on Unitree G1 and is said to have robustly completed language guidance in real messy scenes without using any real robot navigation data training.

**Value.** is not directly applicable to wheeled ground platforms, but the judgment that "travelability depends on the body configuration rather than projection occupation" can be transferred: for mobile platforms with robotic arms, the navigation endpoint and path are also subject to the body posture constraints (echoing OmniNav's inclusion of operational accessibility in endpoint selection). A feasible supervised pipeline for synthetic dynamics is a reusable data generation idea

**Limitations.** There is no numerical evidence, and the "optimal" cannot be verified; the training is entirely in simulation, and the real robot conclusion is a qualitative description; the summary of the safety and recovery mechanism of 29-DoF joint space action is not covered

**Recommendation.** **trace, and** is disassembled according to its data synthesis pipeline. After the main text is published, confirm the specific benchmarks used in its VLN evaluation (the abstract does not specify whether it is the R2R-CE series and should not be comparable by default)

### 5. NavArena: converting 3DGS reconstructions into closed-loop navigation benchmarks
{: id="5-navarena把-3dgs-重建变成可闭环评测的导航基准"}

**Problem.** Fixed 3D Gaussian splash reconstruction provides realistic new perspectives, but lacks traversability constraints, valid targets and closed-loop protocols required for navigation evaluation

**Method.** has three components: the frozen 3DGS model provides first-person RGB-D rendering; the occupancy cost map is derived from Gaussian density and height statistics, supporting reachability and collision queries; and the semantic target candidates are obtained through multi-view open vocabulary mask promotion. The three jointly support the automatic generation and unified closed-loop evaluation of goal-oriented navigation plots.

**Evidence.** covers more than 2,000 scenes and generates 22.2 million expert trajectories; it tests the derived navigation representation through spatial and semantic evaluation, and uses policy playback to demonstrate the diagnostic value of the unified evaluation protocol. Said that all benchmark generation tools, evaluation protocols and derivative assets will be publicly released

**Value.** If is open sourced as scheduled, it can significantly lower the threshold for "using self-collected scenes for navigation evaluation" - the existing process usually requires Habitat/MP3D type of datasets with semantic and accessible annotations. It is of direct significance to teams that want to conduct closed-loop evaluations on their own experimental sites.

**Limitations.** The abstract does not quantitatively verify whether the occupancy map derived from Gaussian density and height statistics agrees with actual traversability; the quality of 22.2 million expert trajectories depends on the occupancy map, and there is error propagation; the abstract does not specify a release date or license

**Recommendation.** **is added to the tracking list, waiting for the code release**. After release, priority will be given to verifying the reliability of the occupancy cost map in low-obstacle and glass/mirror areas.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Air-ground coordination (including UGV): [AGC-VLN](https://arxiv.org/abs/2609.03483v3)**
  - Mechanism: Render the global bird's-eye view of the drone, the pose reported by the UGV, and the anchored target of the VLM together into a shared bird's-eye view of the CAR/GOAL mark with distance annotation, and the UGV uses the frozen VLM to plan the path accordingly.
  - Integration point: Interface design to supplement global spatial context for first-person ground platforms.
  - Prerequisites and risks: CARLA simulation results (Town10HD 100 closed-loop plots, joint success rate 77.0%, 50.0% improvement for the weaker single UAV, 27.0 points higher, 24.0 points higher than the strongest published single baseline Travel UAV 53.0%); there is no corresponding overhead source for indoor scenes.
- **Drone VLN: [AirAnchor](https://arxiv.org/abs/2609.08442v1)**
  - Mechanism: Spatial anchor points bridge the local and the global world: query-driven anchor point landing, persistent object spatial memory, and navigation agents that explicitly integrate two scales.
  - Integration point: Isomorphic to OmniNav’s object memory idea, it can be used as the second design reference for “persistent object knowledge base + landmark prior retrieval”.
  - Prerequisites and risks: Evaluated on AerialVLN, urban outdoor scale; no numerical values are given in the abstract.
- **world model: [WorldAgen](https://arxiv.org/abs/2609.08162v1)**
  - Mechanism: Sampling exploration actions during deployment, collecting real state transitions, and training the world model during lightweight testing.
  - Integration point: Online adaptation of navigation policy after entering a new building.
  - Prerequisites and risks: Verified on CALVIN and LIBERO, both are manipulation tasks; the safety cost of sampling exploration actions in navigation is higher.
- **world model: [ProWAM](https://arxiv.org/abs/2609.06578v1)**
  - Mechanism: Using execution progress as an explicit intermediate representation, the use of imagined futures is modulated between and within progress respectively.
  - Access position: The gate control of "when to trust predictions and when to rely on current observations" in long-term navigation.
  - Prerequisites and risks: It is said to be consistent improvement on the strong VLA/WAM baseline, and the summary does not give the numerical value and benchmark name.
- **failure detection and recovery: [VLA-Corrector](https://arxiv.org/abs/2609.06508v1)**
  - Mechanism: Stage-aware failure verification plus prompt recovery, closed-loop correction of fixed strategies.
  - Integration point: Segmentation failure diagnosis in navigation without retraining the master strategy.
  - Prerequisites and risks: Verification on LIBERO; the "stage" division of navigation is not as clear as operation.
- **failure detection and recovery: [Where Success Breaks](https://arxiv.org/abs/2609.06114v1)**
  - Mechanism: Reframing nudge as failure boundary learning: discovering, locating, and exploiting the boundaries where successful behaviors fail.
  - Access position: The idea of constructing negative samples when fine-tuning the navigation policy.
  - Prerequisites and risks: The abstract does not give numerical values; a simulation environment that can automatically generate failure trajectories is required.
- **evaluation methodology: [No Free Checker](https://arxiv.org/abs/2609.09250v1)**
  - Mechanism: A two-axis framework of availability and credibility, plus nine indicators that make the verifier’s claims verifiable.
  - Integration point: Select success evaluators and runtime monitors for the navigation system.
  - Prerequisites and risks: Overview, no experiments; the conclusion "credibility decreases as availability increases" is the overall observation.
- **evaluation methodology: [MEMOBench](https://arxiv.org/abs/2609.07047v1)**
  - Mechanism: Process-level memory evaluation, distinguishing between "forgetting" and "operation failure", not just the final success rate.
  - Integration point: Evaluation design of the navigation memory module - There is also the problem of confusing forgetting with planning failure.
  - Prerequisites and risks: To build for manipulation tasks, the process-level indicators of the navigation version need to be redesigned.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **O2C-Nav** (zero-shot VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.06476v1)
- **OmniNav** (dynamic environment goal navigation · A): See key analysis. [Source](https://arxiv.org/abs/2609.08159v1)
- **OVMAN** (change-aware navigation benchmark · A): See key analysis. [Source](https://arxiv.org/abs/2609.06424v1)
- **TANGO** (humanoid full-body VLN · A): See key analysis. [Source](https://arxiv.org/abs/2609.09158v1)
- **MobileVLA-R1 2.0** (Mobile Robot Control · A): Alignment of thinking chains to enhance reinforcement learning; SR on VLN-CE is increased by 1.6 points on average, and the overall task success rate of real robot G1 mobile operation is increased by 10.0 points. [Source](https://arxiv.org/abs/2609.06251v1)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **NavArena** (Navigation reference generation · A): See key analysis. [Source](https://arxiv.org/abs/2609.04602v1)
- **MEMOBench** (process-level memory evaluation · B): separates forgetting from operation failure and does not rely on the final success rate. [Source](https://arxiv.org/abs/2609.07047v1)
- **No Free Checker** (Verifier Overview · B): About 150 verifiers are classified according to the two axes of availability and credibility, and nine testability indicators are proposed. [Source](https://arxiv.org/abs/2609.09250v1)
- **LIBERO-RECOVER** (Failure Recovery Evaluation · B): Point out the misleading nature of LIBERO’s nearly full score, and turn to failure recovery capability evaluation. [Source](https://arxiv.org/abs/2609.05178v1)
- **RoboSPA** (Space-Process Complexity Evaluation · B): Evaluate the reasoning performance of VLA with increased space and process complexity. [Source](https://arxiv.org/abs/2609.05324v1)
- **Neural symbolic process reasoning for long-term VLA** (task graph and process memory · B): Use explicit task graphs to encode action dependencies and effective transfers, combined with multi-modal process memory. [Source](https://arxiv.org/abs/2609.05369v1)
- **brain-like hierarchical zero-shot task reasoning framework** (hierarchical planning and verification · B): explicit object state reasoning plus atomic action combination, cost sorting and closed-loop execution verification; clear board 10/10, catch and place 10/10, pyramid stacking 4/5. [Source](https://arxiv.org/abs/2609.05985v1)
- **WorldAgen** (world model training at test time · B): See transferable method. [Source](https://arxiv.org/abs/2609.08162v1)
- **ProWAM** (Progress conditionalization imagination · B): See the migration method. [Source](https://arxiv.org/abs/2609.06578v1)
- **UniMPA** (Memory-Prediction-Action Unification · B): Use the transfer interface of action grounding to jointly handle transfer ambiguity, mismatch between prediction and execution, and mismatch between experience and implementation. [Source](https://arxiv.org/abs/2609.11875v1)
- **An overview of 3D Vision-Language Models** (3D VLM Overview · B): A tutorial-style overview from 3D representation encoding to cross-modal contrast alignment and 3D VLLM, including language-guided 3DGS. [Source](https://arxiv.org/abs/2609.05583v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **VLA-Corrector** (Closed Loop Recovery · B): See Migration Methods. [Source](https://arxiv.org/abs/2609.06508v1)
- **Where Success Breaks** (Failure Boundary Learning · B): See transferable method. [Source](https://arxiv.org/abs/2609.06114v1)
- **FailureSpot** (timestamp-level failure detection · B): Marks efficient timestamp-level failure detection, oriented to long-term execution. [Source](https://arxiv.org/abs/2609.04277v1)
- **ActSafeGuard** (Hard Constraint Guarantee · B): Provides differentiable and training-aligned constraint imposition for flow matching strategies. [Source](https://arxiv.org/abs/2609.11697v1)
- **Reasoning Without Inference Cost** (Training Period Reasoning · B): Use latent semantic scaffolding to obtain reasoning benefits during the training period, and discard it before deployment to eliminate reasoning overhead. [Source](https://arxiv.org/abs/2609.04893v1)
- **What Matters, When?** (Conditional Visual Landing · C): Diagnose the problem of visual target changes with the operation stage and task status as conditional visual landing. [Source](https://arxiv.org/abs/2609.05376v1)
- **RefGuard** (referring identity landing · C): joint target-anchor-reference system landing, dealing with spatial words where similar objects are repeated and reference systems are dependent. [Source](https://arxiv.org/abs/2609.06221v1)
- **ICI-VLA** (Context Mimicry · C): A spatiotemporal alignment demonstration to achieve few-shot test-time adaptation without gradient updates. [Source](https://arxiv.org/abs/2609.07581v1)
- **GIFT** (Target image injection fine-tuning · C): Use the generated target image as a high-level visual guide to inject fine-tuning. [Source](https://arxiv.org/abs/2609.07006v1)
- **LayerRoute** (Layer Hybrid Routing · C): Visual-semantic representation of different layers of VLM routing by action conditions. [Source](https://arxiv.org/abs/2609.06079v1)
- **IMLE-VLA** (Single-step action generation · C): Use IMLE to replace multi-step iterative sampling to eliminate the stop-go motion of the flow matching action head. [Source](https://arxiv.org/abs/2609.10915v1)
- **FreqFM** (Frequency Conditional Flow Matching · C): Explicitly modeling the frequency heterogeneity of action trajectories. [Source](https://arxiv.org/abs/2609.10405v1)
- **Time-frequency geometric cross-attention** (blocked action decoding · C): Treat action chunks as short multivariate trajectories, decoding by frequency and geometric structure instead of linear headers by time step. [Source](https://arxiv.org/abs/2609.09925v1)
- **Large Discrete Policy** (Discrete Behavior Modeling · C): Select actions from a large-scale physically reasonable candidate vocabulary as an alternative to continuous denoising. [Source](https://arxiv.org/abs/2609.07049v1)
- **HuRo** (Human video pre-training · C): A systematic examination of whether robotized human videos can serve as scalable VLA pre-training supervision. [Source](https://arxiv.org/abs/2609.10706v1)
- **RoboDrop** (Post-training data filtering · C): Filter post-training data with local gradient compatibility. [Source](https://arxiv.org/abs/2609.10021v1)
- **ZETA** (cross-embodiment zero-shot migration · C): Provides a unified zero-shot migration definition and a controlled evaluation setting that isolates embodiment changes. [Source](https://arxiv.org/abs/2609.02546v2)
- **VLA-Precision** (real robot online reinforcement learning · C): Asymmetric collaborative bootstrapping alleviates policy drift and large model throughput bottlenecks caused by unreliable value signals. [Source](https://arxiv.org/abs/2609.04355v2)
- **HINT** (Long-term intent inference · C): Infer human intent from general instructions and continuously adapt as visual observations evolve. [Source](https://arxiv.org/abs/2609.02653v2)
- **CR-VLA-Force** (Force-aware compliance control · C): Control-aware compliance VLA for contact-rich operations. [Source](https://arxiv.org/abs/2609.05832v1)
- **FWBC-VLA** (Force-Aware Whole Body Compensation·C): Bridging semantic action generation and physical interaction control for contact-rich mobile operations. [Source](https://arxiv.org/abs/2609.03889v2)
- **DeCAL** (Contact Aware Dexterous Operation · C): Potential co-imagination of contact awareness, handling severe visual occlusion and complex contact dynamics. [Source](https://arxiv.org/abs/2609.09119v1)
- **GloVLA** (global geometry plus local VLA · C): Split end-effector long-range transport with contact-dense interaction upon arrival. [Source](https://arxiv.org/abs/2609.06256v1)
- **FolDeX** (Deformable Object Benchmark · C): A real robot benchmark for dual-arm operation of long-term deformable objects. [Source](https://arxiv.org/abs/2609.10243v1)
- **Language Migration Metrics for Robot Strategies** (Multi-Language VLA · C): Rewrite instructions only with machine translation to add Greek to the Cosmos3 VLA, focusing on the reliability of the measurement tool. [Source](https://arxiv.org/abs/2609.07470v1)

### 5.4 UAVs, communications, and less relevant topics
{: id="54-无人机通信与其他低相关方向"}

- **AGC-VLN** (Air-Ground Coordination VLN · B): See Migration Methods. [Source](https://arxiv.org/abs/2609.03483v3)
- **AirAnchor** (drone zero-shot VLN · B): See Migration Methods. [Source](https://arxiv.org/abs/2609.08442v1)
- **ComVLA** (6G segmentation inference · C): Communication-aware VLA segmentation inference, adapting to wireless link bandwidth constraints. [Source](https://arxiv.org/abs/2609.07838v1)
- **Modal decoupled federated learning** (6G privacy protection · C): Modal decoupled federated learning for 6G embodied intelligence. [Source](https://arxiv.org/abs/2609.09591v1)
- **few-sample VLM soft prompt** (overseas target detection · C): Use soft prompts to perform ten-image few-sample adaptation in aerial photography, industry, medical and other foreign scenes. [Source](https://arxiv.org/abs/2609.11310v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

There are no informational items in this issue. WeChat public account source (we-mp-rss) added 0 items, and both subscription account collections returned 0 items due to rate limiting on the WeChat public platform.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- The assumptions of the **navigation review are being systematically dismantled.** OVMAN is about "the scene remains unchanged between two visits", NavArena is about "only annotated datasets such as Habitat/MP3D can do closed-loop evaluation", LIBERO-RECOVER and MEMOBench are about "the final success rate is enough to generalize the ability". The four directions are different but the common result is: the explanatory power of a single success rate number is declining, and the evaluation protocol needs to be recorded in the report at the same time.
- **"Memory effectiveness" changes from an implicit assumption to an explicit modeling object.** OmniNav's updateable 3D object memory and failed search negative evidence, AirAnchor's persistent object spatial memory, UniMPA's visual-action memory library, and MEMOBench's process-level memory evaluation. The four works deal with the same thing from the four perspectives of navigation, air navigation, operation and evaluation. This digest's assessment: This is the most concentrated methodological convergence point in this issue.
- **inference cost is treated as a first-class constraint rather than a post hoc optimization.** O2C-Nav compresses the number of large model calls, Reasoning Without Inference Cost moves the reasoning benefits to the training period, IMLE-VLA eliminates multi-step sampling, and ComVLA handles wireless link bandwidth limitations. Four of them attack the same problem from different levels. For a real robot navigation deployment, the actual value of this type of work may be greater than a small improvement in the baseline score.
- **Navigation's "traversability" definition begins to be bound to the embodiment configuration.** TANGO's full-body adaptation and OmniNav's inclusion of operational accessibility into end-point selection both indicate that abstracting the robot into a disk is no longer sufficient in mobile manipulation scenarios.

### Research gaps
{: id="研究空白"}

- Zero-shot methods on **R2R-CE lack unified latency-accuracy joint reporting.** O2C-Nav uses the number of calls as a selling point but does not give a precision value in the summary. Most jobs only report SR/SPL but not single-step time consumption, making it impossible to judge "how much faster but worse". This is a low-cost review contribution.
- **Navigation after scene changes currently only has benchmarks and no methods.** The gap between 45.2% of OVMAN’s reference agent and 99.5% of the oracle’s main bottleneck is attributed to the implementation of open vocabulary examples by its error decomposition. There is no method to specifically deal with "the target is no longer in place" in ground-based VLN.
- **simulation to real robot navigation migration lacks quantitative comparison.** TANGO’s real robot conclusion is a qualitative description, and OmniNav’s real robot comparison is an open-loop baseline. There is no work in this issue that presents paired values of the same method on simulation benchmarks and real robots.

### Suggested actions
{: id="建议动作"}

**High priority**

- **carefully read and reproduced**: O2C-Nav: Confirm the R2R-CE/RxR-CE values and comparison settings in the text, and measure the single-step end-to-end delay; the code has been made public.
- **Read closely**: OmniNav: Dismantle the specific form of Bayesian belief modification and negative evidence mechanism, and evaluate the feasibility of separately connecting to the existing frontier exploration module.

**medium priority**

- **joins benchmark tracking**: OVMAN, NavArena: waiting for the release of data and tools. The former is used to evaluate the map aging problem of self-developed systems, and the latter is used for closed-loop evaluation of its own venues.
- **Track**: TANGO: After the main text is published, the benchmark used for VLN evaluation is confirmed (the abstract does not specify whether it belongs to the R2R-CE series and cannot be compared by default); the simulation data synthesis pipeline is disassembled.
- **is used as a selection reference to read**: No Free Checker: Select the verifier type for the success determination and runtime monitoring of the navigation system.

**low priority**

- **Suspended**: VLA method work that only reports LIBERO and no evidence of failure recovery or real robot (most of the 10 articles in this issue) - LIBERO-RECOVER has pointed out that this benchmark is close to saturation.
