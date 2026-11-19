---
layout: post
lang: en
translation_id: vln-weekly-2026-09-12
permalink: /en/vln-weekly-2026-09-12/
source_path: _posts/weekly-reports/2026-09-12-VLN-Weekly.md
source_url: /vln-weekly-2026-09-12/
source_revision_date: 2026-10-24
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly · Issue 4"
date: 2026-09-12
last_modified_at: 2026-10-24
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

- **Of the 46 new entries in this issue, only 1 reports R2R-CE/RxR-CE results** (O2C-Nav), and another 1 reports VLN-CE results (MobileVLA-R1 2.0). Papers on instruction-following navigation in continuous environments are at a low point, while there are 3 new navigation evaluation/benchmark works (NavArena, OVMAN, and the manipulation-oriented MEMOBench). This digest's assessment: the growth in ground language navigation in this issue comes more from "how to evaluate" than from "how to do it".
- **The inference cost of zero-shot VLN-CE is tackled head-on.** O2C-Nav compresses the large-model calls per decision step to one: a training-free structured generator produces candidate waypoints, which are drawn directly on the RGB image as visual markers, and an MLLM picks one for an FMM planner to execute. The paper claims to outperform existing zero-shot methods on R2R-CE and RxR-CE, but the abstract gives no verifiable values. For real-robot deployment, the number of calls, rather than raw success rate, is the practical constraint right now.
- **Navigation evaluation is shifting from "static scene + final success rate" to "scenes change and memory goes stale".** OVMAN builds goals defined by the change itself, such as "go to the chair that was removed" or "go to where the vase used to be". It reports that a zero-shot object-goal navigator succeeds only 11.4% of the time and almost always fails at positions the object has left (former 0.000, removed 0.025); OmniNav addresses the same problem with an incrementally updatable 3D object-scene memory plus Bayesian belief revision. This digest's assessment: static semantic maps are the most fragile link in current ground navigation systems, and these two works point at it from the evaluation end and the method end.
- **LIBERO's ability to discriminate between methods is being openly questioned.** LIBERO-RECOVER points out that SOTA methods are close to 100% success on LIBERO, but success under ideal conditions does not equal true robustness. 10 of the 46 works in this issue use LIBERO as their main benchmark. This digest's assessment: in future screening, work that reports only LIBERO numbers carries much less marginal information, and priority should go to whether it comes with evidence of failure recovery or real-robot results.
- **Humanoid whole-body navigation appears in the form of a full-body VLA.** TANGO directly predicts 29-degree-of-freedom joint-space motion, recasting navigation from 2D path planning into continuous geometric adaptation that involves arm placement, torso adjustment, and gait adjustment. Trained purely in simulation, it is deployed zero-shot on a Unitree G1. This changes the definition of "traversability" and does not apply directly to wheeled platforms, but it has reference value for modeling traversability in narrow, cluttered scenes.

> There are 0 new entries from the WeChat source (we-mp-rss) in this issue. The cause is rate limiting on the WeChat Official Accounts Platform: both subscribed accounts (vision-language navigation, embodied intelligence navigation) fully entered the collection process, and after retries the log reports `rate limit` (translated log message) and `frequencey control, stop at 0`, with 0 updates in total. It was not caused by keyword filtering. All 46 entries in this issue come from arXiv, are independent works, and have no cross-source duplicates.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [O2C-Nav](https://arxiv.org/abs/2609.06476v1)** · Zero-shot VLN-CE (continuous environments)
  - Contribution: Only one MLLM call per step; training-free waypoint generation, with waypoints drawn on the image as visual markers.
  - Evidence: Outperforms existing zero-shot methods on R2R-CE and RxR-CE; **the abstract gives no verifiable numbers**.
  - Why it matters: The only work in this issue reporting on R2R-CE/RxR-CE, and it addresses the deployment bottleneck of inference latency head-on; the code address is provided.
- **A2 · [OmniNav](https://arxiv.org/abs/2609.08159v1)** · Long-horizon goal navigation in dynamic environments
  - Contribution: Continuous inference that combines scene validity, goal belief, and interaction feasibility.
  - Evidence: Highest success rate on semantic ObjectNav and fine-grained instance navigation (no values given); real-robot pick-and-place improves from 53.3% to 71.7% (against a modified open-loop baseline).
  - Why it matters: Rare among navigation works for writing "memory goes stale" and "a failed search is also evidence" into the method, and it has real-robot numbers.
- **A3 · [OVMAN](https://arxiv.org/abs/2609.06424v1)** · Open-vocabulary navigation over two visits, with goals defined by changes
  - Contribution: A new task and benchmark: goals specified by the scene change itself, including queries about an "object has left" vacancy.
  - Evidence: 219 two-visit episodes, each verified solvable three times by an oracle; a zero-shot object-goal navigator reaches 11.4%, with former 0.000 and removed 0.025; a two-visit reference agent reaches 45.2% and the embodied oracle 99.5%.
  - Why it matters: Quantifies a failure mode where current methods score near zero, and the error decomposition points to open-vocabulary instance grounding rather than geometry.
- **A4 · [TANGO](https://arxiv.org/abs/2609.09158v1)** · Whole-body humanoid vision-language navigation in cluttered indoor spaces
  - Contribution: Directly predicts 29-DoF joint motions from language instructions and first-person RGB.
  - Evidence: In simulation it is said to achieve the best vision-language navigation performance and beat modular baselines (no values given); it is deployed zero-shot on a real Unitree G1, with no real-robot navigation data used for training.
  - Why it matters: Extends traversability from 2D occupancy grids to whole-body geometry; its simulation-to-real synthetic data pipeline is worth taking apart.
- **A5 · [MobileVLA-R1 2.0](https://arxiv.org/abs/2609.06251v1)** · Language-guided navigation + quadruped/humanoid mobile manipulation
  - Contribution: Reinforcement learning aligned with supervised chains of thought, with a reasoning-conditioned action decoder.
  - Evidence: Average SR on VLN-CE improves by 1.6 points (over MobileVLA-R1); success on real-robot G1 mobile manipulation tasks improves by 10.0 points.
  - Why it matters: The second work in this issue involving VLN-CE, and it also provides a simulation vs. real-robot comparison.
- **A6 · [NavArena](https://arxiv.org/abs/2609.04602v1)** · Automatically constructed goal-oriented navigation benchmarks from 3DGS reconstructions
  - Contribution: Frozen 3DGS rendering, occupancy cost maps derived from Gaussian density and height, and semantic goals from multi-view open-vocabulary mask lifting.
  - Evidence: Covers more than 2,000 scenes and generates 22.2 million expert trajectories; the tools, evaluation protocol, and derived assets are said to be released.
  - Why it matters: Solves the "reconstructed but no traversability or closed-loop protocol" problem, and may become infrastructure for evaluating on self-built scenes.
- **B1 · [LIBERO-RECOVER](https://arxiv.org/abs/2609.05178v1)** · Failure-recovery evaluation for manipulation models
  - Contribution: Points out that LIBERO's near-perfect scores are misleading and shifts the focus to failure recovery.
  - Evidence: The abstract states that SOTA methods reach near 100% success on LIBERO.
  - Why it matters: It affects how we screen "LIBERO-only" work for this digest from now on.
- **B2 · [No Free Checker](https://arxiv.org/abs/2609.09250v1)** · Survey of robot policy verifiers
  - Contribution: Organizes about 150 verifiers along two axes: availability and trustworthiness.
  - Evidence: Survey conclusion: across the four verifier types, credibility decreases as availability increases.
  - Why it matters: Provides a selection framework for success detection and runtime monitoring in navigation systems.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. O2C-Nav
{: id="1-o2c-nav把零样本-vln-ce-的每步大模型调用压到一次"}

**One large-model call per decision step for zero-shot VLN-CE.**

**Problem.** The abstract states that existing zero-shot VLN-CE methods either rely on pretrained waypoint predictors or query large models several times per step, which brings unacceptable inference latency and computational overhead.

**Method.** Three stages. A training-free structured waypoint generator produces sparse candidate waypoints that carry history information; the candidates are projected directly onto the RGB image as visual markers, serving as an abstract representation; in a single call, the MLLM selects a waypoint or outputs a full goal bounding box, which a fast marching method (FMM) planner converts into a collision-free executable path.

**Evidence.** Evaluated on R2R-CE and RxR-CE; the paper states that it exceeds the current best zero-shot methods. **The abstract gives no values such as SR, SPL, or NE and does not name the compared methods**, so a direct comparison is impossible. Code address https://github.com/kkpsq/O2C-Nav-Code

**Value.** It speaks to the real-time constraints of ground robots: reducing "multiple queries" to "one query" is a gain that converts directly into control frequency. Drawing candidate waypoints on the image as visual markers turns spatial selection into a visual recognition problem that MLLMs handle better, while carrying history memory explicitly.

**Limitations.** No verifiable numbers; the recall of the training-free waypoint generator in cluttered or narrow spaces is unknown; FMM planning depends on reasonably reliable depth and occupancy estimates, and the summary does not cover real-robot performance under noise; the trigger frequency and failure modes of the fallback bounding-box mechanism are not described.

**Recommendation.** **Prioritize reading and reproducing it.** This is the only R2R-CE/RxR-CE work in this issue with code. When reproducing, first confirm the R2R-CE values and comparison settings in the main text (note that continuous R2R-CE and discrete R2R are distinct and not directly comparable), then measure the per-step end-to-end latency.

### 2. OmniNav
{: id="2-omninav把记忆会过期写进导航状态推断"}

**Accounting for stale memory in navigation state inference.**

**Problem.** Three kinds of state in long-horizon goal navigation are only conditionally valid: the scene representation goes stale after an object moves or disappears; a failed search changes the belief about the goal's location; and a geometrically convenient endpoint may not be feasible for the subsequent manipulation.

**Method.** It formalizes long-horizon navigation as continuous inference over a factorized task-state posterior combining scene validity, goal belief, and interaction feasibility. For representation, it builds an updatable 3D object-scene memory incrementally. For exploration, it introduces evidence-aware Bayesian belief revision: a perception-dependent regional prior is derived from semantic context, and failed searches enter the posterior as negative evidence to guide frontier selection. For interaction, manipulation reachability and collision constraints are incorporated into endpoint selection, and execution feedback is propagated through hierarchical closed-loop recovery.

**Evidence.** On the semantic ObjectNav and fine-grained instance navigation benchmarks it claims the highest success rate among compared methods, **but the abstract gives no specific values or names of the compared methods**; it is said to remain robust to goal relocation; real-world pick-and-place success improves from 53.3% to 71.7%, against a modified open-loop baseline. Project page https://omni-nav.github.io/

**Value.** "Treat unsuccessful searches as negative evidence" is a directly portable mechanism, in contrast to most current systems that only do positive semantic frontier selection. Building manipulation reachability into navigation endpoint selection is a practical engineering benefit for mobile manipulation platforms.

**Limitations.** No numerical results are given for the simulation benchmarks, so the size of the improvement cannot be judged; the real-robot comparison of 53.3%→71.7% is against a "modified open-loop baseline" rather than a comparable closed-loop method, so the advantage may partly come from closing the loop itself rather than from Bayesian belief revision; and updating the 3D object memory relies on reliable object-level detection and association.

**Recommendation.** **Read the method section closely, focusing on the specific form of belief revision.** Consider ablating the negative-evidence mechanism on its own and then attaching it to an existing frontier exploration module.

### 3. OVMAN
{: id="3-ovman把场景变化本身作为导航目标"}

**Scene changes as navigation goals.**

**Problem.** Goals in existing navigation benchmarks are defined in the world the agent currently sees. Existing two-visit benchmarks evaluate recall or rearrangement, not navigation. The abstract states that there was no way to express "go to the chair that was removed" or "go to where the vase used to be".

**Method.** It defines a new task: the agent first patrols the scene and returns after a scripted change, and must navigate to a goal specified by the change itself. For two of the six change relations, the answer is a position the object has left, where nothing detectable remains. It releases 219 two-visit episodes, each completed three times by an oracle agent to verify solvability.

**Evidence.** A zero-shot object-goal navigator reaches change-defined goals 11.4% of the time and essentially fails at vacated positions (former 0.000, removed 0.025); the rate at which a self-maintained open-vocabulary map answers past-tense queries is about one third of that when the same map is read at the second visit, even with grounding information provided; a simple two-visit reference agent reaches 45.2% navigation success and the embodied oracle 99.5%. The error decomposition attributes the remaining difficulty to open-vocabulary instance grounding, not to geometry or to choosing the answer once the position is known.

**Value.** It provides a set of clear failure scenarios in which existing methods score near zero, with errors decomposed into specific stages. For home service robots on the ground, "things get moved" is the norm, not an edge case.

**Limitations.** The changes are scripted and may not match the distribution of changes in real homes; 219 episodes is a limited scale; the gap between the reference agent's 45.2% and the oracle's 99.5% shows the task still has a lot of unsolved room, so it should not serve as a primary metric in the short term.

**Recommendation.** **Add it to the benchmark tracking list.** No need to invest in reproduction soon, but when evaluating map staleness in your own navigation system, you can borrow the structure of its "past-tense queries".

### 4. TANGO
{: id="4-tango把人形导航从-2d-路径规划改写为全身几何适配"}

**From 2D paths to whole-body geometric adaptation.**

**Problem.** Humanoid traversal of a cluttered indoor environment is not a 2D path planning problem. It requires continuous, geometry-aware whole-body adaptation, including arm placement, torso adjustment, and gait adjustment, to move without collision in complex three-dimensional space.

**Method.** Given natural-language instructions and first-person RGB observations, it directly predicts 29-degree-of-freedom joint-space motion for downstream whole-body control. Training is done entirely in simulation. Global path planning, kinematic whole-body motion generation, obstacle-aware motion editing, and RL-based tracking synthesize a variety of collision-free traversal behaviors, providing dynamically feasible action supervision for a language-conditioned whole-body policy.

**Evidence.** It is said to achieve the best vision-language navigation performance across extensive simulation experiments and to beat strong modular baselines in difficult scenes that require obstacle avoidance. **The abstract gives no values or specific benchmark names**; it is deployed zero-shot on a Unitree G1 and is said to complete language-guided navigation robustly in real cluttered scenes without any real-robot navigation training data.

**Value.** It does not apply directly to wheeled ground platforms, but the judgment that "traversability depends on body configuration rather than projected footprint" transfers: for mobile platforms with arms, navigation endpoints and paths are also constrained by body posture (echoing OmniNav's inclusion of manipulation reachability in endpoint selection). The feasible synthetic-dynamics supervision pipeline is a reusable data generation idea.

**Limitations.** No numerical evidence, so "best" cannot be verified; training is entirely in simulation and the real-robot conclusions are qualitative; the summary does not cover safety and recovery mechanisms for 29-DoF joint-space actions.

**Recommendation.** **Track it, and take apart its data synthesis pipeline.** Once the full text is out, confirm the specific benchmarks used in its VLN evaluation (the abstract does not say whether they are the R2R-CE series, so assume they are not comparable).

### 5. NavArena
{: id="5-navarena把-3dgs-重建变成可闭环评测的导航基准"}

**Converting 3DGS reconstructions into closed-loop navigation benchmarks.**

**Problem.** Fixed 3D Gaussian splatting reconstructions provide realistic novel views but lack the traversability constraints, valid goals, and closed-loop protocol needed for navigation evaluation.

**Method.** Three components: a frozen 3DGS model provides first-person RGB-D rendering; an occupancy cost map derived from Gaussian density and height statistics supports reachability and collision queries; and semantic goal candidates come from multi-view open-vocabulary mask lifting. Together they support automatic generation of goal-oriented navigation episodes and unified closed-loop evaluation.

**Evidence.** It covers more than 2,000 scenes and generates 22.2 million expert trajectories; it tests the derived navigation representation through spatial and semantic evaluation, and uses policy playback to show the diagnostic value of the unified evaluation protocol. It says the benchmark generation tools, evaluation protocol, and derived assets will all be released publicly.

**Value.** If released as planned, it could significantly lower the barrier to "evaluating navigation on self-collected scenes": existing workflows usually require Habitat/MP3D-style datasets with semantic and traversability annotations. This matters directly for teams that want closed-loop evaluation at their own test sites.

**Limitations.** The abstract gives no quantitative check of whether the occupancy map derived from Gaussian density and height statistics agrees with real traversability; the quality of the 22.2 million expert trajectories depends on that occupancy map, so errors propagate; the abstract states neither a release date nor a license.

**Recommendation.** **Add it to the tracking list and wait for the code release.** After release, first verify the reliability of the occupancy cost map around low obstacles and glass or mirror areas.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Air-ground collaboration (including UGVs): [AGC-VLN](https://arxiv.org/abs/2609.03483v3)**
  - Mechanism: Renders the drone's global bird's-eye view, the pose reported by the UGV, and the VLM-anchored goal into a shared bird's-eye view with CAR/GOAL markers and distance annotations, from which the UGV plans a route with a frozen VLM.
  - Integration point: An interface design for supplying global spatial context to first-person ground platforms.
  - Prerequisites and risks: Results are from CARLA simulation (Town10HD, 100 closed-loop episodes, joint success rate 77.0%, 27.0 points above the weaker single agent at 50.0% (UAV), and 24.0 points above the strongest published single baseline, Travel UAV at 53.0%); indoor scenes have no equivalent overhead source.
- **UAV VLN: [AirAnchor](https://arxiv.org/abs/2609.08442v1)**
  - Mechanism: Spatial anchors bridge the local and global worlds: query-driven anchor grounding, persistent object spatial memory, and a navigation agent that explicitly integrates the two scales.
  - Integration point: Isomorphic to OmniNav's object memory idea, and can serve as a second design reference for a "persistent object knowledge base + landmark prior retrieval".
  - Prerequisites and risks: Evaluated on AerialVLN at urban outdoor scale; the abstract gives no values.
- **World model: [WorldAgen](https://arxiv.org/abs/2609.08162v1)**
  - Mechanism: Samples exploratory actions at deployment, collects real state transitions, and trains a lightweight world model at test time.
  - Integration point: Online adaptation of a navigation policy after entering a new building.
  - Prerequisites and risks: Validated on CALVIN and LIBERO, both manipulation tasks; sampling exploratory actions costs more in safety terms in navigation.
- **World model: [ProWAM](https://arxiv.org/abs/2609.06578v1)**
  - Mechanism: Uses execution progress as an explicit intermediate representation, modulating the use of imagined futures between and within progress stages.
  - Integration point: Gating "when to trust predictions and when to rely on the current observation" in long-horizon navigation.
  - Prerequisites and risks: Said to improve consistently over strong VLA/WAM baselines; the summary gives no values or benchmark names.
- **Failure detection and recovery: [VLA-Corrector](https://arxiv.org/abs/2609.06508v1)**
  - Mechanism: Stage-aware failure verification plus prompted recovery, correcting a fixed policy in closed loop.
  - Integration point: Stage-wise failure diagnosis in navigation without retraining the main policy.
  - Prerequisites and risks: Validated on LIBERO; the "stage" division is less clear in navigation than in manipulation.
- **Failure detection and recovery: [Where Success Breaks](https://arxiv.org/abs/2609.06114v1)**
  - Mechanism: Reframes nudging as failure-boundary learning: discovering, locating, and exploiting the boundaries at which successful behaviors fail.
  - Integration point: An idea for constructing negative samples when fine-tuning navigation policies.
  - Prerequisites and risks: The abstract gives no values; it needs a simulation environment that can automatically generate failure trajectories.
- **Evaluation methodology: [No Free Checker](https://arxiv.org/abs/2609.09250v1)**
  - Mechanism: A two-axis framework of availability and credibility, plus nine metrics that make a verifier's claims testable.
  - Integration point: Choosing success evaluators and runtime monitors for navigation systems.
  - Prerequisites and risks: A survey with no experiments; the conclusion that "credibility decreases as availability increases" is an overall observation.
- **Evaluation methodology: [MEMOBench](https://arxiv.org/abs/2609.07047v1)**
  - Mechanism: Process-level memory evaluation that distinguishes "forgetting" from "manipulation failure" rather than looking only at final success rate.
  - Integration point: Evaluation design for navigation memory modules, which also suffer from confusing forgetting with planning failure.
  - Prerequisites and risks: Built for manipulation tasks; process-level metrics for a navigation version would need to be redesigned.

## 5. Research roundup by category
{: id="五分类速览"}

Relevance labels: A means directly relevant to ground navigation research, B means the mechanism is transferable, and C is for field observation only.

### 5.1 VLN and ObjectNav
{: id="51-vln-与-objectnav"}

- **O2C-Nav** (zero-shot VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.06476v1)
- **OmniNav** (dynamic environment goal navigation · A): See key analysis. [Source](https://arxiv.org/abs/2609.08159v1)
- **OVMAN** (change-aware navigation benchmark · A): See key analysis. [Source](https://arxiv.org/abs/2609.06424v1)
- **TANGO** (humanoid full-body VLN · A): See key analysis. [Source](https://arxiv.org/abs/2609.09158v1)
- **MobileVLA-R1 2.0** (mobile robot control · A): Reinforcement learning aligned with chains of thought; average SR on VLN-CE improves by 1.6 points, and the overall task success rate of real-robot G1 mobile manipulation improves by 10.0 points. [Source](https://arxiv.org/abs/2609.06251v1)

### 5.2 Memory, planning and evaluation
{: id="52-记忆规划与评测"}

- **NavArena** (navigation benchmark generation · A): See key analysis. [Source](https://arxiv.org/abs/2609.04602v1)
- **MEMOBench** (process-level memory evaluation · B): Separates forgetting from manipulation failure instead of relying on final success rate. [Source](https://arxiv.org/abs/2609.07047v1)
- **No Free Checker** (verifier survey · B): Classifies about 150 verifiers along the two axes of availability and credibility, and proposes nine testability metrics. [Source](https://arxiv.org/abs/2609.09250v1)
- **LIBERO-RECOVER** (failure-recovery evaluation · B): Points out that LIBERO's near-perfect scores are misleading, and turns to evaluating failure-recovery ability. [Source](https://arxiv.org/abs/2609.05178v1)
- **RoboSPA** (spatial-procedural complexity evaluation · B): Evaluates VLA reasoning performance as spatial and procedural complexity increase. [Source](https://arxiv.org/abs/2609.05324v1)
- **Neural symbolic process reasoning for long-term VLA** (task graph and procedural memory · B): Uses explicit task graphs to encode action dependencies and valid transitions, combined with multimodal procedural memory. [Source](https://arxiv.org/abs/2609.05369v1)
- **Brain-inspired hierarchical zero-shot task reasoning framework** (hierarchical planning and verification · B): Explicit object-state reasoning plus composition of atomic actions, cost ranking, and closed-loop execution verification; clearing a board 10/10, pick-and-place 10/10, pyramid stacking 4/5. [Source](https://arxiv.org/abs/2609.05985v1)
- **WorldAgen** (test-time world model training · B): See transferable methods. [Source](https://arxiv.org/abs/2609.08162v1)
- **ProWAM** (progress-conditioned imagination · B): See transferable methods. [Source](https://arxiv.org/abs/2609.06578v1)
- **UniMPA** (unified memory-prediction-action · B): Uses an action-grounded transition interface to jointly handle transition ambiguity, prediction-execution mismatch, and experience-implementation mismatch. [Source](https://arxiv.org/abs/2609.11875v1)
- **A survey of 3D Vision-Language Models** (3D VLM survey · B): A tutorial-style survey from 3D representation encoding through cross-modal contrastive alignment to 3D VLLMs, including language-guided 3DGS. [Source](https://arxiv.org/abs/2609.05583v1)

### 5.3 VLA and manipulation
{: id="53-具身-vla--移动操作"}

- **VLA-Corrector** (closed-loop recovery · B): See transferable methods. [Source](https://arxiv.org/abs/2609.06508v1)
- **Where Success Breaks** (failure-boundary learning · B): See transferable methods. [Source](https://arxiv.org/abs/2609.06114v1)
- **FailureSpot** (timestep-level failure detection · B): Efficient failure detection at the timestep level, aimed at long-horizon execution. [Source](https://arxiv.org/abs/2609.04277v1)
- **ActSafeGuard** (hard-constraint guarantees · B): Provides differentiable, training-aligned constraint enforcement for flow-matching policies. [Source](https://arxiv.org/abs/2609.11697v1)
- **Reasoning Without Inference Cost** (training-time reasoning · B): Uses latent semantic scaffolding to gain reasoning benefits during training and discards it before deployment, eliminating inference overhead. [Source](https://arxiv.org/abs/2609.04893v1)
- **What Matters, When?** (conditional visual grounding · C): Diagnoses how the visual target changes with the manipulation stage and task state, treating it as conditional visual grounding. [Source](https://arxiv.org/abs/2609.05376v1)
- **RefGuard** (referring-identity grounding · C): Jointly grounds the target, anchor, and reference frame, handling spatial terms where similar objects repeat and the reference frame matters. [Source](https://arxiv.org/abs/2609.06221v1)
- **ICI-VLA** (in-context imitation · C): Spatiotemporally aligned demonstrations enable few-shot test-time adaptation without gradient updates. [Source](https://arxiv.org/abs/2609.07581v1)
- **GIFT** (goal-image injection fine-tuning · C): Injects a generated goal image into fine-tuning as high-level visual guidance. [Source](https://arxiv.org/abs/2609.07006v1)
- **LayerRoute** (layer-wise hybrid routing · C): Routes visual-semantic representations from different VLM layers according to action conditions. [Source](https://arxiv.org/abs/2609.06079v1)
- **IMLE-VLA** (single-step action generation · C): Uses IMLE in place of multi-step iterative sampling, removing the stop-and-go motion of flow-matching action heads. [Source](https://arxiv.org/abs/2609.10915v1)
- **FreqFM** (frequency-conditioned flow matching · C): Explicitly models the frequency heterogeneity of action trajectories. [Source](https://arxiv.org/abs/2609.10405v1)
- **Time-frequency geometric cross-attention** (chunked action decoding · C): Treats action chunks as short multivariate trajectories and decodes them by frequency and geometric structure rather than with linear heads per timestep. [Source](https://arxiv.org/abs/2609.09925v1)
- **Large Discrete Policy** (discrete behavior modeling · C): Selects actions from a large vocabulary of physically plausible candidates as an alternative to continuous denoising. [Source](https://arxiv.org/abs/2609.07049v1)
- **HuRo** (human video pre-training · C): A systematic examination of whether robotized human videos can serve as scalable pre-training supervision for VLAs. [Source](https://arxiv.org/abs/2609.10706v1)
- **RoboDrop** (post-training data filtering · C): Filters post-training data by local gradient compatibility. [Source](https://arxiv.org/abs/2609.10021v1)
- **ZETA** (zero-shot cross-embodiment transfer · C): Provides a unified definition of zero-shot transfer and a controlled evaluation setting that isolates the change of embodiment. [Source](https://arxiv.org/abs/2609.02546v2)
- **VLA-Precision** (real-robot online reinforcement learning · C): Asymmetric co-bootstrapping alleviates policy drift caused by unreliable value signals and the throughput bottleneck of large models. [Source](https://arxiv.org/abs/2609.04355v2)
- **HINT** (long-horizon intent inference · C): Infers human intent from a general instruction and keeps adapting as visual observations evolve. [Source](https://arxiv.org/abs/2609.02653v2)
- **CR-VLA-Force** (force-aware compliance control · C): A control-aware compliant VLA for contact-rich manipulation. [Source](https://arxiv.org/abs/2609.05832v1)
- **FWBC-VLA** (force-aware whole-body compensation · C): Bridges semantic action generation and physical interaction control for contact-rich mobile manipulation. [Source](https://arxiv.org/abs/2609.03889v2)
- **DeCAL** (contact-aware dexterous manipulation · C): Contact-aware joint latent imagination, handling severe visual occlusion and complex contact dynamics. [Source](https://arxiv.org/abs/2609.09119v1)
- **GloVLA** (global geometry plus local VLA · C): Separates long-range end-effector transport from the contact-rich interaction after arrival. [Source](https://arxiv.org/abs/2609.06256v1)
- **FolDeX** (deformable-object benchmark · C): A real-robot benchmark for long-horizon bimanual manipulation of deformable objects. [Source](https://arxiv.org/abs/2609.10243v1)
- **Language Transfer Metrics for Robot Policies** (multilingual VLA · C): Adds Greek to the Cosmos3 VLA by rewriting instructions with machine translation alone, focusing on the reliability of the measurement tools. [Source](https://arxiv.org/abs/2609.07470v1)

### 5.4 UAVs and other
{: id="54-无人机与其他"}

- **AGC-VLN** (air-ground collaborative VLN · B): See transferable methods. [Source](https://arxiv.org/abs/2609.03483v3)
- **AirAnchor** (UAV zero-shot VLN · B): See transferable methods. [Source](https://arxiv.org/abs/2609.08442v1)
- **ComVLA** (6G split inference · C): Communication-aware split inference for VLAs, adapting to wireless link bandwidth constraints. [Source](https://arxiv.org/abs/2609.07838v1)
- **Modal decoupled federated learning** (6G privacy protection · C): Modality-decoupled federated learning for 6G embodied intelligence. [Source](https://arxiv.org/abs/2609.09591v1)
- **Few-shot VLM soft prompts** (out-of-domain object detection · C): Uses soft prompts for few-shot adaptation from ten images in out-of-domain scenes such as aerial, industrial, and medical imagery. [Source](https://arxiv.org/abs/2609.11310v1)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

There are no news items in this issue. The WeChat source (we-mp-rss) added 0 entries, and collection for both subscribed accounts returned 0 items because of rate limiting on the WeChat Official Accounts Platform.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **The assumptions behind navigation evaluation are being dismantled systematically.** OVMAN challenges "the scene stays the same between two visits", NavArena challenges "only annotated datasets such as Habitat/MP3D can support closed-loop evaluation", and LIBERO-RECOVER and MEMOBench challenge "final success rate is enough to summarize ability". The four directions differ, but the common result is that a single success-rate number explains less and less, so the evaluation protocol must be reported alongside it.
- **"Memory validity" is turning from an implicit assumption into an explicitly modeled object.** OmniNav's updatable 3D object memory and negative evidence from failed searches, AirAnchor's persistent object spatial memory, UniMPA's visual-action memory bank, and MEMOBench's process-level memory evaluation: the four works address the same thing from the perspectives of navigation, aerial navigation, manipulation, and evaluation. This digest's assessment: this is the most concentrated methodological convergence in this issue.
- **Inference cost is treated as a first-class constraint rather than a post hoc optimization.** O2C-Nav compresses the number of large-model calls, Reasoning Without Inference Cost moves the reasoning benefit into training, IMLE-VLA eliminates multi-step sampling, and ComVLA handles wireless link bandwidth limits. Four works attack the same problem at different levels. For real-robot navigation deployment, the practical value of such work may exceed a small gain in baseline score.
- **The definition of "traversability" in navigation is starting to be tied to the embodiment configuration.** TANGO's whole-body adaptation and OmniNav's inclusion of manipulation reachability in endpoint selection both indicate that abstracting the robot as a disk is no longer enough in mobile manipulation scenarios.

### Research gaps
{: id="研究空白"}

- **Zero-shot methods on R2R-CE lack joint latency-accuracy reporting.** O2C-Nav sells on the number of calls but its summary gives no accuracy values. Most works report only SR/SPL and not per-step time, so "how much faster but how much worse" cannot be judged. This is a low-cost evaluation contribution.
- **Navigation after scene changes currently has only benchmarks and no methods.** OVMAN's error decomposition attributes the main bottleneck behind the gap between the reference agent's 45.2% and the oracle's 99.5% to open-vocabulary instance grounding. No method in ground-based VLN specifically handles "the target is no longer where it was".
- **Sim-to-real navigation transfer lacks quantitative comparison.** TANGO's real-robot conclusions are qualitative, and OmniNav's real-robot comparison is against an open-loop baseline. No work in this issue presents paired values for the same method on a simulation benchmark and on a real robot.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read closely and reproduce**: O2C-Nav: confirm the R2R-CE/RxR-CE values and comparison settings in the full text and measure per-step end-to-end latency; the code is released.
- **Read closely**: OmniNav: take apart the specific form of Bayesian belief revision and the negative-evidence mechanism, and assess whether it can be attached on its own to an existing frontier exploration module.

**Medium priority**

- **Join benchmark tracking**: OVMAN and NavArena: wait for the data and tools to be released. The former is for evaluating map staleness in your own system, the latter for closed-loop evaluation at your own sites.
- **Track**: TANGO: once the full text is out, confirm the benchmark used for VLN evaluation (the abstract does not say whether it belongs to the R2R-CE series, so assume it is not comparable); take apart the simulation data synthesis pipeline.
- **Read as a selection reference**: No Free Checker: for choosing verifier types for success detection and runtime monitoring in navigation systems.

**Low priority**

- **Paused**: VLA works that report only LIBERO with no evidence of failure recovery or real-robot results (most of the 10 in this issue); LIBERO-RECOVER has already pointed out that this benchmark is close to saturation.
