---
layout: post
lang: en
translation_id: vln-weekly-2026-09-19
permalink: /en/vln-weekly-2026-09-19/
source_path: _posts/weekly-reports/2026-09-19-VLN-Weekly.md
source_url: /vln-weekly-2026-09-19/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-09-09 to 2026-09-17)"
date: 2026-09-19
period_start: 2026-09-09
period_end: 2026-09-17
issue_number: 5
tags: [VLN, VLA, Embodied Navigation, Embodied Agent, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "EgoPathBench tests first-person waypoint decisions in nine VLMs: Point Path achieves 35.9% success with point-agent geometry, but Embodied Path falls to 2.9% under embodiment constraints. GPT-6-Astra instead reaches 52.0% R2R-CE SR through a complete workflow. This digest attributes current zero-shot navigation performance mainly to system design rather than foundation-model spatial ability. C2Nav supports this interpretation: replacing comparative questions with cardinal questions reduces SR to 12.0%, 28.0%, and 21.0% at three decision points. Embodied agents receive systematic coverage for the first time, including Harness Robotic OS's closed-loop quadruped inspection results."
---
## 1. Key conclusions
{: id="一本期结论"}

- **VLM Waypoint decision-making under embodied geometry constraints is nearly unusable, while system-level workflows pull the score up to 52%.** This is the most noteworthy set of comparisons in this issue. EgoPathBench measures the first-person waypoint decision-making of nine VLMs one by one, and the highest EgoPath Score is only 28.3; under point proxy geometry, Point Path still has a 35.9% success rate. Once embodied geometry constraints are added, the Embodied Path drops to **2.9%**, and the Intent Path 4.0%. GPT-6-Astra ran 52.0% SR on R2R-CE, relying on the complete workflow of observation-decision-execution plus context management. This digest's assessment: The current zero-shot navigation performance is mainly contributed by the **system structure** rather than the spatial capabilities of the foundation model; using VLM directly as a waypoint planner is betting on its weakest ability.
- **R2R-CE is still the main battlefield, but the evaluation protocol is so fragmented that direct comparison is impossible.** 5 articles reported that R2R-CE used four sets of evaluation sets: the complete val-unseen (GroundingVLN), OpenNav's 100 episode protocol (C2Nav), the first 50 items of the protocol (GPT-6-Astra), and a custom 550-item subset (LG-VLN). It would be seriously misleading to interpret GroundingVLN's 69.9% SR side by side with C2Nav's 31.0% SR - the former is the result of the trained method on the full set, and the latter is the result of the zero-shot method on the hundred-episode subset. Episode collections must be aligned before comparison.
- **Asking VLMs to compare instead of reporting numbers is becoming a reusable design principle.** C2Nav's role-reversal ablation is this issue's strongest methodological experiment: changing only the question format from comparative to cardinal/absolute reduces SR from 31.0% to 12.0%, 28.0%, and 21.0% at the spatial, transition, and termination decision points, respectively. AnchorVLN's MCP tool schema enforces "semantics from the VLM, metric quantities from geometry"; no tool accepts parameters in meters or radians. It provides supporting evidence: direct coordinate estimation passes the threshold on 0 of 45 questions, compared with 10 after geometric anchoring. GroundingVLN implements the same principle through pixel goals and a geometric planner in a trained system.
- **Embodied-agent runtimes are beginning to offer complete deployable systems.** Harness Robotic OS organizes four planes—robot runtime, autonomous skills, cognitive-agent runtime, and interaction/operations—and reports closed-loop results from real-robot inspection in a residential community: 100% waypoint reachability, outdoor localization error <10 cm, local obstacle-avoidance response <200 ms, 85–95% hazard detection, and 99% alarm delivery and structured reporting. Together with Finder (agentic closed-loop object search, +15.75 points at the 1 m success threshold on Habitat/HM3D) and AeroWeaver (a drone-swarm agent harness), these studies begin to outline how agent harnesses can support embodied systems.
- **spatial memory has divided into three routes, and there is new work in all three routes in this issue.** ① Retain the attention KV of the geometry foundation model and clip it by relevance (AdaGeoVLN); ② Give up coordinates completely and use visual location topology or persistent spatio-temporal graphs (Navi-Agent, HarnessVLN); ③ Train a latent world model on global spatio-temporal memory to predict future maps and waypoints (GLAM), or compress the memory into segment-level plans for execution (Memory as Plans). All three routes claim to be SOTA under their respective settings, and there is no direct comparison between them.
- **Coverage has been corrected, so this issue's counts cannot be compared directly with the previous issue.** On 2026-09-19, two fixes to the collection configuration added previously missed navigation terms (robot / social / terrain / legged / egocentric navigation, waypoint) and a new embodied Agent / AgentOS source. In the same time window, the revised collection contains 162 entries, including **41 in the main pool (navigation + embodied agents)**; before the fixes, it collected only 70 entries, of which just 14 were relevant. This digest analyzes the main pool; see 5.4 for secondary pools such as VLA/manipulation. The WeChat source contributed 0 entries because Docker was not running and the account remained subject to long-term rate limiting, a situation continuing since 2026-09-05.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [GroundingVLN](https://arxiv.org/abs/2609.18581v1)** · R2R-CE/RxR-CE (continuous environment)
  - Contribution: Using visual grounding as a shared interface for reasoning and action; predicting progress-aligned pixel targets to a geometric planner.
  - Evidence: R2R-CE 69.9% SR, RxR-CE 75.1% SR; only 0.9% of the strongest baseline training data is used; RxR-CE reaches 59.9% SR when trained with R2R only, which is +20.1% higher than the strongest baseline.
  - Reason: This issue is the only trained method that reports SOTA in complete R2R-CE and has an order of magnitude difference in sample efficiency.
- **A2 · [EgoPathBench](https://arxiv.org/abs/2609.16610v1)** · First-person waypoint decision-making evaluation
  - Contribution: Five-task benchmark, based on three criteria: candidate feasibility, adjacent edge legitimacy, and goal attainment, distinguishing point agency from embodied geometry.
  - Evidence: nine VLMs with the highest EgoPath Score 28.3; Point Path 35.9% vs **Embodied Path 2.9%**, Intent Path 4.0%; 31,852 training + 1,111 benchmark questions; fine-tuning Qwen 3.5 4B raised its score from 3.9 to 38.9.
  - Reason: The gap scale between "foundation model space capability" and "navigation system performance" is given, which is a must-read for model selection.
- **A3 · [C2Nav](https://arxiv.org/abs/2609.15142v1)** ·  zero-shot  VLN-CE
  - Contribution: Training independent; VLM only does ordinal comparisons between candidates constructed by the controller, geometry/thresholds/motion amplitudes are left on the robot side.
  - Evidence: OpenNav R2R-CE 100 Protocol: Qwen3-VL-8B-Instruct 41.0% OSR / 31.0% SR / 16.7% SPL; change to GPT-5.5 to reach 54.0% / 44.0% / 29.0%; see conclusion three for role reversal ablation.
  - Reason: Causal evidence of interface design is more valuable than performance numbers and can be directly migrated to your own pipeline.
- **A4 · [Harness Robotic OS](https://arxiv.org/abs/2609.11225v1)** · Embodied Agent runtime (quadruped inspection)
  - Contributions: four planes of robot runtime/autonomous skills/cognitive agent runtime/interactive operation and maintenance; hierarchical work, scenarios, and semantic memory; self-evolving loops for safety gate control.
  - Evidence: Real robot in residential communities: Waypoint reachability 100%, outdoor positioning error <10 cm, local obstacle avoidance response <200 ms, hazard detection 85–95%, alarm and reporting 99%.
  - Reason: This issue is the only embodied Agent runtime that provides complete real robot closed-loop data, a reference implementation in the AgentOS direction.
- **A5 · [GLAM](https://arxiv.org/abs/2609.14561v1)** · ObjectNav（HM3D / Habitat）
  - Contribution: Training target condition latent world model on global spatiotemporal memory, jointly predicting future map representation and robot center waypoint latent variable; JEPA-style latent prediction without RGB reconstruction.
  - Evidence: Under the controlled HM3D-ObjectNav subset reproduction setting, both SR and SPL are better than the reproduced BSC-Nav baseline; **summary does not give specific values**.
  - Reason: A representative work of the third route of spatial memory, forming a tripartite contrast with AdaGeoVLN / Navi-Agent.
- **A6 · [Finder](https://arxiv.org/abs/2609.18058v1)** · Open vocabulary embodied object retrieval (Habitat/HM3D)
  - Contribution: agentic closed-loop search primitive: typed recurrent state string query condition planning, limited range forensics, candidate verification and accept/continue/abort control.
  - Evidence: On Habitat/HM3D and real RGB-D scenes, the average 1 m success rate is strong baseline **+15.75 points**; the same primitive can be transferred to sequence object grounding and embodied object center question answering.
  - Reason: Transform ObjectNav from "static retrieval" to "closed-loop forensics", the mechanism is clear and portable.
- **A7 · [HarnessVLN](https://arxiv.org/abs/2609.15195v2)** · **Discrete** R2R/RxR/ObjectNav (HM3D)
  - Contribution: Training-free Agent Harness unified command following and goal navigation; verification evidence support, geometric feasibility and sub-goal consistency before distribution.
  - Evidence: R2R 60.8%, RxR 53.9%, HM3D-v2 76.0%, HM3D-OVON 59.3% SR, said to be better than previous training-free SOTA; including real robot deployment of humanoid robots.
  - Reason: Note that this is the **discrete** R2R, which cannot be compared side by side with the above R2R-CE numbers.
- **A8 · [AdaGeoVLN](https://arxiv.org/abs/2609.18789v1)** · R2R-CE/RxR-CE, monocular RGB streaming
  - Contribution: GFM hierarchical features by representation depth coupling strategy stages; VGGT global attention KV by instruction relevance/geometric confidence/transfer novelty preserved within bounded budget.
  - Evidence: The abstract only calls "strong performance", and **does not provide a verifiable number**; ablation The conclusion is that multi-depth coupling is significantly better than repeated injection of terminal features.
  - Reason: The mechanism level constitutes a counterexample to the existing GFM usage; the figures need to be verified by the official version.
- **A9 · [Navi-Agent](https://arxiv.org/abs/2609.20388v1)** · zero-shot VLN-CE, no positioning monocular
  - Contribution: Coordinate-free navigation topology (node = visual location, edge = motion transfer), supports approximate self-positioning, progress verification and revisit recovery.
  - Evidence: State-of-the-art in geometrically constrained methods on zero-shot VLN-CE benchmark and real robot, **The abstract does not give specific values** 。
  - Reason: To form a "coordinate or not" route comparison with AdaGeoVLN.
- **A10 · [UNI](https://arxiv.org/abs/2609.20114v1)** · Wheeled robot navigation data collection
  - Contribution: Four-wheeled walker + mobile phone collection of physically constrained human demonstrations, naturally biased toward wheeled-accessible routes.
  - Evidence: 37.2 km real data; 17.4–24.8% trajectory prediction error reduction on set-out UNI demonstration; closed-loop migration to power wheelchairs (curbs/stairs/ramps).
  - Reason: Low cost, bypassing platform-specific teleoperation data path, directly available to ground platform.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. EgoPathBench and GPT-6-Astra: foundation-model spatial ability and system performance
{: id="1-egopathbench-与-gpt-6-astra-对读基础模型的空间能力和导航系统的成绩是两件事"}

It is easy to read these two articles separately, but the whole picture can be seen only when they are put together, so they are analyzed together.

**Problem.** Two sides of the same problem. EgoPathBench asks: Can VLM itself select a sequence of waypoints that are feasible and reach the target from first-person observations. GPT-6-Astra asked: If you put a general large model into a complete navigation workflow without making any navigation fine-tuning, what level can the overall system achieve?

**Method.** EgoPathBench provides five task benchmarks. Each question contains first-person RGB, natural language target and numbered visible waypoint. The model returns passable candidates or ordered routes; evaluates the feasibility of sub-candidates, the legitimacy of adjacent edges, and target arrival, and distinguishes point proxy geometry from embodied geometry. GPT-6-Astra uses the standard observation-decision-execution workflow to directly call the API. Each request brings selected observations, execution feedback and retained progress records. The evaluation object is a complete system including context management and action control.

**Evidence.** EgoPathBench: The highest EgoPath Score among the nine VLMs is only 28.3; the top-ranked model has a Point Path success rate of 35.9%, but an Embodied Path of only 2.9% and an Intent Path of only 4.0%. The dataset contains 31,852 training questions, 1,345 verification questions, and 1,111 benchmark questions. Each route question retains at least one geometrically verified reference route. Using the released training set to fine-tune Qwen 3.5 4B, its EgoPath Score increased from 3.9 to 38.9, with a general increase of 1.4–9.6 points on the three external space benchmarks. GPT-6-Astra: On the first 50 of the 100 R2R-CE val-unseen used by Open-Nav, 52.0% SR, 48.9% SPL, 70.8% nDTW; success is split into 36.0% from accepted active STOP decisions and 16.0% from satisfying the distance criterion at the step limit.

**Value.** The two sets of numbers differ by an order of magnitude, pointing to the same conclusion: **The current performance of zero-shot navigation is mainly contributed by the system structure** - candidate construction, geometry verification, context management, execution feedback - rather than the spatial intelligence of the underlying model itself. This digest's assessment has direct implications for technology selection: the return on investing budget in system structure is currently significantly higher than replacing it with a stronger VLM. EgoPathBench releases the training set. This is particularly useful. The fine-tuning result of 3.9→38.9 shows that this ability is trainable, but no one has trained it.

**Limitations.** The evaluation protocols of the two are different and cannot be subtracted: EgoPathBench measures single-step/short-range waypoint decision-making questions, while GPT-6-Astra measures the task success rate of a complete episode; the former has nine models, and the latter has 50 episodes in a single model (1 episode accounts for 2 percentage points). GPT-6-Astra relies on a closed-source API, with cost and reproducibility limited by the vendor.

**Recommendation.** EgoPathBench high-priority intensive reading and reproduction of fine-tuning experiments - it is the only work in this issue that provides evidence that "this ability is trainable". GPT-6-Astra can be used as a baseline tracking, but its reporting method of termination composition is worth copying.

### 2. GroundingVLN: explicit grounding interfaces improve sample efficiency by an order of magnitude
{: id="2-groundingvln把-grounding-当接口而非中间产物样本效率出现量级差"}

**Problem.** The paper states that there are two coupling gaps: intermediate reasoning is not explicitly anchored to visual evidence; and high-level decision-making lacks precise spatial targets to guide low-level movements.

**Method.** uses grounding in two stages: first "reasoning with grounding", anchoring task-related visual evidence to precise image locations throughout the structured reasoning process; and then "grounding action" to predict pixel targets aligned with the task progress, which are translated into primitive actions by the geometric planner. A time-series aligned grounded inference trajectory dataset GroundingCOTVLN-188K is constructed, and GEAR (Grounded and Execution-Aware RL) is used to align grounded inference and spatial decision-making to downstream execution.

**Evidence.** R2R-CE 69.9% SR, RxR-CE 75.1% SR, called SOTA; the amount of training data is 0.9% of the strongest baseline; in terms of cross-dataset generalization, when only R2R is used for training, RxR-CE reaches 59.9% SR, which is 20.1% higher than the strongest baseline. The abstract does not give SPL, nDTW or baseline method names.

**Value.** The division of labor of pixel target + geometric planner is the implementation of the same idea on the training route as the "semantics from the model, metric quantities from geometry" of C2Nav and AnchorVLN. If the 0.9% data amount is true, it means that the information density of grounded supervision is much higher than that of pure trajectory supervision - this is mutually confirmed by the fine-tuning conclusion of EgoPathBench: the efficiency of waypoint/grounding type supervision has been underestimated for a long time.

**Limitations.** The abstract does not explain the complete settings of the baseline identity and evaluation split, and the SOTA statement cannot be independently verified; the construction cost of the 188K dataset has not been disclosed; no real robot results have been seen.

**Recommendation.** High priority intensive reading. Focus on checking two things: whether the val-unseen corresponding to 69.9% is a complete set; how to define GEAR's execution alignment reward - this is a candidate answer for "inference-execution disconnect".

### 3. C2Nav: comparative questions improve performance over absolute numerical answers
{: id="3-c2nav把-vlm-的回答形式从报数改成比较本身就是性能来源"}

**Problem.** Existing zero-shot VLN-CE systems generally ask for cardinal output from the VLM—waypoints, pixels, orientations, progress values, or absolute arrival decisions, which directly couples generative answers to geometric magnitudes or irreversible commitments.

**Method.** is a training-free framework with three synergistic capabilities: Seeing performs an ordinal Gaze Election between physically verified candidate perspectives; Remembering maintains a compact route sketch and compares adjacent command segment hypotheses; Arriving combines hesitation ladders, lookback comparisons, and reversible walking back to achieve reliable stopping. Geometry, thresholds, motion ranges and execution all stay on the robot side.

**Evidence.** OpenNav R2R-CE 100 protocol: Qwen3-VL-8B-Instruct gets 41.0% OSR / 31.0% SR / 16.7% SPL; replacing the same interface with standard GPT-5.5 gets 54.0% / 44.0% / 29.0%. Ability ablation: After removing Seeing, SR drops to 14.0%, removing Remembering drops to 25.0%, and removing Arriving drops to 29.0%. The matched role reversal experiment - only changing the comparative response form into a cardinal/absolute question - reduced the SR to 12.0%, 28.0%, and 21.0% in the three decision-making positions of space, transfer, and termination respectively. The paper concludes that constrained decision-making interfaces and stronger VLM inference are complementary rather than substitutes for each other.

**Value.** Role reversal separates the two confounding variables of "interface form" and "model capability", which is the only one in this issue. Combined with EgoPathBench's 2.9%, this conclusion has more weight: Since VLM's absolute judgment under embodied geometric constraints is so poor, limiting it to ordinal comparison is not just an engineering trick, but an inevitable choice to maximize strengths and avoid weaknesses.

**Limitations.** 100 episodes has a small sample size, and the weight of a single episode is 1%. The percentage difference between ablation needs to be interpreted with caution; the absolute SR (31.0%/44.0%) is still far from practical; there is no real robot.

**Recommendation.** Read and reproduce the role reversal experiment. On your own system, change Arriving first (to stop the judgment), which will cost the least.

### 4. Harness Robotic OS: complete real-robot results from an embodied-agent runtime
{: id="4-harness-robotic-os具身-agent-运行时第一次给出完整真机闭环数据"}

**Problem.** The deployable autonomous inspection system requires not only robust navigation, but also the connection of heterogeneous sensors, reusable autonomous capabilities, multi-modal scene understanding, human–robot interaction and enterprise-side response in a traceable operational closed loop. The existing quadruped inspection system uses multi-purpose task-specific interfaces to put these functions together, making context coordination, knowledge reuse and controlled adaptation difficult.

**Method.** HROS divides the system into four planes: robot runtime, embodied autonomous skills, cognitive agent runtime, interaction and operation and maintenance. Shared context connects the physical state to agent reasoning; streaming ASR/TTS supports voice task interaction; hierarchical working memory, episodic memory, and semantic memory preserve running knowledge; and the safety-gated self-evolving loop converts execution trajectories into versioned candidate updates, which does not allow unconstrained online modifications. The Argos prototype integrates Vbot quadrupeds, Fast-LIO2 positioning mapping, Hobot-Stereo depth perception, PCT-Planner global planning, EGO-Planner local motion generation, and Qwen3-VL inspection analysis orchestrated by OpenClaw.

**Evidence.** Residential property environment experiment: waypoint accessibility 100%, outdoor positioning error less than 10 cm, local obstacle avoidance response delay less than 200 ms, representative hazard detection rate 85–95%, alarm delivery and structured report generation success rate 99%.

**Value.** This is the only work in this issue that puts "agent architecture" into measurable real robot indicators. It is also a direct sample of the AgentOS direction that users are concerned about. The two designs of three-layer memory (working/situational/semantic) and safety gated self-evolution have direct reference value for the long-term operation of the ground navigation system - especially the "execution trajectory is converted into a candidate update with version, and unconstrained online modification is not allowed", which controls the risk of continuous learning.

**Limitations.** The scenario is a structured residential community inspection. The task is to predefine inspection routes rather than follow open instructions, which is very different from the task distribution of VLN. It reports system integration indicators (reachability, delay, detection rate). There is no control experiment with other agent architectures, so it is impossible to judge the independent contribution of each design. Indicators such as navigation success rate that can be compared horizontally with VLN are not reported.

**Recommendation.** Read the architecture part intensively, focusing on the division basis of three-layer memory and the safety-gate design of self-evolving loop. There is no need to replicate the entire system - its value lies in the way it is organized, not in the specific numbers.

### 5. GLAM and Memory as Plans: a third route to spatial memory
{: id="5-glam-与-memory-as-plans空间记忆的第三条路线"}

**Problem.** Active exploration and semantic navigation require the agent to build memory from local observations, predict how the evolution of the observed spatial memory will support future movements, and convert the predictions into executable plans. Existing approaches either retain the original form of geometric features (high memory cost), or degenerate into topological maps with no predictive ability.

**Method.** GLAM is a latent goal-conditioned world model, trained on global spatiotemporal memory: given a historical map token, navigation target and current pose, it jointly predicts the future map representation and the waypoint latent variable of the robot center, so that the future spatial context and navigation intention fall in the same representation space. It adopts a JEPA-style latent prediction paradigm that operates directly on map-level latent tokens instead of RGB reconstruction, and uses a pre-trained waypoint codec to supervise and decode navigation plans in GLAM NAV. Memory as Plans (MaP-WAM) takes another path: representing memory as a completed segment record containing language instructions and sparse visual context, converting it into a compact plan (next-level language plan + corresponding visual guidance), executed by the World-Action-Progress model, jointly predicting action chunks and execution progress.

**Evidence.** GLAM: The training data is obtained by replaying ObjectNav expert trajectories on Habitat/HM3D v0.2 and cutting into multi-time scale prediction samples; under the controlled reproduction on an HM3D-ObjectNav subset, both SR and SPL are better than the reproduced BSC-Nav baseline. The **abstract does not give specific values**. MaP-WAM: RMBench 83.3% success rate (called SOTA), real robot task 78.0%, and the executor context length is fixed, and the inference delay is approximately constant as the task history grows.

**Value.** What the two have in common is that **does not feed historical observations directly to the executor**: GLAM compresses latent map tokens and predicts their evolution, and MaP-WAM compresses them into segment-level plans. The implication of this for long-range navigation is that latency does not grow with episodes - MaP-WAM explicitly reports this. GLAM's "map-level latent token instead of RGB reconstruction" and AdaGeoVLN's "preserving GFM attention KV" form an interesting opposition: one believes that abstraction should be at the map layer, and the other believes that fidelity should be maintained at the feature layer.

**Limitations.** GLAM provides no public numerical results, and the baseline is the authors' reproduction of BSC-Nav. It is impossible to judge whether it is SOTA or not; the evaluation is limited to the controlled subset of HM3D-ObjectNav. The evidence of MaP-WAM all comes from manipulation tasks (RMBench + real-robot manipulation). The navigation has not been verified, and it is doubtful whether its "segment-level plan" is suitable for fine-grained decision-making of continuous navigation.

**Recommendation.** GLAM tracks the official version and values. MaP-WAM's fixed context length + KV cache design can be borrowed from the navigation executor alone, and this step does not depend on its task settings.

## 4. Transferable methods
{: id="四可迁移方法"}

Only secondary-pool studies with clear transfer paths are included.

- **VLA Inference efficiency: [VLA-ULAP](https://arxiv.org/abs/2609.18663v1)**
  - Mechanism: Cloud large model call alternates with local ultra-lightweight action predictor; ULAP is about 7.4M parameters (including frozen visual encoder), independent training, no VLA hidden state, online verification or server round-trip required.
  - Integration point: Navigation decision-making layer: The remote VLM is only called at key decision points, and the straight/fine-adjust heading is taken over by the local predictor.
  - Prerequisites and risks: The numbers are all from manipulation tasks, and whether the call skip rate of 48.8–76.7% can be maintained under long-range dependence on navigation is unproven.
- **VLA Architecture: [DEM (decoupled) ](https://arxiv.org/abs/2609.18374v2)**
  - Mechanism: Separate visual/language encoder conditions a compact action head, replacing the VLM backbone running billions of parameters per step; the MeanFlow head single forward action chunk.
  - Integration point: The backbone of language-conditioned navigation strategies, especially for vehicle-mounted platforms with limited computing power.
  - Prerequisites and risks: The conclusion is limited to the scope of "trained tasks"; the task distribution of RoboCasa 18 tasks and navigation is very different.
- **VLA Inference efficiency: [rMuscle](https://arxiv.org/abs/2609.19104v1)**
  - Mechanism: Two-stage caching across execution similarities: Context Cache reuses visual token output reduction calculations, and Action Cache reuses neuron activation patterns to reduce weight access.
  - Integration point: inference acceleration in structured navigation scenarios such as fixed patrol routes and repeated round trips.
  - Prerequisites and risks: The premise is that tasks are highly repetitive; the observation similarity of open environment navigation is much lower than that of factory workstations.
- **VLA Architecture: [What Makes an Efficient VLA](https://arxiv.org/abs/2609.13984v1)**
  - Mechanism: Action header performance is mainly determined by initialization rather than decoder architecture; copying the last few transformer layers of the language backbone into the action header is the greatest leverage and has zero latency cost.
  - Integration point: Action header initialization for any self-built VLA-style navigation policy.
  - Prerequisites and risks: The paper states that this is the explanation that best organizes its measurement results rather than proven cause and effect.
- **evaluation methodology: [Libero-CTRL](https://arxiv.org/abs/2609.15940v1)**
  - Mechanism: Paired composite robustness evaluation: Pair single-axis and simultaneous disturbance conditions for the same initial state to distinguish "emergency failure" and "compensation success".
  - Integration point: Navigation robustness evaluation: multi-axis simultaneous disturbances such as lighting × layout × dynamic obstacles.
  - Prerequisites and risks: The core finding is that the two types of transfers cancel each other out in aggregate statistics - up to 29.0% (34.5% in the most severe condition) of paired initial state outcomes that are not reflected in the overall success rate and must be recorded on an instance-by-pair basis.
- **Security: [Beyond Patch Removal](https://arxiv.org/abs/2609.19669v1)**
  - Mechanism: State recovery protocol: adversarial patches are removed at matching action chunk boundaries, and recoverability is measured under the same remaining step budget.
  - Access locations: Adversarial robustness evaluation of navigation strategies and recovery adapter design.
  - Prerequisites and risks: The evidence comes from LIBERO-Long: After OpenVLA-OFT was attacked by EDPA, only 36.2% episodes could be recovered after five chunks (control group 89.9% / 87.0%); the recovery adapter increased the recovery rate from 7.7% to 47.4% under one chunk delay, and the benefits dropped significantly as the delay increased.
- **review: [World-Action Models review ](https://arxiv.org/abs/2609.16074v1)**
  - Mechanism: A unified taxonomy that couples future world prediction and executable action generation, covering representation, transfer modeling, action interfaces, training processes and expansion strategies, and single-column navigation applications.
  - Integration point: The entry document for WAM route entry navigation.
  - Prerequisites and risks: Review nature, no own experiments; project page https://rcl-robotics.github.io/Awesome-World-Action-Models。

## 5. Research roundup by category
{: id="五分类速览"}

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **GroundingVLN** (R2R-CE/RxR-CE): See key analysis. [Source](https://arxiv.org/abs/2609.18581v1)
- **C2Nav** (zero-shot VLN-CE): See key analysis. [Source](https://arxiv.org/abs/2609.15142v1)
- **AdaGeoVLN** (R2R-CE/RxR-CE): GFM hierarchical features are retained at each stage of the characterization depth coupling strategy + bounded KV. [Source](https://arxiv.org/abs/2609.18789v1)
- **GPT-6-Astra workflow** (zero-shot VLN-CE): See key analysis. [Source](https://arxiv.org/abs/2609.20116v1)
- **GLAM** (ObjectNav (HM3D)): See key analysis. [Source](https://arxiv.org/abs/2609.14561v1)
- **HarnessVLN** (Discrete R2R/RxR + ObjectNav): Agent Harness verifies evidence, geometric feasibility and sub-goal consistency before distribution, unifying two types of navigation tasks. [Source](https://arxiv.org/abs/2609.15195v2)
- **Navi-Agent** (zero-shot VLN-CE): Coordinate-free visual location topology replaces geometric positioning, supporting progress verification and revisit recovery. [Source](https://arxiv.org/abs/2609.20388v1)
- **LG-VLN** (zero-shot VLN-CE, monocular): Sharing CleanDIFT dense features to connect geometry and semantics, LangGraph is organized as a directed state diagram; 21.3% SR, 12.1% SPL on the customized 550 episode R2R-CE val-unseen subset. [Source](https://arxiv.org/abs/2609.15098v1)
- **AnchorVLN** (Open Vocabulary VLN): MCP tool schema forces "VLM to provide semantics and geometric determination"; CMU VLN Challenge 2026 instruction followinging 64.4%, removing the controller modeling dropped 13.3 points; 10 of the 45 questions on object reference exceeded the threshold for geometric anchoring, and 0 were directly estimated coordinates question. [Source](https://arxiv.org/abs/2609.12285v1)
- **TADreamer** (land and air dual-modal zero-shot navigation): Use generated video to imagine navigation, and then restore 3D waypoints and motion patterns with consistent metrics through two-stage calibration. [Source](https://arxiv.org/abs/2609.19824v1)
- **CueNav** (universal robot navigation): BEV map transmits the global task context, retains part of the body to expose the specific context, and guides video planning; IDM converts dense light flow into action. [Source](https://arxiv.org/abs/2609.16737v2)
- **RoboFind** (Personalized Object Search (Barrier-free)): Mobile phone teaches target and quadruped execution search; teaching agent produces semantic target portrait and multi-view reference library for reuse in subsequent tasks. [Source](https://arxiv.org/abs/2609.20330v1)
- **LEAP** (Quadruped Navigation + Active Perception): No change in task goals, no coverage/curiosity agent rewards, just the task pressure of the terrain course to make gaze control emerge spontaneously. [Source](https://arxiv.org/abs/2609.17628v1)

### 5.2 Embodied agents / AgentOS (new coverage)
{: id="52-具身-agent--agentos本期新覆盖方向"}

This direction has not been included in the crawling before. It is included in the system for the first time in this issue, so it is listed in a separate section. In addition to Harness Robotic OS, Finder, GLAM, and Memory as Plans analyzed above:

- **[AeroWeaver](https://arxiv.org/abs/2609.18520v1)** (UAV cluster agent harness)
  - Contributions: Connecting semantic decisions to governed skills; role-conditioned local agents for distributed coordination; online refinement of skill selection using role-indexed state-action-reward experience.
  - Evidence: It maintains effective skill execution under test conditions and supports local multi-machine operation without a central agent; reward-guided online updates provide a training-free adaptation path. **does not give a quantitative indicator**.
  - Relationship with ground navigation: The organizational method of "governed skills + online experience refinement" is transferable; however, the coordination needs of cluster and stand-alone ground navigation are quite different.
- **[DeliveryGym](https://arxiv.org/abs/2609.19801v1)** (long-range embodied agent planning environment)
  - Contribution: An RL environment for long-range embodied planning.
  - Evidence: Baseline scores were not given in the abstract.
  - Relationship to ground navigation: The training environment for long-range task decomposition is related to the long-range nature of navigation.
- **[Embodied-BenchForge](https://arxiv.org/abs/2609.13082v1)** (embodied benchmark automatic construction)
  - Contribution: Closed-loop benchmark synthesis: forward artifact synthesis + reverse verification repair; skill orchestration artifact synthesis cooperates with the artifact dependency graph to prevent local defects from propagating downstream.
  - Evidence: No quantitative results are given in the abstract.
  - Relationship with ground navigation: If you want to build your own navigation evaluation set, its idea of "verification by workpiece rather than end-to-end delivery" is worth learning from.
- **[EmbodiedMind](https://arxiv.org/abs/2609.19659v1)** (data screening and RL)
  - Contribution: Adaptive data filtering + prefix tree reinforcement learning.
  - Evidence: The summary does not provide verifiable figures.
  - Relationship with ground navigation: training side method, not directly bound to navigation tasks.
- **[STAGE](https://arxiv.org/abs/2609.13458v1)** (embodied execution semantic migration diagnosis)
  - Contribution: Diagnosing the migration of semantics at grounding execution.
  - Evidence: Benchmark is LIBERO (Operation).
  - Relationship with ground navigation: diagnostic methods can be referenced, and evidence comes from manipulation tasks.
- **[SIMLIFE](https://arxiv.org/abs/2609.19610v1)** (long-range human-machine collaboration)
  - Contribution: Pattern understanding in long-range human-agent collaboration.
  - Evidence: The summary does not provide verifiable figures.
  - Relationship with ground navigation: Human-machine collaboration scenarios have a weak relationship with social navigation.
- **[ReactHuman](https://arxiv.org/abs/2609.10895v1)** (physical grounding benchmark)
  - Contribution: A physics grounding benchmark for human-like reactive decision-making.
  - Evidence: The summary does not provide verifiable figures.
  - Relationship to ground navigation: Evaluation of reactive decision-making, weakly related to dynamic obstacle avoidance.
- **[Autonomy, Social Norms, and Alignment](https://arxiv.org/abs/2609.11660v1)** (theoretical framework)
  - Contribution: A developmental framework of autonomy, social norms and alignment.
  - Evidence: Theoretical work, no experiments.
  - Relationship with ground navigation: partial theory, only field observation.
- **[Exploring 2D backbone effects](https://arxiv.org/abs/2609.17257v1)** (Indoor semantic occupancy prediction)
  - Contribution: Examining the impact of 2D backbone networks on indoor semantic occupancy prediction.
  - Evidence: The summary does not provide verifiable figures.
  - Relationship to terrestrial navigation: Upstream perception module of semantic maps.

### 5.3 Memory, Maps, Planning and Evaluation
{: id="53-记忆地图规划与评测"}

- **EgoPathBench** (waypoint decision evaluation): See key analysis. [Source](https://arxiv.org/abs/2609.16610v1)
- **ENCP** (VLN uncertainty): Episode normalized conformal prediction, recovering progressive coverage guarantees for variable length dependent navigation sequences. [Source](https://arxiv.org/abs/2609.17499v1)
- **UDAV** (off-road waypoint planning uncertainty): Multiple random samplings of medoids are taken as self-consistent nominal routes, and spatial dispersion is used to estimate the uncertainty. Reconsideration is triggered only when the threshold is exceeded; the average ADE on 400 reserved queries dropped from 147.4 to 110.4 pixels (-25.1%). [Source](https://arxiv.org/abs/2609.16368v1)
- **UNI** (navigation data collection): robot-free collection of walkers + mobile phones, physical constraints naturally screen out wheeled feasible routes. [Source](https://arxiv.org/abs/2609.20114v1)
- **Feeling Terrain Before Crossing** (off-road navigation world model): "Feel" the terrain before crossing it, and use the world model to predict passability. [Source](https://arxiv.org/abs/2609.19863v1)
- **TRACER** (multi-robot social navigation): bidirectional rolling time domain, separating single-robot effects and non-additive pair interactions, maintaining persistent belief in response patterns by identity; improving collision-free completion rate on SocialGym2. [Source](https://arxiv.org/abs/2609.18776v1)
- **PRISM** (Social Navigation): Predictive representations of interaction style and movement. [Source](https://arxiv.org/abs/2609.18125v1)
- **Mobile Multi-Robot Navigation**: Handling runtime uncertainty with the Koopman operator. [Source](https://arxiv.org/abs/2609.14058v1)
- **Learning Safe Humanoid Navigation**: Learning safe navigation from reduced-order models. [Source](https://arxiv.org/abs/2609.19272v1)
- **Tuning ROS 2 for Energy-Efficient Navigation** (system tuning): Demonstration of energy-efficiency tuning of ROS 2 navigation stack. [Source](https://arxiv.org/abs/2609.12971v1)
- **Task-Oriented Active Learning of Residual Dynamics** (MPC): Task-oriented active learning of residual dynamics. [Source](https://arxiv.org/abs/2609.19378v1)
- **Steering with Lexicographic Preferences** (deployment period preference): Lexicographic preferences guide the generative strategy, and the strategy weight remains unchanged; the paper claims that it is verified to be effective on a navigation benchmark. [Source](https://arxiv.org/abs/2609.15014v1)
- **OHRID-Retail** (dataset): An open multimodal dataset of human activities in retail environments. [Source](https://arxiv.org/abs/2609.19302v1)
- **SwarmNxt** (cluster platform): open source software and hardware cluster platform. [Source](https://arxiv.org/abs/2609.11382v1)
- **Custom PX4 firmware** (Land-Air Amphibious): PX4 firmware customization for autonomous hybrid land-air missions. [Source](https://arxiv.org/abs/2609.20691v1)
- **UAVs Meet Embodied Intelligence** (UAV embodied intelligence): bridging human intention and flight dynamics with physical-digital AI agents. [Source](https://arxiv.org/abs/2609.18326v1)

### 5.4 Secondary-pool roundup: VLA/manipulation, autonomous driving, and other work
{: id="54-次池速览vla操作--自动驾驶--其他"}

There are a total of 162 windows in this issue, 41 of which are covered by the main pool above; the remaining 121 are classified into the secondary pool according to the "field" field and will not be analyzed in depth. The distribution is: VLA/manipulation 65 items, other 50 items, and autonomous driving 6 items.

Those in the VLA manipulation pool that have potential transfer value related to navigation have been listed in Section 4 "Migration Methods". The rest are purely manipulation work such as robotic arms, force control, dexterous hands, teleoperation, underwater arms, and laboratory automation, as well as training-side methods such as action segmentation, RL fine-tuning, and federated training. There is no navigation-side verification in this issue and will not be discussed one by one. For a complete list, see the "Field" column in `index.md`.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **system structure, rather than model capabilities, is the main source of current zero-shot navigation results. There is an order of magnitude difference between** EgoPathBench's 2.9% (waypoint decision under embodied geometry) and GPT-6-Astra's 52.0% (episode success rate under complete workflow); the role reversal of C2Nav further shows that the same model will lose 19 percentage points when switching back to cardinal questions, while the same interface will only increase by 13 percentage points when switching to a stronger model. All three together support the same judgment.
- **spatial memory shifts from "what to store" to "what to predict".** Among the three routes in this issue, GLAM directly predicts future map representations and waypoint hidden variables, and Memory as Plans converts memory into segment-level plans. Both of them no longer feed historical observations to the executor; AdaGeoVLN's bounded KV retention and Navi-Agent's coordinate-free topology are still "what to save" optimizations. A measurable benefit of predictive memory is that latency does not grow with episodes, which MaP-WAM explicitly reports.
- **agent harness is spilling over from the general LLM realm to embodied systems.** Three independent harness jobs appear in this issue: HarnessVLN (navigation), Harness Robotic OS (quadruped inspection runtime), and AeroWeaver (drone cluster). The common mode is to perform a layer of verification (evidence support, geometric feasibility, sub-goal consistency, governed skills) before distributing actions, and maintain cross-step shared status and hierarchical memory. Harness Robotic OS's safety gated self-evolving loop is the only one that handles "continuously updated risk control."
- The **evaluation methodology itself has become an independent research object.** EgoPathBench points out that existing spatial intelligence benchmarks only measure isolated judgments and cannot measure integrated navigation capabilities; ENCP points out that the coverage guarantee of standard conformal prediction on variable-length dependency episodes is invalid; GPT-6-Astra splits the success rate into active stopping and bottoming determination; Embodied-BenchForge directly agentizes the benchmark construction itself and adds artifact-by-artifact verification.

### Research gaps
{: id="研究空白"}

- The zero-shot review of **R2R-CE lacks recognized protocols.** Four subsets coexist in this issue, and there is no report on the conversion or overlapping relationship with other subsets, making "What level has zero-shot VLN-CE currently reached" impossible to answer at the literature level.
- **suspension judgment is still the weakest single link and lacks special indicators.** GPT-6-Astra's 16.0% bottoming success, and C2Nav's SR drop from 31.0% to 29.0% after removing Arriving all point to this. However, except for these two articles, no one separately reported stop related indicators in this issue.
- The architectural design of **embodied Agent lacks controlled experiments.** The three harness jobs in this issue only report the overall system indicators, and none of them has done ablation of "removing a certain layer of verification/a certain type of memory", so it is impossible to judge the independent contribution of each design - this is in sharp contrast to C2Nav's approach on the navigation side.
- **There is a verification gap between efficiency work and navigation tasks. The inference efficiency work in the** secondary pool was all verified on the LIBERO class operation benchmark, and none was tested on VLN-CE.

### Suggested actions
{: id="建议动作"}

**High priority**

- **intensively read and reproduced**: EgoPathBench's fine-tuning experiment (Qwen 3.5 4B increased from 3.9 to 38.9) - the only work in this issue that proves "embodied waypoint decision-making is trainable".
- **Read closely**: GroundingVLN: Verify the split integrity and baseline identity corresponding to 69.9% SR; dismantle the execution alignment reward design of GEAR.
- **intensively read and partially reproduced**: C2Nav's role reversal experiment; priority is given to using comparative questions in the stop judgment process of the own system.
- **creates a tracking table**: Creates a three-column registration table for R2R-CE results "evaluation subset + number of episodes + whether zero-shot" to prevent direct comparison of cross-protocol numbers.

**Medium priority**

- **Read closely architecture**: Harness Robotic OS's three-layer memory division and security gate self-evolution loop; there is no need to reproduce the entire system.
- **Minimal verification**: Multi-depth GFM coupling of AdaGeoVLN: Change single-point terminal feature injection to early/middle/late three-point coupling, independent of its code.
- **indicator modification**: Fixed splitting of "active STOP success" and "step limit bottoming success" in its own evaluation, refer to GPT-6-Astra.
- **Track**: the official version values and code releases of GLAM and Finder; they represent the two routes of predictive memory and closed-loop forensics respectively.
- **Track**: UNI’s walker acquisition paradigm to evaluate whether it is suitable for data supplementation of its own wheeled platform.

**Low priority**

- **Suspended**: All VLA inference efficiency work in the secondary pool - the mechanism can be used for reference, but its effectiveness on VLN-CE has not been verified in any way.
