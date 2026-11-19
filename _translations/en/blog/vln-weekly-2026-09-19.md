---
layout: post
lang: en
translation_id: vln-weekly-2026-09-19
permalink: /en/vln-weekly-2026-09-19/
source_path: _posts/weekly-reports/2026-09-19-VLN-Weekly.md
source_url: /vln-weekly-2026-09-19/
source_revision_date: 2026-10-24
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly · Issue 5"
date: 2026-09-19
last_modified_at: 2026-10-24
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

- **VLM waypoint decisions under embodied geometry constraints are nearly unusable, yet system-level workflows lift the score to 52%.** This is the most noteworthy comparison in this issue. EgoPathBench tests the first-person waypoint decisions of nine VLMs one by one, and the best EgoPath Score is only 28.3; under point-proxy geometry, Point Path still reaches a 35.9% success rate. Once embodied geometry constraints are added, Embodied Path drops to **2.9%** and Intent Path to 4.0%. GPT-6-Astra reaches 52.0% SR on R2R-CE using a complete observe-decide-execute workflow with context management. This digest's assessment: current zero-shot navigation performance comes mainly from the **system structure**, not from the spatial ability of the foundation model; using a VLM directly as a waypoint planner bets on its weakest skill.
- **R2R-CE is still the main battleground, but evaluation protocols are so fragmented that direct comparison is impossible.** 5 works report on R2R-CE using four different evaluation sets: the full val-unseen (GroundingVLN), the Open-Nav 100-episode protocol (C2Nav), the first 50 episodes of that protocol (GPT-6-Astra), and a custom 550-episode subset (LG-VLN). Reading GroundingVLN's 69.9% SR side by side with C2Nav's 31.0% SR would be seriously misleading: the former is a trained method on the full set, the latter a zero-shot method on a hundred-episode subset. Episode sets must be aligned before any comparison.
- **Asking a VLM to compare rather than to report numbers is becoming a reusable design principle.** C2Nav's role-reversal ablation is the strongest methodological experiment in this issue: changing only the question format from comparative to cardinal/absolute cuts SR from 31.0% to 12.0%, 28.0%, and 21.0% at the spatial, transition, and termination decision points respectively. AnchorVLN's MCP tool schema enforces "semantics from the VLM, metric quantities from geometry": no tool accepts parameters in meters or radians. It supplies supporting evidence: direct coordinate estimation passes the threshold on 0 of 45 questions, versus 10 after geometric anchoring. GroundingVLN realizes the same principle in a trained system through pixel goals and a geometric planner.
- **Embodied-agent runtimes are starting to offer complete deployable systems.** Harness Robotic OS organizes four planes (robot runtime, autonomous skills, cognitive agent runtime, and interaction/operations) and reports closed-loop real-robot inspection results in a residential community: 100% waypoint reachability, outdoor localization error <10 cm, local obstacle-avoidance response <200 ms, 85–95% hazard detection, and 99% alarm delivery and structured reporting. Together with Finder (agentic closed-loop object search, +15.75 points at the 1 m success threshold on Habitat/HM3D) and AeroWeaver (a UAV-swarm agent harness), these works begin to outline how agent harnesses can support embodied systems.
- **Spatial memory has split into three routes, and this issue has new work on all three.** ① Retain the attention KV of a geometry foundation model and clip it by relevance (AdaGeoVLN); ② abandon coordinates entirely and use visual-place topology or persistent spatiotemporal graphs (Navi-Agent, HarnessVLN); ③ train a latent world model on global spatiotemporal memory to predict future maps and waypoints (GLAM), or compress memory into segment-level plans for execution (Memory as Plans). All three claim SOTA under their own settings, and there is no direct comparison among them.
- **Coverage has been corrected, so this issue's counts cannot be compared directly with the previous issue.** On 2026-09-19, two fixes to the collection configuration added previously missed navigation terms (robot / social / terrain / legged / egocentric navigation, waypoint) and a new embodied Agent / AgentOS source. Over the same time window, the revised collection yields 162 entries, including **41 in the main pool (navigation + embodied agents)**; before the fixes it collected only 70 entries, of which just 14 were relevant. This digest analyzes the main pool; see 5.4 for secondary pools such as VLA/manipulation. The WeChat source contributed 0 entries because Docker was not running and the account remained under long-term rate limiting, a situation that has continued since 2026-09-05.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [GroundingVLN](https://arxiv.org/abs/2609.18581v1)** · R2R-CE/RxR-CE (continuous environments)
  - Contribution: Uses visual grounding as a shared interface for reasoning and action; predicts progress-aligned pixel goals for a geometric planner.
  - Evidence: 69.9% SR on R2R-CE and 75.1% SR on RxR-CE using only 0.9% of the strongest baseline's training data; trained on R2R only, it reaches 59.9% SR on RxR-CE, +20.1% over the strongest baseline.
  - Why it matters: The only trained method in this issue reporting SOTA on the full R2R-CE, with an order-of-magnitude advantage in sample efficiency.
- **A2 · [EgoPathBench](https://arxiv.org/abs/2609.16610v1)** · Evaluation of first-person waypoint decisions
  - Contribution: A five-task benchmark based on three criteria (candidate feasibility, adjacent-edge legitimacy, and goal attainment) that distinguishes point-proxy geometry from embodied geometry.
  - Evidence: Across nine VLMs the highest EgoPath Score is 28.3; Point Path 35.9% vs. **Embodied Path 2.9%** and Intent Path 4.0%; 31,852 training + 1,111 benchmark questions; fine-tuning Qwen 3.5 4B raises its score from 3.9 to 38.9.
  - Why it matters: Gives the size of the gap between "spatial ability of foundation models" and "navigation system performance", which is essential reading for model selection.
- **A3 · [C2Nav](https://arxiv.org/abs/2609.15142v1)** · Zero-shot VLN-CE
  - Contribution: Training-free; the VLM only makes ordinal comparisons among candidates constructed by the controller, while geometry, thresholds, and motion magnitudes stay on the robot side.
  - Evidence: On the Open-Nav R2R-CE 100-episode protocol, Qwen3-VL-8B-Instruct reaches 41.0% OSR / 31.0% SR / 16.7% SPL; switching to GPT-5.5 gives 54.0% / 44.0% / 29.0%; see key conclusion three for the role-reversal ablation.
  - Why it matters: Causal evidence about interface design is worth more than performance numbers and can be migrated directly into your own pipeline.
- **A4 · [Harness Robotic OS](https://arxiv.org/abs/2609.11225v1)** · Embodied-agent runtime (quadruped inspection)
  - Contribution: Four planes (robot runtime / autonomous skills / cognitive agent runtime / interaction and operations); hierarchical working, episodic, and semantic memory; a safety-gated self-evolving loop.
  - Evidence: Real robot in a residential community: waypoint reachability 100%, outdoor localization error <10 cm, local obstacle-avoidance response <200 ms, hazard detection 85–95%, alarm and reporting 99%.
  - Why it matters: The only embodied-agent runtime in this issue with complete real-robot closed-loop data, and a reference implementation for the AgentOS direction.
- **A5 · [GLAM](https://arxiv.org/abs/2609.14561v1)** · ObjectNav（HM3D / Habitat）
  - Contribution: Trains a goal-conditioned latent world model on global spatiotemporal memory that jointly predicts future map representations and robot-centric waypoint latents; JEPA-style latent prediction without RGB reconstruction.
  - Evidence: In a controlled reproduction on an HM3D-ObjectNav subset, both SR and SPL beat the reproduced BSC-Nav baseline; **the summary gives no specific values**.
  - Why it matters: A representative of the third spatial-memory route, forming a three-way contrast with AdaGeoVLN / Navi-Agent.
- **A6 · [Finder](https://arxiv.org/abs/2609.18058v1)** · Open-vocabulary embodied object retrieval (Habitat/HM3D)
  - Contribution: An agentic closed-loop search primitive: typed recurrent state, query-conditioned planning, bounded-range evidence gathering, candidate verification, and accept/continue/abort control.
  - Evidence: On Habitat/HM3D and real RGB-D scenes, the average 1 m success rate is **+15.75 points** over strong baselines; the same primitive transfers to sequential object grounding and embodied object-centric question answering.
  - Why it matters: Turns ObjectNav from "static retrieval" into "closed-loop evidence gathering", with a clear and portable mechanism.
- **A7 · [HarnessVLN](https://arxiv.org/abs/2609.15195v2)** · **Discrete** R2R/RxR/ObjectNav (HM3D)
  - Contribution: A training-free agent harness that unifies instruction following and goal navigation; evidence support, geometric feasibility, and sub-goal consistency are verified before actions are dispatched.
  - Evidence: SR of 60.8% on R2R, 53.9% on RxR, 76.0% on HM3D-v2, and 59.3% on HM3D-OVON, said to beat prior training-free SOTA; includes real-robot deployment on a humanoid.
  - Why it matters: Note that this is **discrete** R2R and cannot be compared side by side with the R2R-CE numbers above.
- **A8 · [AdaGeoVLN](https://arxiv.org/abs/2609.18789v1)** · R2R-CE/RxR-CE, monocular RGB streams
  - Contribution: Couples hierarchical GFM features to the policy stages by representation depth; VGGT global-attention KV is retained within a bounded budget according to instruction relevance, geometric confidence, and transition novelty.
  - Evidence: The abstract only says "strong performance" and **gives no verifiable numbers**; the ablation concludes that multi-depth coupling is significantly better than repeatedly injecting terminal features.
  - Why it matters: At the mechanism level it is a counterexample to how GFMs are usually used; the figures need to be verified in the official version.
- **A9 · [Navi-Agent](https://arxiv.org/abs/2609.20388v1)** · Zero-shot VLN-CE, monocular without localization
  - Contribution: A coordinate-free navigation topology (nodes are visual places, edges are motion transitions) supporting approximate self-localization, progress verification, and revisit recovery.
  - Evidence: State of the art among geometry-constrained methods on zero-shot VLN-CE benchmarks and on a real robot; **the abstract gives no specific values**.
  - Why it matters: Forms a "coordinates or no coordinates" route comparison with AdaGeoVLN.
- **A10 · [UNI](https://arxiv.org/abs/2609.20114v1)** · Data collection for wheeled-robot navigation
  - Contribution: A four-wheeled walker plus a phone collects physically constrained human demonstrations, which are naturally biased toward wheel-accessible routes.
  - Evidence: 37.2 km of real data; trajectory prediction error drops by 17.4–24.8% on held-out UNI demonstrations; closed-loop transfer to a power wheelchair (curbs / stairs / ramps).
  - Why it matters: Low cost, bypassing platform-specific teleoperation data, and directly usable for ground platforms.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. EgoPathBench / GPT-6-Astra
{: id="1-egopathbench-与-gpt-6-astra-对读基础模型的空间能力和导航系统的成绩是两件事"}

**Foundation-model spatial ability vs. system performance.**

The two works are easy to read separately, but the full picture appears only when they are put together, so they are analyzed together.

**Problem.** Two sides of one problem. EgoPathBench asks: can a VLM by itself pick a sequence of waypoints that is feasible and reaches the goal from first-person observations? GPT-6-Astra asks: if a general-purpose large model is put into a complete navigation workflow with no navigation fine-tuning at all, what level can the whole system reach?

**Method.** EgoPathBench provides a five-task benchmark. Each question contains first-person RGB, a natural-language goal, and numbered visible waypoints. The model returns traversable candidates or an ordered route; evaluation covers candidate feasibility, adjacent-edge legitimacy, and goal arrival, and distinguishes point-proxy geometry from embodied geometry. GPT-6-Astra uses a standard observe-decide-execute workflow that calls the API directly. Each request carries selected observations, execution feedback, and retained progress records. What is evaluated is a complete system including context management and action control.

**Evidence.** EgoPathBench: the best EgoPath Score among the nine VLMs is only 28.3; the top-ranked model has a Point Path success rate of 35.9%, but Embodied Path of only 2.9% and Intent Path of only 4.0%. The dataset contains 31,852 training questions, 1,345 validation questions, and 1,111 benchmark questions. Each route question retains at least one geometrically verified reference route. Fine-tuning Qwen 3.5 4B on the released training set raises its EgoPath Score from 3.9 to 38.9, with gains of 1.4–9.6 points on three external spatial benchmarks. GPT-6-Astra: on the first 50 of the 100 R2R-CE val-unseen episodes used by Open-Nav, 52.0% SR, 48.9% SPL, 70.8% nDTW; successes split into 36.0% from accepted active STOP decisions and 16.0% from meeting the distance criterion at the step limit.

**Value.** The two sets of numbers differ by an order of magnitude yet point to the same conclusion: **current zero-shot navigation performance comes mainly from system structure** (candidate construction, geometry verification, context management, execution feedback) rather than from the spatial intelligence of the underlying model. This digest's assessment has direct implications for technology choice: money spent on system structure currently returns significantly more than swapping in a stronger VLM. EgoPathBench releases its training set, which is especially useful: the 3.9→38.9 fine-tuning result shows that this ability is trainable, yet no one has trained it.

**Limitations.** The two evaluation protocols differ and cannot be subtracted from each other: EgoPathBench measures single-step / short-range waypoint decision questions, while GPT-6-Astra measures task success over complete episodes; the former covers nine models, the latter a single model on 50 episodes (1 episode is 2 percentage points). GPT-6-Astra relies on a closed-source API, so cost and reproducibility are limited by the vendor.

**Recommendation.** Read EgoPathBench closely at high priority and reproduce its fine-tuning experiment: it is the only work in this issue that provides evidence that "this ability is trainable". GPT-6-Astra can be tracked as a baseline, but its way of reporting the composition of terminations is worth copying.

### 2. GroundingVLN
{: id="2-groundingvln把-grounding-当接口而非中间产物样本效率出现量级差"}

**An explicit grounding interface improves sample efficiency by an order of magnitude.**

**Problem.** The paper identifies two coupling gaps: intermediate reasoning is not explicitly anchored to visual evidence, and high-level decisions lack precise spatial goals to guide low-level movement.

**Method.** Grounding is used in two stages: first "reasoning with grounding", which anchors task-relevant visual evidence to precise image locations throughout the structured reasoning process, and then "grounding action", which predicts pixel goals aligned with task progress and translates them into primitive actions through a geometric planner. It builds GroundingCOTVLN-188K, a temporally aligned dataset of grounded reasoning trajectories, and uses GEAR (Grounded and Execution-Aware RL) to align grounded reasoning and spatial decisions with downstream execution.

**Evidence.** 69.9% SR on R2R-CE and 75.1% SR on RxR-CE, claimed as SOTA; the training data amounts to 0.9% of the strongest baseline's; for cross-dataset generalization, training on R2R only gives 59.9% SR on RxR-CE, 20.1% above the strongest baseline. The abstract gives no SPL, nDTW, or baseline names.

**Value.** The division of labor between pixel goals and a geometric planner is the same idea as C2Nav's and AnchorVLN's "semantics from the model, metric quantities from geometry", implemented on the trained route. If the 0.9% data figure holds, grounded supervision carries far more information per sample than plain trajectory supervision, which is corroborated by EgoPathBench's fine-tuning result: waypoint/grounding-style supervision has long been underestimated in its efficiency.

**Limitations.** The abstract does not describe the identity of the baselines or the full evaluation split, so the SOTA claim cannot be verified independently; the construction cost of the 188K dataset is not disclosed; no real-robot results have been seen.

**Recommendation.** Read closely at high priority, checking two things in particular: whether the val-unseen behind 69.9% is the full set, and how GEAR's execution-alignment reward is defined, which is a candidate answer to the "reasoning-execution disconnect".

### 3. C2Nav
{: id="3-c2nav把-vlm-的回答形式从报数改成比较本身就是性能来源"}

**Comparative questions beat absolute numerical answers.**

**Problem.** Existing zero-shot VLN-CE systems generally ask the VLM for cardinal output (waypoints, pixels, headings, progress values, or absolute arrival decisions), which couples generated answers directly to geometric magnitudes or irreversible commitments.

**Method.** A training-free framework with three cooperating abilities: Seeing performs an ordinal Gaze Election among physically verified candidate views; Remembering maintains a compact route sketch and compares hypotheses about adjacent instruction segments; Arriving combines a hesitation ladder, lookback comparisons, and reversible backtracking to stop reliably. Geometry, thresholds, motion ranges, and execution all stay on the robot side.

**Evidence.** On the Open-Nav R2R-CE 100-episode protocol, Qwen3-VL-8B-Instruct gets 41.0% OSR / 31.0% SR / 16.7% SPL; swapping the same interface to the stronger GPT-5.5 gives 54.0% / 44.0% / 29.0%. Ability ablation: removing Seeing drops SR to 14.0%, removing Remembering to 25.0%, and removing Arriving to 29.0%. In the matched role-reversal experiment, merely changing the response form from comparative to cardinal/absolute questions reduces SR to 12.0%, 28.0%, and 21.0% at the spatial, transition, and termination decision points respectively. The paper concludes that constrained decision interfaces and stronger VLM reasoning are complementary rather than substitutes.

**Value.** Role reversal separates the two confounded variables, "interface form" and "model capability", and is the only experiment of its kind in this issue. Combined with EgoPathBench's 2.9%, the conclusion carries more weight: since a VLM's absolute judgments under embodied geometric constraints are so poor, restricting it to ordinal comparison is not just an engineering trick but the inevitable choice that plays to its strengths and avoids its weaknesses.

**Limitations.** 100 episodes is a small sample, with a single episode worth 1%, so percentage differences between ablations should be read with caution; the absolute SR (31.0%/44.0%) is still far from practical; no real robot.

**Recommendation.** Read it and reproduce the role-reversal experiment. On your own system, change Arriving (the stop decision) first, which costs the least.

### 4. Harness Robotic OS
{: id="4-harness-robotic-os具身-agent-运行时第一次给出完整真机闭环数据"}

**Complete real-robot results from an embodied-agent runtime.**

**Problem.** A deployable autonomous inspection system needs more than robust navigation: it must connect heterogeneous sensors, reusable autonomous skills, multimodal scene understanding, human-robot interaction, and enterprise-side response in one traceable operational closed loop. Existing quadruped inspection systems stitch these functions together with many task-specific interfaces, which makes context coordination, knowledge reuse, and controlled adaptation difficult.

**Method.** HROS divides the system into four planes: robot runtime, embodied autonomous skills, cognitive agent runtime, and interaction and operations. A shared context connects physical state to agent reasoning; streaming ASR/TTS supports voice task interaction; hierarchical working memory, episodic memory, and semantic memory preserve operating knowledge; and a safety-gated self-evolving loop converts execution trajectories into versioned candidate updates and forbids unconstrained online modification. The Argos prototype integrates a Vbot quadruped, Fast-LIO2 localization and mapping, Hobot-Stereo depth perception, PCT-Planner global planning, EGO-Planner local motion generation, and Qwen3-VL inspection analysis orchestrated by OpenClaw.

**Evidence.** Experiments in a residential property environment: waypoint reachability 100%, outdoor localization error under 10 cm, local obstacle-avoidance response latency under 200 ms, representative hazard detection rate 85–95%, and a 99% success rate for alarm delivery and structured report generation.

**Value.** This is the only work in this issue that turns "agent architecture" into measurable real-robot metrics, and it is a direct sample of the AgentOS direction that the user follows. Two designs, three-layer memory (working / episodic / semantic) and safety-gated self-evolution, have direct reference value for long-running ground navigation systems, especially "execution trajectories are converted into versioned candidate updates and unconstrained online modification is not allowed", which contains the risk of continual learning.

**Limitations.** The setting is inspection of a structured residential community, where the task is to follow predefined inspection routes rather than open-ended instructions, which differs greatly from the task distribution of VLN. It reports system integration metrics (reachability, latency, detection rate) with no controlled comparison against other agent architectures, so the independent contribution of each design cannot be judged. Metrics such as navigation success rate that could be compared horizontally with VLN are not reported.

**Recommendation.** Read the architecture section closely, focusing on the basis for dividing the three-layer memory and the safety-gate design of the self-evolving loop. There is no need to replicate the whole system: its value lies in how it is organized, not in the specific numbers.

### 5. GLAM / Memory as Plans
{: id="5-glam-与-memory-as-plans空间记忆的第三条路线"}

**A third route to spatial memory.**

**Problem.** Active exploration and semantic navigation require the agent to build memory from local observations, predict how the observed spatial memory will evolve to support future movement, and turn those predictions into executable plans. Existing approaches either keep the raw form of geometric features (high memory cost) or degrade to topological maps with no predictive ability.

**Method.** GLAM is a latent goal-conditioned world model trained on global spatiotemporal memory: given historical map tokens, a navigation goal, and the current pose, it jointly predicts the future map representation and the robot-centric waypoint latent, so that future spatial context and navigation intent live in the same representation space. It adopts a JEPA-style latent prediction paradigm that works directly on map-level latent tokens instead of reconstructing RGB, and uses a pretrained waypoint codec to supervise and decode navigation plans in GLAM NAV. Memory as Plans (MaP-WAM) takes another path: memory is represented as completed-segment records containing language instructions and sparse visual context, converted into a compact plan (the next-level language plan plus the corresponding visual guidance) and executed by a World-Action-Progress model that jointly predicts action chunks and execution progress.

**Evidence.** GLAM: training data come from replaying ObjectNav expert trajectories on Habitat/HM3D v0.2 and cutting them into multi-timescale prediction samples; in a controlled reproduction on an HM3D-ObjectNav subset, both SR and SPL beat the reproduced BSC-Nav baseline. **The abstract gives no specific values.** MaP-WAM: 83.3% success rate on RMBench (claimed SOTA), 78.0% on real-robot tasks, with a fixed executor context length and roughly constant inference latency as task history grows.

**Value.** What the two share is that they **do not feed historical observations directly to the executor**: GLAM compresses them into latent map tokens and predicts their evolution, while MaP-WAM compresses them into segment-level plans. For long-horizon navigation this means latency does not grow with the episode, which MaP-WAM reports explicitly. GLAM's "map-level latent tokens instead of RGB reconstruction" and AdaGeoVLN's "retain GFM attention KV" form an interesting opposition: one believes abstraction belongs at the map level, the other that fidelity should be kept at the feature level.

**Limitations.** GLAM publishes no numerical results, and its baseline is the authors' own reproduction of BSC-Nav, so whether it is SOTA cannot be judged; evaluation is limited to a controlled HM3D-ObjectNav subset. All of MaP-WAM's evidence comes from manipulation tasks (RMBench + real-robot manipulation); navigation is unverified, and it is doubtful whether its "segment-level plans" suit the fine-grained decisions of continuous navigation.

**Recommendation.** Track GLAM's official version and values. MaP-WAM's fixed context length plus KV cache design can be borrowed for a navigation executor on its own, and this step does not depend on its task setting.

## 4. Transferable methods
{: id="四可迁移方法"}

Only secondary-pool works with a clear transfer path are included.

- **VLA inference efficiency: [VLA-ULAP](https://arxiv.org/abs/2609.18663v1)**
  - Mechanism: Calls to a large cloud model alternate with a local ultra-lightweight action predictor; ULAP has about 7.4M parameters (including the frozen visual encoder), is trained independently, and needs no VLA hidden state, online verification, or server round trip.
  - Integration point: The navigation decision layer: call the remote VLM only at key decision points, and let the local predictor handle straight segments and fine heading adjustments.
  - Prerequisites and risks: All numbers come from manipulation tasks, and whether the call-skip rate of 48.8–76.7% can be sustained under the long-range dependencies of navigation is unproven.
- **VLA architecture: [DEM (decoupled)](https://arxiv.org/abs/2609.18374v2)**
  - Mechanism: Separate visual and language encoders condition a compact action head, replacing a VLM backbone that runs billions of parameters at every step; a MeanFlow head produces an action chunk in a single forward pass.
  - Integration point: The backbone of language-conditioned navigation policies, especially for onboard platforms with limited compute.
  - Prerequisites and risks: The conclusion is limited to "trained tasks"; the 18 RoboCasa tasks differ greatly from the task distribution of navigation.
- **VLA inference efficiency: [rMuscle](https://arxiv.org/abs/2609.19104v1)**
  - Mechanism: Two-stage caching that exploits similarity across executions: a Context Cache reuses visual-token outputs to cut computation, and an Action Cache reuses neuron activation patterns to reduce weight access.
  - Integration point: Inference acceleration in structured navigation scenarios such as fixed patrol routes and repeated round trips.
  - Prerequisites and risks: Assumes highly repetitive tasks; observation similarity in open-environment navigation is much lower than at factory workstations.
- **VLA architecture: [What Makes an Efficient VLA](https://arxiv.org/abs/2609.13984v1)**
  - Mechanism: Action-head performance is determined mainly by initialization rather than decoder architecture; copying the last few transformer layers of the language backbone into the action head gives the greatest leverage at zero latency cost.
  - Integration point: Initializing the action head of any self-built VLA-style navigation policy.
  - Prerequisites and risks: The paper states that this is the explanation that best organizes its measurements, not a proven causal relationship.
- **Evaluation methodology: [Libero-CTRL](https://arxiv.org/abs/2609.15940v1)**
  - Mechanism: Paired compound-robustness evaluation: for the same initial state, single-axis and simultaneous perturbation conditions are paired to distinguish "emergent failure" from "compensated success".
  - Integration point: Navigation robustness evaluation: simultaneous multi-axis perturbations such as lighting × layout × dynamic obstacles.
  - Prerequisites and risks: The core finding is that two kinds of transitions cancel out in aggregate statistics: up to 29.0% of paired initial-state outcomes (34.5% under the most severe condition) are invisible in the overall success rate and must be recorded pair by pair.
- **Safety: [Beyond Patch Removal](https://arxiv.org/abs/2609.19669v1)**
  - Mechanism: A state-recovery protocol: the adversarial patch is removed at a matching action-chunk boundary, and recoverability is measured under the same remaining step budget.
  - Integration point: Adversarial robustness evaluation of navigation policies and the design of recovery adapters.
  - Prerequisites and risks: The evidence is from LIBERO-Long: after OpenVLA-OFT is attacked by EDPA, only 36.2% of episodes can be recovered after five chunks (controls 89.9% / 87.0%); a recovery adapter raises the recovery rate from 7.7% to 47.4% at a one-chunk delay, with benefits dropping markedly as the delay grows.
- **Survey: [World-Action Models survey](https://arxiv.org/abs/2609.16074v1)**
  - Mechanism: A unified taxonomy coupling future world prediction with executable action generation, covering representation, transition modeling, action interfaces, training procedures, and scaling strategies, with a separate section on navigation applications.
  - Integration point: An entry document for approaching the WAM route in navigation.
  - Prerequisites and risks: A survey with no experiments of its own; project page https://rcl-robotics.github.io/Awesome-World-Action-Models.

## 5. Research roundup by category
{: id="五分类速览"}

### 5.1 VLN and ObjectNav
{: id="51-vln-与-objectnav"}

- **GroundingVLN** (R2R-CE/RxR-CE): See key analysis. [Source](https://arxiv.org/abs/2609.18581v1)
- **C2Nav** (zero-shot VLN-CE): See key analysis. [Source](https://arxiv.org/abs/2609.15142v1)
- **AdaGeoVLN** (R2R-CE/RxR-CE): Couples hierarchical GFM features to each stage by representation depth, plus bounded KV retention. [Source](https://arxiv.org/abs/2609.18789v1)
- **GPT-6-Astra workflow** (zero-shot VLN-CE): See key analysis. [Source](https://arxiv.org/abs/2609.20116v1)
- **GLAM** (ObjectNav (HM3D)): See key analysis. [Source](https://arxiv.org/abs/2609.14561v1)
- **HarnessVLN** (Discrete R2R/RxR + ObjectNav): An agent harness verifies evidence, geometric feasibility, and sub-goal consistency before dispatch, unifying two types of navigation tasks. [Source](https://arxiv.org/abs/2609.15195v2)
- **Navi-Agent** (zero-shot VLN-CE): A coordinate-free visual-place topology replaces geometric localization, supporting progress verification and revisit recovery. [Source](https://arxiv.org/abs/2609.20388v1)
- **LG-VLN** (zero-shot VLN-CE, monocular): Shared CleanDIFT dense features link geometry and semantics, and LangGraph organizes the system as a directed state graph; 21.3% SR and 12.1% SPL on the custom 550-episode R2R-CE val-unseen subset. [Source](https://arxiv.org/abs/2609.15098v1)
- **AnchorVLN** (Open Vocabulary VLN): An MCP tool schema enforces "semantics from the VLM, metric determination from geometry"; 64.4% instruction following on the CMU VLN Challenge 2026, dropping 13.3 points when controller modeling is removed; on object-reference questions, 10 of 45 pass the threshold with geometric anchoring versus 0 with directly estimated coordinates. [Source](https://arxiv.org/abs/2609.12285v1)
- **TADreamer** (zero-shot navigation for both ground and air · ): Uses generated video to imagine navigation, then recovers metrically consistent 3D waypoints and motion patterns through two-stage calibration. [Source](https://arxiv.org/abs/2609.19824v1)
- **CueNav** (general robot navigation): A BEV map conveys global task context while the robot body is partly retained to expose embodiment-specific context, guiding video planning; an IDM converts dense optical flow into actions. [Source](https://arxiv.org/abs/2609.16737v2)
- **RoboFind** (personalized object search (accessibility)): A phone teaches the target and a quadruped executes the search; the teaching agent produces a semantic goal profile and a multi-view reference library for reuse in later tasks. [Source](https://arxiv.org/abs/2609.20330v1)
- **LEAP** (quadruped navigation + active perception): No change to the task goal and no coverage/curiosity surrogate rewards; the pressure of the terrain course alone makes gaze control emerge spontaneously. [Source](https://arxiv.org/abs/2609.17628v1)

### 5.2 Embodied agents
{: id="52-具身-agent"}

This direction was not part of earlier crawls. It enters the system for the first time in this issue, so it is listed in its own section. Besides Harness Robotic OS, Finder, GLAM, and Memory as Plans analyzed above:

- **[AeroWeaver](https://arxiv.org/abs/2609.18520v1)** (UAV cluster agent harness)
  - Contribution: Connects semantic decisions to governed skills; role-conditioned local agents for distributed coordination; online refinement of skill selection using role-indexed state-action-reward experience.
  - Evidence: It maintains effective skill execution under test conditions and supports local multi-machine operation without a central agent; reward-guided online updates provide a training-free adaptation path. **No quantitative metrics are given.**
  - Relevance to ground navigation: The organizing approach of "governed skills + online experience refinement" is transferable, but the coordination needs of swarms and of a single ground robot differ greatly.
- **[DeliveryGym](https://arxiv.org/abs/2609.19801v1)** (long-horizon embodied agent planning environment)
  - Contribution: An RL environment for long-horizon embodied planning.
  - Evidence: The abstract gives no baseline scores.
  - Relevance to ground navigation: A training environment for long-horizon task decomposition, related to the long-horizon nature of navigation.
- **[Embodied-BenchForge](https://arxiv.org/abs/2609.13082v1)** (automatic construction of embodied benchmarks)
  - Contribution: Closed-loop benchmark synthesis: forward artifact synthesis plus reverse verification and repair; skill-orchestrated artifact synthesis works with an artifact dependency graph to keep local defects from propagating downstream.
  - Evidence: The abstract gives no quantitative results.
  - Relevance to ground navigation: If you want to build your own navigation evaluation set, its idea of "verifying per artifact rather than only at end-to-end delivery" is worth borrowing.
- **[EmbodiedMind](https://arxiv.org/abs/2609.19659v1)** (data filtering and RL)
  - Contribution: Adaptive data filtering plus prefix-tree reinforcement learning.
  - Evidence: The summary gives no verifiable figures.
  - Relevance to ground navigation: A training-side method not tied directly to navigation tasks.
- **[STAGE](https://arxiv.org/abs/2609.13458v1)** (diagnosing semantic transfer in embodied execution)
  - Contribution: Diagnoses how semantics transfer at the grounding-to-execution stage.
  - Evidence: The benchmark is LIBERO (manipulation).
  - Relevance to ground navigation: The diagnostic method can be borrowed, but the evidence comes from manipulation tasks.
- **[SIMLIFE](https://arxiv.org/abs/2609.19610v1)** (long-horizon human-agent collaboration)
  - Contribution: Pattern understanding in long-horizon human-agent collaboration.
  - Evidence: The summary gives no verifiable figures.
  - Relevance to ground navigation: Human-agent collaboration scenarios are only weakly related to social navigation.
- **[ReactHuman](https://arxiv.org/abs/2609.10895v1)** (physically grounded benchmark)
  - Contribution: A physically grounded benchmark for human-like reactive decision-making.
  - Evidence: The summary gives no verifiable figures.
  - Relevance to ground navigation: Evaluates reactive decision-making, weakly related to dynamic obstacle avoidance.
- **[Autonomy, Social Norms, and Alignment](https://arxiv.org/abs/2609.11660v1)** (theoretical framework)
  - Contribution: A developmental framework for autonomy, social norms, and alignment.
  - Evidence: Theoretical work, no experiments.
  - Relevance to ground navigation: Theory only; field observation.
- **[Exploring 2D backbone effects](https://arxiv.org/abs/2609.17257v1)** (Indoor semantic occupancy prediction)
  - Contribution: Examines how the 2D backbone network affects indoor semantic occupancy prediction.
  - Evidence: The summary gives no verifiable figures.
  - Relevance to ground navigation: An upstream perception module for semantic maps.

### 5.3 Memory, planning and evaluation
{: id="53-记忆规划与评测"}

- **EgoPathBench** (waypoint decision evaluation): See key analysis. [Source](https://arxiv.org/abs/2609.16610v1)
- **ENCP** (VLN uncertainty): Episode-normalized conformal prediction, restoring asymptotic coverage guarantees for variable-length, dependent navigation sequences. [Source](https://arxiv.org/abs/2609.17499v1)
- **UDAV** (uncertainty in off-road waypoint planning): The medoid of multiple random samples is taken as the self-consistent nominal route and spatial dispersion is used to estimate uncertainty; reconsideration is triggered only when a threshold is exceeded; average ADE on 400 held-out queries dropped from 147.4 to 110.4 pixels (-25.1%). [Source](https://arxiv.org/abs/2609.16368v1)
- **UNI** (navigation data collection): Robot-free collection with a walker plus a phone; physical constraints naturally filter for wheel-feasible routes. [Source](https://arxiv.org/abs/2609.20114v1)
- **Feeling Terrain Before Crossing** (off-road navigation world model): "Feels" the terrain before crossing it and uses a world model to predict traversability. [Source](https://arxiv.org/abs/2609.19863v1)
- **TRACER** (multi-robot social navigation): A bidirectional receding horizon separates single-robot effects from non-additive pairwise interactions and keeps persistent per-identity beliefs about response patterns; improves the collision-free completion rate on SocialGym2. [Source](https://arxiv.org/abs/2609.18776v1)
- **PRISM** (Social Navigation): Predictive representations of interaction style and motion. [Source](https://arxiv.org/abs/2609.18125v1)
- **Mobile Multi-Robot Navigation**: Handles runtime uncertainty with the Koopman operator. [Source](https://arxiv.org/abs/2609.14058v1)
- **Learning Safe Humanoid Navigation**: Learns safe navigation from reduced-order models. [Source](https://arxiv.org/abs/2609.19272v1)
- **Tuning ROS 2 for Energy-Efficient Navigation** (system tuning): Demonstrates energy-efficiency tuning of the ROS 2 navigation stack. [Source](https://arxiv.org/abs/2609.12971v1)
- **Task-Oriented Active Learning of Residual Dynamics** (MPC): Task-oriented active learning of residual dynamics. [Source](https://arxiv.org/abs/2609.19378v1)
- **Steering with Lexicographic Preferences** (deployment-time preferences): Lexicographic preferences steer a generative policy while its weights stay unchanged; the paper claims effectiveness on a navigation benchmark. [Source](https://arxiv.org/abs/2609.15014v1)
- **OHRID-Retail** (dataset): An open multimodal dataset of human activities in retail environments. [Source](https://arxiv.org/abs/2609.19302v1)
- **SwarmNxt** (cluster platform): An open-source hardware and software swarm platform. [Source](https://arxiv.org/abs/2609.11382v1)
- **Custom PX4 firmware** (land-air hybrid): Custom PX4 firmware for autonomous hybrid land-air missions. [Source](https://arxiv.org/abs/2609.20691v1)
- **UAVs Meet Embodied Intelligence** (UAV embodied intelligence): Bridges human intent and flight dynamics with physical-digital AI agents. [Source](https://arxiv.org/abs/2609.18326v1)

### 5.4 Secondary pool
{: id="54-次池速览"}

This issue has 162 entries in total, 41 of which are covered by the main pool above; the remaining 121 were placed in the secondary pool according to the "field" attribute and are not analyzed in depth. The distribution is: VLA/manipulation 65, other 50, and autonomous driving 6.

Those in the VLA/manipulation pool with potential transfer value for navigation are already listed in Section 4, "Transferable methods". The rest are purely manipulation works (robotic arms, force control, dexterous hands, teleoperation, underwater arms, laboratory automation) and training-side methods (action segmentation, RL fine-tuning, federated training), with no navigation-side validation in this issue, so they are not discussed one by one. For the full list, see the "Field" column of `index.md`.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **System structure, not model capability, is the main source of current zero-shot navigation results.** There is an order-of-magnitude gap between EgoPathBench's 2.9% (waypoint decisions under embodied geometry) and GPT-6-Astra's 52.0% (episode success under a complete workflow); C2Nav's role reversal further shows that the same model loses 19 percentage points when switched back to cardinal questions, while the same interface gains only 13 percentage points when switched to a stronger model. All three support the same judgment.
- **Spatial memory is shifting from "what to store" to "what to predict".** Of the three routes in this issue, GLAM directly predicts future map representations and waypoint latents, and Memory as Plans converts memory into segment-level plans; neither feeds historical observations to the executor. AdaGeoVLN's bounded KV retention and Navi-Agent's coordinate-free topology are still "what to store" optimizations. One measurable benefit of predictive memory is that latency does not grow with the episode, which MaP-WAM reports explicitly.
- **The agent harness is spilling over from general LLMs into embodied systems.** Three independent harness works appear in this issue: HarnessVLN (navigation), Harness Robotic OS (quadruped inspection runtime), and AeroWeaver (UAV swarm). The common pattern is a verification layer before actions are dispatched (evidence support, geometric feasibility, sub-goal consistency, governed skills), plus shared cross-step state and hierarchical memory. Harness Robotic OS's safety-gated self-evolving loop is the only one that handles the risk control of "continual updating".
- **Evaluation methodology has itself become a research object.** EgoPathBench points out that existing spatial-intelligence benchmarks only measure isolated judgments and cannot measure integrated navigation ability; ENCP points out that the coverage guarantee of standard conformal prediction fails on variable-length dependent episodes; GPT-6-Astra splits success rate into active stopping and fallback determination; and Embodied-BenchForge turns benchmark construction itself into an agentic process with per-artifact verification.

### Research gaps
{: id="研究空白"}

- **Zero-shot evaluation on R2R-CE lacks an agreed protocol.** Four subsets coexist in this issue, with no report on conversion or overlap among them, so "what level has zero-shot VLN-CE reached?" cannot be answered at the literature level.
- **Stop decisions remain the weakest single link and lack dedicated metrics.** GPT-6-Astra's 16.0% fallback success and C2Nav's SR dropping from 31.0% to 29.0% after removing Arriving both point to this. Apart from these two works, however, no one in this issue reports stop-related metrics separately.
- **The architectural design of embodied agents lacks controlled experiments.** The three harness works in this issue report only overall system metrics, and none ablates "removing a certain verification layer / a certain kind of memory", so the independent contribution of each design cannot be judged. This contrasts sharply with how C2Nav works on the navigation side.
- **There is a validation gap between efficiency work and navigation tasks.** All the inference-efficiency work in the secondary pool was validated on LIBERO-style manipulation benchmarks, and none was tested on VLN-CE.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read closely and reproduce**: EgoPathBench's fine-tuning experiment (Qwen 3.5 4B from 3.9 to 38.9), the only work in this issue showing that "embodied waypoint decision-making is trainable".
- **Read closely**: GroundingVLN: verify the split integrity and baseline identities behind 69.9% SR; take apart the execution-alignment reward design of GEAR.
- **Read closely and partially reproduce**: C2Nav's role-reversal experiment; first use comparative questions in the stop-decision step of your own system.
- **Create a tracking table**: a three-column register of R2R-CE results ("evaluation subset + number of episodes + zero-shot or not") to prevent direct comparison of numbers across protocols.

**Medium priority**

- **Read the architecture closely**: Harness Robotic OS's three-layer memory division and safety-gated self-evolving loop; there is no need to reproduce the whole system.
- **Minimal verification**: AdaGeoVLN's multi-depth GFM coupling: change single-point terminal-feature injection into three-point coupling (early / middle / late), independent of its code.
- **Metric change**: split "successful by active STOP" and "successful by step-limit fallback" in your own evaluation, following GPT-6-Astra.
- **Track**: the official-version values and code releases of GLAM and Finder; they represent the two routes of predictive memory and closed-loop evidence gathering respectively.
- **Track**: UNI's walker-based collection paradigm, to assess whether it suits supplementing data for your own wheeled platform.

**Low priority**

- **Paused**: all the VLA inference-efficiency work in the secondary pool: the mechanisms are worth borrowing, but none has been validated on VLN-CE.
