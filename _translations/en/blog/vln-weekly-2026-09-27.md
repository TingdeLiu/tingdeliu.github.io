---
layout: post
lang: en
translation_id: vln-weekly-2026-09-27
permalink: /en/vln-weekly-2026-09-27/
source_path: _posts/weekly-reports/2026-09-27-VLN-Weekly.md
source_url: /vln-weekly-2026-09-27/
source_revision_date: 2026-10-01
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-09-16 to 2026-09-24)"
date: 2026-09-27
period_start: 2026-09-16
period_end: 2026-09-24
issue_number: 6
tags: [VLN, VLA, Embodied Navigation, Embodied Agent, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "GPT-6-Astra reports a new 79.0% zero-shot R2R-CE SR claim with Codex harness and ultra reasoning, but the abstract omits the episode set, inference cost, and stopping criteria. A direct workflow reported 52.0% a week earlier; comparisons with trained methods require verification. Information conditions differ: autonomous, training-free SparseNav reports 42.8% on val-unseen, while Talk2Escape uses oracle or human correction to reach 66.0%. Harness trajectories also feed back into training: fine-tuning Qwen3.5-9B with World Action Agent trajectories raises out-of-distribution success from 1.7% to 43.3%."
---
## 1. Key conclusions
{: id="一本期结论"}

- A new claim of 79.0% appears for **R2R-CE zero-shot, but it is currently impossible to verify what it means.** GPT-6-Astra uses only monocular RGB under the smallest interface of the Codex harness, without navigation fine-tuning, trained waypoint predictors, or pre-built maps. The paper reports that the ultra reasoning level SR is 79.0%, which is 13.0 points higher than the strongest zero-shot and 6.9 points higher than the strongest supervised method. A week earlier, the same model achieved only 52.0% in a workflow that called the API directly (50 items in the Open-Nav 100 episode protocol), and 16.0% of them were reaching the step limit rather than actively stopping. The harness, reasoning level, episode set, and success determination may have changed at the same time between the two articles, and the 27-point gap cannot be attributed. This digest's assessment: If 79.0% follows the 100 episode subset, the half-width of the 95% confidence interval under the binomial distribution is about 8 percentage points, and "exceeding the supervised methods by 6.9 points" falls within the statistical error, and the supervised results are mostly reported on the complete val-unseen (1,839 episodes), not the same set comparison.
- **The numbers on the same R2R-CE are increasingly derived from different information conditions.** Three new results in this issue: SparseNav training-free, autonomous, 42.8% (val-unseen); Talk2Escape asks the algorithm oracle or humans for help when spinning or deviating, 66.0%; GPT-6-Astra uses the ultra reasoning level of the closed-source cutting-edge model, 79.0%. The difference between the three comes first from "what information the system can get", and secondly from the method itself. There is also a direct conflict between the sources: Talk2Escape claims that 66.0% exceeds the current supervised SOTA, and according to the expression of GPT-6-Astra, the best supervised method adopted by the author is 72.1%. GroundingVLN, covered in the previous issue, also reported 69.9% on the complete val-unseen.
- **Harness spreads from navigation to the entire embodiment stack and begins to feed back model training.** There were 3 harness studies in the last issue, and 5 in this issue are directly harness-themed (HarnessPAI, RegenHarness, AdaHVLA, Robo-Harness K1, World Action Agent), and GPT-6-Astra is evaluated in the Codex harness. The emerging common practices are twofold. The first is to only change the harness configuration without changing the model weight, and rely on execution record-driven revision. The second is to distill the execution trajectory of the harness back into the model: HarnessPAI used the data collected by the converged program to fine-tune π0.5, and LIBERO-PRO increased by another 38.8 points; World Action Agent used the harness trajectory to fine-tune Qwen3.5-9B, and the out-of-distribution success rate increased from 1.7% to 43.3%; the student model of Robo-Harness K1 only used 107 teacher episodes. Of these, only AdaHVLA reported navigation-side numbers (NaVILA-LH rose from 22.5% to a maximum of 57.5%).
- **Navigation and VLA both converge to "bounded memory" at the same time: a fixed-size state that replaces the growing context.** On the navigation side, VNT-PA uses pose-indexed keyframe sets to characterize the environment, PointNav (HM3D) SR 93.3%, SPL 90.4%; SparseNav only saves sparse landmarks required by the current sub-instruction; MemCtrl uses about half of the context to achieve an average of +6% and a long instruction subset +20%. On the VLA side, SmoLSTM, MemBodied, and StateMem all replace historical frame stacking with fixed-size memory states. The ablation of SmoLSTM is the most direct: the recurrent state is reset at each step, and the complete task success rate drops from 77.5% to 7.0%.
- **Evaluation studies are beginning to systematically challenge leaderboard rankings.** PopNavShift shows rank-dependent metrics for social navigation strategies: changing only time pressure, there is an 8.6% inversion by robot time ranking, a 22.4% inversion by pedestrian delay, and a 23.9% inversion by worst decile delay. RoboFollow pointed out that language is redundant under "low scene entropy", and the results of nine VLA/WAM in basic settings cannot be reliably transferred to perturbation settings. DPed-VLN combines social safety indicators with navigation efficiency to evaluate. VLN interpretability work shows that navigation progress is internally encoded in the policy.
- There are 124 independent new works in this issue, of which 38 are from the main pool (navigation + embodied agent), and 2 boundary entries (ACE, WORLDS) are analyzed together; 0 WeChat public account sources. The arXiv retrieval interface continues to be unavailable in this issue, and OAI-PMH metadata is used for local matching according to the original search formula; compared in the same time window of the previous issue, the main pool recall is 38/41.

## 2. Priority reading list
{: id="二优先阅读清单"}

1. **[GPT-6-Astra Lights Up](https://arxiv.org/abs/2609.29861v1)** ·  zero-shot  VLN-CE， monocular  RGB
   - Contribution: The general foundation model autonomously decides to observe, move and stop under the minimal interface of Codex harness.
   - Evidence: R2R-CE SR 79.0% (ultra inference); episode set, SPL, inference cost not given in abstract.
   - Reason: Currently the highest zero-shot statement, the caliber needs to be verified before quoting.
2. **[SparseNav](https://arxiv.org/abs/2609.26408v1)** · training-free VLN-CE + quadruped real robot
   - Contribution: Decide what to sense based on the current sub-instruction, and only maintain geometric BEV and sparse landmark memory.
   - Evidence: R2R-CE val-unseen SR 42.8%, RxR-CE val-unseen SR 40.7%.
   - Reasons: Autonomous, training-free, both benchmarks are reported, hardware stacks are common, and the barrier to reproduction is low.
3. **[Talk2Escape](https://arxiv.org/abs/2609.28296v1)** · Conversational error correction VLN
   - Contribution: After detecting rotation or deviation, grounding questions are issued to the oracle or humans to turn open-loop navigation into closed-loop.
   - Evidence: R2R-CE SR 66.0%; consistent improvement on various base agents on R2R-CE, RxR-CE, and VLNVerse.
   - Reason: The detector can be reused independently; its SOTA statement conflicts with other sources.
4. **[VNT-PA](https://arxiv.org/abs/2609.21212v1)** · PointNav（HM3D）
   - Contribution: Attention is determined by pose difference. The keyframe set is the environment representation and can be fused across trajectories.
   - Evidence: HM3D PointNav SR 93.3%, SPL 90.4%.
   - Reason: Historical experience can be reused without building an explicit map.
5. **[AdaHVLA](https://arxiv.org/abs/2609.29204v1)** · Adaptive harness for long-range VLA execution
   - Contribution: Testable hypotheses drive harness revision, and revision maps preserve evidence and effects.
   - Evidence: NaVILA-LH average test success rate 22.5% → maximum 57.5% (simulation).
   - Reason: The only one in this harness job that has navigation side numbers.
6. **[NaviScale](https://arxiv.org/abs/2609.27218v1)** · Semantic Map ObjectNav
   - Contribution: Splicing real floor plans and room-level semantic maps, and generating training data for complete predictions on a large scale.
   - Evidence: HM3D SR 64.3% / SPL 34.8%, MP3D SR 43.1% / SPL 16.8%.
   - Reason: Do not change the prediction structure, only expand the comparison points of the data.
7. **[What do VLM-Based VLN Models Rely on](https://arxiv.org/abs/2609.24576v1)** · VLN interpretability and activation guidance
   - Contribution: Use intervention indicators to measure the causal impact of each modality on decision-making, and extract behavioral activation vectors.
   - Evidence: The summary does not provide verifiable figures.
   - Reason: Provides a direct diagnostic method for "whether the strategy is using instructions".
8. **[DPed-VLN](https://arxiv.org/abs/2609.21504v1)** · Dynamic Pedestrian VLN Benchmark
   - Contribution: 33,093 episodes on Habitat 3.0, jointly evaluating navigation efficiency and social safety.
   - Evidence: The abstract only gives relative conclusions and does not provide verifiable figures.
   - Reason: Ground-based VLN enters the first substantial benchmark in dynamic crowd scenes.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. GPT-6-Astra: no controlled variable separates the 79.0% and 52.0% results
{: id="1-gpt-6-astra-两篇对读790-与-520-之间没有一个变量是被控制的"}

**Problem.** Whether the general foundation model can complete continuous environment VLN without adding any navigation-specific components. The article in the last issue ([How Far Can GPT-6-Astra Go? ](https://arxiv.org/abs/2609.20116v2), the original title GPT-6-Astra in a Navigation Workflow, this issue reappears as v2 after the title is revised, not included in the new addition) has been exposed: correct local judgment does not mean that it can continue to advance and stop at the correct position.

**Method.** The new article uses the smallest interface under the Codex harness, which only provides monocular RGB; the model decides by itself when to observe, how to move, and when to stop; there is no navigation fine-tuning, no trained waypoint predictor, and no pre-built map. The abstract only reports the ultra reasoning level, indicating that multiple reasoning levels were evaluated. The old article directly called the API's observation-decision-execution workflow, and each request brought in selected observations, execution feedback, and retained progress records.

**Evidence.**'s new paper reports R2R-CE SR 79.0%, which is 13.0 points higher than the strongest zero-shot and 6.9 points higher than the strongest supervised method; the abstract does not give SPL, nDTW, episode number and inference cost. The old article reported SR 52.0%, SPL 48.9%, and nDTW 70.8% on 50 of the Open-Nav 100 episodes, of which 36.0% were successful in active STOP and 16.0% in the upper limit of steps.

**Value.** This digest's assessment: If 79.0% is established under a comparable agreement, the "foundation model + minimum harness" has exceeded the specially trained VLN-CE system, and the value of the self-developed navigation module needs to be re-evaluated. This is exactly what the author’s fourth conclusion claims.

**Limitations.**

- The evaluation set is not specified. The "common zero-shot R2R-CE benchmark" in the zero-shot literature mostly refers to the 100 episode subset of Open-Nav (This digest's assessment). If so, with a 95% confidence interval half-width of about 8 percentage points, "6.9 points over supervised" is not statistically significant and is not the same set as the supervised results on the full val-unseen.
- The single-step delay and call cost of ultra inference are not reported, and the availability of real robot cannot be judged.
- The harness, reasoning level, episode set, and success determination may change at the same time between the two articles, and the 27-point gap cannot be attributed separately.
- The author himself also states: Even with ultra reasoning, route execution and target verification are still unreliable, and seemingly reasonable local landmark matching does not necessarily lead to correct completion.

**Recommendation.** Read the text carefully and confirm the episode set, the SR and cost of each reasoning level, and whether success requires active STOP; if the log is made public, count the composition of "non-stop/wrong stop/yaw" in the failed episode. Don’t put 79.0% into a side-by-side comparison with a training-style approach before verifying it.

### 2. SparseNav: selective perception offers a reproducible zero-shot route
{: id="2-sparsenav少感知一点反而是更可复现的零样本路线"}

**Problem.** If map-based VLN performs semantic annotation on all visible objects, it will not only increase the perceptual overhead, but also flood the spatial representation read by the VLM planner with irrelevant objects.

**Method.** Continuously maintains lightweight geometric BEV maps and sparse landmark memory. The instruction manager tracks the progress and gives the landmark query corresponding to the current sub-instruction; only when the queried landmark is visible and its measured position is useful for the next decision, the open vocabulary segmentation is called. VLM selects among mixed candidates of "frontier point + local direction waypoint".

**Evidence.** Paper report: Without any training, R2R-CE val-unseen SR 42.8%, RxR-CE val-unseen SR 40.7%; there is also a sensing strategy and controlled ablation of each component (ablation numbers and SPL are not given in the abstract). The real robot is deployed on Unitree Go2, RealSense D455 is responsible for mapping and landmark grounding, and Livox MID-360 is responsible for positioning. There is no need to pre-build maps and it is verified in multiple indoor environments.

**Value.** Among the three new R2R-CE results in this issue, the only one that simultaneously satisfies the requirements of autonomy, training-free, specify val-unseen segmentation, and report synchronously on RxR-CE; the hardware stack is consistent with common quadruped platforms. "Determining what to sense based on subcommands" can be directly grafted onto any map-based zero-shot VLN system.

**Limitations.** The abstract does not state whether it is a full val-unseen, nor does it state the VLM and single-step latency used; 42.8% is not directly comparable to other zero-shot results reported on a subset of more than 100 episodes. Sparse landmark memory will discard objects that are not mentioned in the command but are useful for obstacle avoidance or relocation.

**Recommendation.** reproduces the instruction conditionalized perception triggering module and measures the actual decrease in the number of perception calls and latency; compare it with the dense semantic map baseline on the same episode set.

### 3. Talk2Escape: the detector is more reusable than the dialogue
{: id="3-talk2escape检测器比对话更值得带走"}

**Problem.** Single-round VLN is an open-loop execution: small deviations caused by perceptual aliasing, sensing noise, and odometry drift continue to accumulate, and the final task fails, and the system has no built-in recovery mechanism.

**Method.** A lightweight visual language module continuously monitors the agent's kinematics; when it detects local rotation or serious trajectory deviation, it converts the first-person observation into a short grounding problem and requests corrective feedback from the algorithm oracle or humans. Frames are independent of the underlying navigation model.

**Evidence.** The paper reports consistent improvements in various base agents on R2R-CE, RxR-CE, and VLNVerse; R2R-CE SR is 66.0%, which is said to exceed the current supervised and zero-shot SOTA; sim-to-real verification was done on Go2 quadruped. The summary does not give the episode set, SPL, average number of questions per episode, nor does it distinguish between the performance of oracle and human feedback.

**Value.** The spin/deviation detector itself does not rely on external feedback and can be used independently as a trigger for "when to replan/when to backtrack". This corresponds exactly to the stop and resume weak link pointed out in the previous issue.

**Limitations.** The corrected information comes from oracles or humans. It is a setting with privileged information. Its SR is not under the same information condition as the autonomous agent. There is a source conflict in the SOTA statement: the supervised best of GPT-6-Astra's new paper inference is 72.1%, and the last issue GroundingVLN reported 69.9% on the full val-unseen, both higher than 66.0%.

**Recommendation.** Read the text to confirm the frequency of questions and the specific settings of the oracle; connect the detector to your own system to create a "self-correcting" version that does not require manual labor, and quantify the net income without external feedback.

### 4. VNT-PA: attention based on pose differences rather than temporal order
{: id="4-vnt-pa让注意力看位姿差而不是看时间顺序"}

**Problem.** The learning navigation policy encodes the observation history in chronological order, and it is difficult to reuse the experience of previously traversing the same environment; systems that can reuse experience usually need to explicitly build a map or topology map first, and then plan on it.

**Method.** The context of the Transformer planner is a set of depth keyframes indexed by camera pose. Using pose as position encoding, attention is determined by the pose difference between key frames, regardless of time order. During training, the shortest path planner on the ground truth grid is imitated; during inference, only the current pose and target position are used to query the spatial context.

**Evidence.** Paper report HM3D verification scene PointNav SR 93.3%, SPL 90.4%; navigation performance and training efficiency are better than the baseline of "encoding the same context as a time series" and "using pose as input feature"; the degradation under positioning noise is gentler than planning based on explicit maps (the abstract does not provide figures for noise experiments).

**Value.** The context is a collection of pose indexes, which can fuse frames from different trajectories during testing, providing a solution for "reusing experience when entering the same environment multiple times" without building an explicit map. It forms a contrast with the previous Navi-Agent: one relies entirely on pose, and the other completely abandons coordinates.

**Limitations.** The task is PointNav (the target is given in coordinates) and does not involve language; it relies on pose estimation; the teaching comes from the shortest path on the ground truth grid, and the real environment does not have such supervision; where the contextual keyframes come from (pre-traversal or accumulation in this round) the abstract does not explain.

**Recommendation.** In-depth study of the keyframe context construction method; replace the "visited area memory" with a keyframe set of pose index in the VLN system for small-scale verification.

### 5. AdaHVLA: this issue's only harness study with navigation results
{: id="5-adahvla本期唯一给出导航侧数字的-harness-工作"}

**Problem.** VLA is good at local control and command following, but long-range tasks require persistent memory and planning; task harness can retain history and track stage progress. The difficulty lies in aligning the two.

**Method.** Refining code-based coordination strategies using robot execution experience. Evidence analysis, harness revision, and behavioral evaluation are completed by different agents in an isolated context; the revision is driven by a testable "coordination hypothesis", and subsequent rollout is used to test whether the expected effect occurs. The stateful revision diagram connects evidence, hypotheses, revisions and observed effects, retains alternative harnesses and adaptation memories, and supports continuous adaptation across tasks and environments.

**Evidence.** The paper reports that the average test success rate of NaVILA-LH in simulation increased from 22.5% to a maximum of 57.5%; the success rate of manipulation tasks on three VLA backbones was up to 30.8 points higher than the initial harness; real robot deployment was a qualitative demonstration. The summary of NaVILA-LH's task composition is not stated; "highest" indicates that 57.5% is the optimal configuration rather than the mean; the comparison is the author's own initial harness, not an external method.

**Value.** The closed loop of's "hypothesis → revision → rollout test" and the same period [RegenHarness](https://arxiv.org/abs/2609.27612v1)'s "fixed regression check + acceptance of modifications only after release authorization" are two constraint strengths of the same problem: the former relies on experimental testing and revision, and the latter relies on gating restriction revision. RegenHarness also clearly distinguishes between "model proposal / controller termination / verified completion" and points out that completion depends on execution history rather than how close to the end - this is directly related to the stopping problem exposed by GPT-6-Astra.

**Limitations.** Only the simulation has numbers; the number of rollouts required for adaptation and the cost are not reported; it is unclear whether the harness revision will overfit the test task. RegenHarness only has real robot cases and no quantitative indicators.

**Recommendation.** Verify the definition of NaVILA-LH and the configuration corresponding to 57.5%; pay attention to the data structure of the revision diagram, which can be directly used to organize the failure case library of the navigation system.

## 4. Transferable methods
{: id="四可迁移方法"}

Only include work where the migration path can be clearly explained.

- **VLA Memory: [SmoLSTM](https://arxiv.org/abs/2609.22854v1), [MemBodied](https://arxiv.org/abs/2609.28256v1)**
  - Mechanism: Fixed size recurrent state replaces history frame stacking. SmoLSTM's complete task success rate on LIBERO-Mem is 77.5%, sub-goal coverage is 85.1%, trainable parameters are 0.04B, and the state is reset at each step, which drops to 7.0%. MemBodied uses the associative state plus the first frame scene anchor. On the five memory tasks of RMBench, the average success rate is 7.81 times that of the stateless strategy and 2.98 times that of ordinary loop memory.
  - Integration point: VLN historical encoding, replacing sliding window or KV cached historical context.
  - Prerequisites and risks: The manipulation task episode is short, and the memory requirements are mostly occlusion and distinguishing objects with the same appearance; navigation requires long-range spatial memory, which needs to be verified separately.
- **VLA inference scheduling: [React When You Need To](https://arxiv.org/abs/2609.22587v1)**
  - Mechanism: Dynamically adjust the inference interval according to the scene changes since the last inference, taking into account the continuity of actions and timely response.
  - Access position: Replanning trigger during action chunk execution, such as a pedestrian suddenly entering the field of view.
  - Prerequisites and risks: The evidence comes from operational scenarios. The average success rate under static and dynamic real robot settings is 95%, which is 55 points higher than the strongest baseline.
- **Perception Instrumentation: [Robo-Harness K1](https://arxiv.org/abs/2609.29389v1)**
  - Mechanism: Expose perception as a tool for the agent to query; the tool call trajectory can directly train the student model. Qwen3.5-9B only uses 107 teacher episodes, the new initial state accuracy is 44.2% (OpenVLA 30.2%), and the remaining task conditions are 13.9% (OpenVLA 0.0%).
  - Integration point: Perception interface design of the navigation agent; isomorphic to the MCP navigation framework of this issue and AnchorVLN of the previous issue.
  - Prerequisites and risks: The numbers are all from manipulation tasks.
- **Harness Trajectory Distillation: [World Action Agent](https://arxiv.org/abs/2609.29964v1), [HarnessPAI](https://arxiv.org/abs/2609.29166v1)**
  - Mechanism: Fine-tune a small model with the successful execution trajectory of the harness. WAA has an average success rate of 75.6% on LIBERO-Pro, and after fine-tuning Qwen3.5-9B with its trajectory, the out-of-distribution success rate rises from 1.7% to 43.3%. Expert data collected by HarnessPAI using the Converged program brings π0.5 up another 38.8 points on LIBERO-PRO.
  - Where to plug in: Distill the successful trajectory of zero-shot VLN harnesses (such as GPT-6-Astra-like systems) into small, deployable models.
  - Prerequisites and risks: The harness's own success rate must be high enough and the trajectory coverage must be diverse enough; what is obtained by distillation is the behavior of the harness, not stronger spatial capabilities.
- **language annotation efficiency: [LADA](https://arxiv.org/abs/2609.27747v1)**
  - Mechanism: First learn the potential action codebook from unlabeled observations, and then map a small number of language instructions to the codebook. With less than 5% language annotations, Bench2Drive’s closed-loop driving score is 87.98 and the success rate is 70.46%, which is the same as or higher than the fully supervised baseline.
  - Integration point: VLN command data is scarce, so you can first use unlabeled navigation trajectories to learn potential action codebooks.
  - Prerequisites and risks: Driving action space is very different from indoor navigation; Bench2Drive is a closed-loop simulation.
- **Parallel Hypothesis and Forensics: [WORLDS](https://arxiv.org/abs/2609.23841v1) (boundary entry, air)**
  - Mechanism: A persistent graph initialized with geographical priors, plus a parallel reasoner retaining competing interpretations and actively requesting evidence, a reviewer processing observations, and a referee deciding to select a target or take another round of evidence collection. CityNav achieved 51.8% SR on all 5,311 test episodes, 15.7 points higher than the published best (OSM-only, high-resolution ortho-protocol); 50.0% vs. 27.9% on 1,000 shared episodes of the same model and budget; censor contribution 5.9 points; had a quadcopter real robot demo.
  - Integration point: ObjectNav/instance navigation with large target ambiguity. Obtain evidence first and then set the target, replacing "leave when you see it".
  - Prerequisites and risks: City-scale aerial scenes rely on geographical prior maps; there are no equivalent prior sources indoors on the ground.
- **review design: [RoboFollow](https://arxiv.org/abs/2609.25636v1)**
  - Mechanism: "High scene entropy" principle, each training scene supports multiple kinematically different task branches, forcing the strategy to rely on language; the four-level perturbation protocol separates understanding and execution.
  - Integration point: When evaluating a self-built VLN, check whether there is an episode where there is only one reasonable route and language is dispensable.
  - Prerequisites and risks: Evidence from a manipulation task that performance on nine VLA/WAM at L0 does not reliably transfer to L1–L3.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **GPT-6-Astra Lights Up** (zero-shot VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.29861v1)
- **How Far Can GPT-6-Astra Go?** (zero-shot VLN-CE · A): The renamed v2 of the v1 analyzed in the previous issue is not included in the new additions in this issue; see the key analysis. [Source](https://arxiv.org/abs/2609.20116v2)
- **SparseNav** (training-free VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.26408v1)
- **Talk2Escape** (conversational error correction VLN · A): See key analysis. [Source](https://arxiv.org/abs/2609.28296v1)
- **VNT-PA** (PointNav · A): See key analysis. [Source](https://arxiv.org/abs/2609.21212v1)
- **NaviScale** (Semantic Map ObjectNav data · A): Splicing 24,000 floor plans of 12,794 properties with MP3D / HM3DSem room-level maps, generating 192,000 semantic maps to train the completion predictor; HM3D SR 64.3% / SPL 34.8%, MP3D SR 43.1% / SPL 16.8%, and real robot deployment. [Source](https://arxiv.org/abs/2609.27218v1)
- **Object-Path Graph** (Open Vocabulary Instance Navigation · A): Object-Path Graph unifies open vocabulary semantic reasoning and topological navigation, and is executed with semantic visual servoing between nodes without the need for dense mass reconstruction; verified by HM3D/Replica and real robot, the abstract does not give a number. [Source](https://arxiv.org/abs/2609.24189v1)
- **DPed-VLN** (Dynamic Pedestrian VLN Benchmark · A): 33,093 episodes, ORCA-controlled humanoid pedestrians, socially constrained expert paths; after LoRA adaptation, NaVILA and StreamVLN are better than the zero-shot version in several success and safety indicators, and the SR/SPL/STL of the self-proposed DPet-RL is the highest. [Source](https://arxiv.org/abs/2609.21504v1)
- **VLN Interpretability** (Interpretability and Activation Guidance · A): The strategy is sensitive to vision, instructions, and visual memory without relying on a single modality; navigation progress is internally encoded, and behavioral activation vectors can be zero-shot migrated to real scenes outside the distribution and improve performance. [Source](https://arxiv.org/abs/2609.24576v1)
- **NaViRrator** (map to instruction · A): start and end point on human-readable map → map coordinate system route skeleton → VLM generates instructions → hand over to pre-trained VLN strategy for execution; real robot SR/SPL is better than directly generating instructions compared with A* skeleton, etc., the abstract does not give a number. [Source](https://arxiv.org/abs/2609.21316v1)
- **Deploying FMs for Embodied Navigation** (Foundation model Deployment · A): TAP uses human habit data mined in the scene for personalized target search (Turtlebot real robot average +18%); MemCtrl uses "memory head" to actively manage context, multi-tasking average +6%, long instruction subset +20%, the context is about half of the baseline. [Source](https://arxiv.org/abs/2609.25666v1)
- **CoRelNav** (Multi-robot Relational Semantic Navigation · A): Task-conditioned multi-machine exploration is coupled with candidate-driven collaborative verification, and observations are aggregated across topological nodes to determine spatial relationships; simulation is better than the baseline, two real robots are deployed, the abstract does not give a number. [Source](https://arxiv.org/abs/2609.27720v1)
- **"Dear LLaVA, Please Drive"** (VLM Navigation Fine-tuning · A): Use a differentiable geometric cost field instead of annotated trajectories, and only rely on binocular depth learning for collision-free paths; task-specific LoRA updates less than 1% of parameters, and the environment SPL is not "competitive" (no number given). [Source](https://arxiv.org/abs/2609.22925v1)
- **MCP Navigation Representation Layer** (LLM + ROS Navigation · A): The occupation grid is converted into a metric image with pose, waypoint-level observations are semantically annotated, and exposed as a standard tool through MCP without changing the ROS navigation stack; the simulation indoor mapping coverage exceeds 97%. [Source](https://arxiv.org/abs/2609.27340v1)
- **ACE** (embodied exploration, boundary entry · A): Evidence grounding perception and exposure-aware movement are combined to alleviate the contradiction between premature termination and excessive continuation; it is said that the navigation task success rate is 18.0% higher than the previous SOTA, and the question and answer exploration efficiency is 10.3% higher. The abstract does not include a benchmark name. [Source](https://arxiv.org/abs/2609.22385v1)
- **ReVNM** (Remote Camera Visual Navigation · B): A single surveillance camera serves as both an observation source and an implicit map. The exo2ego module predicts the depth in front of the robot from a remote perspective; it is only trained in a randomly generated world, eliminating the need for fine-tuning and migration to real robots. [Source](https://arxiv.org/abs/2609.28976v1)

### 5.2 Embodied Agent, memory and planning
{: id="52-具身-agent记忆与规划"}

- **AdaHVLA** (adaptive harness · B): See key analysis. [Source](https://arxiv.org/abs/2609.29204v1)
- **RegenHarness** (robot agent harness · B): See item 5 of the key analysis. [Source](https://arxiv.org/abs/2609.27612v1)
- **HarnessPAI** (Physical AI harness · B): uses code as an executable and evolvable interface, executes the program in an open loop within a round, and uses feedback to revise the program and accumulate skills between rounds; LIBERO-PRO is 61.6 points higher than π0.5, and RoboCasa atomic tasks are 27.2 points higher than WorldDreamer; covering sweepers and foot-mounted platforms. [Source](https://arxiv.org/abs/2609.29166v1)
- **AquaMend** (Belief failure recovery · B): Choose between redetection, rollback, and continuation according to the expected loss on the detection-belief-action graph; 28 of the 32 self-built paired scenarios were recovered, and the average complete loss was 21.6% lower than restarting. The difference with decision-theoretic troubleshooting was not significant after Holm correction. [Source](https://arxiv.org/abs/2609.28973v1)
- **RoboFollow** (Instruction following diagnosis · B): See Section 4. [Source](https://arxiv.org/abs/2609.25636v1)
- **OmniEcho** (spatial audio + navigation · B): OmniEchoBench contains 197 real spatial audio and video scenes, 2,972 question and answer pairs, and 900 first-order Ambisonics navigation samples in 30 real environments; sound source guided navigation is "close to the traditional VLN level", the abstract does not give a number. [Source](https://arxiv.org/abs/2609.23407v2)
- **Active exploratory operation** (Agent-based operation · B): three modules of planning/perception/execution plus fine-grained perception-execution interleaving, when the target is initially invisible, search first and then operate; Find-and-Place task verification, the summary does not give a number. [Source](https://arxiv.org/abs/2609.29091v1)
- **WORLDS** (city-scale language search, boundary entry · B): See Section 4. [Source](https://arxiv.org/abs/2609.23841v1)
- **AquaCap** (underwater code is strategy agent · C): The double-layer agent converts instructions and observations into conditional plans and executable control programs. Failure-aware memory supports closed-loop re-planning; the simulation success rate is 66.43%, and ROV real robot captures and carries. [Source](https://arxiv.org/abs/2609.23133v1)
- **CE⁴L** (Multi-perspective continuous learning benchmark · C): ego / exo / ego-exo four-task continuous learning benchmark, with parameter-efficient subspace routing adapter baseline. [Source](https://arxiv.org/abs/2609.23492v1)
- **PUBG Ally** (in-game conversational agent · C): The language model agent uses tools to read game information and drive a faster control layer; iterative training on nearly 39,000 real-person game data; the positive response to recommendation willingness is 25.1 points higher than the negative response. [Source](https://arxiv.org/abs/2609.29837v1)
- **Listening and Mirroring** (VR empathic dialogue agent · C): 20-subject experiment, verbal attunement is the most reliable source of perceived empathy; it is a VR social agent and was mistakenly classified as an embodied agent when divided into pools. [Source](https://arxiv.org/abs/2609.27246v1)

### 5.3 Social Navigation, Off-Road and Outdoor
{: id="53-社会导航越野与户外"}

- **PopNavShift** (Social Navigation Evaluation · B): Use LLM to generate pedestrian motion parameters for 600 personality records, and compare three types of strategies under 8 crowd conditions and 7,488 paired runs; the ranking reversal ratio changes with the indicator (see conclusion 5). [Source](https://arxiv.org/abs/2609.21838v1)
- **Diffusion-guided online adaptation** (Social Navigation · B): Fixed diffusion policy, training-only noise strategy (DSRL), maintaining basic performance during fine-tuning of the deployment environment; multi-seed diffusion RL strategies are integrated as the basic strategy; hardware-in-the-loop verification. [Source](https://arxiv.org/abs/2609.24317v1)
- **Where Should I Join?** (Language-guided joining of the crowd · B): Recursive spectral partitioning generates a subset of candidate members, language condition image-geometric model sorting, and then uses crowd formation a priori to predict socially compliant joining poses; sub-second reasoning, real robot verification. [Source](https://arxiv.org/abs/2609.28467v1)
- **AcousticDiffusion** (sound source guided rescue navigation · B): The arrival direction of the microphone array is recursively fused into a BEV confidence field, and the conditional diffusion model generates a waypoint trajectory; the average azimuth error of the quadruped real robot is 64.9° (A* 98.2°, RRT 90.4°), and the final distance from the caller is 2.48 m (classic planning) 3.96 m). [Source](https://arxiv.org/abs/2609.21792v1)
- **Verti-WM** (Off-road World Model · C): Frozen Transformer handles rigid terrain, and neural symbolic ground mechanics handles deformable terrain; the prediction error is 34.6% / 21.7% lower than the pure data/pure physics baseline, and the training calculation time is saved 23.6 times; the real vehicle success rate is 80% (direct sim-to-real 40%). [Source](https://arxiv.org/abs/2609.23118v1)
- **TravPro** (off-road traversability ranking · C): Convert existing annotations into region preference pairs, and redistillize preference scores on prototypes of frozen VLM patch tokens; the average pairwise accuracy of five unseen domains is 0.915 (strongest baseline 0.783). [Source](https://arxiv.org/abs/2609.23673v1)
- **Outdoor GNSS Navigation Stack** (ROS 2 Outdoor Navigation · C): Single GNSS-IMU/dual antenna front end is replaceable, multiple controllers share a unified status interface; 800 field operations in the vineyard, average lateral error in in-row mixed mode 0.95 cm (single GNSS+IMU)/0.85 cm (dual antennas). [Source](https://arxiv.org/abs/2609.28933v1)
- **Planning Trajectories that Bounce** (collision tolerance planning · C): The reflection augmentation state diagram divides path classes according to the wall sequence used, and controlled wall collision can reduce execution time and control amount (simulation only). [Source](https://arxiv.org/abs/2609.27145v1)
- **BarrierFormer** (Safety Control · C): Transformer autoregressively generates predictive rolling to replace the model, barrier critic checks CBF constraints along the rolling, no online optimization required during inference. [Source](https://arxiv.org/abs/2609.23896v1)
- **ZIL** (image-point cloud registration · C): zero-shot asynchronous image to LiDAR registration foundation model; 1.4 million frames of training in 7 datasets, translation/rotation error reduced by up to 87%/76%. [Source](https://arxiv.org/abs/2609.22716v1)

### 5.4 Surface vessels, underwater vehicles, and UAVs
{: id="54-水面水下与无人机"}

- **RiverVLN** (Unmanned Surface Vehicle VLN·B): Long-range USV VLN benchmark under continuous river motion; PGT-NAV converts instructions into a visually verifiable semantic stage sequence and maintains the current stage online; Unity-ROS closed-loop average success rate is 0.79, verified by real ships. [Source](https://arxiv.org/abs/2609.23423v1)
- **AquaWorld** (Underwater World Generation · C): Consistent randomization of shared terrain structures; the same budget strategy training verification success rate is 21%, and the simulation-trained visual navigation policy is 95% successful in the physical water tank. [Source](https://arxiv.org/abs/2609.22670v1)
- **PhysAI-Bench** (UAV agent decision benchmark·C): 10,178 decision instances, including MCP tool calls, A2A interactions and 6G network status; among the 29 models, GPT-5.3 has the highest accuracy of 52.00%. [Source](https://arxiv.org/abs/2609.23695v1)

### 5.5 Secondary-pool roundup: VLA/manipulation, autonomous driving, and other work
{: id="55-次池速览vla操作--自动驾驶--其他"}

There are a total of 84 articles in the secondary pool (81 articles on VLA/manipulation, another 1 article AdaHVLA due to the navigation side digital shift to 5.2; 3 "other" articles), and no in-depth analysis will be performed. Those with transfer value have been included in Section 4. Grouped by theme as follows:

-  **Memory and long-range execution (12)** ：[SmoLSTM ](https://arxiv.org/abs/2609.22854v1) 、[StateMem ](https://arxiv.org/abs/2609.22684v1) 、[MemBodied ](https://arxiv.org/abs/2609.28256v1) 、[TaskAnchor ](https://arxiv.org/abs/2609.23580v1) 、[CommitFlow ](https://arxiv.org/abs/2609.21908v1) 、[CARE ](https://arxiv.org/abs/2609.24118v1) 、[H-VLA ](https://arxiv.org/abs/2609.22895v1) 、[X-Planner ](https://arxiv.org/abs/2609.25187v1) 、[TANDEM ](https://arxiv.org/abs/2609.28314v1) 、[LiMA ](https://arxiv.org/abs/2609.28431v1) 、[ActiveArena ](https://arxiv.org/abs/2609.24124v2) 、[SafeLoop ](https://arxiv.org/abs/2609.26313v1)
- **Harness and agent execution (3)**: [Robo-Harness K1](https://arxiv.org/abs/2609.29389v1), [World Action Agent](https://arxiv.org/abs/2609.29964v1), [AR-WAM](https://arxiv.org/abs/2609.23578v1)
-  **3D, geometry and visual representation (12)** ：[Grounded Action Model ](https://arxiv.org/abs/2609.23863v1) 、[Bridge3D ](https://arxiv.org/abs/2609.24525v1) 、[GALA ](https://arxiv.org/abs/2609.21948v1) 、[FOCAL-VLA ](https://arxiv.org/abs/2609.21228v1) 、[HABILIS Brain 0 ](https://arxiv.org/abs/2609.25558v1) ,[Terminal geometry cross-strategy analysis ](https://arxiv.org/abs/2609.21659v2) , [Topological visual cues ](https://arxiv.org/abs/2609.23944v1) , [3D visual language alignment-fusion ](https://arxiv.org/abs/2609.28222v1) 、[InfiNoVA ](https://arxiv.org/abs/2609.27734v1) 、[ActGaze ](https://arxiv.org/abs/2609.28955v1) 、[MaskVLA ](https://arxiv.org/abs/2609.23565v1) 、[PSR ](https://arxiv.org/abs/2609.21753v1)
- **World Model (5)**：[AffordanceWAM](https://arxiv.org/abs/2609.22332v2)、[Think Like a World Model](https://arxiv.org/abs/2609.24682v2)、[Imagine-RL](https://arxiv.org/abs/2609.24033v1)、[Prioritized Rollouts](https://arxiv.org/abs/2609.22879v1), [MachEmbodied-U0](https://arxiv.org/abs/2609.25627v1)
-  **RL post-training and online adaptation (13)** ：[SynthDemo-RL ](https://arxiv.org/abs/2609.21650v1) ,[Asynchronous playback anchoring online post-training ](https://arxiv.org/abs/2609.22888v1) 、[BEE ](https://arxiv.org/abs/2609.27450v1) 、[RouteRLT ](https://arxiv.org/abs/2609.26467v1) ,[Analysis of training after advantage guidance ](https://arxiv.org/abs/2609.28161v1) , [Uncertainty gated exploration noise ](https://arxiv.org/abs/2609.28838v1) 、[ForceRFT ](https://arxiv.org/abs/2609.22840v1) 、[FAN ](https://arxiv.org/abs/2609.21358v1) 、[Self-Adaptive VLA ](https://arxiv.org/abs/2609.30092v1) , [task semantic action calibration ](https://arxiv.org/abs/2609.23650v1) , [BNN full covariance smoothing ](https://arxiv.org/abs/2609.27244v1) , [Brain-inspired hierarchical modular continuous learning ](https://arxiv.org/abs/2609.25146v1) , [VLA federated fine-tuning test bed ](https://arxiv.org/abs/2609.22973v1)
- **Action representation and reasoning efficiency (11)**: [KerColle](https://arxiv.org/abs/2609.22335v1)、[Catch Me If You Can](https://arxiv.org/abs/2609.21022v1)、[Fewer Steps, Better Actions](https://arxiv.org/abs/2609.21216v1)、[React When You Need To](https://arxiv.org/abs/2609.22587v1),[FoldQuantVLA](https://arxiv.org/abs/2609.24433v1),[VLAQuantBench](https://arxiv.org/abs/2609.25376v1),[Decoupled Early Exits](https://arxiv.org/abs/2609.29382v1),[CereVLA](https://arxiv.org/abs/2609.27468v1),[Action Block VLA sim-to-real Pipeline ](https://arxiv.org/abs/2609.21817v1), [Action segmentation: beyond reconstruction error ](https://arxiv.org/abs/2609.25820v1), [Direction-scale decomposition action representation ](https://arxiv.org/abs/2609.28865v1)
- **Review and Safety (9)**: [LIBERO-VPro](https://arxiv.org/abs/2609.24350v1), [IndustrialVLA-Bench](https://arxiv.org/abs/2609.25562v1), [VLA-Scope](https://arxiv.org/abs/2609.21246v1), [SafeSt age](https://arxiv.org/abs/2609.21223v1), [VLPSA](https://arxiv.org/abs/2609.22462v1), [Noise space real-time policy guidance ](https://arxiv.org/abs/2609.21220v1), [CrossSafe](https://arxiv.org/abs/2609.28984v1), [Industrial robot arm backdoor ](https://arxiv.org/abs/2609.26868v1), [ReVeal](https://arxiv.org/abs/2609.23910v1)
-  **Contact, force touch and special scenes (13)** ：[ForeTac-VLA ](https://arxiv.org/abs/2609.20980v1) 、[CompVLA ](https://arxiv.org/abs/2609.23614v1) 、[Opt2VLA ](https://arxiv.org/abs/2609.23968v1) 、[VisForce ](https://arxiv.org/abs/2609.25785v1) 、[VT-Bridge ](https://arxiv.org/abs/2609.22606v1) 、[HEARTH ](https://arxiv.org/abs/2609.23418v1) 、[CableVLA ](https://arxiv.org/abs/2609.25606v1) 、[Imperfection for Precision ](https://arxiv.org/abs/2609.26672v1) 、[SCULPT-VLA ](https://arxiv.org/abs/2609.23275v1) , [Capability-aware shared control ](https://arxiv.org/abs/2609.25369v1) 、[MATE ](https://arxiv.org/abs/2609.26520v1) 、[MedVLA ](https://arxiv.org/abs/2609.25756v1) , [StenoVLA-3D (gastrointestinal endoscopy) ](https://arxiv.org/abs/2609.24187v2)
- **Autonomous driving (4)**: [Beyond the Leaderboard](https://arxiv.org/abs/2609.22582v1), [PRIME](https://arxiv.org/abs/2609.22040v1), [ZYT-World](https://arxiv.org/abs/2609.21712v2), [LADA](https://arxiv.org/abs/2609.27747v1)
- **pre-training data and general model (2)**: [AtomEgo](https://arxiv.org/abs/2609.21461v1), [ME-VLM](https://arxiv.org/abs/2609.24526v2)

### 5.6 News and non-paper items
{: id="56-资讯与非论文"}

None in this issue. WeChat public account source has 0 items in the database.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **Harness is becoming a data engine.** The harness work in the previous issue only reported the overall system indicators; in this issue, there are three jobs (HarnessPAI, World Action Agent, Robo-Harness K1) that distill the harness execution trajectory back to a small model, and all reported significant improvements. This digest's assessment: The same route is likely to appear on the navigation side - first use the closed-source frontier model and harness to build a zero-shot system with high success rate, and then distill the trajectory into a deployable model. 79.0% of GPT-6-Astra, if verified, is upstream of this route.
- **memory design shifts from "how much to store" to "how much to state".** VNT-PA, SparseNav, and MemCtrl on the navigation side, and SmoLSTM, MemBodied, and StateMem on the VLA side, all replace the growing context with fixed or bounded representations. SmoLSTM dropped from 77.5% to 7.0% after resetting the state, but MemCtrl improved with half the context. The two sets of evidence jointly show that the bottleneck is not the context length, but what to remember.
- The comparison dimension of **zero-shot VLN-CE is expanded from "method" to "information conditions".** Whether to use external correction (Talk2Escape), which foundation model and reasoning level to use (GPT-6-Astra), and whether to use training-free autonomy (SparseNav) can already explain the performance differences better than method details. The problem in the previous issue was that the episode subsets were not uniform, and in this issue, the information conditions were not uniform.
- **non-visual signal enters navigation.** In this issue, OmniEcho (spatial audio navigation sample), AcousticDiffusion (microphone array guides quadrupeds), and ReVNM (remote surveillance camera) all make navigation rely on signal sources other than onboard vision. None of the three yet interface with standard VLN benchmarks.

### Research gaps
{: id="研究空白"}

- **zero-shot R2R-CE registration specification missing information condition.** Three types of misaligned variables, namely episode subset, external correction, foundation model and reasoning level, coexist on the same benchmark. No paper reports these three items at the same time.
- There is still no specific indicator for **stop and completion judgment. The author of** GPT-6-Astra admits that target verification under ultra reasoning is still unreliable. RegenHarness advocates that completion depends on the execution history rather than the end distance. The old article in the previous issue had a bottoming success of 16.0%. All three points point to the same link, but no one has separately reported stop-related indicators in this issue.
- **The cost of people in the loop is not transparent.** Talk2Escape does not report the frequency of questions, nor does it provide a curve of "SR changes with the number of interventions", so it is impossible to judge how much labor is required for 66.0%.

### Suggested actions
{: id="建议动作"}

**High priority**

- **intensively read and verified** GPT-6-Astra Lights Up: 79.0% corresponding episode set, SR and cost of each reasoning level, whether success requires active STOP; no direct comparison will be entered before verification.
- **carefully read and reproduced** SparseNav: reproduced the conditional sensing trigger of the command, and measured the decrease in the number of sensing calls and latency.
- **draws on the spin/deviation detector of** Talk2Escape: instead of manual work, it triggers its own re-planning or backtracking to measure the net income.
- **Update the tracking table**: In addition to "evaluation subset + number of episodes + whether zero-shot", the R2R-CE registration table adds three new items: "whether to use external correction", "foundation model and reasoning level" and "whether active STOP is required for success".

**Medium priority**

- **Read closely** VNT-PA: The construction method of key frame context, and evaluate whether it can replace the topology memory in its own system.
- **Verify** AdaHVLA:NaVILA-LH's mission definition corresponds to 57.5% of the configuration.
- **Minimal verification**: Try the fixed size recurrent state (SmoLSTM / MemBodied idea) at the VLN history encoding, and compare it with the sliding window context.
- **Track the** DPed-VLN benchmark release and the LoRA adaptation results of NaVILA / StreamVLN under dynamic pedestrians.

**Low priority**

- **Track whether the** NaviScale dataset is public.
- **Defer** secondary pool VLA quantification and inference acceleration work: the mechanisms can be used for reference, but they have not been verified on VLN-CE.

> **update (2026-10-01)**: GPT-6-Astra Lights Up has released v2 on 2026-09-25. The 79.0% / 76.0% quoted in this report are the single-run results of v1; v2 is changed to the average of three runs at each reasoning level, and ultra is 81.3±2.5% (SPL 71.5±1.7), medium is 75.7±1.5% (SPL 65.6±2.1), still only evaluated on R2R-CE-100. See the [GPT-6-Astra reading in VLN Papers](/en/VLN-Papers/#gpt-6-astra) for details.
