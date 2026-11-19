---
layout: post
lang: en
translation_id: vln-weekly-2026-09-27
permalink: /en/vln-weekly-2026-09-27/
source_path: _posts/weekly-reports/2026-09-27-VLN-Weekly.md
source_url: /vln-weekly-2026-09-27/
source_revision_date: 2026-11-07
translation_updated: 2026-11-19
title: "Embodied Navigation Weekly · Issue 6"
date: 2026-09-27
last_modified_at: 2026-11-07
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

- **A new 79.0% claim appears for zero-shot R2R-CE, but for now it is impossible to verify what it means.** GPT-6-Astra uses only monocular RGB through the smallest interface of the Codex harness, with no navigation fine-tuning, no trained waypoint predictor, and no prebuilt map. The paper reports an SR of 79.0% at the ultra reasoning level, 13.0 points above the strongest zero-shot method and 6.9 points above the strongest supervised method. A week earlier, the same model reached only 52.0% in a workflow that called the API directly (50 episodes from the Open-Nav 100-episode protocol), 16.0% of which came from reaching the step limit rather than stopping actively. The harness, reasoning level, episode set, and success criterion may all have changed between the two papers, so the 27-point gap cannot be attributed. This digest's assessment: if 79.0% was measured on the 100-episode subset, the half-width of the 95% confidence interval under a binomial distribution is about 8 percentage points, so "6.9 points above supervised methods" falls within statistical error; moreover, supervised results are mostly reported on the full val-unseen (1,839 episodes), so this is not a same-set comparison.
- **Numbers on the same R2R-CE increasingly come from different information conditions.** Three new results in this issue: SparseNav, training-free and autonomous, 42.8% (val-unseen); Talk2Escape, which asks an algorithmic oracle or a human for help when the agent spins or deviates, 66.0%; and GPT-6-Astra, which uses the ultra reasoning level of a closed-source frontier model, 79.0%. The differences among the three come first from "what information the system can access" and only second from the method itself. The sources also directly conflict: Talk2Escape claims 66.0% exceeds the current supervised SOTA, while according to GPT-6-Astra the best supervised method it adopts reaches 72.1%. GroundingVLN, covered in the previous issue, also reported 69.9% on the full val-unseen.
- **The harness is spreading from navigation to the whole embodied stack and has begun to feed back into model training.** The last issue had 3 harness works; this issue has 5 directly themed on harnesses (HarnessPAI, RegenHarness, AdaHVLA, Robo-Harness K1, World Action Agent), and GPT-6-Astra is evaluated inside the Codex harness. Two common practices are emerging. The first changes only the harness configuration, not the model weights, and relies on revision driven by execution records. The second distills the harness's execution trajectories back into the model: HarnessPAI used data collected by the converged program to fine-tune π0.5, gaining another 38.8 points on LIBERO-PRO; World Action Agent fine-tuned Qwen3.5-9B on harness trajectories, raising out-of-distribution success from 1.7% to 43.3%; and Robo-Harness K1's student model used only 107 teacher episodes. Of these, only AdaHVLA reports navigation-side numbers (NaVILA-LH rises from 22.5% to at most 57.5%).
- **Navigation and VLA are both converging on "bounded memory": a fixed-size state replaces a growing context.** On the navigation side, VNT-PA represents the environment with pose-indexed keyframe sets, reaching PointNav (HM3D) SR 93.3% and SPL 90.4%; SparseNav stores only the sparse landmarks required by the current sub-instruction; MemCtrl uses about half the context to gain +6% on average and +20% on the long-instruction subset. On the VLA side, SmoLSTM, MemBodied, and StateMem all replace stacked history frames with fixed-size memory states. SmoLSTM's ablation is the most direct: when the recurrent state is reset at every step, full-task success drops from 77.5% to 7.0%.
- **Evaluation studies are starting to challenge leaderboard rankings systematically.** PopNavShift shows that rank-based metrics for social navigation policies are fragile: changing only the time pressure inverts the ranking by robot time in 8.6% of cases, by pedestrian delay in 22.4%, and by worst-decile delay in 23.9%. RoboFollow points out that language is redundant under "low scene entropy", and that results of nine VLA/WAM models in the base setting do not transfer reliably to perturbed settings. DPed-VLN combines social-safety metrics with navigation efficiency in its evaluation. Work on VLN interpretability shows that navigation progress is encoded internally in the policy.
- This issue has 124 independent new works, of which 38 come from the main pool (navigation + embodied agents), and 2 borderline entries (ACE, WORLDS) are analyzed together; the WeChat source contributed 0 entries. The arXiv retrieval interface remained unavailable in this issue, so OAI-PMH metadata was matched locally using the original search expression; compared over the same time window as the previous issue, the main-pool recall is 38/41.

## 2. Priority reading list
{: id="二优先阅读清单"}

1. **[GPT-6-Astra Lights Up](https://arxiv.org/abs/2609.29861v1)** · Zero-shot VLN-CE, monocular RGB
   - Contribution: A general foundation model decides on its own when to observe, move, and stop under the minimal interface of the Codex harness.
   - Evidence: R2R-CE SR of 79.0% (ultra reasoning); the abstract does not give the episode set, SPL, or inference cost.
   - Why it matters: The highest zero-shot claim to date; its protocol must be verified before it is quoted.
2. **[SparseNav](https://arxiv.org/abs/2609.26408v1)** · training-free VLN-CE + quadruped real robot
   - Contribution: Decides what to perceive based on the current sub-instruction, maintaining only a geometric BEV and sparse landmark memory.
   - Evidence: R2R-CE val-unseen SR 42.8%, RxR-CE val-unseen SR 40.7%.
   - Why it matters: Autonomous and training-free, reports both benchmarks, uses a common hardware stack, and has a low barrier to reproduction.
3. **[Talk2Escape](https://arxiv.org/abs/2609.28296v1)** · VLN with conversational error correction
   - Contribution: After detecting spinning or deviation, it asks grounding questions to an oracle or a human, turning open-loop navigation into a closed loop.
   - Evidence: R2R-CE SR of 66.0%; consistent gains over various base agents on R2R-CE, RxR-CE, and VLNVerse.
   - Why it matters: The detector can be reused independently; its SOTA claim conflicts with other sources.
4. **[VNT-PA](https://arxiv.org/abs/2609.21212v1)** · PointNav（HM3D）
   - Contribution: Attention is determined by pose differences; a keyframe set serves as the environment representation and can be fused across trajectories.
   - Evidence: HM3D PointNav SR 93.3%, SPL 90.4%.
   - Why it matters: Past experience can be reused without building an explicit map.
5. **[AdaHVLA](https://arxiv.org/abs/2609.29204v1)** · Adaptive harness for long-horizon VLA execution
   - Contribution: Testable hypotheses drive harness revision, and a revision graph preserves evidence and effects.
   - Evidence: NaVILA-LH average test success goes from 22.5% to at most 57.5% (simulation).
   - Why it matters: The only one of these harness works with navigation-side numbers.
6. **[NaviScale](https://arxiv.org/abs/2609.27218v1)** · Semantic-map ObjectNav
   - Contribution: Stitches real floor plans with room-level semantic maps to generate training data for map completion at scale.
   - Evidence: HM3D SR 64.3% / SPL 34.8%, MP3D SR 43.1% / SPL 16.8%.
   - Why it matters: A comparison point that expands only the data and leaves the prediction architecture unchanged.
7. **[What do VLM-Based VLN Models Rely on](https://arxiv.org/abs/2609.24576v1)** · VLN interpretability and activation steering
   - Contribution: Uses intervention metrics to measure the causal influence of each modality on decisions, and extracts behavioral activation vectors.
   - Evidence: The summary gives no verifiable figures.
   - Why it matters: Provides a direct diagnostic for "is the policy actually using the instruction?".
8. **[DPed-VLN](https://arxiv.org/abs/2609.21504v1)** · VLN benchmark with dynamic pedestrians
   - Contribution: 33,093 episodes on Habitat 3.0, jointly evaluating navigation efficiency and social safety.
   - Evidence: The abstract only gives relative conclusions and no verifiable figures.
   - Why it matters: Ground-based VLN's first substantial benchmark for dynamic crowd scenes.

## 3. Featured work in depth
{: id="三重点工作分析"}

### 1. GPT-6-Astra
{: id="1-gpt-6-astra-两篇对读790-与-520-之间没有一个变量是被控制的"}

**No controlled variable separates the 79.0% and 52.0% results.**

**Problem.** Can a general foundation model complete VLN in continuous environments without any navigation-specific components? The paper analyzed in the last issue ([How Far Can GPT-6-Astra Go?](https://arxiv.org/abs/2609.20116v2), originally titled GPT-6-Astra in a Navigation Workflow, which reappears this issue as v2 after a title revision and is not counted as new) had already shown that correct local judgments do not mean the agent can keep advancing and stop at the right place.

**Method.** The new paper uses the smallest interface under the Codex harness, providing only monocular RGB; the model decides for itself when to observe, how to move, and when to stop; there is no navigation fine-tuning, no trained waypoint predictor, and no prebuilt map. The abstract reports only the ultra reasoning level, implying that several reasoning levels were evaluated. The old paper called the API directly in an observe-decide-execute workflow, with each request carrying selected observations, execution feedback, and retained progress records.

**Evidence.** The new paper reports R2R-CE SR of 79.0%, 13.0 points above the strongest zero-shot method and 6.9 points above the strongest supervised method; the abstract gives no SPL, nDTW, number of episodes, or inference cost. The old paper reported SR 52.0%, SPL 48.9%, and nDTW 70.8% on 50 of the Open-Nav 100 episodes, of which 36.0% succeeded by active STOP and 16.0% by hitting the step limit.

**Value.** This digest's assessment: if 79.0% holds under a comparable protocol, "foundation model + minimal harness" has already surpassed specially trained VLN-CE systems, and the value of in-house navigation modules needs to be reassessed. This is exactly what the author claims in the fourth conclusion.

**Limitations.**

- The evaluation set is not specified. The "common zero-shot R2R-CE benchmark" in the zero-shot literature mostly means the 100-episode subset of Open-Nav (this digest's assessment). If so, with a 95% confidence interval half-width of about 8 percentage points, "6.9 points above supervised" is not statistically significant, and it is not the same set as the supervised results on the full val-unseen.
- The per-step latency and call cost of ultra reasoning are not reported, so availability on a real robot cannot be judged.
- The harness, reasoning level, episode set, and success criterion may all have changed between the two papers, so the 27-point gap cannot be attributed separately.
- The author also states that even with ultra reasoning, route execution and goal verification remain unreliable, and seemingly reasonable local landmark matching does not necessarily lead to correct completion.

**Recommendation.** Read the full text closely and confirm the episode set, the SR and cost at each reasoning level, and whether success requires an active STOP; if logs are released, count the composition of "no stop / wrong stop / yaw" in failed episodes. Do not put 79.0% in a side-by-side comparison with trained methods before verifying it.

### 2. SparseNav
{: id="2-sparsenav少感知一点反而是更可复现的零样本路线"}

**Selective perception offers a reproducible zero-shot route.**

**Problem.** If map-based VLN semantically annotates every visible object, it not only increases perception overhead but also floods the spatial representation read by the VLM planner with irrelevant objects.

**Method.** It continuously maintains a lightweight geometric BEV map and a sparse landmark memory. An instruction manager tracks progress and issues landmark queries corresponding to the current sub-instruction; open-vocabulary segmentation is called only when the queried landmark is visible and its measured position is useful for the next decision. The VLM chooses among mixed candidates of "frontier points + local-direction waypoints".

**Evidence.** The paper reports, without any training, R2R-CE val-unseen SR of 42.8% and RxR-CE val-unseen SR of 40.7%, along with controlled ablations of the sensing strategy and each component (the abstract gives no ablation numbers or SPL). The real robot is deployed on a Unitree Go2, with a RealSense D455 handling mapping and landmark grounding and a Livox MID-360 handling localization; it needs no prebuilt map and is validated in several indoor environments.

**Value.** Among the three new R2R-CE results in this issue, it is the only one that is autonomous, training-free, specifies the val-unseen split, and also reports on RxR-CE; its hardware stack matches common quadruped platforms. "Decide what to perceive from the sub-instruction" can be grafted directly onto any map-based zero-shot VLN system.

**Limitations.** The abstract does not say whether it is the full val-unseen, nor which VLM and per-step latency were used; 42.8% is not directly comparable to other zero-shot results reported on subsets of about 100 episodes. The sparse landmark memory discards objects that are not mentioned in the instruction but are useful for obstacle avoidance or relocalization.

**Recommendation.** Reproduce the instruction-conditioned perception trigger module and measure the actual reduction in perception calls and latency; compare against a dense semantic-map baseline on the same episode set.

### 3. Talk2Escape
{: id="3-talk2escape检测器比对话更值得带走"}

**The detector is more reusable than the dialogue.**

**Problem.** Single-pass VLN executes open-loop: small deviations from perceptual aliasing, sensing noise, and odometry drift keep accumulating until the task fails, and the system has no built-in recovery mechanism.

**Method.** A lightweight vision-language module continuously monitors the agent's kinematics; when it detects local spinning or serious trajectory deviation, it converts the first-person observation into a short grounding question and asks an algorithmic oracle or a human for corrective feedback. The framework is independent of the underlying navigation model.

**Evidence.** The paper reports consistent gains over various base agents on R2R-CE, RxR-CE, and VLNVerse; R2R-CE SR is 66.0%, said to exceed current supervised and zero-shot SOTA; sim-to-real validation was done on a Go2 quadruped. The summary gives no episode set, SPL, or average number of questions per episode, and does not separate the performance of oracle and human feedback.

**Value.** The spin/deviation detector itself does not depend on external feedback and can be used on its own as a trigger for "when to replan / when to backtrack". This lines up exactly with the stop-and-resume weak link pointed out in the previous issue.

**Limitations.** The corrective information comes from oracles or humans, a setting with privileged information, so its SR is not under the same information conditions as autonomous agents. There is a source conflict in the SOTA claim: the best supervised result inferred from the new GPT-6-Astra paper is 72.1%, and GroundingVLN in the last issue reported 69.9% on the full val-unseen, both higher than 66.0%.

**Recommendation.** Read the full text to confirm the frequency of questions and the specific oracle settings; attach the detector to your own system to build a "self-correcting" version that needs no human, and quantify the net gain without external feedback.

### 4. VNT-PA
{: id="4-vnt-pa让注意力看位姿差而不是看时间顺序"}

**Attention based on pose differences rather than temporal order.**

**Problem.** Learned navigation policies encode observation history in chronological order, making it hard to reuse experience from earlier traversals of the same environment; systems that can reuse experience usually first need to build an explicit map or topological graph and then plan on it.

**Method.** The context of the Transformer planner is a set of depth keyframes indexed by camera pose. Pose serves as the positional encoding, so attention is determined by the pose differences between keyframes regardless of temporal order. Training imitates a shortest-path planner on the ground-truth grid; inference queries the spatial context using only the current pose and the goal position.

**Evidence.** The paper reports PointNav SR 93.3% and SPL 90.4% on HM3D validation scenes; navigation performance and training efficiency beat baselines that "encode the same context as a time series" and "use pose as an input feature"; degradation under localization noise is gentler than for planning based on an explicit map (the abstract gives no figures for the noise experiments).

**Value.** The context is a pose-indexed set, so frames from different trajectories can be fused at test time, offering a way to "reuse experience when entering the same environment repeatedly" without building an explicit map. It contrasts with Navi-Agent in the previous issue: one relies entirely on pose, the other abandons coordinates completely.

**Limitations.** The task is PointNav (the goal is given in coordinates) and involves no language; it relies on pose estimation; supervision comes from the shortest path on the ground-truth grid, which real environments do not offer; and the abstract does not explain where the context keyframes come from (a prior traversal or accumulation within the current run).

**Recommendation.** Study the keyframe context construction closely; in a VLN system, replace "visited-area memory" with a pose-indexed keyframe set for small-scale verification.

### 5. AdaHVLA
{: id="5-adahvla本期唯一给出导航侧数字的-harness-工作"}

**The only harness study in this issue with navigation results.**

**Problem.** VLAs are good at local control and instruction following, but long-horizon tasks need persistent memory and planning; a task harness can retain history and track stage progress. The difficulty lies in aligning the two.

**Method.** It refines code-based coordination policies using robot execution experience. Evidence analysis, harness revision, and behavior evaluation are done by different agents in isolated contexts; revisions are driven by testable "coordination hypotheses", and later rollouts test whether the expected effect occurs. A stateful revision graph connects evidence, hypotheses, revisions, and observed effects, retains alternative harnesses and adaptation memory, and supports continuous adaptation across tasks and environments.

**Evidence.** The paper reports that NaVILA-LH's average test success in simulation rises from 22.5% to at most 57.5%; manipulation task success on three VLA backbones is up to 30.8 points above the initial harness; real-robot deployment is a qualitative demonstration. The task composition of NaVILA-LH is not described in the summary; "at most" indicates that 57.5% is the best configuration rather than the mean; and the comparison is against the authors' own initial harness, not an external method.

**Value.** AdaHVLA's closed loop of "hypothesis → revision → rollout test" and the contemporaneous [RegenHarness](https://arxiv.org/abs/2609.27612v1)'s "fixed regression checks + accepting a modification only after release authorization" are two strengths of constraint on the same problem: the former relies on experimental testing and revision, the latter on gating to restrict revisions. RegenHarness also clearly distinguishes "model proposal / controller termination / verified completion" and points out that completion depends on execution history rather than on how close the agent is to the end, which bears directly on the stopping problem exposed by GPT-6-Astra.

**Limitations.** Only the simulation has numbers; the number of rollouts needed for adaptation and their cost are not reported; it is unclear whether harness revision overfits the test tasks. RegenHarness has only real-robot cases and no quantitative metrics.

**Recommendation.** Verify the definition of NaVILA-LH and the configuration corresponding to 57.5%; pay attention to the data structure of the revision graph, which can be used directly to organize a failure-case library for a navigation system.

## 4. Transferable methods
{: id="四可迁移方法"}

Only works whose transfer path can be explained clearly are included.

- **VLA memory: [SmoLSTM](https://arxiv.org/abs/2609.22854v1), [MemBodied](https://arxiv.org/abs/2609.28256v1)**
  - Mechanism: A fixed-size recurrent state replaces stacked history frames. SmoLSTM's full-task success on LIBERO-Mem is 77.5%, with sub-goal coverage of 85.1% and 0.04B trainable parameters; when the state is reset at every step it falls to 7.0%. MemBodied uses an associative state plus a first-frame scene anchor; on the five memory tasks of RMBench, its average success rate is 7.81 times that of a stateless policy and 2.98 times that of ordinary recurrent memory.
  - Integration point: VLN history encoding, replacing sliding windows or KV-cached history context.
  - Prerequisites and risks: Manipulation episodes are short and the memory demand is mostly occlusion and telling apart identical-looking objects; navigation needs long-range spatial memory, which must be verified separately.
- **VLA inference scheduling: [React When You Need To](https://arxiv.org/abs/2609.22587v1)**
  - Mechanism: Dynamically adjusts the inference interval according to scene change since the last inference, balancing action continuity and timely response.
  - Integration point: Triggering replanning during action-chunk execution, for example when a pedestrian suddenly enters the field of view.
  - Prerequisites and risks: The evidence is from manipulation scenarios. The average success rate under static and dynamic real-robot settings is 95%, 55 points above the strongest baseline.
- **Perception instrumentation: [Robo-Harness K1](https://arxiv.org/abs/2609.29389v1)**
  - Mechanism: Exposes perception as a tool the agent can query; tool-call trajectories can train the student model directly. Qwen3.5-9B uses only 107 teacher episodes, reaching 44.2% on new initial states (OpenVLA 30.2%) and 13.9% on the remaining task conditions (OpenVLA 0.0%).
  - Integration point: Perception interface design for navigation agents; isomorphic to the MCP navigation framework in this issue and AnchorVLN in the previous one.
  - Prerequisites and risks: All numbers come from manipulation tasks.
- **Harness trajectory distillation: [World Action Agent](https://arxiv.org/abs/2609.29964v1), [HarnessPAI](https://arxiv.org/abs/2609.29166v1)**
  - Mechanism: Fine-tunes a small model on the harness's successful execution trajectories. WAA reaches an average success rate of 75.6% on LIBERO-Pro, and after Qwen3.5-9B is fine-tuned on its trajectories, out-of-distribution success rises from 1.7% to 43.3%. Expert data collected by HarnessPAI with the Converged program lifts π0.5 by another 38.8 points on LIBERO-PRO.
  - Integration point: Distilling the successful trajectories of zero-shot VLN harnesses (such as GPT-6-Astra-like systems) into small deployable models.
  - Prerequisites and risks: The harness's own success rate must be high enough and the trajectory coverage diverse enough; what distillation yields is the harness's behavior, not stronger spatial ability.
- **Language annotation efficiency: [LADA](https://arxiv.org/abs/2609.27747v1)**
  - Mechanism: First learns a latent action codebook from unlabeled observations, then maps a small number of language instructions onto the codebook. With less than 5% language annotations, the Bench2Drive closed-loop driving score is 87.98 and success rate 70.46%, matching or exceeding the fully supervised baseline.
  - Integration point: VLN instruction data are scarce, so unlabeled navigation trajectories can first be used to learn a latent action codebook.
  - Prerequisites and risks: The driving action space differs greatly from indoor navigation; Bench2Drive is a closed-loop simulation.
- **Parallel hypotheses and evidence gathering: [WORLDS](https://arxiv.org/abs/2609.23841v1) (boundary entry, air)**
  - Mechanism: A persistent graph initialized with geographic priors, plus a parallel reasoner that keeps competing interpretations and actively requests evidence, a reviewer that processes observations, and a referee that decides whether to commit to a target or collect another round of evidence. On CityNav it reaches 51.8% SR on all 5,311 test episodes, 15.7 points above the published best (OSM-only, high-resolution orthophoto protocol); 50.0% vs. 27.9% on 1,000 shared episodes with the same model and budget; the reviewer contributes 5.9 points; there is a quadcopter real-robot demo.
  - Integration point: ObjectNav / instance navigation with high goal ambiguity: gather evidence first and then commit to a target, replacing "leave as soon as you see it".
  - Prerequisites and risks: City-scale aerial scenes rely on geographic prior maps; indoors on the ground there is no equivalent prior source.
- **Evaluation design: [RoboFollow](https://arxiv.org/abs/2609.25636v1)**
  - Mechanism: A "high scene entropy" principle: each training scene supports several kinematically different task branches, forcing the policy to rely on language; a four-level perturbation protocol separates understanding from execution.
  - Integration point: When evaluating your own VLN system, check whether there are episodes with only one reasonable route, where language is dispensable.
  - Prerequisites and risks: Evidence from manipulation tasks: performance of nine VLA/WAM models at L0 does not transfer reliably to L1–L3.

## 5. Research roundup by category
{: id="五分类速览"}

Relevance labels: A means directly relevant to ground navigation research, B means the mechanism is transferable, and C is for field observation only.

### 5.1 VLN and ObjectNav
{: id="51-vln-与-objectnav"}

- **GPT-6-Astra Lights Up** (zero-shot VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.29861v1)
- **How Far Can GPT-6-Astra Go?** (zero-shot VLN-CE · A): The renamed v2 of the v1 analyzed in the previous issue; not counted among this issue's new additions; see the key analysis. [Source](https://arxiv.org/abs/2609.20116v2)
- **SparseNav** (training-free VLN-CE · A): See key analysis. [Source](https://arxiv.org/abs/2609.26408v1)
- **Talk2Escape** (VLN with conversational error correction · A): See key analysis. [Source](https://arxiv.org/abs/2609.28296v1)
- **VNT-PA** (PointNav · A): See key analysis. [Source](https://arxiv.org/abs/2609.21212v1)
- **NaviScale** (semantic-map ObjectNav data · A): Stitches 24,000 floor plans from 12,794 properties with MP3D / HM3DSem room-level maps to generate 192,000 semantic maps for training a completion predictor; HM3D SR 64.3% / SPL 34.8%, MP3D SR 43.1% / SPL 16.8%, plus real-robot deployment. [Source](https://arxiv.org/abs/2609.27218v1)
- **Object-Path Graph** (open-vocabulary instance navigation · A): The Object-Path Graph unifies open-vocabulary semantic reasoning with topological navigation, using semantic visual servoing between nodes without dense metric reconstruction; validated on HM3D/Replica and a real robot, with no numbers in the abstract. [Source](https://arxiv.org/abs/2609.24189v1)
- **DPed-VLN** (VLN benchmark with dynamic pedestrians · A): 33,093 episodes with ORCA-controlled humanoid pedestrians and socially constrained expert paths; after LoRA adaptation, NaVILA and StreamVLN beat their zero-shot versions on several success and safety metrics, and the authors' own DPet-RL achieves the highest SR/SPL/STL. [Source](https://arxiv.org/abs/2609.21504v1)
- **VLN Interpretability** (interpretability and activation steering · A): The policy is sensitive to vision, instructions, and visual memory without relying on a single modality; navigation progress is encoded internally, and behavioral activation vectors transfer zero-shot to out-of-distribution real scenes and improve performance. [Source](https://arxiv.org/abs/2609.24576v1)
- **NaViRrator** (map to instruction · A): Start and end points on a human-readable map → route skeleton in map coordinates → VLM generates instructions → a pretrained VLN policy executes them; on a real robot, SR/SPL beats generating instructions directly and baselines such as an A* skeleton, with no numbers in the abstract. [Source](https://arxiv.org/abs/2609.21316v1)
- **Deploying FMs for Embodied Navigation** (foundation-model deployment · A): TAP uses human habit data mined from the scene for personalized goal search (+18% on average on a real Turtlebot); MemCtrl uses a "memory head" to manage context actively, +6% on average across tasks and +20% on the long-instruction subset, with context about half the baseline. [Source](https://arxiv.org/abs/2609.25666v1)
- **CoRelNav** (multi-robot relational semantic navigation · A): Task-conditioned multi-robot exploration is coupled with candidate-driven collaborative verification, and observations are aggregated across topological nodes to determine spatial relations; better than baselines in simulation, deployed on two real robots, with no numbers in the abstract. [Source](https://arxiv.org/abs/2609.27720v1)
- **"Dear LLaVA, Please Drive"** (VLM navigation fine-tuning · A): Uses a differentiable geometric cost field instead of annotated trajectories and learns collision-free paths from stereo depth alone; task-specific LoRA updates less than 1% of the parameters, and the SPL in the environment is described as "competitive" (no number given). [Source](https://arxiv.org/abs/2609.22925v1)
- **MCP Navigation Representation Layer** (LLM + ROS navigation · A): The occupancy grid is converted into a metric image with pose, waypoint-level observations are semantically annotated, and both are exposed as standard tools through MCP without changing the ROS navigation stack; indoor mapping coverage in simulation exceeds 97%. [Source](https://arxiv.org/abs/2609.27340v1)
- **ACE** (embodied exploration, borderline entry · A): Evidence-grounded perception is combined with exposure-aware movement to ease the tension between premature termination and excessive continuation; it is said to improve navigation task success by 18.0% over the previous SOTA and question-answering exploration efficiency by 10.3%. The abstract does not name the benchmark. [Source](https://arxiv.org/abs/2609.22385v1)
- **ReVNM** (visual navigation with a remote camera · B): A single surveillance camera serves as both the observation source and an implicit map. An exo2ego module predicts the depth in front of the robot from the remote view; it is trained only in randomly generated worlds and transfers to real robots without fine-tuning. [Source](https://arxiv.org/abs/2609.28976v1)

### 5.2 Embodied Agent, memory and planning
{: id="52-具身-agent记忆与规划"}

- **AdaHVLA** (adaptive harness · B): See key analysis. [Source](https://arxiv.org/abs/2609.29204v1)
- **RegenHarness** (robot agent harness · B): See item 5 of the featured work. [Source](https://arxiv.org/abs/2609.27612v1)
- **HarnessPAI** (physical AI harness · B): Uses code as an executable, evolvable interface; programs run open-loop within a round, and feedback revises the program and accumulates skills between rounds; LIBERO-PRO is 61.6 points above π0.5 and RoboCasa atomic tasks are 27.2 points above WorldDreamer; covers robot vacuums and legged platforms. [Source](https://arxiv.org/abs/2609.29166v1)
- **AquaMend** (belief-based failure recovery · B): Chooses among re-detection, rollback, and continuation by expected loss on a detection-belief-action graph; 28 of 32 self-built paired scenarios were recovered, with average total loss 21.6% lower than restarting. The difference from decision-theoretic troubleshooting was not significant after Holm correction. [Source](https://arxiv.org/abs/2609.28973v1)
- **RoboFollow** (instruction-following diagnosis · B): See Section 4. [Source](https://arxiv.org/abs/2609.25636v1)
- **OmniEcho** (spatial audio + navigation · B): OmniEchoBench contains 197 real spatial audio-video scenes, 2,972 question-answer pairs, and 900 first-order Ambisonics navigation samples from 30 real environments; sound-source-guided navigation is "close to the level of traditional VLN", with no number in the abstract. [Source](https://arxiv.org/abs/2609.23407v2)
- **Active exploratory manipulation** (agentic manipulation · B): Three modules (planning / perception / execution) plus fine-grained interleaving of perception and execution: when the target is initially invisible, search first and then manipulate; validated on a Find-and-Place task, with no number in the summary. [Source](https://arxiv.org/abs/2609.29091v1)
- **WORLDS** (city-scale language search, boundary entry · B): See Section 4. [Source](https://arxiv.org/abs/2609.23841v1)
- **AquaCap** (underwater code-as-policy agent · C): A two-layer agent converts instructions and observations into conditional plans and executable control programs. Failure-aware memory supports closed-loop replanning; simulation success is 66.43%, with real-ROV grasping and carrying. [Source](https://arxiv.org/abs/2609.23133v1)
- **CE⁴L** (multi-view continual learning benchmark · C): A four-task continual learning benchmark across ego / exo / ego-exo views, with a parameter-efficient subspace-routing adapter baseline. [Source](https://arxiv.org/abs/2609.23492v1)
- **PUBG Ally** (in-game conversational agent · C): A language-model agent reads game information through tools and drives a faster control layer; iteratively trained on nearly 39,000 real player game records; willingness to follow recommendations is 25.1 points higher for positive responses than for negative ones. [Source](https://arxiv.org/abs/2609.29837v1)
- **Listening and Mirroring** (VR empathic dialogue agent · C): In a 20-subject experiment, verbal attunement is the most reliable source of perceived empathy; it is a VR social agent and was misclassified as an embodied agent when assigning pools. [Source](https://arxiv.org/abs/2609.27246v1)

### 5.3 Social Navigation, Off-Road and Outdoor
{: id="53-社会导航越野与户外"}

- **PopNavShift** (social navigation evaluation · B): An LLM generates pedestrian motion parameters for 600 personality profiles, and three kinds of policies are compared under 8 crowd conditions and 7,488 paired runs; the rate of ranking reversals varies with the metric (see conclusion 5). [Source](https://arxiv.org/abs/2609.21838v1)
- **Diffusion-guided online adaptation** (social navigation · B): Keeps the diffusion policy fixed and trains only a noise policy (DSRL), preserving base performance while fine-tuning for the deployment environment; diffusion RL policies from multiple seeds are ensembled as the base policy; validated hardware-in-the-loop. [Source](https://arxiv.org/abs/2609.24317v1)
- **Where Should I Join?** (language-guided joining of a crowd · B): Recursive spectral partitioning generates candidate group subsets, a language-conditioned image-geometry model ranks them, and a crowd-formation prior then predicts socially compliant joining poses; sub-second inference, validated on a real robot. [Source](https://arxiv.org/abs/2609.28467v1)
- **AcousticDiffusion** (sound-source-guided rescue navigation · B): The microphone array's direction of arrival is fused recursively into a BEV confidence field, and a conditional diffusion model generates waypoint trajectories; on a real quadruped the average azimuth error is 64.9° (A* 98.2°, RRT 90.4°), and the final distance to the caller is 2.48 m (classical planning 3.96 m). [Source](https://arxiv.org/abs/2609.21792v1)
- **Verti-WM** (off-road world model · C): A frozen Transformer handles rigid terrain and neuro-symbolic terramechanics handles deformable terrain; prediction error is 34.6% / 21.7% lower than the pure-data / pure-physics baselines, training compute is reduced 23.6 times, and real-vehicle success is 80% (direct sim-to-real 40%). [Source](https://arxiv.org/abs/2609.23118v1)
- **TravPro** (off-road traversability ranking · C): Converts existing annotations into region preference pairs and distills preference scores onto prototypes of frozen VLM patch tokens; average pairwise accuracy across five unseen domains is 0.915 (strongest baseline 0.783). [Source](https://arxiv.org/abs/2609.23673v1)
- **Outdoor GNSS Navigation Stack** (ROS 2 outdoor navigation · C): The single GNSS-IMU / dual-antenna front end is swappable and several controllers share a unified state interface; across 800 field runs in a vineyard, the average lateral error in mixed in-row mode is 0.95 cm (single GNSS+IMU) / 0.85 cm (dual antenna). [Source](https://arxiv.org/abs/2609.28933v1)
- **Planning Trajectories that Bounce** (collision-tolerant planning · C): A reflection-augmented state graph divides paths into classes by the sequence of walls used, and controlled wall collisions can reduce execution time and control effort (simulation only). [Source](https://arxiv.org/abs/2609.27145v1)
- **BarrierFormer** (safety control · C): A Transformer autoregressively generates predictive rollouts in place of a model, and a barrier critic checks CBF constraints along the rollout; no online optimization is needed at inference. [Source](https://arxiv.org/abs/2609.23896v1)
- **ZIL** (image-to-point-cloud registration · C): A foundation model for zero-shot, asynchronous image-to-LiDAR registration; trained on 1.4 million frames from 7 datasets, translation / rotation error reduced by up to 87% / 76%. [Source](https://arxiv.org/abs/2609.22716v1)

### 5.4 Surface, underwater and UAVs
{: id="54-水面水下与无人机"}

- **RiverVLN** (unmanned surface vehicle VLN · B): A long-horizon USV VLN benchmark with continuous river motion; PGT-NAV converts instructions into a visually verifiable sequence of semantic stages and tracks the current stage online; average closed-loop success in Unity-ROS is 0.79, validated on a real boat. [Source](https://arxiv.org/abs/2609.23423v1)
- **AquaWorld** (underwater world generation · C): Consistent randomization over shared terrain structures; with the same budget, policy-training validation success is 21%, and a visual navigation policy trained in simulation succeeds 95% of the time in a physical water tank. [Source](https://arxiv.org/abs/2609.22670v1)
- **PhysAI-Bench** (UAV agent decision benchmark · C): 10,178 decision instances, including MCP tool calls, A2A interactions, and 6G network states; among 29 models, GPT-5.3 has the highest accuracy at 52.00%. [Source](https://arxiv.org/abs/2609.23695v1)

### 5.5 Secondary pool
{: id="55-次池速览"}

The secondary pool has 84 entries in total (81 on VLA/manipulation, with another 1, AdaHVLA, moved to 5.2 because of its navigation-side numbers; and 3 "other"), and is not analyzed in depth. Those with transfer value are included in Section 4. Grouped by theme:

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

None in this issue. The WeChat source has 0 entries in the database.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **The harness is becoming a data engine.** The harness works in the previous issue reported only overall system metrics; in this issue three works (HarnessPAI, World Action Agent, Robo-Harness K1) distill harness execution trajectories back into a small model, and all report significant gains. This digest's assessment: the same route is likely to appear on the navigation side: first use a closed-source frontier model plus a harness to build a zero-shot system with a high success rate, then distill its trajectories into a deployable model. GPT-6-Astra's 79.0%, if verified, sits upstream of this route.
- **Memory design is shifting from "how much to store" to "how much state to keep".** VNT-PA, SparseNav, and MemCtrl on the navigation side, and SmoLSTM, MemBodied, and StateMem on the VLA side, all replace a growing context with a fixed or bounded representation. SmoLSTM fell from 77.5% to 7.0% after its state was reset, while MemCtrl improved with half the context. Together the two sets of evidence show that the bottleneck is not context length but what to remember.
- **The axes of comparison for zero-shot VLN-CE are widening from "method" to "information conditions".** Whether external correction is used (Talk2Escape), which foundation model and reasoning level are used (GPT-6-Astra), and whether it is training-free and autonomous (SparseNav) already explain performance differences better than method details do. The problem in the previous issue was non-uniform episode subsets; in this issue it is non-uniform information conditions.
- **Non-visual signals are entering navigation.** In this issue OmniEcho (spatial-audio navigation samples), AcousticDiffusion (a microphone array guiding a quadruped), and ReVNM (a remote surveillance camera) all make navigation rely on signal sources other than onboard vision. None of the three yet connects to standard VLN benchmarks.

### Research gaps
{: id="研究空白"}

- **Zero-shot R2R-CE reporting norms lack information conditions.** Three kinds of misaligned variables, namely episode subset, external correction, and foundation model plus reasoning level, coexist on the same benchmark, and no paper reports all three at once.
- **Stop and completion decisions still have no dedicated metric.** The GPT-6-Astra authors admit that goal verification under ultra reasoning remains unreliable. RegenHarness argues that completion depends on execution history rather than distance to the end. The old paper from the previous issue had a 16.0% fallback success. All three point to the same link, yet no one in this issue reports stop-related metrics separately.
- **The cost of keeping people in the loop is not transparent.** Talk2Escape reports neither the frequency of questions nor a curve of "SR vs. number of interventions", so it is impossible to judge how much human effort 66.0% requires.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read closely and verify** GPT-6-Astra Lights Up: the episode set behind 79.0%, the SR and cost at each reasoning level, and whether success requires an active STOP; make no direct comparison before verification.
- **Read closely and reproduce** SparseNav: reproduce the instruction-conditioned perception trigger and measure the reduction in perception calls and latency.
- **Borrow Talk2Escape's spin/deviation detector**: use it to trigger your own replanning or backtracking instead of asking a human, and measure the net gain.
- **Update the tracking table**: besides "evaluation subset + number of episodes + zero-shot or not", add three columns to the R2R-CE register: "whether external correction is used", "foundation model and reasoning level", and "whether success requires an active STOP".

**Medium priority**

- **Read closely** VNT-PA: how the keyframe context is constructed, and whether it can replace the topological memory in your own system.
- **Verify** AdaHVLA: NaVILA-LH's task definition and the configuration corresponding to 57.5%.
- **Minimal verification**: try a fixed-size recurrent state (the SmoLSTM / MemBodied idea) in VLN history encoding and compare it with a sliding-window context.
- **Track** the DPed-VLN benchmark release and the LoRA adaptation results of NaVILA / StreamVLN under dynamic pedestrians.

**Low priority**

- **Track** whether the NaviScale dataset is released.
- **Defer** the quantization and inference-acceleration work for VLAs in the secondary pool: the mechanisms are worth borrowing, but none has been validated on VLN-CE.

> **Update (2026-10-01)**: GPT-6-Astra Lights Up released v2 on 2026-09-25. The 79.0% / 76.0% quoted in this report are single-run results from v1; v2 changes them to the average of three runs at each reasoning level, with ultra at 81.3±2.5% (SPL 71.5±1.7) and medium at 75.7±1.5% (SPL 65.6±2.1), still evaluated only on R2R-CE-100. See the [GPT-6-Astra reading in VLN Papers](/en/Agentic-Navigation-Papers/#gpt-6-astra) for details.
