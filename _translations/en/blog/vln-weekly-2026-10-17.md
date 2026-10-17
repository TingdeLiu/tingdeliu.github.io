---
layout: post
lang: en
translation_id: vln-weekly-2026-10-17
permalink: /en/vln-weekly-2026-10-17/
source_path: _posts/weekly-reports/2026-10-17-VLN-Weekly.md
source_url: /vln-weekly-2026-10-17/
source_revision_date: 2026-10-17
translation_updated: 2026-10-17
title: "Embodied Navigation Weekly (2026-10-08 to 2026-10-15)"
date: 2026-10-17
period_start: 2026-10-08
period_end: 2026-10-15
issue_number: 9
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "This issue adds 175 independent works, with 64 in the automatic primary pool and 5 involving R2R-CE. RSI-Nav reports SR of 68.6%; VLN-Safe exposes collisions in successful episodes, while I2O2I-Nav full-mission success remains 4.02%. The focus is evidence verification, stage handoffs, and failure-guided improvement."
---

## 1. Key conclusions
{: id="key-conclusions"}

- **Continuous VLN gains must be considered alongside collisions.** Of 175 new works, 5 explicitly involve R2R-CE; 4 abstracts provide related numbers, while VLN-AIEP only names the benchmark. [RSI-Nav](https://arxiv.org/abs/2610.15782v1) reports R2R-CE SR of 68.6%; [VLN-Safe](https://arxiv.org/abs/2610.14682v1) finds contact in 38.5–56.8% of successful episodes when sliding after collisions is allowed. Our assessment: report success and safety separately, and distinguish continuous R2R-CE from discrete R2R.
- **Memory must maintain trustworthy state and govern execution.** [MemoryVLN](https://arxiv.org/abs/2610.18478v1) studies spatial grounding and cross-context reuse; [S3-Nav](https://arxiv.org/abs/2610.19431v1) separates persistent state, decision-facing evidence, and execution authority; [AVALON](https://arxiv.org/abs/2610.21586v1) distinguishes human permissions from robot-observed traversability. Our assessment: test evidence withdrawal and completion verification, beyond memory capacity.
- **Long-horizon failures emerge at stage handoffs.** [I2O2I-Nav](https://arxiv.org/abs/2610.18307v1) raises full-mission success from 1.15% to 4.02%, still far below primitive performance; [MultINav](https://arxiv.org/abs/2610.17089v1) incorporates connectivity and visibility changes caused by opening doors and containers. Our assessment: starting each stage at its predecessor's actual endpoint exposes difficulties hidden by isolated subtask evaluation.
- **Self-improvement increasingly targets diagnosed failures.** [WEDGE](https://arxiv.org/abs/2610.19420v1) generates environments around policy weaknesses; [EnactOpt](https://arxiv.org/abs/2610.14977v1) jointly revises context construction and textual skills; [AdaRSI](https://arxiv.org/abs/2610.20474v1) diagnoses invocation before fine-tuning persistently unreliable substeps. Our assessment: navigation research should compare the costs and benefits of correcting evidence, scheduling, and policies; manipulation gains do not establish navigation gains.

This issue adds 175 records corresponding to 175 independent arXiv works, published between 2026-10-08 and 2026-10-15. Another 5 existing papers received version updates and are excluded from the new-work count. There are 0 new WeChat articles. Automatic primary-pool labels comprise 45 navigation and 19 embodied-agent works, totaling 64; the secondary pool comprises 90 VLA/manipulation, 7 autonomous-driving, and 14 other works, totaling 111. Labels have boundary errors: StereoEngine is classified as autonomous driving, Jev-Guard as embodied agents, and ASD and VP-EVT as other. Detailed analyses select directly relevant works from the primary pool; the categorized index follows actual content and covers all 175 works. Numbers below come from archived abstracts; missing splits, comparisons, and absolute scores are not inferred.

## 2. Priority reading list
{: id="priority-reading"}

1. **[VLN-Safe](https://arxiv.org/abs/2610.14682v1)** · Collision evaluation on continuous R2R-CE
   - Contribution: compares sliding and blocking collision handling and adds an independent collision guard.
   - Evidence: switching to blocking reduces SR by 11.7–36.1 percentage points across 6 published agents.
   - Why read: changes how existing navigation results and real-world transfer are interpreted.
2. **[GuidedNav](https://arxiv.org/abs/2610.13122v1)** · Representation supervision for continuous R2R-CE / RxR-CE
   - Contribution: training-time supervision of active subtasks, landmark attention, and spatial representations.
   - Evidence: the abstract states SR improvements of 8.1% / 13.1% on the respective val-unseen splits, without absolute SR; these are not reinterpreted as percentage points.
   - Why read: strengthens instruction–action grounding while limiting online reasoning overhead.
3. **[RSI-Nav](https://arxiv.org/abs/2610.15782v1)** · A language–action data loop for continuous navigation
   - Contribution: generates instructions, filters execution, and immediately updates the same model in shards.
   - Evidence: R2R-CE SR 68.6%, RxR-CE SR 72.4%; splits are unspecified in the abstract.
   - Why read: tests whether synthetic instructions improve in step with the current policy.
4. **[MemoryVLN](https://arxiv.org/abs/2610.18478v1)** · Long-term memory for continuous navigation
   - Contribution: temporal compression, spatial anchoring, and confidence-controlled write-back.
   - Evidence: the abstract reports R2R-CE / RxR-CE SR improvements of 2% / 6.5%, without absolute scores, splits, or clarification of relative versus absolute gains.
   - Why read: separate the contributions of history compression and cross-context reuse.
5. **[VLN-AIEP](https://arxiv.org/abs/2610.13104v1)** · Instruction-execution progress
   - Contribution: learns progress from navigation decision labels and suppresses executed instruction content.
   - Evidence: evaluates discrete R2R, RxR, CVDN, and continuous R2R-CE; the abstract provides no verifiable metric values.
   - Why read: contrasts with representation guidance requiring additional progress annotations.
6. **[I2O2I-Nav](https://arxiv.org/abs/2610.18307v1)** · Continuous indoor–outdoor–indoor missions
   - Contribution: evaluates full missions from actual handoff states and localizes stage failures.
   - Evidence: full-mission SR 1.15% → 4.02%, using the same underlying policies.
   - Why read: stress-tests runtime handoffs and completion verification.
7. **[S3-Nav](https://arxiv.org/abs/2610.19431v1)** · Separating evidence from execution authority
   - Contribution: bounds decision context while retaining persistent spatial-semantic evidence and independent execution validation.
   - Evidence: the abstract names neither environments nor verifiable numbers.
   - Why read: relevant to navigation runtimes, but effectiveness requires full-paper verification.
8. **[WEDGE](https://arxiv.org/abs/2610.19420v1)** · Failure-guided environment generation
   - Contribution: validates weaknesses before changing the environmental factors that control them.
   - Evidence: compares random and hard-example alternatives on ProcTHOR and reports HM3D transfer for some policies; the abstract gives no numbers.
   - Why read: supports targeted ablations beyond merely increasing environment count.

## 3. Detailed analyses
{: id="deep-analysis"}

### 1. VLN-Safe: establish how collisions change actions
{: id="vln-safe"}

**Problem.** Reaching the goal does not imply collision-free motion. Sliding along obstacles can conceal unsafe actions' consequences.

**Method.** The [paper](https://arxiv.org/abs/2610.14682v1) contrasts sliding and blocking on the same navigation task, using depth geometry and recent motion to detect risk. A guard vetoes risky forward actions, escapes, and returns control.

**Evidence.** Under R2R-CE blocking, the guard reduces collision steps by 62–98% and raises SR by 6.9–29.3 percentage points across 5 evaluated agents. These are cross-agent ranges, not one model's score.

**Value.** Our assessment: prioritize replication because it tests reliance on sliding without first retraining the policy.

**Limitations.** Simulated collisions do not measure physical contact damage; the abstract does not establish guard reliability under sensor errors.

**Action.** Fix policies and episodes, compare both collision regimes, and record collision-free success, interventions, and added distance.

### 2. GuidedNav: keep progress and geometry supervision in training
{: id="guidednav"}

**Problem.** Action labels alone may teach correlations without reliably grounding the current instruction segment and its landmarks.

**Method.** The [paper](https://arxiv.org/abs/2610.13122v1) guides pre-action attention and constrains representations with geometry, introducing Guided-R2R and Guided-RxR annotations.

**Evidence.** The abstract reports improvements over compared monocular RGB methods and faster inference, but no absolute SR, latency, hardware, or training cost.

**Value.** Our assessment: test whether structural supervision can replace extra deployment-time textual reasoning and geometric encoding.

**Limitations.** Additional annotations have costs; the abstract cannot disentangle annotation scale, supervision design, and model configuration.

**Action.** Fix backbone and data volume, ablate subtask, landmark, and spatial supervision separately, and measure closed-loop latency.

### 3. RSI-Nav: keep the data generator aligned with the policy
{: id="rsi-nav"}

**Problem.** Labeling all trajectories with a fixed model can leave generated instructions misaligned with an evolving navigator.

**Method.** The [paper](https://arxiv.org/abs/2610.15782v1) shares instruction generation and following within one model: generate language in shards, execute, filter by completion and path efficiency, and update immediately.

**Evidence.** Both continuous VLN benchmarks have explicit SR values, but the abstract omits splits, confidence intervals, and paired gains over a fixed generator.

**Value.** Our assessment: replicate synchronized data–policy updates rather than focusing only on synthetic corpus size.

**Limitations.** Self-generation and filtering may favor already manageable routes and omit difficult instructions. This is a risk to test, not an established paper finding.

**Action.** Fix the trajectory budget, compare one-pass annotation with shard-wise updates, and track long instructions, difficult routes, and instruction diversity.

### 4. MemoryVLN: compressed history still needs spatial provenance
{: id="memoryvln"}

**Problem.** Long histories face context limits, redundant observations, and unstable spatial alignment over time.

**Method.** The [paper](https://arxiv.org/abs/2610.18478v1) samples recent history densely and distant history sparsely, associating semantics, geometry, and pose. Hierarchical memory supports cross-context warm starts, selective retrieval, and confidence-controlled write-back.

**Evidence.** The abstract reports gains on both continuous VLN benchmarks, but omits retrieval and geometry-encoding costs and experimental isolation of cross-context memory.

**Value.** Our assessment: evidence of what was seen where better supports revisitation and repeated search than location-free summaries.

**Limitations.** Check boundaries between test environments, trajectories, and persistent memory before interpreting prior scene exposure as unseen-environment generalization.

**Action.** Separate within-episode and cross-task memory; fix the token budget and compare compression, spatial anchoring, and write-back.

### 5. I2O2I-Nav: primitive competence does not ensure full missions
{: id="i2o2i-nav"}

**Problem.** Isolated stage evaluation misses accumulated endpoint errors, handoff failures, and incorrect completion decisions.

**Method.** The [paper](https://arxiv.org/abs/2610.18307v1) builds physically executable composite scenes and starts each stage at its predecessor's actual endpoint. A modular agent adds completion verification, handoffs, and bounded recovery.

**Evidence.** The benchmark includes 20 composite scenes and 1,374 evaluation episodes. Primitive SR is 40.5–76%, but full-mission SR remains only 4.02%.

**Value.** Our assessment: a more deployment-relevant diagnostic target for ground delivery robots than isolated indoor or outdoor arrival.

**Limitations.** Benchmark results are not real delivery success rates; low full-mission performance does not establish reliable deployment.

**Action.** Record stage-conditional success, handoff states, and false stops, especially at the final ObjectNav stage.

## 4. Transferable methods
{: id="transferable-methods"}

- **Manipulation memory: [REMM](https://arxiv.org/abs/2610.12710v1) / [GAMMA](https://arxiv.org/abs/2610.12793v1).** The former withdraws task conclusions whose supporting evidence changes; the latter audits memory writes and subgoals. Apply to navigation memory and completion checks, requiring object identity, locations, and evidence dependencies. Manipulation gains do not transfer directly.
- **Manipulation scheduling: [When to Ask for Help](https://arxiv.org/abs/2610.21160v1).** Separates trigger quality from slow-module intervention value. Apply to cloud–edge navigation budgets, including latency and harmful interventions rather than judging system benefit from warning accuracy alone.
- **Language evaluation: [Language2Drive](https://arxiv.org/abs/2610.16634v1).** Fixes the starting scene and changes only feasible instructions to test behavioral dependence on language. Transfer to turns, detours, and ordering, after establishing that both instructions are executable.
- **Aerial VLN: [X-Fly](https://arxiv.org/abs/2610.21450v1).** Learns recovery and stopping from policy-visited states. Applicable to ground action chunks, but OpenFly results do not establish R2R-CE performance; dynamics and stopping regions require redesign.
- **Memory evaluation: [MeMoREval](https://arxiv.org/abs/2610.12561v1).** Accumulates experience before future requests are known, then measures memory utility and writing/reading costs. Useful for comparing text, frame, and scene-graph navigation memory; its requests without new observations differ from online navigation.

## 5. Categorized index
{: id="categorized-index"}

A denotes direct ground-navigation relevance; B denotes an explicit transferable mechanism; C denotes field observation or low relevance. These reading categories do not replace automatic library labels. Detailed analyses are not repeated; secondary works are grouped by theme.

### 5.1 Ground VLN / ObjectNav / semantic navigation
{: id="ground-navigation"}

- **[Two-Stage Multi-modal Learning for Dynamic Audio-Visual Navigation](https://arxiv.org/abs/2610.14077v1)** (A): Dynamic audio-visual navigation aligns sound intensity with geometry for unheard sounds.
- **[Visual-Wi-Fi Navigation to Signal-Emitting Devices](https://arxiv.org/abs/2610.12733v1)** (A): ViFiNav combines motion-dependent Wi-Fi bearings with depth to locate small or occluded devices.
- **[GuidedNav](https://arxiv.org/abs/2610.13122v1)** (A): Training-time landmark, progress, and spatial guidance; see detailed analysis.
- **[VLN-AIEP](https://arxiv.org/abs/2610.13104v1)** (A): VLN-AIEP anchors trajectories to landmarks and suppresses already executed instruction segments.
- **[VLN-Safe](https://arxiv.org/abs/2610.14682v1)** (A): Collision handling changes the interpretation of success; see detailed analysis.
- **[Training-Free Multi-Agent Cooperative Embodied Search from Uncertain Vision-Language Priors](https://arxiv.org/abs/2610.15038v1)** (A): Treats language-model priors as hypotheses and jointly plans multi-robot search through graph optimization.
- **[VbotNav-0](https://arxiv.org/abs/2610.14634v1)** (A): Asynchronous cloud subgoals and edge trajectory policies connect indoor and outdoor routes.
- **[ADAPT](https://arxiv.org/abs/2610.15956v1)** (A): DarkNav-Bench / ADAPT uses spatial audio and depth for navigation without visible illumination.
- **[RSI-Nav](https://arxiv.org/abs/2610.15782v1)** (A): RSI-Nav updates instruction generation and navigation together in shards; see detailed analysis.
- **[FireNav](https://arxiv.org/abs/2610.15595v1)** (A): Routes reconstruction evidence by navigation utility under fire-related visual degradation.
- **[RefBeliefNav](https://arxiv.org/abs/2610.15375v1)** (A): RefBeliefNav separates target belief from evidence uncertainty and unifies exploration with revisitation.
- **[LERR-Nav](https://arxiv.org/abs/2610.15362v1)** (A): Learns when to explore, recover, and revisit historical cues.
- **[Estimating the Causal Impact of Social Navigation Signals in Corridor Scenarios](https://arxiv.org/abs/2610.16411v1)** (A): Observational associations of social-navigation signals can oppose their intervention effects.
- **[G2WAM](https://arxiv.org/abs/2610.16558v1)** (A): G2WAM anchors history and goal images in shared coordinates and jointly predicts latent states and actions.
- **[MultINav](https://arxiv.org/abs/2610.17089v1)** (A): MultINav models interaction-driven changes in connectivity or target visibility and verifies graph updates.
- **[PathLift](https://arxiv.org/abs/2610.16103v1)** (A): Uses sensor depth to correct unitless navigation trajectories waypoint by waypoint.
- **[CoSMO](https://arxiv.org/abs/2610.18321v1)** (A): Uses calibrated category sets for semantic safety clearance; coverage guarantees do not imply zero collisions.
- **[DR-VLA](https://arxiv.org/abs/2610.18373v1)** (A): Transfers human egocentric video to quadruped navigation; abstract evidence is mainly pseudo-closed-loop and open-loop.
- **[I2O2I-Nav](https://arxiv.org/abs/2610.18307v1)** (A): Evaluates actual indoor–outdoor stage handoffs; see detailed analysis.
- **[LynxScenes](https://arxiv.org/abs/2610.18093v1)** (A): Outdoor wheel-legged multisensor data with pseudo-closed-loop evaluation; data and tools are planned for release.
- **[MemoryVLN](https://arxiv.org/abs/2610.18478v1)** (A): Spatial anchoring and cross-context memory reuse; see detailed analysis.
- **[SkipWorld](https://arxiv.org/abs/2610.18135v1)** (A): Filters trajectories with latent skip predictions, then reranks them with an autoregressive model.
- **[ASD](https://arxiv.org/abs/2610.17593v1)** (A): ASD distills language goals and navigation reasoning from privileged training-time trajectory feedback.
- **[Cracking Mobile Manipulation](https://arxiv.org/abs/2610.19600v1)** (A): Diagnoses mobile manipulation through planning, base positioning, and local manipulation before targeted intervention.
- **[F2A-ObjectNav](https://arxiv.org/abs/2610.19618v1)** (A): Event-driven target validation, candidate management, recovery, and strategy switching.
- **[WEDGE](https://arxiv.org/abs/2610.19420v1)** (A): WEDGE validates policy weaknesses and generates targeted training environments.
- **[HM3D-M3](https://arxiv.org/abs/2610.20121v1)** (A): HM3D-M3 combines multiple robots, goal modalities, and object goals within one episode.
- **[MoCaP](https://arxiv.org/abs/2610.20998v1)** (A): MoCaP extracts spatial memory and priors for manipulation-ready base configurations from human video.
- **[S3-Nav](https://arxiv.org/abs/2610.19431v1)** (A): Separates persistent state, decision evidence, and execution authority; effectiveness needs full-paper checks.
- **[VLX-Go](https://arxiv.org/abs/2610.19361v1)** (A): Uses staged learning and policy-visited-state refinement for following amid distractors.
- **[AVALON](https://arxiv.org/abs/2610.21586v1)** (A): Seeds topological memory from user floor plans and place clues, with updates verified in situ.
- **[MuSTMM](https://arxiv.org/abs/2610.21429v1)** (A): Cross-scene sound-triggered mobile manipulation uses pose-aligned depth as navigation memory.
- **[Training-Free Target-Aware Refinement of 3D Gaussian Language Map](https://arxiv.org/abs/2610.21619v1)** (A): Minimally injects target priors into existing language maps while suppressing query-independent bias.
- **[VP-EVT](https://arxiv.org/abs/2610.21999v1)** (A): Makes target visibility a motion objective to actively resolve persistent occlusion.

### 5.2 Memory, maps, planning, and evaluation
{: id="memory-planning"}

- **[HorizonNav](https://arxiv.org/abs/2610.13849v1)** (B): Reconstructs plannable space from generated video; imagined feasibility does not establish real traversability.
- **[EnactOpt](https://arxiv.org/abs/2610.14977v1)** (B): Jointly evolves context-construction code and textual skills with fixed models and runtime.
- **[PVFH-Bench](https://arxiv.org/abs/2610.14635v1)** (B): Diagnoses visual hallucinations induced by false textual premises; relevant to absent-target navigation tests.
- **[StateSkill-Agent](https://arxiv.org/abs/2610.14601v1)** (B): StateSkill organizes skills and stage-relevant context as finite-state machines.
- **[Observe, Share, or Predict?](https://arxiv.org/abs/2610.16895v1)** (B): PARTNR compares observation, knowledge sharing, and action prediction; more prediction need not improve cooperation.
- **[Code2Worlds-2](https://arxiv.org/abs/2610.17651v1)** (B): Compiles physical properties from visual scenes, then simulates, diagnoses, and refines executable environments.
- **[MemAct](https://arxiv.org/abs/2610.17640v1)** (B): MemAct combines memory writing, historical retrieval, and active viewpoint acquisition in one loop.
- **[SEAR](https://arxiv.org/abs/2610.18792v1)** (B): Separates retained-experience gains from retries, budget, and task familiarity.
- **[Towards a Data-Centric Understanding of Spatial Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2610.19108v1)** (B): Surveys spatial-reasoning data construction and evaluation, emphasizing supervision and metric definitions.
- **[SCOPER](https://arxiv.org/abs/2610.17657v1)** (B): SCOPER jointly examines evidence memory, execution progress, and failure recovery.
- **[AdaRSI](https://arxiv.org/abs/2610.20474v1)** (B): Repairs skill invocation before fine-tuning the executor on persistently failing substeps.
- **[Dungeon and Visibility World Generator](https://arxiv.org/abs/2610.19523v1)** (B): Generates controllable geometry and visibility for exploration evaluation.
- **[C-Slerp](https://arxiv.org/abs/2610.20409v1)** (B): C-Slerp composes intermediate goals in spherical representations and adapts waypoint count to difficulty.
- **[PhySGA](https://arxiv.org/abs/2610.20560v1)** (B): Verifies scene composition with physics feedback, informing executability checks in environment generation.
- **[HiMeco](https://arxiv.org/abs/2610.20312v1)** (B): HiMeco separates perception, exploration, and planning failures before retaining execution-validated skills.
- **[Towards Agentic Robots](https://arxiv.org/abs/2610.20442v1)** (B): Surveys agentic robotics across models, data, environments, and harnesses.
- **[ACT3D](https://arxiv.org/abs/2610.21783v1)** (B): ACT3D learns viewpoint selection from answer outcomes; spatial-QA accuracy is not navigation success.
- **[GAMMA](https://arxiv.org/abs/2610.12793v1)** (B): GAMMA separates memory writing and reasoning, auditing claims against mechanical visual evidence.
- **[MeMoREval](https://arxiv.org/abs/2610.12561v1)** (B): MeMoREval accumulates experience before future tasks are known, comparing memory modalities and inference costs.
- **[EmbodiedCLBench](https://arxiv.org/abs/2610.13021v1)** (B): EmbodiedCLBench tests whether basic-task experience supports unseen skill compositions.
- **[JoyInside-VLM-Agent](https://arxiv.org/abs/2610.12957v1)** (B): Product-conditioned multimodal policy training includes goal revisions and execution feedback.
- **[DySCOVER](https://arxiv.org/abs/2610.14302v1)** (B): DySCOVER uses prediction error to guide active collection of robot visual-learning experience.
- **[MAGMA-Bench](https://arxiv.org/abs/2610.13534v1)** (B): MAGMA-Bench evaluates interruptions, execution failures, and heterogeneous robot coordination at a skill interface.
- **[MPT](https://arxiv.org/abs/2610.22322v1)** (B): MPT jointly learns bidding-based participation and action policies for decentralized coordination.

### 5.3 Embodied VLA / mobile manipulation
{: id="vla-manipulation"}

- **Manipulation memory, skills, and runtimes (C)**：[Embodied Zero-Shot Task Planning via Hardware-Aware LLM-VLM Integration for Grounded Robotic Manipulation](https://arxiv.org/abs/2610.14106v1); [REMM](https://arxiv.org/abs/2610.12710v1); [Dual-Stream Memory for Vision-Language-Action Policies](https://arxiv.org/abs/2610.13262v1); [Memory Without Curation](https://arxiv.org/abs/2610.14406v1); [A Persian Vision-Language-Action Dataset and Benchmark for Ambiguity- and Infeasibility-Aware Instruction-to-Plan Grounding](https://arxiv.org/abs/2610.15666v1); [Exploring the Limits of Combinatorial Generalization in Embodied Manipulation](https://arxiv.org/abs/2610.15425v1).
- **Manipulation memory, skills, and runtimes (C)**：[SurgiCLAW](https://arxiv.org/abs/2610.15766v1); [Can a Frontier VLM Serve as a Robot Policy for Memory-Dependent Manipulation?](https://arxiv.org/abs/2610.17350v1); [ChunkAct](https://arxiv.org/abs/2610.18874v1); [EqC](https://arxiv.org/abs/2610.18790v1); [ForkCast](https://arxiv.org/abs/2610.18650v1); [Let the Mind Conceive, Let the Hand Achieve](https://arxiv.org/abs/2610.18352v1).
- **Manipulation memory, skills, and runtimes (C)**：[ACRO](https://arxiv.org/abs/2610.19677v1); [Back on Track](https://arxiv.org/abs/2610.19579v1); [Robota-Embed](https://arxiv.org/abs/2610.19495v1); [VLRLA](https://arxiv.org/abs/2610.19517v1); [CarryOn](https://arxiv.org/abs/2610.21453v1); [Memory as Weights](https://arxiv.org/abs/2610.22070v1).
- **Manipulation memory, skills, and runtimes (C)**：[REKI](https://arxiv.org/abs/2610.22178v1).
- **Manipulation evaluation, instructions, and preferences (C)**：[Spa-Bench](https://arxiv.org/abs/2610.12696v1); [JitterVLA](https://arxiv.org/abs/2610.13628v1); [Red-Teaming of Vision-Language-Action Models via Quality Diversity Environment Generation](https://arxiv.org/abs/2610.14415v1); [LIBERO-Industry](https://arxiv.org/abs/2610.16782v2); [VLA-GAUGE](https://arxiv.org/abs/2610.16567v1); [LIBERO-RIG](https://arxiv.org/abs/2610.18843v1).
- **Manipulation evaluation, instructions, and preferences (C)**：[What Matters in Quality-Aware VLA Pretraining](https://arxiv.org/abs/2610.18869v1); [Paraphrase Consensus Is Not Self-Consistency](https://arxiv.org/abs/2610.19438v1); [Do VLAs Really Follow Instructions? Diagnosing How-Level Instruction Following in Vision-Language-Action Models](https://arxiv.org/abs/2610.21251v1); [LIBERO-PeRM](https://arxiv.org/abs/2610.22372v1).
- **Manipulation data, transfer, and policy optimization (C)**：[StereoEngine](https://arxiv.org/abs/2610.12734v1); [Axiom-0](https://arxiv.org/abs/2610.14155v1); [Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies](https://arxiv.org/abs/2610.12861v1); [Noise Space Learning for VLA Adaptation](https://arxiv.org/abs/2610.13943v1); [OGPO](https://arxiv.org/abs/2610.12914v1); [RARA](https://arxiv.org/abs/2610.13578v1).
- **Manipulation data, transfer, and policy optimization (C)**：[Splat2Act](https://arxiv.org/abs/2610.13945v1); [VLMs are Native Action Models](https://arxiv.org/abs/2610.12832v1); [CFG-Policy](https://arxiv.org/abs/2610.14936v1); [Flow-Support-Weighted Proposals for Test-Time Verified VLA Policies](https://arxiv.org/abs/2610.14588v1); [Verify, Act, Distill](https://arxiv.org/abs/2610.14885v1); [CoLA](https://arxiv.org/abs/2610.15337v1).
- **Manipulation data, transfer, and policy optimization (C)**：[Trust-Calibrated VLA Policy Post-Training with Evolving Imagination](https://arxiv.org/abs/2610.16517v1); [UniLAP](https://arxiv.org/abs/2610.16969v1); [Breaking the Average](https://arxiv.org/abs/2610.17983v1); [Post-Training, Finetuning, and Steering](https://arxiv.org/abs/2610.19143v1); [Graph-Reweighted Behavior Cloning with Human Interventions for Precise Robotic Manipulation](https://arxiv.org/abs/2610.20504v1); [Human Demonstrations as Post-Training Data for Any Robot Body](https://arxiv.org/abs/2610.20351v1).
- **Manipulation data, transfer, and policy optimization (C)**：[Multi-Dimensional Featurizers Yield Fine-Grained Steering and Data Shaping in Robot Foundation Models](https://arxiv.org/abs/2610.20767v1); [One-shot Human Video-Guided Adaptation for Vision-Language-Action Models](https://arxiv.org/abs/2610.19746v1); [Multi-Teacher On-Policy Distillation for Flow-Based Vision-Language-Action Models](https://arxiv.org/abs/2610.21522v1).
- **Future prediction and action representations (C)**：[Beyond Actions](https://arxiv.org/abs/2610.13212v2); [Evo-World](https://arxiv.org/abs/2610.12936v1); [Forward Dynamics Models](https://arxiv.org/abs/2610.13117v1); [Vision-Predictor-Action](https://arxiv.org/abs/2610.15556v1); [Delta Flow Matching](https://arxiv.org/abs/2610.15794v1); [FlowVLA](https://arxiv.org/abs/2610.16160v1).
- **Future prediction and action representations (C)**：[GreenWAM](https://arxiv.org/abs/2610.18774v1); [WAVE-0](https://arxiv.org/abs/2610.18568v1); [WoVR+](https://arxiv.org/abs/2610.17922v1); [A Phenomenology of Neural Geometry in Vision-Language-Action Models](https://arxiv.org/abs/2610.20840v1); [Coordi-pi](https://arxiv.org/abs/2610.19753v1); [ObjDream](https://arxiv.org/abs/2610.20223v1).
- **Future prediction and action representations (C)**：[What Matters for World Modeling in Vision-Language-Action Models?](https://arxiv.org/abs/2610.19825v1); [TS-WAM](https://arxiv.org/abs/2610.21922v1); [VLA-Delta](https://arxiv.org/abs/2610.21306v1); [What Should Robot Policies Learn to Predict? Rethinking Supervision Targets in Vision-Language-Action Learning](https://arxiv.org/abs/2610.21915v1).
- **Precision manipulation, multimodality, and body control (C)**：[DEFT-VLA](https://arxiv.org/abs/2610.14854v1); [Learning Dynamic Grasping](https://arxiv.org/abs/2610.14848v1); [PRISM](https://arxiv.org/abs/2610.14704v1); [Recovering Low-Bit Failures in Vision-Language-Action Policies via Quantization-Conditioned Reinforcement Learning](https://arxiv.org/abs/2610.15675v1); [Listen Before Act](https://arxiv.org/abs/2610.16313v1); [Interaction Latents as Actions for Contact-Rich Manipulation](https://arxiv.org/abs/2610.18855v2).
- **Precision manipulation, multimodality, and body control (C)**：[TacitVLA](https://arxiv.org/abs/2610.17540v1); [TRACE](https://arxiv.org/abs/2610.17999v1); [Causal Analysis of Robot Foundation Models Enables Efficiency Gains](https://arxiv.org/abs/2610.20750v1); [ForceBench](https://arxiv.org/abs/2610.19604v1); [FoveaVLA](https://arxiv.org/abs/2610.20064v1); [SPECTRA-VLA](https://arxiv.org/abs/2610.19651v1).
- **Precision manipulation, multimodality, and body control (C)**：[KODA](https://arxiv.org/abs/2610.21152v1).
- **Inference compression, action chunks, and verification (C)**：[Accelerating VLA Inference with Perturbation Informed Execution](https://arxiv.org/abs/2610.19656v2); [Fast-VLA — You Only Think Once](https://arxiv.org/abs/2610.21389v1); [FastVLA — Adaptive Visual Token Compression](https://arxiv.org/abs/2610.22146v1); [Policy-Aligned Predictive Verification for Adaptive VLA Action-Chunk Execution](https://arxiv.org/abs/2610.21434v1); [VeriVLA](https://arxiv.org/abs/2610.21985v1); [When to Ask for Help](https://arxiv.org/abs/2610.21160v1).

### 5.4 Aerial, driving, and lower-relevance directions
{: id="lower-relevance"}

- **Aerial navigation and interaction (C)**：[Verifier-Mediated Multi-LLM Coordination for Multi-Robot Path Planning](https://arxiv.org/abs/2610.12761v1); [BGC-Nav](https://arxiv.org/abs/2610.14702v1); [ClearNav](https://arxiv.org/abs/2610.17419v1); [Benchmarking Interaction-Oriented Aerial Mobility across Heterogeneous Environments](https://arxiv.org/abs/2610.18381v1); [Towards bio-inspired visual navigation](https://arxiv.org/abs/2610.21978v1); [X-Fly](https://arxiv.org/abs/2610.21450v1).
- **Geometric control, path planning, and estimation (C)**：[CeDR-Nav](https://arxiv.org/abs/2610.13007v1); [TAME-Nav](https://arxiv.org/abs/2610.12877v1); [Load-Aware Quadruped Navigation with Online Contact-based Geometry Estimation](https://arxiv.org/abs/2610.14596v1); [Learning Reachability-Based Safety Representations for Real-Time Robot Navigation in Unknown and Dynamic Environments](https://arxiv.org/abs/2610.16522v1); [LiDAR-based Humanoid Navigation in Dynamic, Non-Convex Environments via Spatio-Temporal Safe Corridors](https://arxiv.org/abs/2610.17232v1); [End-to-End Vision-Language-Action Policies for Autonomous Lunar Rover Docking](https://arxiv.org/abs/2610.19224v1).
- **Geometric control, path planning, and estimation (C)**：[Reinforcement Learning for Large-Scale Lifelong Multi-Agent Path Finding with Shared Spatial Fields](https://arxiv.org/abs/2610.18186v1); [RingNav](https://arxiv.org/abs/2610.19044v1); [RoPE-Flow](https://arxiv.org/abs/2610.21086v1); [Vision-Guided Closed-Loop Autonomous Navigation of Light-Driven Janus Microbots](https://arxiv.org/abs/2610.21222v1).
- **Driving and motion-based visual adaptation (C)**：[Reflect-VLA](https://arxiv.org/abs/2610.13453v1); [YOUDrive](https://arxiv.org/abs/2610.14802v1); [Mobility Helps Learning](https://arxiv.org/abs/2610.15846v1); [AUGUstus](https://arxiv.org/abs/2610.16985v1); [Language2Drive](https://arxiv.org/abs/2610.16634v1); [SafeDriveVLA](https://arxiv.org/abs/2610.20609v1).
- **General agents, non-navigation work, and retrieval noise (C)**：[A4-MR](https://arxiv.org/abs/2610.13056v1); [MemForm](https://arxiv.org/abs/2610.14431v1); [Jev-Guard](https://arxiv.org/abs/2610.18326v1); [Two Routes to Interoceptive Prioritisation](https://arxiv.org/abs/2610.18617v1); [Average-Reward Policy Identification from a Single Continuing Trajectory](https://arxiv.org/abs/2610.20486v1); [Beyond the Trial-and-Error Loop](https://arxiv.org/abs/2610.19423v1).
- **General agents, non-navigation work, and retrieval noise (C)**：[Generative World Simulation with DAO and JING](https://arxiv.org/abs/2610.19726v1); [MolHop](https://arxiv.org/abs/2610.19920v1); [Context Matters! Social, Spatial, and Semantic Coordination with an Embodied Agent in Immersive Analytics](https://arxiv.org/abs/2610.21484v1); [Evaluating Inquiry Learning in LLM-Based Agents](https://arxiv.org/abs/2610.21604v1).

### 5.5 News and non-paper items
{: id="news"}

- No new WeChat news this issue. All items above are arXiv records, including surveys, benchmarks, dissertations, and other research formats.


## 6. Trends and recommended actions
{: id="trends-and-actions"}

### Trends
{: id="trends"}

- **Execution verification is becoming a separate research target.** VLN-Safe, S3-Nav, and I2O2I-Nav expose collision, evidence-authority, and handoff problems. Our assessment: measure correct proposals, permission to execute, and actual completion separately.
- **Memory increasingly connects to active evidence acquisition.** MemoryVLN, RefBeliefNav, and MemAct address reuse, revisitation, and new viewpoints. Our assessment: measure repeated search avoided and incorrect beliefs withdrawn.
- **Data and system improvements need failure attribution.** WEDGE, EnactOpt, and AdaRSI intervene on environments, runtimes, and executors. Our assessment: the automatic primary pool contains 64 of 175 works; a large secondary pool does not mean manipulation dominates relevant progress. Allocate reading by research question, not paper count.

### Research gaps
{: id="research-gaps"}

- Abstract-level SR gains do not establish how cross-context memory handles scene changes, conflicting evidence, and test-information isolation.
- Continuous indoor–outdoor missions need joint evaluation of stopping, handoffs, and collision regimes; primitive scores cannot predict full-mission reliability.
- Direct comparisons of added supervision, memory, and guards under matched hardware, sensors, interaction, and training budgets remain missing.

### Recommended actions
{: id="recommended-actions"}

**High priority**

- **Replicate VLN-Safe's evaluation comparison.** Establish whether local R2R-CE uses sliding, then evaluate blocking and the collision guard.
- **Read GuidedNav, RSI-Nav, MemoryVLN, and VLN-AIEP closely.** Record splits, baselines, absolute SR/SPL, gain definitions, and compute budgets before comparing them.
- **Add stage-handoff logging.** Following I2O2I-Nav, record predecessor endpoints, successor readiness, recovery, and evidence for final stopping.

**Medium priority**

- **Track S3-Nav, WEDGE, and RefBeliefNav.** Design small ablations for execution authorization, difficult-environment generation, and uncertainty-guided revisitation.
- **Check transferable memory protocols.** Compare MeMoREval and REMM using invalidated evidence, withdrawal of false completion, and reobservation tasks.

**Low priority**

- **Defer wholesale adoption of manipulation, aerial-navigation, and general-agent systems.** Validate specific interfaces or evaluation mechanisms first; do not extrapolate cross-task success into ground VLN gains.
