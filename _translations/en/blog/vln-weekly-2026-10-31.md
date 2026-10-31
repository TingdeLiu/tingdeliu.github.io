---
layout: post
lang: en
translation_id: vln-weekly-2026-10-31
permalink: /en/vln-weekly-2026-10-31/
source_path: _posts/weekly-reports/2026-10-31-VLN-Weekly.md
source_url: /vln-weekly-2026-10-31/
source_revision_date: 2026-10-31
translation_updated: 2026-10-31
title: "Embodied Navigation Weekly (2026-10-21 to 2026-10-29)"
date: 2026-10-31
period_start: 2026-10-21
period_end: 2026-10-29
issue_number: 11
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "This issue adds 102 independent works, with 9 each in VLN and embodied agentic navigation. MOON reports R2R-CE / RxR-CE SR gains of 5.2 / 6.0 percentage points. EvoLife connects help-seeking with persistent place memory; the deep read also examines its guidance conditions."
---

> **This issue**: 102 new papers / 102 independent works · Tracks: VLN 9 / embodied agentic navigation 9 · Other directions 84 · WeChat 0 · 1 existing paper updated

## 1. Key points
{: id="key-points"}

- **Memory maintenance now covers retention, clearing, and reflection.** [Spatiotemporal Memory](#spatiotemporal-memory) organizes historical evidence, [MOON](#moon) addresses post-deviation inertia, and [RetroNav](#retronav) links failure explanations to scene records.
- **The planning–execution interface becomes a training objective.** [MATCH](#match) jointly optimizes waypoint requests and primitive-action execution. This batch's semantic-safety evaluation also tests risk recognition separately from motion adaptation.
- **Help-seeking enters lifelong navigation harnesses.** [EvoLife](#evolife) connects interaction, action tools, and place memory. Its results depend on guidance conditions; first visits, revisits, and continuous tasks require separate reading.

## 2. Deep reads
{: id="deep-reads"}

### 1. MOON
{: id="moon"}

**Recovery requires deciding when to clear history.**

**[Paper](https://arxiv.org/abs/2610.33546v1)** · Track: VLN · Instruction following · R2R-CE / RxR-CE (continuous environments)

**Problem.** History supports route following but can perpetuate action inertia after a deviation.

**Method.** A deviation judge triggers history clearing and hands control to a GRPO-trained recovery policy. The frozen base receives less than 1% additional parameters.

<div align="center">
<img src="/images/vln/MOON-overview.webp" alt="MOON architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1577/871" />
<figcaption>An alarm clears history and activates recovery; the lower panel shows recovery training.（<a href="https://arxiv.org/html/2610.33546v1#S3.F4">Original figure</a>）</figcaption>
</div>

**Evidence.** The abstract reports SR gains of 5.2 and 6.0 percentage points over the frozen base on R2R-CE and RxR-CE. A random-trigger ablation supports the importance of timing.

**Assessment.** This separates memory errors from action errors. Clearing can harm on-route episodes, so false alarms must be measured alongside recovery. The abstract gives no physical-robot results.

**Next.** Reproduce: compare retained history, periodic clearing, judge-triggered clearing, and recovery from identical deviation states.

### 2. MATCH
{: id="match"}

**Planning targets should adapt with executor capability.**

**[Paper](https://arxiv.org/abs/2610.38248v1)** · Track: VLN · Hierarchical planning and execution · R2R-CE / RxR-CE (continuous environments)

**Problem.** A learned planner paired with a fixed executor can repeatedly request unreachable waypoints.

**Method.** An execution-aware graph combines value estimates and outcome memory with continuous target refinement. Rewards balance ordered route progress and execution residual; both policies update together.

<div align="center">
<img src="/images/vln/MATCH-overview.webp" alt="MATCH architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/520" />
<figcaption>The planner issues requests, the executor produces primitive actions, and shared rollouts update both policies.（<a href="https://arxiv.org/html/2610.38248v1#S2.F2">Original figure</a>）</figcaption>
</div>

**Evidence.** The abstract gives no numerical task scores, claiming leading R2R-CE / RxR-CE results with 0.5B parameters. The paper shows only 4 physical-robot episodes.

**Assessment.** This targets executability while retaining instruction constraints. 4 hardware episodes cannot establish deployment reliability; local arrival is also insufficient evidence of full instruction completion.

**Next.** Read closely: check collision settings and isolate target offsets, joint training, and route rewards.

### 3. EvoLife
{: id="evolife"}

**Connect help-seeking and place memory to the navigation loop.**

**[Paper](https://arxiv.org/abs/2610.39446v1)** · Track: embodied agentic navigation · Interactive navigation / ObjectNav / continuous multi-task navigation

**Problem.** Separating movement, clarification, and memory makes interaction-derived knowledge hard to reuse.

**Method.** A unified harness exposes waypoint, discrete, pixel, and object navigation tools. Dialogue and observations attach to places for later retrieval and revision.

<div align="center">
<img src="/images/vln/EvoLife-overview.webp" alt="EvoLife architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1977/1350" />
<figcaption>Hybrid actions, active interaction, and persistent spatial memory share one decision loop.（<a href="https://arxiv.org/html/2610.39446v1#S3.F2">Original figure</a>）</figcaption>
</div>

**Evidence.** The abstract reports 100% first-visit SR across four 100-episode VLN / ObjectNav sets, averaging 1.87 interactions per episode. Complete-sequence SR is 98% and 94% on LHPR-VLN and GOAT-Bench, respectively (50 sequences each).

**Assessment.** These are assisted-system results. The simulation guide accesses privileged routes and goals, precluding equivalence with unassisted VLN. Real help-seeking costs and incorrect answers need evaluation.

**Next.** Read closely: hold the navigator and interaction budget fixed while comparing absent, restricted, and full guidance.

### 4. RetroNav
{: id="retronav"}

**Failure reflection needs verifiable scene evidence.**

**[Paper](https://arxiv.org/abs/2610.35172v1)** · Track: embodied agentic navigation · Lifelong navigation / repeated attempts

**Problem.** Scene knowledge alone does not explain failed decisions or identify what a retry should change.

**Method.** Unified memory links geometry, semantics, and navigation experience while separating observations from task interpretations. An independent evaluator requests retries; a retrospection agent produces an evidence-grounded brief.

<div align="center">
<img src="/images/vln/RetroNav-overview.webp" alt="RetroNav architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1054" />
<figcaption>Navigation accumulates scene and behavior memory; evaluation and retrospection generate a brief for the next attempt.（<a href="https://arxiv.org/html/2610.35172v1#S1.F1">Original figure</a>）</figcaption>
</div>

**Evidence.** The abstract gives no numerical results, claiming leading performance on five navigation benchmarks.

**Assessment.** This can test whether reflection cites genuine evidence. Repeated attempts add information and budget, so first-attempt success, retry recovery, cumulative path length, and model-call costs should all be reported.

**Next.** Reproduce: compare naive retries, scene-memory-only retries, and evidence-grounded briefs with the same retry budget.

### 5. Spatiotemporal Memory
{: id="spatiotemporal-memory"}

**Memory must support progress, historical evidence, and geometry.**

**[Paper](https://arxiv.org/abs/2610.38318v1)** · Track: embodied agentic navigation · Zero-shot VLN · R2R-CE val-unseen (continuous environments)

**Problem.** Retaining history does not ensure accessible evidence; current views alone invite repeated exploration.

**Method.** Task memory tracks progress, temporal memory compresses pose-anchored observations, and spatial memory accumulates geometry. An 8B observer and an 8B reasoner select waypoints or extra views; geometric control executes them.

<div align="center">
<img src="/images/vln/SpatiotemporalMemory-overview.webp" alt="SpatiotemporalMemory architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1259" />
<figcaption>Three memories support waypoint selection and PREVIEW observations, with further updates after execution.（<a href="https://arxiv.org/html/2610.38318v1#S2.F1">Original figure</a>）</figcaption>
</div>

**Evidence.** The abstract reports that removing temporal memory increases average path length from 16.9 m to 39.1 m on R2R-CE val-unseen, without numerical SR.

**Assessment.** Path efficiency merits attention, but this is more than one “8B model.” Hardware evaluation uses a remote A100 and localization in a prebuilt map, leaving fully onboard operation without prior maps unestablished.

**Next.** Read closely: align localization and action interfaces, then compare paths, SR, tokens, and extra-view costs.

## 3. Also worth reading
{: id="also-worth"}

- **[RepresentNav](https://arxiv.org/abs/2610.31762v1)** · VLN: self-constructed visual features and latent prediction; the abstract reports 83.80% SR on VLNVerse fine-grained.
- **[One Scan is Enough](https://arxiv.org/abs/2610.33794v1)** · VLN: one scan builds training environments; the abstract reports 85% SR across five real environments and 82% in unscanned areas of the same buildings.
- **[MEMVLN](https://arxiv.org/abs/2610.35066v2)** · VLN: bounded episodic memory and route reconstruction; hardware testing holds out only six instructions, requiring scrutiny of repeated-trial protocols.
- **[ASPEN](https://arxiv.org/abs/2610.36285v1)** · VLN: event-triggered semantic reasoning; on RxR-CE val-unseen, VLA calls fall 9.5% and estimated inference time falls 8.1%.
- **[Natural-Language to Nav2](https://arxiv.org/abs/2610.32633v1)** · Agentic navigation: language compiles to JSON policies, passes deterministic checks, and projects into Nav2 costmaps.
- **[PIFR](https://arxiv.org/abs/2610.35576v1)** · Agentic navigation: PIFR separates inserted-task completion, original-task resumption, and full-sequence completion.
- **[HGWM](https://arxiv.org/abs/2610.32554v3)** · Agentic navigation: room–object priors and persistent graph memory support graph matching; HM3D v0.1 SR 62.6%, SPL 31.5%.
- **[ThinkLocoManip](https://arxiv.org/abs/2610.34370v1)** · Agentic navigation: a legged loco-manipulation harness orchestrates navigation, locomotion, and manipulation, retaining recovery context through keyframes.
- **[VISTA](https://arxiv.org/abs/2610.35051v1)** · Transferable mechanism: actively generated rollouts validate harness edits; this could inform retesting after navigation reflection.
- **[LOGIV](https://arxiv.org/abs/2610.40016v1)** · Transferable mechanism: continuous dependency checks and local repair could maintain navigation subgoal preconditions.

## 4. Trends and actions
{: id="trends-and-actions"}

**Trends**

- **Memory gains depend on timing and evidence organization.** [MOON](#moon), [RetroNav](#retronav), and [Spatiotemporal Memory](#spatiotemporal-memory) address clearing, reflection, and retrieval. They warrant comparison under shared budgets rather than by cache length alone.
- **Harnesses need separate accounting for execution, assistance, and recovery.** [MATCH](#match) constrains execution requests, [EvoLife](#evolife) introduces external guidance, and PIFR handles runtime interruption. Final arrival rates cannot explain each component's contribution.

**Research gaps**

- Navigation recovery lacks evaluations jointly fixing action interfaces, history budgets, model calls, and retry counts. Stronger policies remain difficult to distinguish from additional opportunities.
- Privileged guidance, remote inference, and prebuilt-map localization differ from ordinary ground-robot conditions. Removing them individually would establish practical capability boundaries.

**Next steps**

- **High · Reproduce** [MOON](#moon): build matched-state deviation interventions and record false alarms, recovery SR, and cumulative path length.
- **High · Read closely** [EvoLife](#evolife): list guide-visible information and compare absent versus restricted guidance before assessing transfer.
- **Medium · Compare** [MATCH](#match): fix executors and contact settings, then compare target adjustment, executor training, and joint training.
- **Medium · Prototype** Natural-Language to Nav2: begin with keepouts and event-conditioned access, separately measuring parsing errors, incorrect acceptance, and execution deviations.
- **Low · Track** [RetroNav](#retronav) and [Spatiotemporal Memory](#spatiotemporal-memory): measure first attempts, retries, and token costs under equal budgets; defer rankings based solely on final SR.

## 5. Complete work index
{: id="full-index"}

The index covers all 102 newly added independent works; A denotes the two main tracks, B transferable mechanisms, and C broader developments. Deep-read entries link back to their cards.

<details markdown="1">
<summary>VLN and semantic navigation · A · 9 works</summary>

- [RepresentNav](https://arxiv.org/abs/2610.31762v1): Policy-built visual representations and future latents.
- [MOON](#moon): See deep read.
- [One Scan is Enough](https://arxiv.org/abs/2610.33794v1): Scene scans generate instructions and training tasks.
- [Stage-Aware Attribution](https://arxiv.org/abs/2610.34028v1): Instruction-stage-aware attribution across current and historical views.
- [ScheNav](https://arxiv.org/abs/2610.34286v1): Sketches and observations form exploration value fields.
- [ASPEN](https://arxiv.org/abs/2610.36285v1): Command persistence schedules fresh semantic reasoning.
- [OSMA-Bench](https://arxiv.org/abs/2610.34674v1): Tests clearance, speed, and smoothness by obstacle semantics.
- [MEMVLN](https://arxiv.org/abs/2610.35066v2): Bounded episodic memory with masked route reconstruction.
- [MATCH](#match): See deep read.

</details>

<details markdown="1">
<summary>Embodied agentic navigation · A · 9 works</summary>

- [HGWM](https://arxiv.org/abs/2610.32554v3): Room–object priors and persistent graph memory guide search.
- [Natural-Language to Nav2](https://arxiv.org/abs/2610.32633v1): Validated language policies project into Nav2 costmaps.
- [ThinkLocoManip](https://arxiv.org/abs/2610.34370v1): Orchestrates locomotion, navigation, and manipulation with keyframe recovery.
- [OmniBrain-0](https://arxiv.org/abs/2610.36013v1): Distills navigation and manipulation experts into shared decisions.
- [PIFR](https://arxiv.org/abs/2610.35576v1): Separately evaluates preemption, inserted tasks, and resumption.
- [RetroNav](#retronav): See deep read.
- [Spatiotemporal Memory](#spatiotemporal-memory): See deep read.
- [EvoLife](#evolife): See deep read.
- [LLM-Guided Loco-Manipulation](https://arxiv.org/abs/2610.40248v1): Composes interactive navigation through skills, objects, and geometric targets.

</details>

<details markdown="1">
<summary>Transferable mechanisms · B · 19 works</summary>

- [ASTEROID](https://arxiv.org/abs/2610.30105v1): Student-history distillation could inform partially observed decisions.
- [FP32 Master Copy](https://arxiv.org/abs/2610.31859v1): Lost low-precision updates motivate navigation fine-tuning audits.
- [Horizon–Depth Coupling](https://arxiv.org/abs/2610.32102v1): Studies solver depth together with open-loop execution horizon.
- [SceneBloom](https://arxiv.org/abs/2610.32978v1): Physically valid indoor layouts support ObjectNav training.
- [TrajMamba](https://arxiv.org/abs/2610.33916v1): Aligns human-trajectory preprocessing between training and deployment.
- [Interaction-Aware GNN](https://arxiv.org/abs/2610.34060v1): Graph-based waypoint corrections use pedestrian and goal context.
- [LegoOcc++](https://arxiv.org/abs/2610.34093v1): Open-vocabulary occupancy maintains cross-view instance semantics.
- [RoboDuo](https://arxiv.org/abs/2610.34108v1): Checks task completion before invoking atomic recovery skills.
- [FISRF](https://arxiv.org/abs/2610.35686v2): Selects reflection and mentor guidance according to capability.
- [EBench Agent Profiling](https://arxiv.org/abs/2610.35832v1): Compares direct control against harnesses invoking learned policies.
- [RobotAgentBench](https://arxiv.org/abs/2610.34649v1): Evaluates autonomy through task discovery, execution, and recovery.
- [The Trinity](https://arxiv.org/abs/2610.36077v1): Capability trees organize verifiable worlds and harness evolution.
- [TP-JSCC](https://arxiv.org/abs/2610.35489v1): Compresses and transmits visual tokens for 3D question answering.
- [VISTA](https://arxiv.org/abs/2610.35051v1): Active exploration tests edits while retaining harness lineage.
- [LSMM](https://arxiv.org/abs/2610.37203v1): Position-indexed matrix memory supports legged maze exploration.
- [AtomTree](https://arxiv.org/abs/2610.38744v1): Current symbolic states guide subgoal decomposition and hierarchical backtracking.
- [Object Motion Anticipation](https://arxiv.org/abs/2610.38993v1): Adapts online to anticipate motion onset of novel objects.
- [EmbodiExp](https://arxiv.org/abs/2610.39180v1): Composes subgoal experiences through applicability and verified outcomes.
- [LOGIV](https://arxiv.org/abs/2610.40016v1): Continuously verifies task dependencies and constrains local repair.

</details>

<details markdown="1">
<summary>VLA / manipulation · C · 43 works</summary>

- **Demonstrations, data, skills, and simulation（16）**: [EgoForesight](https://arxiv.org/abs/2610.31845v1), [MAST-VLA](https://arxiv.org/abs/2610.32312v1), [Learned Speed Prior](https://arxiv.org/abs/2610.32676v1), [EXE](https://arxiv.org/abs/2610.33685v1), [Speech Conditioning](https://arxiv.org/abs/2610.33866v1), [Dexterous Manipulation Survey](https://arxiv.org/abs/2610.33486v1), [TAD](https://arxiv.org/abs/2610.33650v1), [HuGe](https://arxiv.org/abs/2610.34186v1), [RynnVLA-Latent](https://arxiv.org/abs/2610.34356v1), [Robot-Aligned 3DGS](https://arxiv.org/abs/2610.35626v1), [Self-Labeled Preference Transfer](https://arxiv.org/abs/2610.38000v1), [Tokens of Motion](https://arxiv.org/abs/2610.37330v1), [BURGER](https://arxiv.org/abs/2610.40532v1), [Gym-Anything for Robotics](https://arxiv.org/abs/2610.40565v1), [DEED](https://arxiv.org/abs/2610.40430v1), [TouchBridge](https://arxiv.org/abs/2610.40349v1)
- **Memory, world representations, and action generation（11）**: [CogMemory](https://arxiv.org/abs/2610.34251v1), [PDRVLA](https://arxiv.org/abs/2610.37381v1), [Retrieve to Act](https://arxiv.org/abs/2610.37215v1), [ThinkWorld](https://arxiv.org/abs/2610.37666v1), [JAM](https://arxiv.org/abs/2610.37219v1), [World Dreamer v5-lite](https://arxiv.org/abs/2610.37518v1), [SaiVLA-JEPA](https://arxiv.org/abs/2610.38508v1), [CoRAct](https://arxiv.org/abs/2610.39543v1), [HiFOR](https://arxiv.org/abs/2610.40373v1), [PRIME W0](https://arxiv.org/abs/2610.39528v1), [Smooth Action Chunking](https://arxiv.org/abs/2610.39451v1)
- **Quantization, inference, and onboard hardware（6）**: [AGQ](https://arxiv.org/abs/2610.33074v1), [PoT-LIF](https://arxiv.org/abs/2610.33658v1), [TRIAGE](https://arxiv.org/abs/2610.34519v1), [JevVLA](https://arxiv.org/abs/2610.35010v1), [VLA-HWBench](https://arxiv.org/abs/2610.34717v2), [QbD-VLA](https://arxiv.org/abs/2610.38661v1)
- **Robustness, safety, and evaluation（9）**: [VLA Failure Features](https://arxiv.org/abs/2610.32208v1), [Backtrack](https://arxiv.org/abs/2610.32941v1), [RobustRLlib](https://arxiv.org/abs/2610.32971v1), [Federated VLA Safety](https://arxiv.org/abs/2610.32888v1), [MART-GRPO](https://arxiv.org/abs/2610.32431v1), [Stimulus–Response Safety](https://arxiv.org/abs/2610.34837v1), [VLA-Stem](https://arxiv.org/abs/2610.37378v1), [BEHAVIOR-1K Inference Settings](https://arxiv.org/abs/2610.37917v1), [RoboSeed](https://arxiv.org/abs/2610.40026v1)
- **Mobile manipulation practice（1）**: [Tabletop to Walking Bot](https://arxiv.org/abs/2610.32271v1)

</details>

<details markdown="1">
<summary>Flight, control, and other topics · C · 22 works</summary>

- **Aerial and underwater robotics（6）**: [AirPlayground](https://arxiv.org/abs/2610.29699v1), [SP-Guide](https://arxiv.org/abs/2610.32697v1), [Quadrotor Trajectory Control](https://arxiv.org/abs/2610.34456v1), [Omni-IGN](https://arxiv.org/abs/2610.36968v1), [Embodied at Sea](https://arxiv.org/abs/2610.38199v1), [OpenFlyV2](https://arxiv.org/abs/2610.39287v1)
- **Geometry, control, localization, and driving（6）**: [Continuous-Time Safe Planning](https://arxiv.org/abs/2610.35670v1), [LECO](https://arxiv.org/abs/2610.34929v1), [SG²SLAM](https://arxiv.org/abs/2610.36312v1), [Geomagnetic Gradient Flow](https://arxiv.org/abs/2610.36812v1), [Value-Guided MPPI](https://arxiv.org/abs/2610.38731v1), [CoC-CFG](https://arxiv.org/abs/2610.40002v1)
- **Laboratories, general agents, and other topics（10）**: [VERID](https://arxiv.org/abs/2610.32755v1), [Whale You Go Green?](https://arxiv.org/abs/2610.33191v1), [Patchwork](https://arxiv.org/abs/2610.33582v1), [EMMA](https://arxiv.org/abs/2610.35966v1), [Intrinsic Goals / Drosophila](https://arxiv.org/abs/2610.34638v1), [Language to Drives](https://arxiv.org/abs/2610.35629v1), [LongChang](https://arxiv.org/abs/2610.37649v1), [Auditable Self-Driving Labs](https://arxiv.org/abs/2610.36761v1), [MCP-Net](https://arxiv.org/abs/2610.37127v1), [Live iOS Navigation](https://arxiv.org/abs/2610.39264v1)

</details>
