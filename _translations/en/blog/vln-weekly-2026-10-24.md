---
layout: post
lang: en
translation_id: vln-weekly-2026-10-24
permalink: /en/vln-weekly-2026-10-24/
source_path: _posts/weekly-reports/2026-10-24-VLN-Weekly.md
source_url: /vln-weekly-2026-10-24/
source_revision_date: 2026-10-24
translation_updated: 2026-10-24
title: "Embodied Navigation Weekly (2026-10-15 to 2026-10-22)"
date: 2026-10-24
period_start: 2026-10-15
period_end: 2026-10-22
issue_number: 10
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "This issue adds 119 independent works, including 14 in the VLN track and 5 in embodied agentic navigation, with 2 involving R2R-CE. SimpleVLN reports R2R-CE SR of 76.7% and RxR-CE SR of 80.1%; goal verification, body-dependent passability, and strict-contact execution are this issue's focus."
---

> **This issue**: 119 new papers / 119 independent works · Tracks: VLN 14 / embodied agentic navigation 5 · Other directions 100 · R2R-CE 2 · WeChat 0 · 6 existing papers updated

## 1. Key points
{: id="key-points"}

- **Continuous-navigation performance and instruction fidelity are being tested separately.** [SimpleVLN](#simplevln) reports continuous-benchmark results; NavBind fixes the scene and history while changing fork instructions to test whether actions change with language. The two R2R-CE works provide different kinds of evidence.
- **Finding a candidate object still requires verification.** [CLOSER](#closer) turns verification before stopping into evidence collection across views. Goal recognition, distance estimation, and termination are studied separately rather than all being treated as exploration failures.
- **Reachability and physical passability impose different conditions.** [PASS-VLN](#pass-vln) varies robot width to test reaching versus stopping; [VeSTA](#vesta) optimizes candidate trajectories with strict-contact feedback. They address embodiment constraints and collision execution, respectively.
- **Remembering dialogue does not guarantee locating its referent.** In [DialogMem](#dialogmem), appending dialogue to visual memory substantially improves question answering but barely improves navigation. Object linking, correct retrieval, and location recovery remain separate bottlenecks.

## 2. Deep reads
{: id="deep-reads"}

### 1. SimpleVLN
{: id="simplevln"}

**Preserve the native architecture and jointly improve memory, data, and training.**

**[Paper](https://arxiv.org/abs/2610.23203v1)** · Track: VLN · [My reading guide (Chinese)](https://tingdeliu.github.io/VLN-Papers/#simplevln) · Instruction following · R2R-CE / RxR-CE (continuous environments)

**Problem.** Has the navigation potential of native multimodal models been exhausted before adding specialized modules?

**Method.** Fixed-budget global-local memory and action chunks accompany expert demonstrations, policy-dependent DAgger data, and synthesized tasks in progressive IL→DAgger→RL training.

<div align="center">
<img src="/images/vln/SimpleVLN-overview.webp" alt="SimpleVLN architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1118/692" />
<figcaption>SimpleVLN: History, the current observation, and instructions enter an MLLM that directly predicts an action chunk. (<a href="https://arxiv.org/html/2610.23203v1#S2.F2">Original paper figure 2</a>)</figcaption>
</div>

**Evidence.** The abstract reports R2R-CE SR of 76.7% and RxR-CE SR of 80.1%, without naming the specific splits.

**Assessment.** This is a useful ground-VLN baseline for testing architectural additions. Data and training change together, so ablations, data budgets, and contact settings need checking; the abstract gives no real-robot metrics.

**Next step.** Read closely: confirm evaluation splits, then separate memory, DAgger, and RL gains.

### 2. CLOSER
{: id="closer"}

**Turn stopping into active evidence acquisition.**

**[Paper](https://arxiv.org/abs/2610.28476v1)** · Track: VLN · ObjectNav · Gibson / Habitat-Matterport 3D

**Problem.** Confident single-frame detections can confuse similar objects and cause false stops.

**Method.** A fine-tuned VLM assigns target, non-target, or uncertain labels to views of tracked candidates. Diversity gating and probability fusion combine evidence; a diffusion policy approaches for further observation when uncertainty persists.

<div align="center">
<img src="/images/vln/CLOSER-overview.webp" alt="CLOSER architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1108" />
<figcaption>CLOSER: Candidate tracking and multi-view evidence fusion trigger active approach when uncertain. (<a href="https://arxiv.org/html/2610.28476v1#S3.F2">Original paper figure 2</a>)</figcaption>
</div>

**Evidence.** The abstract reports mean SR gains of 3.6 points and SPL gains of 2.0 points across the two scene collections.

**Assessment.** This could supply stop verification for existing ObjectNav systems. The navigator needs no retraining, but the verifier VLM requires fine-tuning. The abstract gives no real-robot results; extra travel, latency, and false rejection need checking.

**Next step.** Reproduce: compare single-view stopping, passive multi-view fusion, and active verification on false stops and SPL.

### 3. PASS-VLN
{: id="pass-vln"}

**The same instruction should yield different outcomes as robot width changes.**

**[Paper](https://arxiv.org/abs/2610.22812v1)** · Track: VLN · Local language navigation · PASS-VLN

**Problem.** Assuming instructions are executable conceals local passage failures caused by body size.

**Method.** Scene, instruction, start, and reference route remain fixed while body width varies. Obstacle inflation in a depth-derived local BEV encodes width, fused with visual-language context to predict latent action codes for continuous closed-loop execution.

<div align="center">
<img src="/images/vln/PASS-VLN-overview.webp" alt="PASS-VLN architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/726" />
<figcaption>PASS-VLN: Body-width-conditioned geometry fuses with visual-language information before closed-loop action decoding. (<a href="https://arxiv.org/html/2610.22812v1#S3.F2">Original paper figure 2</a>)</figcaption>
</div>

**Evidence.** The abstract reports approximately 12 percentage points higher collision-free success on PASS-VLN than a strong VLN model fine-tuned on the same benchmark, plus unseen-width generalization and real-robot deployment.

**Assessment.** The benchmark separates route misunderstanding from physical blockage. Local commands do not replace long-horizon R2R-CE; no real-robot success rate is given.

**Next step.** Read closely: check stopping criteria for blocked passages and width boundaries under depth error.

### 4. VeSTA
{: id="vesta"}

**Align training feedback with strict-contact execution.**

**[Paper](https://arxiv.org/abs/2610.25687v1)** · Track: VLN · Continuous VLN · Abstract names R2R / RxR without an explicit CE designation

**Problem.** Permissive collision models may let physically infeasible actions complete navigation.

**Method.** Trajectory Relative Group Policy Optimization uses strict-contact feedback to increase executable candidates. A traversability risk critic then selects candidate futures by risk-adjusted utility.

<div align="center">
<img src="/images/vln/VeSTA-overview.webp" alt="VeSTA architecture" width="100%" loading="lazy" decoding="async" style="aspect-ratio:2000/1055" />
<figcaption>VeSTA: Strict-contact feedback reshapes trajectories; a risk critic selects executable candidates. (<a href="https://arxiv.org/html/2610.25687v1#S1.F2">Original paper figure 2</a>)</figcaption>
</div>

**Evidence.** No numerical results in the abstract; the paper reports fewer collision failures under strict contact and controlled real-robot transfer experiments.

**Assessment.** Candidate-distribution improvement and candidate selection need separate attribution. The task uses continuous execution, but the abstract's R2R / RxR names do not establish the CE versions. Results should neither be treated as discrete-benchmark scores nor entered directly into CE rankings.

**Next step.** Read closely: confirm task versions, collision implementation, and hardware sample sizes before comparing SR and collisions.

### 5. DialogMem
{: id="dialogmem"}

**Dialogue memory must link to objects that can be located.**

**[Paper](https://arxiv.org/abs/2610.24767v1)** · Track: embodied agentic navigation · Dialogue memory · Instance-level object navigation / question answering

**Problem.** Visual memory struggles to persistently link human dialogue to specific previously observed objects.

**Method.** Memory is formed from aligned posed RGB-D and dialogue streams, then retrieved for navigation and question answering, with unified comparisons of formation, retrieval, and output reasoning.

<div align="center">
<img src="/images/vln/DialogMem-overview.webp" alt="DialogMem task overview" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1890/1276" />
<figcaption>DialogMem: Memory forms from vision and dialogue, then supports instance navigation and question answering. (<a href="https://arxiv.org/html/2610.24767v1#Sx2.F1">Original paper figure 1</a>)</figcaption>
</div>

**Evidence.** The benchmark contains 142 episodes in 27 real-world scenes, 14,193 QA pairs, and 11,890 navigation instructions. The abstract gives no navigation SR and reports negligible navigation gains from appending dialogue.

**Assessment.** Remembering relevant speech and retrieving the right object and location require separate validation. Real-scene data do not establish closed-loop robot deployment; QA gains do not establish navigation gains.

**Next step.** Read closely: inspect linking, retrieval, and localization diagnostics for error accumulation across stages.

## 3. Also worth a look
{: id="also-worth"}

- **[NavBind](https://arxiv.org/abs/2610.29203v2)** · R2R-CE / RxR-CE: 3,252 fork contexts test instruction–action binding with fixed history.
- **[StopWatch](https://arxiv.org/abs/2610.23841v1)** · ObjectNav: explicit goal grounding and distance estimation, with corrected annotations; no numerical results in the abstract.
- **[ReInsVLN](https://arxiv.org/abs/2610.26545v1)** · Memory use: compresses history and schedules reminders; the abstract says R2R without specifying continuous or discrete execution.
- **[HAVANA](https://arxiv.org/abs/2610.24122v1)** · Two-rate planning: slow latent plans and fast local trajectories, without future-video generation at inference.
- **[CovNav](https://arxiv.org/abs/2610.27677v1)** · ObjectNav: calibrated uncertainty determines when to invoke slow reasoning.
- **[ChairNav](https://arxiv.org/abs/2610.24012v1)** · Physical wheelchair: cross-embodiment pretraining, domain adaptation, and post-training from user interventions.
- **[EchoNav](https://arxiv.org/abs/2610.27616v1)** · Visual navigation: historical fusion and confident occupancy completion distinguish observed free space.
- **[Sparse Topometric Graph](https://arxiv.org/abs/2610.31432v1)** · Cross-session navigation: local geometry and connectivity support map matching.
- **[InstAgent](https://arxiv.org/abs/2610.23098v1)** · Collaboration: behavior trees encode assignment and handoff checks, retaining programs revised through feedback.
- **[Refinable Semantic Mapping](https://arxiv.org/abs/2610.31476v1)** · Dynamic memory: pointwise confidence updates remove noise and obsolete object points.

## 4. Trends and actions
{: id="trends-and-actions"}

**Trends**

- **Navigation evaluation is separating arrival, instruction following, and executability.** [SimpleVLN](#simplevln), NavBind, and [VeSTA](#vesta) provide evidence on performance, language dependence, and physical execution. These dimensions should be reported together; one SR cannot substitute for them.
- **Stopping and handoffs are becoming independent feedback loops.** [CLOSER](#closer), StopWatch, and InstAgent make goal confirmation or stage-completion evidence explicit. Useful comparisons include false stops, missed stops, and execution overhead.
- **Memory increasingly concerns invalidation and localization, beyond retaining more history.** [DialogMem](#dialogmem), Refinable Semantic Mapping, and EchoNav support separate evaluation of object linking, evidence updating, and spatial recovery.

**Research gaps**

- A unified ground-VLN evaluation combining instruction counterfactuals, body width, strict collisions, and stop verification is missing. Whether isolated module gains compose over long horizons remains unverified.
- This issue provides insufficient evidence that dialogue-object memory supports real-robot localization and safe stopping after objects move.

**Next steps**

- **High · Read closely** [SimpleVLN](#simplevln): confirm continuous-benchmark splits, data, and training costs to establish a comparable baseline.
- **High · Reproduce** NavBind: swap instructions in paired fixed fork states, measuring both action changes and directional correctness.
- **High · Reproduce** [CLOSER](#closer): retain the navigator and test verification effects on false stops, success, and path cost.
- **Medium · Read closely** [PASS-VLN](#pass-vln) / [VeSTA](#vesta): identify implementation differences between embodiment constraints and strict contact, then design ablations.
- **Medium · Track** [DialogMem](#dialogmem): follow object-linking and localization interfaces, prioritizing moving-object conditions.

## 5. Complete work index
{: id="full-index"}

A denotes direct relevance, B a transferable mechanism, and C field observation; the five groups below cover all 119 independent works in this issue.

<details markdown="1">
<summary>VLN and semantic navigation · A · 14 works</summary>

- [ChairNav](https://arxiv.org/abs/2610.24012v1): Wheelchair adaptation across embodiments and from user interventions.
- [HAVANA](https://arxiv.org/abs/2610.24122v1): Asynchronous latent plans and fast local trajectories.
- [PASS-VLN](#pass-vln): see deep read.
- [SimpleVLN](#simplevln): see deep read.
- [StopWatch](https://arxiv.org/abs/2610.23841v1): Explicit goal grounding and stopping.
- [EmbodiedWorld-200K](https://arxiv.org/abs/2610.25959v1): Navigation-planning data and interaction training in world models.
- [ReInsVLN](https://arxiv.org/abs/2610.26545v1): Compressing and revisiting historical evidence.
- [VeSTA](#vesta): see deep read.
- [CLOSER](#closer): see deep read.
- [CovNav](https://arxiv.org/abs/2610.27677v1): Selecting slow reasoning with calibrated uncertainty.
- [EchoNav](https://arxiv.org/abs/2610.27616v1): Evidential occupancy completion from historical observations.
- [Navigation Explainability](https://arxiv.org/abs/2610.28920v1): Explaining assistive navigation with obstacles and future trajectories.
- [NavBind](https://arxiv.org/abs/2610.29203v2): Testing language–action dependence at fixed route forks.
- [FETA](https://arxiv.org/abs/2610.30643v1): Adapting through binary feedback on selected critical actions.

</details>

<details markdown="1">
<summary>Embodied agentic navigation · A · 5 works</summary>

- [InstAgent](https://arxiv.org/abs/2610.23098v1): Behavior trees, shared memory, and collaborative handoffs.
- [DialogMem](#dialogmem): see deep read.
- [Structure over Reasoning](https://arxiv.org/abs/2610.25130v1): A small model supplies perceptual evidence to structured navigation modules.
- [DPGP](https://arxiv.org/abs/2610.26359v1): Discovery-probability graphs for hidden cooperation needs.
- [TARSphere](https://arxiv.org/abs/2610.30043v1): Long-horizon tasks under evolving physical and social constraints.

</details>

<details markdown="1">
<summary>Transferable mechanisms and VLA / manipulation · B/C · 76 works</summary>

- [Cambrian-BEV](https://arxiv.org/abs/2610.22700v1): Spatial token aggregation compresses multi-frame histories.
- [FOUND-IT System](https://arxiv.org/abs/2610.23622v1): Integrating monocular scene graphs into mobile manipulation.
- [Robot Metareasoning](https://arxiv.org/abs/2610.23662v1): State-dependent computation and action-commitment budgets.
- [Offline Empowerment](https://arxiv.org/abs/2610.24461v1): Estimating exploration value from offline trajectories.
- [GOFusion](https://arxiv.org/abs/2610.25261v1): Fusing occupancy and Gaussian submaps for active observation.
- [LateMatch](https://arxiv.org/abs/2610.25358v1): Caching intra-image context and deferring cross-image matching.
- [RCRA](https://arxiv.org/abs/2610.25419v1): Adapting conservative navigation to pedestrian reactivity.
- [BetaMem](https://arxiv.org/abs/2610.26554v1): Traceable memory writing, forgetting, and local rule repair.
- [CoopUQ](https://arxiv.org/abs/2610.26852v1): Cooperative occupancy forecasting and uncertainty-aware navigation.
- [ForagerSim](https://arxiv.org/abs/2610.26395v1): Evaluating map-based reasoning in large dynamic worlds.
- [LVMem](https://arxiv.org/abs/2610.27104v1): Linking retrieved text back to original visual evidence.
- [MEAN](https://arxiv.org/abs/2610.26297v1): A networking survey organized around closed-loop embodied tasks.
- [CAPSafe](https://arxiv.org/abs/2610.28381v1): Separating contextual action safety from long-horizon planning.
- [Industrial COMPASS](https://arxiv.org/abs/2610.28432v2): Industrial navigation transfer with digital twins and synthetic augmentation.
- [Externalise the Memory](https://arxiv.org/abs/2610.30478v1): Comparing external retrieval and distillation under fixed policies.
- [VLA Corpus Selection](https://arxiv.org/abs/2610.29646v1): Testing whether counterfactual diagnostics improve corpus selection.
- [PAPAV](https://arxiv.org/abs/2610.30108v1): Reviewing evaluation across perception, anticipation, planning, action, and verification.
- [Sparse Topometric Graph](https://arxiv.org/abs/2610.31432v1): Matching maps across sessions through geometry and connectivity.
- [CARE](https://arxiv.org/abs/2610.30587v1): Critic-guided cloud escalation with shared handoff memory.
- [Refinable Semantic Mapping](https://arxiv.org/abs/2610.31476v1): Pointwise confidence updates remove obsolete spatial memory.
- **Data, representations, and training (17)**: [GRASP](https://arxiv.org/abs/2610.23123v1), [ActiveAlign](https://arxiv.org/abs/2610.23378v1), [Model-to-Policy](https://arxiv.org/abs/2610.23053v1), [GazeAct](https://arxiv.org/abs/2610.24507v1), [Relational Spatial Alignment](https://arxiv.org/abs/2610.24705v1), [ALIGN](https://arxiv.org/abs/2610.26455v1), [FR-VLA](https://arxiv.org/abs/2610.25981v1), [Agricultural VLA](https://arxiv.org/abs/2610.26717v1), [VLDA](https://arxiv.org/abs/2610.25874v1), [PLaCA](https://arxiv.org/abs/2610.27657v1), [Real2Sem2Real](https://arxiv.org/abs/2610.28880v1), [Generative Plan Latents](https://arxiv.org/abs/2610.28166v1), [AtomicRL](https://arxiv.org/abs/2610.30097v1), [SimPT](https://arxiv.org/abs/2610.29161v1), [DRILL](https://arxiv.org/abs/2610.30814v1), [Sensorimotor Alignment](https://arxiv.org/abs/2610.31670v1), [Tactile Manipulation Survey](https://arxiv.org/abs/2610.30438v1)
- **Memory, monitoring, and execution frameworks (14)**: [RePlanMem-VLA](https://arxiv.org/abs/2610.22556v1), [IMEP](https://arxiv.org/abs/2610.23372v1), [VLA Memory Survey](https://arxiv.org/abs/2610.23319v1), [RTPC](https://arxiv.org/abs/2610.24573v1), [Runtime Safety Check](https://arxiv.org/abs/2610.27350v1), [EventSync](https://arxiv.org/abs/2610.27917v1), [FIT-MAC](https://arxiv.org/abs/2610.28109v1), [JARVIS](https://arxiv.org/abs/2610.27618v1), [RoboCasa-Shift](https://arxiv.org/abs/2610.27861v1), [SERA](https://arxiv.org/abs/2610.29560v1), [PAL](https://arxiv.org/abs/2610.29329v1), [Echo](https://arxiv.org/abs/2610.31569v1), [WholeBodyMem](https://arxiv.org/abs/2610.31166v1), [Progress Probes P2](https://arxiv.org/abs/2610.31613v1)
- **Inference, compression, and continual adaptation (12)**: [MambaAct](https://arxiv.org/abs/2610.22951v1), [NestedVLA](https://arxiv.org/abs/2610.23271v3), [FlashRT-PAI](https://arxiv.org/abs/2610.24349v1), [SARAM](https://arxiv.org/abs/2610.24804v1), [VLA-Rolling](https://arxiv.org/abs/2610.25043v3), [PaceVLA](https://arxiv.org/abs/2610.25961v2), [SPAR](https://arxiv.org/abs/2610.26487v1), [PACT-VLA](https://arxiv.org/abs/2610.28019v1), [Kernel-Variance Heads](https://arxiv.org/abs/2610.29730v1), [FocusVLA](https://arxiv.org/abs/2610.31423v1), [SmolVLA-MoE](https://arxiv.org/abs/2610.30732v1), [One-Step Flow Policies](https://arxiv.org/abs/2610.30735v1)
- **Whole-body and multi-arm coordination (5)**: [CoEvolve](https://arxiv.org/abs/2610.23288v2), [VLA-Fabric](https://arxiv.org/abs/2610.23018v1), [HAF-VLA](https://arxiv.org/abs/2610.26584v1), [Ropedia W0](https://arxiv.org/abs/2610.26855v1), [SP-HLM](https://arxiv.org/abs/2610.30044v1)
- **Manipulation capability and robustness benchmarks (4)**: [LIBERO-Pro2.0](https://arxiv.org/abs/2610.24587v1), [BiDex-100](https://arxiv.org/abs/2610.25459v1), [HandsOn](https://arxiv.org/abs/2610.26765v1), [VLA-Retain](https://arxiv.org/abs/2610.27698v1)
- **Visual task redirection (1)**: [TRACO](https://arxiv.org/abs/2610.23566v1)
- **Manipulation diagnosis and applications (3)**: [RoboPhoto-VLA](https://arxiv.org/abs/2610.26287v1), [Failure Attribution](https://arxiv.org/abs/2610.27277v1), [FlowVLA](https://arxiv.org/abs/2610.29692v1)

</details>

<details markdown="1">
<summary>Flight, driving, control, and medical robotics · C · 15 works</summary>

- **Flight and underwater robotics (4)**: [AeroDASH](https://arxiv.org/abs/2610.25407v1), [AQUAACT](https://arxiv.org/abs/2610.25534v1), [MCTS UAV Planning](https://arxiv.org/abs/2610.25804v1), [KinoUAV](https://arxiv.org/abs/2610.30609v1)
- **Autonomous driving (4)**: [Scene-Adaptive VLA](https://arxiv.org/abs/2610.24590v1), [LECDrive](https://arxiv.org/abs/2610.25734v1), [Perception–Action Binding](https://arxiv.org/abs/2610.28184v1), [Reasoning–Action Coupling](https://arxiv.org/abs/2610.30953v1)
- **Localization, geometric traversal, and control (5)**: [IMU-Aided VLP](https://arxiv.org/abs/2610.23188v1), [Adaptive Traversal Tubes](https://arxiv.org/abs/2610.27688v1), [Integrating Flow Matching and Model P…](https://arxiv.org/abs/2610.29377v1), [SAFTER](https://arxiv.org/abs/2610.29149v1), [Balance–Navigation Coordination](https://arxiv.org/abs/2610.29142v1)
- **Endoscopic navigation (2)**: [EndoVLA-OOV](https://arxiv.org/abs/2610.24747v1), [EBV-CAN](https://arxiv.org/abs/2610.30786v1)

</details>

<details markdown="1">
<summary>Virtual worlds, vision, and other work · C · 9 works</summary>

- **Virtual worlds and game agents (4)**: [Mirror](https://arxiv.org/abs/2610.24177v1), [KPHarness](https://arxiv.org/abs/2610.27417v1), [Scaling in Games](https://arxiv.org/abs/2610.28698v1), [HiSTEVE 1.0](https://arxiv.org/abs/2610.31162v1)
- **Segmentation, video physics, and hand reconstruction (3)**: [UGA](https://arxiv.org/abs/2610.26909v1), [PVBench](https://arxiv.org/abs/2610.26711v1), [DewHand](https://arxiv.org/abs/2610.29936v1)
- **Navigation-video generation (1)**: [Map2Vid](https://arxiv.org/abs/2610.25223v1)
- **Mobile GUI (1)**: [TAG-Nav](https://arxiv.org/abs/2610.22672v1)

</details>
