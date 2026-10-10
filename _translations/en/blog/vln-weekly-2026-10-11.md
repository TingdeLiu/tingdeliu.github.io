---
layout: post
lang: en
translation_id: vln-weekly-2026-10-11
permalink: /en/vln-weekly-2026-10-11/
source_path: _posts/weekly-reports/2026-10-11-VLN-Weekly.md
source_url: /vln-weekly-2026-10-11/
source_revision_date: 2026-10-11
translation_updated: 2026-10-11
title: "Embodied Navigation Weekly (2026-10-02 to 2026-10-08)"
date: 2026-10-11
period_start: 2026-10-02
period_end: 2026-10-08
issue_number: 8
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
excerpt: "This issue adds 147 independent works, including 49 in the automatic primary pool and 3 reporting R2R-CE results. NavGPT-3 reports R2R-CE SR 81.51 and StageVLN reports val-unseen SR 56.3%; PG-VP trades navigation success for risk avoidance. Runtime scheduling, viewpoint-anchored memory, and visual exploration states are key themes, with completion, safety, and deployment costs assessed separately."
---

## 1. Key conclusions
{: id="key-conclusions"}

- **Of 147 newly collected works, 3 explicitly report R2R-CE results, but their numbers measure different things.** [NavGPT-3](https://arxiv.org/abs/2610.10787v1) reports SR of 81.51 / 90.43 on continuous R2R-CE / RxR-CE; [StageVLN](https://arxiv.org/abs/2610.05664v1) reports 56.3% SR and 51.4% SPL on R2R-CE val-unseen. [PG-VP](https://arxiv.org/abs/2610.07558v1)'s 84.9% / 83.2% measure guidance toward intended low-risk actions on these continuous benchmarks, with navigation SR falling by 6.8 / 7.9 percentage points. **Our assessment:** align tasks, splits, and safety objectives before comparing performance. These figures do not form a common leaderboard, and none refers to discrete R2R.
- **A central runtime question is becoming who takes control of motion, and when.** NavGPT-3 schedules separate reasoning, acting, and monitoring threads; [SuperNav](https://arxiv.org/abs/2610.12126v1) connects a generalist model to navigation tools through visual points. In simulations that keep evolving during inference, [RT-SAFE](https://arxiv.org/abs/2610.09294v1) finds 12.3 times as many collisions under real-time execution as in matched static evaluation. **Our assessment:** correct answers still require timely execution, interruption, and recovery; task completion alone does not establish closed-loop reliability.
- **Long-term memory now addresses both where evidence was observed and how it changes the next decision.** [LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1) anchors small-object memory to observation viewpoints and tests revalidation after relocation; [MarvisNav](https://arxiv.org/abs/2610.06510v1) displays exploration states directly on visual route candidates. **Our assessment:** for ground robots, reducing repeated search and correcting stale beliefs are more useful tests than the number of frames a context can hold.
- **Compute reduction must preserve decision-relevant information and be measured in the complete loop.** StageVLN confines geometric auxiliaries to training; [LiteNWM](https://arxiv.org/abs/2610.12368v1) scores candidate trajectories using future latent representations; [LaTraNav](https://arxiv.org/abs/2610.11622v1) couples slow semantics with fast planning. **Our assessment:** training-time structural supervision, inference-time predictive compression, and asynchronous scheduling optimize different parts of a system. RTX 5090 speedups do not directly establish onboard performance.

This issue adds 147 records corresponding to 147 independent arXiv works, published between 2026-10-02 and 2026-10-08. Another 5 existing papers received version updates and are excluded from the new-work count. There are 0 new WeChat articles. Automatic primary-pool assignments comprise 32 navigation and 17 embodied-agent works, totaling 49; the secondary pool comprises 87 VLA/manipulation, 1 autonomous-driving, and 10 other works, totaling 98. Automatic labels contain boundary errors: navigation works such as PG-VP and SpikingVLA fall in the secondary pool, while non-embodied work such as IndexAct enters the primary pool. Reading priorities below follow the actual research content; these counts retain the automatic classification. The categorized index covers all 147 works.

## 2. Priority reading list
{: id="priority-reading"}

1. **[NavGPT-3](https://arxiv.org/abs/2610.10787v1)** · Continuous R2R-CE / RxR-CE and runtime scheduling
   - Contribution: separate reasoning, action, and monitoring contexts with interruption and motion-control handoffs.
   - Evidence: full-system SR of 81.51 / 90.43 in the abstract; corresponding splits are unspecified and require protocol checks before comparison.
   - Why read: combines continuous VLN with an AgentOS-like runtime.
2. **[StageVLN](https://arxiv.org/abs/2610.05664v1)** · Training-time spatial supervision for continuous navigation
   - Contribution: geometry, relative heading, and route-progress supervision are training-only.
   - Evidence: a 4B backbone; R2R-CE val-unseen SR 56.3%, SPL 51.4%; RxR-CE SR 54.3%, with its split unspecified in the abstract.
   - Why read: tests representation improvements without adding deployment modules.
3. **[PG-VP](https://arxiv.org/abs/2610.07558v1)** · Non-visual hazards in continuous navigation
   - Contribution: turns thermal or radiation risk into moving virtual obstacles that steer frozen OmniNav.
   - Evidence: intended low-risk action guidance of 84.9% / 83.2% on R2R-CE / RxR-CE val-unseen, at an SR cost of 6.8 / 7.9 percentage points.
   - Why read: exposes the tradeoff between avoiding risk and reaching the goal; the guidance rate is not SR.
4. **[MarvisNav](https://arxiv.org/abs/2610.06510v1)** · Zero-shot ObjectNav
   - Contribution: projects candidate topological nodes and exploration states onto the current view.
   - Evidence: HM3D SR 81.2%, SPL 42.5%; the abstract reports VLM calls at 7.5% of WMNav's.
   - Why read: a concrete memory interface permits controls that hold information content constant.
5. **[LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1)** · Lifelong search with changing object layouts
   - Contribution: starts with empty memory and uses multi-view inspection and viewpoint-anchored memory to find small objects.
   - Evidence: LiSoNav-Eval has 28 indoor scenes and 45 small-object categories; no verifiable SR / SPL values in the abstract.
   - Why read: assumptions match persistent household use.
6. **[SuperNav](https://arxiv.org/abs/2610.12126v1)** · General requests in unfamiliar environments
   - Contribution: skills, tools, and progress management support sustained navigation without navigation-specific MLLM fine-tuning.
   - Evidence: instance-level, multi-object, demand-driven, HM3D, and quadruped evaluations are reported, without absolute metrics.
   - Why read: useful for comparing how runtimes connect general reasoning to specialized execution.
7. **[LiteNWM](https://arxiv.org/abs/2610.12368v1)** · Candidate-trajectory evaluation for visual navigation
   - Contribution: shared visual encoding and joint multi-horizon future latent prediction for scoring.
   - Evidence: macro-averaged offline trajectory error on RECON / SCAND / SACSoN falls 17.56% relative to NoMaD+NWM-XL, with a 128.00-fold RTX 5090 speedup. Real-robot success rises from 43.3% to 83.3% relative to NoMaD.
   - Why read: check candidate counts, hardware, closed-loop trial scale, and scorer-transfer conditions.

## 3. Featured analysis
{: id="featured-analysis"}

### 1. NavGPT-3: interruption and control ownership as runtime capabilities
{: id="navgpt-3"}

**Problem.** Long reasoning and low-latency motion require different execution rhythms.

**Method.** The [paper](https://arxiv.org/abs/2610.10787v1) gives reasoning, acting, and monitoring threads separate contexts, tools, and permissions, managed by a scheduler.

**Evidence.** The abstract reports minimum reaction time falling from 3–19 seconds per language-model decision to 0.5–1 second per action-policy step. Interpret the SR results above with their specific evaluation splits.

**Value.** **Our assessment:** interruption, context isolation, and execution-feedback interfaces are the transferable components.

**Limits.** Minimum reaction time is not worst-case end-to-end latency; benchmark success does not establish general human-level physical competence.

**Next step.** Inspect motion-control handoffs, monitoring triggers, and recovery logs before choosing a replication scope.

### 2. StageVLN: learn spatial structure during training
{: id="stagevln"}

**Problem.** Action supervision alone may fail to retain geometry, orientation, and global progress.

**Method.** The [paper](https://arxiv.org/abs/2610.05664v1) combines hierarchical supervision from a frozen geometry model with relative-heading and expert-route progress objectives, removing auxiliaries at deployment.

**Evidence.** R2R-CE val-unseen SR / SPL are listed above; the abstract lacks a matched-backbone baseline and training costs.

**Value.** **Our assessment:** suitable when deployment architecture is fixed but training can change.

**Limits.** Removing auxiliary modules neither reduces the original backbone's computation nor makes training free.

**Next step.** Ablate geometry, heading, and progress supervision separately under fixed data and training budgets.

### 3. MarvisNav: make memory part of visual route selection
{: id="marvisnav"}

**Problem.** Separate textual history forces the model to infer how memory corresponds to available routes.

**Method.** The [paper](https://arxiv.org/abs/2610.06510v1) maintains local exploration progress in a topological graph and displays candidate nodes with their states in egocentric views.

**Evidence.** The abstract reports HM3D SR / SPL and fewer calls, plus real-robot validation without a trial count.

**Value.** **Our assessment:** presentation may drive gains, making an interface-level replication useful.

**Limits.** Fewer model calls need not reduce total latency; mapping and projection still cost time.

**Next step.** Hold nodes, exploration information, and model constant while comparing text, a separate map, and image overlays.

### 4. LiSoNav / IVAM-Nav: interpret non-detection through viewing conditions
{: id="lisonav"}

**Problem.** Small objects are occluded and moved; last-known positions cannot support sustained search alone.

**Method.** The [paper](https://arxiv.org/abs/2610.10125v1) inspects supporting surfaces from complementary views and associates memory with observation viewpoints for reuse and revalidation.

**Evidence.** The benchmark separates unchanged and relocated targets and requires empty initial scene memory. Scale is listed above; the abstract gives no performance values.

**Value.** **Our assessment:** search efficiency, visibility, and stale memory are tested together.

**Limits.** The entry cannot establish perception cost, sensitivity to pose error, or real-robot reproducibility.

**Next step.** Record old-location rechecking cost, useful observations, and recovery time after relocation separately.

### 5. SuperNav: inspect how general requests become executable targets
{: id="supernav"}

**Problem.** Navigation-specific fine-tuning may constrain generalization to new requests and environments.

**Method.** The [paper](https://arxiv.org/abs/2610.12126v1) retains a generalist MLLM and adds skills, tools, context management, and a shared visual-point interface.

**Evidence.** The abstract reports gains over four baselines on multiple tasks and quadruped deployment, without comparable absolute values.

**Value.** **Our assessment:** the semantic-to-executable-target interface deserves close inspection.

**Limits.** “Any task in any scene” is the title, not evidence of unlimited generalization.

**Next step.** Test feedback and replanning for unknown objects, ambiguous requests, and unreachable targets, beyond successful demonstrations.

## 4. Transferable methods
{: id="transferable-methods"}

- **Intermittent perception: [ALONE](https://arxiv.org/abs/2610.11591v1).** Propagates spatial beliefs through actions and looks again when reliability is inadequate and observation would help. Success is 98% / 97% in two drone simulation families; median new-depth observation fractions are 0.9% / 1.3% among successful trials only. **Our assessment:** useful for ground robots sharing a camera, but wheeled dynamics, moving obstacles, and observation demand in failed trials require new tests.
- **Compact history: [LightVLN](https://arxiv.org/abs/2610.05024v1).** One token per historical frame and local aggregation reduce inputs. Reported Orin NX 16 GB rates are 14.61 Hz inference and 11.13 Hz end-to-end updates in real-to-sim hardware-in-the-loop evaluation. **Our assessment:** borrow the compression mechanism; this is not evidence of autonomous real-world flight or ground-VLN success.
- **Causal-memory evaluation: [EMBER-Bench](https://arxiv.org/abs/2610.05013v1).** Tests both next-action selection and causal traceback. The best of 16 models reaches 61.2% accuracy versus 98.3% averaged over two human evaluators. **Our assessment:** introduce historical constraints such as a previously closed door or relocated goal in ground navigation, and test whether remembered causes change actions correctly. Question answering is still distinct from closed-loop control.
- **Subtask context: [RobotUse](https://arxiv.org/abs/2610.04929v2).** The backend handles geometry, planning, and control; subtasks retain detail and return decision-relevant information, while execution updates a persistent playbook. RoboLab task success is 45%, 6.7 percentage points above CaP-X. **Our assessment:** a useful pattern for navigation subgoal handoffs, subject to new tests of motion-time latency and continuous-state consistency.

## 5. Categorized index
{: id="categorized-index"}


A = directly relevant to ground navigation; B = a clear transferable mechanism or evaluation contribution; C = low-relevance field observation. Manipulation papers are grouped by theme. Editorial placement does not change automatic pool counts.

### 5.1 Ground VLN / ObjectNav / semantic navigation
{: id="ground-navigation"}


- **[LiteNWM](https://arxiv.org/abs/2610.12368v1)** (A): Scores trajectories through future latents; see priority list.

- **[LaTraNav](https://arxiv.org/abs/2610.11622v1)** (A): Asynchronous slow VLM and fast planner; 6.05-fold faster path updates at the same semantic-update rate in the abstract.

- **[SuperNav](https://arxiv.org/abs/2610.12126v1)** (A): General requests connected to navigation tools; see featured analysis.

- **[TAPNAV](https://arxiv.org/abs/2610.10748v1)** (A): Actively touches structures for localization and route planning when vision is unavailable.

- **[AirGroundVLN](https://arxiv.org/abs/2610.10421v1)** (A): Air–ground goal navigation with cross-view memory and regional-to-local planning.

- **[LiSoNav / IVAM-Nav](https://arxiv.org/abs/2610.10125v1)** (A): Lifelong small-object search under relocation; see featured analysis.

- **[SpikingVLA](https://arxiv.org/abs/2610.09710v1)** (A): Asynchronous spiking inference; navigation SR / SPL are mentioned without a named benchmark, preventing direct comparison.

- **[MixVPR Teach-and-Repeat](https://arxiv.org/abs/2610.09631v1)** (A): MixVPR place recognition reduces hardware demands for teach-and-repeat navigation.

- **[SiGNgapore](https://arxiv.org/abs/2610.09488v2)** (A): Sign-based navigation data from real public spaces for sequential sign-following decisions.

- **[NavGPT-3](https://arxiv.org/abs/2610.10787v1)** (A): Navigation thread scheduling and interruption; see featured analysis.

- **[COOL](https://arxiv.org/abs/2610.09358v1)** (A): Infers ownership from human–object interactions and actively refreshes memory for personalized object finding.

- **[RT-SAFE](https://arxiv.org/abs/2610.09294v1)** (A): Simulation continues during inference, separating completion from safety events.

- **[CUSP](https://arxiv.org/abs/2610.07882v1)** (A): Uses annotated risk onset and accumulated alarms for off-road hazards; intervention time is distinct from hazard onset.

- **[PG-VP](https://arxiv.org/abs/2610.07558v1)** (A): Dynamic prompts for non-visual hazards; see priority list. Automatically labeled VLA/manipulation.

- **[Risk-Sensitive Crowd Navigation / AECP](https://arxiv.org/abs/2610.07474v1)** (A): Directional uncertainty ellipsoids and tail-risk constraints address shifts in pedestrian motion.

- **[Semantic-Aware Humanoid Navigation](https://arxiv.org/abs/2610.07396v1)** (A): Addresses hazards missed by elevation maps and command–execution mismatch; distinguish simulated navigation from real locomotion validation.

- **[Ackermann VLN Sim-to-Real](https://arxiv.org/abs/2610.07192v1)** (A): Continuous VLN transfer to an Ackermann platform; no SPL / nDTW values in the abstract.

- **[MarvisNav](https://arxiv.org/abs/2610.06510v1)** (A): Visual exploration memory; see featured analysis.

- **[Dual-VAE Sim-to-Real](https://arxiv.org/abs/2610.06327v1)** (A): Dual VAEs align simulation and real features; nearly 91% in the abstract is an image-classification result, not closed-loop navigation SR.

- **[JESSI](https://arxiv.org/abs/2610.05733v2)** (A): Joint pedestrian perception and social navigation from LiDAR to executable control.

- **[StageVLN](https://arxiv.org/abs/2610.05664v1)** (A): Training-time spatial and progress supervision; see featured analysis.

- **[Tour-guide Social Navigation](https://arxiv.org/abs/2610.05455v1)** (A): Social-force model for following a tour-guide robot; the abstract primarily describes an experimental design.

- **[TACET](https://arxiv.org/abs/2610.03828v1)** (A): Joint control of personal space and locomotion noise extends social navigation beyond distance keeping.

### 5.2 Memory, maps, planning, and evaluation
{: id="memory-planning-evaluation"}


- **[2DGS-Planner](https://arxiv.org/abs/2610.11752v1)** (B): Rasterization queries traversable geometry in Gaussian maps; a geometric planning component.

- **[Mine Odyssey](https://arxiv.org/abs/2610.11328v1)** (B): Long-horizon spatial exploration in reconstructed Minecraft scenes, distinct from real-robot navigation.

- **[Arena 5.0](https://arxiv.org/abs/2610.11220v1)** (A): ROS2 social-navigation simulation and scenario generation.

- **[SCOPE](https://arxiv.org/abs/2610.12431v1)** (B): Control-ready uncertainty for trajectory diffusion, including crowd-navigation evaluation.

- **[Ctrl-CWM](https://arxiv.org/abs/2610.09438v1)** (B): World-model planning generates crowds that adapt to specified objectives.

- **[iAm.md](https://arxiv.org/abs/2610.10962v1)** (B): Deployment evidence and persistent object records support skill feasibility assessment.

- **[System Switch](https://arxiv.org/abs/2610.09683v1)** (B): Studies when to invoke slow reasoning; no variant exits in closed-loop Doom, so offline gains do not establish task success.

- **[ActiveLang](https://arxiv.org/abs/2610.09518v1)** (B): Selects informative views through semantic uncertainty for active open-vocabulary 3D mapping.

- **[VeriFine](https://arxiv.org/abs/2610.08761v1)** (B): Co-evolves policy, curriculum, and verifier to address limitations in feedback quality.

- **[SPW-Nav](https://arxiv.org/abs/2610.08941v1)** (B): Language-driven streaming panorama generation; video quality is distinct from navigation-policy performance.

- **[Evidence-Driven Human-Agent-Robot Teaming](https://arxiv.org/abs/2610.08933v1)** (B): Evidence collection through authority-bounded services; evidence consists of scenarios and a hardware-in-the-loop prototype.

- **[SpaTime](https://arxiv.org/abs/2610.08713v1)** (B): Causal geometry tokens and response-time supervision support streaming spatial reasoning.

- **[EMHO](https://arxiv.org/abs/2610.08432v1)** (B): Revises a frozen model’s harness using execution traces while managing subtask tradeoffs.

- **[Attacca](https://arxiv.org/abs/2610.07785v1)** (B): Trains continuous search–approach–interaction in Minecraft, retaining the state left by previous tasks.

- **[OntoPlan](https://arxiv.org/abs/2610.07649v1)** (B): Symbolic scenes and action preconditions support long-horizon robot planning; automatically labeled other.

- **[Inspect Robots](https://arxiv.org/abs/2610.06306v1)** (B): Modular infrastructure for defining, executing, and terminating physical evaluations.

- **[Embodied Guardrail Benchmark](https://arxiv.org/abs/2610.06122v1)** (B): Separates defense effectiveness, false positives on benign tasks, and runtime latency.

- **[Direction-Conditioned Policies](https://arxiv.org/abs/2610.05087v1)** (B): Conditions goal policies on representation-space direction and distance; not specific to language navigation.

- **[EMBER-Bench](https://arxiv.org/abs/2610.05013v1)** (B): Joint tests of historical causes and next actions; see transferable methods.

- **[RobotUse](https://arxiv.org/abs/2610.04929v2)** (B): Subtask context and persistent playbooks; see transferable methods.

- **[PreAct-Nav](https://arxiv.org/abs/2610.04916v1)** (A): Persistent subgoals and action-conditioned prediction support urban navigation; no absolute metrics in the abstract.

- **[ROMA](https://arxiv.org/abs/2610.06955v2)** (B): Actively obtains visual, acoustic, tactile, and force evidence through object interactions.

### 5.3 Embodied VLA / mobile manipulation
{: id="vla-manipulation"}


- **Views, geometry, and perception–action interfaces（6）· C**：[VersaCamVLA](https://arxiv.org/abs/2610.12451v1), [WARP-VLA](https://arxiv.org/abs/2610.11508v1), [PAIR](https://arxiv.org/abs/2610.09016v1), [Wiring Matters](https://arxiv.org/abs/2610.06318v1), [GeoBridge-VLA](https://arxiv.org/abs/2610.05026v1), [ExStereo](https://arxiv.org/abs/2610.04805v1).

- **Latent world models and future supervision（9）· C**：[PLaW-VLA](https://arxiv.org/abs/2610.12285v1), [HWAM](https://arxiv.org/abs/2610.12026v1), [ACT3](https://arxiv.org/abs/2610.11416v1), [Juno](https://arxiv.org/abs/2610.09940v1), [RobotAPO](https://arxiv.org/abs/2610.09454v1), [ViDAL](https://arxiv.org/abs/2610.08150v1), [SimForcing](https://arxiv.org/abs/2610.06598v1), [ForeAct3D](https://arxiv.org/abs/2610.04607v1), [SUAVE](https://arxiv.org/abs/2610.04009v1).

- **Training, adaptation, and self-improvement（12）· C**：[HT-Policies](https://arxiv.org/abs/2610.12231v1), [CAPABLE](https://arxiv.org/abs/2610.11971v1), [SQAM](https://arxiv.org/abs/2610.10437v1), [DRIVE](https://arxiv.org/abs/2610.09943v1), [EmbodiedRSI](https://arxiv.org/abs/2610.10498v1), [Robo-COP](https://arxiv.org/abs/2610.09228v1), [VLA-ZO](https://arxiv.org/abs/2610.06271v1), [Data Augmentation in VLA Post-Training](https://arxiv.org/abs/2610.05994v1), [TIGER](https://arxiv.org/abs/2610.07527v1), [ProactiveVLA](https://arxiv.org/abs/2610.06999v1), [RoboIRS](https://arxiv.org/abs/2610.04681v3), [PermVLA](https://arxiv.org/abs/2610.04659v1).

- **Language dependence, instructions, and shortcuts（7）· C**：[NegaAlign](https://arxiv.org/abs/2610.11952v1), [Task Scrubbing](https://arxiv.org/abs/2610.10912v1), [Rephrase Before You Act](https://arxiv.org/abs/2610.10526v1), [Do VLAs Understand Instructions?](https://arxiv.org/abs/2610.10178v1), [YUBI-STAG](https://arxiv.org/abs/2610.09718v1), [Encoded but Not in Control](https://arxiv.org/abs/2610.06235v1), [PerturBot](https://arxiv.org/abs/2610.04616v1).

- **Action representation, timing, and execution（9）· C**：[REACT](https://arxiv.org/abs/2610.12007v1), [PathTime-VLA](https://arxiv.org/abs/2610.11771v1), [RoboPace](https://arxiv.org/abs/2610.09696v1), [TempoBridge](https://arxiv.org/abs/2610.09451v1), [ProAct](https://arxiv.org/abs/2610.09170v1), [StairVLA](https://arxiv.org/abs/2610.07756v2), [ESP](https://arxiv.org/abs/2610.07696v1), [RACE](https://arxiv.org/abs/2610.05719v1), [Vela](https://arxiv.org/abs/2610.05230v1).

- **Skills, memory, routing, and verification（8）· C**：[FLOWMEM](https://arxiv.org/abs/2610.12090v1), [EVIS](https://arxiv.org/abs/2610.11418v1), [RV-ICL](https://arxiv.org/abs/2610.06843v1), [EvoMem-VLA](https://arxiv.org/abs/2610.05418v1), [TUD](https://arxiv.org/abs/2610.05025v1), [DiVeR](https://arxiv.org/abs/2610.04933v1), [Local Predictive Sufficiency](https://arxiv.org/abs/2610.04303v1), [SWAP](https://arxiv.org/abs/2610.06926v1).

- **Pruning, certified acceleration, and system costs（6）· C**：[DIVA](https://arxiv.org/abs/2610.09144v1), [CARE](https://arxiv.org/abs/2610.08917v1), [ActTune](https://arxiv.org/abs/2610.08444v1), [VLA-ACL](https://arxiv.org/abs/2610.08133v2), [SAPrune](https://arxiv.org/abs/2610.05273v1), [Beyond LLM Serving](https://arxiv.org/abs/2610.05062v1).

- **Robustness, monitoring, and recovery（8）· C**：[Ancestor-VLM Patch Attacks](https://arxiv.org/abs/2610.09708v1), [SOUL](https://arxiv.org/abs/2610.09496v1), [TMT](https://arxiv.org/abs/2610.09462v1), [SALT](https://arxiv.org/abs/2610.07946v1), [OGAM](https://arxiv.org/abs/2610.05878v1), [Implied Harm in VLA Instructions](https://arxiv.org/abs/2610.05818v1), [VICS-G](https://arxiv.org/abs/2610.05166v2), [Learned Corrector vs. Simple Retreat](https://arxiv.org/abs/2610.06921v1).

- **Data, mobile manipulation, and long-horizon evaluation（11）· C**：[ManiUnit](https://arxiv.org/abs/2610.12089v1), [RoboQuest](https://arxiv.org/abs/2610.10388v1), [VOMMI](https://arxiv.org/abs/2610.08220v1), [SMART](https://arxiv.org/abs/2610.07652v1), [BiGym 2.0](https://arxiv.org/abs/2610.07594v1), [ACG-Bench / AE-VLA](https://arxiv.org/abs/2610.06184v1), [When Does Retrieval Help?](https://arxiv.org/abs/2610.05492v1), [RMMBench](https://arxiv.org/abs/2610.05414v1), [ArticuTable](https://arxiv.org/abs/2610.05249v1), [Grounded in Time](https://arxiv.org/abs/2610.04255v1), [Hybrid Flow Task-and-Motion Planning](https://arxiv.org/abs/2610.04771v1).

- **Contact, touch, and close-range exploration（4）· C**：[OpenViTac](https://arxiv.org/abs/2610.10384v1), [MIM-VLA](https://arxiv.org/abs/2610.08425v1), [Reactive Exploration with Virtual Model Control](https://arxiv.org/abs/2610.08110v1), [AgenticTactileVLA](https://arxiv.org/abs/2610.04391v1).

### 5.4 Aerial navigation, driving, and other lower-relevance directions
{: id="other-directions"}


- **[ALONE](https://arxiv.org/abs/2610.11591v1)** (B): Intermittent drone perception; see transferable methods.

- **[TGIT](https://arxiv.org/abs/2610.10635v1)** (B): Instruction translation before a frozen aerial navigator; the language interface needs ground-task validation.

- **[Sensor-Layout-Agnostic Navigation](https://arxiv.org/abs/2610.08306v1)** (B): Canonicalizes depth-camera layouts and explicitly marks blind spots; evidence is from an aerial platform.

- **[LightVLN](https://arxiv.org/abs/2610.05024v1)** (B): Compact aerial-navigation history and hardware-in-the-loop efficiency; see transferable methods.

- **[WAND](https://arxiv.org/abs/2610.11809v1)** (C): Quadrotor wind rejection and obstacle control without language or semantic tasks.

- **[WareFly-VLA](https://arxiv.org/abs/2610.08526v1)** (C): Language-conditioned worker search and tracking by warehouse drones.

- **[Learning minimum-time navigation policies in two-dimensional flows with a genetic algorithm](https://arxiv.org/abs/2610.12177v1)** (C): Minimum-time control in known two-dimensional flows, without language or semantic navigation.

- **[Visual Swarm Navigation](https://arxiv.org/abs/2610.06400v1)** (C): Visual swarm exploration and control without language tasks.

- **[EM Digital Twin Calibration](https://arxiv.org/abs/2610.07081v2)** (C): Robot measurement paths calibrate an electromagnetic digital twin; the objective is parameter estimation.

- **[Visual Prosthesis Navigation](https://arxiv.org/abs/2610.05772v1)** (C): Human navigation assistance for visual prostheses, with a different task and user population.

- **Driving and road generation（3）· C**：[GeoCoTDrive](https://arxiv.org/abs/2610.10390v1), [Odyssey](https://arxiv.org/abs/2610.06469v1), [Controllable Road Marking Generation](https://arxiv.org/abs/2610.05771v1).

- **Visual understanding, generation, and game interaction（4）· C**：[ContourVLA](https://arxiv.org/abs/2610.12107v1), [WorldBench](https://arxiv.org/abs/2610.10622v1), [Humanity's Sixth Sense](https://arxiv.org/abs/2610.08966v2), [PlaySuite](https://arxiv.org/abs/2610.07127v1).

- **Non-embodied retrieval noise: archive links only（5）· C**：[DRL-DCO](https://arxiv.org/abs/2610.11546v1), [Pelvic Fractures Technology Review](https://arxiv.org/abs/2610.09884v1), [Energy Transition Investment RL](https://arxiv.org/abs/2610.10768v1), [Agentic RCA / E4](https://arxiv.org/abs/2610.08622v1), [IndexAct](https://arxiv.org/abs/2610.07960v1).

### 5.5 News and non-paper items
{: id="news"}


- No new WeChat articles or separate news items; all entries above are papers.

## 6. Trends and actions
{: id="trends-actions"}

### Trends
{: id="trends"}

- **Runtimes are becoming independently testable research objects.** NavGPT-3, SuperNav, and [EMHO](https://arxiv.org/abs/2610.08432v1) examine scheduling, execution interfaces, and experience-driven harness revision respectively. **Our assessment:** system organization needs separate ablations under the same underlying model; improvements cannot all be attributed to model capability.
- **Observation, memory, and action need a verifiable causal chain.** IVAM-Nav, MarvisNav, and EMBER-Bench address visibility, spatial presentation, and historical constraints. **Our assessment:** ask which decision memory changed, and whether that change was correct, beyond retrieval and answering scores.
- **Efficiency must be reported alongside safety and completion.** The 49 automatically assigned primary works are outnumbered by 98 secondary works; manipulation speedups do not directly establish navigation gains. LiteNWM, LaTraNav, and RT-SAFE motivate measuring complete-loop latency, observation freshness, completion, and collisions together.

### Research gaps
{: id="research-gaps"}

- This issue offers no directly aligned continuous-VLN comparison covering splits, observations, interventions, compute budgets, and dynamic environments. Minimum response time especially must not be treated as a worst-case guarantee.
- The abstracts do not jointly validate viewpoint-anchored memory and visual route memory under simultaneous pose drift, object relocation, and perception errors.
- Failures caused by risk avoidance need evaluation beyond SR to distinguish reasonable refusal, necessary detours, and policy degradation.

### Recommended actions
{: id="recommended-actions"}

**High priority**

- **Read NavGPT-3 and StageVLN closely.** Check R2R-CE / RxR-CE splits, training data, inputs, and interruption mechanisms before selecting baselines; do not mix in discrete R2R results.
- **Replicate MarvisNav's memory-presentation controls.** Fix the underlying model and information content; measure revisits, search paths, and actual total latency.
- **Track LiSoNav data and code.** Start with empty memory and separately test search efficiency for relocated and unchanged targets.

**Medium priority**

- **Evaluate PG-VP's safety–success tradeoff.** Keep low-risk action guidance and navigation SR as separate metrics; do not hide costs in a single weighted score.
- **Inspect deployment interfaces in SuperNav, LiteNWM, and LaTraNav.** Measure end-to-end latency, observation-to-execution delay, and recovery on the target hardware.
- **Add continuous-world evaluation.** Following RT-SAFE, keep pedestrians and obstacles moving during inference and retain matched static controls.

**Low priority**

- **Defer importing manipulation scores into navigation leaderboards.** Transfer aerial navigation, game agents, tabletop manipulation, and driving only with explicit task mappings and validation protocols. Track further materials for work lacking closed-loop metrics.
