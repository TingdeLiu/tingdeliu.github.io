---
layout: post
lang: en
translation_id: vln-weekly-2026-08-01
permalink: /en/vln-weekly-2026-08-01/
source_path: _posts/weekly-reports/2026-08-01-VLN-Weekly.md
source_url: /vln-weekly-2026-08-01/
source_revision_date: 2026-09-27
translation_updated: 2026-10-03
title: "Embodied Navigation Weekly (2026-07-22 to 2026-08-01)"
date: 2026-08-01
period_start: 2026-07-22
period_end: 2026-08-01
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: false
author: Tingde Liu
toc: true
published: false
excerpt: "A Tongji University survey in IEEE TASE systematically quantifies the sim-to-real gap: success falls from 66% to 51% for hierarchical methods and from 61% to 22% for end-to-end methods. EA-Nav and X-NavDP make embodiment geometry an explicit conditioning variable. MemVLN and BrainNav compress long histories without token-by-token autoregressive decoding, while CheckVLA and RoboBRIDGE illustrate runtime verification as a component separate from the main policy."
---

* Contents
{:toc}

## 1. Key conclusions
{: id="一本期结论"}

- **The performance gap between simulation and real life has been quantified systematically for the first time, and the hierarchical paradigm is significantly more stable on real robots**: Tongji University's VLN review published in IEEE TASE built a real-robot platform and conducted head-to-head evaluations in 10 real scenarios. The success rate of the hierarchical method dropped from 66% in simulation to 51%, and the end-to-end method dropped from about 61% down to 22%. This is the only work in this issue that provides real robot direct comparison evidence, and it also directly affects the subsequent technical route selection: the success rate of the hierarchical method in real scenarios is more than 2 times that of end-to-end, and the collision rate is about 1/7 (the above are clearly stated by the source).
- **" embodiment geometry " changes from implicit assumptions to explicit condition variables**: EA-Nav uses the robot's length, width, height and maximum spanning height as conditional token injection strategies, and X-NavDP uses FiLM morphological modulation to adapt a single diffusion policy to wheeled/quadruped/humanoid. Both works put on the table the action ambiguity problem of "the same observation corresponds to different optimal actions for different embodiments". This digest's assessment: The bottleneck of cross-embodiment navigation is shifting from data scale to conditional modeling.
- **Compression evolution of real-time decoding and long-horizon memory**: MemVLN uses pyramid resolution contextual memory plus procedural memory of intermediate atomic actions, and BrainNav performs low-rank world model compression based on the minimalist sufficiency principle. Both bypass LLM token-by-token autoregressive decoding, and MemVLN reports reaching 14 FPS.
- **The runtime verification and closed-loop error correction architecture is becoming mature**: In response to the error accumulation of open-loop action chunking, CheckVLA uses the action-conditioned world model to verify the action chunks, RoboBRIDGE and Pigey use the hierarchical orchestrator to asynchronously detect faults and targeted replanning, and the test-time scaling framework on the UAV side uses "generate candidates - self-review - safety priority scoring" to achieve zero training error correction. This digest's assessment: Plug-in runtime verification is becoming a standard component independent of the main policy.

## 2. Priority reading list
{: id="二优先阅读清单"}

- **A1 · [VLN overview and real-world system evaluation ](https://ieeexplore.ieee.org/abstract/document/11600818)** · real robot VLN system evaluation (Tongji University, IEEE TASE 2026-07-09)
  - Contribution: Classify existing methods according to the two orthogonal dimensions of "action paradigm × model paradigm", and conduct head-to-head actual measurements of hierarchical and end-to-end routes in 10 real scenes.
  - Evidence: The SR of the hierarchical method is 66%→51%, and the end-to-end is 61%→22%; the strict success rate (SSR) of both methods is more than 10 percentage points lower than the conventional SR; under the backtracking instruction, the hierarchical SR drops to 0%, and the end-to-end is still 20%.
  - Reason: The article in this issue has the highest value for judging the implementation of ground-based VLN: it provides a quantitative basis that cannot be directly extrapolated from simulation indicators, and the failure mode analysis can directly guide the selection.
- **A2 · [X-NavDP](https://arxiv.org/abs/2607.28560)** · cross-embodiment visual navigation (wheeled/quadruped/humanoid)
  - Contribution: Proposed a diffusion policy RL post-training framework for GQRM group Q-value weighted matching, combined with self-guided trajectory perturbation exploration and FiLM morphological modulation.
  - Evidence: The simulation SR increased from 61.20% to 84.28%, and the real difficult scene increased from 10% to 65%; compared with the native DPMD (global Q normalization) overall SR 71.57%, group normalization brought an improvement of nearly 9 percentage points.
  - Reason: It is directly oriented to ground robots, and provides a complete and reproducible path of the diffusion navigation policy "imitation pre-training + RL post-training", and the code has been open source.
- **A3 · [MemVLN](https://arxiv.org/abs/2607.23504v1)** · Indoor Continuous Environment Navigation (VLN-CE)
  - Contribution: Pyramid-resolution contextual memory + procedural atomic action memory with bypass autoregressive decoding.
  - Evidence: Compared with the Qwen3-VL-4B baseline on R2R-CE and RxR-CE, the SR is increased by 5.8% and 9.7% respectively, and the inference speed is accelerated by 7 times, reaching 14 FPS.
  - Reason: Solving the contradiction between long history preservation and LLM decoding delay is a key reference for low-latency navigation at the edge.
- **A4 · [EA-Nav](https://arxiv.org/abs/2607.19880v2)** · cross-embodiment safe visual navigation (Sweet potato robot × Zhejiang University, ACM MM 2026)
  - Contribution: Use geometry as a conditional token to eliminate action ambiguity; decouple the three modules of "where to go/how risky/how to hide", and use trajectory augmentation to construct high-risk samples to train risk correction.
  - Evidence: Grab about 1,000 hours of first-person video from the Internet, covering 8 types of mobile subjects to build cross-embodiment pre-training data; use Unitree Go2 and TurtleBot4 for real robot verification, and set three embodiment specifications of original/Body+/Body++.
  - Reason: In contrast to X-NavDP: it also handles cross-embodiment, but takes the imitation learning route and does not rely on large-scale interaction and reward design.
- **A5 · [BrainNav](https://arxiv.org/abs/2607.23181v1)** · Indoor Continuous Environment Navigation (VLN-CE)
  - Contribution: Based on the principle of minimalist sufficiency, the logical anchor point model is used to filter noise, and the low-rank compressed world model is used to predict action condition states.
  - Evidence: SOTA 2.0% SR / 1.0% SPL improvement on R2R-CE val-unseen, 0.94% SR / 0.78% SPL improvement on RxR-CE.
  - Reason: Provides actionable design principles for extracting minimalist navigation representations on mobile platforms.

> Note: A total of 8 WeChat public account entries and more than 40 arXiv entries were included in this issue. After cross-source merging, the respective WeChat public account interpretations of EA-Nav, X-NavDP, and Self-in-Space point to the same work as the original arXiv text, and are only analyzed once.

## 3. Analysis of highlighted work
{: id="三重点工作分析"}

### 1. VLN survey and real-world evaluation: quantifying the sim-to-real gap
{: id="1-vln-综述与真实世界系统评测量化仿真到现实的落差"}

**Problem.** The VLN field relies heavily on simulation verification, and the optimal success rate of classic benchmarks has exceeded 85%. However, the horizontal performance of the algorithm in the real world lacks systematic comparison, and the gap between simulation and reality has not been quantified.

**Method.** Two main lines: (1) Establish a four-category framework from the two orthogonal dimensions of "action paradigm" (layered vs end-to-end) and "model paradigm" (discriminative small model vs generative large model); (2) build a standardized physical robot platform and evaluate two mainstream routes in 10 indoor and outdoor real scenes. Instruction types are based on standard step-by-step 70%, intention fuzzy 20%, and backtracking 10% ratio, and add implementation-oriented indicators such as strict success rate (SSR) in addition to SR and Oracle SR.

**Evidence.** The source clearly states: The hierarchical representation method CLASH simulates SR 66%, and the real robot drops to 51% (a drop of 15 percentage points); the end-to-end representation method JanusVLN simulates about 61%, and the real robot drops to 22%. The hierarchical SR in real scenes is more than 2 times that of end-to-end, and the collision rate is about 1/7. The SSR of both methods is more than 10 percentage points lower than the conventional SR. The stratification rate is 35% and end-to-end 25% for fuzzy instructions; the stratification rate is 0% and end-to-end 20% for backtracking instructions. The paper statistics show that 51.7% of existing VLN methods belong to the hierarchical paradigm.

**Value.** provides the only real robot comparison data in this issue that can be used for route selection. The systematic gap between SSR and conventional SR shows that the current algorithm only relies on distance thresholds to determine stopping and cannot understand the semantic stopping conditions in instructions. This is a commonly underestimated source of failure.

**Limitations.** Only one representative method (CLASH/JanusVLN) is selected for each type of route, and the sample size is limited. The conclusion should not be extrapolated to "the hierarchical paradigm is overall better than end-to-end". Backtracking instructions account for only 10%, and the hierarchical approach's 0% success rate is based on a smaller sample.

**Recommendation.** Read the analysis of its failure reasons and the discussion in four directions (memory, reasoning, security, data); incorporate semantic stopping criteria such as SSR into our own evaluation indicators, and don’t just report SR.

### 2. X-NavDP: grouped Q-weighted matching for RL post-training of diffusion navigation policies
{: id="2-x-navdp用分组-q-值加权匹配稳定扩散导航策略的-rl-后训练"}

**Problem.** Diffusion visual navigation (NavDP, NoMaD, etc.) relies on imitating global planning expert trajectories, which has two shortcomings: there is an omniscient global planner during pre-training but only local RGB-D during deployment, resulting in the inability to autonomously escape when encountering dead ends and long obstacles; the dataset does not distinguish the dynamics and size constraints of wheeled/quadruped/humanoid, and the stability decreases when migrating across aircraft models. Among the existing RL fine-tuning solutions, DPPO/DDPO causes gradient oscillation due to the likelihood calculation of the complete denoising chain, DSRL freezes the backbone to limit the exploration space, and DPMD performs global Q normalization on the entire batch, making it impossible to obtain effective gradients in difficult scenarios.

**Method.** four components: (1) **GQRM group Q Weighted matching** - Multiple candidate trajectories sampled under the same observation state are grouped separately to calculate the mean variance. Even if the overall return of a group of trajectories is low, the differences within the group can still be explored; (2) **self-guided structure Trajectory perturbation** - Reuse the target-free conditional branch of the pre-trained model to generate an exploration trajectory, mix it with the target-oriented trajectory and randomly flip the coordinates to replace the Gaussian noise that destroys smoothness; (3) **FiLM Morphological modulation** - only adds a few parameters, uses robot-specific embedded modulation features, and a single model adapts to three types of embodiments; (4) 500+ robot parallel simulation training pipeline based on IsaacLab, combined with MPC hierarchical control and RTC inference timing constraints.

**Evidence.** The source of clearly states: Simulation SR 61.20% → 84.28%, real difficult scene 10% → 65%, and the whole post-training process takes about 12 hours. The real robot includes TurtleBot wheeled, Unitree Go2 quadruped, and Unitree G1 humanoid. It is repeated 10 times in three types of scenarios: laboratory/hall/office. The indicators are SR (reaching the target within 0.5 m) and SPL. The baseline includes iPlanner/ViPlanner/NavDP/DPPO/DSRL/DPMD. ablation: The overall SR of native DPMD is 71.57%, and GQRM brings an improvement of nearly 9 percentage points; the training collapses after removing target disturbance and coordinate flipping, and the overall SR is only 6.23%; compared with direct splicing and embedding tokens, the overall SR of FiLM is improved by 1.44, and the humanoid home scene is improved by 4 percentage points; turning on RTC constraint improves the overall SR by 2.11.

**Value.** ablation data is complete, especially "the SR dropped to 6.23% after removing the perturbation" clearly shows that the exploration strategy rather than the loss function is the key prerequisite for training convergence. The idea of group normalization is applicable to any scenario where "simple samples dominate the gradient", not limited to navigation.

**Limitations.** The real scene is repeated only 10 times per group. The comparison of 10%→65% is based on a small sample and the confidence interval is unknown. The overall conclusion relies on IsaacLab simulation training. How the domain gap from simulation to real robot is absorbed by RL post-training has not been dismantled in public materials.

**Recommendation.** gives priority to reproducing the two parts of GQRM loss and self-guided disturbance (the code has been open source), and first verifies on a single embodiment whether the escape behavior is really caused by group normalization.

### 3. EA-Nav: conditioning safe navigation on embodiment geometry
{: id="3-ea-nav把本体几何作为显式条件变量的安全导航"}

**Problem.** Most visual navigation models treat the robot as a volumeless particle by default, and learn the "picture → action" mapping instead of "picture + body parameters → adaptive action". The paper defines this flaw as **action ambiguity**: In the same door opening screen, the small robot corresponds to "go straight" and the large robot corresponds to "turn right". If the training set mixes trajectories of multiple sizes, the model can only fit compromise routes, and boundary scenes are prone to collision.

**Method.** A multi-stage modular framework that imitates the learning route: in the pre-training stage, about 1000 hours of first-person video are captured from the Internet (covering 8 types of moving subjects such as people, cars, bicycles, cats and dogs) to build a cross-embodiment dataset, and the embodiment geometry (length, width, height, maximum spanable height) is used as a conditional token Inject to resolve ambiguity; in the fine-tuning phase, multi-modal information is injected based on the decoupled architecture, and high-risk samples are generated using trajectory augmentation (scaling and rotation transformation of the safe trajectory and then imported into occupied grid screening), and the two modules of spatial perception and risk correction are trained separately to avoid trajectory distortion caused by collision loss and direct conflict with the imitation target.

**Evidence.** The source clearly states: ACM MM 2026 accepted; The real robot experiment uses Unitree Go2 and TurtleBot4, and sets three embodiment configurations: original size, Body+ (medium expansion), and Body++ (maximum expansion). **Insufficient evidence**: Neither the WeChat public account interpretation nor the arXiv abstract provides quantitative comparison results such as success rate and collision rate. The sentence "High-risk sample identification efficiency is improved by about __" was truncated in the crawled text. The specific value needs to be checked in the original text.

**Value.** and X-NavDP compare two solutions to the same problem: EA-Nav uses imitation learning, clearly pointing out that the RL route requires large-scale interaction and fine reward design, and is difficult to support scalable pre-training. The cost of its data acquisition idea of "using first-person Internet videos for cross-embodiment pre-training" is significantly lower than real robot acquisition.

**Limitations.** lacks public quantification results and cannot determine the actual gain relative to the baseline. The dynamics of the 8 types of mobile subjects covered by Internet videos are quite different from those of robot embodiments. Whether geometric conditions can bridge this gap, there is no targeted ablation in the material.

**Recommendation.** directly reads the arXiv original text to complete the experimental data ([2607.19880v2](https://arxiv.org/abs/2607.19880v2)); the injection method of embodiment geometry token can be directly compared with the FiLM modulation of X-NavDP.

### 4. IMPRINT: enriching semantic maps with web images for long-tail ObjectNav
{: id="4-imprint利用网络图像富化语义地图的长尾-objectnav"}

**Problem.** Zero-shot ObjectNav based on pre-trained VLM usually relies on plain text queries. When faced with fine-grained long-tail targets with strong specificity, the reliability of text-visual similarity matching drops significantly.

**Method.** Plug-and-play framework: Automatically obtain multiple reference images of the target object from the network, encode them with VLM and calculate similarity with local observations on a semantic map, and aggregate multi-view features to form a context-aware positioning response without modifying the underlying motion or mapping strategy.

**Evidence.** built the long-tail evaluation benchmark HSSD-rare based on the Habitat scene; on OVON and HSSD-rare, image enrichment query brings improved semantic alignment accuracy and translates into improved end-to-end navigation success rate (the abstract does not give specific values).

**Value.** zero-shot is a lightweight adaptation route for "enriching text semantics to image multi-modality" in open vocabulary navigation, which can be integrated into the existing mapping pipeline without changing the strategy.

**Limitations.** The gain depends on the quality of the underlying semantic map and the accuracy of the target detector; if the network retrieval image contains noise, the similarity map may suffer from multi-modal confusion.

**Recommendation.** introduces multi-image enhanced positioning offline into the existing semantic mapping pipeline to test the success rate of long-tail target search.

### 5. TEA-AgriVLN: traversability alerts in unstructured agricultural environments
{: id="5-tea-agrivln非结构化农业场景中的可通行性估计警报"}

**Problem.** VLN-CE is mainly verified indoors with clear traversable boundaries; in unstructured scenes such as farmland and orchards, the traffic boundaries are blurred (immature crops have different definitions of physical traversability for different entities), which can easily lead to collisions or accidental entry into dangerous situations.

**Method.** Trafficability Estimation Alert (TEA) module: Online estimates the traversability probability map of the field of view image, performs spatial alignment verification on the navigation network output action and the map, and triggers a Rethinking alarm when the action points to a low traffic probability area, forcing the decision-making network to make secondary corrections.

**Evidence.** On the agricultural VLN-CE benchmark A2A, the SR is improved from 0.47 to 0.54, and the average navigation error (NE) is reduced from 2.91 m to 2.70 m.

**Value.** A lightweight defensive filtering framework that inserts geometric/semantic safety guardrails at the action output, suitable for unstructured field operations.

**Limitations.** Accessibility estimation itself is affected by illumination, occlusion and crop growth cycle; a high false alarm rate will cause the decision-making layer to fall into frequent secondary error correction cycles.

**Recommendation.** draws on the design of "traversability verification + error correction loop before control output" as a dynamic obstacle avoidance backbone for ground navigation strategies.

## 4. Transferable methods
{: id="四可迁移方法"}

- **Drone VLN (zoomed during testing): [No Training, Better Flights](https://arxiv.org/abs/2607.19288)**
  - Mechanism: Three-stage framework of pure inference phase: batch generation of multiple candidate routes → model self-review and correction → safety priority scoring to select the optimal route. Without fine-tuning the weights, the reasoning depth of the serial thinking chain and the search breadth of parallel sampling are combined. The source states that it has refreshed SOTA on three types of test sets: seen scenes, new objects, and unfamiliar maps, and the higher the computing power investment, the higher the success rate.
  - Integration point: The high-level waypoint decision-making link of ground-based VLN, especially low-speed or stop-and-go tasks that can tolerate additional inference delays.
  - Prerequisites and risks: The scaling during testing converts reasoning computing power into power, which directly conflicts with the frame rate requirements of real-time ground control; the delay budget for candidate generation and review needs to be quantified first.
- **Fault Importance Sampling (Reinforcement Learning): [Self-Evolving Learning with Criticality Model](https://arxiv.org/abs/2607.28251v1)**
  - Mechanism: The state-level criticality model predicts future failure probability, guides the sampler to tilt toward high-risk states, and improves data information density.
  - Integration point: Difficult sample mining in the RL fine-tuning phase of ground navigation strategies (such as foot-based complex terrain).
  - Prerequisites and risks: A high-precision fault predictor needs to be learned online, otherwise severe sampling bias will destroy the normal action distribution.
- **Multiple verification and error correction during operation: [CheckVLA](https://arxiv.org/abs/2607.26789v1)**
  - Mechanism: Independently frozen action-conditioned world model, predicts the visual evolution trajectory after the execution of the open-loop action chunk, and rewrites the remaining actions when the deviation exceeds the standard.
  - Access position: Open-loop action monitoring for long-distance navigation prevents cumulative drift caused by sensor noise or physical slippage.
  - Prerequisites and risks: The world model needs to run on the onboard device with a sufficiently low overhead and cannot block the main control loop.
- **Asynchronous calculation and future state prediction: [FutureRTC](https://arxiv.org/abs/2607.24008v1)**
  - Mechanism: Use motion physics priors to forward predict visual features and proprioception at the next moment, and align execution bias caused by calculation delays.
  - Access position: Higher speed (>1 m/s) ground obstacle avoidance navigation to compensate for incoherent actions caused by high-latency reasoning of large models.
  - Prerequisites and risks: The prediction module overhead must be less than the delay difference of a single inference of the navigation model.

## 5. Research roundup by category
{: id="五分类速览"}

The relevance of each label: A is for direct research on ground navigation, B has a transferable mechanism, and C is only for field observation.

### 5.1 Ground-based VLN / ObjectNav / Semantic Navigation
{: id="51-地面-vln--objectnav--语义导航"}

- **[VLN overview + real robot evaluation ](https://ieeexplore.ieee.org/abstract/document/11600818)** (real scene system evaluation · A1): See key analysis. [Source](https://ieeexplore.ieee.org/abstract/document/11600818)
- **[X-NavDP](https://arxiv.org/abs/2607.28560)** (cross-embodiment diffusion policy RL post-training · A1): See the key analysis. ([arXiv](https://arxiv.org/abs/2607.28560)/[Project Page](https://yty-sky.github.io/x-navdp-project-page/)/[Code ](https://github.com/InternRobotics/NavDP/tree/master/baselines/x-navdp))
- **[MemVLN](https://arxiv.org/abs/2607.23504v1)** (VLN-CE real-time reasoning · A1): See key analysis. [Source](https://arxiv.org/abs/2607.23504v1)
- **[EA-Nav](https://arxiv.org/abs/2607.19880v2)** (embodied geometry-aware safe navigation · A1): See key analysis. [Source](https://arxiv.org/abs/2607.19880v2)
- **[BrainNav](https://arxiv.org/abs/2607.23181v1)** (minimalist and fully characterized · A1): See key analysis. [Source](https://arxiv.org/abs/2607.23181v1)
- **[IMPRINT](https://arxiv.org/abs/2607.25106v1)** (long tail ObjectNav · A1): See key analysis. [Source](https://arxiv.org/abs/2607.25106v1)
- **[TEA-AgriVLN](https://arxiv.org/abs/2607.28474v1)** (Agricultural Environment VLN-CE · A1): See key analysis. [Source](https://arxiv.org/abs/2607.28474v1)
- **[BioVLN](https://arxiv.org/abs/2607.26914v1)** (Laboratory Multi-Constraint Navigation · A): Introducing a simulation platform to represent the target as a combination of physical entities, obstacle avoidance safety zones and operating zones, including 47 scenes and 1667 Episodes. [Source](https://arxiv.org/abs/2607.26914v1)
- **[Offline Outdoor VLN](https://arxiv.org/abs/2607.22226v1)** (Offline Outdoor Navigation · A): Evaluate the instruction decomposition performance of 17 small edge models, and propose a lightweight hybrid semantic-geometric target positioning framework. [Source](https://arxiv.org/abs/2607.22226v1)
- **HiMemVLN** (Bionic Hippocampus Memory Navigation · To be verified): WeChat public account title states that it proposes a bionic hippocampus memory system to solve the problem of lost navigation loops. **failed to fetch the main text. There is no abstract and no text link. The content of** has not been verified. It cannot be determined whether it has the same origin as MemVLN. [Source](https://mp.weixin.qq.com/s/TeOdNaHDqteluQTqXtxsnA)

### 5.2 Memory, Maps, Planning and Evaluation
{: id="52-记忆地图规划与评测"}

- **[RoboBRIDGE](https://arxiv.org/abs/2607.27881v1)** (VLA Robust Deployment Framework·B): Multi-module coordination layer, which improves real robot deployment fault tolerance through multi-level fault detection and asynchronous re-planning. [Source](https://arxiv.org/abs/2607.27881v1)
- **[CheckVLA](https://arxiv.org/abs/2607.26789v1)** (runtime check·B): See migration method. Verify the long-view open-loop action chunk based on the action-conditioned world model, combined with conformal verification to trigger corrections. [Source](https://arxiv.org/abs/2607.26789v1)
- **[FutureRTC](https://arxiv.org/abs/2607.24008v1)** (Asynchronous action execution alignment · B): See migration method. Use motion prior to predict the next moment of observation and correct the state to solve the incoherence of asynchronous control trajectories. [Source](https://arxiv.org/abs/2607.24008v1)
- **[Pigey](https://arxiv.org/abs/2607.21725v1)** (Physical Agent Orchestration · B): The closed-loop orchestrator is responsible for high-level task decomposition, low-level policy issuance and runtime result verification, without additional training. [Source](https://arxiv.org/abs/2607.21725v1)
- **[IDR](https://arxiv.org/abs/2607.25516v1)** (Test Period Modal Adaptation · B): Infer-Diagnose-Refine framework that diagnoses dynamic visual dependencies with counterfactual visual zero-filling interventions and training-free corrective actions. [Source](https://arxiv.org/abs/2607.25516v1)
- **[HMP](https://arxiv.org/abs/2607.24083v1)** (Hybrid Motion Prior · B): The imitation learning action prior is distilled into the RVQ discrete codebook, and the high-level strategy directly performs point goal navigation by selecting the codebook action. [Source](https://arxiv.org/abs/2607.24083v1)

### 5.3 Embodied VLA / mobile manipulation
{: id="53-具身-vla--移动操作"}

- **[TurboVLA](https://arxiv.org/abs/2607.27205v1)** (real-time VLA model · B): removes the LLM intermediary, uses V+L→A direct feature interaction, infers at 32 Hz on 4090, and occupies 0.9 GB of video memory. [Source](https://arxiv.org/abs/2607.27205v1)
- **[CoTinyVLA](https://arxiv.org/abs/2607.25487v1)** (Lightweight VLA Distillation · B): 0.9B parametric VLA, dual-view temporal input + multi-stage CoT distillation, outperforming 7B model on LIBERO-Plus. [Source](https://arxiv.org/abs/2607.25487v1)
- **[Self-Evolving Learning](https://arxiv.org/abs/2607.28251v1)** (embodied reinforcement fine-tuning · B): See the transferable method. Use state criticality model to do importance sampling for high failure rate scenarios. [Source](https://arxiv.org/abs/2607.28251v1)
- **[ACE-Data-0](https://arxiv.org/abs/2607.28625v1)** (Embodied Data Engine · C): Supports desktop-level fine operations and room-level whole-limb motion collection, providing 150 hours of multi-modal alignment data. [Source](https://arxiv.org/abs/2607.28625v1)
- **[RedFlow](https://arxiv.org/abs/2607.27782v1)** (failure redirection fine-tuning · C): Flow matching VLA that redirects trajectory-level failures to action-level error correction supervision through context-aware error correction matching. [Source](https://arxiv.org/abs/2607.27782v1)
- **[Cross-Embodiment VLA](https://arxiv.org/abs/2607.27549v1)** (cross-embodiment operation migration · C): Evaluate the cross-embodiment migration performance of bounding boxes, motion descriptions, terminal trajectories and other representations, and the terminal trajectory is the best. [Source](https://arxiv.org/abs/2607.27549v1)
- **[DLAM](https://arxiv.org/abs/2607.27138v1)** (Video Implicit Action Extraction · C): A distributed implicit action model based on diagonal Gaussians, using label-free video reconstruction and reversible constraint learning transformation operators. [Source](https://arxiv.org/abs/2607.27138v1)
- **[Concept Expert](https://arxiv.org/abs/2607.26513v1)** (3D Kinetic Guidance VLA · C): Use 3D visual large models to estimate object kinematic parameters, and dynamically track during inference to provide dense rewards and spatial guidance. [Source](https://arxiv.org/abs/2607.26513v1)
- **[CG-World](https://arxiv.org/abs/2607.26452v1)** (CG dataset and protocol · C): Generate 850,000 segments of world model data containing bones, materials, motion curves and counterfactual intervention branches from the industrial 3D animation pipeline. [Source](https://arxiv.org/abs/2607.26452v1)
- **[SAM3D-Alignment](https://arxiv.org/abs/2607.25912v1)** (3D representation alignment VLA · C): Use SAM3D to distill the 3D target prior during training, and improve the success rate under occlusion and perspective changes without adding 3D input during inference. [Source](https://arxiv.org/abs/2607.25912v1)
- **[HiFi-UMI](https://arxiv.org/abs/2607.25895v1)** (no real robot data collection · C): Portable binocular wide-angle head-mounted UMI sensor, which enables data training without physical robot anchoring by improving hand tracking accuracy. [Source](https://arxiv.org/abs/2607.25895v1)
- **[Data Pyramid](https://arxiv.org/abs/2607.24744v1)** (Review of Embodied Data · C): Combining and operating the embodied data ecology from the five dimensions of real robot, UMI, self/other perspective video, simulation, and graphic Internet. [Source](https://arxiv.org/abs/2607.24744v1)
- **[τ Touch VLA](https://arxiv.org/abs/2607.24485v2)** (Tactile Augmentation VLA·C): Supervise tactile features with future visual signals using the JEPA architecture and provide the TacAura haptic manipulation dataset. [Source](https://arxiv.org/abs/2607.24485v2)
- **[DeVA](https://arxiv.org/abs/2607.24159v1)** (decoupled video action model · C): decouples video experts and action experts, using affordance and other physical explicit guidance to learn dynamics and causality. [Source](https://arxiv.org/abs/2607.24159v1)
- **[VQVLA](https://arxiv.org/abs/2607.24148v1)** (Vector Quantization Accelerated Inference · C): Algorithm-hardware collaboration, dynamically adjust quantization accuracy according to motion status and accelerate GEMM through centroid reuse. [Source](https://arxiv.org/abs/2607.24148v1)
- **[WCM](https://arxiv.org/abs/2607.22999v1)** (World Cognitive Interaction Model · C): Based on the SLAK architecture, it decouples perception, logic, action and knowledge, and uses asynchronous runtime and human-in-the-loop teaching to improve interaction capabilities. [Source](https://arxiv.org/abs/2607.22999v1)
- **[Real2Sim2Real ROCm](https://arxiv.org/abs/2607.22997v1)** (ROCm platform VLA pipeline · C): real robot-simulation-real robot deployment solution based on AMD ROCm, integrating 3D Gaussian splashing and Genesis physics engine. [Source](https://arxiv.org/abs/2607.22997v1)

### 5.4 UAVs, autonomous driving, and less relevant topics
{: id="54-无人机自动驾驶与其他低相关方向"}

- **[No Training, Better Flights](https://arxiv.org/abs/2607.19288)** (UAV VLN test scaling · B (mechanism transferable)): See transferable methods. A three-stage pure inference framework with zero training to achieve candidate generation, self-review and safety priority selection. [Source](https://arxiv.org/abs/2607.19288)
- **[Self in Space](https://arxiv.org/abs/2607.12477)** (UAV Self-Awareness Evaluation (ACM MM 2026) · C): SIS-Bench is proposed to evaluate 26 multi-modal large models from the dual dimensions of spatial cognition and self-awareness and three levels of perception/memory/reasoning; and the SIS-Motion optical flow motion fusion scheme is proposed. ([arXiv](https://arxiv.org/abs/2607.12477)/[Project Page](https://choucisan.github.io/publications/self-in-space/)/[Code ](https://github.com/IntelliSensing/Self-in-Space))
- **[CosFly](https://arxiv.org/abs/2605.19120)** (UAV tracking simulation data generation · C): A 7-step pipeline automatically generates aerial photography materials containing RGB/depth/semantics and bilingual instructions, and the accompanying CosFly-Track dataset is made public. [Source](https://arxiv.org/abs/2605.19120)
- **[MulRobBench](https://arxiv.org/abs/2607.23870v1)** (UAV Protocol and Security Compliance Benchmark · C): Offline protocol-aware evaluation benchmark, containing 3,024 samples, to evaluate compliance under uncertain observations and fuzzy instructions. [Source](https://arxiv.org/abs/2607.23870v1)
- **[SkyEV](https://arxiv.org/abs/2607.18747)** (anti-UAV detection dataset · no navigation component, only recording): RGB-event synchronization dataset, covering camera self-motion and 8 types of UAV long-distance small targets, supporting SAST+YOLOX fusion detection baseline. [Source](https://arxiv.org/abs/2607.18747)

### 5.5 News and non-paper items
{: id="55-资讯与非论文"}

The 8 WeChat public account entries in this issue are all paper interpretations and have been merged according to corresponding work in the above tables. There is no independent information content.

## 6. Trends and suggested actions
{: id="六趋势判断与行动建议"}

### Trends
{: id="趋势"}

- **real robot evaluation begins to reversely correct the simulation conclusion**: This review shows the real robot gap of stratified 66% → 51%, end-to-end 61% → 22%, and the phenomenon that SSR is systematically lower than SR by more than 10 percentage points. This digest's assessment: The marginal value of simply brushing simulation SR is declining, and work that includes real robot verification and security indicators will be recognized faster.
- **cross-embodiment is redefined from "data problem" to "conditional modeling problem"**: EA-Nav (geometric conditional token + imitation learning) and X-NavDP (FiLM morphological modulation + RL post-training) give similar judgments from two technical routes in the same week - without explicit modeling embodiment, mixed data will only allow the strategy to fit compromised actions.
- **Decoupling of action generation and autoregressive decoding**: MemVLN’s procedural memory atomic actions and BrainNav’s low-rank world model compression both bypass token-by-token decoding. MemVLN reports 14 FPS and 7 times acceleration, making it possible to increase the real-time rate at the edge from single digit Hz to more than 10 Hz.
- **runtime verification becomes an independent component**: CheckVLA, RoboBRIDGE, Pigey and the test-time scaling framework on the UAV side, jointly point to the structure of "main policy + plug-in verification layer" instead of continuing to increase the single end-to-end model.

### Research gaps
{: id="研究空白"}

- **Modeling of semantic stop conditions**: Overview Actual measurements show that SSR is generally more than 10 percentage points lower than SR, indicating that "arrival" is currently mainly determined by distance thresholds, and semantic stop conditions such as "stop on the left side of the refrigerator" in instructions lack modeling methods. This is a direction where there are gaps in both indicators and methods.
- **Return and backtracking instructions**: The hierarchical method under the backtracking instruction has a real robot SR of 0% and an end-to-end of 20% (small sample). The fact that topological memory is not good at reentrant logic has been exposed by actual measurements, but there is no work in this issue to design memory structures for this purpose.
- **Unified accessibility standard for unstructured outdoor areas**: TEA-AgriVLN has verified the value of the alarm mechanism, but the coupling determination of plant occlusion, flexible traversable media and wheel/foot motion characteristics still lacks a common definition beyond A2A.

### Suggested actions
{: id="建议动作"}

**High priority**

- **Read** carefully: Tongji VLN Overview’s real-scene evaluation chapter and analysis of four types of failure reasons, and accordingly adjust the indicator composition in our own evaluation protocol (at least add SSR and collision rate).
- **reproduces**: X-NavDP's GQRM loss and self-guided trajectory perturbation (the code has been open source), first verify the source of the escape behavior on a single embodiment.
- **Read closely**: MemVLN's procedural memory atomic action space definition and pyramid situation memory compression algorithm are used in edge-side low-latency navigation models.

**Medium priority**

- **Supplementary check**: Quantitative experimental results of the original EA-Nav arXiv (the interpretation and summary of WeChat public account are not given), compared with the cross-embodiment modeling method of X-NavDP's FiLM modulation.
- **Track**: BioVLN's "physical entity + obstacle avoidance safety zone + operating zone" multi-constraint target definition to evaluate whether it can be migrated to the existing indoor ObjectNav benchmark.

**Low priority**

- **Verify**: HiMemVLN WeChat public account original text (this text capture failed) to confirm whether it and MemVLN are the same work.
