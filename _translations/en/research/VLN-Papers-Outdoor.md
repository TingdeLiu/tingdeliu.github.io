---
layout: post
lang: en
translation_id: vln-papers-outdoor
permalink: /en/VLN-Papers-Outdoor/
source_path: _posts/research/2026-01-06-VLN-Papers-Outdoor.md
source_url: /VLN-Papers-Outdoor/
source_revision_date: 2026-11-11
translation_updated: 2026-11-11
title: "VLN Papers: Outdoor Navigation"
date: 2026-11-11
published: false
translation_scope: partial
categories: research
comments: false
author: Tingde Liu
toc: true
---

Unpublished translation draft. The CausalNav reading has been preserved from the goal-navigation collection; WayTrace and the collection introduction still require translation.

## 2. CausalNav (2026)
{: id="causalnav"}
———First Scene Graph-based Semantic Navigation for Dynamic Outdoor Environments

📄 **Paper**: [arXiv:2601.01872](https://arxiv.org/abs/2601.01872) · 🏛️ **IEEE RA-L**

### Key takeaways
{: id="causalnav-精华-12"}

The core highlight of CausalNav is the deep integration of multi-level scene graph (Embodied Graph) and RAG mechanism to achieve long-range semantic navigation that supports open-vocabulary queries - the design paradigm of "graph as knowledge base" is worth learning from. Second, the hierarchical Embodied Graph construction strategy (from fine-grained object nodes to coarse-grained building and cluster nodes) shows how to unify semantic representation and retrieval at multiple spatial scales. Third, the dynamic object filtering mechanism based on the Spatial-Temporal Corridor can distinguish static, quasi-static and dynamic obstacles without additional annotation, which is a practical solution for processing outdoor dynamic scenes. Fourth, local open source LLM is used to replace commercial APIs to complete hierarchical semantic retrieval, proving that high-quality semantic reasoning can still be achieved on an autonomous platform without the cloud.

---

### 1. Background and problem
{: id="1-研究背景问题-11"}

Autonomous semantic navigation in large-scale outdoor dynamic environments faces three major challenges: semantic understanding of open-vocabulary, dynamic environment adaptation (moving obstacles such as pedestrians and vehicles), and long-term stability. Existing VLN research mainly focuses on static indoor scenes, relying on high-precision maps or large-scale training data, and the long-range navigation robustness in real outdoor dynamic scenes has not been fully verified.

---

### 2. Method and innovations
{: id="2-主要方法创新点-9"}

<div align="center">
  <img src="/images/vln/CausalNav-overview.webp" width="100%" loading="lazy" decoding="async" style="aspect-ratio:715/565" alt="CausalNav overall workflow: integrating three major modules: semantic reasoning, dynamic environment adaptation and Embodied Graph planning" />
<figcaption>
CausalNav overall workflow: integrating three major modules: semantic reasoning, dynamic environment adaptation and Embodied Graph planning
</figcaption>
</div>

CausalNav proposes a semantic navigation framework composed of three core modules:

**Module 1: Open vocabulary target tracking and self-motion estimation**

Use YOLO-World to extract open-vocabulary 2D detection boxes and segmentation masks from RGB images, and perform multi-view target tracking through ByteTrack. Combined with the LiDAR point cloud, the 2D detection is projected into the 3D space to obtain the 3D attitude of the target $$^w\mathbf{T}_{obj}$$. The self-vehicle motion is estimated through LiDAR-IMU odometry (FAST-LIO2), providing an accurate basis for positioning and coordinate transformation.

**Module 2: Dynamic Object Filtering and Embodied Graph Construction**

<div align="center">
  <img src="/images/vln/CausalNav-architecture.webp" width="100%" loading="lazy" decoding="async" style="aspect-ratio:1446/699" alt="CausalNav three-module pipeline architecture: target tracking and self-motion estimation → dynamic filtering and graph construction → graph update and natural language navigation" />
<figcaption>
CausalNav three-module pipeline architecture: target tracking and self-motion estimation → dynamic filtering and graph construction → graph update and natural language navigation
</figcaption>
</div>

- **Space-time corridor filtering**: Encode the historical trajectory of each target as a space-time corridor $$\mathcal{T} = \{^w\mathbf{T}^n_{obj}, \text{3DBBox}_i, t_i\}_{i=1}^n$$. If the target's displacement exceeds the threshold within $k$ steps, it is identified as a dynamic target and removed from the graph, effectively eliminating false nodes caused by motion.

- **Embodied Graph hierarchical construction**: The static environment consists of two types of nodes - the building node $\nu_i^{build}$ comes from the offline map, and the object node $\nu_i^{obj}$ comes from real-time perception. Use LLM to perform hierarchical clustering (spatial-semantic similarity) of nodes to form multi-level abstraction: object layer (Level $L-1$) → building/Place layer (Level $L$) → clustering node (Clustering Node). Each time the self-vehicle moves exceeds the distance threshold $d$, a new self-vehicle node $\nu_i^l$ is added to record the historical trajectory.

- **RAG semantic retrieval**: hierarchical retrieval based on LLM scoring, combined with spatial similarity $\kappa^{spatial}$ and semantic similarity $\kappa^{semantic}$, selecting the node path that best matches the query layer by layer in the graph, supporting open-vocabulary target positioning.

**Module 3: Embodied Graph dynamic update and natural language navigation**

<div align="center">
  <img src="/images/vln/CausalNav-embodied-graph.webp" width="100%" loading="lazy" decoding="async" style="aspect-ratio:715/610" alt="Embodied Graph built in the simulation environment: multi-level fusion of coarse-grained building nodes and fine-grained object nodes (fire hydrants, mailboxes, etc.)" />
<figcaption>
Embodied Graph built in the simulation environment: multi-level fusion of coarse-grained building nodes and fine-grained object nodes (fire hydrants, mailboxes, etc.)
</figcaption>
</div>

- **Global planning**: Parse natural language instructions, infer the target location through RAG retrieval Embodied Graph, and give priority to the Dijkstra shortest path in the historical trajectory; if the target is unreachable, call offline maps or Google Maps to generate coarse-grained routes, and the result is expressed as waypoint sequence $\mathcal{W} = \{w_1, w_2, \ldots, w_n\}$.

- **Local planning**: Use RH-Map for real-time dynamic local map construction, use Informed-RRT* to generate the initial trajectory, and then use NMPC-CBF (Nonlinear Model Predictive Control with Control Barrier Function) for trajectory tracking and dynamic obstacle avoidance to ensure the safety of moving pedestrians/vehicles.

---

### 3. Results and findings
{: id="3-核心结果发现-10"}

<div align="center">
  <img src="/images/vln/CausalNav-real-world-results.webp" width="100%" loading="lazy" decoding="async" style="aspect-ratio:709/997" alt="Navigation experiments at different distance scales in real environments: short-range (130m object-level instructions) and long-range (512m building-level instructions)" />
<figcaption>
Navigation experiments at different distance scales in real environments: short-range (130m object-level instructions) and long-range (512m building-level instructions)
</figcaption>
</div>

**Simulation experiment** (Compare ViNT, NoMaD, GNM, CityWalker):
- Small task: SR 100%, SPL 88.9%, CC 0.2 (best among all methods)
- Medium mission: SR 92%, SPL 82.2%
- Large mission: SR 80%, SPL 66.0%, CC 1.2, TL 141.82m

**Real World Experiment**:
- Short range (130m): ViNT and CausalNav both succeed, other methods fail
- Long range (512m): Only CausalNav successfully completed the task, other methods failed due to collision
- CityWalker's real-world performance is significantly worse than simulation, and it is sensitive to lighting changes and dynamic obstacles.

**ablation experiment**:
- Enable Embodied Graph dynamic updates: SR increased from 78% to 90%, SPL increased from 54.7% to 80.1%
- Optimal hyperparameters: $\alpha=\beta=0.5$, $\gamma=1.5$ (space-semantic balance point)
- Running latency: 105ms/cycle (10Hz), only 11% more overhead than NoMaD

---

### 4. Limitations
{: id="4-局限性-10"}

The robustness of CausalNav under extreme lighting/weather conditions needs to be improved, and the compression and forgetting mechanism of long-range graph memory has not yet been perfected. Problems of graph expansion and retrieval accuracy may occur after extremely long runs.
