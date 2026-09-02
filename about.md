---
layout: page
title: 关于我 (About)
permalink: /about/
lang: zh
---

<div class="lang-banner">
  <span>💡 <strong>Looking for English profile?</strong> Visit the <a href="{{ site.baseurl }}/en/"><strong>English Academic Page</strong></a> (Bio, Publications, Theses &amp; CV).</span>
</div>

我是**刘庭德 (Tingde Liu)**，目前在北京从事**具身智能（Embodied Intelligence）**与机器人自主导航大脑算法研发。硕士毕业于德国汉诺威莱布尼茨大学（[Leibniz Universität Hannover](https://www.uni-hannover.de/), LUH）机电与机器人专业。

我的核心研究方向聚焦于**视觉语言导航（Vision-Language Navigation, VLN）**、**具身智能体系统框架（Embodied Agent Frameworks）**与 **3D 多模态大模型（3DLLM / 空间智能）**。我始终关注一个核心命题：*如何让机器人在真实三维物理世界中具备真正的空间理解、语言推理与自主意图规划能力，而非仅仅停留在表面上的轨迹模仿或离散分类？*

---

## 个人经历 (Experience)

- **具身智能算法工程师** · 北京 (2025 – 至今)  
  就职于具身智能机器人团队，负责**具身导航系统软硬件 Harness 架构**——将传感器多模态感知、3D 空间拓扑记忆、分层决策规划与动作控制执行紧密耦合，构建真实物理机器人可稳定运行的高性能算法栈；主要以视觉语言导航（VLN）与智能体自主空间推理为技术突破口。

- **科研助理 (Research Assistant)** · [汉诺威工业生产研究所 (IPH)](https://www.iph-hannover.de/de/) (2024 – 2025)  
  与 Marc Warnecke 工程师合作，负责工业现场移动机器人实机部署。从事 3D 环境重建、语义建图与自主导航研发，深入攻克了“算法仿真/实验室 Demo”与“工业现场严苛约束下长效稳定运行”之间的巨大鸿沟。

- **科研助理 (Research Associate)** · [汉诺威大学制图与地理信息研究所 (IKG)](https://www.ikg.uni-hannover.de/en/) (2022 – 2024)  
  近两年时间深入探索空间智能与 3D 多模态感知。在 Claus Brenner 教授指导下，研发能够理解真实城市稀疏、无色彩、含噪声车载 LiDAR 点云的大模型 MMS-LLM。硕士毕业论文以满分 **1.0 (with distinction)** 评级通过。

- **科研项目 (Research Project)** · [汉诺威大地测量研究所 (GIH)](https://www.gih.uni-hannover.de/) (2022)  
  与 Jan Hartmann 合作，利用深度几何学习（PointNet++）对高精度地面三维激光扫描仪进行物理误差建模，将测量均方误差从 0.387 mm 大幅降低至 0.009 mm。研究成果发表于期刊 *Journal of Applied Geodesy*（[DOI: 10.1515/jag-2023-0097](https://www.degruyterbrill.com/document/doi/10.1515/jag-2023-0097/html)）。

---

## 关于本博客 (About This Blog)

这里是我系统化梳理、沉淀和分享具身智能技术思考的开放空间。在学习与科研过程中，我习惯将读到的论文、构建的系统和理解的演进整理成文，而不是散落在各处书签和草稿纸上。

本站的核心内容涵盖：
1. **具身智能与大模型前沿综述**：涵盖 [视觉语言导航（VLN）](/VLN-Survey/)、[VLA 端到端模型](/VLA-Survey/)、[世界模型](/World-Models-Survey/)、[空间智能 3D-LLM](/Spatial-Intelligence-Survey/) 等；
2. **论文精读与排行榜**：[VLN 核心论文路线梳理](/VLN-Papers/)、代表性模型对比矩阵与开源链接；
3. **具身机器人软硬件工程实践**：ROS 2 系统架构、强化学习实操与智能体框架搭建心得。

我也希望本站成为一个友好、开放的开源交流空间，非常欢迎同行与读者交流探讨、指正建议。

---

## 核心研究关切 (Research Questions)

我的科研与工程实践围绕以下几个持续探寻的问题展开：

- **3D 空间推理**：语言大模型如何深层次理解和推理真实三维几何空间与拓扑关系？
- **指令遵循与自主探索**：机器人在开放未知环境下，如何依据人类模糊的自然语言指令完成长程导航？
- **Sim-to-Real 跨越**：如何最大程度弥合仿真器环境与真实物理环境在感知噪声与动力学交互上的鸿沟？
- **Robot Agent Harness**：如何构建高鲁棒的具身系统底座——无缝整合感知、情境记忆、长程规划与实时底层控制？
- **自主能动性 (Agency)**：如何赋予机器人真正的“自主意图”：不仅执行预定任务，更能自我反思、动态修正常识与意图？

---

## 学术成果与代表项目

### 已发表论文 & 学术产出
- **期刊论文**：*Error modeling of terrestrial laser scanners using deep learning with PointNet++*  
  **Tingde Liu**, Jan Hartmann. *Journal of Applied Geodesy*, 2023. [DOI: 10.1515/jag-2023-0097](https://www.degruyterbrill.com/document/doi/10.1515/jag-2023-0097/html)
- **硕士论文 (Grade: 1.0)**：*MMS-LLM: Multimodal Large Language Model for Urban LiDAR Point Clouds*  
  **Tingde Liu** (导师: apl. Prof. Dr.-Ing. Claus Brenner, 汉诺威大学 IKG). [代码仓库](https://github.com/TingdeLiu/MMS-LLM)

### 精选开源项目
- [AgentNav](https://github.com/TingdeLiu/AgentNav)：基于 Agent 的真实机器人自主导航系统，自然语言指令控制与分层规划。
- [MMS-LLM](https://github.com/TingdeLiu/MMS-LLM)：车载激光雷达点云多模态大语言模型（Point-BERT + Vicuna-7B），大幅超越 PointLLM 基线。
- [miniagent](https://github.com/TingdeLiu/miniagent)：轻量级 Tool-using AI Agent 框架，支持记忆与工具链扩展。
- [Tyndall-Skills](https://github.com/TingdeLiu/Tyndall-Skills)：面向科研流程与智能体自动化研发的工作流与 Claude Code SKILL 模板集。
- [quant.ai](https://github.com/TingdeLiu/quant.ai)：通过 MCP 在 AI 助手内运行的美股可解释量化投研工具。

---

## 技术技能 (Skills)

- **编程语言**：Python, C++, MATLAB
- **系统与架构**：PyTorch, ROS 2, Linux, CUDA, Docker, Git
- **大模型与多模态**：LLM 微调, 视觉-语言对齐, Point-BERT, CLIP, LLaVA, RAG
- **3D 视觉与机器人**：LiDAR 点云处理, PCL, 3D 高斯泼溅 (3DGS), SLAM, 传感器融合, 运动规划, 强化学习
- **仿真与工具**：Isaac Sim, Habitat, Gazebo, Claude Code

---

## 联系方式 (Contact)

- **GitHub**：[TingdeLiu](https://github.com/TingdeLiu)
- **LinkedIn**：[Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- **Email**：[tingde.liu.luh@gmail.com](mailto:tingde.liu.luh@gmail.com)
- **全部文章**：[归档列表](/archive/)

---

*探索人工智能与机器人技术的无限可能！*
