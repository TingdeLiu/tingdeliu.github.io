---
layout: page
title: About
permalink: /about/
---

# Tingde Liu

I am a robotics and AI engineer with an M.Sc. in Mechatronics and Robotics from Leibniz Universität Hannover (LUH). My current research centers on vision-language navigation (VLN) and vision-language-action models (VLA) — and the broader question of what it would actually take for a robot to understand and act in the world the way we do. I am interested in genuine embodied intelligence, not just systems that appear to navigate, but ones that truly reason about space, language, and intention.

## How I Got Here

My first real encounter with 3D sensing came during my **Studienarbeit** at the [Geodätisches Institut Hannover](https://www.gih.uni-hannover.de/), where I worked alongside Research Associate Jan Hartmann. The problem was deceptively simple: laser scanners are not perfect, and their errors follow physical patterns. Under Jan's guidance I learned to model that uncertainty with deep learning — reducing mean measurement error from 0.387 mm to just 0.009 mm using a PointNet++ architecture. Seeing the work eventually published in the *Journal of Applied Geodesy* ([DOI: 10.1515/jag-2023-0097](https://www.degruyterbrill.com/document/doi/10.1515/jag-2023-0097/html)) was the moment I understood that careful, rigorous research produces results worth sharing.

That experience led me to the **[Institut für Kartographie und Geoinformatik (IKG)](https://www.ikg.uni-hannover.de/en/)** at LUH, where I spent over a year as a research associate working with M.Sc. Leichter Artem and Dr.-Ing. Jeldrik Axmann. The environment there was both demanding and generative. Working alongside them on urban 3D reconstruction and multimodal sensor data exposed me to the gap between academic benchmarks and real-world LiDAR data — sparse, noisy, colorless — and made me want to close it. Their rigour in data quality and system design became a standard I internalized.

That question became my **Masterarbeit**, supervised by apl. Prof. Dr.-Ing. Claus Brenner. I set out to build a multimodal large language model that could genuinely understand urban LiDAR point clouds — not synthetic, color-rich data, but the kind of sparse, intensity-only scans you get from a real city. The result was MMS-LLM ([GitHub](https://github.com/TingdeLiu/MMS-LLM)): a system combining a Point-BERT encoder with a fine-tuned Vicuna-7B, trained on the ikgc17 dataset I built from scratch (4,185 instances, 7,000+ instruction pairs). It outperformed the PointLLM baseline by 40% on classification and description tasks. Prof. Brenner's expectation of intellectual honesty — to understand not just what the numbers say but why — shaped how I think about model evaluation. The thesis was graded **1.0**.

After graduating, I joined **[IPH – Institut für Integrierte Produktion Hannover](https://www.iph-hannover.de/de/)** as a robotics engineer, where I worked with Dipl.-Ing. Marc Warnecke on deploying robots in real industrial environments. The gap between a working demo and a system that runs reliably on a factory floor turned out to be enormous. Marc's practical, problem-first mindset taught me that engineering is ultimately about what works under real constraints — and that lesson has been just as valuable as anything I learned in the lab.

In 2025, after five years abroad, I returned to China — partly out of longing for family, partly drawn by the energy around embodied AI that I could feel building back home. I am now based in Beijing, working as an AI engineer focused on embodied intelligence, with VLA and VLN as my primary research directions. This blog is one way I stay connected to the ideas I care about.

## Research Interests

These experiences converged around a set of questions I keep returning to:

- How can language models reason meaningfully about 3D space?
- What does it take for a robot to navigate using natural instructions?
- How do we bridge the gap between simulation and real-world perception?
- How do we build a **Robot OS** — a coherent harness that integrates perception, memory, planning, and action into a unified system?
- How can **agentic AI** give robots something closer to genuine agency: not just executing instructions, but forming intentions, adapting plans, and acting with purpose?

In practice this means I work on **Vision-Language Navigation (VLN)**, **Vision-Language-Action models (VLA)**, **agentic robot systems**, and the infrastructure that makes embodied intelligence real.

## Technical Skills

**Languages:** Python, C++, MATLAB
**Frameworks:** PyTorch, ROS2, LangChain
**3D Vision:** LiDAR processing, PCL, 3D Gaussian Splatting
**Tools:** Docker, Git, Gazebo, Isaac Sim

## Links

- **GitHub:** [TingdeLiu](https://github.com/TingdeLiu)
- **LinkedIn:** [Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- **Blog:** [Archive](https://tingdeliu.github.io/archive/)
- **Email:** tingde.liu.luh@gmail.com
---

*Continuously learning and exploring the infinite possibilities of AI and Robotics!*
