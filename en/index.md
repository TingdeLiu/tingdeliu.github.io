---
layout: default
title: Tingde Liu | Academic Profile
permalink: /en/
lang: en
---

<div class="academic-page">

  <!-- Header / Bio Section -->
  <header class="academic-header">
    <div class="academic-intro">
      <h1 class="academic-name">Tingde Liu <span class="chinese-name">(刘庭德)</span></h1>
      <p class="academic-subtitle">
        <strong>AI &amp; Robotics Engineer</strong> · Embodied Intelligence<br>
        M.Sc. in Mechatronics &amp; Robotics, <a href="https://www.uni-hannover.de/" target="_blank">Leibniz Universität Hannover</a>
      </p>
      <div class="academic-links">
        <a href="mailto:tingde.liu.luh@gmail.com" class="academic-badge">Email</a>
        <a href="https://github.com/TingdeLiu" target="_blank" class="academic-badge">GitHub</a>
        <a href="https://www.linkedin.com/in/tingde-liu-379818270/" target="_blank" class="academic-badge">LinkedIn</a>
        <a href="{{ site.baseurl }}/about/" class="academic-badge lang-tag">中文简介</a>
      </div>
    </div>
  </header>

  <div class="academic-bio-text">
    <p>
      I am a robotics and AI engineer based in Beijing, working on <strong>Embodied Intelligence</strong> and robot foundation systems. I completed my M.Sc. in Mechatronics and Robotics at Leibniz Universität Hannover (LUH), Germany.
    </p>
    <p>
      My current research centers on <strong>Vision-Language Navigation (VLN)</strong>, <strong>Embodied Agent Frameworks</strong>, and <strong>3D Multimodal Foundation Models (3D-LLM / Spatial Intelligence)</strong>. I am driven by a fundamental question: <em>what does it take for a physical robot to genuinely understand and act in our three-dimensional world — not merely mimicking demonstrations, but reasoning about geometry, language, memory, and intention?</em>
    </p>
  </div>

  <!-- Research Interests -->
  <section class="academic-section">
    <h2 class="academic-heading">Research Interests</h2>
    <div class="interest-tags">
      <span class="interest-tag">Vision-Language Navigation (VLN)</span>
      <span class="interest-tag">Embodied Agent Frameworks</span>
      <span class="interest-tag">Vision-Language-Action Models (VLA)</span>
      <span class="interest-tag">3D Multimodal LLMs / Spatial AI</span>
      <span class="interest-tag">Sim-to-Real Robot Navigation</span>
      <span class="interest-tag">Semantic Mapping &amp; 3D Vision</span>
    </div>
  </section>

  <!-- News & Highlights -->
  <section class="academic-section">
    <h2 class="academic-heading">Highlights</h2>
    <ul class="academic-timeline-list">
      <li>
        <span class="item-date">2025.10 – Present</span>
        <span class="item-content">Working on the <strong>embodied navigation system harness</strong> — binding perception, spatial memory, hierarchical planning, and control into a production-grade stack for real robots.</span>
      </li>
      <li>
        <span class="item-date">2024.08</span>
        <span class="item-content">Completed Master's Thesis on <strong>MMS-LLM</strong> supervised by Prof. Dr.-Ing. Claus Brenner at LUH IKG, awarded with the highest grade <strong>1.0 (with distinction)</strong>.</span>
      </li>
      <li>
        <span class="item-date">2023.05</span>
        <span class="item-content">First-author paper on deep learning error modeling for terrestrial laser scanners published in the <strong>Journal of Applied Geodesy</strong>.</span>
      </li>
    </ul>
  </section>

  <!-- Publications & Theses -->
  <section class="academic-section">
    <h2 class="academic-heading">Publications &amp; Academic Output</h2>

    <div class="pub-card">
      <div class="pub-title">Error modeling of terrestrial laser scanners using deep learning with PointNet++</div>
      <div class="pub-authors"><strong>Tingde Liu</strong>, Jan Hartmann</div>
      <div class="pub-venue"><em>Journal of Applied Geodesy</em>, De Gruyter, 2023</div>
      <div class="pub-desc">
        Investigated physical uncertainty patterns in high-precision LiDAR point clouds. Developed a deep geometric neural network architecture based on PointNet++ to model systemic sensor inaccuracies, reducing the mean measurement error from 0.387 mm down to 0.009 mm.
      </div>
      <div class="pub-links">
        <a href="https://www.degruyterbrill.com/document/doi/10.1515/jag-2023-0097/html" target="_blank" class="pub-btn">Journal Paper</a>
        <a href="https://doi.org/10.1515/jag-2023-0097" target="_blank" class="pub-btn">DOI: 10.1515/jag-2023-0097</a>
      </div>
    </div>

    <div class="pub-card">
      <div class="pub-title">MMS-LLM: Multimodal Large Language Model for Urban LiDAR Point Clouds</div>
      <div class="pub-authors"><strong>Tingde Liu</strong> (Supervised by apl. Prof. Dr.-Ing. Claus Brenner)</div>
      <div class="pub-venue"><em>Master's Thesis</em>, Leibniz Universität Hannover, 2024 — <strong>Grade: 1.0 (Distinction)</strong></div>
      <div class="pub-desc">
        Addressed the gap between clean synthetic benchmarks and real-world urban LiDAR (sparse, noisy, intensity-only). Integrated Point-BERT with a fine-tuned Vicuna-7B model, trained on the custom-constructed <code>ikgc17</code> dataset (4,185 instances, 7,000+ instruction pairs). Outperformed PointLLM baselines by 40% on spatial understanding and 3D captioning.
      </div>
      <div class="pub-links">
        <a href="https://github.com/TingdeLiu/MMS-LLM" target="_blank" class="pub-btn">Code &amp; Dataset</a>
      </div>
    </div>
  </section>

  <!-- Selected Systems & Projects -->
  <section class="academic-section">
    <div class="section-title-wrap">
      <h2 class="academic-heading">Selected Projects</h2>
      <a href="{{ site.baseurl }}/home" class="heading-extra-link">View All Projects →</a>
    </div>

    <div class="project-grid">
      <div class="project-card">
        <div class="project-card-title">
          <a href="https://github.com/TingdeLiu/AgentNav" target="_blank">AgentNav</a>
          <span class="lang-badge">Python</span>
        </div>
        <p>Agent-based navigation system for real robots. Translates natural language instructions into multi-stage spatial goals via LLM reasoning and hierarchical planners.</p>
      </div>

      <div class="project-card">
        <div class="project-card-title">
          <a href="https://github.com/TingdeLiu/MMS-LLM" target="_blank">MMS-LLM</a>
          <span class="lang-badge">Python</span>
        </div>
        <p>Multimodal 3D LLM for urban LiDAR point clouds. Point-BERT + Vicuna-7B, achieving +40% gain over PointLLM baseline. Master's thesis (Grade 1.0).</p>
      </div>

      <div class="project-card">
        <div class="project-card-title">
          <a href="https://github.com/TingdeLiu/miniagent" target="_blank">miniagent</a>
          <span class="lang-badge">Python</span>
        </div>
        <p>Lightweight agentic framework designed for building tool-using, memory-equipped embodied AI agents.</p>
      </div>

      <div class="project-card">
        <div class="project-card-title">
          <a href="https://github.com/TingdeLiu/Tyndall-Skills" target="_blank">Tyndall-Skills</a>
          <span class="lang-badge">Python</span>
        </div>
        <p>A collection of automated research workflows, agent harnesses, and Claude Code SKILL.md templates for AI researchers.</p>
      </div>
    </div>
  </section>

  <!-- Experience & Education -->
  <section class="academic-section">
    <h2 class="academic-heading">Experience &amp; Education</h2>
    
    <div class="exp-item">
      <div class="exp-header">
        <span class="exp-role"><strong>AI Engineer (Embodied Intelligence)</strong></span>
        <span class="exp-time">2025 – Present</span>
      </div>
      <div class="exp-org">Robotics Company / Tyndall Labs · Beijing, China</div>
      <p class="exp-text">
        Leading the embodied navigation system framework — the runtime harness binding perception, spatial memory, topological planning, and policy execution into a unified stack for real physical robots. Focusing on Vision-Language Navigation (VLN) and agentic spatial reasoning.
      </p>
    </div>

    <div class="exp-item">
      <div class="exp-header">
        <span class="exp-role"><strong>Research Assistant</strong></span>
        <span class="exp-time">2024 – 2025</span>
      </div>
      <div class="exp-org">IPH – Institut für Integrierte Produktion Hannover · Hannover, Germany</div>
      <p class="exp-text">
        Worked on deploying mobile robots in real industrial warehouse and production settings. Focused on 3D reconstruction, semantic mapping, and closing the gap between laboratory demos and long-running field reliability.
      </p>
    </div>

    <div class="exp-item">
      <div class="exp-header">
        <span class="exp-role"><strong>Research Associate</strong></span>
        <span class="exp-time">2022 – 2024</span>
      </div>
      <div class="exp-org">Institut für Kartographie und Geoinformatik (IKG), LUH · Hannover, Germany</div>
      <p class="exp-text">
        Investigated multimodal spatial intelligence, developing multimodal large language models capable of processing sparse, noisy urban LiDAR scans.
      </p>
    </div>

    <div class="exp-item">
      <div class="exp-header">
        <span class="exp-role"><strong>M.Sc. in Mechatronics &amp; Robotics</strong></span>
        <span class="exp-time">2019 – 2024</span>
      </div>
      <div class="exp-org">Leibniz Universität Hannover · Hannover, Germany</div>
      <p class="exp-text">
        Master's Thesis Grade: <strong>1.0 (with distinction)</strong>. Core coursework in Robot Perception, Machine Learning, State Estimation, Sensor Fusion, and Motion Control.
      </p>
    </div>
  </section>

  <!-- Research Surveys (International Reader Note) -->
  <section class="academic-section">
    <h2 class="academic-heading">Research Surveys &amp; Technical Notes</h2>
    <div class="academic-callout">
      <div class="callout-icon">💡</div>
      <div class="callout-content">
        <p><strong>Note for International Researchers:</strong></p>
        <p>
          I regularly curate and author extensive, structured surveys covering recent breakthroughs in Embodied AI. These surveys are maintained in Chinese, but are well-organized with complete English paper titles, publication venues, and benchmark leaderboards. They are directly readable using automated translation tools like <a href="https://immersivetranslate.com/" target="_blank">Immersive Translate</a> or Google Translate.
        </p>
      </div>
    </div>

    <ul class="survey-link-list">
      <li>
        <a href="{{ site.baseurl }}/VLN-Survey/"><strong>Vision-Language Navigation (VLN) Comprehensive Survey</strong></a>
        <span class="survey-desc">— Paradigms, historical evolution, 3D representations, and technical taxonomy.</span>
      </li>
      <li>
        <a href="{{ site.baseurl }}/VLN-Papers/"><strong>VLN Core Papers &amp; Benchmark Leaderboard Tracker</strong></a>
        <span class="survey-desc">— In-depth breakdown of 60+ seminal VLN papers and comparative recipes.</span>
      </li>
      <li>
        <a href="{{ site.baseurl }}/VLA-Survey/"><strong>Vision-Language-Action (VLA) Frontier Survey</strong></a>
        <span class="survey-desc">— End-to-end robotics foundation models, tokenization, action chunking, and policy learning.</span>
      </li>
      <li>
        <a href="{{ site.baseurl }}/World-Models-Survey/"><strong>World Models &amp; Video Generation Survey</strong></a>
        <span class="survey-desc">— Generative simulation, physics-informed representations, and prediction in robotics.</span>
      </li>
      <li>
        <a href="{{ site.baseurl }}/Spatial-Intelligence-Survey/"><strong>Spatial Intelligence &amp; 3D Foundation Models Survey</strong></a>
        <span class="survey-desc">— 3D Gaussian Splatting, NeRF, point cloud understanding, and geometric reasoning.</span>
      </li>
    </ul>
  </section>

  <!-- Technical Skills -->
  <section class="academic-section">
    <h2 class="academic-heading">Technical Skills</h2>
    <div class="skills-block">
      <p><strong>Programming &amp; Systems:</strong> Python, C++, MATLAB, ROS 2, Linux, CUDA, Docker, Git</p>
      <p><strong>Deep Learning &amp; AI:</strong> PyTorch, LLM Fine-tuning, Multimodal Alignment, Point-BERT, Transformers, RAG</p>
      <p><strong>3D Vision &amp; Robotics:</strong> LiDAR Processing, PCL, 3D Gaussian Splatting, SLAM, Sensor Fusion, Motion Planning</p>
      <p><strong>Simulation &amp; Tools:</strong> Isaac Sim, Habitat, Gazebo, Claude Code</p>
    </div>
  </section>

</div>
