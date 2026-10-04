<div align="center">

# Tingde Liu · Research Notes

**English** | [简体中文](README_CN.md)

**Research notes on embodied AI, vision-language navigation, and robot learning**

[![Website](https://img.shields.io/badge/Website-tingdeliu.github.io-2563EB?style=flat-square)](https://tingdeliu.github.io/en/)
[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.3-CC0000?style=flat-square&logo=jekyll&logoColor=white)](https://jekyllrb.com/)

[![Stars](https://img.shields.io/github/stars/TingdeLiu/tingdeliu.github.io?style=flat-square&logo=github&color=F5A623)](https://github.com/TingdeLiu/tingdeliu.github.io/stargazers)
[![Code License: MIT](https://img.shields.io/badge/Code-MIT-3DA639?style=flat-square)](LICENSE)
[![Content License: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-EF9421?style=flat-square)](LICENSE-CONTENT)

[English Edition](https://tingdeliu.github.io/en/) · [Research](https://tingdeliu.github.io/en/research/) · [Blog](https://tingdeliu.github.io/en/blog/) · [Projects](https://tingdeliu.github.io/home/) (Chinese only) · [About](https://tingdeliu.github.io/about/) (Chinese only)

</div>

## Overview

This is Tingde Liu's personal academic blog and research knowledge base, featuring surveys, paper readings, and engineering notes on embodied AI, robot navigation, and multimodal learning.

The site is intended for researchers, students, and engineers working in embodied AI and robotics, with a basic understanding of machine learning and deep learning.

Chinese is the source edition. The English edition includes the complete VLN Survey, both VLN paper collections, and six published embodied-navigation weekly digests, with an [English home page](https://tingdeliu.github.io/en/), [Research index](https://tingdeliu.github.io/en/research/), and [Blog index](https://tingdeliu.github.io/en/blog/). Translated articles support language switching. Other long-form articles will be translated gradually; untranslated destinations are marked **Chinese only** below. See the [English edition maintenance guide](docs/english-edition.md) for translation and synchronization checks.

The site aims to provide a systematic, continuously updated account of models, datasets, training methods, system architectures, benchmarks, and deployment on real robots. Long-form surveys establish the broader technical context, while shorter posts capture focused analyses and interim observations.

As of **October 3, 2026**, the collection includes **18 research surveys and paper readings**, **7 technical blog posts**, **6 published embodied-navigation weekly digests**, and **over 1,000** image files. The published Chinese article sources total about **56,000 lines of Markdown**; English translations are counted separately.

Created and maintained by [@TingdeLiu](https://github.com/TingdeLiu), the project has been continuously updated since January 4, 2026. The September 16, 2026 maintenance record documents **5 community feedback submissions, all addressed and closed**; see [MAINTAINERS.md](MAINTAINERS.md).

## Research Areas

The complete article index is organized by topic. Links use the English edition where available; other articles are marked **Chinese only**.

| Area | Article | Focus |
| --- | --- | --- |
| **Vision-Language Navigation** | [VLN Survey](https://tingdeliu.github.io/en/VLN-Survey/) | Task definitions, datasets, and benchmarks; the evolution from modular pipelines to end-to-end navigation agents |
| | [VLN Papers: Instruction Following](https://tingdeliu.github.io/en/VLN-Papers/) | Detailed readings of published or leaderboard-listed work, with leaderboards for three instruction-following task settings |
| | [VLN Papers: Goal Navigation and Extensions](https://tingdeliu.github.io/en/VLN-Papers-Extended/) | Goal-navigation leaderboards, recent preprints, and additional papers beyond the main instruction-following collection |
| **Vision-Language-Action Models** | [VLA Survey](https://tingdeliu.github.io/VLA-Survey/) (Chinese only) | Robot policy learning, action generation, imitation learning, and reinforcement learning |
| | [VLA Paper Readings](https://tingdeliu.github.io/VLA-Papers/) (Chinese only) | Individual paper analyses, RoboDojo leaderboard links, and comparisons under matching experimental settings |
| **Robot Navigation Systems** | [Classical Robot Navigation Survey](https://tingdeliu.github.io/Robot-Navigation-Survey/) (Chinese only) | SLAM, localization and mapping, path planning, and motion control |
| | [Complete ROS 2 Guide](https://tingdeliu.github.io/ROS2-Survey/) (Chinese only) | Communication models, lifecycle nodes, QoS, and real-robot deployment |
| **Multimodal and Spatial Intelligence** | [VLM Survey](https://tingdeliu.github.io/VLM-Survey/) (Chinese only) | An overview of multimodal fusion methods in vision-language models |
| | [Spatial Intelligence Survey](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) (Chinese only) | 3D scene understanding, point clouds, depth estimation, and Gaussian Splatting |
| **World Models and Agents** | [World Models Survey](https://tingdeliu.github.io/World-Models-Survey/) (Chinese only) | Environment modeling and prediction, video-generative world models, and embodied applications |
| | [AI Agent Survey](https://tingdeliu.github.io/AI-Agent-Survey/) (Chinese only) | Reasoning, memory and skills, context engineering, multi-agent collaboration, and the MCP / WebMCP / A2A / MHS connectivity layers and security |
| | [Embodied Agent Paper Readings](https://tingdeliu.github.io/Embodied-Agent-Papers/) (Chinese only) | Closed-loop runtimes (AgentOS / Harness), typed action abstractions, physical orchestration, scene-graph exit-code evaluation, and governance of self-evolution |
| | [Embodied Agent Survey](https://tingdeliu.github.io/Embodied-Agent-Harness-Survey/) (Chinese only) | Integration of agent harnesses (physical guardrails, exit-code evaluation, online critics) with software systems engineering (microservices, ZeroMQ, fast/slow loops) |
| **Learning and Training Methods** | [LLM Training Survey](https://tingdeliu.github.io/LLM-Training-Survey/) (Chinese only) | Pretraining, post-training, alignment, parallelism, and inference acceleration |
| | [Reinforcement Learning Survey](https://tingdeliu.github.io/Reinforcement-Learning-Survey/) (Chinese only) | Algorithms from theoretical foundations to embodied AI applications |
| | [Deep Learning Survey](https://tingdeliu.github.io/Deep-Learning-Survey/) (Chinese only) | Network architectures, optimization, and training techniques |
| | [Machine Learning Survey](https://tingdeliu.github.io/Machine-Learning-Survey/) (Chinese only) | Classical models and statistical learning foundations |
| **Engineering Practice** | [Python Engineering Guide](https://tingdeliu.github.io/Python-Engineering-Survey/) (Chinese only) | Environment and dependency boundaries, src layouts and pyproject, guarantees of uv lockfiles, quality and CI pipelines, CUDA dependency layers, ROS 2 and virtual-environment compatibility, and four levels of experiment reproducibility |

### Technical Blog

These posts are currently available in Chinese only.

| Article | Topic |
| --- | --- |
| [Jev: System One Models and Embodied Navigation](https://tingdeliu.github.io/Jev-Embodied-Navigation/) | Typed decisions, evidence boundaries, and research questions for navigation, memory, and agent collaboration |
| [RoboTTT: A Detailed Analysis](https://tingdeliu.github.io/RoboTTT-Fast-Weights/) | How TTT modules compress history into fast weights for long-term storage and real-time retrieval |
| [Graph Engineering](https://tingdeliu.github.io/graph-engineering/) | Agent graph orchestration and design patterns in the large-model era |
| [Loop Engineering](https://tingdeliu.github.io/loop-engineering/) | Closed-loop approaches to agent engineering |
| [Mixture-of-Transformers](https://tingdeliu.github.io/mixture-of-transformers/) | Modality decoupling and sparsity in multimodal foundation models |
| [Tree-Attention Training](https://tingdeliu.github.io/Tree-Attention-Decoding/) | How Robostral Navigate reduces VLN training tokens by 22× |
| [Harness Engineering](https://tingdeliu.github.io/Harness-Engineering/) | Execution environments and toolchains for agents |

### Embodied-Navigation Weekly Digests

Weekly digests cover new arXiv papers and Chinese community analyses in six parts: key conclusions, reading priorities, detailed analyses, transferable methods, a topic overview, and trends. Each issue emphasizes conclusions, evidence, and reading priorities. Figures that cannot be verified are explicitly qualified by source and measurement scope.

| Issue | Coverage Period |
| --- | --- |
| [Issue 6](https://tingdeliu.github.io/en/vln-weekly-2026-09-27/) | 2026-09-16 ~ 2026-09-24 |
| [Issue 5](https://tingdeliu.github.io/en/vln-weekly-2026-09-19/) | 2026-09-09 ~ 2026-09-17 |
| [Issue 4](https://tingdeliu.github.io/en/vln-weekly-2026-09-12/) | 2026-09-02 ~ 2026-09-10 |
| [Issue 3](https://tingdeliu.github.io/en/vln-weekly-2026-09-05/) | 2026-08-26 ~ 2026-09-03 |
| [Issue 2](https://tingdeliu.github.io/en/vln-weekly-2026-08-30/) | 2026-08-20 ~ 2026-08-29 |
| [Issue 1](https://tingdeliu.github.io/en/vln-weekly-2026-08-22/) | 2026-08-13 ~ 2026-08-20 |

New digests appear in the embodied-navigation weekly section of the [Blog page](https://tingdeliu.github.io/en/blog/).

### Suggested Reading Path

1. Start with the [VLN Survey](https://tingdeliu.github.io/en/VLN-Survey/) for task definitions, datasets, and the evolution of navigation methods.
2. Read the [VLA Survey](https://tingdeliu.github.io/VLA-Survey/) (Chinese only) for unified modeling of vision, language, and robot actions.
3. Explore the [Spatial Intelligence Survey](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) and [World Models Survey](https://tingdeliu.github.io/World-Models-Survey/) (both Chinese only) for 3D understanding and environment prediction.
4. For implementation, move to the engineering sections of the [Complete ROS 2 Guide](https://tingdeliu.github.io/ROS2-Survey/) and [LLM Training Survey](https://tingdeliu.github.io/LLM-Training-Survey/) (both Chinese only).
5. Follow the [weekly digests](https://tingdeliu.github.io/en/blog/) for ongoing developments and reading priorities.

## Content Organization

```text
.
├── _posts/
│   ├── research/          # Long-form surveys and paper readings (categories: research)
│   ├── blog/              # Focused analyses and engineering posts (categories: blog)
│   └── weekly-reports/    # Embodied-navigation weekly digests (categories: weekly)
├── _translations/en/     # English articles, grouped under research/ and blog/
├── en/                   # English home, Research, and Blog entry pages
├── translations/         # Source/translation synchronization snapshots
├── images/               # Topic directories: vln / vla / vlm / wm / si / agent / llm-training ...
├── Analysis/             # Technical reports (VLN analysis, waypoint candidate methods)
├── paper_summary/        # Paper summary drafts (local directory, not version-controlled)
├── _layouts/             # Page layouts: default / post / page
├── _includes/            # Navigation, contents, feedback, footer, and other components
├── _sass/                # Theme style modules
├── js/                   # Article interaction scripts
├── home/ · research/ · blog/   # Project / Research / Blog index pages
├── tags/ · archive/       # Tag search and historical archives
├── .github/workflows/    # GitHub Actions deployment pipeline
├── _config.yml           # Site, navigation, and plugin configuration
├── README.md             # Default README in English
├── README_CN.md          # Chinese README
├── LICENSE               # Site code license (MIT)
├── LICENSE-CONTENT       # Original content license (CC BY 4.0) and third-party image exclusions
├── CONTRIBUTING.md       # Feedback, contribution process, and writing conventions
├── SECURITY.md           # Security reporting and copyright takedown requests
└── MAINTAINERS.md        # Maintainer details and community feedback records
```

Chinese articles use Jekyll's built-in `posts` collection with `/:title/` permalinks. The `categories` value (`research`, `blog`, or `weekly`) determines index-page placement; subdirectories under `_posts/` only organize source files. English articles use the separate `translations` collection with explicit `/en/.../` permalinks.

- The Chinese Research page groups cards into six areas: embodied navigation (VLN), embodied AI, large models and agents, machine learning foundations, robot systems and classical navigation, and engineering practice. A jump bar links to each group; unassigned articles appear under Other.
- The Chinese Blog page lists embodied-navigation weekly digests first, with issue numbers assigned in publication order, followed by technical essays.
- The English Research and Blog pages list available translations. English articles do not add duplicate entries to Chinese feeds, tags, or archives.

## Technical Implementation

- **Static site generation**: Jekyll 4.3, Ruby 3.2, and a custom theme derived from Jekyll Now.
- **Rendering**: kramdown (GFM), Rouge syntax highlighting, MathJax 3 equations, and Mermaid 10 diagrams. Mermaid is loaded from a CDN only when a page contains diagrams.
- **Jekyll plugins**: `jekyll-sitemap`, `jekyll-feed`, `jekyll-paginate`, and `jekyll-seo-tag`.
- **Reading experience**: A contents drawer with scroll-based section highlighting, a reading progress bar, one-click code copying, heading-link copying, image zoom, horizontal scrolling for wide tables, and a back-to-top button.
- **Paper filtering**: Interactive tag filters in the VLN and Embodied Agent paper collections support real-time filtering by multiple technical attributes. VLN filters also work across companion collections.
- **Images**: Most paper and conceptual raster images are served as WebP. New raster illustrations follow the lossless WebP convention; algorithm animations remain GIF and vector diagrams remain SVG.
- **Content feedback**: One-click Issue submission at the end of articles for errors, paper recommendations, and suggested changes, with related Issue status displayed.
- **Deployment**: Pushing to `main` triggers a GitHub Actions build and deployment to GitHub Pages.

## Local Development

### Requirements

- [Ruby 3.2](https://www.ruby-lang.org/en/documentation/installation/) (matching CI)
- Bundler

Check your environment:

```bash
ruby --version
bundle --version
```

### Install and Run

```bash
bundle install
bundle exec jekyll serve
```

The local site runs at [http://127.0.0.1:4000/](http://127.0.0.1:4000/) by default. Build output goes to `_site/`.

Create a production build:

```bash
JEKYLL_ENV=production bundle exec jekyll build
```

On Windows PowerShell, you can also use:

```powershell
ruby -S bundle exec jekyll serve
```

For a production build in PowerShell:

```powershell
$env:JEKYLL_ENV = "production"
ruby -S bundle exec jekyll build
```

### Content Maintenance

Before adding or editing articles, read [CONTRIBUTING.md](CONTRIBUTING.md) for front matter fields, image directories, Mermaid syntax, and equation conventions. One mandatory rule: **after editing any existing article under `_posts/`, update its front matter `date:` to the current date**, so article lists reflect recent updates.

## Content Feedback

If you find an error, a missing important paper, or a technical explanation that could be improved:

- [Open an Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new).
- Use the feedback links at the end of an article to report an error, recommend a paper, or suggest a change.

Include the original paper, official documentation, or reproducible materials where possible to support verification. Submission formats and the handling process are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

The maintainer aims to respond within **3 business days** and explains the decision whether or not a suggestion is adopted. The September 16, 2026 maintenance record lists 5 external feedback submissions, all addressed and closed; individual records are in [MAINTAINERS.md](MAINTAINERS.md#社区反馈处理记录).

## Licensing and Reuse

This repository uses **separate licenses for code and content**.

| Component | License | Scope |
| --- | --- | --- |
| **Code** | [MIT License](LICENSE) | Site implementation, including `_layouts/`, `_includes/`, `_sass/`, `js/`, `style.scss`, `_config.yml`, and `.github/workflows/` |
| **Original content** | [CC BY 4.0](LICENSE-CONTENT) | Article text under `_posts/`, author-created Mermaid diagrams and tables, and repository documentation |
| **Third-party paper figures** | ⚠️ **Excluded** | Illustrations in `images/` sourced from papers, project websites, and official documentation |

### ⚠️ Third-Party Paper Figures

Most images in `images/` are figures cited from academic papers, such as architecture diagrams and experimental results. **Copyright belongs to the original authors and publishers.** The site uses them for academic commentary and teaching. **It does not own their copyright and cannot sublicense them under CC BY 4.0 or any other terms.**

To reuse a paper figure, obtain permission from its author or publisher, or follow the paper's own license terms as indicated on its arXiv page. The basis for this site's use **does not automatically extend to your republication**. See Section 4 of [LICENSE-CONTENT](LICENSE-CONTENT) for details.

If you hold rights to an image and believe its use exceeds permitted quotation, contact the maintainer through the copyright channel in [SECURITY.md](SECURITY.md). After verification, the image will be removed or its permission statement updated promptly.

### Citing This Site

When reusing original content, provide attribution under CC BY 4.0:

```text
Author: Tingde Liu
Source: https://tingdeliu.github.io/<article-path>
License: CC BY 4.0
```

When citing this site's conclusions in academic writing, **also cite the relevant original papers**. This site's contribution is selection, translation, and commentary; credit for the methods belongs to the original authors.

## Contributing and Maintenance

| Document | Contents |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Content feedback and PR submissions, writing and image conventions, and local build checks |
| [MAINTAINERS.md](MAINTAINERS.md) | Maintainer identity and responsibilities, complete feedback records, response commitments, and publishing cadence |
| [SECURITY.md](SECURITY.md) | Private security reporting, the site's attack surface, and copyright takedown requests |

The site uses continuous deployment: pushes to `main` trigger a GitHub Actions build and publication, so the live site follows the latest deployed commit on `main`. Content is updated as needed, without versioned releases. To cite a specific version, link to the corresponding commit.

## Contact

- GitHub: [@TingdeLiu](https://github.com/TingdeLiu)
- LinkedIn: [Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- Email: [tingde.liu.luh@gmail.com](mailto:tingde.liu.luh@gmail.com)
