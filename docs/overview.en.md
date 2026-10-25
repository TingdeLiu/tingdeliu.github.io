# Tingde Liu · Research Notes

**English** | [简体中文](README.zh-CN.md)

**Research notes on embodied AI, vision-language navigation, and robot learning**

[![Website](https://img.shields.io/badge/Website-tingdeliu.github.io-2563EB?style=flat-square)](https://tingdeliu.github.io/en/)
[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.3-CC0000?style=flat-square&logo=jekyll&logoColor=white)](https://jekyllrb.com/)

[![Code License: MIT](https://img.shields.io/badge/Code-MIT-3DA639?style=flat-square)](../LICENSE)
[![Content License: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-EF9421?style=flat-square)](../LICENSE-CONTENT)

[English Edition](https://tingdeliu.github.io/en/) · [Research](https://tingdeliu.github.io/en/research/) · [Blog](https://tingdeliu.github.io/en/blog/) · [Projects](https://tingdeliu.github.io/home/) (Chinese only) · [About](https://tingdeliu.github.io/about/) (Chinese only)

## Overview

This is Tingde Liu's personal academic blog and research knowledge base, featuring surveys, paper readings, and engineering notes on embodied AI, robot navigation, and multimodal learning.

The site is intended for researchers, students, and engineers working in embodied AI and robotics, with a basic understanding of machine learning and deep learning.

Chinese is the source edition. English translations cover the research collection and published navigation digests. Use the [English home page](https://tingdeliu.github.io/en/), [Research index](https://tingdeliu.github.io/en/research/), or [Blog index](https://tingdeliu.github.io/en/blog/) to browse available articles. Articles support language switching and indicate when a translation needs updating. Some technical essays remain Chinese only. See the [English edition maintenance guide](english-edition.md) for synchronization checks.

The site aims to provide a systematic, continuously updated account of models, datasets, training methods, system architectures, benchmarks, and deployment on real robots. Long-form surveys establish the broader technical context, while shorter posts capture focused analyses and interim observations.

Created and maintained by [@TingdeLiu](https://github.com/TingdeLiu). For the current collection, browse the [Research](https://tingdeliu.github.io/en/research/) and [Blog](https://tingdeliu.github.io/en/blog/) indexes.

## Research Areas

The complete article index is organized by topic. Links use the English edition where available; other articles are marked **Chinese only**.

| Area | Article | Focus |
| --- | --- | --- |
| **Vision-Language Navigation** | [VLN Survey](https://tingdeliu.github.io/en/VLN-Survey/) | Task definitions, datasets, and benchmarks; the evolution from modular pipelines to end-to-end navigation agents |
| | [VLN Papers: Instruction Following](https://tingdeliu.github.io/en/VLN-Papers/) | Detailed readings of published or leaderboard-listed work, with leaderboards for three instruction-following task settings |
| | [VLN Papers: Goal Navigation and Extensions](https://tingdeliu.github.io/en/VLN-Papers-Extended/) | Goal-navigation leaderboards, recent preprints, and additional papers beyond the main instruction-following collection |
| **Vision-Language-Action Models** | [VLA Survey](https://tingdeliu.github.io/en/VLA-Survey/) | Robot policy learning, action generation, imitation learning, and reinforcement learning |
| | [VLA Paper Readings](https://tingdeliu.github.io/en/VLA-Papers/) | Individual paper analyses, RoboDojo leaderboard links, and comparisons under matching experimental settings |
| **Robot Navigation Systems** | [Classical Robot Navigation Survey](https://tingdeliu.github.io/en/Robot-Navigation-Survey/) | SLAM, localization and mapping, path planning, and motion control |
| | [Complete ROS 2 Guide](https://tingdeliu.github.io/en/ROS2-Survey/) | Communication models, lifecycle nodes, QoS, and real-robot deployment |
| **Multimodal and Spatial Intelligence** | [VLM Survey](https://tingdeliu.github.io/en/VLM-Survey/) | An overview of multimodal fusion methods in vision-language models |
| | [Spatial Intelligence Survey](https://tingdeliu.github.io/en/Spatial-Intelligence-Survey/) | 3D scene understanding, point clouds, depth estimation, and Gaussian Splatting |
| **World Models and Agents** | [World Models Survey](https://tingdeliu.github.io/en/World-Models-Survey/) | Environment modeling and prediction, video-generative world models, and embodied applications |
| | [AI Agent Survey](https://tingdeliu.github.io/en/AI-Agent-Survey/) | Reasoning, memory and skills, context engineering, multi-agent collaboration, and the MCP / WebMCP / A2A / MHS connectivity layers and security |
| | [Embodied Agent Paper Readings](https://tingdeliu.github.io/en/Embodied-Agent-Papers/) | Closed-loop runtimes (AgentOS / Harness), typed action abstractions, physical orchestration, scene-graph exit-code evaluation, and governance of self-evolution |
| | [Embodied Agent Survey](https://tingdeliu.github.io/en/Embodied-Agent-Harness-Survey/) | Integration of agent harnesses (physical guardrails, exit-code evaluation, online critics) with software systems engineering (microservices, ZeroMQ, fast/slow loops) |
| **Learning and Training Methods** | [LLM Training Survey](https://tingdeliu.github.io/en/LLM-Training-Survey/) | Pretraining, post-training, alignment, parallelism, and inference acceleration |
| | [Reinforcement Learning Survey](https://tingdeliu.github.io/en/Reinforcement-Learning-Survey/) | Algorithms from theoretical foundations to embodied AI applications |
| | [Deep Learning Survey](https://tingdeliu.github.io/en/Deep-Learning-Survey/) | Network architectures, optimization, and training techniques |
| | [Machine Learning Survey](https://tingdeliu.github.io/en/Machine-Learning-Survey/) | Classical models and statistical learning foundations |
| **Engineering Practice** | [Python Engineering Guide](https://tingdeliu.github.io/en/Python-Engineering-Survey/) | Environment and dependency boundaries, src layouts and pyproject, guarantees of uv lockfiles, quality and CI pipelines, CUDA dependency layers, ROS 2 and virtual-environment compatibility, and four levels of experiment reproducibility |

### Technical Blog

The workflow article is available in English; the additional essays listed below are currently in Chinese.

| Article | Topic |
| --- | --- |
| [Self Research: How This Blog Is Maintained with AI](https://tingdeliu.github.io/en/building-blog-with-skills/) | Literature collection, paper readings, survey maintenance, and automated publishing |
| [Jev: System One Models and Embodied Navigation](https://tingdeliu.github.io/Jev-Embodied-Navigation/) | Typed decisions, evidence boundaries, and research questions for navigation, memory, and agent collaboration |
| [RoboTTT: A Detailed Analysis](https://tingdeliu.github.io/RoboTTT-Fast-Weights/) | How TTT modules compress history into fast weights for long-term storage and real-time retrieval |
| [Graph Engineering](https://tingdeliu.github.io/graph-engineering/) | Agent graph orchestration and design patterns in the large-model era |
| [Loop Engineering](https://tingdeliu.github.io/loop-engineering/) | Closed-loop approaches to agent engineering |
| [Mixture-of-Transformers](https://tingdeliu.github.io/mixture-of-transformers/) | Modality decoupling and sparsity in multimodal foundation models |
| [Tree-Attention Training](https://tingdeliu.github.io/Tree-Attention-Decoding/) | How Robostral Navigate reduces VLN training tokens by 22× |
| [Harness Engineering](https://tingdeliu.github.io/Harness-Engineering/) | Execution environments and toolchains for agents |

### Embodied-Navigation Weekly Digests

Digests follow VLN and embodied agentic navigation through key findings, selected paper readings, and a categorized index. Experimental results retain source and evaluation context to help readers decide what to investigate next.

Browse the [English Blog index](https://tingdeliu.github.io/en/blog/) for the complete published series and future issues, or use the [Chinese edition](https://tingdeliu.github.io/blog/).

### Suggested Reading Path

1. Start with the [VLN Survey](https://tingdeliu.github.io/en/VLN-Survey/) for task definitions, datasets, and the evolution of navigation methods.
2. Read the [VLA Survey](https://tingdeliu.github.io/en/VLA-Survey/) for unified modeling of vision, language, and robot actions.
3. Explore the [Spatial Intelligence Survey](https://tingdeliu.github.io/en/Spatial-Intelligence-Survey/) and [World Models Survey](https://tingdeliu.github.io/en/World-Models-Survey/) for 3D understanding and environment prediction.
4. For implementation, move to the engineering sections of the [ROS 2 Core Architecture Guide](https://tingdeliu.github.io/en/ROS2-Survey/) and [LLM Training Survey](https://tingdeliu.github.io/en/LLM-Training-Survey/).
5. Follow the [weekly digests](https://tingdeliu.github.io/en/blog/) for ongoing developments and reading priorities.

## Research Workflow

AI assistants help collect literature, draft paper summaries, extract figures, update surveys, and validate formatting. The author selects papers and reviews key claims against original sources.

[Self Research: How This Blog Is Maintained with AI](https://tingdeliu.github.io/en/building-blog-with-skills/) explains the process. Reusable skills are maintained in the separate [Tyndall-Skills](https://github.com/TingdeLiu/Tyndall-Skills) repository.

## Content Organization

```text
.
├── _posts/               # research/, blog/, and weekly-reports/ Chinese sources
├── _translations/        # English article collection
├── pages/                # Entry pages, including pages/en/; explicit permalinks
├── assets/               # css/ and js/ frontend resources
├── images/               # Article figures grouped by research topic
├── _layouts/             # Page layouts: default / post / page
├── _includes/            # Shared navigation, article, and footer components
├── _sass/                # Sass modules; entry: assets/css/style.scss
├── _data/                # Shared UI labels, research groups, translation status
├── docs/                 # Project and maintenance documentation
│   ├── research-notes/   # Technical analyses, fact checks, and figure audits
│   └── translations/     # Source synchronization snapshots and progress records
├── scripts/              # Content validators and regression checks
├── .github/              # Deployment workflow, contribution and security policies
├── paper_summary/        # Local paper summary drafts; ignored by Git
├── .cache/               # Local temporary files; ignored by Git
├── _config.yml           # Site and build configuration
├── index.html            # Jekyll pagination entry
├── Gemfile · Gemfile.lock # Ruby dependencies
├── README.md             # Concise project entry and documentation links
├── LICENSE               # Site code license (MIT)
└── LICENSE-CONTENT       # Original content license (CC BY 4.0)
```

Chinese articles use Jekyll's built-in `posts` collection with `/:title/` permalinks. The `categories` value (`research`, `blog`, or `weekly`) determines index-page placement; subdirectories under `_posts/` only organize source files. English articles use the separate `translations` collection with explicit `/en/.../` permalinks.

- The Chinese Research page groups cards into six areas: embodied navigation (VLN), embodied manipulation (VLA), embodied agents, large models and spatial intelligence, machine learning foundations, and engineering practice. A jump bar links to each group; unassigned articles appear under Other.
- The Chinese Blog page combines navigation digests and technical essays, with filters for each category.
- The English Research and Blog pages list available translations. English articles do not add duplicate entries to Chinese feeds, tags, or archives.

## Technical Implementation

- **Static site generation**: Jekyll 4.3, Ruby 3.2, and a custom theme derived from Jekyll Now.
- **Rendering**: kramdown (GFM), Rouge syntax highlighting, MathJax 3 equations, and Mermaid 10 diagrams. Mermaid is loaded from a CDN only when a page contains diagrams, and each diagram is rendered lazily, only when it is scrolled within about two screens of the viewport.
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
- Python 3 for content and translation checks

Check your environment:

```bash
ruby --version
bundle --version
```

### Install and Run

```bash
bundle install
python scripts/check_translations.py
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

### Before Publishing

Run the same checks as GitHub Actions, from the repository root:

```bash
python scripts/lint_posts.py
python scripts/check_translations.py
python scripts/check_translation_drafts.py
python -m unittest discover -s scripts/tests
JEKYLL_ENV=production bundle exec jekyll build
python scripts/check_site_styles.py
python scripts/check_english_site.py
```

For the PowerShell build step, use the production command above: set `$env:JEKYLL_ENV = "production"`, then run `ruby -S bundle exec jekyll build`.

### Content Maintenance

Before adding or editing articles, read [CONTRIBUTING.md](../.github/CONTRIBUTING.md) for front matter fields, image directories, Mermaid syntax, and equation conventions. One mandatory rule: **after editing any existing article under `_posts/`, update its front matter `date:` to the current date**, so article lists reflect recent updates.

## Content Feedback

If you find an error, a missing important paper, or a technical explanation that could be improved:

- [Open an Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new).
- Use the feedback links at the end of an article to report an error, recommend a paper, or suggest a change.

Include the original paper, official documentation, or reproducible materials where possible to support verification. Submission formats and the handling process are documented in [CONTRIBUTING.md](../.github/CONTRIBUTING.md).

The maintainer aims to respond within **3 business days** and explains the decision whether or not a suggestion is adopted. The September 16, 2026 maintenance record lists 5 external feedback submissions, all addressed and closed; individual records are in [MAINTAINERS.md](maintainers.md#社区反馈处理记录).

## Licensing and Reuse

This repository uses **separate licenses for code and content**.

| Component | License | Scope |
| --- | --- | --- |
| **Code** | [MIT License](../LICENSE) | Site implementation, including `_layouts/`, `_includes/`, `_sass/`, `assets/js/`, `assets/css/style.scss`, `_config.yml`, and `.github/workflows/` |
| **Original content** | [CC BY 4.0](../LICENSE-CONTENT) | Article text under `_posts/`, author-created Mermaid diagrams and tables, and repository documentation |
| **Third-party paper figures** | ⚠️ **Excluded** | Illustrations in `images/` sourced from papers, project websites, and official documentation |

### ⚠️ Third-Party Paper Figures

Most images in `images/` are figures cited from academic papers, such as architecture diagrams and experimental results. **Copyright belongs to the original authors and publishers.** The site uses them for academic commentary and teaching. **It does not own their copyright and cannot sublicense them under CC BY 4.0 or any other terms.**

To reuse a paper figure, obtain permission from its author or publisher, or follow the paper's own license terms as indicated on its arXiv page. The basis for this site's use **does not automatically extend to your republication**. See Section 4 of [LICENSE-CONTENT](../LICENSE-CONTENT) for details.

If you hold rights to an image and believe its use exceeds permitted quotation, contact the maintainer through the copyright channel in [SECURITY.md](../.github/SECURITY.md). After verification, the image will be removed or its permission statement updated promptly.

### Citing This Site

When reusing original content, provide attribution under CC BY 4.0:

```text
Author: Tingde Liu
Source: https://tingdeliu.github.io/<article-path>
License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
Changes: <state whether the content was modified>
```

When citing this site's conclusions in academic writing, **also cite the relevant original papers**. This site's contribution is selection, translation, and commentary; credit for the methods belongs to the original authors.

## Contributing and Maintenance

| Document | Contents |
| --- | --- |
| [CONTRIBUTING.md](../.github/CONTRIBUTING.md) | Content feedback and PR submissions, writing and image conventions, and local build checks |
| [MAINTAINERS.md](maintainers.md) | Maintainer identity and responsibilities, complete feedback records, response commitments, and publishing cadence |
| [SECURITY.md](../.github/SECURITY.md) | Private security reporting, the site's attack surface, and copyright takedown requests |

The site uses continuous deployment: pushes to `main` trigger a GitHub Actions build and publication, so the live site follows the latest deployed commit on `main`. Content is updated as needed, without versioned releases. To cite a specific version, link to the corresponding commit.

## Contact

- GitHub: [@TingdeLiu](https://github.com/TingdeLiu)
- LinkedIn: [Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- Email: [tingde.liu.luh@gmail.com](mailto:tingde.liu.luh@gmail.com)
