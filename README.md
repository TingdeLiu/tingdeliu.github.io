# Tingde Liu · Research Notes

Research notes on **embodied AI, vision-language navigation, and robot learning** — surveys, in-depth paper readings, and engineering practice by Tingde Liu.

[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-3DA639?style=flat-square)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-EF9421?style=flat-square)](LICENSE-CONTENT)

[**Read in English**](https://tingdeliu.github.io/en/) · [**阅读中文站点**](https://tingdeliu.github.io/) · [中文 README](docs/README.zh-CN.md)

This is a personal research blog for researchers, students, and engineers working in embodied AI and robotics. Articles connect task definitions, methods, datasets, and evaluation with practical questions about training and deployment.

Chinese is the source edition, with English translations of the research collection and published navigation digests. Articles link between languages and indicate when a translation needs updating. Browse the [English research index](https://tingdeliu.github.io/en/research/) or the [中文研究索引](https://tingdeliu.github.io/research/) for the full collection.

## Start reading

| If you want to… | Start here |
| --- | --- |
| Understand vision-language navigation | [VLN Survey](https://tingdeliu.github.io/en/VLN-Survey/) — tasks, benchmarks, and the evolution of navigation methods |
| Compare navigation papers and results | [Instruction Following](https://tingdeliu.github.io/en/VLN-Papers/) · [Goal and Agentic Navigation](https://tingdeliu.github.io/en/VLN-Papers-Extended/) — paper readings, task-specific leaderboards, and tag filters |
| Explore robot manipulation | [VLA Survey](https://tingdeliu.github.io/en/VLA-Survey/) · [VLA Paper Readings](https://tingdeliu.github.io/en/VLA-Papers/) |
| Study embodied agent systems | [Harness and Runtime Architecture](https://tingdeliu.github.io/en/Embodied-Agent-Harness-Survey/) · [Embodied Agent Paper Readings](https://tingdeliu.github.io/en/Embodied-Agent-Papers/) |
| Move from reading to implementation | [ROS 2 Architecture](https://tingdeliu.github.io/en/ROS2-Survey/) · [Python Engineering](https://tingdeliu.github.io/en/Python-Engineering-Survey/) |
| Follow recent navigation research | [Embodied Navigation Weekly](https://tingdeliu.github.io/en/blog/) — key findings, selected papers, and reading priorities |

## Research collection

| Area | Surveys and paper collections |
| --- | --- |
| Embodied navigation | [VLN](https://tingdeliu.github.io/en/VLN-Survey/) · [Classical Robot Navigation](https://tingdeliu.github.io/en/Robot-Navigation-Survey/) |
| Embodied manipulation | [VLA](https://tingdeliu.github.io/en/VLA-Survey/) · [World Models](https://tingdeliu.github.io/en/World-Models-Survey/) |
| Embodied agents | [AI Agents](https://tingdeliu.github.io/en/AI-Agent-Survey/) · [Harness and Runtime Architecture](https://tingdeliu.github.io/en/Embodied-Agent-Harness-Survey/) · [Paper Readings](https://tingdeliu.github.io/en/Embodied-Agent-Papers/) |
| Large models and spatial intelligence | [LLM Training](https://tingdeliu.github.io/en/LLM-Training-Survey/) · [Vision-Language Models](https://tingdeliu.github.io/en/VLM-Survey/) · [Spatial Intelligence](https://tingdeliu.github.io/en/Spatial-Intelligence-Survey/) |
| Learning foundations | [Machine Learning](https://tingdeliu.github.io/en/Machine-Learning-Survey/) · [Deep Learning](https://tingdeliu.github.io/en/Deep-Learning-Survey/) · [Reinforcement Learning](https://tingdeliu.github.io/en/Reinforcement-Learning-Survey/) |
| Engineering practice | [Python Engineering](https://tingdeliu.github.io/en/Python-Engineering-Survey/) · [ROS 2](https://tingdeliu.github.io/en/ROS2-Survey/) |

The [Blog](https://tingdeliu.github.io/en/blog/) combines navigation digests with technical essays. Additional essays are available in the [Chinese edition](https://tingdeliu.github.io/blog/). [Projects](https://tingdeliu.github.io/home/) and [About the author](https://tingdeliu.github.io/about/) are available in Chinese.

## How this blog is maintained

The workflow connects literature collection, paper summaries, figure extraction, survey updates, and automated publishing checks. AI assistants help with retrieval, drafting, formatting, and validation; the author selects papers and reviews claims against original sources.

Read [Self Research: How This Blog Is Maintained with AI](https://tingdeliu.github.io/en/building-blog-with-skills/) for the process and reusable tools. The skills are maintained in a separate repository, [Tyndall-Skills](https://github.com/TingdeLiu/Tyndall-Skills).

## Local development

The site uses Jekyll with a custom Jekyll Now-derived theme and GitHub Pages hosting. Use **Ruby 3.2** (matching CI), **Bundler**, and **Python 3** for content checks. From the repository root:

```bash
bundle install
python scripts/check_translations.py
bundle exec jekyll serve
```

Open [localhost:4000](http://127.0.0.1:4000/). On Windows, use `ruby -S bundle exec jekyll serve` if the Bundler executable is not on your path.

Before publishing, run the same checks as CI:

```bash
python scripts/lint_posts.py
python scripts/check_translations.py
python scripts/check_translation_drafts.py
python -m unittest discover -s scripts/tests
JEKYLL_ENV=production bundle exec jekyll build
python scripts/check_site_styles.py
python scripts/check_english_site.py
```

For the build step in PowerShell, set `$env:JEKYLL_ENV = "production"`, then run `ruby -S bundle exec jekyll build`.

Pushing to `main` runs these checks and deploys through [GitHub Actions](.github/workflows/deploy.yml). See the [contribution guide](.github/CONTRIBUTING.md) for article formatting and maintenance rules.

## Repository layout

| Location | Purpose |
| --- | --- |
| `_posts/` | Chinese articles in `research/`, `blog/`, and `weekly-reports/` |
| `_translations/` | English article collection |
| `pages/` | Chinese and English entry pages, with explicit public permalinks |
| `assets/` | Frontend stylesheets and JavaScript |
| `images/` | Article figures grouped by research topic |
| `_layouts/`, `_includes/`, `_sass/` | Jekyll layouts, shared components, and Sass modules |
| `_data/` | Shared UI labels, research groups, and generated translation status |
| `docs/` | Project guides, maintenance records, research notes, and translation snapshots |
| `scripts/` | Content validation and regression checks |
| `.github/` | Deployment workflow, contribution guide, and security policy |
| `paper_summary/`, `.cache/` | Local paper drafts and temporary working files; ignored by Git and excluded from the site |

See the [documentation index](docs/README.md) for maintenance guides, the [中文项目介绍](docs/README.zh-CN.md) for Chinese readers, and the [detailed English overview](docs/overview.en.md) for the full article index and site features.

## Contributing and licensing

Found an error or a paper worth adding? [Open an issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new) or use the feedback links at the end of an article. Include the article section and an original paper, official document, or reproducible result where possible. See the [contribution guide](.github/CONTRIBUTING.md) for details.

See [maintainer and feedback records](docs/maintainers.md) and the [security and copyright reporting policy](.github/SECURITY.md).

Site code is licensed under [MIT](LICENSE); original writing is licensed under [CC BY 4.0](LICENSE-CONTENT). Third-party figures retain their original rights; see the content license for details.
