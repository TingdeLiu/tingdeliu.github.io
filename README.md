# Tingde Liu · Research Notes

Research surveys, paper readings, and engineering notes on embodied AI, vision-language navigation, and robot learning. Built with Jekyll and published through GitHub Pages.

[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-3DA639?style=flat-square)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-EF9421?style=flat-square)](LICENSE-CONTENT)

[中文站点](https://tingdeliu.github.io/) · [English edition](https://tingdeliu.github.io/en/) · [中文项目介绍](docs/README.zh-CN.md) · [Detailed overview](docs/overview.en.md)

Chinese is the source edition. English translations use a separate Jekyll collection, with language switching and source revision tracking.

## Explore

- [Research](https://tingdeliu.github.io/research/): VLN, VLA, multimodal learning, world models, agents, and robotics.
- [Blog](https://tingdeliu.github.io/blog/): embodied-navigation weekly digests and focused technical notes.
- [Projects](https://tingdeliu.github.io/home/) and [About](https://tingdeliu.github.io/about/).

## Local development

Install Ruby and Bundler, then run these commands from the repository root:

```bash
bundle install
python scripts/check_translations.py
bundle exec jekyll serve
```

Open [localhost:4000](http://127.0.0.1:4000/). On Windows, use `ruby -S bundle exec jekyll serve` if the Bundler executable is not on your path.

Before publishing:

```bash
python scripts/lint_posts.py
python scripts/check_translations.py
python scripts/check_translation_drafts.py
python -m unittest discover -s scripts/tests
bundle exec jekyll build
python scripts/check_site_styles.py
python scripts/check_english_site.py
```

Pushing to `main` runs these checks and deploys through [GitHub Actions](.github/workflows/deploy.yml).

## Repository layout

| Location | Purpose |
| --- | --- |
| `_posts/` | Chinese research articles, blog posts, and weekly digests |
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

The root contains the README, licenses, build configuration, and `index.html` required by pagination. Page URLs are independent of their source locations. The main Sass entry is `assets/css/style.scss` and still builds to `/style.css`.

See the [documentation index](docs/README.md) for directory conventions and maintenance guides.

## Contributing and licensing

Corrections, paper suggestions, and improvements are welcome. Read the [contribution guide](.github/CONTRIBUTING.md) before opening an issue or pull request. Existing articles under `_posts/` must update their front matter `date:` to the editing date.

See [maintainer and feedback records](docs/maintainers.md) and the [security and copyright reporting policy](.github/SECURITY.md).

Site code is licensed under [MIT](LICENSE); original writing is licensed under [CC BY 4.0](LICENSE-CONTENT). Third-party figures retain their original rights; see the content license for details.
