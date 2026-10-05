# 项目文档

根目录保留项目入口、许可证、构建配置和 Jekyll 分页入口。维护资料集中在这里，GitHub 社区规范放在 `.github/`。

## 阅读入口

| 文档 | 内容 |
| --- | --- |
| [中文项目介绍](README.zh-CN.md) | 研究方向、文章索引、技术实现与使用说明 |
| [English overview](overview.en.md) | Detailed project introduction and research index |
| [贡献指南](../.github/CONTRIBUTING.md) | 内容反馈、PR 流程与文章格式约定 |
| [安全与著作权政策](../.github/SECURITY.md) | 安全报告和图片下架请求 |
| [维护者与社区反馈](maintainers.md) | 维护职责、响应约定及历史处理记录 |
| [英文版维护指南](english-edition.md) | 翻译流程、版本同步与发布检查 |
| [研究分析记录](research-notes/) | 事实核验、方法比较及配图审计 |
| [翻译同步记录](translations/) | 已审阅源文的章节哈希和翻译进度 JSON |

## 文件归档约定

- 中文文章放在 `_posts/`，英文文章放在 `_translations/`；研究图片放在 `images/` 的主题目录。
- 网站入口页面放在 `pages/`。每个页面都声明 `permalink`，移动文件时保留网址；根目录 `index.html` 是分页入口。
- 前端资源放在 `assets/css/` 和 `assets/js/`；Sass 模块沿用 Jekyll 的 `_sass/`。主样式仍输出到 `/style.css`。
- 主 Sass 入口使用固定 `/style.css` 路径，必须保留 `sass.sourcemap: never`，避免调试文件继承同一路径覆盖 CSS。构建后运行 `python scripts/check_site_styles.py` 检查样式内容。
- 项目说明和工作记录放在 `docs/`；新的分析报告放在 `docs/research-notes/`，翻译同步快照放在 `docs/translations/`。
- 维护脚本放在 `scripts/`，现有回归检查放在 `scripts/tests/`。
- 临时下载、截图、备份和一次性脚本放在 `.cache/`。本地论文摘要沿用 `paper_summary/`，便于现有摘要工作流继续使用。

`docs/`、`scripts/`、`.cache/`、`paper_summary/` 和本地 `AGENTS.md` 均不发布到网站。Jekyll 构建产物 `_site/` 与运行缓存也不纳入版本控制。
