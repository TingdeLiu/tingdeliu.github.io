<div align="center">

# Tingde Liu · Research Notes

[English](README.md) | **简体中文**

**具身智能、视觉语言导航与机器人学习的中文研究笔记**

[![Website](https://img.shields.io/badge/Website-tingdeliu.github.io-2563EB?style=flat-square)](https://tingdeliu.github.io/)
[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.3-CC0000?style=flat-square&logo=jekyll&logoColor=white)](https://jekyllrb.com/)

[![Stars](https://img.shields.io/github/stars/TingdeLiu/tingdeliu.github.io?style=flat-square&logo=github&color=F5A623)](https://github.com/TingdeLiu/tingdeliu.github.io/stargazers)
[![Code License: MIT](https://img.shields.io/badge/Code-MIT-3DA639?style=flat-square)](LICENSE)
[![Content License: CC BY 4.0](https://img.shields.io/badge/Content-CC%20BY%204.0-EF9421?style=flat-square)](LICENSE-CONTENT)

[访问网站](https://tingdeliu.github.io/) · [研究综述](https://tingdeliu.github.io/research/) · [技术博客](https://tingdeliu.github.io/blog/) · [开源项目](https://tingdeliu.github.io/home/) · [关于作者](https://tingdeliu.github.io/about/)

</div>

## 项目简介

这是 Tingde Liu 的个人学术博客与研究知识库，主要记录具身智能、机器人导航和多模态学习方向的中文综述、论文笔记与工程实践。

内容主要面向具身智能与机器人方向的研究者、学生和工程师，默认读者具备基础的机器学习与深度学习知识。

英文版已提供完整的 VLN 综述、两篇 VLN 论文精读和六期已发布的具身导航周报，以及[英文阅读入口](https://tingdeliu.github.io/en/)、[研究索引](https://tingdeliu.github.io/en/research/)和[博客索引](https://tingdeliu.github.io/en/blog/)，中英文文章可以切换。其他长文将逐步翻译，尚未翻译的链接会标注 Chinese only。翻译与同步检查流程见 [英文版维护说明](docs/english-edition.md)。

本站以系统梳理和持续更新为目标：既关注模型与数据的发展脉络，也关注训练方法、系统架构、评测基准和真实机器人部署。长篇综述用于建立完整的技术脉络，短篇文章用于记录阶段性观察与专题分析。

截至 **2026-10-03**，收录 **18 篇研究综述与论文精读**、**7 篇技术博客**、**6 期已发布的具身导航周报**与 **1000 余个图片文件**。已发布中文文章源文件合计约 **5.6 万行 Markdown**，英文译文另计，仍在持续维护。

项目由 [@TingdeLiu](https://github.com/TingdeLiu) 创建，自 2026-01-04 起持续更新。2026-09-16 的维护记录收录了 **5 条社区反馈，均已处理关闭**，完整记录见 [MAINTAINERS.md](MAINTAINERS.md)。

## 研究方向

按方向组织的完整文章索引，全部为中文长文，可直接点击进入。

| 方向 | 文章 | 内容重点 |
| --- | --- | --- |
| **视觉语言导航** | [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) | 任务定义、数据集与基准，从模块化流水线到端到端导航智能体的演进 |
| | [VLN 论文精读：指令跟随篇](https://tingdeliu.github.io/VLN-Papers/) | 已发表或已上榜工作的精读，附三类指令跟随任务设定的性能排行榜 |
| | [VLN 论文精读：目标导航与扩展篇](https://tingdeliu.github.io/VLN-Papers-Extended/) | 目标导航性能排行榜，以及指令跟随主篇之外的近期预印本与扩展论文 |
| **视觉语言动作模型** | [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) | 机器人策略学习、动作生成、模仿学习与强化学习的路线梳理 |
| | [VLA 论文精读](https://tingdeliu.github.io/VLA-Papers/) | 独立论文解析、RoboDojo 排行榜入口与同设置实验对照 |
| **机器人导航系统** | [传统导航综述](https://tingdeliu.github.io/Robot-Navigation-Survey/) | SLAM、定位建图、路径规划与运动控制等经典方法 |
| | [ROS 2 完全指南](https://tingdeliu.github.io/ROS2-Survey/) | 通信模型、生命周期节点、QoS 与真实机器人部署实战 |
| **多模态与空间智能** | [VLM 综述](https://tingdeliu.github.io/VLM-Survey/) | 视觉语言模型的多模态融合方法全景 |
| | [空间智能综述](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) | 三维场景理解、点云、深度估计与 Gaussian Splatting |
| **世界模型与智能体** | [世界模型综述](https://tingdeliu.github.io/World-Models-Survey/) | 环境建模与预测、视频生成式世界模型及其在具身任务中的应用 |
| | [AI Agent 综述](https://tingdeliu.github.io/AI-Agent-Survey/) | 推理范式、记忆与技能、上下文工程、多智能体协作，以及 MCP / WebMCP / A2A / MHS 四层连接协议与安全 |
| | [Embodied Agent 论文精读](https://tingdeliu.github.io/Embodied-Agent-Papers/) | 具身智能体闭环运行时（AgentOS / Harness）、类型化动作抽象、物理编排、场景图退出码评估与自演化治理 |
| | [Embodied Agent 综述](https://tingdeliu.github.io/Embodied-Agent-Harness-Survey/) | 上层 Agent Harness（物理护栏/退出码评估/在线Critic）与下层软件系统工程（微服务/ZeroMQ/快慢双循环）深度协同 |
| **学习与训练方法** | [LLM 训练综述](https://tingdeliu.github.io/LLM-Training-Survey/) | 预训练、后训练、对齐、并行策略与推理加速 |
| | [强化学习综述](https://tingdeliu.github.io/Reinforcement-Learning-Survey/) | 从理论基础到具身智能场景的算法全景 |
| | [深度学习综述](https://tingdeliu.github.io/Deep-Learning-Survey/) | 网络结构、优化方法与训练技巧的系统梳理 |
| | [机器学习综述](https://tingdeliu.github.io/Machine-Learning-Survey/) | 经典模型与统计学习基础 |
| **工程实践** | [Python 工程实践指南](https://tingdeliu.github.io/Python-Engineering-Survey/) | 环境与依赖边界、src 布局与 pyproject、uv 锁文件的真实保证范围、质量与 CI 流水线，以及 CUDA 依赖分层、ROS 2 与虚拟环境的兼容边界、实验可复现的四个层次 |

### 技术博客

| 文章 | 主题 |
| --- | --- |
| [Jev：System One 模型与具身导航](https://tingdeliu.github.io/Jev-Embodied-Navigation/) | 类型化判断、证据边界，以及导航决策、记忆管理与 Agent 协作的研究问题 |
| [RoboTTT 深度解析](https://tingdeliu.github.io/RoboTTT-Fast-Weights/) | TTT 模块如何把历史压缩进 Fast Weights，实现长时程存储与实时检索 |
| [Graph Engineering](https://tingdeliu.github.io/graph-engineering/) | 大模型时代的智能体图拓扑编排与设计模式 |
| [Loop Engineering](https://tingdeliu.github.io/loop-engineering/) | Agent 工程化的下一代闭环范式 |
| [Mixture-of-Transformers](https://tingdeliu.github.io/mixture-of-transformers/) | 多模态基础模型的模态解耦与稀疏化演进 |
| [树状注意力训练](https://tingdeliu.github.io/Tree-Attention-Decoding/) | Robostral Navigate 如何将 VLN 训练 Token 压缩 22× |
| [Harness Engineering](https://tingdeliu.github.io/Harness-Engineering/) | 面向智能体的执行环境与工具链设计 |

### 具身导航周报

按周汇总 arXiv 新论文与中文社区解读，按「本期结论 → 优先阅读清单 → 重点工作分析 → 可迁移方法 → 分类速览 → 趋势判断」六段组织。只保留结论、证据与阅读优先级，不做论文清单堆砌；无法核实的数字会明确标注来源与口径。

| 期号 | 覆盖区间 |
| --- | --- |
| [第 6 期](https://tingdeliu.github.io/vln-weekly-2026-09-27/) | 2026-09-16 ~ 2026-09-24 |
| [第 5 期](https://tingdeliu.github.io/vln-weekly-2026-09-19/) | 2026-09-09 ~ 2026-09-17 |
| [第 4 期](https://tingdeliu.github.io/vln-weekly-2026-09-12/) | 2026-09-02 ~ 2026-09-10 |
| [第 3 期](https://tingdeliu.github.io/vln-weekly-2026-09-05/) | 2026-08-26 ~ 2026-09-03 |
| [第 2 期](https://tingdeliu.github.io/vln-weekly-2026-08-30/) | 2026-08-20 ~ 2026-08-29 |
| [第 1 期](https://tingdeliu.github.io/vln-weekly-2026-08-22/) | 2026-08-13 ~ 2026-08-20 |

后续新增周报同样发布在 [Blog 页](https://tingdeliu.github.io/blog/) 的「具身导航周报」分区。

### 推荐阅读路径

1. 从 [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) 了解视觉语言导航的任务定义、数据集和技术演进。
2. 阅读 [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) 了解视觉、语言与机器人动作的统一建模。
3. 通过 [空间智能综述](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) 和 [世界模型综述](https://tingdeliu.github.io/World-Models-Survey/) 扩展到三维理解与环境预测。
4. 需要动手实现时，再进入 [ROS 2 完全指南](https://tingdeliu.github.io/ROS2-Survey/) 与 [LLM 训练综述](https://tingdeliu.github.io/LLM-Training-Survey/) 的工程部分。
5. 想持续跟进最新进展，直接看[具身导航周报](https://tingdeliu.github.io/blog/)，每期只给结论与阅读优先级。

## 内容组织

```text
.
├── _posts/
│   ├── research/          # 长篇综述与论文精读（categories: research）
│   ├── blog/              # 专题解析与工程文章（categories: blog）
│   └── weekly-reports/    # 具身导航周报（categories: weekly）
├── images/                # 按研究主题分目录：vln / vla / vlm / wm / si / agent / llm-training ...
├── Analysis/              # 专题技术分析报告（VLN 技术分析、Waypoint 候选方法）
├── paper_summary/         # 论文摘要草稿，供正文引用（本地目录，未纳入版本控制）
├── _layouts/              # 页面布局：default / post / page
├── _includes/             # 导航、目录、反馈、页脚等页面组件
├── _sass/                 # 主题样式模块
├── js/                    # 文章交互脚本
├── home/ · research/ · blog/   # Project / Research / Blog 聚合页
├── tags/ · archive/       # 标签检索页与历史归档页
├── .github/workflows/     # GitHub Actions 部署流水线
├── _config.yml            # 站点、导航与插件配置
├── LICENSE                # 站点代码许可（MIT）
├── LICENSE-CONTENT        # 原创内容许可（CC BY 4.0）+ 第三方图片例外声明
├── CONTRIBUTING.md        # 反馈与贡献流程、写作约定
├── SECURITY.md            # 安全报告渠道与著作权下架请求
└── MAINTAINERS.md         # 维护者说明与社区反馈处理记录
```

文章使用 Jekyll 内置的 `posts` 集合，永久链接为 `/:title/`。`categories` 取 `research` / `blog` / `weekly` 三值，是聚合页分区的唯一依据，`_posts/` 下的子目录仅用于维护源文件。

- Research 页以卡片网格按「具身导航（VLN）／具身智能／大模型与 Agent／机器学习基础／机器人系统与传统导航／工程实践」六组展示，顶部有分组跳转栏，未归组的新文章自动落入「其他」
- Blog 页先列「具身导航周报」（按发布顺序自动编期号），再列「技术随笔」

## 技术实现

- **静态站点生成**：Jekyll 4.3，Ruby 3.2，定制主题（源自 Jekyll Now）
- **内容渲染**：kramdown（GFM）+ Rouge 代码高亮 + MathJax 3 公式 + Mermaid 10 图表，Mermaid 仅在页面含图表时按需加载 CDN，且每张图在滚动到视口附近（约两屏）时才渲染
- **Jekyll 插件**：`jekyll-sitemap`、`jekyll-feed`、`jekyll-paginate`、`jekyll-seo-tag`
- **阅读体验**：章节目录抽屉（随滚动高亮当前小节）、顶部阅读进度条、代码块一键复制、标题锚点复制、配图点击放大、宽表格横向滚动、返回顶部
- **论文检索**：VLN 与 Embodied Agent 论文精读内置交互式标签筛选栏，支持按多维技术特征实时过滤（VLN 篇支持跨篇联动）
- **图片资源**：论文与概念位图主要以 WebP 交付；新增位图按无损 WebP 约定维护，算法演示动画保留 GIF，矢量示意图保留 SVG
- **内容反馈**：文章末尾一键提交 Issue（报告错误 / 推荐论文 / 修改建议），并回显相关 Issue 状态
- **部署**：推送 `main` 分支后由 GitHub Actions 构建并发布到 GitHub Pages

## 本地开发

### 环境要求

- [Ruby 3.2](https://www.ruby-lang.org/en/documentation/installation/)（与 CI 环境一致）
- Bundler

确认环境：

```bash
ruby --version
bundle --version
```

### 安装与运行

```bash
bundle install
bundle exec jekyll serve
```

本地站点默认运行在 [http://127.0.0.1:4000/](http://127.0.0.1:4000/)，构建产物输出到 `_site/`。

生成生产构建：

```bash
JEKYLL_ENV=production bundle exec jekyll build
```

Windows PowerShell 中也可以使用：

```powershell
ruby -S bundle exec jekyll serve
```

PowerShell 生产构建：

```powershell
$env:JEKYLL_ENV = "production"
ruby -S bundle exec jekyll build
```

### 内容维护约定

新增或修改文章前，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，其中记录了 front matter 字段、图片归档目录、Mermaid 与公式写法等约定。其中一条硬性规则：**修改 `_posts/` 下任何已有文章后，必须把 front matter 的 `date:` 更新为当天日期**，以保证列表按最近更新排序。

## 内容反馈

如果你发现内容错误、遗漏的重要论文或可以改进的技术表述，欢迎：

- [提交 Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new)
- 使用文章末尾的反馈入口提交错误报告、论文推荐或修改建议

提交反馈时请尽量附上原始论文、官方文档或可复现资料，方便核验与更新。详细的提交格式与处理流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

维护者以 **3 个工作日内**首次回复为目标，无论是否采纳都会说明理由。2026-09-16 的维护记录列出的 5 条外部反馈均已处理关闭，逐条记录见 [MAINTAINERS.md](MAINTAINERS.md#社区反馈处理记录)。

## 许可与使用

本仓库采用**双许可**：代码与内容分开授权。

| 部分 | 许可证 | 覆盖范围 |
| --- | --- | --- |
| **代码** | [MIT License](LICENSE) | `_layouts/` · `_includes/` · `_sass/` · `js/` · `style.scss` · `_config.yml` · `.github/workflows/` 等站点实现 |
| **原创内容** | [CC BY 4.0](LICENSE-CONTENT) | `_posts/` 下全部文章正文、作者绘制的 Mermaid 图表与表格，以及仓库文档 |
| **第三方论文图片** | ⚠️ **不在授权范围内** | `images/` 下取自论文、项目主页与官方文档的插图 |

### ⚠️ 关于第三方论文图片

`images/` 下的绝大多数图片是从学术论文中引用的插图（架构图、实验结果图等），**著作权归原论文作者及出版方所有**。本站基于学术评述与教学目的引用，**不拥有这些图片的著作权，因此无权以 CC BY 4.0 或任何其他条款转授权**。

如需复用某张论文配图，请直接向原作者或出版方获取授权，或遵循该论文自身的许可条款（arXiv 页面会标注）。本站的合理引用基础**不会随转载传递给你**。详细说明见 [LICENSE-CONTENT](LICENSE-CONTENT) 第 4 节。

若你是某张图片的权利人并认为使用超出合理引用范围，请按 [SECURITY.md](SECURITY.md) 中的著作权联系方式告知，核实后会及时移除或补充授权说明。

### 引用本站

复用原创内容时请按 CC BY 4.0 署名：

```
作者：Tingde Liu
来源：https://tingdeliu.github.io/<文章链接>
许可：CC BY 4.0
```

学术写作中引用本站结论时，**请同时引用相关原始论文**——本站的贡献在于筛选、翻译与评述，方法本身的功劳属于原作者。

## 参与与维护

| 文档 | 内容 |
| --- | --- |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 如何提交内容反馈与 PR、写作与图片约定、本地构建自查清单 |
| [MAINTAINERS.md](MAINTAINERS.md) | 维护者身份与职责、完整的社区反馈处理记录、响应承诺、发布节奏 |
| [SECURITY.md](SECURITY.md) | 安全问题的私密报告渠道、真实攻击面说明、著作权下架请求 |

站点采用持续部署：推送 `main` 后由 GitHub Actions 构建并发布，线上版本始终对应 `main` 的最新提交。内容按需更新，不做版本化发布；需要引用某个确定时刻的版本时，可直接引用对应的 commit 链接。

## 联系方式

- GitHub: [@TingdeLiu](https://github.com/TingdeLiu)
- LinkedIn: [Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- Email: [tingde.liu.luh@gmail.com](mailto:tingde.liu.luh@gmail.com)
