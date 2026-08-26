<div align="center">

# Tingde Liu · Research Notes

**具身智能、视觉语言导航与机器人学习的中文研究笔记**

[![Website](https://img.shields.io/badge/Website-tingdeliu.github.io-2563EB?style=flat-square)](https://tingdeliu.github.io/)
[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.3-CC0000?style=flat-square&logo=jekyll&logoColor=white)](https://jekyllrb.com/)

[访问网站](https://tingdeliu.github.io/) · [研究综述](https://tingdeliu.github.io/research/) · [技术博客](https://tingdeliu.github.io/blog/) · [开源项目](https://tingdeliu.github.io/home/) · [关于作者](https://tingdeliu.github.io/about/)

</div>

## 项目简介

这是 Tingde Liu 的个人学术博客与研究知识库，主要记录具身智能、机器人导航和多模态学习方向的中文综述、论文笔记与工程实践。

内容主要面向具身智能与机器人方向的研究者、学生和工程师，默认读者具备基础的机器学习与深度学习知识。

本站以系统梳理和持续更新为目标：既关注模型与数据的发展脉络，也关注训练方法、系统架构、评测基准和真实机器人部署。长篇综述用于建立完整的技术脉络，短篇文章用于记录阶段性观察与专题分析。

目前收录 **14 篇研究综述**（约 4.3 万行 Markdown）、**6 篇技术博客**与 **800 余张**论文与概念配图，仍在持续维护。

## 研究方向

按方向组织的完整文章索引，全部为中文长文，可直接点击进入。

| 方向 | 文章 | 内容重点 |
| --- | --- | --- |
| **视觉语言导航** | [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) | 任务定义、数据集与基准，从模块化流水线到端到端导航智能体的演进 |
| | [VLN 经典论文](https://tingdeliu.github.io/VLN-Papers/) | 60 篇已发表或已上榜工作的精读，附三类任务设定的性能排行榜 |
| | [VLN 扩展论文](https://tingdeliu.github.io/VLN-Papers-Extended/) | 尚未进入排行榜的近期预印本与扩展增补，按年份倒序 |
| **视觉语言动作模型** | [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) | 机器人策略学习、动作生成、模仿学习与强化学习的路线梳理 |
| **机器人导航系统** | [传统导航综述](https://tingdeliu.github.io/Robot-Navigation-Survey/) | SLAM、定位建图、路径规划与运动控制等经典方法 |
| | [ROS 2 完全指南](https://tingdeliu.github.io/ROS2-Survey/) | 通信模型、生命周期节点、QoS 与真实机器人部署实战 |
| **多模态与空间智能** | [VLM 综述](https://tingdeliu.github.io/VLM-Survey/) | 视觉语言模型的多模态融合方法全景 |
| | [空间智能综述](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) | 三维场景理解、点云、深度估计与 Gaussian Splatting |
| **世界模型与智能体** | [世界模型综述](https://tingdeliu.github.io/World-Models-Survey/) | 环境建模与预测、视频生成式世界模型及其在具身任务中的应用 |
| | [AI Agent 综述](https://tingdeliu.github.io/AI-Agent-Survey/) | 自主推理、规划、记忆、工具调用与闭环执行 |
| **学习与训练方法** | [LLM 训练综述](https://tingdeliu.github.io/LLM-Training-Survey/) | 预训练、后训练、对齐、并行策略与推理加速 |
| | [强化学习综述](https://tingdeliu.github.io/Reinforcement-Learning-Survey/) | 从理论基础到具身智能场景的算法全景 |
| | [深度学习综述](https://tingdeliu.github.io/Deep-Learning-Survey/) | 网络结构、优化方法与训练技巧的系统梳理 |
| | [机器学习综述](https://tingdeliu.github.io/Machine-Learning-Survey/) | 经典模型与统计学习基础 |

### 技术博客

| 文章 | 主题 |
| --- | --- |
| [RoboTTT 深度解析](https://tingdeliu.github.io/RoboTTT-Fast-Weights/) | TTT 模块如何把历史压缩进 Fast Weights，实现长时程存储与实时检索 |
| [Graph Engineering](https://tingdeliu.github.io/graph-engineering/) | 大模型时代的智能体图拓扑编排与设计模式 |
| [Loop Engineering](https://tingdeliu.github.io/loop-engineering/) | Agent 工程化的下一代闭环范式 |
| [树状注意力训练](https://tingdeliu.github.io/Tree-Attention-Decoding/) | Robostral Navigate 如何将 VLN 训练 Token 压缩 22× |
| [Mixture-of-Transformers](https://tingdeliu.github.io/mixture-of-transformers/) | 多模态基础模型的模态解耦与稀疏化演进 |
| [Harness Engineering](https://tingdeliu.github.io/Harness-Engineering/) | 面向智能体的执行环境与工具链设计 |

### 推荐阅读路径

1. 从 [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) 了解视觉语言导航的任务定义、数据集和技术演进。
2. 阅读 [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) 了解视觉、语言与机器人动作的统一建模。
3. 通过 [空间智能综述](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) 和 [世界模型综述](https://tingdeliu.github.io/World-Models-Survey/) 扩展到三维理解与环境预测。
4. 需要动手实现时，再进入 [ROS 2 完全指南](https://tingdeliu.github.io/ROS2-Survey/) 与 [LLM 训练综述](https://tingdeliu.github.io/LLM-Training-Survey/) 的工程部分。

## 内容组织

```text
.
├── _posts/
│   ├── research/          # 长篇综述、论文精读与技术笔记（categories: research）
│   └── blog/              # 专题解析、工程文章与周报（categories: blog）
├── images/                # 按研究主题分目录：vln / vla / vlm / wm / si / agent / llm-training ...
├── paper_summary/         # 论文摘要草稿，供正文引用，不单独发布
├── _layouts/              # 页面布局：default / post / page
├── _includes/             # 导航、目录、反馈、页脚等页面组件
├── _sass/                 # 主题样式模块
├── js/                    # 文章交互脚本
├── home/ · research/ · blog/   # Project / Research / Blog 聚合页
├── tags/ · archive/       # 标签检索页与历史归档页
├── .github/workflows/     # GitHub Actions 部署流水线
├── _config.yml            # 站点、导航与插件配置
└── AGENTS.md              # 仓库写作与维护规范
```

文章使用 Jekyll 内置的 `posts` 集合，永久链接为 `/:title/`。`categories: research` 与 `categories: blog` 是内容分类依据，子目录仅用于维护源文件。

## 技术实现

- **静态站点生成**：Jekyll 4.3，Ruby 3.2，定制主题（源自 Jekyll Now）
- **内容渲染**：kramdown（GFM）+ Rouge 代码高亮 + MathJax 3 公式 + Mermaid 10 图表，Mermaid 仅在页面含图表时按需加载 CDN
- **Jekyll 插件**：`jekyll-sitemap`、`jekyll-feed`、`jekyll-paginate`、`jekyll-seo-tag`
- **阅读体验**：章节目录抽屉（随滚动高亮当前小节）、顶部阅读进度条、代码块一键复制、标题锚点复制、配图点击放大、宽表格横向滚动、返回顶部
- **论文检索**：两篇 VLN 论文精读内置筛选栏，支持按标签过滤并跨篇联动
- **图片资源**：正文配图统一转为无损 WebP 交付，兼顾清晰度与加载速度
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

新增或修改文章前，请先阅读 [AGENTS.md](AGENTS.md)，其中记录了 front matter 字段、图片归档目录、Mermaid 与公式写法等约定。其中一条硬性规则：**修改 `_posts/` 下任何已有文章后，必须把 front matter 的 `date:` 更新为当天日期**，以保证列表按最近更新排序。

## 内容反馈

如果你发现内容错误、遗漏的重要论文或可以改进的技术表述，欢迎：

- [提交 Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new)
- 使用文章末尾的反馈入口提交错误报告、论文推荐或修改建议

提交反馈时请尽量附上原始论文、官方文档或可复现资料，方便核验与更新。

## 使用与引用

本仓库目前未附开源许可证。除非文件另有说明，仓库内容不构成对复制、修改或再分发的授权。部分论文图片与资料的版权属于原作者或原项目，使用时请遵循对应来源的许可要求。

引用本站研究笔记时，建议同时引用相关原始论文并附上对应文章链接。如需大段转载或复用站点主题，请先联系作者。

## 联系方式

- GitHub: [@TingdeLiu](https://github.com/TingdeLiu)
- LinkedIn: [Tingde Liu](https://www.linkedin.com/in/tingde-liu-379818270/)
- Email: [tingde.liu.luh@gmail.com](mailto:tingde.liu.luh@gmail.com)
