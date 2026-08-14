<div align="center">

# Tingde Liu · Research Notes

**具身智能、视觉语言导航与机器人学习的中文研究笔记**

[![Website](https://img.shields.io/badge/Website-tingdeliu.github.io-2563EB?style=flat-square)](https://tingdeliu.github.io/)
[![Deploy](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/TingdeLiu/tingdeliu.github.io/actions/workflows/deploy.yml)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.3-CC0000?style=flat-square&logo=jekyll&logoColor=white)](https://jekyllrb.com/)

[访问网站](https://tingdeliu.github.io/) · [研究综述](https://tingdeliu.github.io/research/) · [技术博客](https://tingdeliu.github.io/blog/) · [关于作者](https://tingdeliu.github.io/about/)

</div>

## 项目简介

这是 Tingde Liu 的个人学术博客与研究知识库，主要记录具身智能、机器人导航和多模态学习方向的中文综述、论文笔记与工程实践。

内容主要面向具身智能与机器人方向的研究者、学生和工程师，默认读者具备基础的机器学习与深度学习知识。

本站以系统梳理和持续更新为目标：既关注模型与数据的发展脉络，也关注训练方法、系统架构、评测基准和真实机器人部署。长篇综述用于建立完整的技术脉络，短篇文章用于记录阶段性观察与专题分析。

## 研究方向

| 方向 | 主要内容 | 入口 |
| --- | --- | --- |
| 视觉语言导航 | VLN、R2R、连续环境导航、拓扑记忆、端到端导航 | [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) · [论文笔记](https://tingdeliu.github.io/VLN-Papers/) |
| 视觉语言动作模型 | VLA、机器人策略学习、动作生成、模仿学习与强化学习 | [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) |
| 机器人导航 | SLAM、定位建图、路径规划、运动控制与 ROS 2 | [导航综述](https://tingdeliu.github.io/Robot-Navigation-Survey/) · [ROS 2](https://tingdeliu.github.io/ROS2-Survey/) |
| 多模态与空间智能 | VLM、三维场景理解、点云、深度估计、Gaussian Splatting | [VLM 综述](https://tingdeliu.github.io/VLM-Survey/) · [空间智能](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) |
| 世界模型与智能体 | 世界建模、AI Agent、规划、记忆、工具调用与闭环执行 | [世界模型](https://tingdeliu.github.io/World-Models-Survey/) · [AI Agent](https://tingdeliu.github.io/AI-Agent-Survey/) |
| 学习与训练方法 | LLM 训练、深度学习、机器学习、强化学习 | [LLM 训练](https://tingdeliu.github.io/LLM-Training-Survey/) · [强化学习](https://tingdeliu.github.io/Reinforcement-Learning-Survey/) |

完整内容可在 [Research](https://tingdeliu.github.io/research/) 和 [Blog](https://tingdeliu.github.io/blog/) 页面浏览。

### 推荐阅读路径

1. 从 [VLN 综述](https://tingdeliu.github.io/VLN-Survey/) 了解视觉语言导航的任务定义、数据集和技术演进。
2. 阅读 [VLA 综述](https://tingdeliu.github.io/VLA-Survey/) 了解视觉、语言与机器人动作的统一建模。
3. 通过 [空间智能综述](https://tingdeliu.github.io/Spatial-Intelligence-Survey/) 和 [世界模型综述](https://tingdeliu.github.io/World-Models-Survey/) 扩展到三维理解与环境预测。

## 内容组织

```text
.
├── _posts/
│   ├── research/          # 长篇综述、论文集合与技术笔记
│   └── blog/              # 周报、专题解析与工程文章
├── images/                # 按研究主题组织的图片资源
├── _layouts/              # Jekyll 页面布局
├── _includes/             # 导航、目录、反馈等页面组件
├── _sass/                 # 主题样式模块
├── js/                    # 搜索与文章交互脚本
├── research/              # Research 聚合页
├── blog/                  # Blog 聚合页
└── _config.yml            # 站点与导航配置
```

文章使用 Jekyll 内置的 `posts` 集合。`categories: research` 与 `categories: blog` 是内容分类依据，子目录仅用于维护源文件。

## 技术实现

- **静态站点生成**：Jekyll 4.3
- **主题**：定制主题，源自 Jekyll Now
- **内容渲染**：Kramdown、Rouge、MathJax
- **站点能力**：文章目录、全文搜索、阅读进度、社区反馈
- **部署**：GitHub Actions 构建并发布到 GitHub Pages

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

本地站点默认运行在 [http://127.0.0.1:4000/](http://127.0.0.1:4000/)。

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
