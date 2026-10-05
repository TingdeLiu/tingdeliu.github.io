# 综述引言配图更新记录

日期：2026-10-02。覆盖 14 篇综述与指南：新增 3 张，替换 11 张。Embodied Agent 图也按相同视觉风格重做。

14 张新图使用内置 `image_gen` 生成，完整提示词、替代文字和图注保存在 [survey-intro-figures-prompts.json](survey-intro-figures-prompts.json)。图片以新文件名保存，原有素材保留。

统一采用浅底、深蓝中文标签、克制的蓝绿与琥珀色、少量明确箭头和具体场景。图用于解释主题与典型机制，不能代替论文架构或实验数据。

| 文章 | 配图 | 调整要点 |
|---|---|---|
| VLA | [查看配图](../../images/vla/vla-survey-intro.webp) | 补齐引言图；以杯子放入托盘为例说明多模态输入、动作块与观测反馈。 |
| ROS 2 | [查看配图](../../images/ros2/ros2-survey-intro.webp) | 补齐引言及主题图；呈现节点协作与客户端、RMW、中间件实现的分层，不把 DDS 画成唯一底座。 |
| Python 工程 | [查看配图](../../images/python/python-engineering-survey-intro.webp) | 补齐引言及主题图；用环境、接口、质量、交付四个环节说明从脚本到项目。 |
| 机器学习 | [查看配图](../../images/ML/machine-learning-survey-intro.webp) | 替换分类维度混杂的旧图；按监督、无监督、自监督、强化学习的学习信号展示，图注明确可组合。 |
| 深度学习 | [查看配图](../../images/DL/deep-learning-survey-intro.webp) | 替换旧图；CNN、RNN、Transformer 并列为架构选择，训练更新单独画出。 |
| LLM 训练 | [查看配图](../../images/llm-training/llm-training-survey-intro.webp) | 替换旧图；基础模型、指令微调、偏好优化与部署工程区分，避免统一必经训练配方。 |
| 强化学习 | [查看配图](../../images/vla/reinforcement-learning-survey-intro.webp) | 替换旧图；动作流向环境，观测与奖励返回智能体，策略更新作为学习支路。 |
| VLN | [查看配图](../../images/vln/vln-survey-intro.webp) | 替换旧图；指令接地、导航决策、移动停止与反馈形成闭环，空间记忆用虚线表示辅助模块。 |
| VLM | [查看配图](../../images/vlm/vlm-survey-intro.webp) | 替换旧图；用红杯问答解释典型生成式 VLM，图注限定其覆盖范围。 |
| AI Agent | [查看配图](../../images/agent/ai-agent-survey-intro.webp) | 替换旧图；突出任务、决策、工具与结果检查，记忆是上下文支撑。 |
| 世界模型 | [查看配图](../../images/wm/world-models-survey-intro.webp) | 替换引言的论文截图；未来分支用虚线与淡色区分预测和实际，覆盖预测、规划与学习。 |
| 空间智能 | [查看配图](../../images/si/spatial-intelligence-survey-intro.webp) | 替换旧图；用同一桌面场景串起感知、表示、关系与交互，图注说明不是技术演进时间线。 |
| 传统机器人导航 | [查看配图](../../images/robotics_navigation/robot-navigation-survey-intro.webp) | 替换旧图；感知定位、地图规划、局部控制和传感器反馈保持清晰。 |
| Embodied Agent | [查看配图](../../images/agent/embodied-agent-survey-intro.webp) | 重做为同系列的浅底、中文大标签与机器人场景；保留任务闭环、接口契约、本地执行闭环与独立保护。 |

## 内容核对依据

图的范围与各篇引言及正文对齐。对容易混淆的技术关系，核对了以下原始材料：

- [scikit-learn 用户指南](https://scikit-learn.org/stable/user_guide)：监督与无监督学习的分类。
- [Sutton 与 Barto《Reinforcement Learning》](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf)：智能体、环境、动作、奖励与长期回报。
- [TRL 文档](https://huggingface.co/docs/trl/main/quickstart)及 [DPO 原论文](https://arxiv.org/abs/2305.18290)：后训练方法可有不同组织方式，DPO 不要求独立奖励模型。
- [ROS 2 官方中间件文档](https://docs.ros.org/en/ros2_documentation/kilted/Concepts/Intermediate/About-Different-Middleware-Vendors.html)：RMW 抽象与 DDS、Zenoh 等实现。

## 验证

- 逐张查看 14 张新图，核对中文标签、箭头方向、概念关系和图注。
- 新图均为 1672 × 941；WebP 编码保留原始像素尺寸，每张约 100–220 KB。
- 14 篇均在引言区拥有一张主题图；新增图与正文第一张具体模型图区分。
- 语义化 figure / figcaption、完整 alt、固有尺寸、延迟加载与异步解码均已配置。
- Jekyll 完整构建通过；构建使用已有本机依赖，临时产物位于工作区 .cache。
- 桌面 1280px、手机 390px 共 28 次浏览器检查通过：图片加载、视口边界、图注、点击放大与放大图注。
- 手机整图侧重主线，细节需点击放大；页面提供放大提示。
- Embodied Agent 重做后再次构建通过；当前浏览器预览确认新图、1672 × 941 尺寸和新图注正常。额外修正保护箭头，明确感知信息进入本地保护，保护再约束控制器。
- 检查记录与页面截图位于 `.cache/survey-intro-figures/`，不进入 Git。
