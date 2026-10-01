# VLA 综述核心事实核验

核验日期：2026-10-01。对象：`_posts/research/2026-01-24-VLA-Survey.md`。只使用论文、作者项目页和官方文档；未逐项审计全文全部模型与数字。研究笔记沿用仓库 `Analysis/` 目录。

## 1. RT-1 与 RT-2 应区分

**建议正文表述**：RT-1 是语言条件化的视觉运动 Transformer，为规模化机器人策略奠定基础；RT-2 则通过视觉语言数据与机器人轨迹的联合微调，把预训练 VLM 的语义知识迁移到动作生成。不能把两者一并描述成“首次将预训练 VLM 转成 VLA”。[RT-1 官方项目](https://robotics-transformer1.github.io/)、[RT-2 官方项目](https://robotics-transformer2.github.io/)。

RT-1 细节需要修正：每图 **81 个空间 token → 8 个 token**；512 是通道维度。Transformer 为 **8 层、19M 参数**，不是正文的11层38M。11维动作由末端位姿6维、夹爪1维、底盘3维、模式1维组成。Table 13 中自回归动作是消融变体，不宜将主模型写成自回归动作解码。来源：[RT-1 §5.1、Table 13](https://arxiv.org/html/2212.06817v2)。

RT-2 主评测包含 PaLM-E 12B、PaLI-X 55B，另对 PaLI-X 的5B/55B作规模消融；不能仅用“5B/55B”涵盖全部版本。[RT-2 项目结果](https://robotics-transformer2.github.io/)。

## 2. ACT、Diffusion Policy 是策略方法，不自动等于 VLA

**建议正文表述**：ACT 使用 CVAE 和动作分块减少有效预测时域；Diffusion Policy 用条件扩散建模动作序列和多峰动作分布。两者的原始工作是视觉运动模仿学习方法，可为 VLA 提供动作建模组件，但仅采用 Transformer、分块或扩散，并不意味着模型具备语言条件或 VLM 预训练。

“解决累积误差”“彻底解决取平均”应改成“缓解累积误差”“能够表达多峰动作分布”；这些模型仍会失败，也受数据覆盖和分布偏移影响。来源：[ACT 官方项目](https://tonyzhaozh.github.io/aloha/)、[Diffusion Policy 官方项目](https://diffusion-policy.cs.columbia.edu/)。

## 3. π₀：参数、控制频率与推理延迟

**建议正文表述**：π₀ 在约3B的PaliGemma上增加约300M动作专家，总计约 **3.3B** 参数；以10步积分生成50步动作块。在论文RTX4090配置下，单次本地推理约73ms；50Hz机器人每执行25步、约0.5秒重新推理。因此 **50Hz是动作执行频率，不是整个视觉语言模型每秒闭环推理50次**。

原文“3B+860M”“3.8B”“把去噪路径拉直后普遍快5–7倍”缺乏该论文支持。可写成“连续动作生成与动作分块支持高频执行”，避免把全部速度收益归因于Flow Matching。来源：[π₀ §IV、Appendix A-D、Table I](https://arxiv.org/html/2410.24164v1)。

## 4. FAST：训练加速不等于推理加速

**建议正文表述**：FAST 用DCT和BPE压缩动作序列，改善高频动作数据上的自回归学习效率。论文报告达到相近性能所需训练算力最高减少约5倍；这不是“15倍推理加速”。在作者RTX4090对照中，π₀-FAST每块约750ms，π₀流模型约100ms以内，自回归版本反而更慢。

数字分别来自训练收敛与推理延迟，不能混写；token压缩比也不等于端到端速度比。来源：[FAST 摘要、§VI-E、§VI-F](https://arxiv.org/html/2501.09747v1)。

## 5. OpenVLA 与 OFT

OpenVLA 由 Prismatic-7B VLM 微调而来，组合DINOv2、SigLIP、投影层和Llama 2 7B；使用970k真实机器人演示。Llama 2本身是语言模型，不宜直接称其为预训练VLM。[OpenVLA 官方模型说明](https://openvla.github.io/)。

“目前最强”“双编码器本身击败55B”应改成带评测范围的结论：论文在29项任务、多种机器人上报告比RT-2-X高16.5个百分点；数据与模型组件共同改变，不能把效果单独归因于双编码器。[OpenVLA 论文摘要](https://arxiv.org/abs/2406.09246)。

**OFT建议正文表述**：在作者LIBERO配置中，OpenVLA-OFT通过并行解码、分块、连续动作表示与L1目标，使动作生成吞吐量约为原版26倍、延迟约为原版1/3。官方概述也使用25–50倍，但正文应优先明确实验条件和指标，不把吞吐倍数称为闭环控制频率倍数。[OFT 官方结果](https://openvla-oft.github.io/)。

## 6. GR00T 双系统的频率属于具体版本与硬件

**建议正文表述**：GR00T N1 将VLM表征与DiT动作模块结合，以Flow Matching训练动作生成。N1论文报告L40上的System 2为10Hz、System 1为120Hz，并在bf16配置下报告生成16步动作块约63.9ms。此处是 **N1具体实现**，不能泛化成所有GR00T版本或任何“双系统”的固定速度；正文的System 2“5Hz”不符合该来源。

还应避免把所有VLM隐表征都等同显式长程规划。来源：[GR00T N1 §1–2](https://arxiv.org/html/2503.14734v1)。

## 7. RoboGen、Genesis、Genesis AI 的关系

**可证实**：RoboGen使用开发团队提供的内部Genesis版本作为仿真后端；论文还说明框架原则上不依赖特定仿真平台。这能证明使用关系，不能反向证明Genesis由RoboGen工程衍生，更不能证明“原班人马商业化”。[RoboGen §4.1及脚注](https://arxiv.org/html/2311.01455v2)。

**可证实**：当前官方文档明确说Genesis World始于2024年12月的学术Genesis项目，现由Genesis AI支持开发。这可以写成有来源的工程延续。[Genesis World 官方文档](https://genesis-world.readthedocs.io/en/latest/)。

**未充分核验**：原Gene 26.5链接本次只返回通用网站与候补登记页，没有可读技术正文，因此200k小时、20–30分钟微调、精细任务能力及完整商业传承叙事不能依据本次访问确认。不要把这视作“模型不存在”的证据；应暂时删掉未证实数字或另寻可追溯的一手报告。[本次访问目标](https://www.genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level)。

## 8. SayCan 并非完全忽略环境的“闭眼规划”

SayCan把语言模型对技能用途的评分与当前状态下技能成功概率相结合，通过affordance/value函数获得环境约束。原始方案的反馈方式有局限，但“闭着眼睛”会掩盖其核心贡献。建议改成“语言规划与低层技能分离，通过价值函数将语言决策落到可执行技能”。[SayCan 官方方法与限制](https://say-can.github.io/)。

## 编辑原则

- 将动作执行频率、重规划频率、块生成延迟、生成吞吐、训练算力分开报告。
- 成功率需附任务、机器人、训练数据和评测条件；绝对增加16.5%应写“16.5个百分点”。
- “首次”“最强”“彻底解决”“商业传承”需要专门证据；无证据时使用可检验的方法描述。
- 把领域发展写成多条相互影响的路线，避免把ACT、扩散、Flow Matching、RL写成严格先后替代的三次跃迁。

## 9. 补充核验：骨干对应、RT-X书目与量化

- **RT-2骨干对应**：PaLI-X对应5B/55B规模实验，PaLM-E对应12B；“PaLI-X 5B / PaLM-E 55B”错误。正文可写“基于PaLI-X（5B、55B）和PaLM-E（12B）的多个变体”。[RT-2 官方项目 Results](https://robotics-transformer2.github.io/)。
- **RT-X书目**：arXiv **2310.08864链接正确**，正式标题是 **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**。应改标题，不必改正确的编号。[arXiv 条目](https://arxiv.org/abs/2310.08864)、[官方项目](https://robotics-transformer-x.github.io/)。
- **OpenVLA量化**：不能写成“量化一定加速”。论文§5.4报告多数GPU上8-bit会因量化开销变慢；A5000上int8约1.2Hz、int4约3Hz。4-bit的主要可证实收益是显存降到bf16一半以下且该实验任务表现接近；吞吐能否超过bf16需按硬件及实现查看Figure 6。附录D.4通过阻塞控制对照说明延迟本身会影响机器人成功率。[OpenVLA §5.4、Appendix D.4](https://arxiv.org/html/2406.09246v3)。
