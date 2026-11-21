---
layout: post
lang: zh-CN
translation_id: vln-weekly-2026-11-21
title: "具身导航周报 · 第 14 期"
date: 2026-11-21
last_modified_at: 2026-11-21
issue_number: 14
period_start: 2026-11-13
period_end: 2026-11-20
permalink: /vln-weekly-2026-11-21/
tags: [VLN, VLA, Embodied Navigation, Weekly Digest, arXiv]
categories: weekly
comments: true
author: Tingde Liu
toc: true
excerpt: "本期选读 26 项工作。VLN-Flash 在固定输入上将动作解码加速 2.57–2.75 倍；CHAC 在 R2R-CE 少调用模型 64.7%，成功率下降 10.82 个百分点。LeaP-VLN 与 AgenticJevNav 分别在子任务切换和执行异常时核验进展。"
---

> **本期数字**：选读 26 项独立工作（窗口内首次提交 25 项 / 既有论文修订 1 项）· 主线：VLN 9 篇 / 具身 Agentic 导航 3 篇 · 其他方向 14 篇 · 公众号 0 条

## 一、本期要点
{: id="key-points"}

- **推理预算有两个可调的环节。** [VLN-Flash](#vln-flash)缩短单次动作解码，[CHAC](#chac)减少重算频率。后者在 R2R-CE 的成功率损失提示，延长执行承诺需要按任务校准。
- **在线学习开始复用失败回合。** [R2A](#r2a)让旧反馈参与后续策略更新。读数时应同时看初始策略、反馈预算与回合顺序。
- **规划器核验的时机开始与失败类型绑定。** [LeaP-VLN](#leap-vln)检查子任务停止是否可信，[AgenticJevNav](#agenticjevnav)在阻塞或进展不足时触发复核。两种信号对应不同的干预点。

## 二、精读
{: id="deep-reads"}

### 1. VLN-Flash
{: id="vln-flash"}

**在小动作空间里，一次前向即可评分多条短计划。**

**[论文](https://arxiv.org/abs/2611.19287v1)** · 主线：VLN · 指令跟随 · R2R-CE / RxR-CE（连续环境）

**问题。** 多步自回归动作解码要依次前向，生成速度受限。执行器若一次走完整段动作，也可能错过更新观察的时机。

**方法。** 用树形注意力在一次前向中给候选动作前缀评分，再根据候选前缀的分数差动态选择本轮执行长度。导航骨干保持冻结。

<div align="center">
<img src="https://arxiv.org/html/2611.19287v1/Architecture.png" alt="VLN-Flash 并行动作解码架构" width="100%" loading="lazy" decoding="async" referrerpolicy="no-referrer" />
<figcaption>图 1：树形注意力让模型在一次前向中为多条候选动作前缀评分。 <a href="https://arxiv.org/html/2611.19287v1/Architecture.png">论文原图</a></figcaption>
</div>

**证据。** 固定输入的动作解码较自回归快 2.75 倍（Uni-NaVid）和 2.57 倍（StreamVLN）；在 RxR-CE val-unseen 上，Uni-NaVid 的 SR 从 55.1% 到 57.8%。Unitree GO2 上，同一 StreamVLN 骨干完成 52/60 条指令，自回归组完成 49/60 条。

**判断。** 同检查点对照有助于区分解码器与骨干的贡献。收益依赖候选动作空间和计划长度，部署时还应单独计算通信等待。

**下一步。** 复现：固定执行长度后测解码延迟，再打开自适应执行测 SR 和 SPL。

### 2. R2A
{: id="r2a"}

**让相似指令下的成功和失败反馈都参与策略更新。**

**[论文](https://arxiv.org/abs/2611.19901v1)** · 主线：VLN · 在线适应 · REVERIE / R2R（离散环境）、R2R-CE（连续环境）

**问题。** 在线 VLN 已获得的反馈往往只用于当前回合，后续遇到相似指令时还要重新试错。

**方法。** 成败记忆按指令相似度检索历史回合，并补充随机样本。更新策略时相对于冻结的预训练先验提高成功轨迹的偏好、压低失败轨迹的偏好，以约束行为漂移。

<div align="center">
<img src="https://arxiv.org/html/2611.19901v1/pipeline.png" alt="R2A 在线导航适应流程" width="100%" loading="lazy" decoding="async" referrerpolicy="no-referrer" />
<figcaption>图 2：检索与当前指令相关的成功和失败回合，结合当前反馈更新导航策略。 <a href="https://arxiv.org/html/2611.19901v1/pipeline.png">论文原图</a></figcaption>
</div>

**证据。** 在离散 REVERIE val-unseen 上，论文报告 DUET 在线适应后，SR 从 46.98% 到 69.84%，SPL 从 33.73% 到 50.98%。保留在线适应但移除回放记忆时，SR 为 64.49%。

**判断。** 这些是在线适应条件下的收益。回合顺序、经验库状态和反馈可用性都会影响测试时能够利用的信息。

**下一步。** 复现：打乱回合顺序，在同一反馈预算下比较仅用当前回合、仅回放成功和成败共同回放。

### 3. CHAC
{: id="chac"}

**省下的模型调用需要和导航成功率一起计账。**

**[论文](https://arxiv.org/abs/2611.25587v1)** · 主线：VLN · 航点执行 · HM3D-OVON / R2R-CE（连续环境）

**问题。** 每次执行都调用视觉语言模型开销较高。缓存航点在转弯、遇阻或接近终点后又可能失效。

**方法。** 冻结策略先预测一串航点，再依据保留的视觉上下文、完成线索与控制器反馈决定是否刷新。策略发出停止信号时，可利用访问历史做至多一次移动修正，无需额外调用模型。

<div align="center">
<img src="https://arxiv.org/html/2611.25587v1/workflow.png" alt="CHAC 航点预测与缓存刷新流程" width="100%" loading="lazy" decoding="async" referrerpolicy="no-referrer" />
<figcaption>图 1：策略先预测航点，执行中根据视觉上下文和控制器反馈决定何时刷新。 <a href="https://arxiv.org/html/2611.25587v1/workflow.png">论文原图</a></figcaption>
</div>

**证据。** HM3D-OVON 和 R2R-CE 的模型调用分别减少 33.4% 和 64.7%，SR 分别下降 0.63 和 10.82 个百分点。四项物理任务每种方法跑 40 次，CHAC 成功 18 次，原生执行成功 14 次。

**判断。** 两套仿真基准的代价相差较大，单看每次调用带来的成功率会掩盖这个取舍。真机试验宜按指令长度和终点判据复核。

**下一步。** 跟踪：绘制不同指令长度下的调用数–SR 曲线，单独标出末端修正的贡献。

### 4. LeaP-VLN
{: id="leap-vln"}

**每次子任务切换都需要有证据支持。**

**[论文](https://arxiv.org/abs/2611.20676v1)** · 主线：具身 Agentic 导航 · 长程指令跟随 · R2R-CE / RxR-CE（连续环境）

**问题。** 长路线分解后，执行器可能提前停止或已偏离路线。未经检查就切换子任务，局部误差会传到后续步骤。

**方法。** 执行器联合学习进度推理和动作预测。停止后，验证器决定推进、续跑或回到最近一次确认的锚点，重新取景并重试当前子任务。

<div align="center">
<img src="https://arxiv.org/html/2611.20676v1/framework_overview_v2.png" alt="LeaP-VLN 进度验证与恢复框架" width="100%" loading="lazy" decoding="async" referrerpolicy="no-referrer" />
<figcaption>图 2：验证器在子任务停止后核验进展，并决定推进、续跑或从已确认的锚点恢复。 <a href="https://arxiv.org/html/2611.20676v1/framework_overview_v2.png">论文原图</a></figcaption>
</div>

**证据。** R2R-CE val-unseen 上，论文报告 SR 为 67.5%，对照 CorrectNav 为 65.1%，两者 SPL 分别为 59.1% 和 62.3%。RxR-CE val-unseen 的 SR 为 69.6%。AgiBot Genie 02 展示了一条含三段子任务的室内路线。

**判断。** 可恢复的锚点提高了故障隔离能力，回退和重试也会增加路径。真机演示说明接口能够落地，规模化收益仍需统计。

**下一步。** 复现：对提前停止和路线偏离分别统计验证混淆率、恢复成功率与额外路径长度。

### 5. AgenticJevNav
{: id="agenticjevnav"}

**只在执行异常时调用有明确权限的复核器。**

**[论文](https://arxiv.org/abs/2611.19912v1)** · 主线：具身 Agentic 导航 · 室内物体搜索 · 六类仿真任务与四足机器人试验

**问题。** 搜索途中可能停滞，远程推理返回的动作也可能因证据过时或目标变化而失效。

**方法。** 阻塞、执行失败或连续两段低进展才触发 Jev，复核范围限于规划器授权的动作集。单执行器在动作前检查版本、时效和是否已消费，另由本地 RGB 反射模块拦截前进动作。

<div align="center">
<img src="https://arxiv.org/html/2611.19912v1/figures/1.png" alt="AgenticJevNav 的物体搜索与事件触发复核示例" width="100%" loading="lazy" decoding="async" referrerpolicy="no-referrer" />
<figcaption>图 1：搜索中的受阻和低进展等异常可触发 Jev 复核，目标更新后会撤销过时请求。 <a href="https://arxiv.org/html/2611.19912v1/figures/1.png">论文原图</a></figcaption>
</div>

**证据。** 六类仿真任务共 60 次，SR 为 73.33%，最强仅 Agent 对照为 51.67%，成功回合均时分别为 140.55 s 和 120.40 s。GENISOM L1 在一处真机场景成功 21/30 次，含限额内确认未找到目标的任务。

**判断。** 将复核限在异常点有助于审计权限与触发依据。整体系统仍要为等待付费，且单场景真机结果不足以估计跨场景表现。

**下一步。** 复现：按阻塞、低进展和证据过期分别记录恢复率、耗时及被拒绝的迟到决策。

## 三、也值得一看
{: id="also-worth"}

- **[ERA](https://arxiv.org/abs/2611.17730v1)** · ObjectNav：用后续视角修正预测深度造成的占据错误，停止前跨视角核验目标。
- **[ROAM](https://arxiv.org/abs/2611.25215v1)** · 户外语义导航：以开放词汇分割将语义风险转为地形代价，接入四足控制。
- **[EvoNav](https://arxiv.org/abs/2611.22321v1)** · 终身导航：将重复观察和搜索反馈汇成环境特定的物体共现知识。
- **[RPSI-Nav](https://arxiv.org/abs/2611.23774v1)** · 具身 Agentic 导航：后续搜索反馈同时更新导航程序和诊断规则。
- **[EasyNav](https://arxiv.org/abs/2611.19894v1)** · ROS 2：显式共享状态和时序约定可借鉴到导航 Agent 与控制器接口。
- **[RePlan-VLA](https://arxiv.org/abs/2611.23843v1)** · 操作：用短程策略执行、监测进展并修订后续计划，可借鉴到导航子任务恢复。

## 四、趋势与行动
{: id="trends-and-actions"}

**趋势**

- **推理算力开始按决策阶段分配。** [VLN-Flash](#vln-flash)对应每次调用的解码代价，[CHAC](#chac)对应刷新频率。两种开销应分开计量，再纳入总耗时。
- **复核逐渐成为可测试的接口。** [LeaP-VLN](#leap-vln)给出转段前的核验点，[AgenticJevNav](#agenticjevnav)给出异常后的触发点。可分别测错过复核和误触发的后果。

**研究空白**

- 在同一长程连续导航基准中，尚需对照独立验证器何时应该介入、延迟多长仍有效，以及预算耗尽前如何交还给本地执行器。

**下一步**

- **高 · 复现** [VLN-Flash](#vln-flash)与[CHAC](#chac)：固定已有骨干和硬件，分别记录单次解码耗时、模型调用次数与 SR/SPL。
- **中 · 评测** [LeaP-VLN](#leap-vln)与[AgenticJevNav](#agenticjevnav)：在同一批长程任务注入提前停止、阻塞和延迟反馈，比较转段核验与异常核验的恢复代价。
- **中 · 复现** [R2A](#r2a)：分别报告冷启动和已有经验库的结果，并为新场景做清空记忆的对照。

## 五、全部工作索引
{: id="full-index"}

以下为本期 26 项选读工作，依次收录两条导航主线（A）、可迁移方法（B）与相邻领域（C）。

<details markdown="1">
<summary>VLN 与语义导航 · 9 篇</summary>

- [VLN-Flash](#vln-flash)：见精读。
- [R2A](#r2a)：见精读。
- [CHAC](#chac)：见精读。
- [ERA](https://arxiv.org/abs/2611.17730v1)：用运动获得的新视角纠正地图错误。
- [ROAM](https://arxiv.org/abs/2611.25215v1)：将语义风险转为四足地形代价。
- [EvoNav](https://arxiv.org/abs/2611.22321v1)：从短期观察汇集长期环境知识。
- [SearchDiffusion](https://arxiv.org/abs/2611.24375v1)：以空间效用图引导物体搜索轨迹。
- [ROS 2 语义导航](https://arxiv.org/abs/2611.20558v1)：开放词汇检测与 Nav2 接入 Gazebo。
- [Racing the Clock](https://arxiv.org/abs/2611.17442v2)：修订版，比较预算约束下的物体搜索教师。

</details>

<details markdown="1">
<summary>具身 Agentic 导航 · 3 篇</summary>

- [LeaP-VLN](#leap-vln)：见精读。
- [AgenticJevNav](#agenticjevnav)：见精读。
- [RPSI-Nav](https://arxiv.org/abs/2611.23774v1)：更新搜索规则与生成规则的机制。

</details>

<details markdown="1">
<summary>可迁移方法 · 8 篇</summary>

- [EasyNav](https://arxiv.org/abs/2611.19894v1)：以显式共享状态连接 ROS 2 导航模块。
- [RePlan-VLA](https://arxiv.org/abs/2611.23843v1)：监测长程操作进度，按反馈改计划。
- [AgentRoboCritic](https://arxiv.org/abs/2611.26613v1)：对照局部意图归因动作级失败。
- [CoachSI](https://arxiv.org/abs/2611.25529v1)：语言教练产生可重标的成功执行轨迹。
- [CRA](https://arxiv.org/abs/2611.22080v1)：将连续语义风险接入视觉导航安全滤波。
- [FedDIA](https://arxiv.org/abs/2611.25637v1)：按场景难度和进步加权联邦更新。
- [Hierarchical Skill Composition](https://arxiv.org/abs/2611.25462v1)：在未见 FourRooms 布局复用预训练技能。
- [Homeward](https://arxiv.org/abs/2611.24455v2)：将变化的相机观测重投影到训练视角。

</details>

<details markdown="1">
<summary>相邻领域 · 6 篇</summary>

- **无人机导航（3）**：[AeroRSI](https://arxiv.org/abs/2611.24077v1)、[Fly as Instructed](https://arxiv.org/abs/2611.22883v1)、[CityNav-VLA](https://arxiv.org/abs/2611.24247v1)
- **VLA 操作（2）**：[A²-VLA](https://arxiv.org/abs/2611.25656v1)、[ScaleVLA](https://arxiv.org/abs/2611.21607v1)
- **自动驾驶（1）**：[Ego2Flow](https://arxiv.org/abs/2611.17720v1)

</details>
