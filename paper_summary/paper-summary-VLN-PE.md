---
## 1. VLN-PE (2025)
———重新思考视觉-语言导航中的具身化差距:物理和视觉差异的全面研究

📄 **Paper**: https://arxiv.org/abs/2507.13019v2

**精华**

这篇论文通过构建物理真实的VLN平台,系统性地揭示了理想化仿真与物理部署之间的巨大差距。核心启示包括:(1) 跨具身数据融合训练可以显著提升模型泛化能力,为统一的跨机器人导航模型奠定基础;(2) 多模态感知(RGB+Depth)比单一RGB更鲁棒,尤其在光照变化环境下;(3) 物理控制器的引入对于腿足机器人至关重要,训练和评估阶段的控制器一致性直接影响性能;(4) 现有MP3D风格数据集的泛化能力有限,小规模域内数据微调即可超越大模型零样本性能;(5) diffusion policy作为连续路径点预测的新范式在VLN任务中展现潜力。

**研究背景/问题**

现有的VLN方法在理想化仿真环境中表现优异,但在部署到真实物理机器人时面临巨大挑战。主要问题包括:当前VLN平台忽视了机器人的物理具身特性(如视点高度、运动动力学、碰撞和跌倒等),并且缺乏对不同机器人类型(轮式、人形、四足)的跨具身支持。研究核心问题是:物理具身约束和视觉环境变化对现有VLN方法的性能影响究竟有多大?

**主要方法/创新点**

<div align="center">
  <img src="/images/VLN-PE-evolution.png" width="100%" />
<figcaption>
VLN任务的演进:从oracle-based导航(2018)到VLN-CE连续导航(2020),再到VLN-PE物理真实导航(2025)
</figcaption>
</div>

论文提出了**VLN-PE平台**,一个基于GRUTopia构建的物理真实VLN基准测试平台,具有以下核心特性:

1. **跨具身支持**:支持人形机器人(Unitree H1, G1)、四足机器人(Unitree Aliengo)和轮式机器人(Jetbot),并提供基于RL的物理控制器API,实现真实的运动动力学模拟

2. **场景多样性**:除了90个MP3D场景外,新增10个高质量合成家居场景(GRScenes)和3DGS在线渲染实验室场景,支持无缝集成更多环境

<div align="center">
  <img src="/images/VLN-PE-platform-overview.png" width="100%" />
<figcaption>
VLN-PE平台概览:支持多种机器人具身、场景类型、光照条件和控制器模式
</figcaption>
</div>

3. **系统性评估框架**:评估三类ego-centric VLN方法
   - **单步端到端方法**:Seq2Seq、CMA(约36M参数)和NaVid(7B参数的视频MLLM)
   - **多步端到端方法**:首次提出RDP(Recurrent Diffusion Policy),使用transformer-based diffusion模块预测连续轨迹路径点
   - **地图基零样本方法**:改进的VLMaps,结合LLM和语义地图进行路径规划

<div align="center">
  <img src="/images/VLN-PE-RDP-framework.png" width="100%" />
<figcaption>
RDP(循环扩散策略)框架:使用GRU维护历史信息,交叉注意力融合视觉-语言特征,Transformer扩散模块预测连续动作序列
</figcaption>
</div>

4. **新数据集**:
   - **R2R-filtered**:过滤楼梯场景后保留8,679/658/1,347个训练/val-seen/val-unseen episodes
   - **GRU-VLN10**:10个合成场景,441/111/1,287个episodes
   - **3DGS-Lab-VLN**:3DGS渲染实验室环境,160训练/640评估episodes

5. **新评估指标**:除了传统的TL、NE、SR、OS、SPL外,新增Fall Rate (FR)和Stuck Rate (StR)来衡量物理真实性挑战

**核心结果/发现**

<div align="center">
  <img src="/images/VLN-PE-main-results.png" width="100%" />
<figcaption>
使用人形机器人Unitree H1在R2R数据集上的主要实验结果对比
</figcaption>
</div>

**零样本迁移性能大幅下降**:
- VLN-CE模型直接迁移到VLN-PE时,SR相对下降约34%
- Seq2Seq-Full、CMA-Full和NaVid的SR分别下降10%、16%和18%
- 这表明现有模型严重过拟合特定仿真平台

**域内微调显著提升**:
- 在VLN-PE上从头训练的CMA(无数据增强)超越了使用175K增强数据训练的CMA-Full
- 小模型CMA+经过微调后,在val-seen上达到SR 28.72,SPL 24.24,超越NaVid的零样本性能

**跨具身敏感性**:
- 四足机器人(相机高度约0.5m)在迁移时几乎完全失败
- 调整相机高度到1.8m可改善人形机器人的迁移性能
- 跨具身联合训练使单一模型在所有机器人类型上达到SoTA性能

**物理控制器的重要性**:
- 训练和评估使用相同控制器时性能最佳
- 使用物理控制器收集数据可降低Fall Rate和Stuck Rate

**多模态鲁棒性**:
- 仅RGB的NaVid在低光照下SR下降12.47%
- RGB+Depth的CMA和RDP受光照影响较小(下降约1-2%)

**MP3D数据集泛化能力有限**:
- 在GRU-VLN10上,RDP用6M参数仅441个训练样本,零样本超越NaVid大模型
- 在3DGS-Lab-VLN上,NaVid完全失败(SR仅5.81),可能是3DGS渲染噪声导致

**扩散策略的潜力**:
- RDP作为首个VLN扩散策略基线,在从头训练时优于Seq2Seq和CMA
- 预测连续密集路径点,可与MPC等控制理论方法结合

**真机实验验证**:
- 使用Unitree Go2机器人进行14个室内场景测试
- VLN-PE微调模型在真实环境中OS达到57.14,SR达到28.57,显著优于VLN-CE训练模型

**局限性**

当前RL-based运动控制器无法可靠处理复杂环境中的楼梯导航,需要过滤相关场景。论文主要聚焦ego-centric视角,未评估panoramic VLN方法。MLLM在精确目标识别和停止决策上仍存在挑战。3DGS渲染引入的像素级噪声可能干扰纯RGB模型,需要进一步研究图像扰动的鲁棒性。
