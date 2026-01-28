---
## 1. VLM4VLA (2026)
———重新审视 Vision-Language-Action 模型中的 Vision-Language 模型

📄 **Paper**: https://arxiv.org/abs/2601.03309

**精华**

这篇论文最值得借鉴的核心思想包括:通过最小化适配管道公平评估不同 VLM 对下游任务性能的影响;发现 VLM 的通用能力与具身控制性能并不强相关,挑战了常见假设;识别出视觉编码器(而非语言组件)是性能瓶颈,揭示了 VLM 预训练目标与具身动作规划需求之间存在领域差距;提出通过向视觉编码器注入控制相关监督信号可获得持续性能提升的策略。

**研究背景/问题**

当前 Vision-Language-Action (VLA) 模型研究主要关注网络架构、训练范式和动作解码方案的改进,但很少系统研究一个核心问题:底层 Vision-Language Model (VLM) 的选择和能力如何影响 VLA 策略的性能。现有工作缺乏公平的实验框架来评估不同 VLM 对下游机器人任务性能的贡献。

**主要方法/创新点**

论文提出了 **VLM4VLA** 框架,这是一个最小化适配管道,通过引入少于 1% 的新参数将通用 VLM 转换为 VLA 策略,确保公平高效的比较。

<div align="center">
  <img src="/images/VLM4VLA-framework-overview.png" width="100%" />
<figcaption>
VLM4VLA 框架概览:展示评估流程、辅助具身任务微调和不同训练策略的影响
</figcaption>
</div>

**核心架构设计**:

<div align="center">
  <img src="/images/VLM4VLA-network-architecture.png" width="100%" />
<figcaption>
VLM4VLA 网络架构:通过可学习的 Action Query token 提取具身相关知识,使用 MLP 解码动作块
</figcaption>
</div>

- 引入可学习的 **Action Query token** 从 VLM 中提取具身相关知识
- 使用简单的 **MLP-based policy head** 解码动作,避免 diffusion/flow-matching 引入的随机性
- 采用 **L1/L2 loss** 而非 diffusion loss,提高推理稳定性和评估鲁棒性
- 所有 VLM 参数(vision encoder、LLM、word embeddings)在下游任务微调时全部训练

**三维实验设计**:

1. **通用能力评估**: 比较 9 个开源 VLM(1B-30B 参数)作为 VLA 骨干网络的性能,包括 Qwen2.5VL/Qwen3VL 系列、Paligemma 系列、Kosmos-2
2. **具身特定能力评估**: 使用 7 种辅助具身任务(visual grounding、depth estimation、trajectory prediction 等)微调 VLM,测试对下游控制任务的影响
3. **模态级消融**: 独立冻结/微调视觉和语言编码器,并测试向 vision encoder 注入控制相关信息(FAST tokenizer)的效果

**评估基准**: 在三个模拟环境上测试
- **Calvin ABC-D**: 训练于 ABC 场景,测试于 D 场景(跨场景泛化)
- **SimplerEnv-Bridge**: 训练于真实 BridgeV2 数据,测试于仿真环境
- **Libero-Long**: 10 个长视距操作任务

**核心发现**:

<div align="center">
  <img src="/images/VLM4VLA-vlm-capability-correlation.png" width="100%" />
<figcaption>
VLM 通用能力与 VLA 性能的线性关系:Calvin 呈强正相关(r=0.839),而 Simpler 和 Libero 几乎无相关性
</figcaption>
</div>

1. **VLM 通用能力是必要但不充分的**: VLM 初始化相比从头训练提供一致性收益,但 VLM 的通用 VQA 能力无法预测其在具身控制任务上的表现
2. **辅助具身任务微调效果有限**: 在 visual pointing、spatial understanding、embodied VQA 等任务上微调 VLM 并未提升下游控制性能,甚至略有下降

<div align="center">
  <img src="/images/VLM4VLA-auxiliary-tasks-performance.png" width="100%" />
<figcaption>
不同辅助 VLM 微调任务的性能表现:所有具身 VQA 任务微调后性能均略低于基线
</figcaption>
</div>

3. **Vision encoder 是关键瓶颈**: 冻结视觉编码器导致显著性能下降(Calvin 上下降 1.0-3.0 分),而冻结 word embeddings 几乎无影响
4. **存在视觉-语言理解与低级控制的语义差距**: 通过向 vision encoder 注入动作 token 预测任务,即使冻结 encoder 也能获得 +18.1% 性能提升,证明 VLM 视觉特征与控制需求存在根本性不对齐

<div align="center">
  <img src="/images/VLM4VLA-training-divergence.png" width="100%" />
<figcaption>
VLM 和 VLA 训练轨迹示意图:两者初期沿相同方向学习,但在某个时间点分歧到不同区域
</figcaption>
</div>

**核心结果/发现**

- **Calvin ABC-D**: Qwen3VL-2B 达到最佳性能(平均完成 4.142 个任务),接近 SOTA VLA(pi0: 3.509)
- **SimplerEnv-Bridge**: 最小的 Kosmos-2 (1.7B) 达到最高成功率(60.4%),超越更大的 Qwen 系列模型
- **Libero-Long**: Qwen3VL-2B 和 Kosmos-2 均达到 55%+ 成功率,优于其他 VLM
- **从头训练性能崩溃**: 不使用 VLM 预训练的模型性能下降 60-70%,证明 VLM 预训练对 VLA 泛化至关重要
- **Real-to-Sim 差距非主因**: 在真实图像上微调 VLM 的动作预测任务后,冻结 vision encoder 仍导致性能下降,表明问题源于视觉-语言任务与低级控制任务的本质差异
- **Vision encoder 微调必要性**: 在 SimplerEnv-Bridge 任务上,解冻 vision encoder 并注入控制信息使性能从 27.6% 提升至 45.7%(+18.1%)

**局限性**

研究未在物理机器人上进行实验,主要受限于公平性和可重复性考虑。虽然分析表明 VLM-VLA 差距源于任务异质性而非简单的 sim-to-real 差距,但真实世界部署仍是最终目标。论文的全面模拟基准结果可为未来研究提供有价值的参考。
