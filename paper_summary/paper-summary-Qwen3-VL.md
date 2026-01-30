---
## 1. Qwen3-VL (2025)
——最强视觉-语言模型系列,原生支持256K上下文

📄 **Paper**: https://arxiv.org/abs/2511.21631v2

**精华**

这篇论文展示了如何构建一个全面的视觉-语言模型系列,值得借鉴的核心思想包括:
1. **平衡文本和多模态能力**:通过square-root reweighting确保多模态训练不损害文本能力,甚至在某些文本任务上超越纯文本模型
2. **渐进式上下文扩展**:采用四阶段预训练(8K→32K→256K),逐步扩展上下文窗口,而不是一步到位
3. **架构优化的实用主义**:Interleaved MRoPE、DeepStack、文本时间戳等创新都针对实际问题(长视频理解、视觉-语言对齐、时序定位)
4. **分层式后训练**:区分non-thinking和thinking变体,针对不同应用场景优化
5. **全栈式能力整合**:将感知(grounding)、推理(reasoning)和行动(agentic)能力统一到单一模型框架中

**研究背景/问题**

现有的视觉-语言模型在发展过程中面临几个关键挑战:一是多模态训练往往会损害底层LLM的语言能力;二是长上下文支持不足,难以处理长文档和长视频;三是在STEM推理、文档理解、视频理解等专业任务上性能参差不齐;四是缺乏统一的框架整合感知、推理和决策能力。Qwen3-VL旨在系统性地解决这些问题。

**主要方法/创新点**

<div align="center">
  <img src="/images/Qwen3-VL-architecture.png" width="100%" />
<figcaption>
Qwen3-VL整体架构:集成视觉编码器和语言模型解码器处理文本、图像和视频等多模态输入。视觉编码器支持动态原生分辨率,通过DeepStack机制将多层视觉特征注入到LLM的对应层中。采用Interleaved MRoPE编码位置信息,并引入文本时间戳标记捕获视频的时序结构
</figcaption>
</div>

Qwen3-VL提出了一个完整的视觉-语言模型系列,包括4个dense模型(2B/4B/8B/32B)和2个MoE模型(30B-A3B/235B-A22B),均原生支持256K token的交错式上下文。

**架构创新**:

1. **Interleaved MRoPE** - 针对Qwen2.5-VL中MRoPE频谱不平衡的问题,将时间(t)、水平(h)、垂直(w)维度交错分布在低频和高频频段,显著改善长视频理解能力

2. **DeepStack跨层融合** - 从ViT的多个层提取视觉特征,通过轻量级残差连接路由到LLM的对应层,增强多层次视觉-语言对齐,不增加额外上下文长度

3. **显式视频时间戳** - 用文本token(如`<3.0 seconds>`)标记视频帧组,替代Qwen2.5-VL中的绝对时间位置编码,提供更简单直接的时序表示,支持seconds和HMS两种格式

**训练策略**:

**预训练**分为四个阶段:
- S0 (67B tokens, 8K): 仅训练merger层进行视觉-语言对齐
- S1 (~1T tokens, 8K): 全参数多模态预训练,混合VL数据和文本数据
- S2 (~1T tokens, 32K): 长上下文预训练,增加文本数据比例和视频/agent数据
- S3 (100B tokens, 256K): 超长上下文适应,聚焦长视频和长文档理解

**后训练**包含三个阶段:
1. SFT - 分为32K和256K两个阶段,提供non-thinking和thinking两个变体
2. Strong-to-Weak蒸馏 - 用text-only数据微调LLM backbone,显著提升推理能力
3. 强化学习 - 分为Reasoning RL(数学、代码、逻辑推理等)和General RL(指令遵循、格式控制等),使用SAPO算法

**数据优化**:

- **高质量caption** - 使用Qwen2.5-VL-32B对web图像重新标注,基于视觉embedding聚类增强稀疏概念覆盖
- **交错文本-图像** - 收集多模态文档,用domain classifier过滤低质量内容,构建256K长序列
- **知识数据** - 覆盖12+语义类别(动物、植物、地标等),采用importance-based采样平衡长尾分布
- **OCR扩展** - 从10种语言扩展到39种语言,合成3000万高质量样本
- **Grounding归一化** - 统一采用[0, 1000]归一化坐标系统,支持2D/3D grounding和counting
- **视频数据** - 密集caption合成(short-to-long策略)和时空grounding数据
- **STEM数据** - 6M图表caption + 60M+ K-12/本科习题 + 12M长CoT推理样本
- **Agent数据** - GUI感知(描述、grounding)+ 自进化轨迹生成框架

**优化技巧**:

- **Square-root reweighting** - 对per-token loss进行平方根归一化,平衡文本和多模态数据贡献
- **分层式训练** - 预训练阶段逐步扩展上下文,后训练阶段区分thinking/non-thinking模式

**核心结果/发现**

**综合性能**:
- 在多模态reasoning任务上(MMMU、MathVista、MathVision等),Qwen3-VL-235B-A22B-Thinking达到SOTA水平
- 在文本任务上超越或持平纯文本模型(如DeepSeek V3、Qwen3-235B),证明多模态训练未损害语言能力
- 小模型(2B/4B/8B)表现出色,8B模型在很多任务上接近Qwen2.5-VL-72B

**长上下文能力**:
- Needle-in-a-Haystack评估:256K token(30分钟视频)内100%准确率,外推到1M token(2小时视频)仍保持99.5%准确率
- MMLongBench-Doc: 57.0%准确率,SOTA表现

**领域专项**:
- **OCR/文档**: OCRBench 920分,支持39种语言,32/39语言准确率>70%
- **2D/3D Grounding**: RefCOCO 91.9%,ODinW-13 48.6 mAP,3D grounding在SUNRGBD上超越Gemini-2.5-Pro 5.2点
- **视频理解**: VideoMME 79.2%,MLVU 84.3%,在长视频理解上超越Gemini-2.5-Pro
- **GUI Agent**: ScreenSpot Pro 62.0%,OSWorld 38.1%,AndroidWorld 63.7%
- **Fine-grained Perception**: 使用工具后V* 93.7%,HRBench4K 85.4%
- **STEM推理**: MathVista 85.8%(thinking),MathVision 74.6%,MMMU 80.6%

**思考模式收益**:
- Thinking模式在推理密集型任务上带来显著提升(如AIME-25: 89.7% vs 74.7%, HMMT-25: 77.4% vs 57.4%)
- 在某些任务上instruct模式反而更好(如RealWorldQA),说明需要针对应用场景选择模式

**局限性**

论文未明确指出局限性,但从架构和实验设计可推断:训练成本较高(四阶段预训练+三阶段后训练),对于资源受限的场景可能难以复现;虽然支持256K上下文,但在超长序列(>256K)上仍需YaRN外推;thinking模式虽然提升推理能力,但会增加推理延迟和成本。
