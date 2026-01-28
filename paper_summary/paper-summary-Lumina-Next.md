---
## 1. Lumina-Next (2024)
——通过 Next-DiT 让 Lumina-T2X 更强更快

📄 **Paper**: https://arxiv.org/abs/2406.xxxxx (NeurIPS 2024)

**精华**

这篇论文展示了如何通过架构优化和推理技术改进 diffusion transformer 的关键方法,值得借鉴的核心思想包括:(1) 使用 3D RoPE 和 sandwich normalization 控制网络激活量级,实现更稳定的训练和更好的分辨率外推能力;(2) 提出 Frequency- 和 Time-Aware Scaled RoPE,针对 diffusion 模型的时间感知特性设计位置编码外推策略;(3) 通过优化时间离散化调度和 Time-Aware Context Drop 显著提升推理效率;(4) 采用 flow matching 框架和 transformer 架构构建统一的多模态生成框架;(5) 使用更强大的 decoder-only LLM 作为文本编码器实现零样本多语言生成能力。

**研究背景/问题**

Lumina-T2X 作为基于 Flow-based Large Diffusion Transformer (Flag-DiT) 的生成模型家族,虽然在多模态生成方面展现出潜力,但仍然面临训练不稳定、推理速度慢和分辨率外推时出现伪影等挑战。现有的 diffusion transformer 架构在扩展到大规模模型和长序列时存在激活量级难以控制的问题,且缺乏针对视觉生成任务优化的位置编码外推策略。

**主要方法/创新点**

<div align="center">
  <img src="/images/Lumina-Next-architecture-comparison.png" width="100%" />
<figcaption>
Flag-DiT 和 Next-DiT 的架构对比,展示了主要改进点
</figcaption>
</div>

**Next-DiT 架构改进:**

1. **3D RoPE 替代 1D RoPE**: 原始 Flag-DiT 使用 1D RoPE 将图像 token 序列化处理,丢失了空间和时间关系。Next-DiT 采用 3D RoPE,将嵌入维度分为三个独立部分,分别编码 x、y、z 轴的位置信息,提供统一且精确的时空位置表示,无需额外的可学习标识符如 [nextline] 和 [nextframe]。

2. **Sandwich Normalization**: 通过在每个 attention 和 MLP 层前后都添加 RMSNorm,有效控制网络激活量级的增长。实验表明,Flag-DiT 中存在的长信号路径会导致激活值在深层累积,特别是在分辨率外推时,早期层的小误差会在后续层被放大。Sandwich normalization 配合 tanh gating 机制,稳定了训练和采样过程。

<div align="center">
  <img src="/images/Lumina-Next-rope-comparison.png" width="100%" />
<figcaption>
1D RoPE 与 2D RoPE 的注意力模式对比,以及不同外推策略在 2K 生成中的效果
</figcaption>
</div>

3. **Grouped-Query Attention**: 采用 GQA 将 32 个 query heads 分成 8 组,共享 8 个 key-value heads,在保持性能的同时降低参数量和计算复杂度,特别适合高分辨率图像生成。

**Frequency- 和 Time-Aware Scaled RoPE:**

针对 3D RoPE 的分辨率外推,论文系统比较了现有方法并提出创新策略:

1. **Frequency-Aware Scaled RoPE**: 识别波长等于训练序列长度的维度 d_target,通过 b' = b · s^(d_head/d_target) 调整频率基,使该维度等效于位置插值,有效减少内容重复问题。

2. **Time-Aware Scaled RoPE**: 考虑 diffusion 模型先生成全局概念再生成局部细节的特性,设计时间相关系数 d_t = (d_head - 1)·t + 1,在去噪早期使用位置插值确保全局结构,在后期逐渐转向 NTK-Aware Scaled RoPE 保留局部细节。

<div align="center">
  <img src="/images/Lumina-Next-convergence.png" width="100%" />
<figcaption>
Next-DiT 在 ImageNet-256 基准上的收敛速度对比
</figcaption>
</div>

**优化采样效率:**

1. **Sigmoid 时间调度**: 分析发现 Flow ODE 的离散化误差在纯噪声(t=0)附近最大,在接近干净数据(t=1)时相对较大。提出 sigmoid 分段函数确保采样开始和结束阶段的步长大于中间步骤,配合高阶 ODE solver(如 midpoint method)实现 5-10 步高质量生成。

2. **Time-Aware Context Drop**: 对 keys 和 values 进行空间平均池化合并相似 token,减少注意力计算冗余。与 Token Merging 不同,该方法只下采样 KV 而保留完整的视觉内容,并且引入时间感知机制,在 t=0 时最大程度 drop token 提升效率,在 t=1 时不 drop 以保持视觉质量,在 1K 分辨率生成中实现 2× 推理加速。

<div align="center">
  <img src="/images/Lumina-Next-resolution-extrapolation.png" width="100%" />
<figcaption>
4× 分辨率外推对比,Lumina-Next 相比其他方法展现出更好的全局一致性和局部细节
</figcaption>
</div>

**统一多模态框架:**

论文将 Lumina-Next 扩展到多种模态:
- **多视图生成**: 引入相对姿态控制和图像条件,支持任意视图数量的灵活推理
- **音频/音乐生成**: 使用 1D 卷积 VAE 编码 mel-spectrogram,配合 CLAP+FLAN-T5 双文本编码器
- **点云生成**: 提出 Time-aware Scaled Fourier feature,实现密度不变的点云生成,可以在 256 点训练后生成 8192 点的高密度点云
- **任意分辨率识别**: 通过 dynamic partitioning 和 masked attention 实现图像识别任务的分辨率外推

**训练数据改进:**

采用 Mixture-of-Captioners (MoC) 策略,使用多个预训练 VLM(BLIP2, LLaVA, SPHINX, Share-GPT4V)生成互补的图像描述,并使用 GPT-4V 对 10 万张高分辨率图像进行多方面整体描述。训练时从 caption pool 随机采样,增强模型对不同风格用户输入的鲁棒性。数据集从 14M 扩展到 20M 高质量图像-文本对。

**核心结果/发现**

1. **架构性能**: Next-DiT 在 ImageNet-256 上训练 100 epochs 即达到 81.6% Top-1 准确率,接近 DeiT-base 训练 300 epochs 的结果。训练 300 epochs 后达到 82.3%,显著超越 DeiT-base 的 81.6%。

2. **文生图质量**: 2B 参数的 Next-DiT 配合 Gemma-2B 文本编码器,在生成质量上超越了 5B Flag-DiT 配合 LLaMA-7B 的 Lumina-T2X,同时显著降低训练和推理成本。

3. **分辨率外推**: 支持训练时未见过的任意分辨率和宽高比生成,在 2K 和全景图生成任务上展现出全局一致性和丰富的局部细节,优于 MultiDiffusion、DemoFusion 和 ScaleCrafter 等方法。

4. **少步采样**: 使用 midpoint solver 配合 sigmoid schedule,在 10-20 个函数评估步数内生成高质量图像,性能持续优于使用 DPM-Solver 的 PixArt-α 和 SDXL。

5. **多语言能力**: 使用 decoder-only LLM 作为文本编码器,实现零样本多语言文生图能力,在 15 种语言测试中展现出比 SDXL 和 PixArt-α 更好的文本理解和文化细微差别捕捉能力。

6. **推理效率**: Time-Aware Context Drop 在 1K 分辨率下实现 2× 加速,且在配合 Flash Attention 时对超高分辨率图像生成的加速更显著。

7. **多模态扩展**:
   - 音乐生成: FAD 3.75, MOS-Q 83.56, MOS-F 85.69,优于 MusicLDM 和 AudioLDM 2
   - 音频生成: FAD 1.03, MOS-Q 77.53, MOS-F 76.52,优于 Make-An-Audio 2 和 AudioLDM 2
   - 多视图生成: 512×512 分辨率支持 1-8 视图灵活推理
   - 点云生成: 在 airplane 和 chair 类别上达到与 PDiffusion 相当的 MMD 和 COV 指标

8. **任意分辨率识别**: 经过 any-resolution fine-tuning 后,在 ImageNet-1K 上达到 84.2% Top-1 准确率,在 1024×1024 分辨率上显著优于 DeiT-base。

**局限性**

尽管 Lumina-Next 在多个方面取得显著改进,但在文本-图像对齐和视觉美学方面仍落后于 Midjourney 和 DALLE 3 等闭源模型。主要差距在于多阶段训练使用的文本-图像对数量:论文将数据集扩展到 2000 万,但仍远小于闭源模型使用的数据规模。此外,使用人类偏好数据进行微调(如 Direct Preference Optimization)对提升图像质量也很重要,这是未来研究的方向。
