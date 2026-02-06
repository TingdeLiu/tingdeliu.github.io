# LLM Training Survey 图片资源

本文档说明如何为博客文章添加论文架构图片。

## 需要的图片

### 1. RLHF 三阶段流程图
- **文件名**: `rlhf-three-steps.png`
- **来源论文**: [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- **图片位置**: 论文 Figure 2
- **下载方法**:
  1. 访问 https://arxiv.org/pdf/2203.02155.pdf
  2. 找到 Figure 2（在第 3 页）
  3. 截图或提取该图片
  4. 保存为 `rlhf-three-steps.png`

### 2. DPO vs RLHF 对比图
- **文件名**: `dpo-vs-rlhf.png`
- **来源论文**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **图片位置**: 论文 Figure 1
- **下载方法**:
  1. 访问 https://arxiv.org/pdf/2305.18290.pdf
  2. 找到 Figure 1（在第 2 页）
  3. 截图或提取该图片
  4. 保存为 `dpo-vs-rlhf.png`

### 3. RLAIF 工作流程图
- **文件名**: `rlaif-workflow.png`
- **来源论文**: [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
- **图片位置**: 论文 Figure 1 或 Figure 2
- **下载方法**:
  1. 访问 https://arxiv.org/pdf/2309.00267.pdf
  2. 找到合适的架构图
  3. 截图或提取该图片
  4. 保存为 `rlaif-workflow.png`

### 4. Constitutional AI 流程图
- **文件名**: `constitutional-ai.png`
- **来源论文**: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- **图片位置**: 论文 Figure 1
- **下载方法**:
  1. 访问 https://arxiv.org/pdf/2212.08073.pdf
  2. 找到 Figure 1（显示两阶段流程）
  3. 截图或提取该图片
  4. 保存为 `constitutional-ai.png`

## 图片处理建议

1. **分辨率**: 建议 1200-1600 像素宽度
2. **格式**: PNG 格式（支持透明背景）
3. **文件大小**: 尽量控制在 500KB 以内
4. **压缩工具**: 可以使用 [TinyPNG](https://tinypng.com/) 压缩图片

## 添加图片的步骤

1. **创建目录**（如果不存在）:
   ```bash
   mkdir -p assets/images/llm-training
   ```

2. **放置图片**:
   将下载的图片放到 `assets/images/llm-training/` 目录下

3. **验证图片路径**:
   确保文件名与博客文章中引用的路径一致

4. **提交到仓库**:
   ```bash
   git add assets/images/llm-training/
   git commit -m "添加偏好对齐方法的架构图"
   git push
   ```

5. **等待部署**:
   GitHub Pages 会自动部署，1-3 分钟后图片即可显示

## 版权说明

这些图片来自学术论文，用于教育和学习目的。根据学术惯例：
- ✅ 在博客中引用论文图片进行学术讨论是允许的
- ✅ 需要标明图片来源（论文标题和作者）
- ✅ 用于非商业、教育目的

## 备选方案

如果无法获取原论文图片，可以：
1. 使用论文中的表格或算法伪代码代替
2. 绘制简化版的架构图
3. 继续使用 Mermaid 图表（虽然不如论文图片专业）

## 联系方式

如有问题，可以在 GitHub Issues 中提出。
