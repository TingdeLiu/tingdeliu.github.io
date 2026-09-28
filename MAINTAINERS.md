# 维护者说明 / Maintainers

## 维护者

| | |
| --- | --- |
| **主要维护者 (Lead Maintainer)** | Tingde Liu ([@TingdeLiu](https://github.com/TingdeLiu)) |
| **角色** | 项目创建者、唯一主维护者；负责全部内容撰写、站点工程、Issue 处理与许可合规 |
| **联系方式** | <tingde.liu.luh@gmail.com> · [LinkedIn](https://www.linkedin.com/in/tingde-liu-379818270/) |
| **维护起始** | 2026-01-04（仓库创建日） |

本项目目前为**单一维护者项目**。这里如实说明这一点，而不是用复数的 "maintainers"
来暗示一个并不存在的团队。如果你有兴趣长期共同维护某个方向（例如接手某一类综述的
更新），欢迎开 Issue 联系。

---

## 维护范围

维护者对以下各项负责：

1. **内容正确性** —— 文章中的方法描述、性能数字、论文归类；收到反馈后回原文核对
2. **内容时效性** —— 具身导航周报按周更新；综述随领域进展持续增补
3. **站点可用性** —— Jekyll 主题、构建流水线、GitHub Pages 部署
4. **反馈处理** —— 全部 Issue 的分诊、核实、答复与关闭
5. **许可合规** —— 代码 / 原创内容 / 第三方论文图片的授权边界维护与下架请求处理

---

## 项目现状（截至 2026-09-16）

以下数字均可在仓库页面直接核验：

| 指标 | 数值 |
| --- | --- |
| Stars | 76 |
| 提交总数 | 718（2026-01-04 起，连续 9 个月每月均有提交） |
| Pull Request | 12 个，全部已合并 |
| 社区 Issue | 5 个，全部已处理关闭（100%） |
| 研究综述与论文精读 | 17 篇 |
| 技术博客 | 6 篇 |
| 具身导航周报 | 已上线 4 期（另有 2 期草稿未发布），持续更新中 |
| Markdown 正文 | 约 5.7 万行 |
| 配图 | 1003 张（统一 WebP 交付） |

> 说明：12 个 PR 中 11 个由维护者本人在特性分支上提交并合并，1 个来自 GitHub Copilot
> 编码代理。**目前尚无外部人类贡献者的 PR**，这里不做模糊表述。外部参与目前集中在
> Issue 反馈渠道，记录见下节。

---

## 社区反馈处理记录

本站每篇文章末尾内置反馈组件，读者可一键提交「错误反馈 / 论文推荐 / 修改建议」。
下表是**全部**历史反馈及其处理结果，来自 5 位互不相同的外部读者：

| Issue | 提交者 | 类型 | 内容 | 提交 → 关闭 | 处理时长 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| [#17](https://github.com/TingdeLiu/tingdeliu.github.io/issues/17) | @feigemicer-cloud | 错误反馈 | 《深度学习综述》RoPE 章节表格未被解析，渲染成纯文字 | 2026-09-01 → 09-02 | 1 天 | ✅ 已修复（定位为表格首行前缺空行导致 kramdown 未解析） |
| [#8](https://github.com/TingdeLiu/tingdeliu.github.io/issues/8) | @cbcbHH | 论文推荐 | 推荐收录 OnFly（无人机零样本 VLN） | 2026-06-22 → 06-25 | 3 天 | ⚖️ **未采纳，已说明理由**：本篇聚焦室内机器人 VLN，暂不覆盖 UAV VLN |
| [#7](https://github.com/TingdeLiu/tingdeliu.github.io/issues/7) | @11klow | 修改建议 | 建议补全论文全称与发表会议/期刊；建议按方法论关键词分类 | 2026-06-17 → 06-18 | 1 天 | ✅ 部分采纳 —— 已上线「已发表论文（会议/期刊）」对照表与交互式标签筛选栏 |
| [#6](https://github.com/TingdeLiu/tingdeliu.github.io/issues/6) | @jiangqs472-sketch | 错误反馈 | 《VLN 论文精读：指令跟随篇》排行榜中 AwareVLN 的 SR/SPL/OS 数值与原论文不符 | 2026-06-08 → 06-11 | 3 天 | ✅ 已核对原论文并修正 |
| [#5](https://github.com/TingdeLiu/tingdeliu.github.io/issues/5) | @DAHM7048 | 错误反馈 | 《机器学习综述》3.20 扩散模型一节 MathJax 公式渲染失败 | 2026-04-14 → 04-17 | 3 天 | ✅ 已修复 |

**汇总**

- 处理率：**5 / 5（100%）**，无挂起、无因超时自动关闭
- 响应时长：最短 1 天，最长 3 天，中位数 3 天
- 每一条均有维护者的实名回复，包括未采纳的那条
- 采纳的反馈均转化为了实际提交；#7 的建议直接促成了论文列表的会议/期刊对照表与标签筛选功能

> 关于 #8 的说明：维护者认为**明确拒绝并给出边界理由，与采纳同样重要**。
> 项目的收录范围需要保持稳定，否则综述会失焦。反馈渠道的价值在于认真答复，
> 而不在于来者不拒。

---

## 处理反馈的原则

收到内容类反馈时，维护者按以下顺序判断：

1. **能否核验** —— 回到原始论文/官方文档确认。无法核验的会要求补充来源，而不是
   凭印象修改
2. **是否在收录范围内** —— 每篇综述有既定的主题边界（例如 VLN 论文精读：指令跟随篇
   聚焦指令跟随类机器人导航）。超出范围的会明确说明并拒绝
3. **是否值得单独成节** —— 重要工作做深度精读，增量改进并入已有小节
4. **无论采纳与否都要答复** —— 不静默关闭

代码类 PR 的判断标准见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 响应承诺

| 渠道 | 首次回复目标 |
| --- | --- |
| 内容 Issue（错误 / 推荐 / 建议） | 3 个工作日内 |
| Pull Request | 5 个工作日内 |
| 安全报告（见 [SECURITY.md](SECURITY.md)） | 3 个工作日内确认，7 个工作日内给出评估 |
| 著作权下架请求 | 3 个工作日内确认，核实后立即处理 |

这些是个人维护者的目标，不是 SLA。历史记录（上表）显示实际响应在 1–3 天区间。
若超时未回复，欢迎在 Issue 中 @ 维护者或直接发邮件催促。

---

## 部署与更新方式

本项目是持续更新的内容站点，不做版本化发布：

- 推送 `main` 后由 GitHub Actions 自动构建并部署到 GitHub Pages
- 线上版本始终对应 `main` 的最新提交，不存在需要单独维护的发布分支
- 需要引用某个确定时刻的内容时，请直接引用对应的 commit 链接，而不是可变的页面 URL
- 安全问题请始终针对 `main` 的当前状态报告，见 [SECURITY.md](SECURITY.md)

---

## 单点风险说明 / Bus Factor

如实声明：本项目的 bus factor 为 **1**。所有内容与维护知识集中在单一维护者处。

缓解措施：

- 全部内容以纯 Markdown 保存在 Git 中，无私有格式、无外部数据库依赖
- 代码采用 [MIT License](LICENSE)，原创内容采用 [CC BY 4.0](LICENSE-CONTENT)，
  任何人都可以在遵守署名要求的前提下 fork 并继续维护
- 构建流程完全基于开源 Jekyll 与 GitHub Actions，无自建基础设施

---

## English Summary

**Lead maintainer:** Tingde Liu ([@TingdeLiu](https://github.com/TingdeLiu)),
creator and sole maintainer since 2026-01-04. Responsible for all content,
site engineering, issue triage, and licensing compliance.

**Project status (2026-09-16):** 76 stars · 718 commits across 9 consecutive
active months · 12 merged PRs · 17 research surveys, 6 technical posts, and an
ongoing weekly digest (~57k lines of Markdown, 1003 figures).

**Community feedback:** 5 issues from 5 distinct external readers, **all 5
resolved and closed (100%)**, with a 1–3 day turnaround and a named reply on
every one — including the one that was declined with a stated scope rationale
(#8). Two accepted reports produced verified content corrections; one
suggestion (#7) led directly to the venue-mapping table and the interactive tag
filter now shipping on the VLN paper list.

**Note on contributors:** 11 of the 12 PRs were authored by the maintainer on
feature branches; 1 came from the GitHub Copilot coding agent. There are no
external human PR contributors to date — stated plainly rather than obscured.

**Deployment:** continuous — pushing to `main` triggers a GitHub Actions build
and deploy to GitHub Pages. There are no versioned releases; cite a specific
commit when a fixed point-in-time reference is needed.

**Bus factor: 1.** Mitigated by plain-Markdown content in Git, permissive
licensing (MIT for code, CC BY 4.0 for original prose), and a fully open-source
build pipeline.
