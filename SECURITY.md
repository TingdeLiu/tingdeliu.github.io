# 安全策略 / Security Policy

## 这个项目的真实攻击面

先把范围说清楚，避免浪费你的时间：

本仓库是一个**纯静态站点**（Jekyll → GitHub Pages）。它**没有后端服务、没有数据库、
不接收用户输入、不存储任何用户数据、不处理认证或支付**。站点唯一的用户交互是
Google Analytics 统计和跳转到 GitHub Issues 的反馈链接。

因此常见的 Web 漏洞类型（SQL 注入、SSRF、越权、会话固定等）在这里不适用。
实际存在的风险面只有以下几类：

| 风险面 | 具体内容 |
| --- | --- |
| **构建供应链** | `.github/workflows/deploy.yml` 中的 GitHub Actions、`Gemfile` 中的 Ruby 依赖 |
| **第三方 CDN** | 页面从 `cdn.jsdelivr.net` 加载 KaTeX 0.19、Mermaid 10、Gitalk，从 `events.vercount.one` 加载访客统计脚本（CDN 被投毒或劫持会导致任意脚本执行）。其中 KaTeX 固定了版本并带 SRI 校验；Mermaid 与 Gitalk 只固定了大版本、没有 SRI；统计脚本是无版本号的固定地址，无法加 SRI |
| **前端注入** | `_layouts/`、`_includes/`、`js/` 中处理文章内容的逻辑，若存在未转义的 HTML 注入点 |
| **仓库与部署配置** | Actions 权限范围、Pages 部署配置、分支保护 |

如果你发现的问题落在上述范围内，非常欢迎报告。

## 支持的版本

站点只有一个生产版本，即 `main` 分支的最新部署（<https://tingdeliu.github.io/>）。
本项目不做版本化发布，也没有需要单独维护的旧版本。请始终针对 `main` 的当前状态报告。

## 如何报告

**请不要通过公开 Issue 报告安全问题。**

按优先级选择：

1. **GitHub Private Vulnerability Reporting**（推荐）
   仓库 [Security 页面](https://github.com/TingdeLiu/tingdeliu.github.io/security/advisories/new) →
   "Report a vulnerability"。全程私密，可直接在其中协作修复。

2. **邮件**：<tingde.liu.luh@gmail.com>，标题请以 `[SECURITY]` 开头。

### 报告中请包含

- 受影响的文件或页面 URL
- 复现步骤（能给出最小复现最好）
- 你判断的影响范围
- 可选：你建议的修复方式

### 响应承诺

| 阶段 | 时间 |
| --- | --- |
| 确认收到 | 3 个工作日内 |
| 初步评估结论 | 7 个工作日内 |
| 修复并部署 | 视严重程度，高危问题优先处理 |

这是**个人维护的开源项目**，没有值班团队。上述时间是维护者的目标而非 SLA；
如果超时未回复，欢迎再发一封邮件催促。

修复后，若你同意，会在对应的 commit 或 Security Advisory 中致谢。本项目**没有
漏洞赏金计划**，请在报告前知悉。

## 不在受理范围内的问题

以下情况会被直接关闭，请不要提交：

- 自动化扫描器输出，未经人工验证、无法说明实际影响
- 缺少 CSP / HSTS / X-Frame-Options 等响应头 —— GitHub Pages 的响应头不受本仓库控制
- 站点使用公共 CDN 本身（这是已知的、有意的权衡，见上表）
- 静态站点上的"缺少 CSRF token""缺少速率限制"一类不适用的报告
- 社会工程、物理攻击、针对维护者个人账号的攻击
- 文章内容中的事实错误 —— 这类请走 [Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new)，
  见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 著作权与内容下架请求

如果你是本站引用的某张论文插图或某段文字的权利人，认为使用超出了合理引用范围，
请同样通过上述邮件联系（标题以 `[COPYRIGHT]` 开头），并注明：

- 具体的图片路径或文章 URL
- 你主张权利的依据

维护者会在核实后**及时移除或补充授权说明**。相关政策见
[LICENSE-CONTENT](LICENSE-CONTENT) 第 4 节。

---

## Security Policy (English Summary)

This is a **static Jekyll site** deployed to GitHub Pages. There is no backend,
no database, no user input handling, and no stored user data. The real attack
surface is limited to: the build supply chain (GitHub Actions, Ruby gems),
third-party CDN assets (KaTeX, Mermaid and Gitalk via jsDelivr, where KaTeX is
version-pinned with SRI hashes and Mermaid and Gitalk are pinned to a major
version only; the Vercount visitor-counter script, which is unversioned and
therefore cannot carry an SRI hash), front-end
injection points in layouts and scripts, and repository/deployment configuration.

**Do not report vulnerabilities in public issues.** Use
[GitHub Private Vulnerability Reporting](https://github.com/TingdeLiu/tingdeliu.github.io/security/advisories/new)
or email <tingde.liu.luh@gmail.com> with a `[SECURITY]` subject prefix.

Acknowledgement within 3 business days, initial assessment within 7 business days.
This is a personally maintained project with no on-call rotation and no bug
bounty program. Only the current state of `main` is supported; there are no
versioned releases.

Out of scope: unverified scanner output, missing HTTP security headers (not
controllable on GitHub Pages), CDN usage itself, and non-applicable findings for
static sites. Copyright takedown requests: email with a `[COPYRIGHT]` prefix.
