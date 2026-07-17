---
layout: post
title: "Loop Engineering：Agent 工程化的下一代闭环范式"
date:   2026-07-17
tags: [Loop Engineering, Agentic Workflow, Harness Engineering, AI Engineering, Claude Code, Codex]
categories: blog
comments: true
author: Tingde Liu
toc: true
excerpt: "Loop Engineering（循环工程）是 2026 年 AI 智能体开发的最核心演进。它将开发者的角色从“亲自 Prompt 的打字员”转变为“设计 Agent 闭环系统的架构师”。本文以专业 Agent 工程师视角，拆解 Loop Engineering 的五大核心原语与持久化状态，剖析典型闭环工作流，并直面验证盲区、认知 surrender 与 Token 成本控制等实战挑战。"
---


# 1. 引言：从“手写 Prompt”到“设计系统”的认知跃迁

在过去两年的 AI 辅助开发中，我们绝大多数人与 AI 编码助手的交互方式都可以总结为一种**“单步对话”模式**：你写一段 Prompt 并共享相关代码，点击发送，阅读返回的代码，发现有 Bug 或遗漏，然后再写一段 Prompt 告诉它如何修改。在这个过程中，AI 是一个被动的工具，而你则是高强度的键盘操作者，需要在每一个回合中手动监督、判断和推进。

但正如 Anthropic 的 Claude Code 负责人 Boris Cherny 所说：**“我不再直接 Prompt Claude 了。我有专门运行的循环（loops）去 Prompt Claude 并弄清楚下一步该做什么。我的工作是编写这些循环。”**

这标志着 AI 工程界一次重大的认知转移：**你不应该再亲自去 Prompt 编码智能体，而是应当设计一套“替你 Prompt 智能体”的系统。**

这种设计自主、迭代式执行周期的系统性方法，被称为 **Loop Engineering（循环工程）**。

<div align="center"><img src="/images/agent/loop-engineering-hero.jpg" width="90%" alt="Loop Engineering Hero" /><figcaption>图 1：Loop Engineering 核心概念——从“手动 Prompt”转向“为 Agent 编写自动化闭环系统”</figcaption></div>

<!-- more -->

传统的手动 Prompt 模式由于需要人类时刻处于闭环中（Human-in-the-loop），极大地限制了 Agent 执行大规模、长周期任务的上限。而在 Loop Engineering 模式下，开发者编写的是“控制循环”：定义清晰的目标、环境约束、验证条件与终止规则，让系统自己在前台反复调试、运行、校验、自我纠错，直到达成目标。

<div align="center">
<svg width="720" height="200" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">
  <!-- Background -->
  <rect width="720" height="200" rx="12" fill="#0f172a" stroke="#1e293b" stroke-width="1.5"/>
  <!-- Left Side: Manual Prompting -->
  <rect x="20" y="20" width="310" height="160" rx="10" fill="#1e293b" stroke="#334155" stroke-width="1"/>
  <text x="175" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="#f8fafc">传统单步 Prompting 模式</text>
  <text x="175" y="65" text-anchor="middle" font-size="11" fill="#94a3b8">（Human-in-the-Loop, 极高交互成本）</text>
  <rect x="40" y="85" width="80" height="40" rx="6" fill="#f59e0b" stroke="#d97706" stroke-width="1"/>
  <text x="80" y="109" text-anchor="middle" font-size="12" font-weight="bold" fill="#78350f">开发者</text>
  <!-- Right Arrow -->
  <path d="M 130 95 L 210 95 L 210 100 L 220 92 L 210 84 L 210 89 L 130 89 Z" fill="#94a3b8"/>
  <text x="175" y="82" text-anchor="middle" font-size="10" fill="#94a3b8">1. 手写 Prompt</text>
  <rect x="230" y="85" width="80" height="40" rx="6" fill="#3b82f6" stroke="#2563eb" stroke-width="1"/>
  <text x="270" y="109" text-anchor="middle" font-size="12" font-weight="bold" fill="#ffffff">LLM Agent</text>
  <!-- Left Arrow -->
  <path d="M 230 120 L 150 120 L 150 115 L 140 123 L 150 131 L 150 126 L 230 126 Z" fill="#94a3b8"/>
  <text x="190" y="145" text-anchor="middle" font-size="10" fill="#94a3b8">2. 人工阅读与纠错</text>
  <!-- Right Side: Loop Engineering -->
  <rect x="390" y="20" width="310" height="160" rx="10" fill="#1e1b4b" stroke="#312e81" stroke-width="1"/>
  <text x="545" y="45" text-anchor="middle" font-size="14" font-weight="bold" fill="#f8fafc">现代 Loop Engineering 模式</text>
  <text x="545" y="65" text-anchor="middle" font-size="11" fill="#a5b4fc">（Autonomous Loop, 系统化编排）</text>
  <rect x="405" y="85" width="70" height="60" rx="6" fill="#10b981" stroke="#059669" stroke-width="1"/>
  <text x="440" y="110" text-anchor="middle" font-size="11" font-weight="bold" fill="#064e3b">开发者</text>
  <text x="440" y="125" text-anchor="middle" font-size="9" fill="#064e3b">设计循环机制</text>
  <path d="M 485 115 L 505 115" stroke="#818cf8" stroke-width="2" stroke-dasharray="3 3"/>
  <polygon points="505,111 513,115 505,119" fill="#818cf8"/>
  <rect x="520" y="85" width="160" height="60" rx="6" fill="#4f46e5" stroke="#4338ca" stroke-width="1"/>
  <text x="600" y="105" text-anchor="middle" font-size="11" font-weight="bold" fill="#ffffff">Agent Execution Loop</text>
  <text x="600" y="120" text-anchor="middle" font-size="9" fill="#c7d2fe">触发器 ➔ 执行 ➔ 判定 ➔ 记忆</text>
  <text x="600" y="135" text-anchor="middle" font-size="9" fill="#c7d2fe">（自动化闭环运行）</text>
</svg>
<figcaption>图 2：单步交互与 Loop Engineering 的对比示意图</figcaption>
</div>

---

# 2. AI Agent 的层级栈：Loop 处于哪一层？

为了精确给 Loop Engineering 定位，我们必须了解现代 AI 智能体开发的技术层级栈（The Agentic Stack）。在业内，开发者通常将整个 Agent系统划分为四个核心层级：

<div align="center">
<svg width="720" height="280" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">
  <!-- Background -->
  <rect width="720" height="280" rx="12" fill="#0b0f19" stroke="#1e293b" stroke-width="1.5"/>
  <!-- Title -->
  <text x="360" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#f8fafc">AI Agent 架构层级划分 (The Agentic Stack)</text>
  <!-- Level 4: Loop Engineering -->
  <rect x="80" y="60" width="560" height="42" rx="6" fill="#312e81" stroke="#4f46e5" stroke-width="2"/>
  <text x="100" y="86" font-size="13" font-weight="bold" fill="#e0e7ff">④ 循环工程 (Loop Engineering)</text>
  <text x="350" y="85" font-size="11" fill="#c7d2fe">自主流程、任务编排、Maker-Checker 机制、状态外部持久化</text>
  <!-- Level 3: Harness Engineering -->
  <rect x="80" y="112" width="560" height="42" rx="6" fill="#064e3b" stroke="#10b981" stroke-width="2"/>
  <text x="100" y="138" font-size="13" font-weight="bold" fill="#d1fae5">③ 驾驭工程 (Harness Engineering)</text>
  <text x="350" y="137" font-size="11" fill="#a7f3d0">运行环境、执行沙箱、工具与接口（MCP）、护栏与安全约束</text>
  <!-- Level 2: Context Engineering -->
  <rect x="80" y="164" width="560" height="42" rx="6" fill="#1e3a8a" stroke="#3b82f6" stroke-width="2"/>
  <text x="100" y="190" font-size="13" font-weight="bold" fill="#dbeafe">② 上下文工程 (Context Engineering)</text>
  <text x="350" y="189" font-size="11" fill="#bfdbfe">知识库、动态检索、RAG、系统级说明文件（AGENTS.md）</text>
  <!-- Level 1: Prompt Engineering -->
  <rect x="80" y="216" width="560" height="42" rx="6" fill="#78350f" stroke="#f59e0b" stroke-width="2"/>
  <text x="100" y="242" font-size="13" font-weight="bold" fill="#fef3c7">① 提示工程 (Prompt Engineering)</text>
  <text x="350" y="241" font-size="11" fill="#fde68a">指令调优、Few-shot 样本、单次问答逻辑、推理思维链控制</text>
</svg>
<figcaption>图 3：现代 AI Agent 架构层级栈</figcaption>
</div>

1. **提示工程 (Prompt Engineering)**：研究如何优化单次向大模型输入的信息结构，以获得更准确的瞬时推理（如 Chain-of-Thought, ReAct 元指令设计）。
2. **上下文工程 (Context Engineering)**：关注如何动态提取、缓存与更新模型在解决特定问题时所需的相关信息（例如在 IDE 中动态检索相关定义、代码片段等）。
3. **驾驭工程 (Harness Engineering)**：为 Agent 搭建安全的执行“疆界”。它设计 Agent 所处的沙箱环境（Sandbox）、系统 API 权限、终端运行工具以及安全限制护栏。
4. **循环工程 (Loop Engineering)**：**层级最高，起编排作用。** 它负责在 Harness 之上构建随时间推移自动运行的驱动周期——包括设定调度触发、并行隔离逻辑、判定任务通过的 Checker 守护机制，以及全局持久化记忆管理。

如果把 Harness 比作 Agent 在其中工作的一个具有完善防爆护栏和工具的**生产实验室**，那么 Loop 就是实验室之上的**自动化生产调度流水线**。

---

# 3. Loop Engineering 的五大核心原语 (Primitives) 与 记忆 (State)

要把一个手动的 Agent 会话，转化为自运行的“循环”，我们需要借助**五大基础工程原语（Primitives）**以及**一个核心状态外部化机制（State）**。这些在当前的 Codex App 与 Claude Code 等一线开发者工具中，都已作为核心产品组件得到高度映射：

| 原语组件 (Primitive) | 核心职责 | Codex App 落地细节 | Claude Code 落地细节 |
| :--- | :--- | :--- | :--- |
| **1. 自动触发器 (Automations)** | 发现与分发任务，确定循环节奏 | Automations 面板（周期配置、测试沙箱、`/goal` 指令） | Scheduled tasks（Cron 运行、Git 提交 Hooks、`/goal` 守护） |
| **2. 分支工作区 (Worktrees)** | 隔离并发任务，防止文件写入冲突 | 内部自动分配 Thread-level 独立分支 | `git worktree` 支持、`isolation: worktree` 配置 |
| **3. 项目技能库 (Skills)** | 固化项目工程规范，杜绝 AI 乱猜 | `SKILL.md`（定义构建、格式和设计规约） | `SKILL.md`（定义在指定目录下，运行时自动挂载） |
| **4. 外部连接器 (Connectors)** | 将 Agent 接入外部生产力工具 | MCP Connectors（支持 Linear, GitHub, Slack 等） | MCP 协议服务器、外部 API 工具集成扩展 |
| **5. 多代理体系 (Sub-agents)** | 实施“ Maker-Checker ”责任分离 | `.codex/agents/` 下定义的子角色（TOML 格式） | `.claude/agents/` 下定义的子代理（Markdown/JSON 配置） |

### ① 自动触发器 (Automations)
自动触发器是 Loop 的“心脏搏动”。它通常是一个定时任务（Cron）或一个事件订阅者。例如，每天清晨，触发器扫描 CI 失败日志、Linear 未决 Bug 看板和 GitHub 社区 Issues。一旦捕获到需要修复的任务，它就启动循环。在会话内部，这对应了 `/goal` 这种“直到达成某种特定判定状态前，持续自我纠正运行”的原语。

### ② 分支工作区 (Worktrees)
当我们让多个 Agent 并行处理不同任务时，代码文件的写冲突是致命的。如果两个 Agent 都在直接改工作目录下的同一个接口文件，就会出现类似“多人共同操作一块黑板”的灾难。Loop Engineering 引入了 `git worktree` 级的隔离：每个 sub-agent 的每一个修复尝试，都在独立的 branch 工作区运行，互不干扰，在验证成功后再以 PR 形式汇入主干。

### ③ 项目技能库 (Skills)
Agent 在启动循环时是“失忆”的。如果完全不加引导，它很容易凭感觉去“盲猜”你们团队的编码规范、打包逻辑或者特殊架构规约。这就是为什么我们需要在仓库中固化 `SKILL.md`。它作为一套外置的“意图载体”，每次循环开始时都会被加载。通过 Skills，我们可以告诉 Agent：“不要使用传统的 X 库，我们公司统一在 utils/ 中封装了 Y 工具；所有提交必须先经过 pnpm lint。”

### ④ 外部连接器 (Connectors)
Loop 不能是一座只能操作本地文件系统的孤岛。通过 Model Context Protocol (MCP) 建立的 Connectors，使循环能够拥有对外交互的“感官与肢体”：
* 在 issue 看板检索最新 Bug 描述（输入通道）。
* 提交 PR 到 GitHub（输出通道）。
* 在 Slack 通报“已自动修复第 #382 号崩溃 Bug，等待 review”（通知机制）。

### ⑤ 多代理体系 (Sub-agents) 与“ Maker-Checker ”分离
**这是 Loop 能够摆脱人工作业的关键所在。** 在人类开发中，我们绝不会允许一个人编写了所有代码，然后自己做 Code Review 批准并发布。LLM 同样如此——写代码的 Agent（Maker）往往会对自己写出的逻辑产生严重的“认知盲区”，倾向于给出全是 OK 的报告。
Loop Engineering 要求我们在循环中实例化两个或多个 Sub-agents，例如：
* **Coder Sub-agent（Maker）**：追求速度，快速生成多份修复草案。
* **Reviewer Sub-agent（Checker）**：拥有最严格的安全与合规 Prompt，甚至使用 reasoning 努力程度更高的模型，对 Maker 生成的差异（Diff）进行挑剔式的安全审查。

---

### 第六要素：全局记忆外部化 (State)
在传统的 LLM 会话中，记忆全靠 context 窗口，但随着循环次数增加，context 会急剧膨胀，并且一旦会话结束或超出限额，记忆就会丢失。
Loop Engineering 倡导**“记忆必须留在磁盘/外部数据库，而不是大模型脑子里”**。我们会专门指定一个在仓库中跟踪的 Markdown 文件（例如 `AGENTS.md`）或连接一个 Linear 数据库。当循环迭代时，它会将每次尝试过的动作、失败的编译器日志、已经通过的单元测试项实时写入该文件。
即使在第 15 次循环时因为网络中断发生重启，Agent 只要重新读取 `AGENTS.md`，就能瞬间恢复“我已经做过什么，接下来需要尝试什么”的全局认知，Repo 本身成了唯一的真相源（Single Source of Truth）。

---

# 4. 一个典型的 Agent 循环运行图解 (The Loop Architecture)

在实际工程中，这五大原语与外部状态是如何联动工作的呢？我们可以看一看下面这个典型的**自动化 Bug 修复与验证循环**：

<div align="center">
<svg width="720" height="300" xmlns="http://www.w3.org/2000/svg" style="font-family:sans-serif">
  <!-- Background -->
  <rect width="720" height="300" rx="12" fill="#0d1117" stroke="#30363d" stroke-width="1.5"/>
  <!-- Title -->
  <text x="360" y="30" text-anchor="middle" font-size="15" font-weight="bold" fill="#f0f6fc">Loop Engineering 典型自主执行闭环流程</text>
  <!-- Step 1: Trigger -->
  <rect x="30" y="60" width="100" height="50" rx="8" fill="#21262d" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="80" y="85" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">1. 触发 &amp; 检索</text>
  <text x="80" y="98" text-anchor="middle" font-size="9" fill="#8b949e">Cron / 任务看板</text>
  <!-- Arrow to Step 2 -->
  <path d="M 130 85 L 170 85" stroke="#8b949e" stroke-width="1.5" fill="none"/>
  <polygon points="170,82 178,85 170,88" fill="#8b949e"/>
  <!-- Step 2: Worktree Isolation -->
  <rect x="180" y="60" width="110" height="50" rx="8" fill="#21262d" stroke="#38bdf8" stroke-width="1.5"/>
  <text x="235" y="85" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">2. 隔离工作区</text>
  <text x="235" y="98" text-anchor="middle" font-size="9" fill="#8b949e">git worktree 创建</text>
  <!-- Arrow to Step 3 -->
  <path d="M 290 85 L 330 85" stroke="#8b949e" stroke-width="1.5" fill="none"/>
  <polygon points="330,82 338,85 330,88" fill="#8b949e"/>
  <!-- Step 3: Maker Subagent -->
  <rect x="340" y="60" width="110" height="50" rx="8" fill="#21262d" stroke="#a855f7" stroke-width="1.5"/>
  <text x="395" y="85" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">3. 编码执行者</text>
  <text x="395" y="98" text-anchor="middle" font-size="9" fill="#c084fc">Subagent: Maker</text>
  <!-- Arrow to Step 4 -->
  <path d="M 450 85 L 490 85" stroke="#8b949e" stroke-width="1.5" fill="none"/>
  <polygon points="490,82 498,85 490,88" fill="#8b949e"/>
  <!-- Step 4: Checker Subagent -->
  <rect x="500" y="60" width="110" height="50" rx="8" fill="#21262d" stroke="#10b981" stroke-width="1.5"/>
  <text x="555" y="85" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">4. 自动化判定</text>
  <text x="555" y="98" text-anchor="middle" font-size="9" fill="#34d399">Subagent: Checker</text>
  <!-- Arrow to Branch -->
  <path d="M 610 85 L 635 85 L 635 140 L 415 140 L 415 180" stroke="#8b949e" stroke-width="1.5" fill="none"/>
  <polygon points="412,180 415,188 418,180" fill="#8b949e"/>
  <text x="525" y="135" text-anchor="middle" font-size="10" fill="#8b949e">通过校验？</text>
  <!-- Feedback / Loop back path (if Fail) -->
  <path d="M 360 215 L 235 215 L 235 140 L 395 140 L 395 120" stroke="#ef4444" stroke-width="1.5" fill="none" stroke-dasharray="4"/>
  <polygon points="392,120 395,112 398,120" fill="#ef4444"/>
  <text x="280" y="230" text-anchor="middle" font-size="10" fill="#ef4444">未通过 ➔ 报错反馈与重试</text>
  <!-- Step 5: Stop Condition & PR (if Pass) -->
  <rect x="360" y="190" width="110" height="50" rx="8" fill="#1f2937" stroke="#10b981" stroke-width="2"/>
  <text x="415" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">5. 停止与交付</text>
  <text x="415" y="228" text-anchor="middle" font-size="9" fill="#34d399">合并代码 / 开 PR</text>
  <!-- Arrow to Step 6 -->
  <path d="M 470 215 L 510 215" stroke="#8b949e" stroke-width="1.5" fill="none"/>
  <polygon points="510,212 518,215 510,218" fill="#8b949e"/>
  <!-- Step 6: Memory State Update -->
  <rect x="520" y="190" width="110" height="50" rx="8" fill="#21262d" stroke="#f43f5e" stroke-width="1.5"/>
  <text x="575" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#f0f6fc">6. 更新全局状态</text>
  <text x="575" y="228" text-anchor="middle" font-size="9" fill="#fb7185">AGENTS.md / Linear</text>
</svg>
<figcaption>图 4：Loop Engineering 典型自主执行闭环流程图</figcaption>
</div>

1. **触发 &amp; 检索 (Trigger &amp; Retrieval)**：事件监听发现有未结任务，触发 Loop。系统拉取对应上下文，并挂载 `SKILL.md`，使 Agent 系统掌握项目规范底盘。
2. **工作区隔离 (Worktree Isolation)**：在本地生成临时的 `git worktree` 与新分支，以保护生产主分支代码。
3. **编码执行 (Execution)**：Maker 子代理开始基于 Bug 报错和现有代码生成修复方案。
4. **自动化判定 (Verification)**：运行测试套件和 Linter，Checker 子代理对修改的代码逻辑及测试通过率进行综合质检：
   * **未通过（Fail）**：把执行错误、断言失败等 Traceback 信息写回 `AGENTS.md`，重新反馈给 Maker 子代理，进入自我纠错的分支闭环。
   * **已通过（Pass）**：在本地完成交付提交，打破当前的迭代，迈向停止条件。
5. **停止与交付 (Termination &amp; PR)**：自动清除本地隔离工作区，向 GitHub 仓库发起包含详细修复说明的 PR。
6. **状态同步 (State Update)**：同步标记外置记忆看板，并在团队 Slack 频道留言，提示人类工程师进入 Review。

---

# 5. 专业 Agent 工程师的设计守则与挑战

尽管 Loop Engineering 看起来无比美妙，但作为一个设计过多套工业级闭环系统的 Agent 工程师，我必须客观地泼一些冷水。当系统从单步交互进化到完全自治的 Loop 时，以下三个挑战将变得极其尖锐：

### 1. 验证盲区 (Verification Deficit) —— 不要把“声明”当成“证词”
在闭环中，最难设计的永远是**停止规则（Stop Condition）**。LLM 对“我已经完成了任务”的自信度与代码的实际质量通常是完全脱钩的。
如果我们仅靠 Agent 自己的一句“我觉得改好了”来作为循环终止条件，这个系统大概率会以各种诡异的手段蒙混过关。
**工程法则**：停止条件必须依托**具备强确定性的工具反馈**，而非大模型的文本陈述。你必须强制闭环对准硬性考核指标，例如：“必须通过 `npm run test:unit`，同时 ESLint 报错数为 0，且新生成的代码测试覆盖率不低于 80%”。

### 2. 理解 rot 与认知 surrender (Cognitive Surrender) —— 警惕“理解债”的累积
当闭环系统运行得足够流畅，每天早上你都能收到 10 个自动生成的 PR 时，人性的弱点就会暴露出来——你变得越来越倾向于无条件相信机器。
你不再细致审查每一行 Diff，而是简单地点下 Merge 键。我将这种现象称为**“认知投降” (Cognitive Surrender)**。
这会带来极高昂的**“理解债” (Comprehension Debt)**。软件工程是变动的，当你在一个月后需要重构某个核心模块时，你会突然绝望地发现，系统中有一大半核心逻辑都是你从未深入看过的，而这时候写代码的那些 Agent 早就丢弃了当初的上下文。
**工程法则**：即使 Loop 再完美，人类依然是质量的最后一道闸门。你设计 Loop 的初衷是为了节约繁琐的打字和调试时间，好让你腾出精力进行高阶的架构决策和深度 Code Review。

### 3. Token 成本控制 (Token Governance) —— 规避循环内的“自循环死亡螺旋”
当 Maker 子代理写出的代码有 Bug，Checker 反馈了报错，Maker 在尝试修复时又引入了新的 Bug……如果你的纠错逻辑 Prompt 写得不够稳健，系统极易陷入**死循环陷阱（Death Spiral）**。
在这个螺旋里，Agent 在几分钟内就能刷掉上千万的 Token，产生出长达数百个回合的无效尝试，给公司账单带来灾难。
**工程法则**：每一个 Loop 都必须设计“最高预算护栏”：
* 每次循环的最大执行尝试（如限制 `MaxAttempts = 5`）。
* 会话 Token 预算软、硬护栏（超出指定额度必须向人类请求人工介入挂起）。
* 对重试逻辑做退避衰减分析，当第 3 次重试失败信息与之前相同时，主动中断并抛出异常。

---

# 6. 结语：构建循环，保持思考

Loop Engineering 不仅是技术上的变革，更是软件工程研发范式的深刻变迁。它让我们认识到，在 LLM 时代：
* 优秀的程序员擅长**编写高质量的代码**。
* 优秀的 AI 辅助程序员擅长**撰写高质量的 Prompt**。
* 而下一代的 **Agent 工程师，则擅长设计让 Prompt 与代码自我运转、自我纠错并不断收敛的系统循环**。

但在这个机器可以跑在最前面不知疲倦迭代时，最不可替代的，依然是站在循环终点、冷眼旁观并进行终极把关的——**你自己的审慎思维**。

构建你的循环，但不要沦为循环的看客。保持你作为工程师的灵魂。
