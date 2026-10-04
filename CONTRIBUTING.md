# 贡献指南

感谢你愿意为这个研究笔记站点提供反馈。本站的内容以中文长篇综述和论文精读为主，
最有价值的贡献往往不是代码，而是**指出事实错误、补充遗漏的重要工作、修正技术表述**。

本文档说明三件事：反馈怎么提、内容怎么改、代码怎么改。

---

## 一、内容反馈（最常见，也最欢迎）

### 提交入口

两个入口，效果相同：

1. 文章末尾的反馈组件 —— 会带上文章标题自动生成 Issue 标题，推荐使用
2. 直接 [新建 Issue](https://github.com/TingdeLiu/tingdeliu.github.io/issues/new)

### 三类反馈

| 类型 | 适用情形 | 请附上 |
| --- | --- | --- |
| `[错误反馈]` | 事实错误、数字对不上、公式或代码写错、排版失效 | 文章链接 + 章节锚点 + 你认为正确的版本及依据 |
| `[论文推荐]` | 某个方向漏掉了重要工作 | 论文标题 + arXiv/DOI 链接 + 一两句话说明为什么值得收录 |
| `[修改建议]` | 表述不清、结构可以更好、术语翻译不准 | 具体段落 + 建议改法 |

### 对证据的要求

这是本站唯一的硬性要求：**涉及事实和数字的反馈，请附上可核验的来源**——原始论文、
官方文档、可复现的实验记录，或指明是哪个版本的哪一页。

原因很直接：本站文章大量引用论文中的性能数字和方法细节，作者会逐条回到原文核对
后才修改。没有来源的反馈不是不受欢迎，只是核实成本会高很多，处理会慢。

如果你自己也不确定，明说"我不确定，但感觉这里和论文 Table 3 对不上"同样有价值，
比不提好。

### 会得到什么回应

- 通常 **3 个工作日内**首次回复，说明是否采纳以及核实结论
- 采纳的修改会直接提交，并在 Issue 中附上对应 commit 链接
- 不采纳的会说明理由，而不是直接关闭

历史处理记录见 [MAINTAINERS.md](MAINTAINERS.md)。

---

## 二、直接提交内容修改（PR）

小的错字、链接失效、格式问题，欢迎直接提 PR，不必先开 Issue。

**较大的内容改动（新增一节、重写某个方法的解读、新增一篇论文精读）请先开 Issue 讨论。**
这类文章有既定的组织逻辑和篇幅预算，直接提交大段内容大概率需要返工。

### 内容约定

文章源文件在 `_posts/` 下，按用途分三个目录：

```
_posts/research/        # 长篇综述与论文精读   →  categories: research
_posts/blog/            # 专题解析与工程文章   →  categories: blog
_posts/weekly-reports/  # 具身导航周报         →  categories: weekly
```

注意：**目录只用于组织源文件，聚合页分区的唯一依据是 front matter 里的
`categories` 字段**，两者必须对应。永久链接固定为 `/:title/`。

#### 硬性规则：改了文章就要更新日期

> 修改 `_posts/` 下任何已有文章后，必须把 front matter 的 `date:` 更新为当天日期。

站点列表按 `date` 倒序排列，这是"最近更新"排序能生效的前提。忘记更新会导致修改过的
文章沉在列表底部。

#### 图片

- 放在 `images/<主题>/` 下，主题目录沿用现有划分：`vln` / `vla` / `vlm` / `wm` /
  `si` / `agent` / `llm-training` / `robotics_navigation` 等
- 命名用 `<论文或方法名>-<内容>.webp`，例如 `OccPlanner-architecture.webp`
- **位图**（论文截图、架构图、实验结果图）统一转为**无损 WebP** 后提交，不要提交原始 PNG/JPG
- **矢量示意图保留 SVG**、**算法演示动画保留 GIF**——这两类转 WebP 只会掉质量或丢失动画，不要转
- **来源论文必须在正文中标注**

> ⚠️ **关于第三方图片的重要提醒**
>
> 不要提交你没有权利提交的图片。如果图片取自论文，请在 PR 描述中写明出处论文和
> 该论文的许可条款（arXiv 页面左下角会标注）。本仓库以学术评述和教学为目的引用
> 论文插图，这些图片的著作权属于原作者，**不在本站 CC BY 4.0 授权范围内**，
> 详见 [LICENSE-CONTENT](LICENSE-CONTENT) 第 4 节。

#### 写法约定

- Markdown 解析器是 kramdown（GFM 模式）
- 公式用 MathJax 3 语法
- 图表用 Mermaid 10，**仅在页面含图表时才会加载 CDN**
- Mermaid 节点文字里避免出现 `{{` 和 `}}`，Jekyll 的 Liquid 模板会先把它吃掉，
  本地能 parse 不代表线上不报错；需要花括号时用 HTML 实体或改写表述

---

## 三、代码修改

站点代码（`_layouts/` `_includes/` `_sass/` `js/` `.github/workflows/`）采用
MIT License，欢迎改进。

### 本地运行

环境要求：Ruby 3.2（与 CI 一致）+ Bundler。

```bash
bundle install
bundle exec jekyll serve
```

默认在 <http://127.0.0.1:4000/>。Windows PowerShell 下用 `ruby -S bundle exec jekyll serve`。

生产构建（提 PR 前请至少跑一次，确认不报错）：

```bash
JEKYLL_ENV=production bundle exec jekyll build
```

### 提交前自查

- [ ] `JEKYLL_ENV=production bundle exec jekyll build` 能通过，无 Liquid / kramdown 报错
- [ ] 改动涉及的页面在本地浏览器里实际打开看过
- [ ] 改了已有文章的话，front matter 的 `date:` 已更新为当天
- [ ] 新增图片是 WebP，且已标注来源
- [ ] 没有提交 `_site/`、`.jekyll-cache/` 等构建产物

### Commit message

沿用 [Conventional Commits](https://www.conventionalcommits.org/)，本仓库常用类型：

```
docs(research): add OccPlanner deep-dive to VLN Papers Extended
feat(vln):      add interactive tag filter to paper list
style(research): re-layout Section 5 Mermaid diagram
perf:           lazy-load VLN paper images
ci:             bump Pages actions to Node 24 runtimes
fix:            correct broken anchor in VLA survey
```

scope 用方向名（`vln` / `vla` / `research` / `weekly`）或留空。

### PR 说明请写清楚

- 改了什么、为什么改
- 内容类改动：依据的论文或文档链接
- 代码类改动：本地验证方式，必要时附截图

---

## 四、行为准则

就一条：**对事不对人，把论据放在结论前面。**

这是一个技术笔记仓库，讨论会集中在"这个数字对不对""这个方法归类是否合适"这类
问题上。指出错误越直接越好，但请针对内容本身。

对人身攻击、与内容无关的推广、以及明显未经核实的批量提交，维护者会直接关闭
且不再回应。

---

## 五、许可

提交贡献即表示你同意：

- 对**代码**的贡献以 [MIT License](LICENSE) 授权
- 对**原创文字内容**的贡献以 [CC BY 4.0](LICENSE-CONTENT) 授权
- 你有权提交这些内容，且未侵犯第三方著作权

---

有任何流程上的疑问，直接开 Issue 问即可。
