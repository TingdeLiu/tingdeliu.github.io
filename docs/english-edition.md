# English edition: scope and maintenance

The English edition contains the complete VLN Survey, VLN Papers: Instruction
Following (55 readings), and VLN Papers: Goal Navigation and Extensions
(34 main readings and 3 related readings, 37 total), and all eight existing
embodied-navigation weekly digests (six published and two unpublished), plus `/en/`, `/en/research/`, and
`/en/blog/`. Both paper collections have localized filters, leaderboards,
and reciprocal English companion links. The Traditional Robot Navigation
Algorithms survey also has a complete English edition. The three agent articles
(AI Agents, Embodied Agent Harness and Runtime Architecture, and Embodied Agent
Paper Readings) are translated in full. Other surveys remain Chinese-only.

## Content contract

- Chinese is the source edition. Translate the complete article; preserve its
  conclusions, caveats, formulas, numbers, citations, and reference numbering.
- Use natural research English. Distinguish paper findings from the author's
  interpretation and proposed experiments. Do not strengthen claims.
- A translation date does not change the source verification date.
- Reuse original English paper figures; translate captions, alternative text,
  Mermaid labels, and explanatory graphics made for this site.
- Preserve source heading IDs and explicit anchors in the English edition so
  shared links and language switching continue to reach the same section.
  The initial translation retains existing Chinese IDs for compatibility.

## Files and metadata

English articles are grouped by content type under `_translations/en/`:

```text
_translations/en/
├── research/
│   ├── VLN-Survey.md
│   ├── VLN-Papers.md
│   ├── VLN-Papers-Extended.md
│   ├── Robot-Navigation-Survey.md
│   ├── AI-Agent-Survey.md
│   ├── Embodied-Agent-Harness-Survey.md
│   └── Embodied-Agent-Papers.md
└── blog/
    ├── vln-weekly-2026-08-01.md
    └── ... (eight weekly digests through 2026-09-27)
```

Research surveys and paper readings belong in `research/`; technical blog posts
and weekly digests belong in `blog/`. Digests retain `categories: weekly`.
Set `categories` to match the content type
and declare an explicit `/en/.../` permalink so moving files does not change URLs.
When moving an existing translation, update its `translation` path in the
snapshot under `translations/`; preserve the reviewed source hashes and dates.

English articles render through the shared post layout.
They do not become extra Chinese posts, feed entries, or tag/archive
results. English entry pages live in `en/`. UI strings live in `_data/ui.yml`;
JavaScript interaction labels are in `js/article-ui.js`.

The English Research index shows article cards without an extra reading-path,
companion-link, or author section. Its sidebar contains Project, Research, Blog,
and About; Project links to the existing `/home/` page, Blog links to `/en/blog/`,
and About links to the existing `/about/` page.
Language switching stays in the page's language selector.

Each article declares `lang`, `translation_id`, `permalink`, `source_path`,
`source_url`, `source_revision_date`, and `translation_updated`. Its source has
the same `translation_id`. The same ID pairs equivalent index pages, too.
Do not place incomplete drafts in the published translations collection.

Use `content-link.html` for companion article links. It selects the translated
target by `source_url` when available and otherwise labels the Chinese link.
New translations must preserve any fragments used by existing references.
Only genuine equivalent pages receive reciprocal `hreflang` annotations;
each language has its own canonical URL. URL language determines rendering.

## Update workflow

### Blog and weekly digests

The English Blog lists the six published weekly digests in reverse publication order,
with explicit `issue_number`, `period_start`, and `period_end` metadata. Each
digest preserves the original publication date; `source_revision_date` reflects
the last source commit before translation, not the translation date. The latest
issue includes the source's 2026-10-01 correction and links to the English
GPT-6-Astra paper reading. The two early source drafts (2026-08-01 and
2026-08-15) also have complete English translations but retain `published: false`.

Public source prose used machine translation assistance followed by terminology,
conclusion, limitation, numerical, and Markdown review. Links and numeric values
were protected during translation. Each issue has a synchronization snapshot
under `translations/vln-weekly-*.en.json`; the shared site validator checks the
complete issue set, heading IDs, source links, numerical values, language pairs,
card order, and exclusion from Chinese feeds and Research cards. Technical essays
remain untranslated and are not shown as English articles.

Desktop validation at 1440×1000 passed for the six published issues: card order,
article contents links, section-preserving language switching, Blog navigation,
and exclusion from the three-card Research collection. No horizontal overflow,
unrendered emphasis, or page errors occurred. The built-site validator reports
12 English pages with zero errors. Both unpublished translations also passed
rendered source-link and heading-ID checks (19 matching headings each) and are
absent from the production output. All 16 source/translation weekly files pass
the post linter, and the 11 existing regression tests pass.

### VLN Papers completion and draft workflow

`_translations/en/research/VLN-Papers.md` contains the complete 55-paper
instruction-following collection, including leaderboards, the component matrix,
analysis, references, captions, and interactive controls. The four earlier
reviewed readings were retained. Remaining prose used machine assistance,
followed by terminology, negation, numerical, markup, and targeted technical
review. This is a translation of the source article, not a new independent
verification of every underlying paper.

`translations/vln-papers.en.progress.json` records completion and validation.
`translations/vln-papers.en.json` is the authoritative synchronization snapshot.
Repeated section labels are scoped by each paper's stable anchor so additions
elsewhere do not renumber their keys. Source checks cover per-paper equations,
figures, external citations, and table values; the opening leaderboards and
comparison tables are included in numerical checks too.

Desktop is the primary reading and validation target. At 1440×1000, the complete
page has 524 matching heading IDs, 52 tables, 233 captioned figures, 23 code
blocks, 1,243 rendered mathematical expressions, and 19 Mermaid diagrams.
All 55 paper wrappers, AND tag filters, companion-collection links,
three leaderboards, best-value bolding, contents links, figure enlargement, and
code copying were checked. No page errors or formula/diagram errors occurred.

MapNav's R2R OS/SR headers were corrected in both editions against
[arXiv v5, Table 1](https://arxiv.org/html/2502.13451v5#S4.T1). DualVLN's
illustrative code now preserves query gradients through a frozen VLM, matching
[Appendix A.2](https://arxiv.org/html/2512.08186v1#A2), and its real-world RGB-D
and odometry pipeline is distinguished from RGB-only simulation.

### Goal-navigation and extensions completion

`_translations/en/research/VLN-Papers-Extended.md` contains all 37 readings,
five goal-navigation leaderboard groups, related-reading notes, references,
134 figures with English captions and alternative text, and 17 localized
Mermaid diagrams. The full page has 37 tables. Harness Robotic OS links to the
companion Embodied Agent Paper Readings article.

Prose used machine translation assistance followed by terminology, numerical,
negation, protected-element, and editorial review. The numerical audit restored
a missing ReMEmbR accuracy of 0.61; the negation review corrected SparseNav's
warning that more complete semantic maps are not always better. Original
leaderboard rows were retained with explicit label translations to prevent
omitted textual cells or inconsistent filter vocabulary. This process does not
constitute independent verification of every underlying paper.

Descriptive TeX labels, units, and reward predicates are translated using an
explicit normalization map. Formula checks still detect changed mathematical
values and predicates. Raw table column counts supplement numerical checks.
Two malformed figure wrappers in the Chinese source were repaired in both
editions; paper content and reported experimental values were retained.

`translations/vln-papers-extended.en.json` records source synchronization, and
`translations/vln-papers-extended.en.progress.json` records completion and the
review process. The published-site validator checks both paper collections.
Main/extended filters search 92 papers in total and link directly to the
corresponding English collection. Research cards and Survey companion links
also resolve to the English extension.

Desktop verification at 1440×1000 passed: all 37 paper wrappers, 53 rows across
five leaderboards, 273 matching heading IDs, 37 tables, 134 images, 762 MathJax
expressions, and 17 Mermaid diagrams. AND tag filtering, open-source/paradigm
filters, nonstandard-row hiding, figure enlargement, contents navigation,
section-preserving language switching, English cross-page search results, and
the research card were checked. No page, formula, diagram, or horizontal
overflow errors occurred. The Jekyll build, synchronized-source checks, post
linter, published-site validator, and 11 regression tests pass.

### Traditional robot navigation

`_translations/en/research/Robot-Navigation-Survey.md` translates all 12 main
sections, from perception and localization through mapping, planning, tracking,
kinematics, motion control, and ROS stack integration. Its 164 heading IDs match
the Chinese edition, preserving section links and language switching. It retains
62 tables, 46 images, and 33 demonstration videos. Table values, external
citations, and mathematical expressions are checked against the source; only
explicit descriptive TeX labels and direction subscripts are normalized.

Public prose used translation assistance followed by terminology, markup,
numerical, and targeted technical review. This translates the source's claims
and does not independently reverify every algorithm comparison or ROS default.
The English edition has a new SVG navigation overview and 13 localized vector
diagrams, plus English captions and alternative text. Existing raster figures
and demonstration videos are retained; embedded text in those original media
may remain Chinese. Source synchronization is recorded in
`translations/robot-navigation-survey.en.json`.

The source tracker supports repeated survey subheadings by scoping subsequent
occurrences to their enclosing section hierarchy. Existing snapshot keys and
paper-anchor behavior are retained.

Verified on 2026-10-04: Jekyll build, built-site validation (13 English pages,
zero errors), post lint (zero errors), and all 12 regression tests pass.
At 1440×1000, all 18 Mermaid diagrams and 407 MathJax expressions render without
errors. Contents navigation, reciprocal section-preserving language switching,
and the English Research card pass. The 390×844 layout has no horizontal page
overflow; long equations scroll within their own containers. The new article's
source snapshot is synchronized. The unrelated VLN Papers snapshot currently
needs updates, so the repository-wide strict synchronization command remains
nonzero until that collection is reviewed.

### Agent surveys and paper readings

The English edition includes the complete AI Agent survey, the complete
Embodied Agent Harness survey, and all eight Embodied Agent paper readings.
Chinese publication dates, original section IDs, equations, paper citations,
experimental values, and code interfaces are retained. English Research cards,
reciprocal language switching, companion links, and the paper collection's AND
tag filter use the shared site infrastructure.

Public prose used machine translation assistance with terminology, numerical,
negation, and markup review. Chinese numerical units require explicit
conversion (for example, 100 万 tokens means one million tokens). The English
overview figures are local SVGs; original paper figures are reused with English
captions and alternative text. This translation preserves the source's claims
and evidence boundaries; it is not a fresh verification of every paper,
product release, benchmark, or library version.

Source revisions are 2026-10-02 for the two surveys and 2026-10-04 for the
paper collection; translation updates are dated 2026-10-04. Synchronization
snapshots are stored under `translations/ai-agent-survey.en.json`,
`translations/embodied-agent-harness-survey.en.json`, and
`translations/embodied-agent-papers.en.json`. The release validator checks
heading IDs, table dimensions and values, equations, external citations,
Research cards, language pairs, and exclusion from the Chinese feed.

Verified on 2026-10-04: all 355 section IDs match their Chinese sources, with
95 tables, 60 figures, 63 rendered Mermaid diagrams, and 175 MathJax
expressions. The build and built-site validator pass (16 English pages, zero
errors), as do all 14 regression tests and lint for the six agent files.
At 1440×1000 and 390×844, reciprocal section switching, all eight paper
wrappers, AND filtering and reset, and the three Research cards pass. There
are no page, formula, diagram, raw-emphasis, or horizontal-overflow errors.
Long inline equations scroll locally in the English edition. The unrelated
VLN Papers source snapshot remains stale and still requires its own review.

### Future drafts

Future incomplete drafts use `published: false` and `translation_scope: partial`.
They are excluded from normal output, language pairs, cards, and translated
companion links. Record section-level progress without claiming a reviewed
whole-source snapshot. Validate with `python scripts/check_translation_drafts.py
--strict`. To review them locally:

```text
bundle exec jekyll build --unpublished --destination _site-drafts
python -m http.server 4174 --bind 127.0.0.1 --directory _site-drafts
```

Never upload `_site-drafts`. When a translation is complete, remove the draft
metadata, pair its Chinese source, and record its synchronized source snapshot.

### Published translations

1. Update the Chinese source as usual.
2. Run `python scripts/check_translations.py` to see changed, added, or removed
   sections. This generates `_data/translation_status.json` for the build.
3. Translate the changed sections and review surrounding context, tables,
   equations, diagrams, links, and terminology. Retain reviewed English prose
   elsewhere instead of regenerating the whole document.
4. Update `source_revision_date` and `translation_updated` in the English file.
5. After reviewing, run
   `python scripts/check_translations.py --record vln-survey --date YYYY-MM-DD`.
   Commit the resulting `translations/vln-survey.en.json` with the translation.
   Recording accepts the current source as reviewed; it does not translate it.
6. Run the checks below and inspect desktop rendering; mobile is a secondary check.

Snapshots contain per-section SHA-256 hashes, including source metadata, and
report additions/removals as well as modifications. They never modify English
content automatically. CI warns on stale translations and renders a visible
notice without preventing publication of Chinese updates. Use `--strict` to
require synchronization for an English release. Missing or mismatched records
are build errors. A plain Jekyll build without the preprocessing step shows
an explicit unchecked-status message instead of claiming synchronization.
While stale or unchecked, language switching opens the counterpart at its top
because section IDs may have changed. Structural parity checks run only when
synchronized; standalone English page and link checks always run.

## Validation

```text
python scripts/check_translations.py --strict
python scripts/lint_posts.py
python -m unittest discover -s scripts/tests
bundle exec jekyll build
python scripts/check_english_site.py
```

Also inspect language switching with section fragments, mobile contents,
code and link copying, image enlargement, horizontal tables, English feedback
forms, and Chinese article regression. Confirm that formulas, reference URLs,
and all 10 main sections are preserved. All prose and diagram labels should be
English except language selectors, legacy IDs, and explicitly marked links.

## Terminology

| Chinese | English | Usage |
| --- | --- | --- |
| 视觉语言导航 | vision-language navigation (VLN) | Expand at first use |
| 指令跟随 | instruction following | Distinguish from goal search |
| 语言落地 | language grounding | Alignment to observations/actions |
| 目标指代 | object grounding / referring expression | Match the task context |
| 连续环境 | continuous environment | Does not imply continuous actions |
| 快慢双系统 | fast-slow dual system | Distinguish control hierarchy from context updates |
| 路点 | waypoint | Preserve action-interface meaning |
| 路径忠实度／保真度 | path fidelity | Separate from endpoint success |
| 免训练 | training-free | Preserve the source's qualification |
| 具身形态 | embodiment / morphology | Select according to context |
| 回溯 | backtracking | Spatial recovery, not model backpropagation |
| 卡住 | immobilization / getting stuck | Physical execution failure |
| 测地距离 | geodesic distance | Do not replace with Euclidean distance |
| 真机 | real robot | Separate from physics-based simulation |

## First-release checklist

- [x] Full survey translation with original structure, equations, and references
- [x] English entry pages and links to the existing Blog and About pages
- [x] Shared language switching, localized reading controls, and feedback
- [x] Explicit labels for Chinese-only companion articles
- [x] Per-section source snapshots and visible stale notices
- [x] Build, structural checks, and desktop/mobile verification

Verified on 2026-10-03: Jekyll build; 117 matching headings, 32 tables, formulas
and citation links; four synchronization tests; desktop and 390 px mobile
Chrome interactions; all 13 Mermaid diagrams and 45 MathJax expressions render.
A simulated stale build with an extra Chinese heading also passes while
displaying its warning and switching languages without a section fragment.
The post linter reports no errors and 26 existing image-size warnings elsewhere.
