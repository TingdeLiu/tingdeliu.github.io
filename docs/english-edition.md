# English edition: scope and maintenance

The English edition contains the complete VLN Survey and VLN Papers: Instruction
Following (all 55 readings), plus `/en/` and `/en/research/`. The extensions
collection and other surveys remain Chinese-only until their translations are
complete. The instruction-following paper filters and leaderboards are localized.

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
│   └── VLN-Papers.md
└── blog/
    └── .gitkeep
```

Research surveys and paper readings belong in `research/`; technical blog posts
belong in `blog/`. The future `VLN-Papers-Extended.md` translation belongs in
`research/` when complete. Set `categories` to match the content type
and declare an explicit `/en/.../` permalink so moving files does not change URLs.
When moving an existing translation, update its `translation` path in the
snapshot under `translations/`; preserve the reviewed source hashes and dates.

English articles render through the shared post layout.
They do not become extra Chinese posts, feed entries, or tag/archive
results. English entry pages live in `en/`. UI strings live in `_data/ui.yml`;
JavaScript interaction labels are in `js/article-ui.js`.

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
All 55 paper wrappers, AND tag filters, Chinese-only extended-collection links,
three leaderboards, best-value bolding, contents links, figure enlargement, and
code copying were checked. No page errors or formula/diagram errors occurred.

MapNav's R2R OS/SR headers were corrected in both editions against
[arXiv v5, Table 1](https://arxiv.org/html/2502.13451v5#S4.T1). DualVLN's
illustrative code now preserves query gradients through a frozen VLM, matching
[Appendix A.2](https://arxiv.org/html/2512.08186v1#A2), and its real-world RGB-D
and odometry pipeline is distinguished from RGB-only simulation.

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
- [x] English entry pages and author introduction
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
