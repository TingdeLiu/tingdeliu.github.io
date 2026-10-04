#!/usr/bin/env python3
r"""Lint Jekyll posts against the rules in CLAUDE.md.

Errors (exit 1):
  front-matter   missing layout/title/date
  curly-quote    non-ASCII quote used as an HTML attribute delimiter
  div-blank-line blank line inside <div align="center"> ... </div>
  math-underscore single-$ inline math containing `}_` (must be $$...$$)
  missing-image  local /images/... reference that does not exist
  control-char   control character (BEL, backspace, lone CR, ...) or a TAB glued to a LaTeX
                 command tail on a math line: a `\t`/`\a`/`\b`/`\r` command was pasted as the
                 control character and the formula is broken
  split-math-range  a range split across adjacent formulas can wrap between endpoints
  currency-range   two dollar currency amounts can be misinterpreted as math delimiters
Warnings (exit 0 unless --strict):
  gif            GIF referenced; use mp4 via <video>
  big-image      referenced image over 300KB
  mermaid-label  in a ```mermaid block, an unquoted edge label containing ( ) [ ] { } (write -->|"a(b)"| C),
                 or the invalid thick-link form ==="text"===>. Mermaid then fails to parse the whole diagram
                 and the page silently shows no diagram
  pipe-in-math   a bare `|` inside a formula on a normal text line: kramdown parses any line with a
                 pipe as a table row and splits the formula across cells. Write `\mid` for conditional
                 bars and `\lvert x \rvert` for absolute values (whole-line $$...$$ blocks and real
                 table rows are not checked)

Usage: python scripts/lint_posts.py [--strict] [--fix] [files...]   (default: _posts/**/*.md)
  --fix  rewrite math-underscore spans as $$...$$ (the only auto-fixable rule)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_IMAGE_BYTES = 300_000
CURLY = "“”‘’"

FENCE = re.compile(r"^\s*(```|~~~)")
TAG_CURLY = re.compile(r"<[A-Za-z][^<>]*?\s[\w:-]+\s*=\s*[" + CURLY + r"]")
DIV_OPEN = re.compile(r'<div\s+align="center"\s*>')
DIV_BLOCK = re.compile(r'<div\s+align="center"\s*>(.*?)</div>', re.S)
MATH_BLOCK = re.compile(r"\$\$.*?\$\$", re.S)
BS = "\\"  # backslash, so `\$` (escaped dollar) is not treated as math
MATH_INLINE = re.compile(r"(?<![$" + re.escape(BS) + r"])\$(?!\$)([^$\n]+?)(?<!" + re.escape(BS) + r")\$(?!\$)")
RANGE_MATH = re.compile(r"(?<![$\\])(?P<delimiter>\${1,2})(?P<expression>[^$\n]+?)(?P=delimiter)(?!\$)")
RANGE_JOIN = re.compile(r"[ \t]*(?:[-–—~～]|至)[ \t]*")
CURRENCY_RANGE = re.compile(r"(?<![$\\])\$\d+(?:\.\d+)?[ \t]*[-–—~～][ \t]*\$\d+(?:\.\d+)?(?![\d$])")
INLINE_CODE = re.compile(r"`[^`\n]*`")
# kramdown treats `_` after a closing brace as possible emphasis (even before an
# alphanumeric), so two of them in one paragraph swallow the text between them.
BRACE_UNDERSCORE = re.compile(r"[})\]]_")
IMG_REF = re.compile(r'src="(/images/[^"]+)"|\]\((/images/[^)\s]+)\)')
# C0 control characters except TAB and LF. CRLF is normalised before linting, so any CR left over is a lone one.
CONTROL_CHAR = re.compile(r"[\x00-\x08\x0b-\x1f]")
# A TAB right before the tail of a command that starts with `t` (\text \theta \tau \times \tilde \top \tag \to
# \triangle \tfrac \tan), on a line that has math delimiters, after non-indent text: the backslash was lost.
TAB_IN_MATH = re.compile(
    r"^(?=[^\n]*(?:\$|" + re.escape(BS) + r"\(|" + re.escape(BS) + r"\[))[^\n]*?\S[ ]*\t(?:ext|heta|au|imes|ilde|op|ag|o|riangle|frac|an)\b",
    re.M)
# A formula span ($$..$$ or $..$) on one line, and a pipe that is not escaped.
MATH_SPAN = re.compile(r"\$\$.+?\$\$|" + MATH_INLINE.pattern)
BARE_PIPE = re.compile(r"(?<!" + re.escape(BS) + r")\|")
WHOLE_LINE_BLOCK = re.compile(r"^\s*\$\$.*\$\$\s*$")
MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)\n```", re.S)
# edge label between pipes right after an arrow, not starting with a quote, containing a bracket character
MERMAID_BAD_LABEL = re.compile(r"(?:-{2,}>?|={2,}>|-\.+->?)\|(?!\")([^|\n\"]*[()\[\]{}][^|\n\"]*)\|")
MERMAID_BAD_THICK = re.compile(r"={3,}\"")


def mask_code(text):
    """Blank out fenced code and inline code, keeping offsets and newlines."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(" " * len(line))
        elif in_fence:
            out.append(" " * len(line))
        else:
            out.append(INLINE_CODE.sub(lambda m: " " * len(m.group()), line))
    return "\n".join(out)


def lineno(text, pos):
    return text.count("\n", 0, pos) + 1


def lint(path):
    errs, warns = [], []
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    fm = re.match(r"---\n(.*?)\n---\n", raw, re.S)
    body_start = fm.end() if fm else 0
    if not fm:
        errs.append((1, "front-matter", "missing front matter"))
    else:
        for key in ("layout", "title", "date"):
            if not re.search(rf"^{key}:", fm.group(1), re.M):
                errs.append((1, "front-matter", f"missing `{key}`"))

    for m in CONTROL_CHAR.finditer(raw):
        errs.append((lineno(raw, m.start()), "control-char",
                     f"control character {m.group()!r}; a LaTeX command such as \\a, \\b or \\r was pasted as a control character"))

    text = mask_code(raw)
    body = text[body_start:]
    off = body_start

    pos = off
    for line in body.split("\n"):
        ranges = list(RANGE_MATH.finditer(line))
        for left, right in zip(ranges, ranges[1:]):
            if RANGE_JOIN.fullmatch(line[left.end():right.start()]):
                errs.append((lineno(text, pos + left.start()), "split-math-range",
                              "range endpoints are separate formulas; use one inline formula or non-wrapping text"))
        for match in CURRENCY_RANGE.finditer(line):
            errs.append((lineno(text, pos + match.start()), "currency-range",
                          "dollar currency range can be parsed as math; use non-wrapping USD text"))
        pos += len(line) + 1

    for m in TAB_IN_MATH.finditer(body):
        errs.append((lineno(text, off + m.start()), "control-char",
                     "TAB glued to a LaTeX command tail (e.g. `<TAB>ext{...}`); a `\\t...` command lost its backslash"))

    for m in TAG_CURLY.finditer(body):
        errs.append((lineno(text, off + m.start()), "curly-quote",
                     "curly quote as HTML attribute delimiter; use ASCII \""))

    for m in DIV_BLOCK.finditer(body):
        inner = m.group(1)
        if re.search(r"\n[ \t]*\n", inner):
            errs.append((lineno(text, off + m.start()), "div-blank-line",
                         'blank line inside <div align="center"> block'))
    for m in DIV_OPEN.finditer(body):
        if not DIV_BLOCK.match(body, m.start()):
            errs.append((lineno(text, off + m.start()), "div-blank-line",
                         '<div align="center"> never closed'))

    no_block = MATH_BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), body)
    for m in MATH_INLINE.finditer(no_block):
        hit = BRACE_UNDERSCORE.search(m.group(1))
        if hit:
            errs.append((lineno(text, off + m.start()), "math-underscore",
                         f"`${m.group(1)[:40]}...$` contains `}}_`; use $$...$$"))

    pos = off
    for line in body.split("\n"):
        ln = lineno(text, pos)
        pos += len(line) + 1
        if line.lstrip().startswith("|") or WHOLE_LINE_BLOCK.match(line):
            continue
        for m in MATH_SPAN.finditer(line):
            if BARE_PIPE.search(m.group()):
                warns.append((ln, "pipe-in-math",
                              f"`{m.group()[:40]}` has a bare `|`; kramdown splits this line into table cells "
                              "(use \\mid or \\lvert..\\rvert)"))
                break

    for block in MERMAID_BLOCK.finditer(raw):
        base = lineno(raw, block.start(1)) - 1
        for i, line in enumerate(block.group(1).split("\n")):
            bad = MERMAID_BAD_LABEL.search(line)
            if bad:
                warns.append((base + 1 + i, "mermaid-label",
                              f"unquoted edge label `|{bad.group(1)[:30]}|` has brackets; write |\"...\"| or the diagram will not render"))
            elif MERMAID_BAD_THICK.search(line):
                warns.append((base + 1 + i, "mermaid-label",
                              "invalid thick-link syntax ===\"text\"===>; use ==>|\"text\"| instead"))

    seen = set()
    for m in IMG_REF.finditer(body):
        ref = (m.group(1) or m.group(2)).split("?")[0].split("#")[0]
        ln = lineno(text, off + m.start())
        f = ROOT / ref.lstrip("/")
        if not f.exists():
            errs.append((ln, "missing-image", ref))
            continue
        if ref in seen:
            continue
        seen.add(ref)
        if f.suffix.lower() == ".gif":
            warns.append((ln, "gif", f"{ref} -> convert to mp4 <video>"))
        elif f.stat().st_size > MAX_IMAGE_BYTES:
            warns.append((ln, "big-image", f"{ref} ({f.stat().st_size // 1024}KB)"))
    return errs, warns


def fix_math(path):
    """Rewrite flagged single-$ spans as $$...$$ in place. Returns number fixed."""
    data = path.read_bytes().decode("utf-8")
    bom = "﻿" if data.startswith("﻿") else ""
    crlf = "\r\n" in data
    raw = data[len(bom):].replace("\r\n", "\n")
    fm = re.match(r"---\n(.*?)\n---\n", raw, re.S)
    start = fm.end() if fm else 0
    body = mask_code(raw)[start:]
    no_block = MATH_BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), body)
    spans = [m for m in MATH_INLINE.finditer(no_block) if BRACE_UNDERSCORE.search(m.group(1))]
    for m in reversed(spans):
        a, b = start + m.start(), start + m.end()
        raw = raw[:a] + "$" + raw[a:b] + "$" + raw[b:]
    if spans:
        out = raw.replace("\n", "\r\n") if crlf else raw
        path.write_bytes((bom + out).encode("utf-8"))
    return len(spans)



def main(argv):
    strict = "--strict" in argv
    fix = "--fix" in argv
    files = [Path(a) for a in argv if not a.startswith("--")]
    if not files:
        files = sorted((ROOT / "_posts").rglob("*.md")) + sorted((ROOT / "_translations").rglob("*.md"))
    n_err = n_warn = 0
    for p in files:
        if fix:
            n = fix_math(p)
            if n:
                print(f"{p}: fixed {n} math-underscore span(s)")
        errs, warns = lint(p)
        try:
            rel = p.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            rel = p.as_posix()
        for ln, rule, msg in errs:
            print(f"{rel}:{ln}: error [{rule}] {msg}")
        for ln, rule, msg in warns:
            print(f"{rel}:{ln}: warning [{rule}] {msg}")
        n_err += len(errs)
        n_warn += len(warns)
    print(f"\n{len(files)} files, {n_err} errors, {n_warn} warnings")
    return 1 if n_err or (strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
