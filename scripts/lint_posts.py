#!/usr/bin/env python3
"""Lint Jekyll posts against the rules in CLAUDE.md.

Errors (exit 1):
  front-matter   missing layout/title/date
  curly-quote    non-ASCII quote used as an HTML attribute delimiter
  div-blank-line blank line inside <div align="center"> ... </div>
  math-underscore single-$ inline math containing `}_` (must be $$...$$)
  missing-image  local /images/... reference that does not exist
Warnings (exit 0 unless --strict):
  gif            GIF referenced; use mp4 via <video>
  big-image      referenced image over 300KB

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
INLINE_CODE = re.compile(r"`[^`\n]*`")
# kramdown treats `_` after a closing brace as possible emphasis (even before an
# alphanumeric), so two of them in one paragraph swallow the text between them.
BRACE_UNDERSCORE = re.compile(r"[})\]]_")
IMG_REF = re.compile(r'src="(/images/[^"]+)"|\]\((/images/[^)\s]+)\)')


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

    text = mask_code(raw)
    body = text[body_start:]
    off = body_start

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
