#!/usr/bin/env python3
"""De-AI-ify markdown docs: remove em-dashes (—), en-dashes (–) and semicolons
from prose, while leaving every form of code/math untouched.

Regions that are NEVER modified:
  - Fenced code blocks delimited by ``` or ~~~ (any length)
  - Inline code spans delimited by backticks ` `` ``` ...
  - Block math delimited by $$ ... $$ (may span multiple lines)
  - Inline math delimited by $ ... $ on a single line
  - YAML front-matter at the very top of the file (--- ... ---)

Replacement rules (prose only):
  - en-dash  '–'  -> '-'                          (works for ranges and compounds)
  - em-dash  '—'  -> ': '   if the preceding token looks like a label
                            (heading line, list bullet, or closing backtick/bracket)
                  -> ', '   otherwise
  - ';\\n'        -> '.\\n' and capitalise the first letter of the next non-empty line
  - '; '         -> ', '
  - bare ';'     -> ','

Usage:
    python3 scripts/depunct_docs.py docs/ [docs2/ ...]

The script edits files in place. It prints a per-file change summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")
DIRECTIVE_INFO_RE = re.compile(r"^\s*\{([\w-]+)\}")

# MyST directives whose body is human-readable prose (not code). When we see
# one of these as a fenced directive, rewrite its body too — only the opening
# and closing fence markers stay verbatim.
PROSE_BODY_DIRECTIVES = {
    "admonition", "seealso", "note", "warning", "tip", "caution",
    "important", "attention", "danger", "error", "hint",
    "figure",  # caption text below options
    "tab-item", "tab-set",
    "grid", "grid-item", "grid-item-card",
}


def _is_prose_body_fence(info_string: str) -> bool:
    m = DIRECTIVE_INFO_RE.match(info_string)
    return bool(m and m.group(1).lower() in PROSE_BODY_DIRECTIVES)


def split_top_level(src: str) -> list[tuple[str, str]]:
    """Split a markdown file into a list of (kind, text) chunks at the
    line-block level. kind is one of: 'frontmatter', 'fence', 'prose'.

    For backtick-fenced MyST directives whose body is prose (admonition,
    seealso, figure, ...), the opening and closing fence markers are emitted
    as 'fence' chunks and the body as a 'prose' chunk so its punctuation
    gets rewritten too.
    """
    chunks: list[tuple[str, str]] = []
    lines = src.split("\n")
    n = len(lines)
    i = 0

    # Optional YAML front-matter at very top.
    if n > 0 and lines[0].rstrip() == "---":
        j = 1
        while j < n and lines[j].rstrip() != "---":
            j += 1
        if j < n:
            chunks.append(("frontmatter", "\n".join(lines[0 : j + 1])))
            i = j + 1

    prose_buf: list[str] = []
    while i < n:
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            if prose_buf:
                chunks.append(("prose", "\n".join(prose_buf)))
                prose_buf = []
            _fence_indent, fence_marker, info = m.group(1), m.group(2), m.group(3)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            open_line = line
            body_lines: list[str] = []
            close_line: str | None = None
            i += 1
            while i < n:
                m_close = FENCE_RE.match(lines[i])
                if (
                    m_close
                    and m_close.group(2)[0] == fence_char
                    and len(m_close.group(2)) >= fence_len
                    and m_close.group(3).strip() == ""
                ):
                    close_line = lines[i]
                    i += 1
                    break
                body_lines.append(lines[i])
                i += 1
            if _is_prose_body_fence(info):
                chunks.append(("fence", open_line))
                if body_lines:
                    chunks.append(("prose", "\n".join(body_lines)))
                if close_line is not None:
                    chunks.append(("fence", close_line))
            else:
                all_lines = [open_line, *body_lines]
                if close_line is not None:
                    all_lines.append(close_line)
                chunks.append(("fence", "\n".join(all_lines)))
            continue
        prose_buf.append(line)
        i += 1

    if prose_buf:
        chunks.append(("prose", "\n".join(prose_buf)))
    return chunks


# ---------------------------------------------------------------------------
# Prose tokenisation: protect inline code and math.
# ---------------------------------------------------------------------------

# Order matters: longest/most specific first.
PROTECT_RE = re.compile(
    r"(?P<bmath>\$\$[\s\S]*?\$\$)"     # block math (multiline)
    r"|(?P<imath>(?<!\\)\$[^\$\n]+?\$)"  # inline math (single line)
    r"|(?P<code>`+[^`\n]+?`+)"          # inline code
)


_PH_RE = re.compile(r"\x00PH(\d+)\x00")


def transform_prose(text: str) -> str:
    """Apply punctuation rewrites to a chunk of prose, protecting math/code.

    Protected spans are replaced with opaque placeholders BEFORE rewriting
    so structural regexes (e.g. heading detection) can still match across
    inline-code spans, then restored verbatim after rewriting.
    """
    placeholders: list[str] = []

    def _save(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"\x00PH{len(placeholders) - 1}\x00"

    masked = PROTECT_RE.sub(_save, text)
    rewritten = _rewrite(masked)

    def _restore(m: re.Match[str]) -> str:
        return placeholders[int(m.group(1))]

    return _PH_RE.sub(_restore, rewritten)


# ---------------------------------------------------------------------------
# Core rewrites
# ---------------------------------------------------------------------------

# em-dash that follows a label-like token: heading, list bullet, closing
# inline-code/link/bold. Captured group is kept verbatim.
#
# Note: plain trailing words are intentionally NOT treated as labels, because
# in prose `X — Y` is much more often a parenthetical/contrast than a label.
# A comma reads more natively in those cases (handled by the fall-through).
LABEL_EMDASH_RE = re.compile(
    r"(?P<label>"
    r"^\s{0,3}#{1,6}\s+[^\n—]*?"            # heading line content
    r"|^\s{0,3}[-*+]\s+[^\n—]*?"            # bullet list item content
    r"|^\s{0,3}\d+\.\s+[^\n—]*?"            # ordered list item content
    r"|^\s*:::+\s*\{[^}\n]+\}[^\n—]*?"      # MyST directive title line
    r"|`[^`\n]+`"                            # closing inline code
    r"|\]\([^)\n]+\)"                        # closing markdown link
    r"|\*\*[^*\n]+\*\*"                      # closing bold
    r")\s+—\s+",
    re.MULTILINE,
)


def _replace_em_after_label(match: re.Match[str]) -> str:
    return f"{match.group('label')}: "


def _capitalize_first(line: str) -> str:
    for k, ch in enumerate(line):
        if ch.isalpha():
            return line[:k] + ch.upper() + line[k + 1 :]
        if ch not in " \t":
            return line
    return line


def _semicolon_newline(match: re.Match[str]) -> str:
    """Replace `; \n[ws]*<line>` with `.\n[ws]*<Line>` (capitalise first letter)."""
    prefix_ws = match.group("pre") or ""
    next_line = match.group("next") or ""
    return f".\n{prefix_ws}{_capitalize_first(next_line)}"


def _rewrite(text: str) -> str:
    if not text:
        return text

    # 1. en-dash everywhere -> ASCII hyphen
    text = text.replace("–", "-")

    # 2. em-dash with label context -> ": "
    #    Apply in a loop to catch nested/multiple per line.
    prev = None
    while prev != text:
        prev = text
        text = LABEL_EMDASH_RE.sub(_replace_em_after_label, text)

    # 3. remaining em-dashes -> ", "
    text = re.sub(r"\s*—\s*", ", ", text)

    # 4. Semicolons.
    #    4a. End-of-line / followed by blank line + new paragraph that starts uppercase:
    text = re.sub(
        r";[ \t]*\n(?P<pre>[ \t]*)(?P<next>[^\n]*)",
        _semicolon_newline,
        text,
    )
    #    4b. Mid-sentence "; " -> ", "
    text = re.sub(r";[ \t]+", ", ", text)
    #    4c. Bare ";" without trailing whitespace -> ","
    text = text.replace(";", ",")

    # 5. Tidy: collapse ", ," and ", ." which a bad input could produce.
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",\s*\.", ".", text)

    return text


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def process_file(path: Path) -> dict:
    original = path.read_text(encoding="utf-8")
    chunks = split_top_level(original)
    rebuilt = "\n".join(
        transform_prose(text) if kind == "prose" else text for kind, text in chunks
    )

    # Count what changed.
    stats = {
        "em_before": original.count("—"),
        "en_before": original.count("–"),
        "semi_before": original.count(";"),
        "em_after": rebuilt.count("—"),
        "en_after": rebuilt.count("–"),
        "semi_after": rebuilt.count(";"),
        "changed": rebuilt != original,
    }

    if stats["changed"]:
        path.write_text(rebuilt, encoding="utf-8")
    return stats


def iter_markdown(roots: list[Path]):
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".md":
            yield root
            continue
        for p in root.rglob("*.md"):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes but do not write files.",
    )
    args = ap.parse_args()

    total = {"em": 0, "en": 0, "semi": 0, "files_changed": 0, "files_seen": 0}
    for p in sorted(iter_markdown(args.paths)):
        total["files_seen"] += 1
        if args.dry_run:
            original = p.read_text(encoding="utf-8")
            chunks = split_top_level(original)
            rebuilt = "\n".join(
                transform_prose(text) if k == "prose" else text for k, text in chunks
            )
            changed = rebuilt != original
            em_d = original.count("—") - rebuilt.count("—")
            en_d = original.count("–") - rebuilt.count("–")
            semi_d = original.count(";") - rebuilt.count(";")
        else:
            stats = process_file(p)
            changed = stats["changed"]
            em_d = stats["em_before"] - stats["em_after"]
            en_d = stats["en_before"] - stats["en_after"]
            semi_d = stats["semi_before"] - stats["semi_after"]

        if changed:
            total["files_changed"] += 1
            total["em"] += em_d
            total["en"] += en_d
            total["semi"] += semi_d
            print(
                f"  {p}: -{em_d} em-dash, -{en_d} en-dash, -{semi_d} semicolon"
            )

    print()
    print(
        f"Summary: {total['files_changed']}/{total['files_seen']} files changed, "
        f"removed {total['em']} em-dash, {total['en']} en-dash, "
        f"{total['semi']} semicolon."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
