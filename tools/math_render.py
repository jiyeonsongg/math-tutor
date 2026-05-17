"""Render mixed text + LaTeX in Streamlit using KaTeX (no single-`$` delimiters, so currency stays readable)."""

from __future__ import annotations

import html
import json
import re
import uuid

import streamlit as st
import streamlit.components.v1 as components

_KATEX_CSS = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css"
_KATEX_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"
_KATEX_AUTO = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"

_KATEX_OPTS_JSON = json.dumps(
    {
        "delimiters": [
            {"left": "\\[", "right": "\\]", "display": True},
            {"left": "\\(", "right": "\\)", "display": False},
            {"left": "$$", "right": "$$", "display": True},
        ],
        "throwOnError": False,
        "strict": "ignore",
    }
)


def _iframe_height(text: str, min_h: int = 140, max_h: int = 960) -> int:
    """Estimate iframe height: wrapped lines, not only newline count."""
    t = str(text)
    newline_blocks = max(1, t.count("\n") + 1)
    # ~55–65 chars per line at typical component width
    char_lines = max(1, (len(t) + 54) // 55)
    lines = max(newline_blocks, char_lines)
    display_bonus = t.count("\\[") * 48 + t.count("$$") * 48
    est = int(56 + lines * 28 + display_bonus)
    return max(min_h, min(max_h, est))


def render_math_text(text: str) -> None:
    """Typeset `text` with KaTeX. Use LaTeX inside `\\( … \\)` or `\\[ … \\]` (or `$$ … $$` blocks)."""
    if not text or not str(text).strip():
        return
    body = html.escape(str(text), quote=True)
    uid = uuid.uuid4().hex[:10]
    h = _iframe_height(str(text))
    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="{_KATEX_CSS}"/>
<style>
  html {{ height: 100%; margin: 0; overflow: hidden; }}
  body {{
    height: 100%;
    margin: 0;
    padding: 4px 6px 12px 6px;
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
    box-sizing: border-box;
    font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
    font-size: 1.05rem;
    line-height: 1.58;
    color: #1f2937;
    background: transparent;
  }}
  .math-root {{ white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; }}
  .katex {{ font-size: 1.05em; }}
</style>
<script src="{_KATEX_JS}"></script>
<script src="{_KATEX_AUTO}"></script>
</head><body>
<div id="m{uid}" class="math-root">{body}</div>
<script>
  (function () {{
    var el = document.getElementById("m{uid}");
    if (!el || typeof renderMathInElement === "undefined") return;
    renderMathInElement(el, {_KATEX_OPTS_JSON});
  }})();
</script>
</body></html>"""
    components.html(page, height=h, scrolling=True)


def to_matplotlib_math(text: str | None) -> str | None:
    """Convert tutor LaTeX delimiters to matplotlib mathtext (`$...$`)."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return text if text is not None else None
    t = re.sub(r"\\\[(.+?)\\\]", r"$\1$", t, flags=re.DOTALL)
    t = re.sub(r"\\\((.+?)\\\)", r"$\1$", t, flags=re.DOTALL)
    return t


def _markdown_for_streamlit(text: str) -> str:
    """Map LaTeX delimiters used by the tutor to Streamlit markdown math."""
    t = str(text)
    t = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", t, flags=re.DOTALL)
    t = re.sub(r"\\\((.+?)\\\)", r"$\1$", t, flags=re.DOTALL)
    return t


def render_feedback_text(text: str) -> None:
    """Render agent feedback (headings, lists, bold) via Streamlit markdown."""
    if not text or not str(text).strip():
        return
    st.markdown(_markdown_for_streamlit(text))
