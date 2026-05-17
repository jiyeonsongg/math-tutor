"""Structured web search hits and Streamlit-friendly display for lecture materials."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from tools.web_search import search_resource_hits


def fetch_learning_resource_hits(
    *,
    grade: int,
    topic: str,
    limit_per_group: int = 4,
) -> list[dict[str, str]]:
    topic = (topic or "math").strip()
    query_groups: list[tuple[str, str]] = [
        (
            f"site:youtube.com grade {grade} math {topic} lesson",
            "Video lessons",
        ),
        (
            f"grade {grade} math {topic} tutorial explained examples",
            "Guides and practice",
        ),
    ]
    return search_resource_hits(query_groups, limit_per_query=limit_per_group)


def _parse_legacy_markdown(text: str) -> list[dict[str, str]]:
    """Parse old markdown blobs saved before structured resources."""
    hits: list[dict[str, str]] = []
    current_group = "More resources"
    title = ""
    url = ""
    snippet_lines: list[str] = []

    def flush() -> None:
        nonlocal title, url, snippet_lines
        if title or url:
            hits.append(
                {
                    "title": title or "Resource",
                    "url": url,
                    "snippet": " ".join(snippet_lines)[:400],
                    "group": current_group,
                }
            )
        title = ""
        url = ""
        snippet_lines = []

    for raw in text.splitlines():
        line = raw.strip()
        if line == "---":
            flush()
            continue
        if not line:
            continue
        if line.startswith("## Query:"):
            flush()
            q = line.replace("## Query:", "", 1).strip()
            current_group = "Video lessons" if "youtube" in q.lower() else "Guides and practice"
            continue
        if line.startswith("**") and line.endswith("**"):
            flush()
            title = line.strip("*").strip()
            continue
        if line.startswith("http://") or line.startswith("https://"):
            url = line
            continue
        if not line.startswith("("):
            snippet_lines.append(line)
    flush()
    return hits


def normalize_learning_resources(data: Any) -> list[dict[str, str]]:
    if not data:
        return []
    if isinstance(data, list):
        out: list[dict[str, str]] = []
        for item in data:
            if isinstance(item, dict) and item.get("url"):
                out.append(
                    {
                        "title": str(item.get("title") or "Resource"),
                        "url": str(item["url"]),
                        "snippet": str(item.get("snippet") or "")[:400],
                        "group": str(item.get("group") or "More resources"),
                    }
                )
        return _dedupe_hits(out)
    if isinstance(data, str):
        return _dedupe_hits(_parse_legacy_markdown(data))
    return []


def _dedupe_hits(hits: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for h in hits:
        url = h.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(h)
    return out


def _host_label(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if "youtube" in host or "youtu.be" in host:
            return "YouTube"
        return host or "Link"
    except Exception:  # noqa: BLE001
        return "Link"


def render_learning_resources(data: Any) -> None:
    import streamlit as st

    hits = normalize_learning_resources(data)
    if not hits:
        st.caption("No extra videos or guides were found for this topic right now.")
        return

    by_group: dict[str, list[dict[str, str]]] = {}
    for h in hits:
        by_group.setdefault(h["group"], []).append(h)

    for group, items in by_group.items():
        with st.expander(f"{group} ({len(items)})", expanded=False):
            for i, item in enumerate(items):
                title = item["title"]
                url = item["url"]
                host = _host_label(url)
                st.markdown(f"**{title}**")
                st.caption(f"{host}")
                if item.get("snippet"):
                    st.write(item["snippet"])
                st.link_button("Open", url, use_container_width=False)
                if i < len(items) - 1:
                    st.divider()
