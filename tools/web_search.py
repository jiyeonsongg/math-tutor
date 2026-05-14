import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _format_hit(doc: Any) -> str:
    if isinstance(doc, dict):
        title = doc.get("title") or ""
        url = doc.get("url") or ""
        body = (
            doc.get("markdown")
            or doc.get("description")
            or doc.get("content", "")
            or ""
        )
    else:
        title = getattr(doc, "title", None) or ""
        url = getattr(doc, "url", None) or ""
        body = (
            getattr(doc, "markdown", None)
            or getattr(doc, "description", None)
            or ""
        )
    body = (body or "")[:2500]
    return f"**{title}**\n{url}\n\n{body}\n\n---\n"


def _duckduckgo_search(queries: list[str], limit_per_query: int) -> str:
    from ddgs import DDGS

    parts: list[str] = []
    with DDGS() as ddgs:
        for q in queries:
            parts.append(f"## Query: {q}\n")
            try:
                results = ddgs.text(q, max_results=limit_per_query)
                for r in results:
                    title = r.get("title") or ""
                    url = r.get("href") or ""
                    body = (r.get("body") or "")[:2500]
                    parts.append(f"**{title}**\n{url}\n\n{body}\n\n---\n")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Web search failed for query %r: %s", q, exc)
                parts.append("(No results for this search.)\n")
    return "\n".join(parts).strip()


def firecrawl_search(queries: list[str], limit_per_query: int = 4) -> str:
    key = os.getenv("FIRECRAWL_API_KEY")
    if not key:
        try:
            return _duckduckgo_search(queries, limit_per_query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Web search fallback failed: %s", exc)
            return ""
    from firecrawl import FirecrawlApp

    app = FirecrawlApp(api_key=key)
    parts: list[str] = []
    for q in queries:
        try:
            resp = app.search(q, limit=limit_per_query)
            data = getattr(resp, "data", None) or []
            parts.append(f"## Query: {q}\n")
            for doc in data:
                parts.append(_format_hit(doc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl search failed for query %r: %s", q, exc)
            parts.append(f"## Query: {q}\n")
            parts.append("(No results for this search.)\n")
    return "\n".join(parts).strip()
