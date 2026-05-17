import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ResourceHit = dict[str, str]


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


def _hit_from_ddgs(r: dict[str, Any], group: str) -> ResourceHit | None:
    url = (r.get("href") or "").strip()
    if not url:
        return None
    return {
        "title": (r.get("title") or "Resource").strip(),
        "url": url,
        "snippet": ((r.get("body") or "")[:400]).strip(),
        "group": group,
    }


def _hit_from_firecrawl(doc: Any, group: str) -> ResourceHit | None:
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
    url = (url or "").strip()
    if not url:
        return None
    return {
        "title": (title or "Resource").strip(),
        "url": url,
        "snippet": (body or "")[:400].strip(),
        "group": group,
    }


def search_resource_hits(
    query_groups: list[tuple[str, str]],
    limit_per_query: int = 4,
) -> list[ResourceHit]:
    """Run searches; return deduped hits with human-readable group labels."""
    key = os.getenv("FIRECRAWL_API_KEY")
    hits: list[ResourceHit] = []
    seen: set[str] = set()

    def add(hit: ResourceHit | None) -> None:
        if not hit:
            return
        url = hit["url"]
        if url in seen:
            return
        seen.add(url)
        hits.append(hit)

    if key:
        from firecrawl import FirecrawlApp

        app = FirecrawlApp(api_key=key)
        for query, group in query_groups:
            try:
                resp = app.search(query, limit=limit_per_query)
                data = getattr(resp, "data", None) or []
                for doc in data:
                    add(_hit_from_firecrawl(doc, group))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Firecrawl resource search failed for %r: %s", query, exc)
    else:
        from ddgs import DDGS

        with DDGS() as ddgs:
            for query, group in query_groups:
                try:
                    for r in ddgs.text(query, max_results=limit_per_query):
                        add(_hit_from_ddgs(r, group))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Resource search failed for %r: %s", query, exc)

    return hits


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
