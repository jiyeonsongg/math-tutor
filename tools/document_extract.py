"""Extract plain text from uploads and public URLs for the textbook sample field."""

from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


def _google_doc_export_txt(url: str) -> str | None:
    m = re.search(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", url)
    if not m:
        return None
    doc_id = m.group(1)
    return f"https://docs.google.com/document/d/{doc_id}/export?format=txt"


def text_from_upload(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


def text_from_public_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must start with http:// or https://")

    export = _google_doc_export_txt(url)
    headers = {"User-Agent": "MathTutorStudyApp/1.0 (educational text extraction)"}

    with httpx.Client(timeout=25.0, follow_redirects=True, headers=headers) as client:
        if export:
            r = client.get(export)
            r.raise_for_status()
            return r.text

        r = client.get(url)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()

        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            return text_from_upload("sample.pdf", r.content)

        if (
            "application/vnd.openxmlformats-officedocument" in ct
            or url.lower().endswith(".docx")
        ):
            return text_from_upload("sample.docx", r.content)

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)


def merge_excerpt_parts(*parts: str, max_chars: int = 12000) -> str:
    chunks = [p.strip() for p in parts if p and p.strip()]
    if not chunks:
        return ""
    merged = "\n\n---\n\n".join(chunks)
    return merged[:max_chars]
