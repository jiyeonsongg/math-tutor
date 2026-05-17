"""Anonymous per-browser identity (cookie when available, else Streamlit session id)."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from extra_streamlit_components import CookieManager

_COOKIE_NAME = "jinni_visitor_id"
_COOKIE_DAYS = 400
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _valid_visitor_id(value: str | None) -> str | None:
    if value and _UUID_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def _streamlit_session_visitor_id() -> str:
    """Per-browser tab id from Streamlit (no extra packages)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None and getattr(ctx, "session_id", None):
            return f"st_{ctx.session_id}"
    except Exception:  # noqa: BLE001
        pass
    return f"st_{uuid.uuid4()}"


@st.cache_resource
def _cookie_manager() -> CookieManager:
    from extra_streamlit_components import CookieManager

    return CookieManager(key="jinni_cookie_manager")


def _ensure_via_cookie() -> str | None:
    try:
        from extra_streamlit_components import CookieManager  # noqa: F401
    except ImportError:
        return None

    cm = _cookie_manager()
    from_cookie = _valid_visitor_id(cm.get(cookie=_COOKIE_NAME))
    if from_cookie:
        return from_cookie

    all_cookies = cm.get_all() or {}
    from_all = _valid_visitor_id(all_cookies.get(_COOKIE_NAME))
    if from_all:
        return from_all

    new_id = str(uuid.uuid4())
    expires = datetime.now() + timedelta(days=_COOKIE_DAYS)
    cm.set(
        _COOKIE_NAME,
        new_id,
        expires_at=expires,
        key="jinni_set_visitor_cookie",
    )
    st.session_state.visitor_id = new_id
    st.rerun()
    return new_id  # unreachable after rerun


def ensure_visitor_id() -> str:
    """Stable id for this browser; prefers cookie, falls back to Streamlit session."""
    cached = st.session_state.get("visitor_id")
    if cached:
        return str(cached)

    via_cookie = _ensure_via_cookie()
    if via_cookie:
        st.session_state.visitor_id = via_cookie
        return via_cookie

    vid = _streamlit_session_visitor_id()
    st.session_state.visitor_id = vid
    return vid
