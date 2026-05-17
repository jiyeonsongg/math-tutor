"""Anonymous per-browser identity via cookie (no login)."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

import streamlit as st
from extra_streamlit_components import CookieManager

_COOKIE_NAME = "jinni_visitor_id"
_COOKIE_DAYS = 400
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


@st.cache_resource
def _cookie_manager() -> CookieManager:
    return CookieManager(key="jinni_cookie_manager")


def _valid_visitor_id(value: str | None) -> str | None:
    if value and _UUID_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def ensure_visitor_id() -> str:
    """Return a stable id for this browser; set cookie on first visit."""
    cached = st.session_state.get("visitor_id")
    if cached and _valid_visitor_id(cached):
        return cached

    cm = _cookie_manager()
    from_cookie = _valid_visitor_id(cm.get(cookie=_COOKIE_NAME))
    if from_cookie:
        st.session_state.visitor_id = from_cookie
        return from_cookie

    all_cookies = cm.get_all() or {}
    from_all = _valid_visitor_id(all_cookies.get(_COOKIE_NAME))
    if from_all:
        st.session_state.visitor_id = from_all
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
