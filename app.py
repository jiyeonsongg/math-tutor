from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from main import graph
from tools.learning_resources import render_learning_resources
from tools.math_render import render_feedback_text, render_math_text
from tools.document_extract import merge_excerpt_parts, text_from_public_url, text_from_upload
from tools.figure_render import render_question_diagram
from tools.quiz_tools import (
    format_question_refs_in_text,
    normalize_question_id,
    normalize_questions_list,
)

SESSION_FILE = Path(__file__).resolve().parent / "study_sessions.json"


def _display_label(value: str) -> str:
    """Turn internal ids like word_problem into readable labels."""
    return value.replace("_", " ").strip()


def _inject_pastel_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --pastel-bg: #f2f2f2;
            --pastel-peach: #fde2c4;
            --pastel-pink: #fbcce7;
            --pastel-sky: #c5e9ff;
            --pastel-lime: #d5ff80;
            --pastel-yellow: #fff176;
            --pastel-lavender: #d1c4e9;
            --pastel-text: #2d3436;
          }
          .stApp {
            background: linear-gradient(
              165deg,
              var(--pastel-bg) 0%,
              #faf8fc 45%,
              var(--pastel-peach) 100%
            );
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(
              180deg,
              var(--pastel-sky) 0%,
              var(--pastel-lavender) 100%
            );
          }
          [data-testid="stSidebar"] * {
            color: var(--pastel-text);
          }
          h1, h2, h3, h4 {
            color: var(--pastel-text);
          }
          div[data-testid="stMetric"] {
            background: var(--pastel-yellow);
            border: 1px solid var(--pastel-lavender);
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
          }
          div[data-testid="stExpander"] details {
            border: 1px solid var(--pastel-lavender);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.55);
          }
          div[data-testid="stAlert"] {
            border-radius: 12px;
          }
          .stSuccess {
            background-color: var(--pastel-lime) !important;
            color: var(--pastel-text) !important;
          }
          .stInfo {
            background-color: var(--pastel-sky) !important;
            color: var(--pastel-text) !important;
          }
          div.stButton > button[kind="primary"] {
            background: linear-gradient(
              90deg,
              var(--pastel-lavender),
              var(--pastel-pink)
            );
            color: var(--pastel-text);
            border: 1px solid #c9b8e8;
            border-radius: 999px;
            font-weight: 600;
          }
          div.stButton > button[kind="primary"]:hover {
            border-color: var(--pastel-text);
            filter: brightness(1.03);
          }
          div.stButton > button[kind="secondary"] {
            background: var(--pastel-peach);
            color: var(--pastel-text);
            border-radius: 999px;
          }
          [data-testid="stSpinner"] label {
            white-space: nowrap;
          }
          @media (max-width: 640px) {
            [data-testid="stSpinner"] label {
              white-space: normal;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_disk_sessions() -> list[dict[str, Any]]:
    if not SESSION_FILE.is_file():
        return []
    try:
        raw = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_disk_sessions(rows: list[dict[str, Any]]) -> None:
    SESSION_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_memory(*, include_saved_sessions: bool = False) -> None:
    """Clear in-app study state. Optionally wipe saved session history and disk file."""
    st.session_state.active_pack = None
    st.session_state.last_feedback = None
    st.session_state.selected_cycle_idx = None
    if include_saved_sessions:
        st.session_state.cycles = []
        _save_disk_sessions([])
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (
            k.startswith("wrong_")
            or k.startswith("correct_")
            or k.startswith("diag_zoom_")
        ):
            del st.session_state[k]
    st.rerun()


def _init_session_state() -> None:
    if "cycles" not in st.session_state:
        st.session_state.cycles = _load_disk_sessions()
    if "active_pack" not in st.session_state:
        st.session_state.active_pack = None
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None
    if "selected_cycle_idx" not in st.session_state:
        st.session_state.selected_cycle_idx = None


def _sidebar_settings() -> dict[str, Any]:
    st.sidebar.header("💡Study setup")
    grade = st.sidebar.selectbox("Grade", list(range(1, 13)), index=7)
    level = st.sidebar.selectbox(
        "Level",
        ["basic", "challenge", "honor"],
        index=0,
        format_func=lambda x: x.capitalize(),
    )
    sections = st.sidebar.text_input(
        "Sections / topics",
        placeholder="e.g. linear equations, slope-intercept form",
    )
    num_questions = st.sidebar.slider("Number of questions", 1, 20, 5)

    uploaded = st.sidebar.file_uploader(
        "Optional: textbook sample file (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        help="Legacy Word .doc is not supported; save as .docx or PDF.",
    )
    textbook_url = st.sidebar.text_input(
        "Optional: public link to notes or textbook pages",
        placeholder="https://… (web page, PDF, Google Doc, or hosted .docx)",
        help="Must be publicly reachable without login. Google Docs: use Share → Anyone with the link.",
    )

    file_text = ""
    if uploaded is not None:
        try:
            file_text = text_from_upload(uploaded.name, uploaded.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Could not read file: {exc}")
            file_text = ""

    url_text = ""
    if textbook_url.strip():
        try:
            url_text = text_from_public_url(textbook_url)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.warning(f"Could not load URL (you can still generate without it): {exc}")

    excerpt = merge_excerpt_parts(file_text, url_text, max_chars=12000)

    st.sidebar.divider()
    st.sidebar.subheader("📚Past Study Sessions")
    if not st.session_state.cycles:
        st.sidebar.caption("Completed study cycles will appear here with your score.")
    else:
        for idx, c in enumerate(reversed(st.session_state.cycles)):
            real_idx = len(st.session_state.cycles) - 1 - idx
            label = f"{c.get('sections', 'Session')[:28]}… — {c.get('score_percent', 0)}%"
            if st.sidebar.button(label, key=f"cycle_{real_idx}"):
                st.session_state.selected_cycle_idx = real_idx
                st.session_state.last_feedback = {
                    "mistake_analysis": c.get("mistake_analysis", ""),
                    "learning_resources": c.get("learning_resources", ""),
                    "cycle_summary": c.get("cycle_summary", ""),
                    "score_percent": c.get("score_percent", 0),
                    "questions": c.get("questions", []),
                    "wrong_question_ids": c.get("wrong_question_ids", []),
                }

    st.sidebar.divider()
    st.sidebar.subheader("🧠Memory")
    st.sidebar.caption(
        "Reset clears the current practice set and feedback from the screen. "
        "Optionally also remove every saved session below."
    )
    wipe_history = st.sidebar.checkbox(
        "Also clear all saved sessions",
        value=False,
        key="reset_include_saved_sessions",
    )
    if st.sidebar.button("Reset memory", type="secondary"):
        reset_memory(include_saved_sessions=wipe_history)

    return {
        "grade": int(grade),
        "level": level,
        "sections": sections,
        "num_questions": int(num_questions),
        "textbook_excerpt": excerpt,
    }


def _generate_practice_set(settings: dict[str, Any]) -> None:
    if not settings["sections"].strip():
        st.error("Please enter the sections or topics you are studying (sidebar).")
        return
    with st.spinner("Searching the web and drafting questions…"):
        out = graph.invoke(
            {
                "messages": [],
                "current_agent": "classification_agent",
                "intent": "quiz",
                "grade": settings["grade"],
                "level": settings["level"],
                "sections": settings["sections"].strip(),
                "num_questions": settings["num_questions"],
                "textbook_excerpt": settings["textbook_excerpt"],
            }
        )
    st.session_state.active_pack = {
        "pack_id": str(uuid.uuid4()),
        "settings": settings,
        "research_snippets": out.get("research_snippets", ""),
        "questions": normalize_questions_list(out.get("questions", [])),
    }
    st.session_state.last_feedback = None
    st.session_state.selected_cycle_idx = None
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Math tutor", layout="wide")
    _inject_pastel_theme()
    _init_session_state()

    st.title("🧞Jinni: Your Intelligent Math Tutor💫")
    st.caption(
        "Generate practice from your grade and topics, check off what you got right, "
        "and get a short analysis plus links to follow-up materials."
    )

    settings = _sidebar_settings()

    col_go, _ = st.columns([1, 4])
    with col_go:
        gen_top = st.button("Generate practice set", type="primary", key="gen_top")

    if gen_top:
        _generate_practice_set(settings)

    if st.session_state.selected_cycle_idx is not None and st.session_state.last_feedback:
        st.info("Showing a saved session from the sidebar. Generate a new set to practice again.")

    pack = st.session_state.active_pack
    if pack:
        with st.expander("How this set was researched (web snippets)", expanded=False):
            st.markdown(pack.get("research_snippets") or "_No research text._")

        st.subheader("Questions")
        st.caption("Check the box for each problem you solved correctly.")
        pack_id = pack.get("pack_id", "default")
        correct: set[str] = set()
        for i, q in enumerate(pack["questions"], start=1):
            qid = normalize_question_id(str(q.get("id", "")), fallback_index=i)
            label = qid
            fk = q.get("format_kind")
            hdr = f"**{label}** — _{q.get('concept', '')}_"
            if fk:
                hdr += f" · _{_display_label(fk)}_"
            st.markdown(hdr)
            render_math_text(q["question"])
            if q.get("diagram"):
                render_question_diagram(q["diagram"], control_key=f"{pack_id}_{qid}")
            if st.checkbox(
                "I got this one correct",
                key=f"correct_{pack_id}_{qid}",
            ):
                correct.add(qid)

            with st.expander(f"Show answer — {label}", expanded=False):
                render_math_text(q.get("answer", "") or "")

        if st.button("Submit results and get feedback", type="primary"):
            qs = normalize_questions_list(pack["questions"])
            if not qs:
                st.error("No questions in this pack.")
            else:
                base = {**pack["settings"], "questions": qs, "research_snippets": pack.get("research_snippets", "")}
                all_ids = {q["id"] for q in qs}
                wrong_ids = sorted(all_ids - correct)
                base["wrong_question_ids"] = wrong_ids
                total = len(qs)
                perfect = len(wrong_ids) == 0 and total > 0
                spinner = (
                    "Gathering resources and your feedback…"
                    if perfect
                    else "Analyzing mistakes and gathering lecture-style resources…"
                )
                with st.spinner(spinner):
                    fb = graph.invoke(
                        {
                            "messages": [],
                            "current_agent": "classification_agent",
                            "intent": "feedback",
                            **base,
                        }
                    )
                wrong_n = len(wrong_ids)
                score = round(100.0 * (total - wrong_n) / total, 1) if total else 0.0
                fb["score_percent"] = score
                st.session_state.last_feedback = fb
                row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "grade": base["grade"],
                    "level": base["level"],
                    "sections": base["sections"],
                    "num_questions": total,
                    "questions": qs,
                    "wrong_question_ids": base["wrong_question_ids"],
                    "score_percent": score,
                    "mistake_analysis": fb.get("mistake_analysis", ""),
                    "learning_resources": fb.get("learning_resources", ""),
                    "cycle_summary": fb.get("cycle_summary", ""),
                }
                st.session_state.cycles.append(row)
                _save_disk_sessions(st.session_state.cycles)
                st.success("Cycle saved — find it in the sidebar with your score.")
                st.rerun()

    fb = st.session_state.last_feedback
    if fb:
        st.divider()
        st.subheader("Feedback for this cycle")
        score_pct = fb.get("score_percent", 0)
        perfect = score_pct == 100.0 or score_pct == 100
        st.metric(
            "Score (self-reported)",
            "100%🎉" if perfect else f"{score_pct}%",
        )
        if perfect:
            st.success("Perfect score — nice work!")
        focus_heading = "Great job!" if perfect else "What to focus on next"
        st.markdown(f"### {focus_heading}")
        render_feedback_text(
            format_question_refs_in_text(fb.get("mistake_analysis", "") or "")
        )
        st.markdown("### Videos and lessons to explore")
        render_learning_resources(fb.get("learning_resources"))
        st.markdown("### Summary")
        render_feedback_text(
            format_question_refs_in_text(fb.get("cycle_summary", "") or "")
        )

    if st.session_state.last_feedback is not None:
        st.divider()
        col_bottom, _ = st.columns([1, 4])
        with col_bottom:
            if st.button("Generate practice set", type="primary", key="gen_bottom"):
                _generate_practice_set(settings)


if __name__ == "__main__":
    main()
