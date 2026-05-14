import json
import logging
import os
import uuid
from typing import Any, Literal, get_args

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

from agents.state import TutorState
from tools.web_search import firecrawl_search

logger = logging.getLogger(__name__)

FormatKind = Literal[
    "word_problem",
    "pure_symbolic",
    "explain_show_work",
    "representation_reading",
    "compare_or_select",
    "error_spotting",
    "constraint_or_inequality",
    "piecewise_table_or_cases",
    "short_proof_or_justify",
    "applied_modeling",
    "parametric_or_generalization",
    "sequence_or_pattern",
]


class QuizItem(BaseModel):
    id: str = Field(description="Stable id for this item, e.g. q1")
    question: str
    answer: str
    concept: str = Field(description="Short math topic label, e.g. linear equations")
    format_kind: FormatKind = Field(
        description="Question presentation style; must vary across the batch (no duplicate format_kind)."
    )


class QuizBatch(BaseModel):
    items: list[QuizItem] = Field(default_factory=list)


def chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    kwargs: dict[str, Any] = {"model": model}
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        kwargs["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", "0.35"))
    return ChatOpenAI(**kwargs)


def _level_rubric(level: str) -> str:
    return {
        "basic": (
            "BASIC: one–two clear steps, familiar procedures, minimal reading load. "
            "Still require understanding, not only mechanical plug-in."
        ),
        "challenge": (
            "CHALLENGE (NOT remedial): expect 3–7 coherent steps, combine two ideas, "
            "decode a modest word-problem setup, or justify a non-obvious intermediate claim. "
            "Avoid one-line template drills (e.g. only plug into y=mx+b with all numbers given up front). "
            "If a textbook excerpt is provided, match its *density of reasoning* and notation style: "
            "if the excerpt shows multi-step or contest-style work, your problems must feel that hard; "
            "do not simplify below the excerpt unless the excerpt itself is elementary."
        ),
        "honor": (
            "HONORS / ENRICHMENT: multi-constraint setups, proof sketches, counterexamples, "
            "parameter dependence, or problems where the solver must *design* an approach. "
            "Align strictly with any provided textbook excerpt difficulty when present."
        ),
    }.get(level.lower(), "Match the selected level faithfully; do not default to easier work.")


def _format_diversity_rules(n: int) -> str:
    kinds = list(get_args(FormatKind))
    return (
        "Each item must set `format_kind` to one of these literal values exactly "
        f"(spelling matters): {', '.join(kinds)}. "
        f"Across this batch of {n} questions, **do not reuse** the same `format_kind` "
        f"until every distinct kind has been used at least once (for n ≤ {len(kinds)} use all different kinds). "
        "Rotate styles: story context, bare algebra, explain/justify, read a table or piecewise description, "
        "compare strategies, find-the-error, inequalities, patterns, short proof, modeling with units, etc."
    )


def _quiz_system_message(level: str) -> str:
    return (
        "You are an expert mathematics assessment author. Generate original, correct problems "
        "that match the student's grade, topic, and difficulty level.\n\n"
        f"LEVEL RUBRIC: {_level_rubric(level)}\n\n"
        f"FORMAT DIVERSITY: {_format_diversity_rules(20)}\n\n"
        "TEXTBOOK / EXCERPT: When `textbook_excerpt` is non-empty, treat it as the difficulty and style anchor. "
        "Mirror vocabulary, abstraction, and step count — not wording. Never copy long phrases verbatim.\n\n"
        "Formatting (required):\n"
        "- Put every equation, inequality, or symbolic expression in LaTeX using "
        r"`\(` and `\)` for inline math, or `\[` and `\]` on their own lines for display math."
        "\n- Do NOT use bare `$` for math. Do NOT wrap math in single `$…$`.\n"
        "- For money, write \"12 dollars\" / \"USD 12\", never a `$` currency sign.\n"
        "- Use proper LaTeX: `\\times`, `\\cdot`, `\\frac{a}{b}`, subscripts `x_1`, superscripts `x^2`, "
        "`\\leq`, `\\geq`.\n"
        "- Complete, grammatical statements only.\n"
        "When web search is available, use it to sample *current* problem styles and difficulty cues for the topic; "
        "still write original items."
    )


def _quiz_user_payload(
    *,
    grade: int,
    level: str,
    sections: str,
    n: int,
    excerpt: str,
    research: str,
) -> str:
    return json.dumps(
        {
            "grade": grade,
            "level": level,
            "sections": sections,
            "num_questions": n,
            "textbook_excerpt": excerpt[:24000] if excerpt else None,
            "web_research_excerpt": research[:16000] if research else None,
            "task": (
                f"Produce exactly {n} distinct quiz items. Honor the level rubric and format diversity rules. "
                "Answers: concise but show key results using LaTeX where needed."
            ),
        },
        ensure_ascii=False,
    )


def _quiz_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {"type": "web_search", "search_context_size": "high"},
    ]
    vs = (os.getenv("OPENAI_VECTOR_STORE_ID") or "").strip()
    if vs:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [vs],
                "max_num_results": 12,
            }
        )
    return tools


def _generate_via_openai_responses(
    *,
    model: str,
    level: str,
    user_payload: str,
) -> QuizBatch | None:
    if os.getenv("OPENAI_QUIZ_USE_RESPONSES", "1").strip().lower() in ("0", "false", "no"):
        return None
    client = OpenAI()
    instructions = _quiz_system_message(level)
    tools = _quiz_tools()
    kwargs: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": user_payload,
        "text_format": QuizBatch,
        "tools": tools,
        "tool_choice": "auto",
        "max_output_tokens": int(os.getenv("OPENAI_QUIZ_MAX_OUTPUT", "12000")),
        "max_tool_calls": int(os.getenv("OPENAI_QUIZ_MAX_TOOL_CALLS", "16")),
    }
    if os.getenv("OPENAI_QUIZ_REASONING", "1").strip().lower() not in ("0", "false", "no"):
        kwargs["reasoning"] = {"effort": os.getenv("OPENAI_REASONING_EFFORT", "medium")}
    try:
        resp = client.responses.parse(**kwargs)
        parsed = resp.output_parsed
        if parsed is None:
            logger.warning("Responses API returned no parsed QuizBatch")
            return None
        return parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAI Responses quiz generation failed (%s); falling back to Chat Completions.", exc)
        try:
            del kwargs["reasoning"]
            resp = client.responses.parse(**kwargs)
            parsed = resp.output_parsed
            return parsed
        except Exception as exc2:  # noqa: BLE001
            logger.warning("Responses retry without reasoning failed: %s", exc2)
            return None


def research_for_quiz(state: TutorState) -> dict[str, Any]:
    grade = state.get("grade", 6)
    level = state.get("level", "basic")
    sections = state.get("sections", "").strip() or "general middle school math"
    n = max(1, min(int(state.get("num_questions", 5)), 20))
    excerpt = (state.get("textbook_excerpt") or "").strip()

    queries = [
        f"grade {grade} math {level} difficulty problems {sections} multi-step",
        f"grade {grade} {sections} contest worksheet challenging",
    ]
    if excerpt:
        queries.append(f"grade {grade} {sections} textbook style assessment items")

    snippets = firecrawl_search(queries, limit_per_query=3)
    return {"research_snippets": snippets}


def generate_questions(state: TutorState) -> dict[str, Any]:
    grade = state.get("grade", 6)
    level = state.get("level", "basic")
    sections = state.get("sections", "").strip() or "core grade-level skills"
    n = max(1, min(int(state.get("num_questions", 5)), 20))
    research = state.get("research_snippets", "")
    excerpt = (state.get("textbook_excerpt") or "").strip()

    model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    user_payload = _quiz_user_payload(
        grade=grade,
        level=level,
        sections=sections,
        n=n,
        excerpt=excerpt,
        research=research,
    )

    batch: QuizBatch | None = _generate_via_openai_responses(
        model=model,
        level=level,
        user_payload=user_payload,
    )

    if batch is None:
        sys = SystemMessage(content=_quiz_system_message(level))
        human = HumanMessage(
            content=user_payload
            + f"\n\nReturn exactly {n} items with distinct ids q1..q{n} (or uuid-like ids). "
            "Every question and answer must follow the LaTeX delimiter rules in the system message."
        )
        structured = chat_model().with_structured_output(QuizBatch)
        batch = structured.invoke([sys, human])

    items = batch.items[:n]
    out: list[dict[str, Any]] = []
    for it in items:
        qid = it.id.strip() or str(uuid.uuid4())[:8]
        out.append(
            {
                "id": qid,
                "question": it.question.strip(),
                "answer": it.answer.strip(),
                "concept": it.concept.strip(),
                "format_kind": it.format_kind,
            }
        )
    while len(out) < n:
        i = len(out) + 1
        out.append(
            {
                "id": f"q{i}",
                "question": f"(Placeholder) Grade {grade} problem {i} on {sections}",
                "answer": "N/A — regenerate if you see this.",
                "concept": sections,
                "format_kind": "pure_symbolic",
            }
        )
    return {"questions": out}


def analyze_mistakes(state: TutorState) -> dict[str, Any]:
    wrong = set(state.get("wrong_question_ids") or [])
    questions = state.get("questions") or []
    wrong_items = [q for q in questions if q.get("id") in wrong]
    grade = state.get("grade", 6)
    level = state.get("level", "basic")
    sections = state.get("sections", "")

    sys = SystemMessage(
        content=(
            "You analyze student mistakes in math self-study. "
            "Infer likely misconceptions, procedural slips, or gaps in prerequisite knowledge. "
            "Be supportive and specific. Do not blame the student. "
            "When you write formulas, use LaTeX inside \\( and \\) for inline math; "
            "do not use bare `$` for math or currency (say 'dollars' or USD)."
        )
    )
    human = HumanMessage(
        content=json.dumps(
            {
                "grade": grade,
                "level": level,
                "sections": sections,
                "wrong_questions": wrong_items,
                "note": "Student checked the problems they believe they solved correctly; unchecked items are treated as needing review.",
            },
            ensure_ascii=False,
        )
    )
    resp = chat_model().invoke([sys, human])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"mistake_analysis": text.strip()}


def find_learning_resources(state: TutorState) -> dict[str, Any]:
    grade = state.get("grade", 6)
    sections = state.get("sections", "")
    questions = state.get("questions") or []
    wrong = set(state.get("wrong_question_ids") or [])
    concepts = sorted(
        {q.get("concept", "") for q in questions if q.get("id") in wrong and q.get("concept")}
    )
    concept_str = ", ".join(concepts) if concepts else sections

    queries = [
        f"site:youtube.com grade {grade} math {concept_str} lesson",
        f"grade {grade} math {concept_str} explained tutorial examples",
    ]
    snippets = firecrawl_search(queries, limit_per_query=4)
    return {"learning_resources": snippets}


def summarize_cycle(state: TutorState) -> dict[str, Any]:
    grade = state.get("grade", 6)
    analysis = state.get("mistake_analysis", "")
    resources = state.get("learning_resources", "")
    questions = state.get("questions") or []
    wrong = set(state.get("wrong_question_ids") or [])
    total = len(questions)
    wrong_n = len(wrong)
    score = round(100.0 * (total - wrong_n) / total, 1) if total else 0.0

    sys = SystemMessage(
        content=(
            "Write a short wrap-up (3–6 sentences) for a student and parent. "
            "Mention what went well, what to review next, and how to use the suggested materials. "
            "Plain language, no bullet lists unless truly helpful. "
            "If you include a formula, use LaTeX inside \\( and \\); avoid `$` math delimiters."
        )
    )
    human = HumanMessage(
        content=json.dumps(
            {
                "grade": grade,
                "score_percent": score,
                "mistake_analysis": analysis[:8000],
                "resource_notes": resources[:8000],
            },
            ensure_ascii=False,
        )
    )
    resp = chat_model().invoke([sys, human])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"cycle_summary": text.strip(), "score_percent": score}
