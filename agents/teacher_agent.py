from typing import Any

from agents.state import TutorState
from tools.quiz_tools import analyze_mistakes, find_learning_resources


def teacher_agent(state: TutorState) -> dict[str, Any]:
    analysis = analyze_mistakes(state)
    merged = dict(state)
    merged.update(analysis)
    resources = find_learning_resources(merged)
    return {**analysis, **resources}
