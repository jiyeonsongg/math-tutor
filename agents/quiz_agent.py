from typing import Any

from agents.state import TutorState
from tools.quiz_tools import generate_questions, research_for_quiz


def quiz_agent(state: TutorState) -> dict[str, Any]:
    research = research_for_quiz(state)
    merged = dict(state)
    merged.update(research)
    generated = generate_questions(merged)
    return {**research, **generated}
