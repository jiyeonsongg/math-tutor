from typing import Any

from agents.state import TutorState
from tools.quiz_tools import summarize_cycle


def feynman_agent(state: TutorState) -> dict[str, Any]:
    return summarize_cycle(state)
