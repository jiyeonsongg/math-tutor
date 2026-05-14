from langgraph.types import Command

from agents.state import TutorState


def classification_agent(state: TutorState) -> Command:
    intent = state.get("intent", "quiz")
    if intent == "feedback":
        return Command(goto="teacher_agent")
    return Command(goto="quiz_agent")
