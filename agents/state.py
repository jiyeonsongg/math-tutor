from typing import Annotated, Any, Literal, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TutorState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    current_agent: str
    intent: NotRequired[Literal["quiz", "feedback"]]
    grade: NotRequired[int]
    level: NotRequired[str]
    sections: NotRequired[str]
    num_questions: NotRequired[int]
    textbook_excerpt: NotRequired[str]
    research_snippets: NotRequired[str]
    questions: NotRequired[list[dict[str, Any]]]
    wrong_question_ids: NotRequired[list[str]]
    mistake_analysis: NotRequired[str]
    learning_resources: NotRequired[str]
    cycle_summary: NotRequired[str]
    score_percent: NotRequired[float]
