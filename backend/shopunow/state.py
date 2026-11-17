from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    user_id: str
    session_id: str

    # None until sentiment_node runs; after that, always one of these exact values —
    # matches schemas.models.SentimentResult, which is what actually constrains the LLM's output.
    sentiment: Optional[Literal["positive", "neutral", "negative"]]
    tone: Optional[Literal["calm", "frustrated", "urgent", "appreciative"]]

    # Stays a plain string, not a Literal: department ids come from data/departments.py's
    # runtime registry (plus the special "general" value), which can grow without a matching
    # type-level change — unlike sentiment/tone, this set isn't fixed by definition.
    department: Optional[str]
    is_supported: bool
    needs_escalation: bool
    escalation_reason: Optional[str]
    blocked_by_guardrail: bool  # True only for a guardrail breach, not a plain human handoff

    retrieved_docs: list[dict]
    used_web_search: bool
    draft_answer: Optional[str]
    reflection_count: int
    reflection_feedback: Optional[str]

    final_response: Optional[str]
    incident_log: list[str]
