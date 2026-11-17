"""Conditional-edge functions used to wire the graph. Kept separate from build.py so the
routing decisions are easy to read and test on their own."""

from langgraph.graph import END

from shopunow.data.departments import DEPARTMENT_IDS
from shopunow.state import SupportState

DEPARTMENT_AGENT_NODES = [f"{dept_id}_agent" for dept_id in DEPARTMENT_IDS]
ALL_ANSWER_NODES = [*DEPARTMENT_AGENT_NODES, "general_agent"]


def route_after_router(state: SupportState) -> str:
    department = state.get("department")
    if state.get("needs_escalation") or not department:
        return "escalation_agent"
    if department == "general":
        return "general_agent"
    return f"{department}_agent"


def route_after_reflection(state: SupportState) -> str:
    if not state.get("reflection_feedback"):
        return "compliance_agent"
    department = state.get("department")
    return "general_agent" if department == "general" else f"{department}_agent"


def route_after_compliance(state: SupportState) -> str:
    return "escalation_agent" if state.get("needs_escalation") else END
