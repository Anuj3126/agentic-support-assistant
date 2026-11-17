"""Wires every node into the StateGraph:

sentiment -> router -> (one of 6 department RAG agents, the general web-search agent, or
                         straight to escalation)
           -> reflection (may loop back to the same agent once)
           -> compliance -> end, with escalation reachable from router/compliance too.

This is also where guardrail breaches are handled, in one place, for every node: an agent
raises a GuardrailViolation (prompt injection, a compliance breach, ...) and doesn't need to
know what happens next — _guarded(), applied to every node below, catches it, stops that
node's normal output from going any further, and redirects the turn to a human handoff.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, default_retry_on

from shopunow.agents.compliance_agent import compliance_node
from shopunow.agents.escalation_agent import escalation_node
from shopunow.agents.general_agent import general_node
from shopunow.agents.rag_agent import make_department_agent
from shopunow.agents.reflection import reflection_node
from shopunow.agents.router import router_node
from shopunow.agents.sentiment import sentiment_node
from shopunow.data.departments import DEPARTMENT_IDS
from shopunow.graph.routing import (
    ALL_ANSWER_NODES,
    route_after_compliance,
    route_after_reflection,
    route_after_router,
)
from shopunow.guardrails.exceptions import GuardrailViolation
from shopunow.memory.checkpointer import get_checkpointer
from shopunow.state import SupportState

DEFAULT_ESCALATION_MESSAGE = "Let's get a human teammate to help you with this one."


def _guarded(node_fn):
    """Wraps a node so any GuardrailViolation it raises stops the flow right there and routes
    to escalation, instead of every agent needing its own try/except for this."""

    def wrapped(state: SupportState) -> dict:
        try:
            return node_fn(state)
        except GuardrailViolation as exc:
            incident_log = state.get("incident_log", [])
            return {
                "needs_escalation": True,
                "blocked_by_guardrail": True,
                "escalation_reason": DEFAULT_ESCALATION_MESSAGE,
                "incident_log": incident_log + [f"{type(exc).__name__}: {exc}"],
            }

    wrapped.__name__ = node_fn.__name__
    return wrapped


def _retry_on(exc: BaseException) -> bool:
    # Guardrail exceptions are deliberate stops, not transient failures — never retry them.
    # (In practice _guarded() already absorbs them before a retry would be considered; this
    # is a defensive second layer in case one ever escapes that.)
    if isinstance(exc, GuardrailViolation):
        return False
    return default_retry_on(exc)


LLM_RETRY_POLICY = RetryPolicy(max_attempts=3, retry_on=_retry_on)


def build_graph():
    builder = StateGraph(SupportState)

    builder.add_node("sentiment", _guarded(sentiment_node), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("router", _guarded(router_node), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("general_agent", _guarded(general_node), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("reflection", _guarded(reflection_node), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("compliance_agent", _guarded(compliance_node), retry_policy=LLM_RETRY_POLICY)
    builder.add_node("escalation_agent", escalation_node)

    for dept_id in DEPARTMENT_IDS:
        builder.add_node(
            f"{dept_id}_agent",
            _guarded(make_department_agent(dept_id)),
            retry_policy=LLM_RETRY_POLICY,
        )

    builder.add_edge(START, "sentiment")
    builder.add_edge("sentiment", "router")

    builder.add_conditional_edges(
        "router", route_after_router, ["escalation_agent", *ALL_ANSWER_NODES]
    )

    for node in ALL_ANSWER_NODES:
        builder.add_edge(node, "reflection")

    builder.add_conditional_edges(
        "reflection", route_after_reflection, [*ALL_ANSWER_NODES, "compliance_agent"]
    )
    builder.add_conditional_edges(
        "compliance_agent", route_after_compliance, ["escalation_agent", END]
    )
    builder.add_edge("escalation_agent", END)

    return builder.compile(checkpointer=get_checkpointer())
