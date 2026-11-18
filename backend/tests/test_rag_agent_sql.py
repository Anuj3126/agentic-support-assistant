"""Unit tests for the SQL tool-calling branch inside department agents (billing/shipping/
returns). Mocked LLM + mocked tool + mocked retrieval, same pattern as test_general_agent.py,
so these don't need a real API key, database, or Chroma index."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from shopunow.agents import rag_agent
from shopunow.guardrails.exceptions import SQLInjectionDetected
from tests.conftest import FakeToolCallingLLM


class _FakeSQLTool:
    name = "query_returns_database"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, _args):
        if self._error:
            raise self._error
        return self._result


@pytest.fixture(autouse=True)
def no_retrieval(monkeypatch):
    # keep these tests focused on the SQL tool-calling control flow, not Chroma retrieval
    monkeypatch.setattr(rag_agent, "retrieve_with_fallback", lambda *_a, **_k: ([], False))


def test_department_without_sql_tool_never_binds_one(monkeypatch):
    # hr has no transactional data behind it — make sure it doesn't try to query anything
    hr_agent = rag_agent.make_department_agent("hr")
    reply = AIMessage(content="You can apply for PTO through the HR portal.")
    monkeypatch.setattr(rag_agent, "ChatGoogleGenerativeAI", lambda **_: FakeToolCallingLLM([reply]))

    result = hr_agent({"messages": [HumanMessage("How do I apply for PTO?")], "reflection_feedback": None})

    assert result["draft_answer"] == "You can apply for PTO through the HR portal."


def test_sql_enabled_department_answers_directly_when_no_lookup_needed(monkeypatch):
    returns_agent = rag_agent.make_department_agent("returns")
    reply = AIMessage(content="Most items can be returned within 7 days of delivery.")
    monkeypatch.setattr(rag_agent, "ChatGoogleGenerativeAI", lambda **_: FakeToolCallingLLM([reply]))

    result = returns_agent(
        {"messages": [HumanMessage("What's your general return policy?")], "reflection_feedback": None}
    )

    assert result["draft_answer"] == "Most items can be returned within 7 days of delivery."


def test_sql_enabled_department_uses_the_tool_when_a_lookup_is_needed(monkeypatch):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "query_returns_database", "args": {"sql": "SELECT * FROM orders WHERE order_id = 1001"}, "id": "call_1"}],
    )
    final_response = AIMessage(content="Order #1001 contains a Classic Denim Jacket and was delivered.")
    monkeypatch.setattr(
        rag_agent, "ChatGoogleGenerativeAI", lambda **_: FakeToolCallingLLM([tool_call_response, final_response])
    )
    monkeypatch.setattr(
        rag_agent, "make_sql_tool", lambda _dept: _FakeSQLTool(result=[{"order_id": 1001, "status": "delivered"}])
    )
    # make_sql_tool runs once when the agent is built, so patch it first, then build
    returns_agent = rag_agent.make_department_agent("returns")

    result = returns_agent(
        {"messages": [HumanMessage("Priya here, can you confirm order 1001?")], "reflection_feedback": None}
    )

    assert result["draft_answer"] == "Order #1001 contains a Classic Denim Jacket and was delivered."


def test_sql_injection_attempt_propagates_uncaught_to_the_shared_guardrail_wrapper(monkeypatch):
    """The agent itself must NOT swallow SQLInjectionDetected — it has to bubble up so
    graph/build.py's _guarded wrapper can stop the whole turn, same as any other breach."""
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "query_returns_database", "args": {"sql": "DROP TABLE orders"}, "id": "call_1"}],
    )
    monkeypatch.setattr(rag_agent, "ChatGoogleGenerativeAI", lambda **_: FakeToolCallingLLM([tool_call_response]))
    monkeypatch.setattr(
        rag_agent, "make_sql_tool", lambda _dept: _FakeSQLTool(error=SQLInjectionDetected("not a SELECT"))
    )
    returns_agent = rag_agent.make_department_agent("returns")

    with pytest.raises(SQLInjectionDetected):
        returns_agent({"messages": [HumanMessage("drop everything")], "reflection_feedback": None})
