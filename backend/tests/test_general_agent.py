from langchain_core.messages import AIMessage, HumanMessage

from shopunow.agents import general_agent
from tests.conftest import FakeToolCallingLLM as _FakeToolCallingLLM


class _FakeWebSearchTool:
    name = "web_search"

    def __init__(self, results):
        self._results = results

    def invoke(self, _args):
        return self._results


class _FakeInventoryTool:
    name = "query_general_database"

    def __init__(self, rows):
        self._rows = rows

    def invoke(self, _args):
        return self._rows


def test_general_agent_answers_directly_without_searching_when_it_already_knows(monkeypatch):
    reply = AIMessage(content="ShopUNow sells clothing, DIY supplies, books, and toys.")
    monkeypatch.setattr(general_agent, "ChatGoogleGenerativeAI", lambda **_: _FakeToolCallingLLM([reply]))

    result = general_agent.general_node({"messages": [HumanMessage("What does ShopUNow sell?")]})

    assert result["used_web_search"] is False
    assert result["retrieved_docs"] == []
    assert "clothing" in result["draft_answer"]


def test_general_agent_calls_web_search_when_the_model_asks_for_it(monkeypatch):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "removing a wine stain"}, "id": "call_1"}],
    )
    final_response = AIMessage(content="Blot it with cold water, then apply a stain remover.")
    monkeypatch.setattr(
        general_agent, "ChatGoogleGenerativeAI", lambda **_: _FakeToolCallingLLM([tool_call_response, final_response])
    )
    monkeypatch.setattr(
        general_agent,
        "web_search",
        _FakeWebSearchTool([{"title": "Stain removal guide", "url": "https://example.com", "content": "Use cold water."}]),
    )

    result = general_agent.general_node({"messages": [HumanMessage("How do I remove a wine stain?")]})

    assert result["used_web_search"] is True
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["department"] == "general"
    assert result["draft_answer"] == "Blot it with cold water, then apply a stain remover."


def test_general_agent_calls_inventory_tool_for_stock_questions(monkeypatch):
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "query_general_database",
                "args": {"sql": "SELECT stock_qty FROM inventory WHERE name = 'Classic Denim Jacket'"},
                "id": "call_1",
            }
        ],
    )
    final_response = AIMessage(content="We currently have 19 Classic Denim Jackets in stock.")
    monkeypatch.setattr(
        general_agent, "ChatGoogleGenerativeAI", lambda **_: _FakeToolCallingLLM([tool_call_response, final_response])
    )
    monkeypatch.setattr(general_agent, "inventory_tool", _FakeInventoryTool([{"stock_qty": 19}]))

    result = general_agent.general_node(
        {"messages": [HumanMessage("How many Classic Denim Jackets do you have in stock?")]}
    )

    assert result["used_web_search"] is False
    assert result["draft_answer"] == "We currently have 19 Classic Denim Jackets in stock."
