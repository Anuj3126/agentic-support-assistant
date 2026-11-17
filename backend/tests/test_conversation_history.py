"""Proves the specific mechanism that makes multi-turn memory actually work (not just the
checkpointer plumbing, which test_graph_e2e.py's continuity test covers): the generation
agents are given recent conversation history, not just the latest message, so a natural
follow-up question can actually be answered using something said earlier in the same session.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from shopunow.agents.general_agent import general_node
from shopunow.agents.rag_agent import make_department_agent

pytestmark = pytest.mark.live


def test_general_agent_recalls_something_from_earlier_in_the_conversation():
    state = {
        "messages": [
            HumanMessage("Hi, my name is Priya. What does ShopUNow sell?"),
            AIMessage("Hi Priya! ShopUNow sells clothing, DIY supplies, books, and toys."),
            HumanMessage("Thanks — quick check, what name did I just give you?"),
        ]
    }

    result = general_node(state)

    assert "priya" in result["draft_answer"].lower()


def test_department_agent_recalls_something_from_earlier_in_the_conversation():
    returns_agent = make_department_agent("returns")
    state = {
        "messages": [
            HumanMessage("Hi, my name is Priya. I want to return a jacket I bought last week."),
            AIMessage(
                "Hi Priya! You can start a return from My Orders > Return Item within 7 days "
                "of delivery, as long as it's in original condition with tags attached."
            ),
            HumanMessage("Great — one more time, what's my name and what am I returning?"),
        ],
        "reflection_feedback": None,
    }

    result = returns_agent(state)

    assert "priya" in result["draft_answer"].lower()
    assert "jacket" in result["draft_answer"].lower()
