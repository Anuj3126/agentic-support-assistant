from shopunow.agents import reflection
from shopunow.schemas.models import ReflectionCritique
from tests.conftest import FakeChatModel


def test_needs_revision_sets_feedback_and_increments_count(monkeypatch):
    critique = ReflectionCritique(needs_revision=True, feedback="didn't answer the actual question")
    monkeypatch.setattr(reflection, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(critique))

    result = reflection.reflection_node(
        {"messages": [_msg("where is my order?")], "draft_answer": "hello!", "reflection_count": 0}
    )

    assert result["reflection_count"] == 1
    assert result["reflection_feedback"] == "didn't answer the actual question"


def test_no_revision_needed_clears_feedback(monkeypatch):
    critique = ReflectionCritique(needs_revision=False, feedback="looks good")
    monkeypatch.setattr(reflection, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(critique))

    result = reflection.reflection_node(
        {"messages": [_msg("where is my order?")], "draft_answer": "it's on the way", "reflection_count": 0}
    )

    assert result["reflection_feedback"] is None


def test_hitting_max_reflections_suppresses_further_revision(monkeypatch):
    critique = ReflectionCritique(needs_revision=True, feedback="still not quite right")
    monkeypatch.setattr(reflection, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(critique))
    monkeypatch.setattr(reflection.settings, "max_reflections", 2)

    # already reflected once (reflection_count=1) — this call pushes it to the cap
    result = reflection.reflection_node(
        {"messages": [_msg("where is my order?")], "draft_answer": "still vague", "reflection_count": 1}
    )

    assert result["reflection_count"] == 2
    assert result["reflection_feedback"] is None  # at the cap, so don't ask for another revision


def _msg(text: str):
    from langchain_core.messages import HumanMessage

    return HumanMessage(text)
