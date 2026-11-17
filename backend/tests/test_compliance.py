import pytest
from langchain_core.messages import HumanMessage

from shopunow.agents import compliance_agent
from shopunow.graph.build import _guarded
from shopunow.guardrails.exceptions import ComplianceViolation
from shopunow.schemas.models import ComplianceVerdict
from tests.conftest import FakeChatModel


def test_compliant_draft_becomes_final_response(monkeypatch):
    verdict = ComplianceVerdict(is_compliant=True, reason="fine")
    monkeypatch.setattr(compliance_agent, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(verdict))

    result = compliance_agent.compliance_node({"draft_answer": "Here's your answer."})

    assert result["final_response"] == "Here's your answer."
    assert result["messages"][0].content == "Here's your answer."  # recorded for next turn's history


def test_non_compliant_draft_raises_instead_of_being_sent(monkeypatch):
    verdict = ComplianceVerdict(is_compliant=False, reason="rude tone toward the user")
    monkeypatch.setattr(compliance_agent, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(verdict))

    with pytest.raises(ComplianceViolation):
        compliance_agent.compliance_node({"draft_answer": "Figure it out yourself."})


def test_full_chain_a_real_violation_never_reaches_the_user(monkeypatch):
    """Proves the actual mechanism end to end: compliance_node raises because the (mocked)
    LLM verdict said non-compliant, and the graph's shared guardrail wrapper is what turns
    that into a blocked escalation — the bad draft itself is discarded, not just flagged."""
    verdict = ComplianceVerdict(is_compliant=False, reason="used profanity")
    monkeypatch.setattr(compliance_agent, "ChatGoogleGenerativeAI", lambda **_: FakeChatModel(verdict))

    guarded_compliance = _guarded(compliance_agent.compliance_node)
    result = guarded_compliance(
        {
            "messages": [HumanMessage("you're all useless")],
            "draft_answer": "Figure it out yourself, idiot.",
            "incident_log": [],
        }
    )

    assert "final_response" not in result  # the bad draft never made it out
    assert result["needs_escalation"] is True
    assert result["blocked_by_guardrail"] is True
    assert "ComplianceViolation" in result["incident_log"][0]
