import pytest

from shopunow.graph.build import _guarded
from shopunow.guardrails.exceptions import ComplianceViolation, PromptInjectionDetected


def test_guarded_converts_guardrail_violation_into_escalation():
    def node(_state):
        raise PromptInjectionDetected("looked like a jailbreak attempt")

    result = _guarded(node)({"incident_log": []})

    assert result["needs_escalation"] is True
    assert result["blocked_by_guardrail"] is True
    assert "PromptInjectionDetected" in result["incident_log"][0]


def test_guarded_appends_to_existing_incident_log_instead_of_replacing_it():
    def node(_state):
        raise ComplianceViolation("rude tone")

    result = _guarded(node)({"incident_log": ["earlier_incident"]})

    assert result["incident_log"] == ["earlier_incident", "ComplianceViolation: rude tone"]


def test_guarded_passes_through_normal_results_unchanged():
    def node(state):
        return {"draft_answer": "all good", "echoed": state["foo"]}

    result = _guarded(node)({"foo": "bar", "incident_log": []})

    assert result == {"draft_answer": "all good", "echoed": "bar"}


def test_guarded_does_not_swallow_other_exceptions():
    def node(_state):
        raise ValueError("a real bug, not a guardrail")

    with pytest.raises(ValueError):
        _guarded(node)({"incident_log": []})
