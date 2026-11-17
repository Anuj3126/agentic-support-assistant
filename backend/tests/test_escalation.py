from shopunow.agents.escalation_agent import GUARDRAIL_REFUSAL, escalation_node


def test_guardrail_breach_gets_a_flat_refusal_not_a_callback_offer():
    result = escalation_node({"blocked_by_guardrail": True})

    assert result["final_response"] == GUARDRAIL_REFUSAL
    assert result["messages"][0].content == GUARDRAIL_REFUSAL


def test_genuine_escalation_still_offers_a_human_callback():
    result = escalation_node(
        {"blocked_by_guardrail": False, "escalation_reason": "This doesn't fall under a department we can help with directly."}
    )

    assert "teammate will reach out" in result["final_response"]
    assert "name, phone number, and email" in result["final_response"]
