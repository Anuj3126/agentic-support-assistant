from shopunow.utils.markdown_report import format_turn_report


def test_report_includes_department_sentiment_and_response():
    report = format_turn_report(
        {
            "department": "billing",
            "sentiment": "neutral",
            "tone": "calm",
            "needs_escalation": False,
            "retrieved_docs": [{"text": "doc1"}, {"text": "doc2"}],
            "final_response": "Your refund will land in 5-7 business days.",
        }
    )

    assert "billing" in report
    assert "neutral" in report and "calm" in report
    assert "Escalated:** no" in report
    assert "Docs retrieved:** 2" in report
    assert "Your refund will land in 5-7 business days." in report


def test_report_handles_missing_fields_gracefully():
    report = format_turn_report({})

    assert "n/a" in report
    assert "Escalated:** no" in report
