import json

from shopunow import main as main_module


def test_collect_escalation_form_writes_details_and_confirms(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(main_module, "ESCALATIONS_LOG", log_path)

    answers = iter(["Asha Rao", "9876543210", "asha@example.com"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    main_module.collect_escalation_form("billing")

    printed = capsys.readouterr().out
    assert "Asha Rao" in printed
    assert "reach out to you soon" in printed

    logged = json.loads(log_path.read_text().strip())
    assert logged == {
        "department": "billing",
        "name": "Asha Rao",
        "phone": "9876543210",
        "email": "asha@example.com",
    }


def test_collect_escalation_form_allows_skipping_fields(tmp_path, monkeypatch):
    log_path = tmp_path / "escalations.jsonl"
    monkeypatch.setattr(main_module, "ESCALATIONS_LOG", log_path)

    answers = iter(["", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    main_module.collect_escalation_form(None)

    logged = json.loads(log_path.read_text().strip())
    assert logged == {"department": None, "name": None, "phone": None, "email": None}
