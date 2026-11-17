"""Terminal entrypoint — runs the assistant standalone, independent of any frontend.

Run: `python -m shopunow.main`
"""

import json

from langchain_core.messages import HumanMessage

from shopunow.agents.escalation_agent import format_escalation_confirmation
from shopunow.config import PACKAGE_DIR
from shopunow.db.schema import LOGIN_NAMES
from shopunow.graph.build import build_graph
from shopunow.memory.checkpointer import make_thread_id
from shopunow.schemas.models import EscalationForm
from shopunow.utils.markdown_report import format_turn_report

ESCALATIONS_LOG = PACKAGE_DIR / "escalations.jsonl"


def fresh_turn_input(user_text: str, name: str) -> dict:
    """Resets per-turn scratch fields while leaving message history / incident_log alone,
    so stale routing/reflection state from a prior turn can't leak into this one."""
    return {
        "messages": [HumanMessage(user_text)],
        "user_id": name,
        "sentiment": None,
        "tone": None,
        "department": None,
        "is_supported": False,
        "needs_escalation": False,
        "escalation_reason": None,
        "blocked_by_guardrail": False,
        "retrieved_docs": [],
        "used_web_search": False,
        "draft_answer": None,
        "reflection_count": 0,
        "reflection_feedback": None,
        "final_response": None,
    }


def save_escalation(department: str | None, form: EscalationForm) -> None:
    """Shared with the Streamlit frontend — the CLI collects the form via input(), the
    frontend via st.form, but both end up calling this to append the same log file."""
    with ESCALATIONS_LOG.open("a") as f:
        f.write(json.dumps({"department": department, **form.model_dump()}) + "\n")


def collect_escalation_form(department: str | None) -> None:
    print("\n(You can press Enter to skip any field.)")
    name = input("Name: ").strip() or None
    phone = input("Phone: ").strip() or None
    email = input("Email: ").strip() or None
    form = EscalationForm(name=name, phone=phone, email=email)

    print("\n" + format_escalation_confirmation(form) + "\n")
    save_escalation(department, form)


def choose_login() -> str:
    print("Who are you logging in as?")
    for i, option in enumerate(LOGIN_NAMES, start=1):
        print(f"  {i}. {option['name']} ({option['kind']})")

    while True:
        choice = input("Enter a number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(LOGIN_NAMES):
            return LOGIN_NAMES[int(choice) - 1]["name"]
        print("Please enter a number from the list above.")


def main() -> None:
    print("ShopUNow Support Assistant (type 'exit' to quit)\n")
    name = choose_login()
    thread_id = make_thread_id(name)
    config = {"configurable": {"thread_id": thread_id}}

    graph = build_graph()

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        result = graph.invoke(fresh_turn_input(user_text, name), config=config)
        print("\n" + format_turn_report(result))

        if result.get("needs_escalation") and not result.get("blocked_by_guardrail"):
            collect_escalation_form(result.get("department"))


if __name__ == "__main__":
    main()
