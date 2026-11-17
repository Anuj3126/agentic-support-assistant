"""Human-handoff node. Reached two different ways, which get two different responses:

- A genuine handoff (negative sentiment or an unsupported department) — a real person asked
  a real question we can't fully help with here. We hand off to a human and collect contact
  details (the CLI/frontend does that right after this node runs, as a simple one-shot side
  flow rather than part of the graph).
- A guardrail breach (prompt injection or a compliance violation) — not someone to call back,
  just a request we won't process. A flat refusal, no contact form, no callback promise.
"""

from langchain_core.messages import AIMessage

from shopunow.schemas.models import EscalationForm
from shopunow.state import SupportState

DEFAULT_REASON = "Let's get a human teammate to help you with this one."
GUARDRAIL_REFUSAL = (
    "I'm not able to process that request. If you have a ShopUNow support question, "
    "I'm happy to help with that instead."
)


def escalation_node(state: SupportState) -> dict:
    if state.get("blocked_by_guardrail"):
        return {"final_response": GUARDRAIL_REFUSAL, "messages": [AIMessage(GUARDRAIL_REFUSAL)]}

    reason = state.get("escalation_reason") or DEFAULT_REASON
    message = (
        f"{reason} A ShopUNow support teammate will reach out to you shortly. "
        "In the meantime, could you share your name, phone number, and email so they can "
        "follow up?"
    )
    return {"final_response": message, "messages": [AIMessage(message)]}


def format_escalation_confirmation(form: EscalationForm) -> str:
    return (
        "Thanks — we've got your details:\n"
        f"- Name: {form.name or 'n/a'}\n"
        f"- Phone: {form.phone or 'n/a'}\n"
        f"- Email: {form.email or 'n/a'}\n\n"
        "Someone from our team will reach out to you soon."
    )
