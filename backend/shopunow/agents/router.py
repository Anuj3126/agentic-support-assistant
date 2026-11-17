"""Second node: checks for prompt injection, classifies the department, and decides whether
this turn needs a human instead of an AI workflow.

Per the project's compulsory rule: negative sentiment OR an unsupported department always
escalates, regardless of which department the query would otherwise match. A detected prompt
injection raises PromptInjectionDetected — this function doesn't catch it itself; the graph
builder wraps every node with a single shared guardrail handler that stops the flow and routes
to escalation, so individual agents don't each need their own try/except for this.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from shopunow.config import settings
from shopunow.data.departments import DEPARTMENTS
from shopunow.guardrails.prompt_injection import check_prompt_injection
from shopunow.schemas.models import RouteDecision
from shopunow.state import SupportState

ROUTER_PROMPT = """You are the routing agent for ShopUNow's support assistant. Your job is to
read one incoming message from either an internal employee or an external customer, and decide
which department is best equipped to help them.

Here are the departments you can route to, with who they serve and what they handle:
{department_list}

Read the message carefully and decide:
1. If it clearly matches one of the departments above, set department to that department's id
   and is_supported to true.
2. If it's a reasonable ShopUNow- or retail-related question that doesn't fit any single
   department (for example "what products do you sell" or "what are your store hours"), set
   department to "general" and is_supported to true — a general-purpose agent will handle it.
3. If it doesn't fit either of those (sales pitches, unrelated small talk, questions about
   other companies, or anything with nothing to do with ShopUNow), set department to null and
   is_supported to false, so it can be handed off to a human.

Explain your reasoning briefly before you decide.

Message to classify: {message}"""


def router_node(state: SupportState) -> dict:
    last_message = state["messages"][-1].content
    check_prompt_injection(last_message)

    department_list = "\n".join(f"- {d.id}: {d.label} — {d.description}" for d in DEPARTMENTS.values())
    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(
        ROUTER_PROMPT.format(department_list=department_list, message=last_message)
    )

    negative_sentiment = state.get("sentiment") == "negative"
    needs_escalation = negative_sentiment or not decision.is_supported

    escalation_reason = None
    if negative_sentiment:
        escalation_reason = "We noticed this query needs extra care — connecting you with a human."
    elif not decision.is_supported:
        escalation_reason = "This doesn't fall under a department we can help with directly."

    return {
        "department": decision.department,
        "is_supported": decision.is_supported,
        "needs_escalation": needs_escalation,
        "escalation_reason": escalation_reason,
    }
