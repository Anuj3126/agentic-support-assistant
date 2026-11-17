"""Critiques the draft answer before it reaches the user; can send it back for one revision."""

from langchain_google_genai import ChatGoogleGenerativeAI

from shopunow.config import settings
from shopunow.schemas.models import ReflectionCritique
from shopunow.state import SupportState

REFLECTION_PROMPT = """You are the reflection agent for ShopUNow's support assistant. Another
agent has already drafted a response — your job is to critique it honestly before it reaches
the user, the way a careful editor would, and decide whether it needs another pass.

User's original question: {query}
Draft answer to review: {draft}

Check it against these three things:
1. Does it actually answer what the user asked, directly and completely?
2. Is it consistent with the information it was given, without inventing details it couldn't
   have known?
3. Is the tone warm and appropriately helpful, rather than curt, robotic, or evasive?

If the draft holds up on all three, set needs_revision to false and leave feedback brief. Only
set needs_revision to true for a genuine problem worth fixing — a wrong or missing answer, a
fabricated detail, or a tone that would land badly — not for a minor wording preference. When
you do ask for a revision, make the feedback specific enough that the next draft can actually
act on it."""


def reflection_node(state: SupportState) -> dict:
    query = state["messages"][-1].content
    draft = state.get("draft_answer") or ""

    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    critic = llm.with_structured_output(ReflectionCritique)
    critique: ReflectionCritique = critic.invoke(REFLECTION_PROMPT.format(query=query, draft=draft))

    reflection_count = state.get("reflection_count", 0) + 1
    at_limit = reflection_count >= settings.max_reflections

    return {
        "reflection_count": reflection_count,
        "reflection_feedback": critique.feedback if critique.needs_revision and not at_limit else None,
    }
