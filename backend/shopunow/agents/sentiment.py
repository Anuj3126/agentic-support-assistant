"""First node in the graph: reads the user's sentiment and tone before anything else runs."""

from langchain_google_genai import ChatGoogleGenerativeAI

from shopunow.config import settings
from shopunow.schemas.models import SentimentResult
from shopunow.state import SupportState

SENTIMENT_PROMPT = """You are the sentiment-analysis agent for ShopUNow's support assistant.
Before any other agent sees this message, you read it and judge how the person is feeling —
this decides whether they get routed to an AI workflow or straight to a human.

Read the message below, written by either an internal employee or an external customer, and
classify its overall sentiment. Most support messages describe some kind of problem — a late
order, a billing error, a broken laptop — and that alone does NOT make them negative; reporting
an issue is the normal, expected shape of a support request. What decides sentiment is whether
the person sounds upset, not whether something has gone wrong.

- negative: there's real frustration, anger, or distress in how it's written — repeated
  unresolved issues, feeling let down or ignored, sarcasm, or explicitly saying they're unhappy.
  Example: "This is the third time this has happened and nobody is helping me!"
- neutral: a plain, matter-of-fact report of a problem or a routine question, with no sign of
  distress. Example: "My order hasn't shipped yet, what's going on?" or "I was charged twice,
  can you help?" — these are just normal support requests, not complaints about being
  mistreated, even though something did go wrong.
- positive: calm, satisfied, or appreciative, with no problem being reported at all.

Also pick the one tone that best fits:
- calm: even-keeled, no particular emotional charge
- frustrated: annoyed or let down, whether or not sentiment is negative overall
- urgent: in a hurry, treating this as time-sensitive
- appreciative: expressing thanks or satisfaction

Message: {message}"""


def sentiment_node(state: SupportState) -> dict:
    last_message = state["messages"][-1].content
    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    result: SentimentResult = llm.with_structured_output(SentimentResult).invoke(
        SENTIMENT_PROMPT.format(message=last_message)
    )
    return {"sentiment": result.sentiment, "tone": result.tone}
