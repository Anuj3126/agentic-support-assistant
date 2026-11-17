"""Handles two kinds of queries that don't belong to a specific department:

1. Questions about ShopUNow itself — what it sells, store hours, how it operates. ShopUNow
   is a fictional company for this project, so a real web search for "ShopUNow" wouldn't
   turn up anything real — instead, those basics are just given to the agent directly as a
   company profile in its prompt, the same way a real support agent would already know them.
2. Broader retail/shopping questions that aren't about ShopUNow specifically, but that a
   helpful retail assistant could reasonably weigh in on (product care, gift ideas, general
   shopping guidance) — these genuinely benefit from the real internet, so the agent is given
   the web_search tool and decides for itself when it actually needs to use it.

Guarded either way to stay on-topic rather than answer as a general-purpose assistant.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from shopunow.config import settings
from shopunow.guardrails.conduct_policy import CONDUCT_POLICY
from shopunow.state import SupportState
from shopunow.tools.web_search_tool import to_docs, web_search

COMPANY_PROFILE = """ShopUNow is a retail company selling clothing, DIY (do it yourself)
supplies, books, and toys, both online and through a small number of physical stores. Store
hours are 9 AM-8 PM local time, seven days a week. Most products can be returned within 7 days
of delivery (the Returns department handles specifics), and orders ship via regional courier
partners (the Shipping department handles tracking and delays)."""

SYSTEM_PROMPT = f"""{CONDUCT_POLICY}

You are ShopUNow's general support agent. You step in for questions that don't belong to a
specific department (HR, IT, Facilities, Billing, Shipping, Returns), which come in two kinds:

1. Questions about ShopUNow itself. You already know the basics — they're given to you below
   as a company profile. Answer directly from that; there's no need to search for it.
2. Broader retail or shopping questions that aren't about ShopUNow specifically (product care,
   gift ideas, general shopping guidance). For these, use the web_search tool if your own
   knowledge isn't enough to give a good answer.

Company profile:
{COMPANY_PROFILE}

Guardrail — stay strictly in scope: only answer if the question is genuinely about ShopUNow or
retail/shopping more broadly. If it's unrelated (general trivia, other companies, personal or
financial advice, or anything outside what a retail company's assistant should be answering),
politely explain that you can only help with ShopUNow-related questions, and do not attempt to
answer the unrelated part at all."""


HISTORY_WINDOW = 6  # recent prior messages to include, for a sensible follow-up question


def general_node(state: SupportState) -> dict:
    history = state["messages"][-(HISTORY_WINDOW + 1) : -1]
    query = state["messages"][-1].content
    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    messages = [SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(query)]

    response = llm.bind_tools([web_search]).invoke(messages)
    if not response.tool_calls:
        return {"retrieved_docs": [], "used_web_search": False, "draft_answer": response.text}

    messages.append(response)
    docs = []
    for call in response.tool_calls:
        results = web_search.invoke(call["args"])
        docs.extend(to_docs(results, department="general"))
        messages.append(ToolMessage(content=json.dumps(results), tool_call_id=call["id"]))

    draft = llm.invoke(messages).text
    return {"retrieved_docs": docs, "used_web_search": True, "draft_answer": draft}
