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
supplies, books, and toys, both online and through a small network of physical stores across
North America, Europe, and Asia-Pacific. Store hours are 9 AM-8 PM local time, seven days a
week, at all physical locations.

ShopUNow Rewards is our free loyalty program — customers earn points on every purchase,
redeemable for discounts, and can join from "My Account > Rewards." Gift wrapping is available
at checkout for a small fee, and digital gift cards (no expiry) can be bought and redeemed
directly on the website. A general clothing size guide, in both metric and imperial units, is
available on every clothing product page.

Most products can be returned within 7 days of delivery (the Returns department handles
specifics), and orders ship to over 40 countries via regional courier partners (the Shipping
department handles tracking, delays, and coverage questions). ShopUNow runs seasonal sales
during major shopping periods, and has a free mobile app (iOS and Android) with app-only deals."""

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
MAX_TOOL_ROUNDS = 4  # a backstop against a runaway loop, not an expected depth


def general_node(state: SupportState) -> dict:
    history = state["messages"][-(HISTORY_WINDOW + 1) : -1]
    query = state["messages"][-1].content
    llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.google_api_key)
    messages = [SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(query)]
    tool_llm = llm.bind_tools([web_search])

    # A bounded loop, not a single round — tools must stay bound on every call, not just the
    # first, or a model that wants a second search after seeing the first result gets cut off
    # into an empty response instead of being able to continue.
    docs, used_web_search = [], False
    response = tool_llm.invoke(messages)
    rounds = 0
    while response.tool_calls and rounds < MAX_TOOL_ROUNDS:
        messages.append(response)
        for call in response.tool_calls:
            results = web_search.invoke(call["args"])
            docs.extend(to_docs(results, department="general"))
            used_web_search = True
            messages.append(ToolMessage(content=json.dumps(results), tool_call_id=call["id"]))
        response = tool_llm.invoke(messages)
        rounds += 1

    return {"retrieved_docs": docs, "used_web_search": used_web_search, "draft_answer": response.text}
