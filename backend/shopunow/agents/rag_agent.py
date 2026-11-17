"""Builds one graph node per department (HR, IT, Facilities, Billing, Shipping, Returns) —
each retrieves from its own slice of the shared Chroma knowledge base, then generates an
answer. All six share this one implementation; they differ only in which department they're
configured for. Retrieval itself (including the web-search fallback) lives in
rag/retriever.py — this file turns retrieved context (plus, for Billing/Shipping/Returns, a
live database lookup) into an answer.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from shopunow.config import settings
from shopunow.data.departments import DEPARTMENTS
from shopunow.db.schema import DEPARTMENT_TABLES
from shopunow.guardrails.conduct_policy import CONDUCT_POLICY
from shopunow.rag.retriever import retrieve_with_fallback
from shopunow.state import SupportState
from shopunow.tools.sql_tool import make_sql_tool

SYSTEM_TEMPLATE = """{conduct_policy}

You are the {label} agent for ShopUNow, a retail company selling clothing, DIY (do it
yourself) supplies, books, and toys. {description}

Below is the most relevant reference material retrieved for this question. Read it carefully,
then answer the user's question clearly and completely, in your own words — don't just quote
the reference material verbatim. If the reference material genuinely doesn't contain an answer
to what they're asking, say so honestly rather than guessing or inventing a policy or detail
that isn't there, and suggest they can be connected with a human for anything you can't cover.

Reference material:
{context}
{sql_instructions}
Earlier turns in this conversation are included for context — use them if this question
follows up on something said before, but the reference material above is what answers this
specific question."""

SQL_INSTRUCTIONS_TEMPLATE = """
You also have a read-only database lookup tool for this department. Use it whenever the user
gives you something to look up by — their name, email, or an order number — instead of
answering a specific "did I buy X" / "what's the status of my order" question from general
policy alone. Only look up what's actually needed to answer the question, and never fabricate
order or account details that the lookup didn't actually return.
"""

# How many earlier messages to carry into the prompt for conversational continuity (e.g. "and
# what about my other order?"). Bounded so a long session doesn't grow the prompt unbounded.
HISTORY_WINDOW = 6


def _build_context(docs: list[dict]) -> str:
    return "\n\n".join(f"- {d['text']}" for d in docs) or "No relevant documents found."


def _revise_query(query: str, reflection_feedback: str | None) -> str:
    if not reflection_feedback:
        return query
    return f"{query}\n\n(Revision note from a prior review: {reflection_feedback})"


def make_department_agent(department_id: str):
    dept = DEPARTMENTS[department_id]
    sql_tool = make_sql_tool(department_id) if department_id in DEPARTMENT_TABLES else None

    def _agent(state: SupportState) -> dict:
        history = state["messages"][-(HISTORY_WINDOW + 1) : -1]
        query = _revise_query(state["messages"][-1].content, state.get("reflection_feedback"))
        docs, used_web_search = retrieve_with_fallback(query, department_id, settings.rag_top_k)

        system_message = SystemMessage(
            SYSTEM_TEMPLATE.format(
                conduct_policy=CONDUCT_POLICY,
                label=dept.label,
                description=dept.description,
                context=_build_context(docs),
                sql_instructions=SQL_INSTRUCTIONS_TEMPLATE if sql_tool is not None else "",
            )
        )
        llm = ChatGoogleGenerativeAI(
            model=settings.chat_model, google_api_key=settings.google_api_key
        )
        messages = [system_message, *history, HumanMessage(query)]

        if sql_tool is None:
            draft = llm.invoke(messages).text
            return {"retrieved_docs": docs, "used_web_search": used_web_search, "draft_answer": draft}

        response = llm.bind_tools([sql_tool]).invoke(messages)
        if not response.tool_calls:
            return {"retrieved_docs": docs, "used_web_search": used_web_search, "draft_answer": response.text}

        messages.append(response)
        for call in response.tool_calls:
            # execute_safe_select (inside the tool) raises SQLInjectionDetected on anything
            # unsafe — deliberately not caught here, so it propagates to the shared guardrail
            # wrapper in graph/build.py and stops the turn, same as any other guardrail breach.
            result = sql_tool.invoke(call["args"])
            messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

        draft = llm.invoke(messages).text
        return {"retrieved_docs": docs, "used_web_search": used_web_search, "draft_answer": draft}

    _agent.__name__ = f"{department_id}_agent"
    return _agent
