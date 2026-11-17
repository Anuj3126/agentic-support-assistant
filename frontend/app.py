"""ShopUNow Support Assistant — Streamlit frontend.

A thin UI over the exact same `shopunow` backend the CLI (shopunow/main.py) uses — no API
layer in between, just a direct import, so the backend keeps working standalone from a
terminal too.

Run (from the backend's venv, so `shopunow` is already installed/importable):
    cd backend && source .venv/bin/activate && pip install streamlit
    streamlit run ../frontend/app.py
"""

import json
import sys
import uuid
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st
from langchain_core.messages import HumanMessage

from shopunow.agents.escalation_agent import format_escalation_confirmation
from shopunow.data.departments import DEPARTMENTS
from shopunow.graph.build import build_graph
from shopunow.main import fresh_turn_input, save_escalation
from shopunow.memory.checkpointer import make_thread_id
from shopunow.schemas.models import EscalationForm
from shopunow.utils.markdown_report import format_turn_report

st.set_page_config(page_title="ShopUNow Support Assistant", page_icon="🛍️")


@st.cache_resource
def get_graph():
    # Built once per server process (not per rerun) — the checkpointer that gives every
    # user their own session memory lives on this one shared graph object.
    return build_graph()


if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, content) tuples, for redrawing the thread
    st.session_state.escalation_pending = False
    st.session_state.escalation_department = None  # may legitimately be None (unsupported query)
    # Unique per browser session by default — the compiled graph (and its MemorySaver) is
    # shared process-wide across every session via st.cache_resource, so two anonymous users
    # both left on a fixed default like "guest" would collide on the same thread_id and see
    # each other's conversation. Typing a real, shared id is still how one person picks up
    # the same conversation again across visits/devices.
    st.session_state.user_id = f"guest-{uuid.uuid4().hex[:8]}"

with st.sidebar:
    st.title("🛍️ ShopUNow")
    st.caption("Agentic support assistant — capstone demo")
    st.session_state.user_id = st.text_input("Your user id", value=st.session_state.user_id)
    st.divider()
    st.markdown("**Internal**")
    for d in DEPARTMENTS.values():
        if d.audience == "internal":
            st.markdown(f"- {d.label}")
    st.markdown("**External**")
    for d in DEPARTMENTS.values():
        if d.audience == "external":
            st.markdown(f"- {d.label}")
    st.markdown("- General *(anything else in scope)*")

st.title("ShopUNow Support Assistant")
st.caption("Ask an HR/IT/Facilities question as an employee, or a Billing/Shipping/Returns question as a customer.")

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_text = st.chat_input("Type your question...")

if user_text:
    st.session_state.chat.append(("user", user_text))
    with st.chat_message("user"):
        st.markdown(user_text)

    graph = get_graph()
    thread_id = make_thread_id(st.session_state.user_id, "default")
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        fresh_turn_input(user_text, st.session_state.user_id, "default"), config=config
    )

    response_text = result.get("final_response") or "(no response)"
    with st.chat_message("assistant"):
        st.markdown(response_text)
        with st.expander("Details"):
            st.markdown(format_turn_report(result))

    st.session_state.chat.append(("assistant", response_text))
    st.session_state.escalation_pending = bool(
        result.get("needs_escalation") and not result.get("blocked_by_guardrail")
    )
    st.session_state.escalation_department = result.get("department")

if st.session_state.escalation_pending:
    with st.form("escalation_form", clear_on_submit=True):
        st.markdown("**We'll connect you with a human teammate — leave your details:**")
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        if st.form_submit_button("Submit"):
            form = EscalationForm(name=name or None, phone=phone or None, email=email or None)
            save_escalation(st.session_state.escalation_department, form)
            confirmation = format_escalation_confirmation(form)
            st.session_state.chat.append(("assistant", confirmation))
            st.session_state.escalation_pending = False
            st.rerun()
