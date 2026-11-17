"""ShopUNow Support Assistant — Streamlit frontend.

A thin UI over the exact same `shopunow` backend the CLI (shopunow/main.py) uses — no API
layer in between, just a direct import, so the backend keeps working standalone from a
terminal too.

Run (from the backend's venv, so `shopunow` is already installed/importable):
    cd backend && source .venv/bin/activate && pip install streamlit
    streamlit run ../frontend/app.py
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import streamlit as st

from shopunow.agents.escalation_agent import format_escalation_confirmation
from shopunow.data.departments import DEPARTMENTS
from shopunow.db.schema import LOGIN_NAMES
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


LOGIN_OPTIONS = [f"{n['name']} ({n['kind']})" for n in LOGIN_NAMES]


def _name_from_option(option: str) -> str:
    return option.rsplit(" (", 1)[0]


if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, content) tuples, for redrawing the thread
    st.session_state.escalation_pending = False
    st.session_state.escalation_department = None  # may legitimately be None (unsupported query)
    st.session_state.user_id = _name_from_option(LOGIN_OPTIONS[0])

with st.sidebar:
    st.title("🛍️ ShopUNow")
    st.caption("Agentic support assistant — capstone demo")
    chosen_option = st.selectbox("Logged in as", LOGIN_OPTIONS)
    chosen_name = _name_from_option(chosen_option)
    if chosen_name != st.session_state.user_id:
        # switching identities mid-session starts a fresh display — each login has its own
        # checkpointed conversation (thread_id), so the old messages belong to someone else
        st.session_state.user_id = chosen_name
        st.session_state.chat = []
        st.session_state.escalation_pending = False
        st.session_state.escalation_department = None
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
    thread_id = make_thread_id(st.session_state.user_id)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(fresh_turn_input(user_text, st.session_state.user_id), config=config)

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
