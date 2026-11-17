"""Formats a resolved turn into a clean Markdown block for the CLI (and later, Streamlit)."""


def format_turn_report(state: dict) -> str:
    lines = [
        "## Support Turn Summary",
        "",
        f"- **Department:** {state.get('department') or 'n/a'}",
        f"- **Sentiment:** {state.get('sentiment') or 'n/a'} ({state.get('tone') or 'n/a'})",
        f"- **Escalated:** {'yes' if state.get('needs_escalation') else 'no'}",
    ]
    if state.get("used_web_search"):
        lines.append("- **Source:** web search fallback (knowledge base had nothing relevant)")
    if state.get("retrieved_docs"):
        lines.append(f"- **Docs retrieved:** {len(state['retrieved_docs'])}")
    lines += ["", "**Response:**", "", state.get("final_response") or ""]
    return "\n".join(lines)
