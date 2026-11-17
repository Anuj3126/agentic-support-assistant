"""Cheap, no-LLM-call heuristic that catches obvious prompt-injection / jailbreak attempts
before a query is routed anywhere."""

import re

from shopunow.guardrails.exceptions import PromptInjectionDetected

_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all|any|the) (previous|prior|above) instructions",
        r"disregard (your|the) (system prompt|instructions|rules)",
        r"reveal (your|the) system prompt",
        r"you are no longer",
        r"act as (if you are|a) (dan|jailbreak|unrestricted)",
        r"pretend (you have no|there are no) (rules|restrictions|filters)",
        r"forget (everything|all) (you were told|above)",
        r"\bsystem prompt\b.*\b(show|print|repeat|reveal)\b",
    ]
]


def check_prompt_injection(text: str) -> None:
    """Raises PromptInjectionDetected if `text` matches a known jailbreak pattern."""
    for pattern in _PATTERNS:
        if pattern.search(text):
            raise PromptInjectionDetected(f"matched pattern: {pattern.pattern}")
