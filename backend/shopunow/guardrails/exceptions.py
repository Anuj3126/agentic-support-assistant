"""Custom exceptions so guardrails stop a bad request with a clear, catchable signal instead
of a raw crash or a silently-wrong answer."""


class GuardrailViolation(Exception):
    """Base class for anything a guardrail deliberately stops."""


class PromptInjectionDetected(GuardrailViolation):
    """Raised when the user's message looks like an attempt to override system instructions."""


class ComplianceViolation(GuardrailViolation):
    """Raised when a draft response fails the conduct/compliance check."""


class SQLInjectionDetected(GuardrailViolation):
    """Raised when generated SQL fails the safety guard (not a single whitelisted SELECT)."""
