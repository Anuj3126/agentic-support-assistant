"""Pydantic models used as structured-output contracts between graph nodes. Kept to the
minimum the graph actually needs — one per agent that returns structured data."""

from typing import Literal, Optional

from pydantic import BaseModel


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    tone: Literal["calm", "frustrated", "urgent", "appreciative"]


class RouteDecision(BaseModel):
    department: Optional[str] = None  # best-matching department id, or null if none fit
    is_supported: bool
    reasoning: str


class ReflectionCritique(BaseModel):
    needs_revision: bool
    feedback: str


class ComplianceVerdict(BaseModel):
    is_compliant: bool
    reason: str


class EscalationForm(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
