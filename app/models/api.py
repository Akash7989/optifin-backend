"""Request/response contracts for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.models.profile import UserProfileSchema


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class IngestRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class IngestResponse(BaseModel):
    status: Literal["incomplete", "ready"]
    questions: list[str]
    profile: Optional[UserProfileSchema]
    extracted: dict[str, Any] = Field(
        default_factory=dict, description="Values stated so far, before validation (for live chips).")


class ExplainRequest(BaseModel):
    profile: UserProfileSchema
    results: dict[str, Any]


class ExplainResponse(BaseModel):
    narrative_explanation: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
