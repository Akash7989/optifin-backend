"""Structured extraction: chat transcript -> validated UserProfileSchema.

The LLM only transcribes values the user stated. Anything it leaves null, or that fails
validation, becomes a clarifying question. The two derivations the user may leave implicit
(annual -> monthly income, target year -> horizon) are done here in Python, not by the model.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from google import genai
from google.genai import types
from pydantic import ValidationError

from app.agent.client import generate_content, get_client
from app.models.profile import ExtractedProfile, UserProfileSchema

__all__ = ["UserProfileSchema", "extract_financial_profile", "extract_profile_state", "build_profile"]

SYSTEM_INSTRUCTION = """You extract a personal finance profile from a chat between a user and an assistant.
Rules:
- Record only values the user explicitly stated. Never guess, infer or assume a value; use null instead.
- Do not perform any arithmetic: no sums, averages, divisions or conversions between time periods.
  If the user gives an annual income, fill net_annual_income and leave net_monthly_income null.
  If the user gives several separate EMIs or loans without a total, leave that field null.
- Converting number words is allowed: "1.2 lakh" = 120000, "1.5 crore" = 15000000, "25k" = 25000.
- Use 0 only when the user clearly says they have none (e.g. "no loans", "no insurance").
- If the user corrects an earlier value, use the latest one."""

# Ordered by importance: critical fields first.
CLARIFYING_QUESTIONS: dict[str, str] = {
    "age": "How old are you?",
    "net_monthly_income": "What is your take-home (post-tax) income per month?",
    "target_goal_cost_today": "What would your goal cost if you bought it today?",
    "goal_horizon_years": "In how many years do you want to reach this goal?",
    "risk_tolerance": ("How would you describe your investment risk appetite: conservative, "
                       "moderate or aggressive?"),
    "existing_emis": "What is the total of your current monthly EMIs? Say 0 if you have no loans.",
    "outstanding_liabilities": ("How much loan principal do you still owe in total? "
                                "Say 0 if you have no debts."),
    "liquid_assets": ("How much do you hold in savings, FDs and mutual funds that you could "
                      "withdraw if needed?"),
    "current_monthly_sip": "How much do you currently invest each month through SIPs?",
    "current_life_cover": ("What is the total cover of any term or life insurance you hold? "
                           "Say 0 if you have none."),
}

INVALID_VALUE_QUESTIONS: dict[str, str] = {
    "age": "Could you confirm your age? We support planning for ages 18 to 70.",
    "goal_horizon_years": "Could you confirm how many years away your goal is (between 1 and 40)?",
}


def build_profile(
    extracted: ExtractedProfile, today: date | None = None
) -> tuple[Optional[UserProfileSchema], list[str]]:
    """Validate an extraction; return the profile, or None plus clarifying questions."""
    today = today or date.today()
    values = extracted.model_dump(exclude={"net_annual_income", "goal_target_year"})

    if values["net_monthly_income"] is None and extracted.net_annual_income is not None:
        values["net_monthly_income"] = extracted.net_annual_income / 12
    if values["goal_horizon_years"] is None and extracted.goal_target_year is not None:
        values["goal_horizon_years"] = extracted.goal_target_year - today.year

    problems = {name for name, value in values.items() if value is None}
    try:
        return UserProfileSchema.model_validate(values), []
    except ValidationError as exc:
        invalid = {str(err["loc"][0]) for err in exc.errors()} - problems

    questions = [q for field, q in CLARIFYING_QUESTIONS.items() if field in problems]
    for field in CLARIFYING_QUESTIONS:
        if field in invalid:
            questions.append(INVALID_VALUE_QUESTIONS.get(
                field, f"Could you double-check the value you gave for {field.replace('_', ' ')}?"))
    return None, questions


def extract_profile_state(
    chat_history: str,
    client: genai.Client | None = None,
    today: date | None = None,
) -> tuple[ExtractedProfile, Optional[UserProfileSchema], list[str]]:
    """Raw extraction plus the validated profile (or clarifying questions)."""
    client = client or get_client()
    response = generate_content(
        client,
        contents=chat_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ExtractedProfile,
            temperature=0.0,
        ),
    )
    extracted = response.parsed
    if not isinstance(extracted, ExtractedProfile):
        extracted = ExtractedProfile.model_validate_json(response.text or "{}")
    return (extracted, *build_profile(extracted, today))


def extract_financial_profile(
    chat_history: str,
    client: genai.Client | None = None,
    today: date | None = None,
) -> tuple[Optional[UserProfileSchema], list[str]]:
    """Extract a profile from the transcript; returns (profile, []) or (None, questions)."""
    _, profile, questions = extract_profile_state(chat_history, client, today)
    return profile, questions
