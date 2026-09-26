"""User financial profile schemas.

UserProfileSchema is the validated, complete profile the engines consume.
ExtractedProfile is what the LLM is asked to emit: every field is nullable so the model
can say "not mentioned" instead of inventing a value.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskTolerance = Literal["conservative", "moderate", "aggressive"]


class UserProfileSchema(BaseModel):
    age: int = Field(ge=18, le=70)
    net_monthly_income: float = Field(gt=0)
    existing_emis: float = Field(ge=0)
    liquid_assets: float = Field(ge=0)
    outstanding_liabilities: float = Field(ge=0)
    current_monthly_sip: float = Field(ge=0)
    current_life_cover: float = Field(ge=0)
    target_goal_cost_today: float = Field(gt=0)
    goal_horizon_years: int = Field(ge=1, le=40)
    risk_tolerance: RiskTolerance


_AMOUNT = "Amount in rupees as a plain number (convert lakh/crore words to digits). null if not stated."


class ExtractedProfile(BaseModel):
    age: Optional[int] = Field(None, description="Current age in years. null if not stated.")
    net_monthly_income: Optional[float] = Field(
        None, description="Take-home pay per month, only if the user states a monthly figure. " + _AMOUNT)
    net_annual_income: Optional[float] = Field(
        None, description="Take-home pay per year, only if the user states an annual figure "
                          "(e.g. '18 LPA'). Do not divide it into a monthly figure. " + _AMOUNT)
    existing_emis: Optional[float] = Field(
        None, description="Total of current monthly loan EMIs. 0 if the user says they have no loans. " + _AMOUNT)
    liquid_assets: Optional[float] = Field(
        None, description="Savings, FDs and mutual funds that can be withdrawn. 0 if the user says none. " + _AMOUNT)
    outstanding_liabilities: Optional[float] = Field(
        None, description="Total outstanding loan principal. 0 if the user says no debts. " + _AMOUNT)
    current_monthly_sip: Optional[float] = Field(
        None, description="Current monthly SIP investment. 0 if the user says none. " + _AMOUNT)
    current_life_cover: Optional[float] = Field(
        None, description="Total sum assured of existing term/life insurance. 0 if the user says none. " + _AMOUNT)
    target_goal_cost_today: Optional[float] = Field(
        None, description="Price of the goal (e.g. home) in today's money. " + _AMOUNT)
    goal_horizon_years: Optional[int] = Field(
        None, description="Years until the goal, only if stated as a number of years. null otherwise.")
    goal_target_year: Optional[int] = Field(
        None, description="Calendar year of the goal, only if the user names a year (e.g. 2031). null otherwise.")
    risk_tolerance: Optional[RiskTolerance] = Field(
        None, description="Investment risk appetite, only if the user expresses one. null otherwise.")
