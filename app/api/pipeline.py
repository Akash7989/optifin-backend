"""Orchestration: profile -> engines -> product matching -> result vector.

The loan engine runs first because its FOIR/LTV result sets the downpayment the goal
simulation must reach; the insurance and Monte Carlo engines then run in parallel threads.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from app.adapters.macro_adapter import get_real_estate_inflation, get_repo_rate
from app.adapters.product_matrix import (
    ILLUSTRATIVE,
    TERM_AGE_COHORTS,
    match_loan_offers,
    quote_term_premiums,
)
from app.config import GSEC_10Y_YIELD
from app.engines.goal_simulator import (
    DEFAULT_DOWNPAYMENT_RATIO,
    DEFAULT_PATHS,
    DEFAULT_TARGET_SUCCESS,
    future_target_cost,
    run_monte_carlo_simulation,
)
from app.engines.insurance_engine import DEFAULT_WAGE_GROWTH, calculate_net_insurance_need
from app.engines.loan_engine import calculate_loan_capacity
from app.models.api import ChatMessage
from app.models.profile import UserProfileSchema

RISK_EQUITY_ALLOCATION = {"conservative": 0.30, "moderate": 0.60, "aggressive": 0.80}
ASSUMED_CREDIT_SCORE = 750  # the profile does not capture a credit score yet
NO_MATCH_SPREAD = 0.03  # over repo, used when no catalog lender matches
SIMULATION_SEED = 42  # same profile -> same projections across requests


def transcript_from_messages(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in messages)


async def run_simulation(profile: UserProfileSchema) -> dict[str, Any]:
    repo_rate = get_repo_rate()
    real_estate_inflation = get_real_estate_inflation()
    equity_ratio = RISK_EQUITY_ALLOCATION[profile.risk_tolerance]
    notes: list[str] = []

    loan_offers = match_loan_offers(
        "home_loan", profile.net_monthly_income, ASSUMED_CREDIT_SCORE, repo_rate=repo_rate
    )
    if loan_offers:
        home_loan_rate = loan_offers[0].interest_rate
    else:
        home_loan_rate = repo_rate + NO_MATCH_SPREAD
        notes.append("No catalog lender matched the income; loan priced at repo + 3%.")

    property_value = future_target_cost(
        profile.target_goal_cost_today, profile.goal_horizon_years, real_estate_inflation
    )
    loan = calculate_loan_capacity(
        net_monthly_income=profile.net_monthly_income,
        existing_emis=profile.existing_emis,
        property_value=property_value,
        annual_rate=home_loan_rate,
        borrower_age_at_start=profile.age + profile.goal_horizon_years,
        years_until_purchase=profile.goal_horizon_years,
        income_growth=DEFAULT_WAGE_GROWTH,
    )
    downpayment_ratio = min(1.0, max(DEFAULT_DOWNPAYMENT_RATIO, loan.required_downpayment_ratio))

    insurance, goal = await asyncio.gather(
        asyncio.to_thread(
            calculate_net_insurance_need,
            age=profile.age,
            annual_income=profile.net_monthly_income * 12,
            outstanding_liabilities=profile.outstanding_liabilities,
            future_goal_commitments=downpayment_ratio * profile.target_goal_cost_today,
            liquid_assets=profile.liquid_assets,
            current_term_cover=profile.current_life_cover,
        ),
        asyncio.to_thread(
            run_monte_carlo_simulation,
            current_cost=profile.target_goal_cost_today,
            horizon_years=profile.goal_horizon_years,
            monthly_sip=profile.current_monthly_sip,
            equity_ratio=equity_ratio,
            initial_wealth=profile.liquid_assets,
            inflation=real_estate_inflation,
            downpayment_ratio=downpayment_ratio,
            seed=SIMULATION_SEED,
        ),
    )

    term_quotes = []
    if insurance.net_required_cover <= 0:
        notes.append("Existing cover and liquid assets meet the protection need; no term quotes.")
    elif not TERM_AGE_COHORTS[0] <= profile.age <= TERM_AGE_COHORTS[-1]:
        notes.append(f"Term premium tables cover ages {TERM_AGE_COHORTS[0]}-{TERM_AGE_COHORTS[-1]}.")
    else:
        term_quotes = quote_term_premiums(profile.age, insurance.net_required_cover)

    return {
        "assumptions": {
            "repo_rate": repo_rate,
            "discount_rate": GSEC_10Y_YIELD,
            "real_estate_inflation": real_estate_inflation,
            "income_growth": DEFAULT_WAGE_GROWTH,
            "assumed_credit_score": ASSUMED_CREDIT_SCORE,
            "equity_allocation": equity_ratio,
            "home_loan_rate": home_loan_rate,
            "downpayment_ratio": downpayment_ratio,
            "simulation_paths": DEFAULT_PATHS,
            "simulation_seed": SIMULATION_SEED,
            "target_success_probability": DEFAULT_TARGET_SUCCESS * 100,
        },
        "loan": asdict(loan),
        "insurance": asdict(insurance),
        "goal": asdict(goal),
        "downpayment_funding": {
            "required_downpayment": goal.target_downpayment,
            "liquid_assets_today": profile.liquid_assets,
            "median_projected_corpus": goal.wealth_p50,
            "median_shortfall": max(0.0, goal.target_downpayment - goal.wealth_p50),
        },
        "products": {
            "illustrative": ILLUSTRATIVE,
            "home_loans": [asdict(o) for o in loan_offers],
            "term_insurance": [asdict(q) for q in term_quotes],
        },
        "notes": notes,
    }
