"""Life insurance needs engine: Human Life Value and term cover deficit.

Mortality comes from IALM 2012-14 Ult. (ages 18-70), stored in data/ialm_2012_14.json.
Ages are whole years (age last birthday); cash flows are annual, discounted at the
10-year G-Sec anchor rate by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import DATA_DIR, GSEC_10Y_YIELD

IALM_JSON_PATH = DATA_DIR / "ialm_2012_14.json"

DEFAULT_RETIREMENT_AGE = 60
DEFAULT_WAGE_GROWTH = 0.06
DEFAULT_PERSONAL_CONSUMPTION_RATIO = 0.30


@lru_cache(maxsize=None)
def load_mortality_rates(path: Path = IALM_JSON_PATH) -> dict[int, float]:
    """Return {age: q_x} from the IALM JSON dataset."""
    with open(path) as f:
        return {int(age): float(q) for age, q in json.load(f)["qx"].items()}


def survival_probability(age: int, years: int, qx: dict[int, float] | None = None) -> float:
    """t_p_x = product over k = 0..t-1 of (1 - q_{x+k})."""
    if years < 0:
        raise ValueError(f"years must be non-negative, got {years}")
    rates = load_mortality_rates() if qx is None else qx
    p = 1.0
    for k in range(years):
        if age + k not in rates:
            raise ValueError(f"no mortality rate for age {age + k}")
        p *= 1.0 - rates[age + k]
    return p


def human_life_value(
    current_age: int,
    annual_income: float,
    retirement_age: int = DEFAULT_RETIREMENT_AGE,
    wage_growth: float = DEFAULT_WAGE_GROWTH,
    personal_consumption_ratio: float = DEFAULT_PERSONAL_CONSUMPTION_RATIO,
    discount_rate: float = GSEC_10Y_YIELD,
    qx: dict[int, float] | None = None,
) -> float:
    """Actuarial present value of the family's share of future earnings.

    HLV = sum over t = 1..(R - x) of
          I * (1 + g)^t * (1 - c) * t_p_x / (1 + i)^t
    """
    if annual_income < 0:
        raise ValueError("annual_income must be non-negative")
    if not 0 <= personal_consumption_ratio <= 1:
        raise ValueError("personal_consumption_ratio must be between 0 and 1")
    if discount_rate <= -1 or wage_growth <= -1:
        raise ValueError("rates must be greater than -100%")

    hlv = 0.0
    for t in range(1, retirement_age - current_age + 1):
        family_income = annual_income * (1 + wage_growth) ** t * (1 - personal_consumption_ratio)
        hlv += family_income * survival_probability(current_age, t, qx) / (1 + discount_rate) ** t
    return hlv


@dataclass(frozen=True)
class CoverDeficit:
    human_life_value: float
    outstanding_liabilities: float
    future_goal_commitments: float
    gross_need: float
    liquid_assets: float
    current_term_cover: float
    available_resources: float
    net_required_cover: float
    surplus: float  # resources beyond the gross need; 0 when there is a deficit


def insurance_cover_deficit(
    human_life_value: float,
    outstanding_liabilities: float = 0.0,
    future_goal_commitments: float = 0.0,
    liquid_assets: float = 0.0,
    current_term_cover: float = 0.0,
) -> CoverDeficit:
    """Net Required Term Cover = max(0, Gross Need - Liquid Assets - Current Term Cover)."""
    amounts = {
        "human_life_value": human_life_value,
        "outstanding_liabilities": outstanding_liabilities,
        "future_goal_commitments": future_goal_commitments,
        "liquid_assets": liquid_assets,
        "current_term_cover": current_term_cover,
    }
    for name, value in amounts.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    gross_need = human_life_value + outstanding_liabilities + future_goal_commitments
    available = liquid_assets + current_term_cover
    return CoverDeficit(
        human_life_value=human_life_value,
        outstanding_liabilities=outstanding_liabilities,
        future_goal_commitments=future_goal_commitments,
        gross_need=gross_need,
        liquid_assets=liquid_assets,
        current_term_cover=current_term_cover,
        available_resources=available,
        net_required_cover=max(0.0, gross_need - available),
        surplus=max(0.0, available - gross_need),
    )


def calculate_net_insurance_need(
    age: int,
    annual_income: float,
    outstanding_liabilities: float,
    future_goal_commitments: float,
    liquid_assets: float,
    current_term_cover: float,
    retirement_age: int = DEFAULT_RETIREMENT_AGE,
) -> CoverDeficit:
    """HLV at default assumptions, then the net term cover deficit."""
    hlv = human_life_value(age, annual_income, retirement_age=retirement_age)
    return insurance_cover_deficit(
        human_life_value=hlv,
        outstanding_liabilities=outstanding_liabilities,
        future_goal_commitments=future_goal_commitments,
        liquid_assets=liquid_assets,
        current_term_cover=current_term_cover,
    )
