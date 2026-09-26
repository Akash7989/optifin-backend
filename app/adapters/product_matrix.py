"""Normalized product catalog for Indian lenders and term insurers.

ILLUSTRATIVE DATA: the rules and rates below are representative placeholders shaped like
published Indian products. They are NOT actual quotes or eligibility criteria of the named
institutions and must be replaced with sourced data before being shown as real offers.

- Loans are repo-linked (EBLR): rate = RBI repo rate + spread for the borrower's credit band.
- Term premiums are annual, per Rs 1 crore sum assured, for a non-smoking male with cover to
  age 60, including GST.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.macro_adapter import get_repo_rate

ILLUSTRATIVE = True
CREDIT_SCORE_MIN, CREDIT_SCORE_MAX = 300, 900
ONE_CRORE = 10_000_000
TERM_AGE_COHORTS = (25, 30, 35, 40, 45, 50)


@dataclass(frozen=True)
class CreditBand:
    min_score: int
    spread: float  # over repo rate


@dataclass(frozen=True)
class LoanProduct:
    lender: str
    loan_type: str
    min_monthly_income: float
    bands: tuple[CreditBand, ...]  # sorted by min_score descending; last band is the floor


@dataclass(frozen=True)
class TermPlan:
    insurer: str
    plan: str
    premium_per_crore: dict[int, float]  # annual premium keyed by age cohort


def _bands(*pairs: tuple[int, float]) -> tuple[CreditBand, ...]:
    return tuple(CreditBand(score, spread) for score, spread in pairs)


LOAN_PRODUCTS: tuple[LoanProduct, ...] = (
    LoanProduct("SBI", "home_loan", 25_000,
                _bands((800, 0.0190), (750, 0.0205), (700, 0.0230), (650, 0.0280))),
    LoanProduct("HDFC Bank", "home_loan", 30_000,
                _bands((800, 0.0200), (750, 0.0215), (700, 0.0245))),
    LoanProduct("ICICI Bank", "home_loan", 30_000,
                _bands((800, 0.0210), (750, 0.0225), (700, 0.0255), (675, 0.0290))),
    LoanProduct("SBI", "personal_loan", 15_000,
                _bands((800, 0.0450), (750, 0.0525), (700, 0.0625))),
    LoanProduct("HDFC Bank", "personal_loan", 25_000,
                _bands((800, 0.0475), (750, 0.0575), (725, 0.0700))),
    LoanProduct("ICICI Bank", "personal_loan", 30_000,
                _bands((780, 0.0500), (730, 0.0600), (700, 0.0750))),
)

TERM_PLANS: tuple[TermPlan, ...] = (
    TermPlan("HDFC Life", "Standard term plan",
             {25: 9_500, 30: 11_800, 35: 15_600, 40: 22_400, 45: 33_500, 50: 51_000}),
    TermPlan("Tata AIA", "Standard term plan",
             {25: 9_000, 30: 11_200, 35: 14_900, 40: 21_300, 45: 31_800, 50: 48_500}),
    TermPlan("ICICI Prudential Life", "Standard term plan",
             {25: 9_800, 30: 12_100, 35: 16_000, 40: 23_000, 45: 34_600, 50: 52_800}),
)


@dataclass(frozen=True)
class LoanOffer:
    lender: str
    loan_type: str
    credit_band_min_score: int
    spread: float
    repo_rate: float
    interest_rate: float


@dataclass(frozen=True)
class TermQuote:
    insurer: str
    plan: str
    age: int
    sum_assured: float
    annual_premium: float


def match_loan_offers(
    loan_type: str,
    monthly_income: float,
    credit_score: int,
    repo_rate: float | None = None,
) -> list[LoanOffer]:
    """Offers the applicant qualifies for, cheapest first."""
    if loan_type not in {p.loan_type for p in LOAN_PRODUCTS}:
        raise ValueError(f"unknown loan_type {loan_type!r}")
    if monthly_income < 0:
        raise ValueError("monthly_income must be non-negative")
    if not CREDIT_SCORE_MIN <= credit_score <= CREDIT_SCORE_MAX:
        raise ValueError(f"credit_score must be between {CREDIT_SCORE_MIN} and {CREDIT_SCORE_MAX}")
    repo = get_repo_rate() if repo_rate is None else repo_rate

    offers = []
    for product in LOAN_PRODUCTS:
        if product.loan_type != loan_type or monthly_income < product.min_monthly_income:
            continue
        band = next((b for b in product.bands if credit_score >= b.min_score), None)
        if band is None:
            continue
        offers.append(LoanOffer(product.lender, loan_type, band.min_score, band.spread, repo,
                                repo + band.spread))
    return sorted(offers, key=lambda o: (o.interest_rate, o.lender))


def _premium_per_crore(plan: TermPlan, age: int) -> float:
    """Exact cohort rate, or linear interpolation between neighbouring cohorts."""
    if age in plan.premium_per_crore:
        return plan.premium_per_crore[age]
    lower = max(c for c in TERM_AGE_COHORTS if c < age)
    upper = min(c for c in TERM_AGE_COHORTS if c > age)
    weight = (age - lower) / (upper - lower)
    low, high = plan.premium_per_crore[lower], plan.premium_per_crore[upper]
    return low + weight * (high - low)


def quote_term_premiums(age: int, sum_assured: float = ONE_CRORE) -> list[TermQuote]:
    """Annual premium from each insurer for the given age and cover, cheapest first."""
    if not TERM_AGE_COHORTS[0] <= age <= TERM_AGE_COHORTS[-1]:
        raise ValueError(f"age must be between {TERM_AGE_COHORTS[0]} and {TERM_AGE_COHORTS[-1]}")
    if sum_assured <= 0:
        raise ValueError("sum_assured must be positive")
    quotes = [
        TermQuote(p.insurer, p.plan, age, sum_assured,
                  round(_premium_per_crore(p, age) * sum_assured / ONE_CRORE, 2))
        for p in TERM_PLANS
    ]
    return sorted(quotes, key=lambda q: (q.annual_premium, q.insurer))
