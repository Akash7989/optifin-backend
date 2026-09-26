"""Home loan capacity engine: FOIR affordability and RBI loan-to-value limits.

- LTV: RBI caps individual housing loans at 90% of property value up to Rs 30 lakh,
  80% above Rs 30 lakh up to Rs 75 lakh, and 75% above Rs 75 lakh.
- FOIR (Fixed Obligation to Income Ratio): all EMIs, existing plus new, as a share of net
  monthly income. RBI does not prescribe a single FOIR; the tiers below reflect common
  Indian bank practice (higher incomes are allowed a higher ratio) and are configurable.
- EMIs use the standard reducing-balance formula with monthly compounding.
"""

from __future__ import annotations

from dataclasses import dataclass

LAKH = 100_000

# (property value up to and including, max LTV)
RBI_LTV_TIERS: tuple[tuple[float, float], ...] = (
    (30 * LAKH, 0.90),
    (75 * LAKH, 0.80),
    (float("inf"), 0.75),
)

# (net monthly income up to and including, max FOIR)
FOIR_TIERS: tuple[tuple[float, float], ...] = (
    (50_000, 0.40),
    (100_000, 0.50),
    (200_000, 0.55),
    (float("inf"), 0.60),
)

DEFAULT_MAX_TENURE_YEARS = 20
DEFAULT_MAX_AGE_AT_MATURITY = 60


def rbi_ltv_cap(property_value: float) -> float:
    if property_value <= 0:
        raise ValueError("property_value must be positive")
    return next(ltv for limit, ltv in RBI_LTV_TIERS if property_value <= limit)


def foir_limit(net_monthly_income: float) -> float:
    if net_monthly_income <= 0:
        raise ValueError("net_monthly_income must be positive")
    return next(foir for limit, foir in FOIR_TIERS if net_monthly_income <= limit)


def emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Monthly instalment for a reducing-balance loan."""
    if principal < 0 or annual_rate < 0:
        raise ValueError("principal and annual_rate must be non-negative")
    if tenure_months <= 0:
        raise ValueError("tenure_months must be positive")
    r = annual_rate / 12
    if r == 0:
        return principal / tenure_months
    growth = (1 + r) ** tenure_months
    return principal * r * growth / (growth - 1)


def principal_for_emi(monthly_emi: float, annual_rate: float, tenure_months: int) -> float:
    """Largest loan a given EMI can service (inverse of `emi`)."""
    if monthly_emi < 0 or annual_rate < 0:
        raise ValueError("monthly_emi and annual_rate must be non-negative")
    if tenure_months <= 0:
        return 0.0
    r = annual_rate / 12
    if r == 0:
        return monthly_emi * tenure_months
    growth = (1 + r) ** tenure_months
    return monthly_emi * (growth - 1) / (r * growth)


@dataclass(frozen=True)
class LoanCapacity:
    property_value: float
    projected_monthly_income: float
    existing_emis: float
    current_foir: float  # existing EMIs / projected income
    foir_limit: float
    max_total_emi: float
    available_emi: float
    annual_rate: float
    tenure_years: int
    max_loan_by_foir: float
    ltv_cap: float
    max_loan_by_ltv: float
    eligible_loan: float
    binding_constraint: str  # "foir" or "ltv"
    required_downpayment: float  # property value not covered by the eligible loan
    required_downpayment_ratio: float
    emi_on_eligible_loan: float


def calculate_loan_capacity(
    net_monthly_income: float,
    existing_emis: float,
    property_value: float,
    annual_rate: float,
    borrower_age_at_start: int,
    years_until_purchase: int = 0,
    income_growth: float = 0.0,
    max_tenure_years: int = DEFAULT_MAX_TENURE_YEARS,
    max_age_at_maturity: int = DEFAULT_MAX_AGE_AT_MATURITY,
) -> LoanCapacity:
    """Maximum home loan for a property, limited by both FOIR and RBI LTV.

    Income is projected to the purchase date at `income_growth`; existing EMIs are assumed
    to continue unchanged. Tenure ends by `max_age_at_maturity`.
    """
    if existing_emis < 0:
        raise ValueError("existing_emis must be non-negative")
    if years_until_purchase < 0:
        raise ValueError("years_until_purchase must be non-negative")
    if income_growth <= -1:
        raise ValueError("income_growth must be greater than -100%")

    income = net_monthly_income * (1 + income_growth) ** years_until_purchase
    limit = foir_limit(income)
    max_total_emi = limit * income
    available_emi = max(0.0, max_total_emi - existing_emis)
    tenure_years = max(0, min(max_tenure_years, max_age_at_maturity - borrower_age_at_start))

    by_foir = principal_for_emi(available_emi, annual_rate, tenure_years * 12)
    ltv = rbi_ltv_cap(property_value)
    by_ltv = ltv * property_value
    eligible = min(by_foir, by_ltv)
    downpayment = property_value - eligible

    return LoanCapacity(
        property_value=property_value,
        projected_monthly_income=income,
        existing_emis=existing_emis,
        current_foir=existing_emis / income,
        foir_limit=limit,
        max_total_emi=max_total_emi,
        available_emi=available_emi,
        annual_rate=annual_rate,
        tenure_years=tenure_years,
        max_loan_by_foir=by_foir,
        ltv_cap=ltv,
        max_loan_by_ltv=by_ltv,
        eligible_loan=eligible,
        binding_constraint="ltv" if by_ltv <= by_foir else "foir",
        required_downpayment=downpayment,
        required_downpayment_ratio=downpayment / property_value,
        emi_on_eligible_loan=emi(eligible, annual_rate, tenure_years * 12) if tenure_years else 0.0,
    )
