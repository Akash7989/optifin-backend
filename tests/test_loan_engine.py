import pytest

from app.engines.loan_engine import (
    LAKH,
    calculate_loan_capacity,
    emi,
    foir_limit,
    principal_for_emi,
    rbi_ltv_cap,
)

# --- RBI LTV tiers ---

@pytest.mark.parametrize(
    "value, cap",
    [(10 * LAKH, 0.90), (30 * LAKH, 0.90), (30 * LAKH + 1, 0.80), (75 * LAKH, 0.80),
     (75 * LAKH + 1, 0.75), (500 * LAKH, 0.75)],
)
def test_rbi_ltv_tiers(value, cap):
    assert rbi_ltv_cap(value) == cap


def test_ltv_rejects_non_positive_value():
    with pytest.raises(ValueError, match="property_value"):
        rbi_ltv_cap(0)


# --- FOIR tiers ---

@pytest.mark.parametrize(
    "income, limit",
    [(30_000, 0.40), (50_000, 0.40), (50_001, 0.50), (100_000, 0.50), (150_000, 0.55),
     (200_001, 0.60)],
)
def test_foir_tiers(income, limit):
    assert foir_limit(income) == limit


def test_foir_rejects_non_positive_income():
    with pytest.raises(ValueError, match="net_monthly_income"):
        foir_limit(0)


# --- EMI maths ---

def test_emi_reference_value():
    # Rs 10 lakh at 8.5% for 20 years: widely published EMI of Rs 8,678.
    assert emi(10 * LAKH, 0.085, 240) == pytest.approx(8_678.23, abs=0.01)


def test_emi_zero_rate():
    assert emi(120_000, 0.0, 12) == 10_000


def test_principal_for_emi_inverts_emi():
    assert principal_for_emi(8_678.23, 0.085, 240) == pytest.approx(10 * LAKH, abs=1)
    assert principal_for_emi(10_000, 0.0, 12) == 120_000


def test_principal_for_zero_tenure_is_zero():
    assert principal_for_emi(50_000, 0.08, 0) == 0.0


@pytest.mark.parametrize(
    "fn, args, message",
    [
        (emi, (-1, 0.08, 12), "non-negative"),
        (emi, (100, -0.01, 12), "non-negative"),
        (emi, (100, 0.08, 0), "tenure_months"),
        (principal_for_emi, (-1, 0.08, 12), "non-negative"),
        (principal_for_emi, (100, -0.01, 12), "non-negative"),
    ],
)
def test_emi_validation(fn, args, message):
    with pytest.raises(ValueError, match=message):
        fn(*args)


# --- Loan capacity ---

def test_foir_binding_case_hand_calculation():
    # Income 1.5 lakh -> FOIR 55% -> 82,500 total; minus 18,000 existing = 64,500 available.
    cap = calculate_loan_capacity(150_000, 18_000, 200 * LAKH, 0.085, borrower_age_at_start=35)
    assert cap.foir_limit == 0.55
    assert cap.max_total_emi == pytest.approx(82_500)
    assert cap.available_emi == pytest.approx(64_500)
    assert cap.tenure_years == 20
    assert cap.max_loan_by_foir == pytest.approx(64_500 / 8_678.23 * 10 * LAKH, rel=1e-6)
    assert cap.ltv_cap == 0.75 and cap.max_loan_by_ltv == 150 * LAKH
    assert cap.binding_constraint == "foir"
    assert cap.eligible_loan == cap.max_loan_by_foir
    assert cap.required_downpayment == pytest.approx(200 * LAKH - cap.eligible_loan)
    assert cap.emi_on_eligible_loan == pytest.approx(64_500)
    assert cap.current_foir == pytest.approx(0.12)


def test_ltv_binding_case():
    # High income, 60 lakh property: LTV 80% caps the loan at 48 lakh.
    cap = calculate_loan_capacity(400_000, 0, 60 * LAKH, 0.085, borrower_age_at_start=30)
    assert cap.binding_constraint == "ltv"
    assert cap.eligible_loan == 48 * LAKH
    assert cap.required_downpayment == pytest.approx(12 * LAKH)
    assert cap.required_downpayment_ratio == pytest.approx(0.20)


def test_existing_emis_above_foir_leave_no_capacity():
    cap = calculate_loan_capacity(40_000, 20_000, 40 * LAKH, 0.085, borrower_age_at_start=30)
    assert cap.available_emi == 0
    assert cap.eligible_loan == 0
    assert cap.required_downpayment_ratio == 1.0
    assert cap.emi_on_eligible_loan == 0


def test_tenure_capped_by_age_at_maturity():
    cap = calculate_loan_capacity(150_000, 0, 100 * LAKH, 0.085, borrower_age_at_start=48)
    assert cap.tenure_years == 12


def test_no_tenure_left_means_no_loan():
    cap = calculate_loan_capacity(150_000, 0, 100 * LAKH, 0.085, borrower_age_at_start=62)
    assert cap.tenure_years == 0
    assert cap.eligible_loan == 0 and cap.emi_on_eligible_loan == 0


def test_income_projected_to_purchase_date():
    cap = calculate_loan_capacity(100_000, 0, 100 * LAKH, 0.085, borrower_age_at_start=35,
                                  years_until_purchase=5, income_growth=0.06)
    assert cap.projected_monthly_income == pytest.approx(100_000 * 1.06**5)
    assert cap.foir_limit == 0.55  # projected income crosses into the next tier


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"existing_emis": -1}, "existing_emis"),
        ({"years_until_purchase": -1}, "years_until_purchase"),
        ({"income_growth": -1.0}, "income_growth"),
    ],
)
def test_capacity_validation(kwargs, message):
    args = {"net_monthly_income": 100_000, "existing_emis": 0, "property_value": 50 * LAKH,
            "annual_rate": 0.085, "borrower_age_at_start": 30, **kwargs}
    with pytest.raises(ValueError, match=message):
        calculate_loan_capacity(**args)
