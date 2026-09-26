import csv

import pytest

from app.config import DATA_DIR
from app.engines.insurance_engine import (
    human_life_value,
    insurance_cover_deficit,
    load_mortality_rates,
    survival_probability,
)

# Published IALM 2012-14 rates used in hand calculations below.
Q58, Q59 = 0.009651, 0.010393


# --- Dataset ---

def test_dataset_covers_ages_18_to_70():
    assert sorted(load_mortality_rates()) == list(range(18, 71))


def test_dataset_matches_full_ialm_table():
    with open(DATA_DIR / "ialm_2012_14.csv", newline="") as f:
        full = {int(r["age"]): float(r["qx"]) for r in csv.DictReader(f)}
    assert all(full[age] == q for age, q in load_mortality_rates().items())


# --- Survival probability ---

def test_survival_probability_hand_calculation():
    assert survival_probability(58, 2) == pytest.approx((1 - Q58) * (1 - Q59))


def test_survival_probability_zero_years():
    assert survival_probability(40, 0) == 1.0


def test_survival_probability_rejects_negative_years():
    with pytest.raises(ValueError, match="non-negative"):
        survival_probability(40, -1)


def test_survival_probability_beyond_dataset():
    with pytest.raises(ValueError, match="age 71"):
        survival_probability(65, 10)


# --- Human Life Value: near-retirement cohort (age 58), hand-calculated ---

def test_hlv_age_58_hand_calculation():
    income = 1_200_000
    p1 = 1 - Q58
    p2 = p1 * (1 - Q59)
    expected = (
        income * 1.06 * 0.70 * p1 / 1.07
        + income * 1.06**2 * 0.70 * p2 / 1.07**2
    )
    assert human_life_value(58, income) == pytest.approx(expected)
    assert human_life_value(58, income) == pytest.approx(1_632_050, rel=1e-6)


# --- Human Life Value: young cohort (age 25) ---

def test_hlv_age_25_matches_formula():
    income = 1_000_000
    qx = load_mortality_rates()
    expected, p = 0.0, 1.0
    for t in range(1, 36):
        p *= 1 - qx[25 + t - 1]
        expected += income * 1.06**t * 0.70 * p / 1.07**t
    assert human_life_value(25, income) == pytest.approx(expected)


def test_hlv_age_25_bounds():
    # With g < i and mortality > 0, HLV is below the undiscounted 35 years x 70% of income.
    hlv = human_life_value(25, 1_000_000)
    assert 0.7 * 1_000_000 * 25 < hlv < 0.7 * 1_000_000 * 35


def test_hlv_young_exceeds_near_retirement():
    assert human_life_value(25, 1_000_000) > 10 * human_life_value(58, 1_000_000)


def test_hlv_no_mortality_and_growth_equals_discount():
    # With q = 0 and g = i, each year contributes exactly I * (1 - c).
    flat = {age: 0.0 for age in range(18, 71)}
    hlv = human_life_value(30, 500_000, wage_growth=0.07, discount_rate=0.07, qx=flat)
    assert hlv == pytest.approx(30 * 500_000 * 0.70)


def test_hlv_scales_linearly_with_income():
    assert human_life_value(40, 2_000_000) == pytest.approx(2 * human_life_value(40, 1_000_000))


def test_hlv_zero_at_or_after_retirement():
    assert human_life_value(60, 1_000_000) == 0.0
    assert human_life_value(65, 1_000_000) == 0.0


def test_hlv_full_consumption_is_zero():
    assert human_life_value(30, 1_000_000, personal_consumption_ratio=1.0) == 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"annual_income": -1}, "annual_income"),
        ({"personal_consumption_ratio": 1.5}, "personal_consumption_ratio"),
        ({"personal_consumption_ratio": -0.1}, "personal_consumption_ratio"),
        ({"discount_rate": -1.0}, "rates"),
        ({"wage_growth": -1.0}, "rates"),
    ],
)
def test_hlv_validation(kwargs, message):
    args = {"current_age": 30, "annual_income": 1_000_000, **kwargs}
    with pytest.raises(ValueError, match=message):
        human_life_value(**args)


# --- Net insurance cover deficit ---

def test_cover_deficit():
    result = insurance_cover_deficit(
        human_life_value=10_000_000,
        outstanding_liabilities=3_000_000,
        future_goal_commitments=2_000_000,
        liquid_assets=1_500_000,
        current_term_cover=5_000_000,
    )
    assert result.gross_need == 15_000_000
    assert result.available_resources == 6_500_000
    assert result.net_required_cover == 8_500_000
    assert result.surplus == 0.0


def test_liquid_assets_exceed_all_needs():
    hlv = human_life_value(58, 1_200_000)
    result = insurance_cover_deficit(
        human_life_value=hlv,
        outstanding_liabilities=500_000,
        future_goal_commitments=1_000_000,
        liquid_assets=5_000_000,
    )
    assert result.net_required_cover == 0.0
    assert result.surplus == pytest.approx(5_000_000 - (hlv + 1_500_000))
    assert result.surplus > 0


def test_resources_exactly_meet_need():
    result = insurance_cover_deficit(1_000_000, liquid_assets=400_000, current_term_cover=600_000)
    assert result.net_required_cover == 0.0
    assert result.surplus == 0.0


def test_cover_deficit_rejects_negative_amounts():
    with pytest.raises(ValueError, match="liquid_assets"):
        insurance_cover_deficit(1_000_000, liquid_assets=-1)


def test_calculate_net_insurance_need_combines_hlv_and_deficit():
    from app.engines.insurance_engine import calculate_net_insurance_need

    result = calculate_net_insurance_need(
        age=58, annual_income=1_200_000, outstanding_liabilities=600_000,
        future_goal_commitments=400_000, liquid_assets=300_000, current_term_cover=1_000_000,
    )
    assert result.human_life_value == pytest.approx(1_632_050, rel=1e-6)
    assert result.gross_need == pytest.approx(1_632_050 + 1_000_000, rel=1e-6)
    assert result.net_required_cover == pytest.approx(1_632_050 + 1_000_000 - 1_300_000, rel=1e-6)
