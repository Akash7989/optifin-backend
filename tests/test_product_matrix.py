import pytest

from app.adapters.product_matrix import (
    ILLUSTRATIVE,
    LOAN_PRODUCTS,
    TERM_AGE_COHORTS,
    TERM_PLANS,
    match_loan_offers,
    quote_term_premiums,
)

REPO = 0.055


# --- Catalog integrity ---

def test_catalog_is_flagged_illustrative():
    assert ILLUSTRATIVE is True


def test_credit_bands_sorted_descending():
    for product in LOAN_PRODUCTS:
        scores = [b.min_score for b in product.bands]
        assert scores == sorted(scores, reverse=True)


def test_worse_credit_never_gets_lower_spread():
    for product in LOAN_PRODUCTS:
        spreads = [b.spread for b in product.bands]
        assert spreads == sorted(spreads)


def test_term_plans_cover_all_cohorts_and_rise_with_age():
    for plan in TERM_PLANS:
        assert tuple(plan.premium_per_crore) == TERM_AGE_COHORTS
        rates = list(plan.premium_per_crore.values())
        assert all(a < b for a, b in zip(rates, rates[1:]))


# --- Loan matching ---

def test_top_band_home_loan_offers_sorted_by_rate():
    offers = match_loan_offers("home_loan", 100_000, 810, repo_rate=REPO)
    assert [o.lender for o in offers] == ["SBI", "HDFC Bank", "ICICI Bank"]
    assert [o.interest_rate for o in offers] == pytest.approx([0.0740, 0.0750, 0.0760])
    assert all(o.credit_band_min_score == 800 for o in offers)


def test_band_selection_by_credit_score():
    offers = {o.lender: o for o in match_loan_offers("home_loan", 100_000, 760, repo_rate=REPO)}
    assert offers["SBI"].spread == 0.0205
    assert offers["HDFC Bank"].credit_band_min_score == 750


def test_band_boundary_is_inclusive():
    offers = {o.lender: o for o in match_loan_offers("home_loan", 100_000, 750, repo_rate=REPO)}
    assert offers["SBI"].credit_band_min_score == 750


def test_low_score_excludes_stricter_lenders():
    # 680: below HDFC Bank's 700 floor, within SBI (650) and ICICI Bank (675).
    offers = match_loan_offers("home_loan", 100_000, 680, repo_rate=REPO)
    assert [o.lender for o in offers] == ["SBI", "ICICI Bank"]


def test_income_threshold_excludes_lenders():
    offers = match_loan_offers("home_loan", 27_000, 810, repo_rate=REPO)
    assert [o.lender for o in offers] == ["SBI"]


def test_no_offers_below_all_floors():
    assert match_loan_offers("home_loan", 100_000, 600, repo_rate=REPO) == []
    assert match_loan_offers("personal_loan", 10_000, 850, repo_rate=REPO) == []


def test_personal_loans_priced_above_home_loans():
    home = match_loan_offers("home_loan", 100_000, 790, repo_rate=REPO)
    personal = match_loan_offers("personal_loan", 100_000, 790, repo_rate=REPO)
    assert min(o.interest_rate for o in personal) > max(o.interest_rate for o in home)


def test_rate_tracks_repo_from_macro_adapter(monkeypatch):
    monkeypatch.setenv("OPTIFIN_REPO_RATE", "0.06")
    offer = match_loan_offers("home_loan", 100_000, 810)[0]
    assert offer.repo_rate == 0.06
    assert offer.interest_rate == pytest.approx(0.06 + 0.019)


@pytest.mark.parametrize(
    "args, message",
    [
        (("car_loan", 50_000, 750), "loan_type"),
        (("home_loan", -1, 750), "monthly_income"),
        (("home_loan", 50_000, 299), "credit_score"),
        (("home_loan", 50_000, 901), "credit_score"),
    ],
)
def test_loan_validation(args, message):
    with pytest.raises(ValueError, match=message):
        match_loan_offers(*args, repo_rate=REPO)


# --- Term insurance quotes ---

def test_cohort_quotes_sorted_cheapest_first():
    quotes = quote_term_premiums(30)
    assert [q.insurer for q in quotes] == ["Tata AIA", "HDFC Life", "ICICI Prudential Life"]
    assert [q.annual_premium for q in quotes] == [11_200, 11_800, 12_100]


def test_interpolates_between_cohorts():
    # Age 33 is 3/5 of the way from 30 to 35: 11,200 + 0.6 * (14,900 - 11,200) = 13,420.
    tata = next(q for q in quote_term_premiums(33) if q.insurer == "Tata AIA")
    assert tata.annual_premium == pytest.approx(13_420)


def test_scales_with_sum_assured():
    quote = next(q for q in quote_term_premiums(40, 20_000_000) if q.insurer == "HDFC Life")
    assert quote.annual_premium == 44_800
    assert quote.sum_assured == 20_000_000


def test_edge_cohorts_supported():
    assert quote_term_premiums(25)[0].annual_premium == 9_000
    assert quote_term_premiums(50)[0].annual_premium == 48_500


@pytest.mark.parametrize("age", [24, 51])
def test_age_outside_cohorts_rejected(age):
    with pytest.raises(ValueError, match="age"):
        quote_term_premiums(age)


def test_non_positive_sum_assured_rejected():
    with pytest.raises(ValueError, match="sum_assured"):
        quote_term_premiums(30, 0)
