import numpy as np
import pytest

from app.engines.mortality import MortalityTable, ialm_2012_14


@pytest.fixture(scope="module")
def ialm():
    return ialm_2012_14()


@pytest.fixture
def toy():
    # Three-age closed table, small enough to value by hand.
    return MortalityTable(name="toy", min_age=60, qx=np.array([0.1, 0.5, 1.0]))


# --- IALM 2012-14 data integrity (values as printed in the IAI publication) ---

@pytest.mark.parametrize(
    "age, qx",
    [(2, 0.000915), (16, 0.00077), (30, 0.000977), (45, 0.002579), (60, 0.011162),
     (82, 0.07535), (100, 0.397733), (114, 0.959214), (115, 1.0)],
)
def test_ialm_published_values(ialm, age, qx):
    assert ialm.q(age) == qx


def test_ialm_shape(ialm):
    assert (ialm.min_age, ialm.max_age, ialm.qx.size) == (2, 115, 114)
    assert ialm.name == "IALM 2012-14 Ult."


def test_ialm_increasing_from_age_26(ialm):
    # Rates dip slightly between ages 22 and 25, then rise at every age.
    assert np.all(np.diff(ialm.qx[26 - 2 :]) > 0)


# --- Survival and death probabilities ---

def test_one_year_survival(ialm):
    assert ialm.survival_probability(30, 1) == pytest.approx(0.999023)


def test_two_year_survival(ialm):
    assert ialm.survival_probability(30, 2) == pytest.approx((1 - 0.000977) * (1 - 0.001005))


def test_zero_years_survival_is_one(ialm):
    assert ialm.survival_probability(40, 0) == 1.0


def test_survival_beyond_terminal_age_is_zero(ialm):
    assert ialm.survival_probability(114, 2) == 0.0
    assert ialm.survival_probability(100, 50) == 0.0


def test_death_probability(toy):
    assert toy.death_probability(60, 2) == pytest.approx(1 - 0.9 * 0.5)


def test_deferred_death_probability(toy):
    # Survive age 60 (0.9), then die at 61 (0.5).
    assert toy.deferred_death_probability(60, 1) == pytest.approx(0.45)
    assert toy.deferred_death_probability(60, 0, 3) == pytest.approx(1.0)


# --- Life expectancy ---

def test_curtate_life_expectancy_toy(toy):
    # 1p = 0.9, 2p = 0.45, 3p = 0
    assert toy.curtate_life_expectancy(60) == pytest.approx(1.35)
    assert toy.complete_life_expectancy(60) == pytest.approx(1.85)


def test_curtate_life_expectancy_near_terminal_age(ialm):
    assert ialm.curtate_life_expectancy(114) == pytest.approx(1 - 0.959214)
    assert ialm.curtate_life_expectancy(115) == 0.0


def test_life_expectancy_decreases_with_age(ialm):
    e = [ialm.curtate_life_expectancy(a) for a in range(30, 100)]
    assert all(a > b for a, b in zip(e, e[1:]))


# --- Actuarial present values (toy table, i = 10%) ---

V = 1 / 1.1


def test_term_assurance_toy(toy):
    assert toy.term_assurance(60, 2, rate=0.10) == pytest.approx(V * 0.1 + V**2 * 0.45)


def test_whole_life_assurance_toy(toy):
    assert toy.whole_life_assurance(60, rate=0.10) == pytest.approx(
        V * 0.1 + V**2 * 0.45 + V**3 * 0.45
    )


def test_pure_endowment_toy(toy):
    assert toy.pure_endowment(60, 2, rate=0.10) == pytest.approx(V**2 * 0.45)


def test_endowment_assurance_toy(toy):
    assert toy.endowment_assurance(60, 2, rate=0.10) == pytest.approx(
        V * 0.1 + V**2 * 0.45 + V**2 * 0.45
    )


def test_annuity_due_toy(toy):
    assert toy.annuity_due(60, rate=0.10) == pytest.approx(1 + V * 0.9 + V**2 * 0.45)
    assert toy.annuity_due(60, 2, rate=0.10) == pytest.approx(1 + V * 0.9)


def test_zero_term_values(toy):
    assert toy.term_assurance(60, 0, rate=0.10) == 0.0
    assert toy.annuity_due(60, 0, rate=0.10) == 0.0
    assert toy.pure_endowment(60, 0, rate=0.10) == 1.0


# --- Actuarial identities on IALM at the default 7% G-Sec rate ---

@pytest.mark.parametrize("age", [25, 40, 60, 85])
def test_whole_life_identity(ialm, age):
    d = 0.07 / 1.07
    assert ialm.whole_life_assurance(age) == pytest.approx(1 - d * ialm.annuity_due(age))


@pytest.mark.parametrize("age, n", [(30, 20), (45, 15), (60, 10)])
def test_endowment_identity(ialm, age, n):
    d = 0.07 / 1.07
    assert ialm.endowment_assurance(age, n) == pytest.approx(1 - d * ialm.annuity_due(age, n))


def test_zero_interest_whole_life_assurance_is_one(ialm):
    assert ialm.whole_life_assurance(50, rate=0.0) == pytest.approx(1.0)


def test_zero_interest_annuity_equals_expectation_plus_one(ialm):
    assert ialm.annuity_due(50, rate=0.0) == pytest.approx(ialm.curtate_life_expectancy(50) + 1)


def test_default_rate_is_gsec_anchor(ialm):
    assert ialm.annuity_due(40) == ialm.annuity_due(40, rate=0.07)


# --- Validation ---

@pytest.mark.parametrize("age", [1, 116, -5])
def test_age_out_of_range(ialm, age):
    with pytest.raises(ValueError, match="outside table range"):
        ialm.q(age)


def test_negative_years(ialm):
    with pytest.raises(ValueError, match="years must be non-negative"):
        ialm.survival_probability(40, -1)


def test_negative_deferral(ialm):
    with pytest.raises(ValueError, match="deferral must be non-negative"):
        ialm.deferred_death_probability(40, -1)


def test_negative_term(ialm):
    with pytest.raises(ValueError, match="term must be non-negative"):
        ialm.term_assurance(40, -1)


def test_invalid_rate(ialm):
    with pytest.raises(ValueError, match="rate"):
        ialm.annuity_due(40, rate=-1.0)


@pytest.mark.parametrize(
    "qx, message",
    [
        (np.array([]), "non-empty"),
        (np.array([[0.1, 1.0]]), "non-empty"),
        (np.array([0.1, 1.2]), r"\[0, 1\]"),
        (np.array([-0.1, 1.0]), r"\[0, 1\]"),
        (np.array([0.1, 0.5]), "closed"),
    ],
)
def test_invalid_table(qx, message):
    with pytest.raises(ValueError, match=message):
        MortalityTable(name="bad", min_age=0, qx=qx)


def test_from_csv_rejects_gaps(tmp_path):
    path = tmp_path / "gap.csv"
    path.write_text("age,qx\n10,0.1\n12,1\n")
    with pytest.raises(ValueError, match="consecutive"):
        MortalityTable.from_csv(path, name="gap")
