import math
import time

import numpy as np
import pytest

from app.engines.goal_simulator import (
    DT,
    blended_portfolio_params,
    future_target_cost,
    simulate_goal,
    terminal_wealth_components,
)

# --- Goal valuation ---

def test_future_target_cost_hand_calculation():
    # 50 lakh today, 10 years at 7%: 1.07^10 = 1.967151357...
    assert future_target_cost(5_000_000, 10) == pytest.approx(9_835_756.79, abs=0.01)


def test_downpayment_is_20_percent_of_future_cost():
    result = simulate_goal(5_000_000, 10, monthly_sip=10_000, equity_ratio=0.5, seed=1)
    assert result.future_target_cost == pytest.approx(9_835_756.79, abs=0.01)
    assert result.target_downpayment == pytest.approx(1_967_151.36, abs=0.01)


# --- Portfolio parameters ---

@pytest.mark.parametrize(
    "ratio, mu, sigma",
    [(0.0, 0.07, 0.04), (1.0, 0.12, 0.15), (0.6, 0.10, 0.106), (0.5, 0.095, 0.095)],
)
def test_blended_portfolio_params(ratio, mu, sigma):
    assert blended_portfolio_params(ratio) == pytest.approx((mu, sigma))


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_equity_ratio_out_of_range(ratio):
    with pytest.raises(ValueError, match="equity_ratio"):
        blended_portfolio_params(ratio)


# --- Path dynamics ---

def test_zero_volatility_matches_closed_form():
    # 12 months, SIP 10,000 invested at the start of each month, W0 = 1 lakh.
    a, b = terminal_wealth_components(100_000, 12, mu=0.12, sigma=0.0, n_paths=3)
    g = math.exp(0.12 * DT)
    annuity = sum(g**k for k in range(1, 13))
    assert np.allclose(a, 100_000 * math.exp(0.12))
    assert np.allclose(a + 10_000 * b, 100_000 * math.exp(0.12) + 10_000 * annuity)


def test_decomposition_matches_month_by_month_recursion():
    months, paths, sip, mu, sigma = 24, 50, 5_000, 0.10, 0.106
    a, b = terminal_wealth_components(200_000, months, mu, sigma, paths, np.random.default_rng(7))

    z = np.random.default_rng(7).standard_normal((paths, months))
    w = np.full(paths, 200_000.0)
    for m in range(months):
        w = (w + sip) * np.exp((mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * z[:, m])
    assert np.allclose(a + sip * b, w)


def test_gbm_mean_and_median_without_sip():
    # Lump sum only: E[W_T] = W0 e^{mu T}, median = W0 e^{(mu - sigma^2/2) T}.
    a, _ = terminal_wealth_components(
        1_000_000, 120, mu=0.12, sigma=0.15, n_paths=10_000, rng=np.random.default_rng(0)
    )
    assert a.mean() == pytest.approx(1_000_000 * math.exp(1.2), rel=0.03)
    assert np.median(a) == pytest.approx(1_000_000 * math.exp(1.0875), rel=0.03)


def test_simulation_shape_validation():
    with pytest.raises(ValueError, match="months"):
        terminal_wealth_components(0, 0, 0.1, 0.1)
    with pytest.raises(ValueError, match="n_paths"):
        terminal_wealth_components(0, 12, 0.1, 0.1, n_paths=0)


def test_default_rng_runs():
    a, b = terminal_wealth_components(1_000, 12, 0.1, 0.1, n_paths=5)
    assert a.shape == b.shape == (5,)


# --- Output metrics ---

def test_seed_makes_results_reproducible():
    r1 = simulate_goal(10_000_000, 10, 20_000, 0.6, seed=123)
    r2 = simulate_goal(10_000_000, 10, 20_000, 0.6, seed=123)
    assert r1 == r2


def test_percentiles_are_ordered():
    r = simulate_goal(10_000_000, 15, 20_000, 0.8, seed=5)
    assert 0 < r.wealth_p10 < r.wealth_p50 < r.wealth_p90


def test_on_track_goal_needs_no_extra_sip():
    r = simulate_goal(5_000_000, 10, monthly_sip=50_000, equity_ratio=0.5, seed=3)
    assert r.success_probability >= 85
    assert r.recommended_sip_delta == 0
    assert r.required_monthly_sip == 50_000


def test_initial_wealth_alone_can_meet_goal():
    r = simulate_goal(5_000_000, 5, monthly_sip=0, equity_ratio=0.0,
                      initial_wealth=5_000_000, seed=3)
    assert r.success_probability == 100.0
    assert r.recommended_sip_delta == 0


def test_goal_seek_finds_minimum_sip():
    args = dict(current_cost=10_000_000, horizon_years=10, equity_ratio=0.6, seed=42)
    r = simulate_goal(monthly_sip=5_000, **args)
    assert r.success_probability < 85
    assert r.recommended_sip_delta > 0
    assert r.required_monthly_sip == 5_000 + r.recommended_sip_delta

    # Same seed = same market paths, so the recommended SIP must hit 85% ...
    assert simulate_goal(monthly_sip=r.required_monthly_sip, **args).success_probability >= 85
    # ... and anything materially lower must not.
    assert simulate_goal(monthly_sip=r.required_monthly_sip - 2, **args).success_probability < 85


def test_goal_seek_from_zero_sip():
    r = simulate_goal(10_000_000, 10, monthly_sip=0, equity_ratio=1.0, seed=9)
    assert r.success_probability == 0.0
    assert r.required_monthly_sip > 0


def test_higher_equity_raises_median_wealth():
    low = simulate_goal(10_000_000, 15, 20_000, 0.2, seed=11)
    high = simulate_goal(10_000_000, 15, 20_000, 0.9, seed=11)
    assert high.wealth_p50 > low.wealth_p50
    assert high.wealth_p90 - high.wealth_p10 > low.wealth_p90 - low.wealth_p10


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"current_cost": 0}, "current_cost"),
        ({"horizon_years": 0}, "horizon_years"),
        ({"monthly_sip": -1}, "non-negative"),
        ({"initial_wealth": -1}, "non-negative"),
        ({"downpayment_ratio": 0}, "downpayment_ratio"),
        ({"target_success": 1.0}, "target_success"),
    ],
)
def test_simulate_goal_validation(kwargs, message):
    args = {"current_cost": 5_000_000, "horizon_years": 10, "monthly_sip": 10_000,
            "equity_ratio": 0.5, **kwargs}
    with pytest.raises(ValueError, match=message):
        simulate_goal(**args)


# --- Performance: 10,000 paths, including goal seek, under 300 ms ---

@pytest.mark.parametrize("horizon_years", [10, 20, 30])
def test_performance_under_300ms(horizon_years):
    def run():
        return simulate_goal(20_000_000, horizon_years, 5_000, 0.7, seed=1)

    run()  # warm-up
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        result = run()
        timings.append(time.perf_counter() - start)
    assert result.recommended_sip_delta > 0  # goal seek was exercised
    assert min(timings) < 0.300, f"best of 3 took {min(timings) * 1000:.0f} ms"


# --- Yearly wealth trajectory ---

from app.engines.goal_simulator import _log_returns, yearly_wealth_percentiles  # noqa: E402


def test_trajectory_matches_month_by_month_recursion():
    months, paths, sip, w0 = 36, 200, 5_000, 100_000
    log_returns = _log_returns(months, 0.10, 0.106, paths, np.random.default_rng(3))
    w = np.full(paths, float(w0))
    yearly = [w.copy()]
    for m in range(months):
        w = (w + sip) * np.exp(log_returns[:, m])
        if (m + 1) % 12 == 0:
            yearly.append(w.copy())
    trajectory = yearly_wealth_percentiles(w0, sip, log_returns)
    assert [t.year for t in trajectory] == [0, 1, 2, 3]
    for point, wealth in zip(trajectory, yearly):
        assert point.p10 == pytest.approx(np.percentile(wealth, 10))
        assert point.p50 == pytest.approx(np.percentile(wealth, 50))
        assert point.p90 == pytest.approx(np.percentile(wealth, 90))


def test_trajectory_endpoints_match_result():
    r = simulate_goal(10_000_000, 10, 20_000, 0.6, initial_wealth=500_000, seed=8)
    assert len(r.trajectory) == 11
    start, end = r.trajectory[0], r.trajectory[-1]
    assert start.p10 == start.p50 == start.p90 == pytest.approx(500_000)
    assert (end.p10, end.p50, end.p90) == pytest.approx((r.wealth_p10, r.wealth_p50, r.wealth_p90))


def test_trajectory_percentiles_ordered_each_year():
    r = simulate_goal(10_000_000, 15, 20_000, 0.8, seed=4)
    assert all(t.p10 <= t.p50 <= t.p90 for t in r.trajectory)
