"""Stochastic goal simulation: Monte Carlo of a SIP portfolio against a home downpayment.

The portfolio follows Geometric Brownian Motion with monthly steps:

    W_{m+1} = (W_m + SIP) * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z_m),  dt = 1/12

Because W_n is linear in the SIP, every path's terminal wealth can be written as

    W_n = W_0 * exp(S_0) + SIP * sum_{m=0}^{n-1} exp(S_m),   S_m = sum_{k=m}^{n-1} log-return_k

so one (paths x months) matrix of shocks prices any SIP. The goal seek reuses the same
shocks (common random numbers), which makes success probability monotone in the SIP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 1.0 / 12.0

DEFAULT_REAL_ESTATE_INFLATION = 0.07
DEFAULT_DOWNPAYMENT_RATIO = 0.20
DEFAULT_TARGET_SUCCESS = 0.85
DEFAULT_PATHS = 10_000

# Benchmarks: Nifty 50 for equity, G-Sec / liquid funds for debt.
EQUITY_MU, EQUITY_SIGMA = 0.12, 0.15
DEBT_MU, DEBT_SIGMA = 0.07, 0.04

GOAL_SEEK_TOLERANCE = 1.0  # rupees per month


@dataclass(frozen=True)
class YearlyPercentiles:
    year: int
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class GoalSimulationResult:
    future_target_cost: float
    target_downpayment: float
    portfolio_mu: float
    portfolio_sigma: float
    success_probability: float  # percent of paths (0-100)
    wealth_p10: float  # bear case
    wealth_p50: float  # median
    wealth_p90: float  # bull case
    required_monthly_sip: float  # minimum SIP for the target success rate
    recommended_sip_delta: float  # extra monthly SIP needed; 0 if already on track
    trajectory: tuple[YearlyPercentiles, ...] = ()  # wealth percentiles by year at the current SIP


def future_target_cost(
    current_cost: float, horizon_years: float, inflation: float = DEFAULT_REAL_ESTATE_INFLATION
) -> float:
    return current_cost * (1 + inflation) ** horizon_years


def blended_portfolio_params(equity_ratio: float) -> tuple[float, float]:
    """Return (mu, sigma) as a linear blend of equity and debt benchmarks."""
    if not 0.0 <= equity_ratio <= 1.0:
        raise ValueError("equity_ratio must be between 0 and 1")
    mu = equity_ratio * EQUITY_MU + (1 - equity_ratio) * DEBT_MU
    sigma = equity_ratio * EQUITY_SIGMA + (1 - equity_ratio) * DEBT_SIGMA
    return mu, sigma


def terminal_wealth_components(
    initial_wealth: float,
    months: int,
    mu: float,
    sigma: float,
    n_paths: int = DEFAULT_PATHS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) per path such that terminal wealth = A + SIP * B."""
    if months < 1:
        raise ValueError("months must be at least 1")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")
    rng = np.random.default_rng() if rng is None else rng

    return _components(initial_wealth, _log_returns(months, mu, sigma, n_paths, rng))


def _log_returns(
    months: int, mu: float, sigma: float, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    z = rng.standard_normal((n_paths, months))
    return (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * z


def _decompose(log_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (total log-growth per path, growth[:, m] from month m to the horizon)."""
    prefix = np.cumsum(log_returns, axis=1)
    total = prefix[:, -1]
    return total, np.exp(total[:, None] - (prefix - log_returns))


def _components(initial_wealth: float, log_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total, growth = _decompose(log_returns)
    return initial_wealth * np.exp(total), growth.sum(axis=1)


def _yearly_percentiles(
    initial_wealth: float, monthly_sip: float, total: np.ndarray, growth: np.ndarray
) -> list[YearlyPercentiles]:
    # Wealth after m months: W_m = (W_0 * e^total + SIP * sum_{j<m} growth_j) / growth_m,
    # the closed form of the monthly recursion (growth_n = 1 at the horizon).
    n_paths, months = growth.shape
    contributions = np.cumsum(growth, axis=1)
    end_value = initial_wealth * np.exp(total)
    columns = [np.full(n_paths, float(initial_wealth))]
    for m in range(12, months + 1, 12):
        growth_m = growth[:, m] if m < months else 1.0
        columns.append((end_value + monthly_sip * contributions[:, m - 1]) / growth_m)
    p10, p50, p90 = np.percentile(np.column_stack(columns), [10, 50, 90], axis=0)
    return [
        YearlyPercentiles(year=y, p10=float(p10[y]), p50=float(p50[y]), p90=float(p90[y]))
        for y in range(len(columns))
    ]


def yearly_wealth_percentiles(
    initial_wealth: float, monthly_sip: float, log_returns: np.ndarray
) -> list[YearlyPercentiles]:
    """10th/50th/90th percentile wealth at the start and at the end of every year."""
    return _yearly_percentiles(initial_wealth, monthly_sip, *_decompose(log_returns))


def _success_rate(a: np.ndarray, b: np.ndarray, sip: float, target: float) -> float:
    return float(np.mean(a + sip * b >= target))


def _goal_seek_sip(
    a: np.ndarray, b: np.ndarray, target: float, required_success: float, start_sip: float
) -> float:
    """Binary search for the smallest SIP (to within GOAL_SEEK_TOLERANCE) meeting the target."""
    lo = start_sip
    hi = max(start_sip, 1_000.0)
    while _success_rate(a, b, hi, target) < required_success:
        lo, hi = hi, hi * 2
    while hi - lo > GOAL_SEEK_TOLERANCE:
        mid = (lo + hi) / 2
        if _success_rate(a, b, mid, target) >= required_success:
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi))


def simulate_goal(
    current_cost: float,
    horizon_years: int,
    monthly_sip: float,
    equity_ratio: float,
    initial_wealth: float = 0.0,
    inflation: float = DEFAULT_REAL_ESTATE_INFLATION,
    downpayment_ratio: float = DEFAULT_DOWNPAYMENT_RATIO,
    target_success: float = DEFAULT_TARGET_SUCCESS,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = None,
) -> GoalSimulationResult:
    """Simulate saving for a home downpayment and goal-seek the SIP if success is too low."""
    if current_cost <= 0:
        raise ValueError("current_cost must be positive")
    if horizon_years < 1:
        raise ValueError("horizon_years must be at least 1")
    if monthly_sip < 0 or initial_wealth < 0:
        raise ValueError("monthly_sip and initial_wealth must be non-negative")
    if not 0 < downpayment_ratio <= 1:
        raise ValueError("downpayment_ratio must be in (0, 1]")
    if not 0 < target_success < 1:
        raise ValueError("target_success must be between 0 and 1")

    cost = future_target_cost(current_cost, horizon_years, inflation)
    target = downpayment_ratio * cost
    mu, sigma = blended_portfolio_params(equity_ratio)

    log_returns = _log_returns(horizon_years * 12, mu, sigma, n_paths, np.random.default_rng(seed))
    total, growth = _decompose(log_returns)
    a, b = initial_wealth * np.exp(total), growth.sum(axis=1)
    terminal = a + monthly_sip * b
    success = float(np.mean(terminal >= target))
    p10, p50, p90 = np.percentile(terminal, [10, 50, 90])

    required_sip = (
        monthly_sip
        if success >= target_success
        else _goal_seek_sip(a, b, target, target_success, monthly_sip)
    )
    return GoalSimulationResult(
        future_target_cost=cost,
        target_downpayment=target,
        portfolio_mu=mu,
        portfolio_sigma=sigma,
        success_probability=100 * success,
        wealth_p10=float(p10),
        wealth_p50=float(p50),
        wealth_p90=float(p90),
        required_monthly_sip=required_sip,
        recommended_sip_delta=required_sip - monthly_sip,
        trajectory=tuple(_yearly_percentiles(initial_wealth, monthly_sip, total, growth)),
    )


run_monte_carlo_simulation = simulate_goal
