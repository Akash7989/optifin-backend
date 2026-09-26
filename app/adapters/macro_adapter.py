"""Indian macroeconomic benchmarks with static fallbacks and environment-variable overrides.

Overrides are annual rates written as decimal fractions (e.g. OPTIFIN_REPO_RATE=0.055).
They are read on every call, so changing the environment takes effect without a restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

FALLBACK_AS_OF = "baseline supplied at project setup; verify against RBI / MoSPI releases"


@dataclass(frozen=True)
class Indicator:
    key: str
    env_var: str
    fallback: float
    description: str


GSEC_10Y_YIELD = Indicator("gsec_10y_yield", "OPTIFIN_GSEC_10Y_YIELD", 0.0705,
                           "10-year Government of India bond yield")
REPO_RATE = Indicator("repo_rate", "OPTIFIN_REPO_RATE", 0.0650, "RBI policy repo rate")
CPI_INFLATION = Indicator("cpi_inflation", "OPTIFIN_CPI_INFLATION", 0.0550,
                          "Long-term urban CPI inflation")
REAL_ESTATE_INFLATION = Indicator("real_estate_inflation", "OPTIFIN_REAL_ESTATE_INFLATION", 0.0700,
                                  "Long-term residential real estate inflation")

INDICATORS = (GSEC_10Y_YIELD, REPO_RATE, CPI_INFLATION, REAL_ESTATE_INFLATION)


@dataclass(frozen=True)
class IndicatorValue:
    key: str
    value: float
    source: str  # "env" or "fallback"
    description: str


def _resolve(indicator: Indicator) -> IndicatorValue:
    raw = os.environ.get(indicator.env_var)
    if raw is None or raw.strip() == "":
        return IndicatorValue(indicator.key, indicator.fallback, "fallback", indicator.description)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{indicator.env_var}={raw!r} is not a number") from None
    if not 0 <= value < 1:
        raise ValueError(
            f"{indicator.env_var}={raw!r} must be a decimal fraction in [0, 1), e.g. 0.065 for 6.5%"
        )
    return IndicatorValue(indicator.key, value, "env", indicator.description)


def get_gsec_10y_yield() -> float:
    return _resolve(GSEC_10Y_YIELD).value


def get_repo_rate() -> float:
    return _resolve(REPO_RATE).value


def get_cpi_inflation() -> float:
    return _resolve(CPI_INFLATION).value


def get_real_estate_inflation() -> float:
    return _resolve(REAL_ESTATE_INFLATION).value


def get_macro_snapshot() -> dict[str, IndicatorValue]:
    """All indicators with their values and whether each came from the environment."""
    return {ind.key: _resolve(ind) for ind in INDICATORS}
