"""Life contingency engine based on the IALM 2012-14 mortality table.

IALM 2012-14 (Ult.) gives graduated ultimate mortality rates for male insured lives
that were medically underwritten at inception, by age last birthday, ages 2 to 115
(q_115 = 1). Source: Institute of Actuaries of India, effective 1 April 2019.

Conventions:
- Ages and terms are whole years (age last birthday).
- Deaths are assumed to occur at the end of the year of death (discrete assurances);
  annuities are payable annually in advance (annuity-due).
- Interest rates are annual effective rates; the default is the 10-year G-Sec yield.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import DATA_DIR, GSEC_10Y_YIELD

IALM_2012_14_PATH = DATA_DIR / "ialm_2012_14.csv"


@dataclass(frozen=True)
class MortalityTable:
    name: str
    min_age: int
    qx: np.ndarray  # qx[i] is the one-year death probability at age min_age + i

    def __post_init__(self) -> None:
        if self.qx.ndim != 1 or self.qx.size == 0:
            raise ValueError("qx must be a non-empty 1-D array")
        if np.any((self.qx < 0) | (self.qx > 1)):
            raise ValueError("qx values must lie in [0, 1]")
        if self.qx[-1] != 1.0:
            raise ValueError("table must be closed: q at the terminal age must equal 1")

    @classmethod
    def from_csv(cls, path: Path, name: str) -> MortalityTable:
        with open(path, newline="") as f:
            rows = [(int(r["age"]), float(r["qx"])) for r in csv.DictReader(f)]
        ages = [a for a, _ in rows]
        if ages != list(range(ages[0], ages[0] + len(ages))):
            raise ValueError(f"ages in {path} must be consecutive integers")
        return cls(name=name, min_age=ages[0], qx=np.array([q for _, q in rows]))

    @property
    def max_age(self) -> int:
        return self.min_age + self.qx.size - 1

    def _check_age(self, age: int) -> None:
        if not self.min_age <= age <= self.max_age:
            raise ValueError(f"age {age} outside table range {self.min_age}-{self.max_age}")

    @staticmethod
    def _check_non_negative(value: int, label: str) -> None:
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}")

    def _survival_curve(self, age: int, years: int) -> np.ndarray:
        """Return [0p_x, 1p_x, ..., years p_x]; zero beyond the terminal age."""
        start = age - self.min_age
        px = 1.0 - self.qx[start : start + years]
        curve = np.concatenate(([1.0], np.cumprod(px)))
        return np.pad(curve, (0, years + 1 - curve.size))

    def q(self, age: int) -> float:
        """One-year death probability q_x."""
        self._check_age(age)
        return float(self.qx[age - self.min_age])

    def survival_probability(self, age: int, years: int) -> float:
        """t p_x: probability a life aged x survives t more years."""
        self._check_age(age)
        self._check_non_negative(years, "years")
        return float(self._survival_curve(age, years)[-1])

    def death_probability(self, age: int, years: int) -> float:
        """t q_x: probability a life aged x dies within t years."""
        return 1.0 - self.survival_probability(age, years)

    def deferred_death_probability(self, age: int, deferral: int, years: int = 1) -> float:
        """u|t q_x: probability of surviving u years then dying within the next t."""
        self._check_non_negative(deferral, "deferral")
        return self.survival_probability(age, deferral) - self.survival_probability(
            age, deferral + years
        )

    def curtate_life_expectancy(self, age: int) -> float:
        """e_x: expected number of whole future years lived."""
        self._check_age(age)
        return float(self._survival_curve(age, self.max_age - age + 1)[1:].sum())

    def complete_life_expectancy(self, age: int) -> float:
        """Approximate complete expectation, assuming deaths spread uniformly over each year."""
        return self.curtate_life_expectancy(age) + 0.5

    def _cash_flows(self, age: int, term: int | None, rate: float):
        self._check_age(age)
        if rate <= -1:
            raise ValueError("rate must be greater than -100%")
        n = self.max_age - age + 1 if term is None else term
        self._check_non_negative(n, "term")
        kpx = self._survival_curve(age, n)
        v = 1.0 / (1.0 + rate)
        return n, kpx, v ** np.arange(n + 1)

    def term_assurance(self, age: int, term: int, rate: float = GSEC_10Y_YIELD) -> float:
        """A^1_{x:n}: present value of 1 paid at end of year of death within n years."""
        n, kpx, disc = self._cash_flows(age, term, rate)
        deaths = kpx[:-1] - kpx[1:]  # k|q_x for k = 0..n-1
        return float(np.dot(disc[1:], deaths))

    def whole_life_assurance(self, age: int, rate: float = GSEC_10Y_YIELD) -> float:
        """A_x: present value of 1 paid at end of year of death."""
        return self.term_assurance(age, self.max_age - age + 1, rate)

    def pure_endowment(self, age: int, term: int, rate: float = GSEC_10Y_YIELD) -> float:
        """nE_x: present value of 1 paid at time n if alive."""
        n, kpx, disc = self._cash_flows(age, term, rate)
        return float(disc[n] * kpx[n])

    def endowment_assurance(self, age: int, term: int, rate: float = GSEC_10Y_YIELD) -> float:
        """A_{x:n}: 1 paid at end of year of death, or at time n if alive."""
        return self.term_assurance(age, term, rate) + self.pure_endowment(age, term, rate)

    def annuity_due(self, age: int, term: int | None = None, rate: float = GSEC_10Y_YIELD) -> float:
        """ä_{x:n} (or whole-life ä_x if term is None): 1 per year in advance while alive."""
        n, kpx, disc = self._cash_flows(age, term, rate)
        return float(np.dot(disc[:n], kpx[:n]))


@lru_cache(maxsize=1)
def ialm_2012_14() -> MortalityTable:
    """IALM 2012-14 ultimate table (male, medically underwritten, age last birthday)."""
    return MortalityTable.from_csv(IALM_2012_14_PATH, name="IALM 2012-14 Ult.")
