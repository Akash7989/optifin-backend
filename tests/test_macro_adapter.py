import pytest

from app.adapters import macro_adapter as macro

ENV_VARS = ["OPTIFIN_GSEC_10Y_YIELD", "OPTIFIN_REPO_RATE", "OPTIFIN_CPI_INFLATION",
            "OPTIFIN_REAL_ESTATE_INFLATION"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_fallback_baselines():
    assert macro.get_gsec_10y_yield() == 0.0705
    assert macro.get_repo_rate() == 0.065
    assert macro.get_cpi_inflation() == 0.055
    assert macro.get_real_estate_inflation() == 0.07


def test_env_override_is_read_on_each_call(monkeypatch):
    monkeypatch.setenv("OPTIFIN_REPO_RATE", "0.055")
    assert macro.get_repo_rate() == 0.055
    monkeypatch.setenv("OPTIFIN_REPO_RATE", "0.0525")
    assert macro.get_repo_rate() == 0.0525


def test_blank_env_uses_fallback(monkeypatch):
    monkeypatch.setenv("OPTIFIN_CPI_INFLATION", "  ")
    assert macro.get_cpi_inflation() == 0.055


def test_snapshot_reports_source(monkeypatch):
    monkeypatch.setenv("OPTIFIN_GSEC_10Y_YIELD", "0.064")
    snap = macro.get_macro_snapshot()
    assert set(snap) == {"gsec_10y_yield", "repo_rate", "cpi_inflation", "real_estate_inflation"}
    assert (snap["gsec_10y_yield"].value, snap["gsec_10y_yield"].source) == (0.064, "env")
    assert (snap["repo_rate"].value, snap["repo_rate"].source) == (0.065, "fallback")


def test_non_numeric_env_rejected(monkeypatch):
    monkeypatch.setenv("OPTIFIN_REAL_ESTATE_INFLATION", "seven")
    with pytest.raises(ValueError, match="not a number"):
        macro.get_real_estate_inflation()


@pytest.mark.parametrize("raw", ["6.5", "-0.01", "1"])
def test_percent_style_or_out_of_range_env_rejected(monkeypatch, raw):
    monkeypatch.setenv("OPTIFIN_REPO_RATE", raw)
    with pytest.raises(ValueError, match="decimal fraction"):
        macro.get_repo_rate()
