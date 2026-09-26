import os
from datetime import date
from types import SimpleNamespace

import pytest

from app.agent import client as client_module
from app.agent.client import MODEL
from app.agent.explainer import (
    DISCLAIMER,
    format_number,
    generate_plan_explanation,
    untraceable_numbers,
)
from app.agent.extractor import (
    CLARIFYING_QUESTIONS,
    UserProfileSchema,
    build_profile,
    extract_financial_profile,
)
from app.models.profile import ExtractedProfile

TODAY = date(2026, 9, 27)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses):
        self.models = FakeModels(responses)


def parsed(**fields):
    return SimpleNamespace(parsed=ExtractedProfile(**fields), text=None)


# --- Mock transcripts and the extraction a faithful model returns for them ---

COMPLETE_TRANSCRIPT = """User: I'm 32 and take home 1.5 lakh a month.
Assistant: Thanks. Any loans?
User: A car loan, EMI is 18k, about 6 lakh still outstanding.
User: I have 8 lakh in FDs and mutual funds, and a 25k SIP running.
User: My term cover is 1 crore. I want to buy a 1.2 crore flat in 7 years. I'm fine with moderate risk."""

COMPLETE_EXTRACTION = dict(
    age=32, net_monthly_income=150_000, existing_emis=18_000, liquid_assets=800_000,
    outstanding_liabilities=600_000, current_monthly_sip=25_000, current_life_cover=10_000_000,
    target_goal_cost_today=12_000_000, goal_horizon_years=7, risk_tolerance="moderate",
)

PARTIAL_TRANSCRIPT = """User: Hi, I'm 29 and want to buy a house worth 90 lakh.
User: I earn about 1.1 lakh per month after tax."""

PARTIAL_EXTRACTION = dict(age=29, net_monthly_income=110_000, target_goal_cost_today=9_000_000)


# --- Extraction pipeline ---

def test_complete_transcript_yields_profile():
    fake = FakeClient(parsed(**COMPLETE_EXTRACTION))
    profile, questions = extract_financial_profile(COMPLETE_TRANSCRIPT, client=fake, today=TODAY)
    assert questions == []
    assert profile == UserProfileSchema(**COMPLETE_EXTRACTION)


def test_extraction_request_enforces_schema_and_determinism():
    fake = FakeClient(parsed(**COMPLETE_EXTRACTION))
    extract_financial_profile(COMPLETE_TRANSCRIPT, client=fake, today=TODAY)
    call = fake.models.calls[0]
    assert call.model == MODEL
    assert call.contents == COMPLETE_TRANSCRIPT
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is ExtractedProfile
    assert call.config.temperature == 0.0
    assert "Never guess" in call.config.system_instruction


def test_missing_fields_produce_ordered_questions():
    fake = FakeClient(parsed(**PARTIAL_EXTRACTION))
    profile, questions = extract_financial_profile(PARTIAL_TRANSCRIPT, client=fake, today=TODAY)
    assert profile is None
    missing = ["goal_horizon_years", "risk_tolerance", "existing_emis", "outstanding_liabilities",
               "liquid_assets", "current_monthly_sip", "current_life_cover"]
    assert questions == [CLARIFYING_QUESTIONS[f] for f in missing]


def test_explicit_zero_is_not_treated_as_missing():
    extraction = {**COMPLETE_EXTRACTION, "existing_emis": 0, "outstanding_liabilities": 0,
                  "current_life_cover": 0}
    profile, questions = build_profile(ExtractedProfile(**extraction), TODAY)
    assert questions == []
    assert profile.existing_emis == 0 and profile.current_life_cover == 0


def test_annual_income_converted_in_python():
    extraction = {**COMPLETE_EXTRACTION, "net_monthly_income": None, "net_annual_income": 1_800_000}
    profile, _ = build_profile(ExtractedProfile(**extraction), TODAY)
    assert profile.net_monthly_income == 150_000


def test_monthly_income_takes_precedence_over_annual():
    extraction = {**COMPLETE_EXTRACTION, "net_annual_income": 2_400_000}
    profile, _ = build_profile(ExtractedProfile(**extraction), TODAY)
    assert profile.net_monthly_income == 150_000


def test_target_year_converted_to_horizon():
    extraction = {**COMPLETE_EXTRACTION, "goal_horizon_years": None, "goal_target_year": 2031}
    profile, _ = build_profile(ExtractedProfile(**extraction), TODAY)
    assert profile.goal_horizon_years == 5


def test_target_year_in_the_past_asks_for_confirmation():
    extraction = {**COMPLETE_EXTRACTION, "goal_horizon_years": None, "goal_target_year": 2025}
    profile, questions = build_profile(ExtractedProfile(**extraction), TODAY)
    assert profile is None
    assert questions == ["Could you confirm how many years away your goal is (between 1 and 40)?"]


def test_out_of_range_age_asks_for_confirmation():
    profile, questions = build_profile(ExtractedProfile(**{**COMPLETE_EXTRACTION, "age": 75}), TODAY)
    assert profile is None
    assert questions == ["Could you confirm your age? We support planning for ages 18 to 70."]


def test_negative_amount_asks_generic_confirmation():
    extraction = {**COMPLETE_EXTRACTION, "liquid_assets": -5}
    _, questions = build_profile(ExtractedProfile(**extraction), TODAY)
    assert questions == ["Could you double-check the value you gave for liquid assets?"]


def test_missing_and_invalid_fields_reported_together():
    extraction = {**COMPLETE_EXTRACTION, "age": 12, "risk_tolerance": None}
    _, questions = build_profile(ExtractedProfile(**extraction), TODAY)
    assert questions == [CLARIFYING_QUESTIONS["risk_tolerance"],
                         "Could you confirm your age? We support planning for ages 18 to 70."]


def test_empty_transcript_asks_everything():
    fake = FakeClient(parsed())
    profile, questions = extract_financial_profile("User: hello", client=fake, today=TODAY)
    assert profile is None
    assert questions == list(CLARIFYING_QUESTIONS.values())


def test_falls_back_to_json_text_when_parsed_missing():
    response = SimpleNamespace(parsed=None, text=ExtractedProfile(**COMPLETE_EXTRACTION).model_dump_json())
    profile, questions = extract_financial_profile(COMPLETE_TRANSCRIPT, client=FakeClient(response),
                                                   today=TODAY)
    assert questions == [] and profile.age == 32


def test_default_today_is_used():
    profile, _ = build_profile(ExtractedProfile(**COMPLETE_EXTRACTION))
    assert profile.goal_horizon_years == 7


# --- Client setup ---

def test_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(client_module.MissingApiKeyError):
        client_module.get_client()


def test_client_reads_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    c = client_module.get_client()
    assert c is client_module.get_client()  # cached per key


def test_extractor_uses_env_client(monkeypatch):
    fake = FakeClient(parsed(**COMPLETE_EXTRACTION))
    monkeypatch.setattr("app.agent.extractor.get_client", lambda: fake)
    profile, _ = extract_financial_profile(COMPLETE_TRANSCRIPT, today=TODAY)
    assert profile.age == 32


# --- Explainer ---

PROFILE = UserProfileSchema(**COMPLETE_EXTRACTION)
RESULTS = {
    "foir_ratio_percent": 12,
    "max_additional_emi": 57_000,
    "human_life_value": 20_153_923.4,
    "net_required_term_cover": 11_753_923.4,
    "goal_success_probability_percent": 41.3,
    "required_monthly_sip": 48_200,
}


def text(t):
    return SimpleNamespace(text=t)


GOOD_SUMMARY = ("Current EMIs of 18,000 leave repayment capacity of 57,000 per month. "
                "The human life value is 2,01,53,923, leaving a protection gap of 1,17,53,923. "
                "The goal succeeds in 41.3 percent of simulated paths; a SIP of 48,200 is required.")


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0"), (999, "999"), (1_000, "1,000"), (150_000, "1,50,000"), (12_345_678.9, "1,23,45,678.9"),
     (41.333, "41.33"), (0.5, "0.5"), (-250_000, "-2,50,000")],
)
def test_format_number_indian_grouping(value, expected):
    assert format_number(value) == expected


def test_explanation_appends_disclaimer_and_passes_formatted_figures():
    fake = FakeClient(text(GOOD_SUMMARY))
    out = generate_plan_explanation(PROFILE, RESULTS, client=fake)
    assert out == f"{GOOD_SUMMARY}\n\n{DISCLAIMER}"
    call = fake.models.calls[0]
    assert "Calculated results:\n  Foir ratio percent: 12\n" in call.contents
    assert "  Human life value: 2,01,53,923\n" in call.contents  # whole rupees
    assert "  Net monthly income: 1,50,000\n" in call.contents
    assert "  Goal success probability percent: 41.3\n" in call.contents  # small values keep decimals
    assert '"' not in call.contents
    assert "Never salesy" in call.config.system_instruction
    assert "Never calculate" in call.config.system_instruction


def test_disclaimer_text_is_exact():
    assert DISCLAIMER == (
        "Simulated projections based on IALM 2012-14 actuarial benchmarks and standard banking "
        "FOIR guidelines. This platform is an educational simulation tool, not an authorized SEBI "
        "or IRDAI financial advisor."
    )


def test_untraceable_numbers_detected():
    source = '{"sip": "48,200", "p": "41.3"}'
    assert untraceable_numbers("SIP of 48,200 gives 41.3 percent in 3 cases", source) == set()
    assert untraceable_numbers("SIP of 48,200 would roughly double to 96,400", source) == {96_400}
    assert untraceable_numbers("Based on IALM 2012-14 tables", source) == set()


def test_invented_number_triggers_retry():
    fake = FakeClient(text("Your SIP should rise by about 23,000 per month."), text(GOOD_SUMMARY))
    out = generate_plan_explanation(PROFILE, RESULTS, client=fake)
    assert out.startswith(GOOD_SUMMARY)
    assert len(fake.models.calls) == 2
    assert "23,000" in fake.models.calls[1].contents


def test_repeated_invention_falls_back_to_deterministic_summary():
    bad = text("Returns of 14% will make you a crorepati by 2040.")
    out = generate_plan_explanation(PROFILE, RESULTS, client=FakeClient(bad, bad))
    assert out.startswith("Summary of calculated results:")
    assert "\nHuman life value: 2,01,53,923\n" in out
    assert "\nGoal success probability percent: 41.3\n" in out
    assert out.endswith(DISCLAIMER)


def test_empty_reply_retries_then_falls_back():
    fake = FakeClient(text(""), text(None))
    out = generate_plan_explanation(PROFILE, RESULTS, client=fake)
    assert len(fake.models.calls) == 2
    assert fake.models.calls[1].contents == fake.models.calls[0].contents
    assert out.startswith("Summary of calculated results:")


def test_scalar_lists_are_rendered_as_bullets():
    fake = FakeClient(text("Two scenarios were run."))
    generate_plan_explanation(PROFILE, {"scenarios": ["base", "stress"]}, client=fake)
    assert "  Scenarios:\n    - base\n    - stress" in fake.models.calls[0].contents


def test_nested_and_non_numeric_results_are_passed_through():
    results = {"tiers": [{"ltv": 0.8, "ok": True}], "note": "within limits"}
    fake = FakeClient(text("The loan is within limits."))
    generate_plan_explanation(PROFILE, results, client=fake)
    contents = fake.models.calls[0].contents
    assert "  Tiers:\n    -\n      Ltv: 0.8\n      Ok: Yes\n" in contents
    assert "  Note: within limits" in contents


def test_explainer_uses_env_client(monkeypatch):
    fake = FakeClient(text(GOOD_SUMMARY))
    monkeypatch.setattr("app.agent.explainer.get_client", lambda: fake)
    assert generate_plan_explanation(PROFILE, RESULTS).endswith(DISCLAIMER)


# --- Live extraction fidelity (runs only when GEMINI_API_KEY is set) ---

live = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")


@live
def test_live_complete_transcript():
    profile, questions = extract_financial_profile(COMPLETE_TRANSCRIPT, today=TODAY)
    assert questions == []
    assert profile == UserProfileSchema(**COMPLETE_EXTRACTION)


@live
def test_live_partial_transcript_does_not_invent_values():
    profile, questions = extract_financial_profile(PARTIAL_TRANSCRIPT, today=TODAY)
    assert profile is None
    assert CLARIFYING_QUESTIONS["risk_tolerance"] in questions
    assert CLARIFYING_QUESTIONS["existing_emis"] in questions


@live
def test_live_annual_income_and_target_year():
    transcript = COMPLETE_TRANSCRIPT.replace("take home 1.5 lakh a month", "earn 18 LPA post tax")
    transcript = transcript.replace("in 7 years", "in 2031")
    profile, _ = extract_financial_profile(transcript, today=TODAY)
    assert profile.net_monthly_income == 150_000
    assert profile.goal_horizon_years == 5


# --- Model fallback on overload ---

from google.genai import errors as genai_errors  # noqa: E402


def _api_error(code):
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": "x", "status": "X"}}, None)


class ScriptedModels:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.tried = []

    def generate_content(self, *, model, contents, config):
        self.tried.append(model)
        outcome = self.outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _scripted(**outcomes):
    return SimpleNamespace(models=ScriptedModels({k.replace("_", "-"): v for k, v in outcomes.items()}))


@pytest.mark.parametrize("code", [429, 500, 503])
def test_transient_error_falls_back_to_next_model(code):
    fake = _scripted(primary=_api_error(code), backup="ok")
    assert client_module.generate_content(fake, "hi", None, models=("primary", "backup")) == "ok"
    assert fake.models.tried == ["primary", "backup"]


def test_config_errors_are_not_retried_on_other_models():
    fake = _scripted(primary=_api_error(404), backup="ok")
    with pytest.raises(genai_errors.ClientError):
        client_module.generate_content(fake, "hi", None, models=("primary", "backup"))
    assert fake.models.tried == ["primary"]


def test_last_model_error_is_raised():
    fake = _scripted(primary=_api_error(503), backup=_api_error(503))
    with pytest.raises(genai_errors.ServerError):
        client_module.generate_content(fake, "hi", None, models=("primary", "backup"))
    assert fake.models.tried == ["primary", "backup"]


def test_default_candidates_are_primary_then_fallbacks(monkeypatch):
    monkeypatch.setattr(client_module, "MODEL", "primary")
    monkeypatch.setattr(client_module, "FALLBACK_MODELS", ("backup",))
    fake = _scripted(primary=_api_error(503), backup="ok")
    assert client_module.generate_content(fake, "hi", None) == "ok"
    assert fake.models.tried == ["primary", "backup"]
