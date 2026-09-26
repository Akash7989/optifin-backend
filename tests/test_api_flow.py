"""End-to-end API flow: mock transcript -> extraction -> calculation -> explanation.

Gemini is replaced by a scripted fake client; everything else (FastAPI routing, validation,
extraction post-processing, engines, product matching, explainer guard) runs for real.
Run with -s to see the printed response payloads.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.agent.explainer import DISCLAIMER
from app.main import app
from app.models.profile import ExtractedProfile

TRANSCRIPT = [
    {"role": "user", "content": "I'm 32 and take home 1.5 lakh a month."},
    {"role": "assistant", "content": "Thanks. Do you have any loans?"},
    {"role": "user", "content": "A car loan, EMI 18k, about 6 lakh outstanding."},
    {"role": "user", "content": "8 lakh in FDs and mutual funds, and a 25k SIP."},
    {"role": "user", "content": "Term cover of 1 crore. I want a 1.2 crore flat in 7 years. "
                                "Moderate risk is fine."},
]

EXTRACTED = dict(
    age=32, net_monthly_income=150_000, existing_emis=18_000, liquid_assets=800_000,
    outstanding_liabilities=600_000, current_monthly_sip=25_000, current_life_cover=10_000_000,
    target_goal_cost_today=12_000_000, goal_horizon_years=7, risk_tolerance="moderate",
)


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def install_fake_llm(monkeypatch, *responses):
    fake = SimpleNamespace(models=FakeModels(responses))
    monkeypatch.setattr("app.agent.extractor.get_client", lambda: fake)
    monkeypatch.setattr("app.agent.explainer.get_client", lambda: fake)
    return fake


def show(title, payload):
    print(f"\n===== {title} =====\n{json.dumps(payload, indent=2, ensure_ascii=False)}")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def no_model_fallbacks(monkeypatch):
    # Keep the scripted fake independent of any fallback models set in the local .env.
    monkeypatch.setattr("app.agent.client.FALLBACK_MODELS", ())


# --- Complete pipeline ---

def test_full_pipeline(client, monkeypatch):
    fake = install_fake_llm(
        monkeypatch,
        SimpleNamespace(parsed=ExtractedProfile(**EXTRACTED), text=None),
        SimpleNamespace(text="Current EMIs of 18,000 are within the FOIR limit. The goal success "
                             "probability is 12.19 at the current SIP of 25,000."),
    )

    # 1. Extraction
    ingest = client.post("/api/chat/ingest", json={"messages": TRANSCRIPT})
    assert ingest.status_code == 200
    body = ingest.json()
    show("POST /api/chat/ingest", body)
    assert body["status"] == "ready" and body["questions"] == []
    assert body["extracted"] == EXTRACTED
    assert fake.models.calls[0].contents.startswith("User: I'm 32")
    assert "\nAssistant: Thanks." in fake.models.calls[0].contents

    # 2. Calculation
    run = client.post("/api/simulate/run", json=body["profile"])
    assert run.status_code == 200
    results = run.json()
    show("POST /api/simulate/run", results)

    loan, insurance, goal = results["loan"], results["insurance"], results["goal"]
    assert set(results) == {"assumptions", "loan", "insurance", "goal", "downpayment_funding",
                            "products", "notes"}
    funding = results["downpayment_funding"]
    assert funding["median_shortfall"] == pytest.approx(
        goal["target_downpayment"] - goal["wealth_p50"])
    assert funding["liquid_assets_today"] == 800_000
    trajectory = goal["trajectory"]
    assert [p["year"] for p in trajectory] == list(range(8))
    assert trajectory[0]["p50"] == pytest.approx(800_000)
    assert trajectory[-1]["p50"] == pytest.approx(goal["wealth_p50"])
    assert insurance["outstanding_liabilities"] == 600_000
    assert insurance["gross_need"] == pytest.approx(
        insurance["human_life_value"] + insurance["outstanding_liabilities"]
        + insurance["future_goal_commitments"])
    assert loan["eligible_loan"] <= loan["max_loan_by_ltv"] == pytest.approx(
        loan["ltv_cap"] * loan["property_value"])
    assert loan["ltv_cap"] == 0.75  # property above Rs 75 lakh
    assert goal["future_target_cost"] == pytest.approx(loan["property_value"])
    assert goal["target_downpayment"] == pytest.approx(loan["required_downpayment"])
    assert results["assumptions"]["downpayment_ratio"] >= 0.20
    assert insurance["net_required_cover"] == pytest.approx(
        insurance["gross_need"] - insurance["available_resources"])
    assert 0 <= goal["success_probability"] <= 100
    assert goal["recommended_sip_delta"] > 0
    assert results["products"]["illustrative"] is True
    assert [o["lender"] for o in results["products"]["home_loans"]] == ["SBI", "HDFC Bank",
                                                                        "ICICI Bank"]
    assert results["assumptions"]["home_loan_rate"] == results["products"]["home_loans"][0][
        "interest_rate"]
    assert len(results["products"]["term_insurance"]) == 3

    # Same profile -> identical projections (fixed simulation seed).
    assert client.post("/api/simulate/run", json=body["profile"]).json() == results

    # 3. Explanation
    explain = client.post("/api/simulate/explain",
                          json={"profile": body["profile"], "results": results})
    assert explain.status_code == 200
    narrative = explain.json()["narrative_explanation"]
    show("POST /api/simulate/explain", explain.json())
    assert narrative.endswith(DISCLAIMER)
    assert "12.19" in fake.models.calls[1].contents


# --- Ingest ---

def test_ingest_incomplete_returns_questions(client, monkeypatch):
    install_fake_llm(monkeypatch, SimpleNamespace(
        parsed=ExtractedProfile(age=29, net_monthly_income=110_000), text=None))
    response = client.post("/api/chat/ingest",
                           json={"messages": [{"role": "user", "content": "I'm 29, earn 1.1L"}]})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "incomplete" and body["profile"] is None
    assert body["questions"][0] == "What would your goal cost if you bought it today?"
    assert body["extracted"] == {"age": 29, "net_monthly_income": 110_000}


@pytest.mark.parametrize(
    "payload",
    [{}, {"messages": []}, {"messages": [{"role": "system", "content": "x"}]},
     {"messages": [{"role": "user", "content": ""}]}],
)
def test_ingest_rejects_malformed_messages(client, payload):
    response = client.post("/api/chat/ingest", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --- Simulation edge cases ---

def test_simulate_rejects_invalid_profile(client):
    response = client.post("/api/simulate/run", json={**EXTRACTED, "age": 12})
    error = response.json()["error"]
    assert response.status_code == 422
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"] == ["body", "age"]


def test_simulate_low_income_and_older_profile_adds_notes(client):
    profile = {**EXTRACTED, "age": 55, "net_monthly_income": 20_000, "existing_emis": 0,
               "current_life_cover": 0, "goal_horizon_years": 3}
    results = client.post("/api/simulate/run", json=profile).json()
    assert results["products"]["home_loans"] == []
    assert results["products"]["term_insurance"] == []
    assert results["assumptions"]["home_loan_rate"] == pytest.approx(0.065 + 0.03)
    assert any("No catalog lender" in n for n in results["notes"])
    assert any("ages 25-50" in n for n in results["notes"])


def test_funding_shortfall_is_zero_when_median_exceeds_target(client):
    profile = {**EXTRACTED, "current_monthly_sip": 200_000}
    funding = client.post("/api/simulate/run", json=profile).json()["downpayment_funding"]
    assert funding["median_shortfall"] == 0


def test_simulate_fully_covered_profile_needs_no_term_quotes(client):
    profile = {**EXTRACTED, "current_life_cover": 100_000_000}
    results = client.post("/api/simulate/run", json=profile).json()
    assert results["insurance"]["net_required_cover"] == 0
    assert results["products"]["term_insurance"] == []
    assert any("no term quotes" in n for n in results["notes"])


# --- Error envelopes ---

def test_missing_api_key_returns_503_envelope(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/chat/ingest", json={"messages": TRANSCRIPT})
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "llm_not_configured", "details": None,
                                         "message": "The language model is not configured on "
                                                    "the server."}}


def test_llm_upstream_error_returns_502_envelope(client, monkeypatch):
    overloaded = genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}, None)
    install_fake_llm(monkeypatch, overloaded)
    response = client.post("/api/chat/ingest", json={"messages": TRANSCRIPT})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_upstream_error"
    assert response.json()["error"]["details"] == {"upstream_status": 503}


def test_engine_value_error_returns_422_envelope(client, monkeypatch):
    def boom(profile):
        raise ValueError("horizon too long")
    monkeypatch.setattr("app.main.run_simulation", boom)
    response = client.post("/api/simulate/run", json=EXTRACTED)
    assert response.status_code == 422
    assert response.json()["error"] == {"code": "invalid_input", "message": "horizon too long",
                                        "details": None}


def test_unexpected_error_returns_500_without_leaking(monkeypatch):
    def boom(profile):
        raise RuntimeError("secret internals")
    monkeypatch.setattr("app.main.run_simulation", boom)
    response = TestClient(app, raise_server_exceptions=False).post("/api/simulate/run",
                                                                   json=EXTRACTED)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text


def test_unknown_route_returns_envelope(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_404"


# --- Health and CORS ---

def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_cors_allows_frontend_origin(client):
    response = client.options("/api/simulate/run", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_other_origins(client):
    response = client.options("/api/simulate/run", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in response.headers
