"""OptiFin HTTP API.

Run locally:  uvicorn app.main:app --reload   (from optifin-be/)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.genai import errors as genai_errors
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agent.client import MissingApiKeyError
from app.agent.explainer import generate_plan_explanation
from app.agent.extractor import extract_profile_state
from app.api.pipeline import run_simulation, transcript_from_messages
from app.models.api import (
    ErrorEnvelope,
    ExplainRequest,
    ExplainResponse,
    IngestRequest,
    IngestResponse,
)
from app.models.profile import UserProfileSchema

logger = logging.getLogger(__name__)

ERROR_RESPONSES = {code: {"model": ErrorEnvelope} for code in (422, 500, 502, 503)}

app = FastAPI(title="OptiFin API", version="0.1.0", responses=ERROR_RESPONSES)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Error envelopes: {"error": {"code", "message", "details"}} ---

def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details}}
    return JSONResponse(status_code=status, content=jsonable_encoder(body))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(422, "validation_error", "Request body failed validation.",
                  [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()])


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))


@app.exception_handler(ValueError)
async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
    return _error(422, "invalid_input", str(exc))


@app.exception_handler(MissingApiKeyError)
async def _llm_not_configured(request: Request, exc: MissingApiKeyError) -> JSONResponse:
    return _error(503, "llm_not_configured", "The language model is not configured on the server.")


@app.exception_handler(genai_errors.APIError)
async def _llm_upstream(request: Request, exc: genai_errors.APIError) -> JSONResponse:
    logger.warning("Gemini API error %s: %s", exc.code, exc.message)
    return _error(502, "llm_upstream_error", "The language model service is unavailable. Please retry.",
                  {"upstream_status": exc.code})


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal_error", "An unexpected error occurred.")


# --- Routes ---

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat/ingest", response_model=IngestResponse)
def ingest_chat(request: IngestRequest) -> IngestResponse:
    extracted, profile, questions = extract_profile_state(
        transcript_from_messages(request.messages)
    )
    return IngestResponse(
        status="ready" if profile else "incomplete",
        questions=questions,
        profile=profile,
        extracted=extracted.model_dump(exclude_none=True),
    )


@app.post("/api/simulate/run")
async def simulate_run(profile: UserProfileSchema) -> dict[str, Any]:
    return await run_simulation(profile)


@app.post("/api/simulate/explain", response_model=ExplainResponse)
def simulate_explain(request: ExplainRequest) -> ExplainResponse:
    return ExplainResponse(
        narrative_explanation=generate_plan_explanation(request.profile, request.results)
    )
