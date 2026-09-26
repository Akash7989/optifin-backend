"""Shared Google GenAI client configuration.

Settings (environment or optifin-be/.env):
- GEMINI_API_KEY: required.
- OPTIFIN_GEMINI_MODEL: primary model (default gemini-2.5-flash).
- OPTIFIN_GEMINI_FALLBACK_MODELS: comma-separated models tried in order when the previous
  one is overloaded or rate-limited. Configuration errors (bad key, unknown model) are
  never retried on another model.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

MODEL = os.environ.get("OPTIFIN_GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
FALLBACK_MODELS = tuple(
    m.strip() for m in os.environ.get("OPTIFIN_GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class MissingApiKeyError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _client_for_key(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    """Client authenticated with GEMINI_API_KEY from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError("GEMINI_API_KEY environment variable is not set")
    return _client_for_key(api_key)


def generate_content(
    client: genai.Client,
    contents: str,
    config: types.GenerateContentConfig,
    models: tuple[str, ...] | None = None,
):
    """generate_content on the primary model, falling back on overload or rate limits."""
    candidates = models or (MODEL, *FALLBACK_MODELS)
    for i, model in enumerate(candidates):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code not in TRANSIENT_STATUS_CODES or i == len(candidates) - 1:
                raise
            logger.warning("Gemini model %s unavailable (%s); trying %s",
                           model, exc.code, candidates[i + 1])
