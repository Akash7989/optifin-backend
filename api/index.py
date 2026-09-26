"""Vercel serverless entry point: exposes the FastAPI app as an ASGI function."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402,F401
