"""AMFI India NAV adapter: parses the public daily NAV file and caches benchmark NAVs.

Feed format (semicolon-separated, with section and fund-house lines between blocks):

    Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date
    Open Ended Schemes(Debt Scheme - Liquid Fund)
    SBI Mutual Fund
    119800;INF200K01UT4;-;SBI LIQUID FUND;Direct Plan;Growth;4450.6576;27-Sep-2026

Older files omit the Plan and Option columns; columns are located by header name.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from app.config import DATA_DIR

AMFI_NAV_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
NAV_CACHE_PATH = DATA_DIR / "nav_cache.json"
CACHE_TTL = timedelta(hours=24)
REQUEST_TIMEOUT_SECONDS = 30

# Benchmarks identified by AMFI scheme code (stable across scheme renames).
BENCHMARK_SCHEMES: dict[str, str] = {
    "nifty_50_index": "120716",  # UTI Nifty 50 Index Fund - Direct Plan - Growth
    "nifty_next_50_index": "120684",  # ICICI Prudential Nifty Next 50 Index Fund - Direct - Growth
    "liquid": "119800",  # SBI Liquid Fund - Direct Plan - Growth
}


class AmfiError(RuntimeError):
    pass


class NavRecord(BaseModel):
    scheme_code: str
    scheme_name: str
    plan: str | None = None
    option: str | None = None
    nav: float
    nav_date: date
    category: str | None = None
    fund_house: str | None = None


def parse_nav_text(text: str) -> dict[str, NavRecord]:
    """Parse the AMFI NAV file into {scheme_code: NavRecord}; rows without a numeric NAV are skipped."""
    lines = iter(text.splitlines())
    header = next((line for line in lines if line.startswith("Scheme Code")), None)
    if header is None:
        raise AmfiError("AMFI NAV header row not found")
    columns = [c.strip() for c in header.split(";")]
    idx = {name: columns.index(name) if name in columns else None for name in
           ("Scheme Code", "Scheme Name", "Plan", "Option", "Net Asset Value", "Date")}
    for required in ("Scheme Code", "Scheme Name", "Net Asset Value", "Date"):
        if idx[required] is None:
            raise AmfiError(f"AMFI NAV header missing column {required!r}")

    records: dict[str, NavRecord] = {}
    category = fund_house = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if ";" not in line:
            if line.startswith(("Open Ended", "Close Ended", "Interval Fund")):
                category = line[line.find("(") + 1 : line.rfind(")")] if "(" in line else line
            else:
                fund_house = line
            continue
        fields = [f.strip() for f in line.split(";")]
        if len(fields) != len(columns):
            continue
        try:
            nav = float(fields[idx["Net Asset Value"]])
            nav_date = datetime.strptime(fields[idx["Date"]], "%d-%b-%Y").date()
        except ValueError:
            continue  # "N.A." NAVs or malformed dates
        code = fields[idx["Scheme Code"]]
        records[code] = NavRecord(
            scheme_code=code,
            scheme_name=fields[idx["Scheme Name"]],
            plan=fields[idx["Plan"]] if idx["Plan"] is not None else None,
            option=fields[idx["Option"]] if idx["Option"] is not None else None,
            nav=nav,
            nav_date=nav_date,
            category=category,
            fund_house=fund_house,
        )
    return records


def extract_benchmarks(records: dict[str, NavRecord]) -> dict[str, NavRecord]:
    missing = [key for key, code in BENCHMARK_SCHEMES.items() if code not in records]
    if missing:
        raise AmfiError(f"benchmark schemes missing from AMFI feed: {', '.join(missing)}")
    return {key: records[code] for key, code in BENCHMARK_SCHEMES.items()}


def _read_cache(path: Path) -> tuple[datetime, dict[str, NavRecord]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        navs = {key: NavRecord.model_validate(v) for key, v in payload["navs"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if set(navs) != set(BENCHMARK_SCHEMES):
        return None
    return fetched_at, navs


def _write_cache(path: Path, fetched_at: datetime, navs: dict[str, NavRecord]) -> None:
    payload = {
        "source": AMFI_NAV_URL,
        "fetched_at": fetched_at.isoformat(),
        "navs": {key: rec.model_dump(mode="json") for key, rec in navs.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


async def _fetch_nav_text(client: httpx.AsyncClient | None) -> str:
    if client is not None:
        response = await client.get(AMFI_NAV_URL)
    else:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as own_client:
            response = await own_client.get(AMFI_NAV_URL)
    response.raise_for_status()
    return response.text


async def get_benchmark_navs(
    cache_path: Path = NAV_CACHE_PATH,
    ttl: timedelta = CACHE_TTL,
    client: httpx.AsyncClient | None = None,
    now: datetime | None = None,
) -> dict[str, NavRecord]:
    """Latest NAVs for the benchmark funds, served from a JSON cache for up to `ttl`.

    If the AMFI fetch fails and a stale cache exists, the stale values are returned.
    """
    now = now or datetime.now(timezone.utc)
    cached = _read_cache(cache_path)
    if cached is not None and now - cached[0] < ttl:
        return cached[1]

    try:
        navs = extract_benchmarks(parse_nav_text(await _fetch_nav_text(client)))
    except (httpx.HTTPError, AmfiError):
        if cached is not None:
            return cached[1]
        raise

    _write_cache(cache_path, now, navs)
    return navs
