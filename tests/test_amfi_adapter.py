import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.adapters.amfi_adapter import (
    AMFI_NAV_URL,
    AmfiError,
    extract_benchmarks,
    get_benchmark_navs,
    parse_nav_text,
)

HEADER = ("Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;"
          "Plan;Option;Net Asset Value;Date")

# Excerpt shaped like the live NAVAll.txt feed (8-column format).
SAMPLE_FEED = f"""{HEADER}

Open Ended Schemes(Debt Scheme - Liquid Fund)

SBI Mutual Fund

119800;INF200K01UT4;-;SBI LIQUID FUND;Direct Plan;Growth;4450.6576;27-Sep-2026
119801;INF200K01UU2;-;SBI LIQUID FUND;Direct Plan;IDCW;N.A.;27-Sep-2026

Open Ended Schemes(Other Scheme - Index Funds)

ICICI Prudential Mutual Fund

120684;INF109K01Y80;-;ICICI Prudential Nifty Next 50 Index Fund;Direct Plan;Growth;66.8167;25-Sep-2026

UTI Mutual Fund

120716;INF789F01XA0;-;UTI Nifty 50 Index Fund;Direct Plan;Growth;162.9607;25-Sep-2026
999999;INF000000000;-;Broken Row;Direct Plan;Growth;12.34
888888;INF000000001;-;Bad Date Fund;Direct Plan;Growth;10.00;2026-09-25
"""

LEGACY_FEED = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
Open Ended Schemes(Debt Scheme - Liquid Fund)
SBI Mutual Fund
119800;INF200K01UT4;-;SBI Liquid Fund - Direct Plan - Growth;4000.1234;02-Jan-2025
"""


# --- Parser ---

def test_parse_extracts_fields_and_context():
    records = parse_nav_text(SAMPLE_FEED)
    liquid = records["119800"]
    assert liquid.scheme_name == "SBI LIQUID FUND"
    assert (liquid.plan, liquid.option) == ("Direct Plan", "Growth")
    assert liquid.nav == 4450.6576
    assert liquid.nav_date == date(2026, 9, 27)
    assert liquid.category == "Debt Scheme - Liquid Fund"
    assert liquid.fund_house == "SBI Mutual Fund"
    assert records["120716"].fund_house == "UTI Mutual Fund"
    assert records["120716"].category == "Other Scheme - Index Funds"


def test_parse_skips_na_short_and_malformed_rows():
    records = parse_nav_text(SAMPLE_FEED)
    assert "119801" not in records  # N.A. NAV
    assert "999999" not in records  # missing column
    assert "888888" not in records  # unparseable date
    assert set(records) == {"119800", "120684", "120716"}


def test_parse_legacy_six_column_format():
    rec = parse_nav_text(LEGACY_FEED)["119800"]
    assert rec.nav == 4000.1234 and rec.nav_date == date(2025, 1, 2)
    assert rec.plan is None and rec.option is None


def test_parse_close_ended_category_without_parentheses():
    feed = f"{HEADER}\nClose Ended Schemes\nSome AMC\n1;X;-;Fund;Direct Plan;Growth;10;01-Jan-2026\n"
    assert parse_nav_text(feed)["1"].category == "Close Ended Schemes"


def test_parse_rejects_missing_header():
    with pytest.raises(AmfiError, match="header row not found"):
        parse_nav_text("<html>Service unavailable</html>")


def test_parse_rejects_header_without_nav_column():
    with pytest.raises(AmfiError, match="Net Asset Value"):
        parse_nav_text("Scheme Code;Scheme Name;Date\n")


def test_extract_benchmarks():
    navs = extract_benchmarks(parse_nav_text(SAMPLE_FEED))
    assert {k: v.nav for k, v in navs.items()} == {
        "nifty_50_index": 162.9607,
        "nifty_next_50_index": 66.8167,
        "liquid": 4450.6576,
    }


def test_extract_benchmarks_reports_missing_scheme():
    records = parse_nav_text(SAMPLE_FEED)
    del records["120684"]
    with pytest.raises(AmfiError, match="nifty_next_50_index"):
        extract_benchmarks(records)


# --- Cache behaviour ---

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def _client(body: str = SAMPLE_FEED, status: int = 200, calls: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, text=body)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _failing_client():
    def handler(request):
        raise httpx.ConnectError("network down", request=request)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(**kwargs):
    return asyncio.run(get_benchmark_navs(**kwargs))


def test_fetch_writes_cache(tmp_path):
    cache, calls = tmp_path / "nav_cache.json", []
    navs = _run(cache_path=cache, client=_client(calls=calls), now=NOW)
    assert calls == [AMFI_NAV_URL]
    assert navs["liquid"].nav == 4450.6576
    payload = json.loads(cache.read_text())
    assert payload["fetched_at"] == NOW.isoformat()
    assert payload["navs"]["nifty_50_index"]["nav"] == 162.9607


def test_fresh_cache_skips_network(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client(), now=NOW)
    calls = []
    navs = _run(cache_path=cache, client=_client(calls=calls), now=NOW + timedelta(hours=23))
    assert calls == []
    assert navs["nifty_next_50_index"].nav == 66.8167


def test_expired_cache_refetches(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client(), now=NOW)
    newer = SAMPLE_FEED.replace("4450.6576;27-Sep-2026", "4451.0000;28-Sep-2026")
    calls = []
    later = NOW + timedelta(hours=24)
    navs = _run(cache_path=cache, client=_client(newer, calls=calls), now=later)
    assert len(calls) == 1
    assert navs["liquid"].nav == 4451.0
    assert json.loads(cache.read_text())["fetched_at"] == later.isoformat()


def test_network_failure_falls_back_to_stale_cache(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client(), now=NOW)
    navs = _run(cache_path=cache, client=_failing_client(), now=NOW + timedelta(days=3))
    assert navs["liquid"].nav == 4450.6576


def test_http_error_falls_back_to_stale_cache(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client(), now=NOW)
    navs = _run(cache_path=cache, client=_client("down", status=503), now=NOW + timedelta(days=2))
    assert navs["nifty_50_index"].nav == 162.9607


def test_network_failure_without_cache_raises(tmp_path):
    with pytest.raises(httpx.ConnectError):
        _run(cache_path=tmp_path / "nav_cache.json", client=_failing_client(), now=NOW)


def test_bad_feed_without_cache_raises(tmp_path):
    with pytest.raises(AmfiError):
        _run(cache_path=tmp_path / "nav_cache.json", client=_client("garbage"), now=NOW)


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "nav_cache.json"
    cache.write_text("{not json")
    calls = []
    navs = _run(cache_path=cache, client=_client(calls=calls), now=NOW)
    assert len(calls) == 1 and navs["liquid"].nav == 4450.6576


def test_cache_missing_a_benchmark_is_ignored(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client(), now=NOW)
    payload = json.loads(cache.read_text())
    del payload["navs"]["liquid"]
    cache.write_text(json.dumps(payload))
    calls = []
    _run(cache_path=cache, client=_client(calls=calls), now=NOW)
    assert len(calls) == 1


def test_default_client_is_created(tmp_path, monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=SAMPLE_FEED))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=transport, **kw))
    navs = _run(cache_path=tmp_path / "nav_cache.json", now=NOW)
    assert navs["liquid"].nav == 4450.6576


def test_default_now_uses_current_time(tmp_path):
    cache = tmp_path / "nav_cache.json"
    _run(cache_path=cache, client=_client())
    fetched_at = datetime.fromisoformat(json.loads(cache.read_text())["fetched_at"])
    assert datetime.now(timezone.utc) - fetched_at < timedelta(minutes=1)
