"""Downloading the Midcap-100 broker PDFs inside Trendlyne's quota.

Trendlyne allows this account 70 report opens a day and 300 a month, and counts every request,
retries included. Over the limit, every report link redirects to /member/daily-runout/ - an HTML
page, not a PDF. A fetch that ignores this spends the day's allowance, then hammers that page.
"""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_midcap_reports as fm  # noqa: E402

PDF = b"%PDF-1.7\n" + b"x" * 20_000
RUNOUT_URL = "https://trendlyne.com/member/daily-runout/"
_ids = itertools.count(90_000)


class Resp:
    def __init__(self, body: bytes, url: str, ctype: str):
        self.content, self.url, self.headers = body, url, {"Content-Type": ctype}


class FakeTrendlyne:
    """Serves a PDF for each report until `allowance` opens are used, then the runout page."""

    def __init__(self, allowance: int = 10_000):
        self.allowance = allowance
        self.requested: list[str] = []
        self.headers: dict = {}

    def get(self, url, timeout=None, allow_redirects=True):
        self.requested.append(url)
        if len(self.requested) > self.allowance:
            return Resp(b"<html><title>Daily limit reached | Trendlyne</title></html>",
                        RUNOUT_URL, "text/html; charset=utf-8")
        return Resp(PDF, "https://trendlyne-media-mumbai-new.s3.amazonaws.com/r.pdf",
                    "application/pdf")


def row(ticker: str, no: int, day: str, broker: str = "Motilal Oswal") -> fm.Row:
    rid = next(_ids)
    return fm.Row(batch=None, idx=None, company=ticker.title(), ticker=ticker, report_no=no,
                  report_date=day, broker=broker, rating="Buy", target=100.0, change_vs_prev=None,
                  headline=None, filename=f"{ticker}_{no}_{day}.pdf",
                  url=f"https://trendlyne.com/get-document/report/pdf/{rid}/")


@pytest.fixture(autouse=True)
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "ROOT", tmp_path)
    monkeypatch.setattr(fm, "OUT", tmp_path / "Broker_Reports")


def on_disk(rows) -> int:
    return sum(1 for r in rows if fm.target_path(r).exists())


def test_the_daily_runout_page_stops_the_run_at_once():
    rows = [row(t, 1, "2026-09-01") for t in ("AAA", "BBB", "CCC", "DDD")]
    site = FakeTrendlyne(allowance=1)

    tally = fm.fetch(rows, None, None, delay=0, session=site)

    assert len(site.requested) == 2          # one report, then the runout page - nothing after it
    assert tally["saved"] == 1
    assert on_disk(rows) == 1                # the runout page is never saved under a .pdf name


def test_a_run_never_opens_more_reports_than_trendlyne_allows_in_a_day():
    rows = [row(f"C{i:03d}", 1, "2026-09-01") for i in range(150)]
    site = FakeTrendlyne()                   # would happily serve all 150

    fm.fetch(rows, None, None, delay=0, session=site)

    assert 0 < len(site.requested) <= 70


def test_companies_with_no_report_on_disk_come_before_anyone_s_second_report():
    have = row("AAA", 1, "2026-09-10")
    rows = [have, row("AAA", 2, "2026-08-01"),
            row("BBB", 1, "2026-09-05"), row("BBB", 2, "2026-07-01"), row("BBB", 3, "2026-06-01"),
            row("CCC", 1, "2026-09-20")]
    fm.target_path(have).parent.mkdir(parents=True)
    fm.target_path(have).write_bytes(PDF)
    site = FakeTrendlyne()

    fm.fetch(rows, None, None, delay=0, limit=2, session=site)

    assert set(site.requested) == {rows[2].url, rows[5].url}    # BBB #1 and CCC #1


def test_reports_on_a_dead_host_are_never_requested():
    rows = [row("AAA", 1, "2026-09-01", broker="Geojit BNP Paribas"), row("BBB", 1, "2026-09-01")]
    site = FakeTrendlyne()

    fm.fetch(rows, None, None, delay=0, session=site)

    assert site.requested == [rows[1].url]
