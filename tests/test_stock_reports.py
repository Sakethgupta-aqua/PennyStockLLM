"""Trendlyne Stock Reports: dated monthly snapshots, filed from downloads or fetched with a cookie,
never overwritten, and never more opens than Trendlyne allows in a day."""
from datetime import date

import pytest

from pennystock.data import documents
from pennystock.data import stock_reports as sr

PDF = b"%PDF-1.7\n" + b"z" * 60_000 + b"\n%%EOF\n"


@pytest.fixture(autouse=True)
def lib(tmp_path, monkeypatch):
    monkeypatch.setattr(documents.settings, "market_db", tmp_path / "t.db")
    monkeypatch.setattr(documents.settings, "project_root", tmp_path)
    monkeypatch.setattr(documents.settings, "stock_reports_dir", tmp_path / "Stock_Reports")
    return tmp_path / "Stock_Reports"


@pytest.mark.parametrize("name, want", [
    ("AUBANK-StockReport-20261003-2318.pdf", ("AUBANK", "20261003-2318", None)),
    ("GVT&D-StockReport-20261003-2355__TL67.pdf", ("GVT&D", "20261003-2355", "67")),
    ("AUBANK-StockReport-20261003-2318 (1).pdf", None),
    ("report.pdf", None)])
def test_trendlyne_download_names_are_parsed(name, want):
    got = sr.parse_name(name)
    if want is None:
        assert got is None
    else:
        assert (got.symbol, got.stamp, got.tl_id) == want


def test_a_snapshot_is_filed_under_its_safe_folder_and_catalogued(lib):
    p = sr.file_snapshot(PDF, "GVT&D", "20261003-2355", "67")
    assert p == lib / "GVT_D" / "GVT_D-StockReport-20261003-2355.pdf"
    d = documents.get("stock_report:trendlyne:67:20261003-2355")
    assert (d.symbol, d.doc_date, d.kind) == ("GVT&D", "2026-10-03", "stock_report")


def test_a_newer_snapshot_never_replaces_an_older_one(lib):
    sr.file_snapshot(PDF, "AUBANK", "20261003-2318", "54902")
    sr.file_snapshot(PDF + b"new", "AUBANK", "20261104-0910", "54902")
    assert sorted(p.name for p in (lib / "AUBANK").glob("*.pdf")) == [
        "AUBANK-StockReport-20261003-2318.pdf", "AUBANK-StockReport-20261104-0910.pdf"]


def test_the_monthly_rule_sees_this_months_snapshot(lib):
    sr.file_snapshot(PDF, "AUBANK", "20261003-2318", "54902")
    assert sr.has_snapshot_in_month("AUBANK", date(2026, 10, 30))
    assert not sr.has_snapshot_in_month("AUBANK", date(2026, 11, 1))


def test_a_legacy_bundle_splits_into_its_reports():
    blob = (b"ASTRAL\t122\t" + str(len(PDF)).encode() + b"\tASTRAL-StockReport-20261003-2345.pdf\n" + PDF
            + b"ATGL\t114071\t" + str(len(PDF)).encode() + b"\tATGL-StockReport-20261003-2346.pdf\n" + PDF)
    got = sr.unpack_bundle(blob)
    assert [(s, i, n) for s, i, _, n in got] == [("ASTRAL", "122", "ASTRAL-StockReport-20261003-2345.pdf"),
                                                 ("ATGL", "114071", "ATGL-StockReport-20261003-2346.pdf")]
    assert all(b == PDF for _, _, b, _ in got)


def test_reports_already_on_disk_are_registered_from_the_index(lib):
    folder = lib / "GVT_D"
    folder.mkdir(parents=True)
    (folder / "GVT_D-StockReport-20261003-2355.pdf").write_bytes(PDF)
    (lib / "index.csv").write_text("symbol,trendlyne_id,generated,file,bytes,sha256,source_url\n"
                                   "GVT&D,67,2026-10-03 23:55,Stock_Reports/GVT_D/GVT_D-StockReport-20261003-2355.pdf,,,\n")
    assert sr.register_existing() == 1
    assert sr.register_existing() == 0
    assert documents.get("stock_report:trendlyne:67:20261003-2355").symbol == "GVT&D"


class Resp:
    def __init__(self, url, body=b"", headers=None):
        self.url, self.content, self.headers, self.status_code = url, body, headers or {}, 200


class FakeTrendlyne:
    """The research page redirects to /stock/<id>/...; get-pdf serves PDFs until `allowance` is
    spent, then every open lands on the daily-runout page."""

    def __init__(self, allowance=1000):
        self.allowance, self.pdf_opens, self.headers = allowance, 0, {}

    def get(self, url, timeout=None, allow_redirects=True):
        if "/research-reports/stock/" in url:
            sym = url.rstrip("/").split("/")[-1]
            return Resp(f"https://trendlyne.com/research-reports/stock/{100 + len(sym)}/{sym}/x/")
        self.pdf_opens += 1
        if self.pdf_opens > self.allowance:
            return Resp("https://trendlyne.com/member/daily-runout/", b"<html>limit</html>")
        return Resp(url, PDF, {"Content-Disposition": 'inline; filename="X-StockReport-20261004-0100.pdf"'})


def test_the_fetcher_stops_at_the_runout_page(lib):
    site = FakeTrendlyne(allowance=2)
    tally = sr.fetch(["A", "BB", "CCC", "DDDD"], site, date(2026, 10, 4), delay=0)
    assert site.pdf_opens == 3 and tally["saved"] == 2 and tally["daily_limit"] == 1


def test_the_fetcher_never_opens_more_than_the_daily_allowance(lib):
    site = FakeTrendlyne()
    sr.fetch([f"S{i}" for i in range(90)], site, date(2026, 10, 4), delay=0)
    assert 0 < site.pdf_opens <= 70


def test_the_fetcher_skips_a_company_with_a_snapshot_this_month(lib):
    sr.file_snapshot(PDF, "A", "20261002-0900", "1")
    site = FakeTrendlyne()
    tally = sr.fetch(["A", "BB"], site, date(2026, 10, 4), delay=0)
    assert site.pdf_opens == 1 and tally["this_month"] == 1


def test_the_index_lists_every_snapshot_newest_first_per_company(lib):
    sr.file_snapshot(PDF, "AUBANK", "20261003-2318", "54902")
    sr.file_snapshot(PDF + b"n", "AUBANK", "20261104-0910", "54902")
    sr.file_snapshot(PDF, "GVT&D", "20261003-2355", "67")
    lines = sr.write_index().read_text().splitlines()
    assert lines[0] == "symbol,trendlyne_id,generated,file,bytes,sha256,source_url"
    assert [ln.split(",")[:3] for ln in lines[1:]] == [["AUBANK", "54902", "2026-11-04 09:10"],
                                                      ["AUBANK", "54902", "2026-10-03 23:18"],
                                                      ["GVT&D", "67", "2026-10-03 23:55"]]


def test_reports_on_disk_count_even_when_the_catalog_was_rebuilt(lib):
    """market.db is rebuildable; a lost catalog must not make the fetcher spend opens on reports
    that are already on disk this month."""
    (lib / "A").mkdir(parents=True)
    (lib / "A" / "A-StockReport-20261002-0900.pdf").write_bytes(PDF)
    (lib / "index.csv").write_text("symbol,trendlyne_id,generated,file,bytes,sha256,source_url\n"
                                   "A,1,2026-10-02 09:00,Stock_Reports/A/A-StockReport-20261002-0900.pdf,,,\n")
    site = FakeTrendlyne()
    tally = sr.fetch(["A", "BB"], site, date(2026, 10, 4), delay=0)
    assert site.pdf_opens == 1 and tally["this_month"] == 1
