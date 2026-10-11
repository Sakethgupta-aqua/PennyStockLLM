"""Which broker reports exist, and which company each is about.

Every case is shaped like the real Broker_Reports/ folder on 2026-10-02: twelve flat Trendlyne
downloads described by metadata.jsonl, and 99 per-company folders described by manifest.csv files,
with AU SF Bank's report present in BOTH.
"""
import csv
import hashlib
import json

from pennystock.data.broker_corpus import discover
from pennystock.data.broker_reports import BrokerReport, merge_listing_metadata

TICKERS = {"AUBANK": "AUBANK", "TRENT": "TRENT", "Trent": "TRENT", "360ONE": "360ONE"}
resolve = TICKERS.get          # stands in for the NSE resolver: exact names/tickers only


def _jsonl(folder, rows):
    (folder / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _manifest(folder, ticker, rows):
    d = folder / ticker
    d.mkdir(parents=True, exist_ok=True)
    cols = ["company", "ticker", "report_no", "report_date", "broker", "rating", "target",
            "change_vs_prev", "headline", "filename", "url", "downloaded"]
    with (d / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return d


def _pdf(path, body=b"%PDF-1.4 report"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


AU_FLAT = {"report_id": "100115", "date": "04 Sep 2026", "stock": "AU SF Bank",
           "broker": "ICICI Securities Limited", "ltp": "1065.00", "target": "1250.00",
           "price_at_reco": "1072.00 (-0.65%)", "reco_type": "Buy"}
AU_ROW = {"company": "AU SF Bank", "ticker": "AUBANK", "report_no": "1", "report_date": "2026-09-04",
          "broker": "ICICI Securities Limited", "rating": "Buy", "target": "1250",
          "headline": "Entering the big league", "filename": "AUBANK_1.pdf",
          "url": "https://trendlyne.com/get-document/report/pdf/100115/"}


def test_a_report_listed_twice_becomes_one_record(tmp_path):
    """AU SF Bank's note is a flat download AND report #1 in AUBANK/. Counting it twice would put
    one broker's opinion into the consensus as two brokers agreeing."""
    flat = _pdf(tmp_path / "trendlyne_100115_AU-SF-Bank.pdf")
    _jsonl(tmp_path, [{**AU_FLAT, "local_path": str(flat)}])
    _pdf(_manifest(tmp_path, "AUBANK", [AU_ROW]) / "AUBANK_1.pdf")

    recs = discover(tmp_path, resolve=resolve)
    assert len(recs) == 1
    r = recs[0]
    assert (r.report_id, r.symbol, r.report_date, r.rating, r.target) == \
        ("100115", "AUBANK", "2026-09-04", "Buy", 1250.0)
    assert r.price == 1072.0              # the price AT the recommendation, not today's LTP
    assert r.doc_id == hashlib.sha256(b"%PDF-1.4 report").hexdigest()[:16]


def test_a_name_that_does_not_resolve_leaves_the_symbol_empty(tmp_path):
    """Trendlyne's 'AU SF Bank' is not an NSE name. Without the manifest's ticker the honest answer
    is 'unknown' - a guessed symbol would file the call against the wrong company."""
    flat = _pdf(tmp_path / "trendlyne_100115_AU-SF-Bank.pdf")
    _jsonl(tmp_path, [{**AU_FLAT, "local_path": str(flat)}])
    assert discover(tmp_path, resolve=resolve)[0].symbol is None


def test_a_listed_report_without_its_pdf_is_still_a_record(tmp_path):
    """296 of the 297 midcap PDFs sit behind a login. Their rating, target and date are still
    facts, and Layer 2 needs them."""
    _manifest(tmp_path, "360ONE", [{"company": "360 One Wam", "ticker": "360ONE", "report_no": "1",
                                    "report_date": "2026-09-18", "broker": "Motilal Oswal",
                                    "rating": "Buy", "target": "1300", "filename": "360ONE_1.pdf",
                                    "url": "https://trendlyne.com/get-document/report/pdf/100339/"}])
    r = discover(tmp_path, resolve=resolve)[0]
    assert (r.symbol, r.broker, r.pdf, r.doc_id, r.report_id) == \
        ("360ONE", "Motilal Oswal", None, None, "100339")


def test_a_manifest_ticker_the_universe_does_not_know_is_not_trusted(tmp_path):
    _manifest(tmp_path, "TYPO", [{**AU_ROW, "ticker": "AUBNK", "company": "Unknown Co",
                                  "url": "https://trendlyne.com/get-document/report/pdf/1/"}])
    assert discover(tmp_path, resolve=resolve)[0].symbol is None


def test_a_flat_listing_resolves_by_company_name(tmp_path):
    flat = _pdf(tmp_path / "trendlyne_100132_Trent.pdf")
    _jsonl(tmp_path, [{"report_id": "100132", "date": "04 Sep 2026", "stock": "Trent",
                       "broker": "Joindre Capital Services", "ltp": "2853.00", "target": "3357.00",
                       "price_at_reco": "2853.00", "reco_type": "Buy", "local_path": str(flat)}])
    r = discover(tmp_path, resolve=resolve)[0]
    assert (r.symbol, r.report_date, r.price, r.target) == ("TRENT", "2026-09-04", 2853.0, 3357.0)


def test_a_pdf_dropped_into_a_company_folder_is_found(tmp_path):
    """A report saved by hand into Broker_Reports/AUBANK/ belongs to AUBANK even though no manifest
    lists it; its broker and date are read from the PDF at ingest."""
    _manifest(tmp_path, "AUBANK", [AU_ROW])
    extra = _pdf(tmp_path / "AUBANK" / "downloaded_by_hand.pdf", b"%PDF-1.4 another")
    recs = {r.pdf.name if r.pdf else None: r for r in discover(tmp_path, resolve=resolve)}
    assert recs["downloaded_by_hand.pdf"].symbol == "AUBANK"
    assert recs["downloaded_by_hand.pdf"].pdf == extra


def test_a_moved_project_still_finds_flat_pdfs_by_file_name(tmp_path):
    """metadata.jsonl stores ABSOLUTE paths; moving the project must not orphan every report."""
    _pdf(tmp_path / "trendlyne_100132_Trent.pdf")
    _jsonl(tmp_path, [{"report_id": "100132", "date": "04 Sep 2026", "stock": "Trent",
                       "broker": "Joindre", "reco_type": "Buy",
                       "local_path": "/old/home/Broker_Reports/trendlyne_100132_Trent.pdf"}])
    assert discover(tmp_path, resolve=resolve)[0].pdf == tmp_path / "trendlyne_100132_Trent.pdf"


def test_the_listing_feeds_the_existing_metadata_merge(tmp_path):
    flat = _pdf(tmp_path / "trendlyne_100115_AU-SF-Bank.pdf")
    _jsonl(tmp_path, [{**AU_FLAT, "local_path": str(flat)}])
    _pdf(_manifest(tmp_path, "AUBANK", [AU_ROW]) / "AUBANK_1.pdf")
    rep = merge_listing_metadata(BrokerReport(path="x.pdf"), discover(tmp_path, resolve=resolve)[0].listing())
    assert (rep.company, rep.broker, rep.rating, rep.cmp, rep.target_price, rep.report_date) == \
        ("AU SF Bank", "ICICI Securities Limited", "BUY", 1072.0, 1250.0, "2026-09-04")
