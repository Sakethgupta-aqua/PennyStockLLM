"""Broker reports join the documents catalog where they lie - downloaded ones only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from catalog_documents import broker_document  # noqa: E402

from pennystock.data.broker_corpus import ReportRecord  # noqa: E402


def test_a_downloaded_broker_report_becomes_a_catalog_entry(tmp_path):
    pdf = tmp_path / "AUBANK_1.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    got = broker_document(ReportRecord(report_id="100115", symbol="AUBANK", broker="ICICI Securities",
                                       report_date="2026-09-04", headline="Entering the big league",
                                       pdf=pdf, doc_id="abc"))
    assert got == {"doc_key": "broker_report:trendlyne:100115", "symbol": "AUBANK", "kind": "broker_report",
                   "doc_date": "2026-09-04", "source": "trendlyne", "path": pdf,
                   "title": "ICICI Securities: Entering the big league",
                   "source_url": "https://trendlyne.com/get-document/report/pdf/100115/"}


def test_a_pdf_without_a_trendlyne_id_is_keyed_by_its_content_hash(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    got = broker_document(ReportRecord(symbol="TRENT", broker="Joindre", report_date="2026-09-04", pdf=pdf,
                                       doc_id="9f2c"))
    assert got["doc_key"] == "broker_report:sha:9f2c" and got["source_url"] is None


def test_listed_but_not_downloaded_reports_are_not_catalogued():
    assert broker_document(ReportRecord(report_id="1", symbol="X", report_date="2026-01-01")) is None
