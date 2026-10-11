"""The documents catalog: one row per kept file, idempotent fetches, never half a PDF."""
import hashlib

import pytest

from pennystock.data import documents as docs

PDF = b"%PDF-1.7\n" + b"x" * 6000 + b"\n%%EOF\n"


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(docs.settings, "market_db", tmp_path / "test.db")
    monkeypatch.setattr(docs.settings, "project_root", tmp_path)
    return tmp_path


def save(root, key="filing:nse:1", data=PDF, name="a.pdf", **kw):
    args = dict(doc_key=key, symbol="aubank", kind="filing", doc_date="2026-09-24", source="nse",
                data=data, dest=root / "Corporate_Filings" / "AUBANK" / name,
                category="investor_presentation", min_bytes=5000)
    return docs.save_file(**(args | kw))


def test_a_saved_file_is_catalogued_with_its_fingerprint(project):
    path = save(project)
    d = docs.get("filing:nse:1")
    assert path.read_bytes() == PDF
    assert (d.symbol, d.local_path, d.bytes) == ("AUBANK", "Corporate_Filings/AUBANK/a.pdf", len(PDF))
    assert d.sha256 == hashlib.sha256(PDF).hexdigest()


def test_saving_the_same_key_again_never_rewrites_the_file(project):
    first = save(project)
    second = save(project, data=PDF + b"changed", name="b.pdf")
    assert second == first and first.read_bytes() == PDF
    assert not (project / "Corporate_Filings" / "AUBANK" / "b.pdf").exists()
    assert len(docs.list_documents()) == 1


def test_html_or_truncated_bytes_are_never_stored(project):
    assert save(project, data=b"<html><title>Trendlyne login</title></html>" * 200) is None
    assert save(project, key="filing:nse:2", data=b"%PDF-1.7 tiny") is None
    assert docs.list_documents() == []
    assert not list(project.rglob("*.pdf"))


def test_a_crash_mid_write_leaves_no_partial_file_and_a_rerun_completes(project, monkeypatch):
    real_replace = docs.os.replace

    def crash(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(docs.os, "replace", crash)
    with pytest.raises(OSError):
        save(project)
    folder = project / "Corporate_Filings" / "AUBANK"
    assert list(folder.iterdir()) == []                 # no a.pdf, no .part- leftovers
    assert docs.get("filing:nse:1") is None
    monkeypatch.setattr(docs.os, "replace", real_replace)
    assert save(project).read_bytes() == PDF


def test_a_file_that_vanished_is_fetched_again(project):
    save(project).unlink()
    assert not docs.has("filing:nse:1")
    assert save(project).read_bytes() == PDF and docs.has("filing:nse:1")


def test_an_unknown_kind_is_refused_before_anything_is_written(project):
    with pytest.raises(ValueError):
        save(project, kind="newsletter")
    assert not list(project.rglob("*.pdf"))


def test_registering_a_file_already_in_place_is_idempotent(project):
    f = project / "Broker_Reports" / "AUBANK" / "x.pdf"
    f.parent.mkdir(parents=True)
    f.write_bytes(PDF)
    kw = dict(doc_key="broker_report:trendlyne:100115", symbol="AUBANK", kind="broker_report",
              doc_date="2026-09-04", source="trendlyne", path=f)
    assert docs.register(**kw) is True
    assert docs.register(**kw) is False
    assert docs.counts() == {"broker_report": {"files": 1, "companies": 1, "bytes": len(PDF)}}


def test_verify_reports_missing_and_changed_files(project):
    a = save(project, key="k1", name="a.pdf")
    b = save(project, key="k2", name="b.pdf")
    a.unlink()
    b.write_bytes(PDF + b"tampered")
    got = docs.verify()
    assert (got["missing"], got["changed"]) == (["k1"], ["k2"])


@pytest.mark.parametrize("symbol, folder", [("M&MFIN", "M_MFIN"), ("GVT&D", "GVT_D"), ("360ONE", "360ONE")])
def test_folder_names_match_the_broker_reports_rule(symbol, folder):
    assert docs.folder_name(symbol) == folder


# --- completeness: a download cut off mid-transfer is not a stored file (19 found on 2026-10-05) ----
def real_pdf() -> bytes:
    """A genuine one-page PDF, so the "does it open" half of the check has something real to open."""
    import io

    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def test_a_pdf_cut_off_mid_download_is_not_complete():
    assert docs.is_complete_pdf(b"%PDF-1.7\n" + b"x" * 6000 + b"\n%%EOF\n")
    assert not docs.is_complete_pdf(b"%PDF-1.7\n" + b"x" * 6000)          # the 4,030,464-byte kind
    assert not docs.is_complete_pdf(b"<html>error</html>")


def test_a_truncated_download_is_never_stored(project):
    assert save(project, data=b"%PDF-1.7\n" + b"x" * 6000) is None


def test_verify_lists_files_that_are_incomplete_on_disk(project):
    save(project, data=real_pdf(), min_bytes=1)
    assert docs.verify()["incomplete"] == []
    path = project / "Corporate_Filings" / "AUBANK" / "a.pdf"
    path.write_bytes(path.read_bytes()[:-40])                                # cut the trailer off
    assert docs.verify()["incomplete"] == ["filing:nse:1"]


def test_replacing_a_broken_file_updates_its_size_and_hash(project):
    save(project)
    good = b"%PDF-1.7\n" + b"y" * 6000 + b"\n%%EOF\n"
    assert docs.replace_file("filing:nse:1", b"%PDF-1.7\n truncated") is None
    path = docs.replace_file("filing:nse:1", good)
    d = docs.get("filing:nse:1")
    assert path.read_bytes() == good and d.bytes == len(good) and d.sha256 == hashlib.sha256(good).hexdigest()
