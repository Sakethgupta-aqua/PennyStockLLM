"""
Filings search index: every NSE filing's pages extracted into filings.db (keyword search over all of them)
and the passages worth reading embedded into the Chroma collection "filings".

  python scripts/index_filings.py                  # extract new filings, then embed new passages
  python scripts/index_filings.py --status         # what is extracted, embedded, OCR'd
  python scripts/index_filings.py --repair         # re-download filings stored incomplete, then index
  python scripts/index_filings.py --symbol AUBANK  # one company only
  python scripts/index_filings.py --no-embed       # text + keyword index only (minutes, no GPU)
  python scripts/index_filings.py --reextract --reindex   # after an extractor / chunker change

The first full run extracts ~147,000 pages in about 6 minutes, OCRs ~2,700 text-less pages (~1 hour one
at a time) and embeds ~90,000 passages (~3 hours on the Mac's GPU, measured 2026-10-05); later runs touch
only new filings.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                    # noqa: E402
from pennystock.data import documents, filing_text, filings  # noqa: E402


def tesseract_ocr(path: Path, page_index: int) -> str:
    """One page rendered at 300 dpi and read by Tesseract (the optional [ocr] extra)."""
    import pypdfium2 as pdfium
    import pytesseract
    doc = pdfium.PdfDocument(str(path))
    try:
        image = doc[page_index].render(scale=300 / 72).to_pil()
        return pytesseract.image_to_string(image, lang="eng")
    finally:
        doc.close()


def print_status() -> None:
    st = filing_text.status()
    print("\nFILINGS SEARCH INDEX")
    print(f"  filings catalogued  {st['filings']:>7}   ({st['companies']} companies)")
    print(f"  unique files        {st['files']:>7}   ({st['files_embedded']} with passages embedded)")
    print(f"  pages               {st['pages']:>7}   {st['pages_by_kind']}  OCR'd: {st['ocr_pages']}")
    if st["older_extractor"]:
        print(f"  ! {st['older_extractor']} file(s) extracted by an older extractor (run --reextract)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--repair", action="store_true", help="re-download filings stored incomplete first")
    ap.add_argument("--symbol")
    ap.add_argument("--reextract", action="store_true")
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("pdfminer", "httpx", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.status:
        print_status()
        return 0
    if args.repair:
        from pennystock.data.nse_client import nse_session
        get = filings.nse_getter(nse_session("RELIANCE"), timeout=settings.filings_repair_timeout_s)
        print("repair:", filings.repair(get))
    docs = documents.list_documents(args.symbol, "filing")
    ocr = None if args.no_ocr else tesseract_ocr
    print("text:", filing_text.index_texts(docs, reextract=args.reextract, ocr=ocr))
    if not args.no_embed:
        from pennystock.retrieval.filing_search import index_passages
        symbols = {args.symbol.upper()} if args.symbol else None
        print("passages:", index_passages(reindex=args.reindex, symbols=symbols))
    print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
