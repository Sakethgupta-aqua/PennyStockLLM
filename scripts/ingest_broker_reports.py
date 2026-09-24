"""
Ingest broker research PDFs into their OWN Chroma collection.

  python scripts/ingest_broker_reports.py --dry-run          # parse only, embed nothing
  python scripts/ingest_broker_reports.py                    # ingest Broker_Reports/
  python scripts/ingest_broker_reports.py --dir ./more_pdfs --reset

Deliberately SEPARATE from `knowledge_base`. That collection holds investing books - Graham,
Piotroski-style forensics, behavioural finance - which the analyst is told to treat as
methodology. Broker notes are somebody's OPINION, and sell-side ratings skew heavily to Buy. Mixed
into one collection, broker optimism would start answering "what does margin of safety mean?".
Kept apart, the analyst can quote "ICICI Securities rated UltraTech HOLD on 04 Sep 2026" as an
attributed view while the frameworks stay clean.

Idempotent: a report already ingested (same file, same size) is skipped unless --reset is passed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ingest_broker")

from pennystock.config import settings                                          # noqa: E402
from pennystock.data.broker_reports import (extract_report, load_listing_metadata,  # noqa: E402
                                            merge_listing_metadata, to_chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Broker_Reports", help="folder of broker PDFs")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, embed nothing")
    ap.add_argument("--reset", action="store_true", help="drop the collection and rebuild it")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    folder = Path(args.dir)
    pdfs = sorted(folder.glob("*.pdf"))[: args.limit]
    if not pdfs:
        log.error(f"no PDFs in {folder.resolve()}")
        return 1
    meta = load_listing_metadata(folder)
    log.info(f"{len(pdfs)} PDFs, {len(meta)} listing rows available")

    all_chunks, summary = [], []
    for i, pdf in enumerate(pdfs, 1):
        try:
            rep = extract_report(pdf)
        except Exception as e:
            log.warning(f"  [{i}/{len(pdfs)}] {pdf.name[:44]} FAILED {type(e).__name__}: {e}")
            continue
        if row := meta.get(str(pdf.resolve())):
            rep = merge_listing_metadata(rep, row)
        chunks = to_chunks(rep)
        for c in chunks:                       # provenance for dedupe and for citation
            c["metadata"]["file"] = pdf.name
        all_chunks.extend(chunks)
        fwd = sum(1 for k in rep.estimates if k.endswith(("E", "P")))
        total = rep.chars_kept + rep.chars_dropped
        summary.append({"file": pdf.name, "company": rep.company, "broker": rep.broker,
                        "rating": rep.rating, "target": rep.target_price,
                        "forward_years": fwd, "chunks": len(chunks),
                        "boilerplate_pct": round(rep.chars_dropped / total * 100) if total else 0})
        log.info(f"  [{i}/{len(pdfs)}] {str(rep.company)[:22]:<24}{str(rep.broker)[:18]:<20}"
                 f"{str(rep.rating):<8} {len(chunks):>4} chunks  {fwd} fwd-yrs  "
                 f"{summary[-1]['boilerplate_pct']:>2}% legalese dropped")

    print(f"\n{len(all_chunks)} chunks from {len(summary)} reports")
    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["metadata"]["chunk_type"]] = by_type.get(c["metadata"]["chunk_type"], 0) + 1
    print(f"  by type: {by_type}")
    dropped = [s for s in summary if s["boilerplate_pct"]]
    if dropped:
        print(f"  boilerplate dropped: median {sorted(s['boilerplate_pct'] for s in dropped)[len(dropped)//2]}% "
              f"of characters across {len(dropped)} reports")

    if args.dry_run:
        (folder / "extraction_report.json").write_text(json.dumps(summary, indent=2))
        print(f"\ndry run - nothing embedded. Wrote {folder / 'extraction_report.json'}")
        return 0

    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from pennystock.retrieval.store import _embeddings

    db = Chroma(persist_directory=str(settings.vector_db_dir), embedding_function=_embeddings(),
                collection_name=settings.broker_collection)
    if args.reset:
        db.reset_collection()
        log.info(f"reset collection '{settings.broker_collection}'")
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in all_chunks]
    for i in range(0, len(docs), 200):                        # batch so a big run streams
        db.add_documents(docs[i:i + 200])
        log.info(f"  embedded {min(i + 200, len(docs))}/{len(docs)}")
    print(f"\n{len(docs)} chunks in collection '{settings.broker_collection}' "
          f"(knowledge_base untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
