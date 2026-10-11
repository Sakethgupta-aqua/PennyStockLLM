"""
Read every broker research report into the two places the system uses it.

  python scripts/ingest_broker_reports.py --status       # what exists, what is missing, what is done
  python scripts/ingest_broker_reports.py --dry-run      # parse and report; write nothing
  python scripts/ingest_broker_reports.py --no-embed     # KPI tables only (fast, loads no model)
  python scripts/ingest_broker_reports.py                # KPI tables + vector store, new work only
  python scripts/ingest_broker_reports.py --reparse      # re-read every PDF
  python scripts/ingest_broker_reports.py --reset        # rebuild the vector collection from scratch

Where each part of a report goes:
  the NUMBERS     market.db broker_documents / broker_metrics   (Layer 1: sector KPIs, estimates)
  the ARGUMENT    Chroma collection 'broker_reports'            (Layer 3: one company per search)
  the CALL        the ledger - run scripts/backfill_broker_predictions.py (Layer 2's input)

Reports are found by broker_corpus.discover(): the flat downloads described by metadata.jsonl and
the per-company folders described by manifest.csv, merged so a report present in both is read once.
Only research ON A LISTED COMPANY is ingested; daily market notes and unlisted IPOs resolve to no
symbol and are reported, not stored.

Idempotent: a PDF is identified by a hash of its bytes. One already read by the current parser and
already embedded is skipped; a parser upgrade (PARSER_VERSION) re-reads old documents by itself.
The vector collection stays SEPARATE from knowledge_base: broker notes are opinion with a structural
Buy bias, and must never answer a methodology question about margin of safety.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("ingest_broker")

from pennystock.config import settings                                          # noqa: E402
from pennystock.data.broker_corpus import discover, iso_date, resolve_company   # noqa: E402
from pennystock.data.broker_metrics import (clear_embedded, counts,             # noqa: E402
                                            mark_embedded, parsed_documents, save_document)
from pennystock.data.broker_reports import (PARSER_VERSION, extract_report,     # noqa: E402
                                            merge_listing_metadata, to_chunks)


NO_SYMBOL = "company did not resolve to an NSE symbol (daily notes, unlisted IPOs)"


def _vector_count() -> int | None:
    """Chunks in the broker collection, read without loading the embedding model."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        return client.get_or_create_collection(settings.broker_collection).count()
    except Exception as e:
        log.warning(f"vector store unreadable: {type(e).__name__}: {e}")
        return None


def status(records, folder: str) -> int:
    """Everything needed to analyse the broker reports, and what is still missing."""
    from pennystock.data.ledger import load_predictions
    on_disk = [r for r in records if r.pdf]
    unlisted = [r for r in records if not r.symbol]
    c = counts()
    calls = load_predictions(source="broker")
    print(f"\nBROKER REPORTS ({Path(folder).resolve()})")
    print(f"  reports listed or on disk      {len(records):>5}")
    print(f"  PDFs on disk                   {len(on_disk):>5}")
    print(f"  PDFs still to download         {len(records) - len(on_disk):>5}")
    print(f"  not about a listed company     {len(unlisted):>5}   (daily notes, unlisted IPOs - never ingested)")
    print(f"  parsed into KPI tables         {c['documents']:>5}   ({c['metric_rows']} numbers, "
          f"{c['symbols']} companies; {c['parsed_by_older_parser']} by an older parser)")
    vectors = _vector_count()
    print(f"  embedded for search            {c['embedded']:>5}   ({vectors if vectors is not None else '?'} chunks)")
    print(f"  broker calls in the ledger     {len(calls):>5}")
    missing = len(records) - len(on_disk)
    pending = sum(1 for r in on_disk if r.symbol) - c["embedded"]
    print("\nNEXT")
    if missing:
        print(f"  {missing} PDFs sit behind Trendlyne's login. Export your browser cookie, then:\n"
              f"    python scripts/fetch_midcap_reports.py --cookies cookies.txt\n"
              f"  (it opens at most 70 a day - Trendlyne's limit - so this takes daily rounds)")
    if pending > 0 or c["parsed_by_older_parser"]:
        print("  python scripts/ingest_broker_reports.py")
    print("  python scripts/backfill_broker_predictions.py     # ratings + targets -> ledger (Layer 2)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="Broker_Reports", help="folder of broker PDFs")
    ap.add_argument("--status", action="store_true", help="report what is done and what is missing")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, write nothing")
    ap.add_argument("--no-embed", action="store_true", help="KPI tables only; skip the vector store")
    ap.add_argument("--reparse", action="store_true", help="re-read every PDF")
    ap.add_argument("--reset", action="store_true", help="rebuild the vector collection from scratch")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    records = discover(args.dir)
    if args.status:
        return status(records, args.dir)

    done = parsed_documents()
    store = None
    if not (args.dry_run or args.no_embed):
        from pennystock.retrieval.broker_search import broker_db, drop_legacy_chunks
        store = broker_db()
        if args.reset:
            store.reset_collection()
            clear_embedded()
            done = parsed_documents()
            log.info(f"reset collection '{settings.broker_collection}'")
        if dropped := drop_legacy_chunks(store):
            log.info(f"removed {dropped} chunks written before documents carried an id")

    work = []
    skipped: Counter = Counter()
    unresolved: list[str] = []
    for r in [r for r in records if r.pdf][: args.limit]:
        if not r.symbol and r.company:
            # the listing names a company the NSE universe does not know: a daily market note, an
            # unlisted IPO - or a listed name spelt Trendlyne's way ("AU SF Bank")
            skipped[NO_SYMBOL] += 1
            unresolved.append(r.company)
            continue
        version, embedded = done.get(r.doc_id, (0, None))
        parse = args.reparse or version < PARSER_VERSION
        embed = store is not None and (parse or embedded is None)
        if parse or embed or args.dry_run:
            work.append((r, parse, embed))
    log.info(f"{len(records)} reports known, {sum(1 for r in records if r.pdf)} on disk, "
             f"{len(work)} need work")

    numbers = embedded_chunks = 0
    for i, (r, parse, embed) in enumerate(work, 1):
        try:
            rep = extract_report(r.pdf)
        except Exception as e:
            log.warning(f"[{i}/{len(work)}] {r.pdf.name[:44]} unreadable: {type(e).__name__}: {e}")
            skipped["unreadable PDF"] += 1
            continue
        rep = merge_listing_metadata(rep, r.listing())
        # a PDF no listing describes is identified from its own cover page
        symbol, broker = r.symbol or resolve_company(rep.company), r.broker or rep.broker
        report_date = r.report_date or iso_date(rep.report_date)
        if not symbol:
            skipped[NO_SYMBOL] += 1
            unresolved.append(rep.company or r.pdf.name)
            continue
        if not (broker and report_date):
            skipped["no broker or date"] += 1
            log.warning(f"[{i}/{len(work)}] {r.pdf.name[:44]}: skipped - broker/date unknown")
            continue
        rep.report_date = report_date            # ISO everywhere downstream, newest-first sorts
        line = (f"[{i}/{len(work)}] {symbol:<12}{str(broker)[:22]:<24}{report_date}  "
                f"{len(rep.estimate_rows):>4} table numbers")
        if args.dry_run:
            log.info(line + "  (dry run)")
            continue
        if parse:
            numbers += save_document(doc_id=r.doc_id, symbol=symbol, broker=broker,
                                     report_date=report_date, source_file=str(r.pdf),
                                     rows=rep.estimate_rows, report_id=r.report_id,
                                     company=rep.company, rating=rep.rating, pages=rep.n_pages)
        if embed:
            from pennystock.retrieval.broker_search import replace_document_chunks
            chunks = to_chunks(rep)
            for c in chunks:
                c["metadata"].update(symbol=symbol, file=r.pdf.name)
            embedded_chunks += replace_document_chunks(store, r.doc_id, chunks)
            mark_embedded([r.doc_id])
        log.info(line)

    print(f"\n{len(work)} report(s) processed: {numbers} numbers stored, "
          f"{embedded_chunks} research chunks embedded")
    for why, n in skipped.most_common():
        print(f"  skipped {n}: {why}")
    if unresolved:
        print(f"    {', '.join(sorted(set(unresolved)))}\n"
              "    If one of these IS a listed company, move its PDF into Broker_Reports/<TICKER>/.")
    print("Run with --status to see what is still missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
