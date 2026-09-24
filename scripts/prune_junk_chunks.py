"""
Remove index / TOC / bibliography chunks that are ALREADY stored in the vector DB.

Re-ingesting cannot fix this on its own: ingestion upserts, so junk written before the filter
existed stays put. This scans the live collection and deletes the junk by id - no re-embedding,
so it takes seconds rather than the ~10 minutes a full rebuild would.

  python scripts/prune_junk_chunks.py                 # DRY RUN - shows what would go
  python scripts/prune_junk_chunks.py --apply         # actually delete
  python scripts/prune_junk_chunks.py --apply --collection knowledge_base
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.config import settings                     # noqa: E402
from pennystock.data.text_quality import junk_reason       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete junk chunks from an existing Chroma collection.")
    ap.add_argument("--collection", default=settings.collection)
    ap.add_argument("--db-dir", default=str(settings.vector_db_dir))
    ap.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    ap.add_argument("--show", type=int, default=3, help="how many examples to print")
    args = ap.parse_args()

    import chromadb
    col = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    got = col.get(include=["documents", "metadatas"])
    ids, docs, metas = got["ids"], got["documents"], got["metadatas"]
    print(f"Scanning '{args.collection}': {len(ids)} chunks ...\n")

    junk_ids, by_book, by_reason, samples = [], Counter(), Counter(), []
    for cid, text, meta in zip(ids, docs, metas):
        reason = junk_reason(text)
        if reason:
            junk_ids.append(cid)
            by_book[str((meta or {}).get("source", "?"))[:44]] += 1
            by_reason[reason.split(" (")[0]] += 1
            if len(samples) < args.show:
                samples.append((reason, " ".join(text.split())[:110]))

    if not junk_ids:
        print("No junk chunks found - nothing to do."); return 0

    print(f"JUNK: {len(junk_ids)} of {len(ids)} chunks ({len(junk_ids)/len(ids):.1%})\n")
    print("  by reason:")
    for r, n in by_reason.most_common():
        print(f"     {n:5}  {r}")
    print("\n  by book:")
    for b, n in by_book.most_common():
        print(f"     {n:5}  {b}")
    print("\n  examples:")
    for reason, text in samples:
        print(f"     [{reason}] {text}...")

    if not args.apply:
        print(f"\nDRY RUN - nothing deleted. Re-run with --apply to remove these {len(junk_ids)} chunks.")
        return 0

    for i in range(0, len(junk_ids), 500):                 # delete in batches
        col.delete(ids=junk_ids[i:i + 500])
    print(f"\nDeleted {len(junk_ids)} chunks. Collection now holds {col.count()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
