"""
Search one company's NSE filings - what management said, why a rating moved, what was disclosed.

  python scripts/search_filings.py "NIM guidance" --symbol AUBANK
  python scripts/search_filings.py "why was the rating downgraded" --symbol GAYAPROJ --type credit_rating
  python scripts/search_filings.py "capex plans" --symbol APARINDS --as-of 2026-06-30
  python scripts/search_filings.py "key audit matters" --symbol AUBANK --type annual_report --top 5

Every result says company, document type, date, fiscal period and page(s), and where the file is.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.data.filing_text import indexed_count      # noqa: E402
from pennystock.retrieval.filing_search import search_filings  # noqa: E402

TYPES = ("earnings_call", "investor_presentation", "credit_rating", "results", "annual_report",
         "material_event")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--type", action="append", choices=TYPES, help="repeat to allow several types")
    ap.add_argument("--as-of", help="YYYY-MM-DD[THH:MM]: only filings the market had seen by then")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--mode", choices=("keyword", "dense", "fused", "fused_rerank"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if not indexed_count(args.symbol):
        print(f"{args.symbol.upper()}: no filings indexed "
              f"(python scripts/index_filings.py --symbol {args.symbol})")
        return 1
    as_of = args.as_of if not args.as_of or "T" in args.as_of else f"{args.as_of}T23:59"
    hits = asyncio.run(search_filings(args.symbol, args.question, doc_types=args.type, as_of=as_of,
                                      top_n=args.top, mode=args.mode))
    if not hits:
        print("not found in the filings on file")
        return 0
    for i, d in enumerate(hits, 1):
        head, _, body = d.page_content.partition("\n")
        print(f"\n{i}. {head}")
        print(textwrap.indent(textwrap.fill(" ".join(body.split())[:700], 110), "   "))
        print(f"   [{d.metadata.get('path')}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
