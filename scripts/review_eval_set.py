"""
Read and curate the golden eval set without hand-editing JSON.

The set is JSONL (one object per line) because eval_rag.py streams it and build_eval_set.py
appends to it - so it cannot be pretty-printed. This renders it instead: dense, aligned, and
wrapped to your terminal, with the long source_text hidden until you ask for it.

  python scripts/review_eval_set.py                 # compact listing of everything
  python scripts/review_eval_set.py --answers       # include each reference answer
  python scripts/review_eval_set.py --show 3        # one row in full, with the source passage
  python scripts/review_eval_set.py --verify 1,4,7  # mark rows verified (1-based)
  python scripts/review_eval_set.py --verify-all
  python scripts/review_eval_set.py --unverify 4
  python scripts/review_eval_set.py --drop 2,9      # delete bad rows for good
  python scripts/review_eval_set.py --pending       # only the rows still needing a decision
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
KEY_ORDER = ["verified", "source_book", "question", "reference_answer", "source_id", "source_text"]


def load() -> list[dict]:
    if not GOLDEN.exists():
        print(f"No golden set at {GOLDEN}. Run scripts/build_eval_set.py first.")
        sys.exit(0)
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def save(rows: list[dict]) -> None:
    GOLDEN.write_text("".join(
        json.dumps({k: r[k] for k in KEY_ORDER if k in r}, ensure_ascii=False) + "\n" for r in rows))


def short_books(rows: list[dict]) -> dict[str, str]:
    """Stable short codes so the book column stays narrow and content gets the space."""
    codes, used = {}, set()
    for r in rows:
        b = r.get("source_book", "?")
        if b in codes:
            continue
        stem = Path(b).stem.replace("dokumen.pub_", "").replace("_", "-")
        parts = [p for p in stem.replace("-", " ").split() if p]
        code = ("".join(p[0] for p in parts[:3]) or stem[:3]).upper()[:4]
        while code in used:
            code += "*"
        used.add(code)
        codes[b] = code
    return codes


def parse_idx(spec: str, n: int) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit() or not (1 <= int(part) <= n):
            print(f"ignoring '{part}' - expected 1..{n}")
            continue
        out.append(int(part) - 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Review and curate the golden eval set.")
    ap.add_argument("--answers", action="store_true", help="show each reference answer too")
    ap.add_argument("--show", type=int, help="print one row in full, including the source passage")
    ap.add_argument("--verify", help="comma-separated row numbers to mark verified")
    ap.add_argument("--unverify", help="comma-separated row numbers to unmark")
    ap.add_argument("--verify-all", action="store_true")
    ap.add_argument("--drop", help="comma-separated row numbers to delete permanently")
    ap.add_argument("--pending", action="store_true", help="list only rows not yet verified")
    args = ap.parse_args()

    rows = load()
    n = len(rows)

    if args.show:
        if not 1 <= args.show <= n:
            print(f"row {args.show} does not exist (1..{n})"); return 1
        r = rows[args.show - 1]
        width = min(shutil.get_terminal_size((110, 24)).columns, 118)
        print("=" * width)
        print(f"ROW {args.show}/{n}   verified={r.get('verified')}   {r.get('source_book')}")
        print("=" * width)
        for label, key in (("QUESTION", "question"), ("ANSWER", "reference_answer"),
                           ("SOURCE PASSAGE", "source_text")):
            print(f"\n{label}:")
            print(textwrap.fill(" ".join(str(r.get(key, "")).split()), width=width,
                                initial_indent="  ", subsequent_indent="  "))
        return 0

    if args.verify_all or args.verify or args.unverify or args.drop:
        if args.verify_all:
            for r in rows:
                r["verified"] = True
            print(f"marked all {n} rows verified")
        for i in parse_idx(args.verify or "", n):
            rows[i]["verified"] = True
        for i in parse_idx(args.unverify or "", n):
            rows[i]["verified"] = False
        if args.drop:
            keep = set(range(n)) - set(parse_idx(args.drop, n))
            dropped = n - len(keep)
            rows = [r for i, r in enumerate(rows) if i in keep]
            print(f"deleted {dropped} row(s)")
        save(rows)
        rows = load()
        n = len(rows)

    width = min(shutil.get_terminal_size((110, 24)).columns, 118)
    codes = short_books(rows)
    listing = [(i, r) for i, r in enumerate(rows, 1)
               if not args.pending or not r.get("verified")]
    qwidth = width - 14

    print(f"\n{'#':>3} {'V':^3} {'BOOK':<5} QUESTION")
    print("-" * width)
    for i, r in listing:
        mark = "OK" if r.get("verified") else "."
        q = textwrap.wrap(" ".join(str(r.get("question", "")).split()), qwidth) or [""]
        print(f"{i:>3} {mark:^3} {codes.get(r.get('source_book','?'), '?'):<5} {q[0]}")
        for cont in q[1:]:
            print(f"{'':>3} {'':^3} {'':<5} {cont}")
        if args.answers:
            a = textwrap.wrap(" ".join(str(r.get("reference_answer", "")).split()), qwidth - 2)
            for j, line in enumerate(a[:3]):
                print(f"{'':>3} {'':^3} {'':<5} {'-> ' if j == 0 else '   '}{line}")
    print("-" * width)
    done = sum(1 for r in rows if r.get("verified"))
    print(f"{n} rows | {done} verified | {n - done} pending")
    print("  legend: " + "  ".join(f"{c}={Path(b).stem[:26]}" for b, c in codes.items()))
    if n - done:
        print("\n  verify with:  python scripts/review_eval_set.py --verify 1,2,5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
