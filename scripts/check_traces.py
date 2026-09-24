"""
Run the Layer-0 assertions over saved traces and print an error-analysis table.

  python scripts/check_traces.py                 # all traces
  python scripts/check_traces.py --last 5        # the 5 most recent
  python scripts/check_traces.py --file traces/20260903T142201-a1b2c3.json
  python scripts/check_traces.py --json          # machine-readable (CI)

Exit code is 1 if any applicable check failed, so this doubles as a CI gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pennystock.evals.assertions import run_all       # noqa: E402
from pennystock.tracing import load_traces            # noqa: E402

MARK = {True: "PASS", False: "FAIL", None: " -  "}


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer-0 assertion runner over saved traces.")
    ap.add_argument("--last", type=int, help="only the N most recent traces")
    ap.add_argument("--file", help="check a single trace file")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = ap.parse_args()

    traces = ([json.loads(Path(args.file).read_text())] if args.file
              else load_traces(limit=args.last))
    if not traces:
        print("No traces found. Run `python demo.py` first - every run writes one.")
        return 0

    rows, stats = [], defaultdict(lambda: {"applicable": 0, "failed": 0})
    for t in traces:
        checks = run_all(t)
        rows.append((t, checks))
        for c in checks:
            if c["applicable"]:
                stats[c["name"]]["applicable"] += 1
                stats[c["name"]]["failed"] += 0 if c["passed"] else 1

    if args.json:
        print(json.dumps({"traces": [{"trace_id": t.get("trace_id"), "checks": c} for t, c in rows],
                          "summary": {k: dict(v) for k, v in stats.items()}}, indent=2))
    else:
        names = [c["name"] for c in rows[0][1]]
        print("\n" + "=" * 78)
        print(f"  LAYER-0 ASSERTIONS  ({len(traces)} trace(s))")
        print("=" * 78)
        for t, checks in rows:
            fails = [c for c in checks if c["applicable"] and not c["passed"]]
            print(f"\n{t.get('trace_id','?')}  {t.get('resolved_symbol') or '-'}"
                  f"   {'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
            print(f"   q: {(t.get('query') or '')[:70]}...")
            for c in checks:
                state = MARK[None] if not c["applicable"] else MARK[c["passed"]]
                print(f"   [{state}] {c['name']:<24} {c['detail'][:60]}")
        print("\n" + "-" * 78)
        print("  FAILURE RATE BY CHECK  (where to focus next)")
        print("-" * 78)
        for name in names:
            s = stats[name]
            if not s["applicable"]:
                print(f"   {name:<26}  n/a")
            else:
                print(f"   {name:<26}  {s['failed']}/{s['applicable']}  ({s['failed']/s['applicable']:6.1%})")

    total_fail = sum(v["failed"] for v in stats.values())
    print(f"\n  total failures: {total_fail}")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
