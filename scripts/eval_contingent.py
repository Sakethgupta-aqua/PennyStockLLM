"""Score the stored contingent-liability extractions against the hand-checked gold set.

A VERIFIED extraction passes when its disputed claims and its guarantees given for others both match the
gold to within Rs 0.5 Cr for both years. The number that must stay at zero is VERIFIED-BUT-WRONG: a total the
analyst would quote as exact that is not. A report that is not verified falls back to the indicative excerpt,
so it costs coverage, never correctness.

  python scripts/eval_contingent.py            # exits 1 if any verified extraction is wrong
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pennystock.data import contingent_liabilities as cl   # noqa: E402

GOLD = ROOT / "eval" / "contingent_gold.jsonl"
TOLERANCE_CR = 0.5


def _matches(got: dict, want: list[float]) -> bool:
    return all(abs(got[k] - w) <= TOLERANCE_CR for k, w in zip(("current", "previous"), want))


def main() -> int:
    rows = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    verified = right = wrong = 0
    for g in rows:
        r = cl.stored(g["symbol"])
        if r is None:
            print(f"  NOT READ      {g['symbol']}")
            continue
        if r.get("content_sha") != g["content_sha"]:
            print(f"  STALE GOLD    {g['symbol']} (a newer annual report was read)")
            continue
        if r["status"] != "verified":
            print(f"  NOT VERIFIED  {g['symbol']:<12} status {r['status']}; warnings {r.get('warnings')}")
            continue
        verified += 1
        t = r["totals"]
        ok = _matches(t["disputed_claims"], g["disputed_claims_cr"]) and \
            _matches(t["guarantees_for_others"], g["guarantees_for_others_cr"])
        right += ok
        wrong += not ok
        print(f"  {'PASS' if ok else 'WRONG':<13} {g['symbol']:<12} claims {t['disputed_claims']} vs "
              f"{g['disputed_claims_cr']}; for others {t['guarantees_for_others']} vs "
              f"{g['guarantees_for_others_cr']}")
    print(f"\n{len(rows)} gold reports: verified {verified} ({verified / max(len(rows), 1):.0%}), "
          f"right {right}, VERIFIED-BUT-WRONG {wrong}; accuracy of verified {right / max(verified, 1):.0%}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
