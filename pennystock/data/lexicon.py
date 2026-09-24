"""Loughran-McDonald financial sentiment lexicon.

Generic sentiment tools are actively WRONG on financial text: "liability", "tax", "cost" and
"depreciation" all read as negative to a general-purpose lexicon and are neutral accounting terms
to an analyst. Loughran & McDonald built their word lists from ~4,000 SEC filings precisely to fix
that, and it is the accepted standard in the finance literature.

Seven categories matter here; two of them are the real prize:
    Positive / Negative   ordinary tone
    Uncertainty           "approximately", "possibly", "risk"
    Litigious             legal exposure creeping into the language
    StrongModal           "will", "must", "definitely"   <- confident commitment
    WeakModal             "may", "might", "could"        <- hedging
    Constraining          "restricted", "obligation"

The strong/weak modal SPLIT is the single most useful signal in an earnings call: management that
said "we WILL commission the plant in Q3" last quarter and says "we MAY commission it" this quarter
has changed its mind without ever admitting it.

The full dictionary (~86k words, 8.9MB CSV) is downloaded once and cached as a compact JSON. If the
download is unavailable the module falls back to a small built-in core list, so analysis degrades
in precision but never breaks.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from pennystock.config import settings

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parent / "resources" / "lm_lexicon.json"
# canonical name -> column header in the LM master CSV (the modal columns are underscored)
CATEGORIES = {"Negative": "Negative", "Positive": "Positive", "Uncertainty": "Uncertainty",
              "Litigious": "Litigious", "StrongModal": "Strong_Modal",
              "WeakModal": "Weak_Modal", "Constraining": "Constraining"}

# Minimal fallback so the module always works offline. Deliberately small and obvious.
FALLBACK = {
    "Positive": {"strong", "growth", "improve", "improved", "improvement", "gain", "gains", "profit",
                 "profitable", "success", "successful", "opportunity", "opportunities", "efficient",
                 "robust", "record", "outperform", "favorable", "favourable", "benefit", "achieve"},
    "Negative": {"loss", "losses", "decline", "declined", "weak", "weakness", "adverse", "against",
                 "difficult", "challenging", "challenge", "delay", "delayed", "deferred", "shortfall",
                 "impairment", "litigation", "default", "downturn", "slowdown", "pressure", "risk"},
    "Uncertainty": {"may", "might", "could", "approximately", "uncertain", "uncertainty", "possible",
                    "possibly", "assume", "assumption", "believe", "depends", "unclear", "variable"},
    "Litigious": {"litigation", "court", "plaintiff", "arbitration", "tribunal", "statutory",
                  "regulatory", "compliance", "penalty", "lawsuit", "appeal"},
    "StrongModal": {"will", "must", "shall", "always", "definitely", "certainly", "clearly",
                    "undoubtedly", "never", "highest", "strongly"},
    "WeakModal": {"may", "might", "could", "possibly", "perhaps", "appears", "seems", "sometimes",
                  "suggests", "conceivably", "depending"},
    "Constraining": {"restricted", "obligation", "commitment", "covenant", "constraint", "mandatory",
                     "required", "limitation"},
}

LM_URL = ("https://drive.google.com/uc?export=download&"
          "id=17CmUZM9hGUdGYjCXcjQLyybjTrcjrhik")


def _build_cache() -> dict[str, list[str]]:
    """Download the LM master dictionary once and reduce it to per-category word lists."""
    import csv
    import io

    import requests

    r = requests.get(LM_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    r.raise_for_status()
    rows = csv.DictReader(io.StringIO(r.text))
    out: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for row in rows:
        word = (row.get("Word") or "").strip().lower()
        if not word:
            continue
        for canonical, column in CATEGORIES.items():
            val = (row.get(column) or "0").strip()
            if val not in ("", "0"):          # the cell holds the year the word entered the list
                out[canonical].append(word)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({k: sorted(v) for k, v in out.items()}))
    logger.info(f"LM lexicon cached: { {k: len(v) for k, v in out.items()} }")
    return out


@lru_cache(maxsize=1)
def lexicon() -> dict[str, set[str]]:
    """Category -> set of words. Cached on disk; falls back to the built-in core list."""
    if CACHE_PATH.exists():
        try:
            return {k: set(v) for k, v in json.loads(CACHE_PATH.read_text()).items()}
        except Exception as e:
            logger.warning(f"LM cache unreadable ({e}); rebuilding")
    if settings.lexicon_auto_download:
        try:
            return {k: set(v) for k, v in _build_cache().items()}
        except Exception as e:
            logger.warning(f"LM download failed ({e}); using the built-in fallback list")
    return {k: set(v) for k, v in FALLBACK.items()}


def is_fallback() -> bool:
    """True when running on the small built-in list rather than the full dictionary."""
    return not CACHE_PATH.exists()
