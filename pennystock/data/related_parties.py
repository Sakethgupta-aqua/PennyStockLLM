"""Related-party transactions from the company's own half-yearly disclosure to NSE (SEBI Reg 23(9), XBRL).

The classic Indian governance check is value leaving a listed company for entities its promoters own. The
disclosure itemises every transaction - entity, counterparty, relationship (free text), type, amount - so it
can be totalled by kind of relationship instead of scraped from a PDF table. Subsidiaries are eliminated on
consolidation, so what matters is what flows OUTSIDE them, against the same half-year's revenue (~10% is the
long-standing threshold). The relationship text cannot always say whose group a counterparty belongs to:
"common control" is a promoter's group in a family conglomerate and a joint-venture partner's group at DIXON
(Bharti Airtel holds 47.58% of a Dixon subsidiary), so that class is named for what it is and the
counterparties are always shown.
"""
from __future__ import annotations

import re
from collections import Counter

from pennystock.data.integrated_filings import RptDisclosure, RptRow

CLASSES = ("subsidiary", "jv_associate", "promoter_kmp_entity", "group_or_partner", "people", "other")
_PEOPLE_WORDS = r"(kmp|key\s+manageri\w*|directors?|relatives?|promoters?|individuals?|whole[\s-]?time)"
_OWNED_BY_PEOPLE = re.compile(
    r"(enterprise|entit|compan|firm|trust|concern|llp|huf)\w*\b.{0,60}?\b(owned|controlled|influenced|"
    rf"over\s+which|in\s+which|where)\b.{{0,100}}?\b{_PEOPLE_WORDS}", re.I)
_PROMOTER_ENTITY = re.compile(r"promoter['’]?s?\s+(group|compan|entit|firm|enterprise)", re.I)
_OWN_CONTROL = re.compile(r"over\s+which\s+(the\s+)?(holding\s+|listed\s+)?(company|entity)\b.{0,40}"
                          r"\b(control|influence)", re.I)
_FELLOW = re.compile(r"fellow\s+subsidiar", re.I)
# a subsidiary OF another company (DIXON: "Wholly Owned Subsidiary of Lightanium", its JV's) is outside
# the group
_SUB_OF_OTHER = re.compile(r"subsidiar\w*\s+of\s+(?!(the\s+)?(company|listed|holding|parent)\b)", re.I)
_SUBSIDIARY = re.compile(r"subsidiar", re.I)
_JV = re.compile(r"joint\s+venture|\bjv\b|associate", re.I)
# a controlling or large shareholder: PSUs (NTPC, IRCTC) name the government 'Controlling Authority'
_GROUP = re.compile(r"holding\s+compan|common\s+control|significant\s+influence|investing\s+part|venturer|"
                    r"group\s+compan|controlling\s+authority|government|ministry|president\s+of\s+india|"
                    r"holding\s+more\s+than|substantial\s+shareholder", re.I)
_PEOPLE = re.compile(rf"\b{_PEOPLE_WORDS}|managing|chief\s+(executive|financial)|company\s+secretary|"
                     r"\bceo\b|\bcfo\b", re.I)
_LOANLIKE = re.compile(r"\bloan|guarantee|advance|deposit|investment|security\s+given|inter[\s-]?corporate",
                       re.I)
HOW_TO_JUDGE = (
    "Subsidiaries are inside the group (eliminated on consolidation). Judge what flows OUTSIDE them against "
    "the half-year's revenue (~10% is the long-standing threshold) and say who the counterparties are: "
    "promoter_kmp_entity is the classic channel for value to leave; group_or_partner can be a promoter's "
    "group OR a joint-venture partner's group - the names tell which. Amounts are as disclosed and can "
    "include taxes.")


def classify(relationship: str, listed_word: str | None = None) -> str:
    """The kind of related party. `listed_word` (the listed company's first name-word) tells 'Subsidiary of
    Swiggy Limited' - its own subsidiary - from 'Subsidiary of Lightanium' - a joint venture's."""
    r = relationship or ""
    if _OWNED_BY_PEOPLE.search(r) or _PROMOTER_ENTITY.search(r):
        return "promoter_kmp_entity"
    if _OWN_CONTROL.search(r):                     # entities the listed company itself controls or influences
        return "jv_associate"
    if _FELLOW.search(r):
        return "group_or_partner"
    m = _SUB_OF_OTHER.search(r)
    if m:
        own = listed_word and listed_word.lower() in r[m.end():].lower()
        return "subsidiary" if own else "jv_associate"
    if _SUBSIDIARY.search(r):
        return "subsidiary"
    if _JV.search(r):
        return "jv_associate"
    if _GROUP.search(r):
        return "group_or_partner"
    if _PEOPLE.search(r):
        return "people"
    return "other"


def _describe(cls: str, r: RptRow) -> dict:
    kind = r.details if r.type.lower().startswith("any other") and r.details else r.type
    return {"entity": r.entity, "counterparty": r.counterparty, "class": cls, "relationship": r.relationship,
            "type": kind, "amount_cr": r.amount_cr}


def _listed_word(d: RptDisclosure) -> str | None:
    """The listed company's first name-word, from the entity that enters most of the transactions."""
    names = Counter(r.entity for r in d.rows if r.entity)
    if not names:
        return None
    word = re.match(r"[A-Za-z]{3,}", names.most_common(1)[0][0].strip())
    return word.group(0) if word else None


def summarize_rpt(d: RptDisclosure) -> dict:
    # amounts count by size: a sign encodes direction (KALYANKJIL prints a released guarantee as -158.14)
    listed = _listed_word(d)
    rows = [(classify(r.relationship, listed), r) for r in d.rows if r.amount_cr]
    by_class: dict[str, float] = {}
    for cls, r in rows:
        by_class[cls] = by_class.get(cls, 0.0) + abs(r.amount_cr)
    outside = [(c, r) for c, r in rows if c != "subsidiary"]
    outside_total = round(sum(abs(r.amount_cr) for _, r in outside), 2)
    pk_total = round(by_class.get("promoter_kmp_entity", 0.0), 2)

    def pct(x: float) -> float | None:
        return round(x / d.revenue_cr * 100, 1) if d.revenue_cr else None

    loans = [(c, r) for c, r in outside if c != "people" and _LOANLIKE.search(f"{r.type} {r.details}")]
    return {
        "source": "NSE Reg 23(9) XBRL",
        "period": f"{d.half} ({d.period_start} to {d.period_end})",
        "filed": (d.broadcast_at or "")[:11],
        "transactions": len(rows),
        "half_year_revenue_cr": d.revenue_cr,
        "revenue_note": d.revenue_note,
        "by_class_cr": {c: round(v, 2) for c, v in by_class.items()},
        "outside_subsidiaries_cr": outside_total,
        "outside_subsidiaries_pct_of_revenue": pct(outside_total),
        "promoter_kmp_entities_cr": pk_total,
        "promoter_kmp_entities_pct_of_revenue": pct(pk_total),
        "largest_outside_subsidiaries": [_describe(c, r) for c, r in
                                         sorted(outside, key=lambda cr: -abs(cr[1].amount_cr))[:6]],
        "loans_guarantees_outside_subsidiaries": [_describe(c, r) for c, r in
                                                  sorted(loans, key=lambda cr: -abs(cr[1].amount_cr))[:5]],
        "how_to_judge": HOW_TO_JUDGE,
    }
