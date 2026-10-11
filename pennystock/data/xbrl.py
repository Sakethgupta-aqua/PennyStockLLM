"""Just enough XBRL to read NSE's filings: each fact's name, context (period, and whether the context is a
breakdown along some dimension), value, unit and decimals.

NSE's instances are flat and small (15-650 KB): a regex pass reads one in milliseconds and needs no taxonomy
download. Names are matched WITHOUT their prefix (in-bse-shp:, in-capmkt:), which differs by taxonomy and
year. Segment and line-item breakdowns reuse the same concept names in dimensional contexts, so callers that
want the company's own figure filter on `context.dimensional` (an integrated filing's segment revenue is not
its revenue)."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date

_CONTEXT = re.compile(r"<(?:xbrli:)?context\b[^>]*?\bid=\"([^\"]+)\"[^>]*>(.*?)</(?:xbrli:)?context>", re.S)
_START = re.compile(r"<(?:xbrli:)?startDate>\s*([^<\s]+)")
_END = re.compile(r"<(?:xbrli:)?endDate>\s*([^<\s]+)")
_INSTANT = re.compile(r"<(?:xbrli:)?instant>\s*([^<\s]+)")
_DIMENSION = re.compile(r"explicitMember|typedMember")
_FACT = re.compile(r"<([A-Za-z][\w\-]*):([A-Za-z]\w*)\b([^>]*?\bcontextRef=\"([^\"]+)\"[^>]*?)"
                   r"(?:/>|>(.*?)</\1:\2\s*>)", re.S)
_UNIT = re.compile(r"\bunitRef=\"([^\"]+)\"")
_DECIMALS = re.compile(r"\bdecimals=\"([^\"]+)\"")


@dataclass(frozen=True)
class Context:
    id: str
    start: str | None
    end: str | None
    instant: str | None
    dimensional: bool

    @property
    def days(self) -> int | None:
        if not (self.start and self.end):
            return None
        return (date.fromisoformat(self.end[:10]) - date.fromisoformat(self.start[:10])).days + 1


@dataclass(frozen=True)
class Fact:
    name: str
    value: str
    context: Context
    unit: str | None
    decimals: str | None


def contexts(xml: str) -> dict[str, Context]:
    out = {}
    for cid, body in _CONTEXT.findall(xml):
        s, e, i = _START.search(body), _END.search(body), _INSTANT.search(body)
        out[cid] = Context(cid, s.group(1) if s else None, e.group(1) if e else None,
                           i.group(1) if i else None, bool(_DIMENSION.search(body)))
    return out


def facts(xml: str) -> list[Fact]:
    ctx = contexts(xml)
    out = []
    for _prefix, name, attrs, cref, value in _FACT.findall(xml):
        if cref in ctx:
            u, d = _UNIT.search(attrs), _DECIMALS.search(attrs)
            out.append(Fact(name, html.unescape((value or "").strip()), ctx[cref],
                            u.group(1) if u else None, d.group(1) if d else None))
    return out


def number(fact: Fact | None) -> float | None:
    if fact is None:
        return None
    try:
        return float(fact.value.replace(",", ""))
    except ValueError:
        return None
