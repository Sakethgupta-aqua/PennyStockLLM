"""Passages: each filing type cut along its own structure, every passage opening with where it came from.

Research behind the cuts (docs/superpowers/specs/2026-10-05-filings-search-design.md, section 2):
structure-aware chunking beats fixed windows on financial reports and halves indexing cost; simple
splits with ~25% overlap lead and tiny "semantic" chunks fall 8-12 points behind; earnings calls read
best as a question kept with its answer; and a short context written into each chunk cuts failed
retrievals by about a third - here written from the catalog (company, type, date, period, page,
speaker), so it costs no model call.

  earnings_call          prepared remarks split by size; Q&A as analyst question + management answers
  investor_presentation  one passage per slide (image-only slides skipped)
  results                prose pages only - statement tables stay keyword-only
  annual_report          key sections only (MD&A, Board's report, risk, auditor, contingent, related party)
  credit_rating, material_event   the filing's pages, merged up to the size cap
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pennystock.config import settings

CHUNKER_VERSION = 1
_MIN_SLIDE_CHARS = 40


@dataclass
class Passage:
    text: str                   # the context line, then the body
    page_from: int
    page_to: int
    section: str | None = None
    speaker: str | None = None


def context_line(company: str, doc_type: str, doc_date: str, fiscal_period: str | None, page_from: int,
                 page_to: int, section: str | None = None, speaker: str | None = None) -> str:
    pages = f"p.{page_from}" if page_from == page_to else f"pp.{page_from}-{page_to}"
    parts = [company, doc_type.replace("_", " "), doc_date, fiscal_period, pages,
             section.replace("_", " ") if section else None, speaker]
    return " · ".join(p for p in parts if p)


def _units(text: str, max_chars: int) -> list[str]:
    """Paragraphs; a paragraph over the cap becomes sentences; a sentence over the cap is cut."""
    out: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                out.append(sent[:cut])
                sent = sent[cut:].lstrip()
            if sent:
                out.append(sent)
    return out


def recursive_split(text: str, max_chars: int, overlap: float) -> list[str]:
    """Pieces of at most `max_chars`, cut at paragraph, then sentence, then word boundaries; each piece
    after the first starts with the tail (`overlap` x max_chars) of the one before."""
    parts: list[str] = []
    cur = ""
    keep = int(max_chars * overlap)
    for unit in _units(text, max_chars):
        if cur and len(cur) + 2 + len(unit) > max_chars:
            parts.append(cur)
            tail = cur[-keep:] if keep else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = f"{tail}\n\n{unit}" if tail and len(tail) + 2 + len(unit) <= max_chars else unit
        else:
            cur = f"{cur}\n\n{unit}" if cur else unit
    if cur:
        parts.append(cur)
    return parts


def _cap() -> tuple[int, float]:
    return settings.filings_passage_chars, settings.filings_overlap


def _by_pages(pages: list[dict]) -> list[tuple[str, int, int]]:
    """Consecutive pages merged up to the cap, so each piece knows its page range; a page that is too
    long on its own is split within itself."""
    max_chars, overlap = _cap()
    out: list[tuple[str, int, int]] = []
    buf, first, last = "", None, None
    for p in pages:
        text = p["text"].strip()
        if not text:
            continue
        if len(text) > max_chars:
            if buf:
                out.append((buf, first, last))
                buf, first = "", None
            out += [(piece, p["page"], p["page"]) for piece in recursive_split(text, max_chars, overlap)]
            continue
        if buf and len(buf) + 2 + len(text) > max_chars:
            out.append((buf, first, last))
            buf, first = "", None
        buf = f"{buf}\n\n{text}" if buf else text
        first = first if first is not None else p["page"]
        last = p["page"]
    if buf:
        out.append((buf, first, last))
    return out


def _page_of(snippet: str, pages: list[dict]) -> int:
    probe = " ".join(snippet.split())[:40]
    for p in pages:
        if probe and probe in " ".join(p["text"].split()):
            return p["page"]
    return pages[0]["page"] if pages else 1


def _call_passages(meta: dict, pages: list[dict]) -> list[Passage] | None:
    """Speaker turns from the transcript parser: prepared remarks by size, then each analyst question
    with the management answers that follow it. None when the parser finds too few turns."""
    from pennystock.data.transcripts import Transcript, parse
    t = parse(Transcript(symbol=meta.get("symbol", ""), doc_date=meta.get("doc_date", ""), title="", url="",
                         text="\n".join(p["text"] for p in pages)))
    if len(t.turns) < 10:
        return None
    max_chars, overlap = _cap()
    groups: list[tuple[str, list]] = []
    prepared = [tn for tn in t.prepared if tn.role == "management"]
    if prepared:
        groups.append(("prepared remarks", prepared))
    for tn in t.qa:
        if tn.role == "moderator":
            continue
        if tn.role == "analyst" or not groups or groups[-1][0] == "prepared remarks":
            groups.append((f"Q&A - {tn.speaker}" if tn.role == "analyst" else "Q&A", [tn]))
        else:
            groups[-1][1].append(tn)
    out: list[Passage] = []
    for label, turns in groups:
        body = "\n".join(f"{tn.speaker}: {tn.text}" for tn in turns)
        first, last = _page_of(turns[0].text, pages), _page_of(turns[-1].text, pages)
        for piece in recursive_split(body, max_chars, overlap):
            out.append(Passage(text=piece, page_from=first, page_to=max(first, last), speaker=label))
    return out


def passages_for(meta: dict, pages: list[dict]) -> list[Passage]:
    """The embeddable passages of one filing. `meta`: symbol, company, doc_type, doc_date, fiscal_period;
    `pages`: page, text, kind, section."""
    doc_type = meta["doc_type"]
    max_chars, overlap = _cap()
    prose = [p for p in pages if p.get("kind", "prose") == "prose" and p["text"].strip()]
    if doc_type == "earnings_call":
        raw = _call_passages(meta, prose)
        if raw is None:
            raw = [Passage(t, a, b) for t, a, b in _by_pages(prose)]
    elif doc_type == "investor_presentation":
        raw = [Passage(p["text"].strip(), p["page"], p["page"]) for p in prose
               if len(p["text"].strip()) >= _MIN_SLIDE_CHARS]
        raw = [Passage(piece, r.page_from, r.page_to) for r in raw
               for piece in recursive_split(r.text, max_chars, overlap)]
    elif doc_type == "annual_report":
        raw = []
        run: list[dict] = []
        for p in prose + [None]:
            if (p is not None and run and p.get("section") == run[-1].get("section")
                    and p["page"] == run[-1]["page"] + 1):
                run.append(p)
                continue
            if run:
                raw += [Passage(t, a, b, section=run[0].get("section")) for t, a, b in _by_pages(run)]
            run = [p] if p is not None else []
    else:                                   # results (prose pages only), credit_rating, material_event
        raw = [Passage(t, a, b) for t, a, b in _by_pages(prose)]
    company = meta.get("company") or meta.get("symbol", "")
    return [Passage(text=context_line(company, doc_type, meta.get("doc_date", ""), meta.get("fiscal_period"),
                                      r.page_from, r.page_to, r.section, r.speaker) + "\n" + r.text,
                    page_from=r.page_from, page_to=r.page_to, section=r.section, speaker=r.speaker)
            for r in raw]
