"""Earnings-call transcript intelligence - reading what management SAYS, not just what they report.

The numbers tell you what already happened. The call tells you what management believes is about
to happen, and - more usefully - what they are avoiding. The research is unusually clear here:
NLP signals from earnings calls have produced >12% annualised alpha in long-short tests, adding
~50% on top of traditional earnings measures, and the Q&A section carries more information than
the prepared remarks. Deceptive management language is statistically detectable.

Four signals are computed, all deterministic and all in code:

  1. TONE          Loughran-McDonald financial sentiment on management's own words.
  2. HEDGING       weak-modal / (weak + strong) modal language. "We WILL commission the plant"
                   becoming "we MAY commission the plant" is a downgrade nobody announces.
  3. EVASION       For every analyst question, did management actually answer it? Measured by
                   content-word overlap plus a non-answer phrase list. The Q&A is where scripted
                   confidence meets an unscripted question.
  4. NEW RISK      Negative / uncertainty words that appear this quarter and did NOT appear last
                   quarter. New worry words are how a problem enters the language before it
                   enters the numbers.

Guidance sentences are EXTRACTED (forward-looking cues) rather than judged - the analyst LLM does
the comparing, because "did they quietly drop the target?" is a language judgement, not arithmetic.
That keeps the split this codebase already follows: compute in code, judge in the model.

Transcripts are cached as extracted TEXT in market.db, so a 550KB PDF is downloaded once.
"""
from __future__ import annotations

import io
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from pennystock.config import settings
from pennystock.data.lexicon import lexicon
from pennystock.data.nse_client import nse_api, nse_session

logger = logging.getLogger(__name__)

_SCHEMA = """CREATE TABLE IF NOT EXISTS transcripts (
    url TEXT PRIMARY KEY, symbol TEXT NOT NULL, doc_date TEXT, title TEXT,
    text TEXT, n_chars INTEGER, fetched_at TEXT)"""

# announcement descriptions that carry a call transcript
TRANSCRIPT_HINT = re.compile(r"transcript|earnings\s+call|con\.?\s*call|conference\s+call", re.I)
SPEAKER = re.compile(r"^([A-Z][A-Za-z.'\- ]{2,40}):\s*(.*)$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

# The moderator INTRODUCES every analyst ("...from the line of Satyadeep Jain with Ambit Capital"),
# which identifies the analysts by name far more reliably than guessing from turn length.
ANALYST_INTRO = re.compile(r"line of\s+([A-Za-z][A-Za-z.'\- ]{2,40}?)\s+(?:with|from|of)\s", re.I)
# The real Q&A boundary. NOT any mention of "question": the moderator's opening welcome always says
# a Q&A will follow, which would otherwise put the boundary at the very first turn.
QA_START = re.compile(r"\bfirst question\b|we will now (?:begin|take up) the question", re.I)

# forward-looking language - the sentences that constitute guidance
GUIDANCE_CUES = re.compile(
    r"\b(we (?:expect|anticipate|target|aim|plan|intend|guide|project)|guidance|outlook|"
    r"going forward|next (?:quarter|year|fiscal)|by (?:FY|Q[1-4]|20\d\d)|on track|"
    r"capex of|capacity of|commission|ramp[- ]up|targeting|we will|we should be able)\b", re.I)

# phrases that answer a question without answering it
# HIGH-PRECISION deflection markers. These are things management says INSTEAD of answering, and
# unlike word-overlap heuristics they almost never misfire.
NON_ANSWER = re.compile(
    r"\b(we (?:do not|don't|can't|cannot) (?:provide|give|guide|comment|disclose|share|quantify)|"
    r"cannot comment|can't comment|not in a position to|too early to (?:say|comment|tell)|"
    r"difficult to (?:say|predict|comment|quantify)|hard to (?:say|predict|quantify)|"
    r"as i (?:said|mentioned|explained) (?:earlier|before|already)|"
    r"let(?:'s| us) (?:take|discuss|do) (?:this|that|it) offline|(?:get|come) back to you|"
    r"we (?:will|would) not (?:like to|want to|be able to) (?:comment|share|disclose)|"
    r"no specific (?:number|guidance|timeline)|i(?:'m| am) not going to (?:comment|give|get into)|"
    r"i (?:don't|do not) (?:know|have that)|not something we (?:disclose|share|guide)|"
    r"we have not (?:guided|disclosed)|prefer not to)\b", re.I)

STOPWORDS = set("""a an the and or but if of in on at to for with from by as is are was were be been
being this that these those it its we our you your they their he she his her i me my what which who
whom how when where why do does did done have has had can could would should will may might must
about into over under than then so such not no nor very just also more most some any all each
per cent percent crore lakh million billion year quarter thanks thank yes okay sir please
""".split())


@dataclass
class Turn:
    speaker: str
    text: str
    role: str = "unknown"          # management | analyst | moderator


@dataclass
class Transcript:
    symbol: str
    doc_date: str
    title: str
    url: str
    text: str = ""
    turns: list[Turn] = field(default_factory=list)
    prepared: list[Turn] = field(default_factory=list)
    qa: list[Turn] = field(default_factory=list)


# ------------------------------------------------------------------ storage
def _connect():
    con = sqlite3.connect(str(settings.market_db))
    con.executescript(_SCHEMA)
    return con


def save_transcript(symbol: str, doc_date: str, title: str, url: str, text: str) -> None:
    with _connect() as con:
        con.execute("INSERT INTO transcripts (url, symbol, doc_date, title, text, n_chars, fetched_at)"
                    " VALUES (?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET"
                    "  text=excluded.text, n_chars=excluded.n_chars, fetched_at=excluded.fetched_at",
                    (url, symbol.upper(), doc_date, title, text, len(text),
                     datetime.now().isoformat(timespec="seconds")))


def load_cached(symbol: str, limit: int) -> list[Transcript]:
    with _connect() as con:
        rows = con.execute(
            "SELECT symbol, doc_date, title, url, text FROM transcripts WHERE symbol=?"
            " ORDER BY fetched_at DESC LIMIT ?", (symbol.upper(), limit)).fetchall()
    return [Transcript(*r) for r in rows]


# ------------------------------------------------------------------ discovery + download
def discover(symbol: str, limit: int) -> list[dict]:
    """Transcript announcements for a symbol, newest first."""
    session = nse_session(symbol)
    rows = nse_api(session, f"corporate-announcements?index=equities&symbol={symbol.upper()}") or []
    out = []
    for r in rows:
        blob = f"{r.get('desc', '')} {r.get('attchmntText', '')}"
        f = r.get("attchmntFile") or ""
        if TRANSCRIPT_HINT.search(blob) and f.lower().endswith(".pdf") and "transcript" in f.lower():
            out.append({"date": str(r.get("an_dt", ""))[:11], "title": str(r.get("desc", "")), "url": f})
        if len(out) >= limit:
            break
    return out, session


def download_text(session, url: str) -> str:
    """Fetch a transcript PDF and extract its text. Returns '' on any failure."""
    try:
        from pypdf import PdfReader
        r = session.get(url, timeout=settings.transcript_timeout_s)
        if r.status_code != 200:
            logger.warning(f"transcript download {r.status_code}: {url}")
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        logger.warning(f"transcript extraction failed ({type(e).__name__}: {e}): {url}")
        return ""


def fetch_transcripts(symbol: str, limit: int | None = None) -> list[Transcript]:
    """Cache-first: return up to `limit` recent transcripts, downloading only what is missing."""
    limit = limit or settings.transcript_max_quarters
    symbol = symbol.upper()
    with _connect() as con:
        have = {r[0] for r in con.execute("SELECT url FROM transcripts WHERE symbol=?", (symbol,))}
    try:
        found, session = discover(symbol, limit)
    except Exception as e:
        logger.warning(f"transcript discovery failed for {symbol}: {e}")
        return load_cached(symbol, limit)

    for item in found:
        if item["url"] in have:
            continue
        text = download_text(session, item["url"])
        if text.strip():
            save_transcript(symbol, item["date"], item["title"], item["url"], text)

    with _connect() as con:
        rows = con.execute(
            "SELECT symbol, doc_date, title, url, text FROM transcripts WHERE symbol=? AND url IN"
            f" ({','.join('?' * len(found))})" if found else
            "SELECT symbol, doc_date, title, url, text FROM transcripts WHERE symbol=?",
            ([symbol] + [f["url"] for f in found]) if found else (symbol,)).fetchall()
    order = {f["url"]: i for i, f in enumerate(found)}
    tx = [Transcript(*r) for r in rows]
    tx.sort(key=lambda t: order.get(t.url, 999))          # newest first, as discovered
    return tx[:limit]


# ------------------------------------------------------------------ parsing
def parse(t: Transcript) -> Transcript:
    """Split raw text into speaker turns, label roles, and separate prepared remarks from Q&A.

    Three real-world quirks are handled explicitly, each found by inspecting actual NSE filings:
      1. The PDF opens with a covering letter whose lines ("Registered Office:", "Scrip Code:")
         look exactly like speaker labels. Everything before the first Moderator turn is dropped.
      2. The moderator's WELCOME always promises a Q&A session, so keying the boundary on the word
         "question" puts it at turn one. We key on "first question" instead.
      3. Roles are assigned from the moderator's own introductions rather than guessed: every
         analyst is introduced by name, so anyone not introduced and not the moderator is
         management.
    """
    turns: list[Turn] = []
    current: Turn | None = None
    for line in t.text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = SPEAKER.match(line)
        if m:
            if current:
                turns.append(current)
            current = Turn(speaker=m.group(1).strip(), text=m.group(2))
        elif current:
            current.text += " " + line
    if current:
        turns.append(current)

    # (1) drop the covering-letter preamble
    first_mod = next((i for i, tn in enumerate(turns) if "moderator" in tn.speaker.lower()), None)
    if first_mod is not None:
        turns = turns[first_mod:]

    # (3) analysts are the people the moderator introduces
    analysts = set()
    for tn in turns:
        if "moderator" in tn.speaker.lower():
            for name in ANALYST_INTRO.findall(tn.text):
                analysts.add(" ".join(name.split()).lower())

    # (2) the real Q&A boundary
    qa_start = next((i for i, tn in enumerate(turns)
                     if "moderator" in tn.speaker.lower() and QA_START.search(tn.text)), None)
    if qa_start is None:
        qa_start = int(len(turns) * 0.4) if turns else 0

    for tn in turns:
        low = tn.speaker.lower()
        tn.role = ("moderator" if "moderator" in low else
                   "analyst" if low in analysts else "management")
    t.turns, t.prepared, t.qa = turns, turns[:qa_start], turns[qa_start:]
    return t


def management_text(t: Transcript) -> str:
    return " ".join(tn.text for tn in t.turns if tn.role == "management")


# ------------------------------------------------------------------ signals
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def tone(text: str) -> dict:
    """Loughran-McDonald category counts, rates per 1,000 words, and the hedging ratio."""
    lx = lexicon()
    words = _words(text)
    n = len(words) or 1
    counts = {cat: sum(1 for w in words if w in vocab) for cat, vocab in lx.items()}
    pos, neg = counts["Positive"], counts["Negative"]
    strong, weak = counts["StrongModal"], counts["WeakModal"]
    return {
        "words": n,
        "positive_per_1k": round(pos / n * 1000, 2),
        "negative_per_1k": round(neg / n * 1000, 2),
        "uncertainty_per_1k": round(counts["Uncertainty"] / n * 1000, 2),
        "litigious_per_1k": round(counts["Litigious"] / n * 1000, 2),
        "net_tone": round((pos - neg) / (pos + neg), 3) if (pos + neg) else None,
        "hedging_ratio": round(weak / (weak + strong), 3) if (weak + strong) else None,
        "strong_modal": strong, "weak_modal": weak,
    }


def qa_pairs(t: Transcript) -> list[tuple[str, str]]:
    """(analyst question, full management answer) pairs from the Q&A section.

    An answer routinely spans SEVERAL consecutive management turns - the CEO starts, the CFO adds
    the numbers. Taking only the first turn made complete answers look evasive, so every management
    turn up to the next analyst or moderator is joined into one answer.
    """
    pairs: list[tuple[str, str]] = []
    question: str | None = None
    answer: list[str] = []
    for tn in t.qa:
        if tn.role == "management" and question:
            answer.append(tn.text)
            continue
        if question and answer:                      # the reply is complete
            pairs.append((question, " ".join(answer)))
            question, answer = None, []
        if tn.role == "analyst" and "?" in tn.text:
            question, answer = tn.text, []
    if question and answer:
        pairs.append((question, " ".join(answer)))
    return pairs


def evasion(t: Transcript) -> dict:
    """How often management deflected instead of answering.

    Two signals are reported SEPARATELY on purpose, because they are not equally trustworthy:

      explicit_non_answers  management literally says it will not answer ("we don't guide on that",
                            "too early to say", "let's take it offline"). High precision - this is
                            the number to act on, and the only one that raises a flag.
      possible_deflections  a heuristic: a substantive question whose own terms never appear in the
                            answer. Noisy, because management legitimately answers "the exceptional
                            hit" with "it's an Ind AS versus US GAAP difference" - same topic,
                            different words. Reported for a human to skim, never used as a flag.

    Conflating the two produced an 82% "evasion rate" on a perfectly normal call, which is exactly
    how a screening signal becomes noise people learn to ignore.
    """
    pairs = qa_pairs(t)
    if not pairs:
        return {"questions_analysed": 0, "note": "no question/answer pairs detected"}
    flagged = []
    judged = 0
    for q, a in pairs:
        q_words = {w for w in _words(q) if w not in STOPWORDS and len(w) > 3}
        a_words = set(_words(a))
        non_answer = bool(NON_ANSWER.search(a))
        # a short follow-up ("And Hindalco would also make 0.25%?") has no content words to match
        # on, so overlap is only meaningful for a substantive question
        substantive = len(q_words) >= 5
        overlap = (len(q_words & a_words) / len(q_words)) if q_words else 1.0
        ignores = substantive and overlap < 0.15
        terse = substantive and len(a.split()) < 25 and len(q.split()) > 40
        judged += 1
        if non_answer or ignores or terse:
            reason = ("explicit non-answer" if non_answer else
                      "answer ignores the question's terms (heuristic)" if ignores else
                      "very short answer to a detailed question (heuristic)")
            flagged.append({"reason": reason, "high_confidence": non_answer,
                            "overlap": round(overlap, 2),
                            "question": " ".join(q.split())[:220],
                            "answer": " ".join(a.split())[:220]})
    explicit = [f for f in flagged if f["high_confidence"]]
    return {
        "questions_analysed": judged,
        "explicit_non_answers": len(explicit),
        "explicit_non_answer_rate": round(len(explicit) / judged, 2) if judged else 0.0,
        "possible_deflections": len(flagged) - len(explicit),
        "examples": (explicit + [f for f in flagged if not f["high_confidence"]])
                    [: settings.transcript_evasion_limit],
        "note": "explicit_non_answers is the reliable number (management said it would not answer). "
                "possible_deflections is a noisy word-overlap heuristic - skim, do not score.",
    }


def guidance_sentences(t: Transcript, limit: int | None = None) -> list[str]:
    """Forward-looking sentences from management - the raw material for a quarter-on-quarter diff."""
    limit = limit or settings.transcript_guidance_limit
    out = []
    for sent in SENTENCE.split(management_text(t)):
        s = " ".join(sent.split())
        if 40 < len(s) < 320 and GUIDANCE_CUES.search(s):
            out.append(s)
    return out[:limit]


def new_risk_language(current: Transcript, previous: Transcript, limit: int = 12) -> list[str]:
    """Negative / uncertainty words used this quarter that were absent last quarter."""
    lx = lexicon()
    worry = lx["Negative"] | lx["Uncertainty"]
    now = Counter(w for w in _words(management_text(current)) if w in worry)
    before = set(_words(management_text(previous)))
    fresh = [(w, c) for w, c in now.most_common() if w not in before and c >= 2]
    return [f"{w} (x{c})" for w, c in fresh[:limit]]


# ------------------------------------------------------------------ report
def summarize_transcripts(transcripts: list[Transcript]) -> dict:
    """Compact, bounded commentary picture across the most recent calls."""
    if not transcripts:
        return {"transcripts": "unavailable",
                "note": "No earnings-call transcript found for this symbol. Treat management "
                        "commentary as UNCHECKED - not as clean."}
    parsed = [parse(t) for t in transcripts]
    latest = parsed[0]
    prior = parsed[1] if len(parsed) > 1 else None

    tone_now = tone(management_text(latest))
    tone_prev = tone(management_text(prior)) if prior else None
    ev = evasion(latest)

    flags: list[str] = []
    if tone_prev:
        for key, label, worse_when_up in (("net_tone", "net tone", False),
                                          ("hedging_ratio", "hedging", True),
                                          ("uncertainty_per_1k", "uncertainty language", True)):
            a, b = tone_now.get(key), tone_prev.get(key)
            if a is None or b is None:
                continue
            delta = round(a - b, 3)
            if worse_when_up and delta > 0.05:
                flags.append(f"{label.capitalize()} ROSE {delta:+} vs the previous call - "
                             "management is hedging more than it did last quarter.")
            if not worse_when_up and delta < -0.05:
                flags.append(f"{label.capitalize()} FELL {delta:+} vs the previous call - "
                             "the language has turned more negative.")
    if ev.get("explicit_non_answer_rate", 0) >= settings.transcript_non_answer_flag:
        flags.append(f"{ev['explicit_non_answers']} of {ev['questions_analysed']} analyst questions "
                     "were explicitly NOT answered ('we don't guide on that', 'too early to say'). "
                     "The Q&A is where scripted confidence meets an unscripted question.")
    if tone_now.get("hedging_ratio") is not None and tone_now["hedging_ratio"] > 0.6:
        flags.append(f"Heavily hedged language (weak-modal share {tone_now['hedging_ratio']}) - "
                     "management is committing to very little.")

    return {
        "calls_analysed": len(parsed),
        "latest_call": {"date": latest.doc_date, "title": latest.title,
                        "speakers": len({t_.speaker for t_ in latest.turns}),
                        "questions": len(qa_pairs(latest))},
        "tone_latest": tone_now,
        "tone_previous": tone_prev,
        "evasion": ev,
        "guidance_latest": guidance_sentences(latest),
        "guidance_previous": guidance_sentences(prior) if prior else [],
        "new_risk_language": new_risk_language(latest, prior) if prior else [],
        "commentary_flags": flags or ["No language-based red flags versus the previous call."],
        "how_to_read": ("Compare guidance_latest against guidance_previous: a target that was stated "
                        "last quarter and is missing this quarter is the highest-signal item in any "
                        "transcript. Quote management's own words when you make the point."),
    }
