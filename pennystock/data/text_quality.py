"""Chunk quality filter - keeps index/TOC/bibliography pages out of the RAG corpus.

These pages retrieve *well* (they are packed with topic keywords) but say nothing, so they
steal slots from real prose. On a Cipla run, 1-3 of the 8 retrieved chunks were page-number
soup like "90-101, 103-106, 109-111 ... inflation and, 47-56" or a book bibliography.

Shared by scripts/ingest_documents.py (filter at write time) and scripts/prune_junk_chunks.py
(clean what is already stored), so both use exactly the same definition of junk.
"""
from __future__ import annotations

MIN_WORDS = 40            # a real idea needs some room
MAX_DIGIT_RATIO = 0.15    # index pages are mostly page numbers
MAX_PUNCT_PER_WORD = 0.25 # index/bibliography entries are comma & semicolon soup
MAX_UPPER_RATIO = 0.40    # ALL-CAPS heading lists / running heads


def junk_reason(text: str) -> str | None:
    """Return why this chunk is junk, or None if it is usable prose."""
    text = str(text or "")
    words = text.split()
    if len(words) < MIN_WORDS:
        return f"too short ({len(words)}w)"
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > MAX_DIGIT_RATIO:
        return f"number-dense ({digit_ratio:.0%}) - index/table page"
    punct_per_word = (text.count(",") + text.count(";")) / len(words)
    if punct_per_word > MAX_PUNCT_PER_WORD:
        return f"punctuation-dense ({punct_per_word:.2f}/word) - index/bibliography"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > MAX_UPPER_RATIO:
        return "mostly capitals - heading list"
    return None


def looks_like_junk(text: str) -> bool:
    return junk_reason(text) is not None
