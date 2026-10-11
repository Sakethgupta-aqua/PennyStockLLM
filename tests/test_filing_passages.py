"""Passages: each filing type cut along its own structure, every passage saying where it came from."""
from __future__ import annotations

from pennystock.retrieval import filing_passages as fp

META = {"symbol": "AUBANK", "company": "AU Small Finance Bank", "doc_type": "earnings_call",
        "doc_date": "2026-07-25", "fiscal_period": "Q1 FY27"}

def page(n, text, kind="prose", section=None):
    return {"page": n, "text": text, "kind": kind, "section": section}

def test_every_passage_opens_with_where_it_came_from():
    line = fp.context_line("AU Small Finance Bank", "earnings_call", "2026-07-25", "Q1 FY27", 7, 7,
                           speaker="Q&A - Rahul Jain")
    assert line == "AU Small Finance Bank · earnings call · 2026-07-25 · Q1 FY27 · p.7 · Q&A - Rahul Jain"

def test_a_long_text_splits_on_paragraphs_with_overlap_and_never_beyond_the_cap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 150 for i in range(12))
    parts = fp.recursive_split(text, 2000, 0.25)
    assert all(len(p) <= 2000 for p in parts) and len(parts) > 3
    assert parts[1].split()[0:2] != ["Paragraph", "0"] and any(w in parts[1] for w in parts[0].split()[-20:])

def test_an_earnings_call_keeps_each_question_with_its_answer():
    call = ("Moderator: Ladies and gentlemen, welcome. We will now begin the question-and-answer session. "
            "The first question is from the line of Rahul Jain from Goldman Sachs.\n"
            "Rahul Jain: What is the NIM guidance for FY27?\n"
            "Sanjay Agarwal: We expect NIM of 5.8% to 6.0% as deposit costs ease.\n"
            "Moderator: The next question is from the line of Priya Shah from Kotak.\n"
            "Priya Shah: How is credit cost trending?\n"
            "Vivek Tripathi: Credit cost should stay near 1.5%.\n") * 3
    ps = fp.passages_for(META, [page(1, call)])
    qa = [p for p in ps if "NIM guidance" in p.text]
    assert qa and "5.8% to 6.0%" in qa[0].text and qa[0].speaker.startswith("Q&A")

def test_a_call_the_speaker_parser_cannot_read_falls_back_to_plain_splits():
    ps = fp.passages_for(META, [page(1, "Our deposits grew strongly this quarter. " * 200)])
    assert ps and all(p.speaker is None for p in ps)

def test_presentations_give_one_passage_per_slide_and_skip_image_slides():
    ps = fp.passages_for({**META, "doc_type": "investor_presentation"},
                         [page(1, "Q1 FY27 highlights: advances up 22% " * 5), page(2, "", "prose"),
                          page(3, "CASA ratio 31% " * 5)])
    assert [(p.page_from, p.page_to) for p in ps] == [(1, 1), (3, 3)]

def test_table_pages_and_annual_report_boilerplate_are_never_embedded():
    res = fp.passages_for({**META, "doc_type": "results"},
                          [page(1, "Press release: profit rose 18% on loan growth " * 10),
                           page(2, "12,345 6,789 " * 50, "table")])
    assert [p.page_from for p in res] == [1]
    ar = fp.passages_for({**META, "doc_type": "annual_report"},
                         [page(1, "CSR spend details " * 30, "boilerplate"),
                          page(2, "Margins improved " * 30, "prose", "mdna")])
    assert [p.section for p in ar] == ["mdna"]
