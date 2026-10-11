"""Scoring the filings search against its golden set: did a result cover the answer page of the right file?"""
from collections import Counter

from pennystock.evals import filings_eval as fe

ROW = {"content_sha": "s1", "page_from": 7, "page_to": 8}

def test_a_result_counts_when_it_covers_the_answer_page_of_the_same_file():
    assert fe.page_hit({"content_sha": "s1", "page_from": 8, "page_to": 9}, ROW)
    assert not fe.page_hit({"content_sha": "s2", "page_from": 7, "page_to": 7}, ROW)
    assert not fe.page_hit({"content_sha": "s1", "page_from": 10, "page_to": 11}, ROW)

def test_recall_and_mrr_per_configuration():
    hit = {"content_sha": "s1", "page_from": 7, "page_to": 7}
    miss = {"content_sha": "s9", "page_from": 1, "page_to": 1}
    out = fe.score({"fused": [[miss, hit], [miss] * 12]}, [ROW, ROW], ks=(1, 5))
    assert out["fused"] == {"recall@1": 0.0, "recall@5": 0.5, "mrr": 0.25, "n": 2}


def test_a_question_that_points_at_the_document_is_rejected():
    assert fe.usable_question("What NIM range did management guide for FY27 on the earnings call?")[0]
    assert not fe.usable_question("What does the passage say about margins?")[0]
    assert not fe.usable_question("Margins?")[0]


def test_the_sample_spreads_over_document_types_and_companies():
    items = [(f"ar{i}", {"doc_type": "annual_report", "symbol": "BIG"}) for i in range(500)]
    items += [(f"ar-s{i}", {"doc_type": "annual_report", "symbol": f"S{i}"}) for i in range(5)]
    items += [(f"ec{i}", {"doc_type": "earnings_call", "symbol": "BIG" if i % 2 else "MID"})
              for i in range(40)]
    items += [("cr1", {"doc_type": "credit_rating", "symbol": "TINY"})]
    meta = dict(items)
    picked = fe.sample_passages(items, 12, seed=1)
    types = Counter(meta[p]["doc_type"] for p in picked)
    assert len(picked) == len(set(picked)) == 12
    assert types["credit_rating"] == 1                                   # a scarce type still gets in
    assert abs(types["annual_report"] - types["earnings_call"]) <= 1      # the rest split evenly
    ar_companies = Counter(meta[p]["symbol"] for p in picked if meta[p]["doc_type"] == "annual_report")
    assert ar_companies["BIG"] == 1                       # 500 passages of one company do not swamp it
    assert fe.sample_passages(items, 12, seed=1) == picked     # same seed, same sample
