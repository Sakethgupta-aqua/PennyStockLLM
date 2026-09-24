from pennystock.data.transcripts import (Transcript, Turn, evasion, guidance_sentences,
                                         management_text, new_risk_language, parse, qa_pairs,
                                         summarize_transcripts, tone)

# a miniature transcript with the same quirks as a real NSE filing
RAW = """Registered Office: 21st Floor, Mumbai - 400013
Scrip Code: 500440 National Stock Exchange
Sub: Transcript - Q1 FY27 Earnings Call.
Moderator: Ladies and gentlemen, welcome to the Earnings Call. After the presentation we will
conduct a question and answer session.
Satish Pai: Good morning. Revenue grew strongly and margins improved. We will commission the
new smelter in Q3 and we expect capex of INR 5000 crores this year. Our outlook remains robust.
Moderator: The first question comes from the line of Ritesh Shah with Investec. Please go ahead.
Ritesh Shah: Can you quantify the expected margin impact from the new alumina refinery capacity?
Satish Pai: We don't provide guidance on segment margins at this stage.
Moderator: Next question comes from the line of Sumangal Nevatia with Kotak.
Sumangal Nevatia: What is the timeline for the copper expansion project and its funding mix?
Satish Pai: The copper expansion is on track for FY28 and will be funded from internal accruals.
"""

PRIOR = RAW.replace("We will commission the new smelter in Q3", "We will commission the smelter in Q2")


def make(text=RAW, date="13-Aug-2026") -> Transcript:
    return parse(Transcript(symbol="TEST", doc_date=date, title="call", url=f"u-{date}", text=text))


def test_cover_letter_preamble_is_dropped():
    t = make()
    speakers = {turn.speaker for turn in t.turns}
    assert "Registered Office" not in speakers and "Scrip Code" not in speakers
    assert "Moderator" in speakers


def test_roles_come_from_moderator_introductions():
    t = make()
    roles = {turn.speaker: turn.role for turn in t.turns}
    assert roles["Satish Pai"] == "management"      # never introduced -> management
    assert roles["Ritesh Shah"] == "analyst"        # "from the line of Ritesh Shah with Investec"
    assert roles["Sumangal Nevatia"] == "analyst"
    assert roles["Moderator"] == "moderator"


def test_qa_boundary_ignores_the_opening_welcome():
    """The welcome says 'question and answer session'; the split must be at 'first question'."""
    t = make()
    assert any("Revenue grew strongly" in turn.text for turn in t.prepared)
    assert all("Revenue grew strongly" not in turn.text for turn in t.qa)


def test_qa_pairs_and_explicit_non_answer():
    t = make()
    pairs = qa_pairs(t)
    assert len(pairs) == 2
    ev = evasion(t)
    assert ev["explicit_non_answers"] == 1          # "we don't provide guidance"
    assert ev["examples"][0]["high_confidence"] is True


def test_tone_measures_hedging():
    strong = tone("We will deliver. We must execute. We shall succeed.")
    weak = tone("We may deliver. It might happen. It could possibly occur.")
    assert strong["hedging_ratio"] == 0.0
    assert weak["hedging_ratio"] == 1.0             # all weak modal -> fully hedged


def test_guidance_sentences_are_forward_looking():
    g = " ".join(guidance_sentences(make()))
    assert "commission" in g and "capex" in g


def test_new_risk_language_only_flags_new_worry_words():
    cur = make(RAW.replace("Our outlook remains robust.",
                           "Conditions are challenging and demand is weak. Challenging weak."))
    prev = make(PRIOR, date="26-May-2026")
    fresh = " ".join(new_risk_language(cur, prev))
    assert "challenging" in fresh and "weak" in fresh


def test_summary_is_graceful_with_no_transcripts():
    out = summarize_transcripts([])
    assert out["transcripts"] == "unavailable" and "UNCHECKED" in out["note"]


def test_summary_runs_end_to_end():
    out = summarize_transcripts([make(), make(PRIOR, date="26-May-2026")])
    assert out["calls_analysed"] == 2
    assert out["tone_latest"]["words"] > 0
    assert "how_to_read" in out
