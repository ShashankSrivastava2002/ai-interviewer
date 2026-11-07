"""Tests for evidence verification, /100 scores and bands (scoring.py)."""
from backend import scoring
from backend.prompts import COMPETENCY_KEYS

PLAN = {"sections": [{"key": k, "expectations": [f"{k} exp A", f"{k} exp B"]} for k in COMPETENCY_KEYS]}
ANSWER = ("So I have been assigned. To a— for making a platform which is which is an MCP platform. "
          "uh I have hosted it over the Kubernetes and I have solved about 20 to 30 use cases")
QUESTIONS = [
    {"number": 1, "id": 11, "verbatim": "I moved for scale, three million shipments per day", "skipped": False, "section": "role_alignment"},
    {"number": 2, "id": 12, "verbatim": ANSWER, "skipped": False, "section": "technical_background"},
    {"number": 3, "id": 13, "verbatim": None, "skipped": True, "section": "impact_ownership"},
]
STATS = {"fluency_score": 4.0}


def comp(key, levels, quotes, score=None, questions=()):
    c = {"key": key, "assessment": "a", "score_rationale": "why",
         "expectations": [{"id": f"{key}.{i + 1}", "level": lv, "evidence": [{"question": 2, "quote": q}] if q else []}
                          for i, (lv, q) in enumerate(zip(levels, quotes))],
         "improvements": [{"title": "Use I", "detail": "d", "seen_in": [2, 99]}], "strengths": [],
         "questions": list(questions)}
    if score is not None:
        c["score"] = score
    return c


def test_bands():
    assert scoring.band(29) == "Weak"
    assert scoring.band(30) == "Moderate"
    assert scoring.band(49) == "Moderate"
    assert scoring.band(55) == "Good"
    assert scoring.band(75) == "Excellent"
    assert scoring.band(None) is None


def test_quote_matching_is_verbatim_but_tolerant_of_punctuation():
    assert scoring.quote_in_answer("which is which is an MCP platform", ANSWER)
    assert scoring.quote_in_answer("I have hosted it over the Kubernetes.", ANSWER)
    assert not scoring.quote_in_answer("I architected the whole Kubernetes cluster myself", ANSWER)


def test_real_words_ignores_fillers_and_tags():
    assert scoring.real_words("um [breath] uh research") == 1
    assert scoring.real_words("[breath]") == 0


def test_llm_scores_are_final_and_untouched():
    raw = {"overall_summary": "x", "doing_well": ["a", " "], "improve": ["b"], "competencies": [
        comp("role_alignment", [2, 2], ["To a— for making a platform", "which is which is an MCP platform"], score=100),
        comp("technical_background", [3, 2], ["I have hosted it over the Kubernetes", "solved about 20 to 30 use cases"],
             score=63, questions=[{"question": 2, "rating": "good", "verdict": "v", "strong_answer": "s"}]),
        comp("impact_ownership", [3, 1], ["I single-handedly led a team of fifty", "which is which is"], score=55.4),
        comp("communication", [1, 1], ["which is which is an MCP", "So I have been assigned"], score=-5),
    ]}
    r = scoring.build_report(raw, PLAN, QUESTIONS, STATS)
    c = {x["key"]: x for x in r["competencies"]}
    # no blending, no caps: the model's numbers pass through (only clamped to 0–100 and rounded)
    assert [c[k]["score"] for k in ("role_alignment", "technical_background", "impact_ownership", "communication")] == [100, 63, 55, 0]
    assert r["overall_score"] == round((100 + 63 + 55 + 0) / 4)
    assert r["overall_band"] == scoring.band(r["overall_score"])
    # levels are kept as given; an invented quote is only FLAGGED
    io = c["impact_ownership"]
    assert [e["level"] for e in io["expectations"]] == [3, 1]
    assert io["expectations"][0]["evidence"][0]["verified"] is False
    assert io["expectations"][1]["evidence"][0]["verified"] is True
    assert io["unverified_quotes"] == 1
    assert c["technical_background"]["questions"][0]["rating"] == "Good"
    assert c["role_alignment"]["improvements"][0]["seen_in"] == [2]
    assert r["doing_well"] == ["a"]
    # an answered question the model forgot still gets an (unrated) card
    cards = {q["question"]: q["rating"] for x in r["competencies"] for q in x["questions"]}
    assert cards == {1: "Not rated", 2: "Good"}


def test_missing_score_stays_missing():
    raw = {"competencies": [comp("role_alignment", [2, 0], ["To a— for making a platform", None])]}
    r = scoring.build_report(raw, PLAN, QUESTIONS, STATS)
    c = {x["key"]: x for x in r["competencies"]}
    assert c["role_alignment"]["score"] is None and c["role_alignment"]["band"] is None
    assert r["overall_score"] is None


def test_disfluency_rate_fact():
    assert scoring.disfluency_rate({"word_count": 100, "fillers": {"total": 6}, "repetitions": 4, "false_starts": 1, "stammers": 0}) == 11.0
    assert scoring.disfluency_rate({"word_count": 0}) is None


def test_question_score_is_direct_and_band_is_derived():
    raw = {"competencies": [comp("technical_background", [2], ["I have hosted it over the Kubernetes"], score=60,
                                 questions=[{"question": 2, "score": 42, "rating": "Excellent", "verdict": "v", "strong_answer": "s"},
                                            {"question": 1, "score": 81}])]}
    r = scoring.build_report(raw, PLAN, QUESTIONS, STATS)
    qs = {q["question"]: q for c in r["competencies"] for q in c["questions"]}
    assert (qs[2]["score"], qs[2]["rating"]) == (42, "Moderate")    # the label follows the number, not the model's word
    assert (qs[1]["score"], qs[1]["rating"]) == (81, "Excellent")
