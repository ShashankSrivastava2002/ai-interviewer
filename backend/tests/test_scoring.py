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


def comp(key, levels, quotes):
    return {"key": key, "assessment": "a",
            "expectations": [{"id": f"{key}.{i + 1}", "level": lv, "evidence": [{"question": 2, "quote": q}] if q else []}
                             for i, (lv, q) in enumerate(zip(levels, quotes))],
            "improvements": [{"title": "Use I", "detail": "d", "seen_in": [2, 99]}], "strengths": [],
            "questions": [{"question": 2, "rating": "moderate", "verdict": "v", "strong_answer": "s"}] if key == "technical_background" else []}


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


def test_scores_evidence_and_overall_average():
    raw = {"overall_summary": "x", "doing_well": ["a"], "improve": ["b"], "competencies": [
        comp("role_alignment", [2, 0], ["To a— for making a platform", None]),
        comp("technical_background", [3, 2], ["I have hosted it over the Kubernetes", "solved about 20 to 30 use cases"]),
        # quote not in the transcript -> level forced to 0 (Not Assessed)
        comp("impact_ownership", [3, 1], ["I single-handedly led a team of fifty", "which is which is"]),
        comp("communication", [1, 1], ["which is which is an MCP", "So I have been assigned"]),
    ]}
    r = scoring.build_report(raw, PLAN, QUESTIONS, STATS)
    c = {x["key"]: x for x in r["competencies"]}
    assert c["role_alignment"]["score"] == 67               # one assessed expectation at 2/3
    assert c["technical_background"]["score"] == 83         # (3+2)/6
    io = c["impact_ownership"]
    assert io["expectations"][0]["level"] == 0 and io["expectations"][0]["unverified_quotes_dropped"] == 1
    assert io["score"] == 33
    # communication: rubric 33 blended 70/30 with delivery 40
    assert c["communication"]["rubric_score"] == 33 and c["communication"]["delivery_score"] == 40
    assert c["communication"]["score"] == round(0.7 * 33 + 0.3 * 40)
    scores = [x["score"] for x in r["competencies"]]
    assert r["overall_score"] == round(sum(scores) / 4)
    assert r["overall_band"] == scoring.band(r["overall_score"])
    # invalid question numbers dropped from seen_in; rating normalised
    assert c["role_alignment"]["improvements"][0]["seen_in"] == [2]
    assert c["technical_background"]["questions"][0]["rating"] == "Moderate"
    # every answered question gets a card somewhere (Q1 was forgotten by the model)
    cards = [q["question"] for x in r["competencies"] for q in x["questions"]]
    assert sorted(cards) == [1, 2]
