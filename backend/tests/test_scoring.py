"""Tests for strict scoring: evidence checks, substance gates, points, bands (scoring.py)."""
from backend import scoring
from backend.prompts import COMPETENCY_KEYS

PLAN = {"sections": [{"key": k, "expectations": [f"{k} exp A", f"{k} exp B"]} for k in COMPETENCY_KEYS]}
LONG = ("So I was the main backend engineer on our MCP platform. I designed the workflow and agent orchestration "
        "layer myself, deployed the whole stack on Kubernetes, set up the LLM gateway for cost governance, and "
        "onboarded around twenty to thirty use cases, ten of which are now in production serving about one hundred "
        "thousand LLM calls per day across teams.")


def q(n, text, section, skipped=False):
    words = len(text.split()) if text else 0
    return {"number": n, "id": 100 + n, "verbatim": text, "skipped": skipped, "section": section, "content_words": words}


def comp(key, levels_quotes, questions=()):
    return {"key": key, "assessment": "a",
            "expectations": [{"id": f"{key}.{i + 1}", "level": lv, "evidence": [{"question": qn, "quote": qt}] if qt else []}
                             for i, (lv, qn, qt) in enumerate(levels_quotes)],
            "improvements": [{"title": "Use I", "detail": "d", "seen_in": [1, 99]}], "strengths": [],
            "questions": [{"question": n, "rating": r, "verdict": "v", "strong_answer": "s"} for n, r in questions]}


def test_bands():
    assert scoring.band(29) == "Weak"
    assert scoring.band(30) == "Moderate"
    assert scoring.band(49) == "Moderate"
    assert scoring.band(55) == "Good"
    assert scoring.band(75) == "Excellent"


def test_quote_matching_is_verbatim_but_tolerant_of_punctuation():
    assert scoring.quote_in_answer("deployed the whole stack on Kubernetes", LONG)
    assert scoring.quote_in_answer("I designed the workflow, and agent orchestration", LONG)
    assert not scoring.quote_in_answer("I architected the whole Kubernetes cluster myself", LONG)


def test_the_bad_interview_scores_weak():
    """Reproduces the real report that wrongly came out at 48/100 'Moderate'."""
    qs = [
        q(1, "um By the way what do you call the system alo What are you already talking um How am I engineering "
             "AI engineer don't you have anything to say I've completed my graduation from his t- honours and I'm "
             "looking forward to work as AI engineer", "role_alignment"),
        q(2, "Systems uh system", "role_alignment"),
        q(3, "Research", "role_alignment"),
        q(4, "", "technical_background"),
        q(5, "I tend to use AI for editing videos", "technical_background"),
        q(7, "um", "technical_background"),
    ]
    raw = {"overall_summary": "x", "doing_well": [], "improve": ["a"], "competencies": [
        comp("role_alignment", [(1, 1, "looking forward to work as AI engineer"), (1, 2, "Systems uh system")],
             [(1, "Weak"), (2, "Weak"), (3, "Weak")]),
        # the model was lenient here: "Partially met" on a one-liner -> must be lowered to Not met
        comp("technical_background", [(2, 5, "I tend to use AI for editing videos"), (0, 0, None)], [(5, "Moderate")]),
        comp("impact_ownership", [(0, 0, None), (0, 0, None)]),
        comp("communication", [(1, 1, "don't you have anything to say"), (1, 2, "Systems uh system")]),
    ]}
    r = scoring.build_report(raw, PLAN, qs, {"fluency_score": 3.0})
    c = {x["key"]: x for x in r["competencies"]}
    assert c["technical_background"]["expectations"][0]["level"] == 1
    assert "lowered" in c["technical_background"]["expectations"][0]["note"]
    assert c["role_alignment"]["score"] == 10                       # two Not met
    assert c["technical_background"]["score"] == 5                  # Not met + No evidence
    assert c["impact_ownership"]["score"] == 0 and c["impact_ownership"]["band"] == "Not reached"
    assert c["communication"]["score"] == round(0.7 * 10 + 0.3 * 30)
    assert r["overall_score"] < 15 and r["overall_band"] == "Weak"
    # a 7-word answer can't be rated better than Weak
    assert c["technical_background"]["questions"][0]["rating"] == "Weak"


def test_strong_interview_scores_well_and_gates_need_substance():
    qs = [q(1, LONG, "role_alignment"), q(2, LONG, "technical_background"),
          q(3, LONG, "impact_ownership"), q(4, LONG, "communication")]
    quote = "deployed the whole stack on Kubernetes"
    raw = {"overall_summary": "x", "doing_well": ["a"], "improve": ["b"], "competencies": [
        comp(k, [(3, n, quote), (2, n, quote)], [(n, "Excellent")]) for n, k in enumerate(COMPETENCY_KEYS, 1)]}
    r = scoring.build_report(raw, PLAN, qs, {"fluency_score": 8.0})
    c = {x["key"]: x for x in r["competencies"]}
    assert c["technical_background"]["score"] == round((100 + 45) / 2)
    assert c["technical_background"]["questions"][0]["rating"] == "Excellent"
    assert r["overall_band"] == "Good"


def test_fabricated_quotes_are_dropped():
    qs = [q(1, LONG, "impact_ownership")]
    raw = {"competencies": [comp("impact_ownership", [(3, 1, "I single-handedly led a team of fifty"), (0, 0, None)])]}
    r = scoring.build_report(raw, PLAN, qs, {"fluency_score": None})
    e = r["competencies"][2]["expectations"][0]
    assert e["level"] == 0 and e["unverified_quotes_dropped"] == 1 and e["label"] == "No evidence"


def test_seen_in_filtered_and_forgotten_questions_get_cards():
    qs = [q(1, LONG, "role_alignment"), q(2, "short answer here", "technical_background")]
    raw = {"competencies": [comp("role_alignment", [(2, 1, "deployed the whole stack on Kubernetes")], [(1, "good")])]}
    r = scoring.build_report(raw, PLAN, qs, {"fluency_score": 5.0})
    c = {x["key"]: x for x in r["competencies"]}
    assert c["role_alignment"]["improvements"][0]["seen_in"] == [1]
    assert c["role_alignment"]["questions"][0]["rating"] == "Good"
    assert [x["question"] for x in c["technical_background"]["questions"]] == [2]
    assert c["technical_background"]["questions"][0]["rating"] == "Weak"
