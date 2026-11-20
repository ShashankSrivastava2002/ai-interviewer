"""evaluate=false: the report has empty scores and no LLM call is made."""
from backend import scoring
from backend.prompts import COMPETENCY_KEYS


def test_transcript_only_report_has_empty_scores():
    r = scoring.transcript_only_report()
    assert r["evaluated"] is False and r["overall_score"] is None and r["overall_band"] is None
    assert [c["key"] for c in r["competencies"]] == list(COMPETENCY_KEYS)
    assert all(c["score"] is None and c["band"] is None and c["questions"] == [] for c in r["competencies"])
