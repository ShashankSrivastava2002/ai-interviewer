"""Tests for the second report call (original answer + feedback -> strong answer)."""
import asyncio

from backend import llm, prompts, strong_answers

QUESTIONS = [
    {"number": 1, "text": "Why this role?", "verbatim": "um I like AI", "skipped": False},
    {"number": 2, "text": "Tell me about a project", "verbatim": "I built an MCP platform", "skipped": False},
    {"number": 3, "text": "Skipped one", "verbatim": None, "skipped": True},
]


def report():
    return {"competencies": [{
        "name": "Role Alignment", "improvements": [{"title": "Link to role", "detail": "Be specific", "seen_in": [1]}],
        "questions": [{"question": 1, "score": 20, "verdict": "Too vague", "strong_answer": ""},
                      {"question": 2, "score": 60, "verdict": "No result", "strong_answer": ""}]}]}


def run(rep):
    asyncio.run(strong_answers.add_strong_answers(rep, {"role": "SDE"}, QUESTIONS, None, "m", None))


def test_prompt_has_original_answer_and_feedback():
    msg = prompts.strong_answer_user_message({"role": "SDE"}, report(), QUESTIONS)
    assert "um I like AI" in msg and "Too vague" in msg and "Link to role: Be specific" in msg
    assert '"question": 3' not in msg


def test_answers_fill_cards_without_touching_scores(monkeypatch):
    async def fake(*a, **k):
        return {"answers": [{"question": 1, "strong_answer": "Better"}, {"question": "2", "strong_answer": "Best"}]}
    monkeypatch.setattr(llm, "chat", fake)
    rep = report()
    run(rep)
    cards = rep["competencies"][0]["questions"]
    assert [c["strong_answer"] for c in cards] == ["Better", "Best"] and [c["score"] for c in cards] == [20, 60]
    assert rep["strong_answers_incomplete"] is False


def test_failure_keeps_report_and_flags_it(monkeypatch):
    async def fake(*a, **k):
        raise llm.LLMError("down")
    monkeypatch.setattr(llm, "chat", fake)
    rep = report()
    run(rep)
    assert rep["strong_answers_incomplete"] is True and rep["competencies"][0]["questions"][0]["score"] == 20
