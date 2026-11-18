"""
strong_answers.py — second report call: rewrites each answer from the original answer + the feedback.

Runs after scoring and never changes a score. If it fails, the scored report is still kept.
"""
from __future__ import annotations

import logging
from typing import Any

from . import llm, prompts

log = logging.getLogger("strong_answers")


async def add_strong_answers(report: dict[str, Any], setup: dict[str, Any], questions: list[dict[str, Any]],
                             ctx: dict[str, str | None] | None, model: str, thinking: str | None) -> None:
    """Fills each question card's "strong_answer" in place and flags the report if any are missing."""
    cards = {c["question"]: c for comp in report["competencies"] for c in comp["questions"]}
    if not cards:
        report["strong_answers_incomplete"] = False
        return
    try:
        raw = await llm.chat(prompts.STRONG_ANSWER_SYSTEM,
                             [{"role": "user", "content": prompts.strong_answer_user_message(setup, report, questions, ctx)}],
                             json_mode=True, max_tokens=min(40000, 12000 + 1000 * len(cards)), model=model,
                             temperature=None if model.startswith("gemini-3") else 0.2, thinking=thinking,
                             timeout=180 + 8 * len(cards))
        items = raw.get("answers") if isinstance(raw, dict) else None
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                card = cards.get(int(item.get("question")))
            except (TypeError, ValueError):
                continue
            text = str(item.get("strong_answer") or "").strip()
            if card and text:
                card["strong_answer"] = text
    except llm.LLMError as e:
        log.warning("event=strong_answers_failed error=%s", e)
    missing = sum(1 for c in cards.values() if not c.get("strong_answer"))
    report["strong_answers_incomplete"] = missing > 0
    log.info("event=strong_answers written=%d missing=%d", len(cards) - missing, missing)
