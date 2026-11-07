"""
scoring.py — turns the Reporter LLM's raw JSON into the final, trustworthy report.

Everything numeric is decided here in Python, not by the LLM:

  * Evidence check: every quote must really occur in the candidate's verbatim answer
    (fuzzy word match >= 85%). Unverifiable quotes are dropped; an expectation left with no
    verified evidence becomes 0 = Not Assessed (the model may not guess).
  * Competency score /100 = mean of assessed expectation levels (1–3) / 3 * 100.
    Communication & Clarity blends the rubric with the measured delivery score:
        70% rubric + 30% Python fluency score (0–10 -> 0–100).
  * Overall /100 = plain average of the four competency scores.
  * Bands: Weak < 30 <= Moderate < 50 <= Good < 75 <= Excellent.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .prompts import COMPETENCIES, COMPETENCY_KEYS, plan_for_prompt

BANDS = [(75, "Excellent"), (50, "Good"), (30, "Moderate"), (0, "Weak")]
LEVEL_LABELS = {3: "Met", 2: "Partially Met", 1: "Not Met", 0: "Not Assessed"}
COMMUNICATION_DELIVERY_WEIGHT = 0.3
EVIDENCE_MATCH = 0.85


def band(score: float | None) -> str | None:
    if score is None:
        return None
    for cutoff, name in BANDS:
        if score >= cutoff:
            return name
    return "Weak"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def quote_in_answer(quote: str, answer: str) -> bool:
    """True when (almost) every word of the quote appears, in order, in the answer."""
    q, a = _words(quote), _words(answer)
    if len(q) < 2 or not a:
        return False
    m = SequenceMatcher(None, a, q, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    return matched / len(q) >= EVIDENCE_MATCH


def _int_list(v: Any, valid: set[int]) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        if n in valid and n not in out:
            out.append(n)
    return sorted(out)


def _items(v: Any, valid: set[int], limit: int = 3) -> list[dict[str, Any]]:
    out = []
    for it in (v or [])[:limit]:
        if isinstance(it, dict) and (it.get("title") or it.get("detail")):
            out.append({"title": str(it.get("title", "")), "detail": str(it.get("detail", "")),
                        "seen_in": _int_list(it.get("seen_in", []), valid)})
        elif isinstance(it, str):
            out.append({"title": it, "detail": "", "seen_in": []})
    return out


def _rating(v: Any) -> str:
    v = str(v or "").strip().capitalize()
    return v if v in ("Weak", "Moderate", "Good", "Excellent") else "Moderate"


def build_report(raw: Any, plan: dict[str, Any], questions: list[dict[str, Any]],
                 session_stats: dict[str, Any]) -> dict[str, Any]:
    """
    raw:        Reporter JSON
    plan:       the session's plan
    questions:  [{"number", "id", "verbatim", "skipped"}]
    """
    if not isinstance(raw, dict):
        raise ValueError("report is not an object")
    answers = {q["number"]: q.get("verbatim") or "" for q in questions if not q.get("skipped") and q.get("verbatim")}
    valid_numbers = {q["number"] for q in questions}
    answered_numbers = set(answers)
    raw_comps = {c.get("key"): c for c in raw.get("competencies") or [] if isinstance(c, dict)}
    plan_sections = {s["key"]: s for s in plan_for_prompt(plan)}

    competencies = []
    assigned: set[int] = set()
    for meta in COMPETENCIES:
        key = meta["key"]
        rc = raw_comps.get(key, {})
        raw_exps = {str(e.get("id")): e for e in rc.get("expectations") or [] if isinstance(e, dict)}

        expectations = []
        for i, pe in enumerate(plan_sections.get(key, {}).get("expectations", [])):
            re_ = raw_exps.get(pe["id"]) or (list(raw_exps.values())[i] if i < len(raw_exps) else {})
            try:
                level = max(0, min(3, int(re_.get("level", 0))))
            except (TypeError, ValueError):
                level = 0
            evidence, dropped = [], 0
            for ev in re_.get("evidence") or []:
                if not isinstance(ev, dict):
                    continue
                try:
                    qn = int(ev.get("question"))
                except (TypeError, ValueError):
                    qn = -1
                quote = str(ev.get("quote", "")).strip().strip('"“”')
                # Accept the quote if it's in the cited answer — or in any answer (wrong number cited).
                hit = qn if qn in answers and quote_in_answer(quote, answers[qn]) else next(
                    (n for n, a in answers.items() if quote_in_answer(quote, a)), None)
                if hit is not None:
                    evidence.append({"question": hit, "quote": quote})
                else:
                    dropped += 1
            if level > 0 and not evidence:
                level = 0  # no verifiable transcript moment -> Not Assessed
            expectations.append({"id": pe["id"], "text": pe["text"], "level": level,
                                 "label": LEVEL_LABELS[level], "evidence": evidence[:3],
                                 "unverified_quotes_dropped": dropped})

        assessed = [e["level"] for e in expectations if e["level"] > 0]
        rubric = round(100 * sum(assessed) / (3 * len(assessed))) if assessed else None
        score = rubric
        delivery = None
        if key == "communication":
            fs = session_stats.get("fluency_score")
            delivery = round(fs * 10) if fs is not None else None
            if rubric is not None and delivery is not None:
                score = round((1 - COMMUNICATION_DELIVERY_WEIGHT) * rubric + COMMUNICATION_DELIVERY_WEIGHT * delivery)
            elif rubric is None:
                score = delivery

        qcards = []
        for qc in rc.get("questions") or []:
            if not isinstance(qc, dict):
                continue
            try:
                n = int(qc.get("question"))
            except (TypeError, ValueError):
                continue
            if n not in answered_numbers or n in assigned:
                continue
            assigned.add(n)
            qcards.append({"question": n, "rating": _rating(qc.get("rating")),
                           "verdict": str(qc.get("verdict", "")), "strong_answer": str(qc.get("strong_answer", ""))})

        competencies.append({
            "key": key, "name": meta["name"], "description": meta["description"],
            "score": score, "band": band(score), "rubric_score": rubric, "delivery_score": delivery,
            "assessment": str(rc.get("assessment", "")),
            "expectations": expectations,
            "improvements": _items(rc.get("improvements"), valid_numbers),
            "strengths": _items(rc.get("strengths"), valid_numbers),
            "questions": sorted(qcards, key=lambda x: x["question"]),
        })

    # Any answered question the model forgot gets a card under its planned section.
    by_key = {c["key"]: c for c in competencies}
    for q in questions:
        if q["number"] in answered_numbers and q["number"] not in assigned:
            target = by_key.get(q.get("section") or "", by_key["technical_background"])
            target["questions"].append({"question": q["number"], "rating": "Moderate", "verdict": "", "strong_answer": ""})
            target["questions"].sort(key=lambda x: x["question"])

    scored = [c["score"] for c in competencies if c["score"] is not None]
    overall = round(sum(scored) / len(scored)) if scored else None
    return {
        "version": 2,
        "overall_score": overall,
        "overall_band": band(overall),
        "overall_summary": str(raw.get("overall_summary", "")),
        "doing_well": [str(x) for x in (raw.get("doing_well") or [])][:3],
        "improve": [str(x) for x in (raw.get("improve") or [])][:3],
        "competencies": competencies,
        "session_stats": session_stats,
    }
