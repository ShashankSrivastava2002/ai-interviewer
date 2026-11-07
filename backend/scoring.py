"""
scoring.py — turns the Reporter LLM's raw JSON into the final report. Every number is decided
here, in Python, with strict rules so a weak interview can never look "Moderate" or "Good".

1. Evidence check
   Every quote must really occur in the candidate's verbatim answer (fuzzy word match >= 85%).
   Unverifiable quotes are dropped; an expectation left with no verified evidence has level 0.

2. Substance gates (the LLM can't give credit to answers that say almost nothing)
   Content words = spoken words excluding fillers, pauses and non-speech tags.
     Met (3)            needs a cited answer with >= 40 content words
     Partially met (2)  needs a cited answer with >= 20 content words
   Otherwise the level is lowered to what the evidence can support.
   Question ratings: < 15 content words -> Weak at most, < 35 -> Moderate at most.

3. Points per expectation
     Met 100 · Partially met 45 · Not met 10 · No evidence 0
   "No evidence" counts as 0: if you didn't demonstrate it, you get no credit for it.

4. Competency score /100 = average points of its expectations.
   Communication & Clarity = rubric × (0.5 + 0.5 × delivery/100), delivery measured in fluency.py.
   Fluent delivery keeps the rubric score; heavy fillers/pauses can halve it. Delivery can never ADD points,
   so fluent-but-empty or off-topic answers stay low.
   A competency the interview never reached scores 0 and is labelled "Not reached".

5. Overall /100 = plain average of all four competencies.
   Bands: Weak < 30 <= Moderate < 50 <= Good < 75 <= Excellent.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .prompts import COMPETENCIES, plan_for_prompt

BANDS = [(75, "Excellent"), (50, "Good"), (30, "Moderate"), (0, "Weak")]
LEVEL_LABELS = {3: "Met", 2: "Partially Met", 1: "Not Met", 0: "No evidence"}
NOT_PROBED = "Not probed"
LEVEL_POINTS = {3: 100, 2: 45, 1: 10, 0: 0}
MIN_WORDS_FOR_LEVEL = {3: 40, 2: 20}
RATING_CAPS = [(15, "Weak"), (35, "Moderate")]          # content words -> best possible rating
RATING_ORDER = ["Weak", "Moderate", "Good", "Excellent"]
COMMUNICATION_DELIVERY_WEIGHT = 0.5   # max share of the rubric that poor delivery can remove
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
    return v if v in RATING_ORDER else "Weak"


def cap_rating(rating: str, content_words: int) -> str:
    for limit, best in RATING_CAPS:
        if content_words < limit and RATING_ORDER.index(rating) > RATING_ORDER.index(best):
            return best
    return rating


def build_report(raw: Any, plan: dict[str, Any], questions: list[dict[str, Any]],
                 session_stats: dict[str, Any]) -> dict[str, Any]:
    """
    raw:        Reporter JSON
    plan:       the session's plan
    questions:  [{"number", "id", "verbatim", "skipped", "section", "content_words"}]
    """
    if not isinstance(raw, dict):
        raise ValueError("report is not an object")
    answers = {q["number"]: q.get("verbatim") or "" for q in questions if not q.get("skipped") and q.get("verbatim")}
    words_in = {q["number"]: int(q.get("content_words") or len(_words(q.get("verbatim") or ""))) for q in questions}
    valid_numbers = {q["number"] for q in questions}
    answered_numbers = set(answers)
    asked_sections = {q.get("section") for q in questions if q.get("section")}
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
            note = ""
            if level > 0 and not evidence:
                level, note = 0, "no verifiable quote"
            # Substance gate: short answers can't support Met / Partially met.
            best_words = max((words_in.get(e["question"], 0) for e in evidence), default=0)
            while level >= 2 and best_words < MIN_WORDS_FOR_LEVEL[level]:
                level -= 1
                note = f"lowered: the cited answer has only {best_words} content words"
            # Never asked about -> left out of the score (it's the interview's gap, not the candidate's).
            # Missing flag defaults to probed (strict).
            counted = not (level == 0 and re_.get("probed") is False)
            expectations.append({"id": pe["id"], "text": pe["text"], "level": level, "counted": counted,
                                 "label": LEVEL_LABELS[level] if counted else NOT_PROBED,
                                 "points": LEVEL_POINTS[level] if counted else None,
                                 "evidence": evidence[:3], "unverified_quotes_dropped": dropped, "note": note})

        reached = key in asked_sections or any(e["level"] > 0 for e in expectations)
        counted = [e["points"] for e in expectations if e["counted"]]
        rubric = round(sum(counted) / len(counted)) if counted else 0
        if not reached:
            rubric = 0
        score = rubric
        delivery = None
        if key == "communication":
            fs = session_stats.get("fluency_score")
            delivery = round(fs * 10) if fs is not None else 0
            score = round(rubric * (1 - COMMUNICATION_DELIVERY_WEIGHT + COMMUNICATION_DELIVERY_WEIGHT * delivery / 100))
            reached = reached or bool(answered_numbers)

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
            qcards.append({"question": n, "rating": cap_rating(_rating(qc.get("rating")), words_in.get(n, 0)),
                           "verdict": str(qc.get("verdict", "")), "strong_answer": str(qc.get("strong_answer", ""))})

        competencies.append({
            "key": key, "name": meta["name"], "description": meta["description"],
            "score": score, "band": band(score) if reached else "Not reached", "reached": reached,
            "rubric_score": rubric, "delivery_score": delivery,
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
            target["questions"].append({"question": q["number"], "rating": cap_rating("Moderate", words_in.get(q["number"], 0)),
                                        "verdict": "", "strong_answer": ""})
            target["questions"].sort(key=lambda x: x["question"])

    overall = round(sum(c["score"] for c in competencies) / len(competencies)) if competencies else 0
    answered = len(answered_numbers)
    substantive = sum(1 for n in answered_numbers if words_in.get(n, 0) >= 20)
    return {
        "version": 2,
        "scoring": "strict-v2",
        "overall_score": overall,
        "overall_band": band(overall),
        "overall_summary": str(raw.get("overall_summary", "")),
        "doing_well": [str(x) for x in (raw.get("doing_well") or [])][:3],
        "improve": [str(x) for x in (raw.get("improve") or [])][:3],
        "coverage": {"questions_asked": len(questions), "answered": answered, "substantive_answers": substantive,
                     "sections_reached": sum(1 for c in competencies if c["reached"])},
        "competencies": competencies,
        "session_stats": session_stats,
    }
