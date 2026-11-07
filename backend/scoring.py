"""
scoring.py — turns the Reporter LLM's JSON into the final report.

The LLM's scores are FINAL. Python does not blend, cap or adjust any score or rating.
All strict boundaries live in the prompt (prompts.REPORT_SYSTEM), and the model gets the
measured facts it can't reliably count itself (fillers/min, disfluencies per 100 words,
"I" vs "we" share, words per answer).

Python only:
  * validates/cleans the JSON shape (clamps to 0–100, drops invalid question numbers)
  * checks every evidence quote against the verbatim transcript and FLAGS unverified ones
    for display (verified: true/false). It never changes a level or a score.
  * Overall /100 = plain average of the four competency scores (HackerRank-style aggregate).
  * Bands: Weak < 30 <= Moderate < 50 <= Good < 75 <= Excellent.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .prompts import COMPETENCIES, plan_for_prompt

BANDS = [(75, "Excellent"), (50, "Good"), (30, "Moderate"), (0, "Weak")]
LEVEL_LABELS = {3: "Met", 2: "Partially Met", 1: "Not Met", 0: "Not Assessed"}
EVIDENCE_MATCH = 0.85
RATINGS = ("Weak", "Moderate", "Good", "Excellent")
_FILLER = re.compile(r"^(u+m+|u+h+m*|e+r+m*|a+h+|h+m+|m{2,}|m+h+m+|e+h+)$")


def band(score: float | None) -> str | None:
    if score is None:
        return None
    for cutoff, name in BANDS:
        if score >= cutoff:
            return name
    return "Weak"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def real_words(text: str) -> int:
    """Words that carry content: no [breath]/[laughter] tags, no um/uh."""
    text = re.sub(r"\[[^\]]*\]", " ", text or "")
    return sum(1 for w in _words(text) if not _FILLER.match(w))


def disfluency_rate(st: dict[str, Any] | None) -> float | None:
    """Fillers + repetitions + false starts + stammers per 100 words of an answer (a fact for the prompt)."""
    if not st or not st.get("word_count"):
        return None
    d = (st.get("fillers") or {}).get("total", 0) + st.get("repetitions", 0) + st.get("false_starts", 0) + st.get("stammers", 0)
    return round(100.0 * d / max(st["word_count"], 1), 1)


def quote_in_answer(quote: str, answer: str) -> bool:
    """True when (almost) every word of the quote appears, in order, in the answer."""
    q, a = _words(quote), _words(answer)
    if len(q) < 2 or not a:
        return False
    m = SequenceMatcher(None, a, q, autojunk=False)
    matched = sum(b.size for b in m.get_matching_blocks())
    return matched / len(q) >= EVIDENCE_MATCH


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _int_list(v: Any, valid: set[int]) -> list[int]:
    out = []
    for x in v if isinstance(v, list) else [v]:
        n = _int(x)
        if n in valid and n not in out:
            out.append(n)
    return sorted(out)


def _items(v: Any, valid: set[int], limit: int = 3) -> list[dict[str, Any]]:
    out = []
    for it in (v or [])[:limit]:
        if isinstance(it, dict) and (it.get("title") or it.get("detail")):
            out.append({"title": str(it.get("title", "")), "detail": str(it.get("detail", "")),
                        "seen_in": _int_list(it.get("seen_in", []), valid)})
        elif isinstance(it, str) and it.strip():
            out.append({"title": it, "detail": "", "seen_in": []})
    return out


def _score(v: Any) -> int | None:
    try:
        return max(0, min(100, round(float(v))))
    except (TypeError, ValueError):
        return None


def build_report(raw: Any, plan: dict[str, Any], questions: list[dict[str, Any]],
                 session_stats: dict[str, Any]) -> dict[str, Any]:
    """
    raw:        Reporter JSON
    plan:       the rubric the reporter scored against
    questions:  [{"number", "id", "verbatim", "skipped", "section", "stats"}]
    """
    if not isinstance(raw, dict):
        raise ValueError("report is not an object")
    answers = {q["number"]: q.get("verbatim") or "" for q in questions if not q.get("skipped") and q.get("verbatim")}
    valid_numbers = {q["number"] for q in questions}
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
            level = _int(re_.get("level"))
            level = level if level in LEVEL_LABELS else 0
            evidence = []
            for ev in re_.get("evidence") or []:
                if not isinstance(ev, dict):
                    continue
                qn = _int(ev.get("question"))
                quote = str(ev.get("quote", "")).strip().strip('"“”')
                if not quote:
                    continue
                # Flag (never drop or re-score): is the quote really in the cited answer — or any answer?
                verified = bool(qn in answers and quote_in_answer(quote, answers[qn]))
                if not verified:
                    alt = next((n for n, a in answers.items() if quote_in_answer(quote, a)), None)
                    if alt is not None:
                        qn, verified = alt, True
                evidence.append({"question": qn, "quote": quote, "verified": verified})
            expectations.append({"id": pe["id"], "text": pe["text"], "level": level,
                                 "label": LEVEL_LABELS[level], "evidence": evidence[:3]})

        qcards = []
        for qc in rc.get("questions") or []:
            if not isinstance(qc, dict):
                continue
            n = _int(qc.get("question"))
            if n not in answers or n in assigned:
                continue
            assigned.add(n)
            rating = str(qc.get("rating", "")).strip().capitalize()
            qcards.append({"question": n, "rating": rating if rating in RATINGS else "Not rated",
                           "verdict": str(qc.get("verdict", "")), "strong_answer": str(qc.get("strong_answer", ""))})

        score = _score(rc.get("score"))
        fs = session_stats.get("fluency_score")
        competencies.append({
            "key": key, "name": meta["name"], "description": meta["description"],
            "score": score, "band": band(score),
            "delivery_score": round(fs * 10) if (key == "communication" and fs is not None) else None,
            "score_rationale": str(rc.get("score_rationale", "")),
            "assessment": str(rc.get("assessment", "")),
            "expectations": expectations,
            "unverified_quotes": sum(1 for e in expectations for ev in e["evidence"] if not ev["verified"]),
            "improvements": _items(rc.get("improvements"), valid_numbers),
            "strengths": _items(rc.get("strengths"), valid_numbers),
            "questions": sorted(qcards, key=lambda x: x["question"]),
        })

    # Any answered question the model forgot still gets a card (unrated) under its planned section.
    by_key = {c["key"]: c for c in competencies}
    for q in questions:
        if q["number"] in answers and q["number"] not in assigned:
            target = by_key.get(q.get("section") or "", by_key["technical_background"])
            target["questions"].append({"question": q["number"], "rating": "Not rated", "verdict": "", "strong_answer": ""})
            target["questions"].sort(key=lambda x: x["question"])

    scored = [c["score"] for c in competencies if c["score"] is not None]
    overall = round(sum(scored) / len(scored)) if scored else None
    return {
        "version": 4,
        "overall_score": overall,
        "overall_band": band(overall),
        "overall_summary": str(raw.get("overall_summary", "")),
        "doing_well": [str(x) for x in (raw.get("doing_well") or []) if str(x).strip()][:3],
        "improve": [str(x) for x in (raw.get("improve") or []) if str(x).strip()][:3],
        "competencies": competencies,
        "session_stats": session_stats,
    }
