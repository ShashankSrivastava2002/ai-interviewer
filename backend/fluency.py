"""
fluency.py — fluency statistics, computed deterministically in Python.

The LLM never counts anything: it receives these numbers and must use them as-is.

Definitions
-----------
spoken span     first word start -> last word end of an answer (seconds)
speaking time   spoken span minus detected pause time
filler rate     fillers per minute of spoken span
WPM             non-filler words / speaking time (fillers and pauses excluded)
silent %        pause time / spoken span
thinking time   recording start -> first word
"""
from __future__ import annotations

from collections import Counter
from typing import Any

IDEAL_WPM = (120, 160)

SPOKEN_KINDS = ("word", "repetition", "stammer", "false_start")


def _groups(tokens: list[dict[str, Any]], kind: str) -> int:
    """Count items of a kind; multi-token items share a 'group' and count once."""
    seen = set()
    n = 0
    for i, t in enumerate(tokens):
        if t["kind"] != kind:
            continue
        g = t.get("group")
        if g:
            if g in seen:
                continue
            seen.add(g)
        n += 1
    return n


def answer_stats(tokens: list[dict[str, Any]], thinking_time: float | None, audio_duration: float | None) -> dict[str, Any]:
    words = [t for t in tokens if t["kind"] != "pause"]
    pauses = [t for t in tokens if t["kind"] == "pause"]
    fillers = [t for t in tokens if t["kind"] == "filler"]

    if words:
        span = max(0.0, max(t["end"] for t in words) - min(t["start"] for t in words))
    else:
        span = 0.0
    pause_time = sum(p["duration"] for p in pauses)
    speaking = max(span - pause_time, 0.0)
    minutes = span / 60.0
    content_words = sum(1 for t in tokens if t["kind"] in SPOKEN_KINDS)

    filler_counts = Counter(t["norm"] or t["text"].lower() for t in fillers)

    stats: dict[str, Any] = {
        "word_count": content_words,
        "spoken_seconds": round(span, 2),
        "speaking_seconds": round(speaking, 2),
        "audio_seconds": round(audio_duration or 0.0, 2),
        "thinking_time": round(thinking_time or 0.0, 2),
        "fillers": {
            "total": len(fillers),
            "per_minute": round(len(fillers) / minutes, 2) if minutes > 0 else 0.0,
            "counts": dict(filler_counts.most_common()),
        },
        "stammers": _groups(tokens, "stammer"),
        "repetitions": _groups(tokens, "repetition"),
        "false_starts": _groups(tokens, "false_start"),
        "pauses": {
            "count": len(pauses),
            "short": sum(1 for p in pauses if p.get("category") == "short"),
            "long": sum(1 for p in pauses if p.get("category") == "long"),
            "very_long": sum(1 for p in pauses if p.get("category") == "very_long"),
            "longest": round(max((p["duration"] for p in pauses), default=0.0), 2),
            "average": round(pause_time / len(pauses), 2) if pauses else 0.0,
            "total_seconds": round(pause_time, 2),
            "silent_pct": round(100.0 * pause_time / span, 1) if span > 0 else 0.0,
        },
        "wpm": round(content_words / (speaking / 60.0)) if speaking > 1.0 else 0,
    }
    stats["fluency_score"] = fluency_score(stats) if content_words >= 3 else None
    return stats


def fluency_score(s: dict[str, Any]) -> float:
    """
    0–10 score from transparent penalties (shown in the README):
      fillers           -0.35 per filler/min above 2/min        (max -3)
      stammers/reps/FS  -0.5 per event/min                      (max -2.5)
      silence           -0.06 per % silent above 15%            (max -2)
      long pauses       -0.3 per pause >= 1.5 s                 (max -1.5)
      pace              -0.03 per WPM outside 120–160           (max -1.5)
    """
    minutes = max(s["spoken_seconds"] / 60.0, 1 / 60)
    score = 10.0
    score -= min(3.0, max(0.0, s["fillers"]["per_minute"] - 2.0) * 0.35)
    events = (s["stammers"] + s["repetitions"] + s["false_starts"]) / minutes
    score -= min(2.5, events * 0.5)
    score -= min(2.0, max(0.0, s["pauses"]["silent_pct"] - 15.0) * 0.06)
    score -= min(1.5, (s["pauses"]["long"] + s["pauses"]["very_long"]) * 0.3)
    wpm = s["wpm"]
    lo, hi = IDEAL_WPM
    off = (lo - wpm) if wpm < lo else (wpm - hi) if wpm > hi else 0
    score -= min(1.5, off * 0.03)
    return round(max(0.0, min(10.0, score)), 1)


def session_stats(per_answer: list[dict[str, Any]]) -> dict[str, Any]:
    """
    per_answer: [{"question_id", "index", "stats"}] for answered (not skipped) questions.
    """
    answered = [a for a in per_answer if a.get("stats")]
    total_span = sum(a["stats"]["spoken_seconds"] for a in answered)
    total_speaking = sum(a["stats"]["speaking_seconds"] for a in answered)
    total_words = sum(a["stats"]["word_count"] for a in answered)
    filler_counter: Counter = Counter()
    for a in answered:
        filler_counter.update(a["stats"]["fillers"]["counts"])
    fillers_total = sum(filler_counter.values())
    pause_counts = sum(a["stats"]["pauses"]["count"] for a in answered)
    pause_time = sum(a["stats"]["pauses"]["total_seconds"] for a in answered)
    thinking = [a["stats"]["thinking_time"] for a in answered]
    scores = [a["stats"]["fluency_score"] for a in answered if a["stats"].get("fluency_score") is not None]

    return {
        "answers": len(answered),
        "total_spoken_seconds": round(total_span, 1),
        "fillers": {
            "total": fillers_total,
            "per_minute": round(fillers_total / (total_span / 60.0), 2) if total_span > 0 else 0.0,
            "top": [{"word": w, "count": c} for w, c in filler_counter.most_common(6)],
        },
        "stammers": sum(a["stats"]["stammers"] for a in answered),
        "repetitions": sum(a["stats"]["repetitions"] for a in answered),
        "false_starts": sum(a["stats"]["false_starts"] for a in answered),
        "pauses": {
            "count": pause_counts,
            "longest": round(max((a["stats"]["pauses"]["longest"] for a in answered), default=0.0), 2),
            "average": round(pause_time / pause_counts, 2) if pause_counts else 0.0,
            "silent_pct": round(100.0 * pause_time / total_span, 1) if total_span > 0 else 0.0,
        },
        "avg_thinking_time": round(sum(thinking) / len(thinking), 2) if thinking else 0.0,
        "wpm": round(total_words / (total_speaking / 60.0)) if total_speaking > 1 else 0,
        "ideal_wpm": list(IDEAL_WPM),
        "fluency_score": round(sum(scores) / len(scores), 1) if scores else None,
        "trend": [
            {"question_id": a["question_id"], "index": a["index"], "score": a["stats"].get("fluency_score")}
            for a in answered
        ],
    }
