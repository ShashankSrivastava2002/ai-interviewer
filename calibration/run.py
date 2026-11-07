"""
Calibration check: is the report model's score correct?

Runs each case through the exact same scoring path as the app (prompts.REPORT_SYSTEM + the fixed
rubric + REPORT_LLM_MODEL) several times and compares the overall score with a target range.

    .venv/bin/python calibration/run.py                 # text cases only
    .venv/bin/python calibration/run.py --db-session 1  # also score a saved interview from data/interviews.db
    .venv/bin/python calibration/run.py --runs 1        # quicker

Cases (add your own as transcripts in the same "You/Interviewer" format):
  hackerrank_screen.txt   real HackerRank screen; HackerRank scored it 45/63/55/32 -> 49 Moderate
  strong_candidate.txt    crisp, headline-first, "I"-owned, quantified answers -> should be 75+
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from backend import db, disfluency, fluency, llm, prompts, scoring  # noqa: E402
from backend.config import get_settings  # noqa: E402

KEYS = ["role_alignment", "technical_background", "impact_ownership", "communication"]
JD = "AI Engineer role focused on building and deploying production-grade Generative AI and agentic systems."

# name, source, overall target range (inclusive), per-competency reference (optional)
TEXT_CASES = [
    ("HackerRank real screen", HERE / "hackerrank_screen.txt", (42, 56),
     {"role_alignment": 45, "technical_background": 63, "impact_ownership": 55, "communication": 32}),
    ("Strong crisp candidate", HERE / "strong_candidate.txt", (75, 100), None),
]
DB_TARGET = (0, 25)   # the saved "bad interview"


def from_text(path: Path) -> list[dict]:
    """Parse a You/Interviewer transcript into numbered Q&A with verbatim disfluency stats."""
    raw = path.read_text()
    turns = re.findall(r"^(You|Interviewer)\n(\d\d:\d\d)\n(.*?)(?=\n\n(?:You|Interviewer)\n\d\d:\d\d\n|\Z)", raw, re.S | re.M)
    qa: list[dict] = []
    for who, _, text in turns:
        text = text.strip()
        if who == "Interviewer":
            if len(text) < 20 and qa:  # "Sure" / "Okay" is not a question
                continue
            qa.append({"q": text, "a": []})
        elif qa:
            qa[-1]["a"].append(text)
    out = []
    for n, it in enumerate(qa, 1):
        words, t = [], 1.0
        for w in " ".join(it["a"]).split():  # no audio: ~150 wpm timing, no pause data
            words.append({"word": w, "start": t, "end": t + 0.3, "confidence": 0.95})
            t += 0.4
        toks = disfluency.build_timeline(words, [])
        out.append({"id": n, "number": n, "kind": "question", "section": None, "text": it["q"], "skipped": False,
                     "verbatim": disfluency.to_plain_verbatim(toks) if words else None,
                     "stats": fluency.answer_stats(toks, 1.0, t) if words else None})
    return out


def from_db(session_id: int) -> list[dict]:
    db.init_db()
    with db.get_db() as dbs:
        qs, ans = db.questions_for(dbs, session_id), db.answers_for(dbs, session_id)
        out = []
        for q in qs:
            a = ans.get(q.id)
            st = a.stats if a and not a.skipped else None
            if st and "pronouns" not in st:  # stats saved before newer facts existed
                st = fluency.answer_stats(a.tokens, a.thinking_time, a.duration)
            out.append({"id": q.id, "number": q.position + 1, "kind": q.kind, "section": q.section, "text": q.text,
                        "skipped": bool(a and a.skipped),
                        "verbatim": disfluency.to_plain_verbatim(a.tokens) if a and not a.skipped else None,
                        "stats": st})
        return out


async def score_once(questions: list[dict], model: str) -> tuple[float, dict]:
    per = [{"question_id": q["id"], "index": q["number"], "stats": q["stats"]} for q in questions if q["stats"]]
    stats = fluency.session_stats(per)
    setup = {"role": "AI Engineer", "level": "Mid", "interview_type": "technical", "minutes": 30}
    plan = prompts.standard_plan()
    t0 = time.time()
    raw = await llm.chat(prompts.REPORT_SYSTEM,
                         [{"role": "user", "content": prompts.report_user_message(setup, plan, questions, stats, {"jd": JD})}],
                         json_mode=True, max_tokens=60000, model=model, timeout=600,
                         temperature=None if model.startswith("gemini-3") else 0)
    return time.time() - t0, scoring.build_report(raw, plan, questions, stats)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--db-session", type=int, help="also score this saved interview (target 0–25)")
    args = ap.parse_args()

    model = get_settings().resolved_report_model
    cases = [(n, from_text(p), tgt, ref) for n, p, tgt, ref in TEXT_CASES]
    if args.db_session:
        cases.append((f"Saved interview #{args.db_session}", from_db(args.db_session), DB_TARGET, None))
    print(f"Report model: {model} · {args.runs} run(s) per case\n")

    failures, results = 0, {}
    for name, questions, (lo, hi), ref in cases:
        runs = await asyncio.gather(*(score_once(questions, model) for _ in range(args.runs)), return_exceptions=True)
        print(f"### {name}   target overall {lo}–{hi}")
        print(f"{'':12}" + "".join(f"{k[:13]:>15}" for k in KEYS) + f"{'overall':>9}")
        if ref:
            print(f"{'reference':12}" + "".join(f"{ref[k]:>15}" for k in KEYS))
        ok_reports = []
        for i, r in enumerate(runs, 1):
            if isinstance(r, Exception):
                print(f"run {i}: ERROR {r}")
                failures += 1
                continue
            sec, rep = r
            ok_reports.append(rep)
            c = {x["key"]: x for x in rep["competencies"]}
            overall = rep["overall_score"]
            verdict = "PASS" if overall is not None and lo <= overall <= hi else "FAIL"
            failures += verdict == "FAIL"
            print(f"run {i} {sec:4.0f}s" + "".join(f"{str(c[k]['score']):>15}" for k in KEYS) + f"{str(overall):>9}  {verdict}")
        if ok_reports:
            for x in ok_reports[0]["competencies"]:
                print(f"   · {x['name']} {x['score']}: {x['score_rationale'][:220]}")
        results[name] = ok_reports
        print()

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": model, "results": results}, indent=1))
    print(f"{'ALL PASS' if not failures else f'{failures} run(s) outside target'} · details: {out.relative_to(HERE.parent)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
