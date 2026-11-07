"""
Evaluator harness — runs scripted interviews of known quality through the REAL pipeline:

  verbatim words -> pause detection -> disfluency tagging -> delivery stats (Python)
  -> Planner LLM (real) -> Reporter LLM (real) -> evidence checks + strict scoring (Python)

Only speech-to-text is replaced (the scripted words are fed in with synthetic timings), and the
questions are scripted so every case is asked the same thing. Uses the LLM settings in backend/.env,
so it makes real API calls (2 per case).

Run from the project root:
    .venv/bin/python -m evals.run_evals            # all cases + repeats
    .venv/bin/python -m evals.run_evals A_excellent E_non_answers
Results are saved to evals/results/<timestamp>.json
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

# Isolated data dir; skip loading the local speech model (we feed words directly).
_TMP = tempfile.mkdtemp(prefix="evals-")
os.environ.update(DATA_DIR=_TMP, STT_PROVIDER="deepgram", DEEPGRAM_API_KEY="unused", LOG_LEVEL="WARNING")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import audio, db, main  # noqa: E402
from evals.cases import CASES, JD, NOTES, QUESTIONS, REPEATS, RESUME, ROLE  # noqa: E402

SEC_PER_WORD = 0.36  # ~165 words per minute
_current: dict = {}


def words_from(text: str) -> list[dict]:
    """Scripted answer -> STT-style words with timings. '[p2.5]' inserts a 2.5 s pause."""
    out, t = [], 0.9  # 0.9 s thinking time
    for tok in text.split():
        m = re.fullmatch(r"\[p([\d.]+)\]", tok)
        if m:
            t += float(m.group(1))
            continue
        out.append({"word": tok, "start": round(t, 2), "end": round(t + 0.28, 2), "confidence": 0.95})
        t += SEC_PER_WORD
    return out


async def fake_convert(src: Path, dst: Path) -> Path:
    end = (_current["words"][-1]["end"] if _current["words"] else 0) + 0.6
    with wave.open(str(dst), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.SAMPLE_RATE)
        wf.writeframes(np.zeros(int(end * audio.SAMPLE_RATE), dtype=np.int16).tobytes())
    return dst


async def fake_stt(path: Path) -> list[dict]:
    return list(_current["words"])


audio.convert_to_wav = fake_convert
main.stt_service.transcribe = fake_stt


def run_case(client: TestClient, case: dict) -> dict:
    t0 = time.perf_counter()
    with db.get_db() as dbs:
        sess = db.InterviewSession(role=ROLE, level="Mid", interview_type="technical", duration_minutes=30,
                                   num_questions=len(QUESTIONS), resume_text=RESUME, jd_text=JD, notes_text=NOTES,
                                   scripted_questions=json.dumps(QUESTIONS))
        dbs.add(sess)
        dbs.commit()
        dbs.refresh(sess)
        sess.plan_json = json.dumps(asyncio.run(main.make_plan(sess)))   # real Planner LLM
        dbs.add(sess)
        dbs.commit()
        first = asyncio.run(main.ask_llm_next(dbs, sess))                # scripted: no LLM
        sid, qid = sess.id, first.question.id

    for ans in case["answers"]:
        _current["words"] = words_from(ans)
        r = client.post(f"/api/sessions/{sid}/answer", data={"question_id": qid},
                        files={"audio": ("a.webm", b"x" * 1000, "audio/webm")})
        r.raise_for_status()
        nxt = r.json()["next"]
        if nxt["type"] == "end":
            break
        qid = nxt["question"]["id"]

    r = client.post(f"/api/sessions/{sid}/feedback")                    # real Reporter LLM + strict scoring
    if r.status_code != 200:
        return {"id": case["id"], "error": r.text[:300]}
    rep = r.json()["report"]
    comps = {c["key"]: c for c in rep["competencies"]}
    lowered = sum(1 for c in rep["competencies"] for e in c["expectations"] if e.get("note", "").startswith("lowered"))
    dropped = sum(e.get("unverified_quotes_dropped", 0) for c in rep["competencies"] for e in c["expectations"])
    ratings = [q["rating"] for c in rep["competencies"] for q in c["questions"]]
    return {
        "id": case["id"], "label": case["label"], "seconds": round(time.perf_counter() - t0, 1),
        "overall": rep["overall_score"], "band": rep["overall_band"],
        "scores": {k: comps[k]["score"] for k in comps},
        "delivery": comps["communication"].get("delivery_score"),
        "levels": {k: [e["level"] for e in comps[k]["expectations"]] for k in comps},
        "ratings": ratings, "lowered_by_gate": lowered, "quotes_dropped": dropped,
        "doing_well": rep["doing_well"], "summary": rep["overall_summary"],
    }


def check(res: dict, case: dict, by_id: dict) -> list[str]:
    """Returns a list of failures (empty = pass)."""
    if "error" in res:
        return [f"error: {res['error']}"]
    fails = []
    lo, hi = case["expect"]["overall"]
    if not lo <= res["overall"] <= hi:
        fails.append(f"overall {res['overall']} not in [{lo}, {hi}]")
    tb = case["expect"].get("technical_below")
    if tb is not None and res["scores"]["technical_background"] >= tb:
        fails.append(f"technical {res['scores']['technical_background']} should be < {tb}")
    ref = case["expect"].get("communication_below")
    if ref and ref in by_id and "error" not in by_id[ref]:
        if res["scores"]["communication"] >= by_id[ref]["scores"]["communication"]:
            fails.append(f"communication {res['scores']['communication']} should be below {ref}'s "
                         f"{by_id[ref]['scores']['communication']}")
    return fails


def main_cli() -> None:
    only = set(sys.argv[1:])
    cases = [c for c in CASES if not only or c["id"] in only]
    db.init_db()
    results, by_id = [], {}
    with TestClient(main.app) as client:
        for case in cases:
            print(f"… {case['id']}", flush=True)
            res = run_case(client, case)
            results.append(res)
            by_id.setdefault(case["id"], res)
        for rid in ([] if only else REPEATS):
            case = next(c for c in CASES if c["id"] == rid)
            print(f"… {rid} (repeat)", flush=True)
            res = run_case(client, case)
            res["repeat_of"] = rid
            results.append(res)

    print("\n" + "=" * 112)
    print(f"{'case':28} {'overall':>8} {'band':10} {'role':>5} {'tech':>5} {'impact':>6} {'comm':>5} {'deliv':>5} "
          f"{'gated':>5} {'dropQ':>5}  result")
    print("-" * 112)
    passed = 0
    for res in results:
        case = next(c for c in CASES if c["id"] == res["id"])
        if "error" in res:
            print(f"{res['id']:28} ERROR {res['error'][:70]}")
            continue
        fails = [] if res.get("repeat_of") else check(res, case, by_id)
        if not res.get("repeat_of"):
            passed += not fails
        s = res["scores"]
        name = res["id"] + (" (repeat)" if res.get("repeat_of") else "")
        print(f"{name:28} {res['overall']:>8} {res['band']:10} {s['role_alignment']:>5} {s['technical_background']:>5} "
              f"{s['impact_ownership']:>6} {s['communication']:>5} {res['delivery'] if res['delivery'] is not None else '-':>5} "
              f"{res['lowered_by_gate']:>5} {res['quotes_dropped']:>5}  {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    for rid in ([] if only else REPEATS):
        a = by_id.get(rid)
        b = next((r for r in results if r.get("repeat_of") == rid), None)
        if a and b and "error" not in a and "error" not in b:
            print(f"stability {rid}: {a['overall']} vs {b['overall']} (Δ {abs(a['overall'] - b['overall'])})")
    print(f"\n{passed}/{len(cases)} cases within the expected range")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"details: {path}")


if __name__ == "__main__":
    main_cli()
