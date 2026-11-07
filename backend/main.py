"""
main.py — FastAPI app: API routes + serves the frontend at "/".

Flow (modelled on a recruiter-style AI screen):
  1. POST /api/sessions          Planner builds a plan (4 competency sections + expectations),
                                 Interviewer asks the opening question
  2. POST /api/sessions/{id}/start   the candidate starts the clock (time-boxed, default 30 min)
  3. POST /api/sessions/{id}/answer  verbatim STT -> pauses -> disfluencies -> stats -> next step
                                     (follow-up up to 2 per main question / next question / end)
  4. POST /api/sessions/{id}/feedback Reporter scores every expectation 0–3 with verbatim evidence;
                                     scoring.py turns that into /100 scores and bands

Run from the project root:
    uvicorn backend.main:app --reload
then open http://localhost:8000
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audio, db, disfluency, fluency, llm, prompts, scoring
from .config import PROJECT_DIR, get_settings, setup_logging
from .stt import STTError, STTService

settings = get_settings()
setup_logging(settings.log_level)
log = logging.getLogger("api")

FRONTEND_DIR = PROJECT_DIR / "frontend"
CLOSING_TEXT = "That's the end of our interview. Thank you for your time — I'm preparing your feedback now."
MAX_FOLLOWUPS = 2           # per main question
RECORDING_GAP_S = 0.8       # silence between answers in the full-session recording


# ---------------------------------------------------------------------------
# Lifespan: DB + speech model loaded once at startup
# ---------------------------------------------------------------------------

stt_service = STTService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    await stt_service.start()
    log.info("event=startup stt=%s llm=%s model=%s ffmpeg=%s",
             settings.stt_provider, settings.llm_provider, settings.resolved_llm_model, audio.ffmpeg_available())
    yield


app = FastAPI(title="AI Voice Interviewer", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        log.info("event=request method=%s path=%s status=%s ms=%.0f",
                 request.method, request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.exception_handler(llm.LLMError)
async def _llm_err(_: Request, e: llm.LLMError):
    return JSONResponse(status_code=502, content={"detail": str(e)})


@app.exception_handler(STTError)
async def _stt_err(_: Request, e: STTError):
    return JSONResponse(status_code=502, content={"detail": str(e)})


@app.exception_handler(audio.AudioError)
async def _audio_err(_: Request, e: audio.AudioError):
    return JSONResponse(status_code=422, content={"detail": str(e)})


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionCreate(BaseModel):
    role: str = Field(default="Software Engineer", max_length=120)
    level: Literal["Fresher", "Mid", "Senior"] = "Mid"
    interview_type: Literal["technical", "behavioural", "mixed"] = "mixed"
    duration_minutes: int = Field(default=30, ge=5, le=90)
    resume_text: Optional[str] = Field(default=None, max_length=20000)
    jd_text: Optional[str] = Field(default=None, max_length=20000)
    notes_text: Optional[str] = Field(default=None, max_length=10000)
    retry_of: Optional[int] = Field(default=None, description="Re-run the main questions of this session")


class QuestionOut(BaseModel):
    id: int
    number: int                      # "Question N" — follow-ups are numbered too
    text: str
    kind: Literal["question", "followup"]
    main_index: int
    section: Optional[str] = None
    total: int                       # max main questions for this time box


class SessionCreated(BaseModel):
    session_id: int
    duration_minutes: int
    sections: list[dict[str, str]]
    question: QuestionOut


class StartResult(BaseModel):
    started_at: str
    duration_minutes: int


class NextStep(BaseModel):
    type: Literal["question", "followup", "end"]
    text: str
    question: Optional[QuestionOut] = None
    elapsed_seconds: float = 0


class AnswerResult(BaseModel):
    question_id: int
    skipped: bool
    verbatim_tokens: list[dict[str, Any]]
    stats: dict[str, Any]
    next: NextStep


class AnswerOut(BaseModel):
    skipped: bool
    duration: float
    thinking_time: float
    tokens: list[dict[str, Any]]
    stats: dict[str, Any]
    has_audio: bool


class QuestionDetail(BaseModel):
    id: int
    number: int
    position: int
    main_index: int
    kind: str
    section: Optional[str] = None
    text: str
    answer: Optional[AnswerOut] = None


class SessionDetail(BaseModel):
    id: int
    created_at: str
    started_at: Optional[str] = None
    role: str
    level: str
    interview_type: str
    duration_minutes: int
    session_seconds: float
    status: str
    plan: Optional[dict[str, Any]] = None
    questions: list[QuestionDetail]
    recording: Optional[dict[str, Any]] = None
    report: Optional[dict[str, Any]] = None


class SessionSummary(BaseModel):
    id: int
    created_at: str
    role: str
    level: str
    interview_type: str
    duration_minutes: int
    status: str
    overall_score: Optional[float] = None     # /100 (v2 reports); older reports were /10
    fluency_score: Optional[float] = None     # /10, computed from audio
    report_version: Optional[int] = None


class MicCheckResult(BaseModel):
    text: str
    tokens: list[dict[str, Any]]
    filler_detected: bool
    fillers: list[str]
    provider: str


class Health(BaseModel):
    ok: bool
    ffmpeg: bool
    stt: dict[str, Any]
    llm: dict[str, Any]
    missing: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clean(text: Optional[str]) -> Optional[str]:
    return text.strip() or None if text else None


def max_main_questions(minutes: int) -> int:
    """Upper bound on main questions; follow-ups come on top. ~1 main question per 3.5 min."""
    return max(3, round(minutes / 3.5))


def q_out(q: db.Question, sess: db.InterviewSession) -> QuestionOut:
    return QuestionOut(id=q.id, number=q.position + 1, text=q.text, kind=q.kind, main_index=q.main_index,
                       section=q.section, total=sess.num_questions)


def load_session(dbs, session_id: int) -> db.InterviewSession:
    sess = dbs.get(db.InterviewSession, session_id)
    if not sess:
        raise HTTPException(404, "Interview session not found.")
    return sess


def ctx_of(sess: db.InterviewSession) -> dict[str, Optional[str]]:
    """Resume, job description and misc notes for the prompts."""
    return {"resume": sess.resume_text, "jd": sess.jd_text, "notes": sess.notes_text}


def session_plan(sess: db.InterviewSession) -> Optional[dict[str, Any]]:
    """The stored plan, always including the standard recruiter expectations."""
    return prompts.full_plan(json.loads(sess.plan_json)) if sess.plan_json else None


def elapsed_seconds(sess: db.InterviewSession, qs: list[db.Question]) -> float:
    start = sess.started_at or (qs[0].created_at if qs else None)
    if start is None:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - aware(start)).total_seconds())


def _wav_path(a: db.Answer) -> Optional[Path]:
    if not a.audio_path:
        return None
    p = Path(a.audio_path)
    w = p.with_name(f"{p.stem}.16k.wav")
    return w if w.exists() else None


def recording_chapters(sess, qs, answers) -> list[dict[str, Any]]:
    """Start offset of each answer inside the full-session recording."""
    chapters, t = [], 0.0
    for q in qs:
        a = answers.get(q.id)
        w = _wav_path(a) if a and not a.skipped else None
        if not w:
            continue
        chapters.append({"question": q.position + 1, "question_id": q.id, "start": round(t, 2)})
        t += audio.wav_duration(w) + RECORDING_GAP_S
    return chapters


async def build_recording(sess, qs, answers) -> Optional[Path]:
    """Concatenate every answer into one file (data/audio/{id}/session.m4a) for the report player."""
    parts = []
    for q in qs:
        a = answers.get(q.id)
        w = _wav_path(a) if a and not a.skipped else None
        if w:
            samples, _ = audio.read_wav(w)
            parts.append(samples)
            parts.append(np.zeros(int(audio.SAMPLE_RATE * RECORDING_GAP_S), dtype=np.int16))
    if not parts:
        return None
    folder = settings.audio_dir / str(sess.id)
    full_wav, out = folder / "session.16k.wav", folder / "session.m4a"
    with wave.open(str(full_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(audio.SAMPLE_RATE)
        wf.writeframes(np.concatenate(parts).tobytes())
    try:
        await audio.encode_aac(full_wav, out)
        full_wav.unlink(missing_ok=True)
        return out
    except audio.AudioError:
        return full_wav  # fall back to WAV if AAC encoding is unavailable


def _recording_file(sess_id: int) -> Optional[Path]:
    folder = settings.audio_dir / str(sess_id)
    for name in ("session.m4a", "session.16k.wav"):
        if (folder / name).exists():
            return folder / name
    return None


def session_detail(dbs, sess: db.InterviewSession) -> SessionDetail:
    qs = db.questions_for(dbs, sess.id)
    answers = db.answers_for(dbs, sess.id)
    rep = db.report_for(dbs, sess.id)
    out_qs = []
    last_answer_at = None
    for q in qs:
        a = answers.get(q.id)
        if a:
            last_answer_at = max(filter(None, [last_answer_at, aware(a.created_at)]))
        out_qs.append(QuestionDetail(
            id=q.id, number=q.position + 1, position=q.position, main_index=q.main_index, kind=q.kind,
            section=q.section, text=q.text,
            answer=None if a is None else AnswerOut(
                skipped=a.skipped, duration=a.duration, thinking_time=a.thinking_time,
                tokens=a.tokens, stats=a.stats,
                has_audio=bool(a.audio_path and Path(a.audio_path).exists()),
            ),
        ))
    start = sess.started_at or (qs[0].created_at if qs else None)
    session_seconds = (last_answer_at - aware(start)).total_seconds() if (start and last_answer_at) else 0.0
    rec = _recording_file(sess.id)
    return SessionDetail(
        id=sess.id, created_at=iso(sess.created_at), started_at=iso(sess.started_at), role=sess.role,
        level=sess.level, interview_type=sess.interview_type, duration_minutes=sess.duration_minutes or 30,
        session_seconds=round(max(0.0, session_seconds), 1), status=sess.status, plan=session_plan(sess),
        questions=out_qs,
        recording={"url": f"/api/sessions/{sess.id}/recording", "chapters": recording_chapters(sess, qs, answers)} if rec else None,
        report=json.loads(rep.report_json) if rep else None,
    )


def _ext_for(upload: UploadFile) -> str:
    ct = (upload.content_type or "").lower()
    name = (upload.filename or "").lower()
    if "mp4" in ct or name.endswith((".mp4", ".m4a")):
        return ".mp4"
    if "ogg" in ct or name.endswith(".ogg"):
        return ".ogg"
    if "wav" in ct or name.endswith(".wav"):
        return ".wav"
    return ".webm"


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            f.write(chunk)
    if dest.stat().st_size < 200:
        raise audio.AudioError("The recording was empty. Check your microphone and try again.")


async def analyse_audio(src: Path, wav: Path) -> dict[str, Any]:
    """ffmpeg -> STT -> pauses -> disfluency timeline -> stats."""
    await audio.convert_to_wav(src, wav)
    words = await stt_service.transcribe(wav)
    p = audio.detect_pauses(wav, words)
    tokens = disfluency.build_timeline(words, p["pauses"])
    stats = fluency.answer_stats(tokens, p["thinking_time"], p["duration"])
    return {"words": words, "tokens": tokens, "stats": stats, **p}


# ---------------------------------------------------------------------------
# Planner + Interviewer
# ---------------------------------------------------------------------------

async def make_plan(sess: db.InterviewSession) -> dict[str, Any]:
    system = prompts.planner_system(sess.role, sess.level, sess.interview_type, sess.duration_minutes, ctx_of(sess))
    raw = await llm.chat(system, [{"role": "user", "content": "Create the interview plan."}], json_mode=True, max_tokens=2500)
    by_key = {s.get("key"): s for s in (raw.get("sections") or [])} if isinstance(raw, dict) else {}
    plan = {"sections": []}
    for c in prompts.COMPETENCIES:
        s = by_key.get(c["key"], {})
        exps = [str(e) for e in (s.get("expectations") or []) if str(e).strip()][:prompts.MAX_TAILORED]
        plan["sections"].append({"key": c["key"], "expectations": exps,
                                 "sample_questions": [str(q) for q in (s.get("sample_questions") or [])][:3]})
    return prompts.full_plan(plan)


def _normalise_reply(reply: Any) -> tuple[str, str, Optional[str]]:
    if not isinstance(reply, dict):
        return "question", "", None
    kind = str(reply.get("type", "question")).lower().strip()
    if kind in ("follow-up", "follow_up"):
        kind = "followup"
    if kind not in ("question", "followup", "end"):
        kind = "question"
    section = str(reply.get("section", "")).strip() or None
    if section not in prompts.COMPETENCY_KEYS:
        section = None
    return kind, str(reply.get("text", "")).strip(), section


def _create_question(dbs, sess, text: str, kind: str, main_index: int, position: int, section: Optional[str],
                     elapsed: float = 0.0) -> NextStep:
    q = db.Question(session_id=sess.id, position=position, main_index=main_index, kind=kind, text=text, section=section)
    dbs.add(q)
    dbs.commit()
    dbs.refresh(q)
    return NextStep(type=kind, text=text, question=q_out(q, sess), elapsed_seconds=round(elapsed, 1))


def build_history(qs, answers, note: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "user", "content": "Please begin the interview."}]
    for q in qs:
        msgs.append({"role": "assistant", "content": json.dumps({"type": q.kind, "section": q.section, "text": q.text})})
        a = answers.get(q.id)
        if a is None:
            content = "(no answer)"
        elif a.skipped:
            content = "(The candidate skipped this question.)"
        else:
            content = "Candidate's answer (verbatim transcript): " + (disfluency.to_plain_verbatim(a.tokens) or "(no speech detected)")
        msgs.append({"role": "user", "content": content})
    msgs[-1]["content"] += "\n\n" + note
    return msgs


async def ask_llm_next(dbs, sess: db.InterviewSession, end_now: bool = False) -> NextStep:
    """Decide and persist the next step: follow-up, next main question, or end."""
    qs = db.questions_for(dbs, sess.id)
    answers = db.answers_for(dbs, sess.id)
    current = qs[-1] if qs else None
    main_asked = max((q.main_index for q in qs), default=0)
    elapsed = elapsed_seconds(sess, qs)
    minutes = sess.duration_minutes or 30

    def finish(text: str = CLOSING_TEXT) -> NextStep:
        sess.status = "finished"
        dbs.add(sess)
        dbs.commit()
        return NextStep(type="end", text=text, elapsed_seconds=round(elapsed, 1))

    if end_now:
        return finish()

    # "Retry same questions" sessions follow the original script, no follow-ups.
    if sess.scripted_questions:
        script = json.loads(sess.scripted_questions)
        if main_asked >= len(script):
            return finish()
        item = script[main_asked]
        text, section = (item["text"], item.get("section")) if isinstance(item, dict) else (item, None)
        return _create_question(dbs, sess, text, "question", main_asked + 1, len(qs), section, elapsed)

    followups_used = sum(1 for q in qs if current and q.kind == "followup" and q.main_index == current.main_index)
    skipped_current = bool(current and current.id in answers and answers[current.id].skipped)
    followups_left = 0 if (current is None or skipped_current) else MAX_FOLLOWUPS - followups_used
    max_main = sess.num_questions

    # Hard stops: time is up, or nothing left to ask.
    if qs and elapsed >= minutes * 60 - 45:
        return finish()
    if main_asked >= max_main and followups_left <= 0:
        return finish()

    coverage = {k: 0 for k in prompts.COMPETENCY_KEYS}
    for q in qs:
        if q.section in coverage:
            coverage[q.section] += 1
    note = prompts.state_note(question_number=len(qs), elapsed_min=elapsed / 60, minutes=minutes,
                              main_asked=main_asked, max_main=max_main, followups_left=max(0, followups_left),
                              coverage=coverage)
    system = prompts.interviewer_system(sess.role, sess.level, sess.interview_type, minutes,
                                        ctx_of(sess), session_plan(sess))
    messages = build_history(qs, answers, note)
    reply = await llm.chat(system, messages, json_mode=True, max_tokens=500)
    kind, text, section = _normalise_reply(reply)

    # Enforce the rules in code, not just in the prompt.
    if kind == "end" and main_asked < max_main and elapsed < minutes * 60 * 0.75:
        messages[-1]["content"] += "\nThere is still plenty of time: ask the next main question now, preferring an uncovered section."
        reply = await llm.chat(system, messages, json_mode=True, max_tokens=500)
        kind, text, section = _normalise_reply(reply)
    llm_said_end = kind == "end"
    if not qs:
        kind = "question"
    if kind == "followup" and followups_left <= 0:
        kind = "question" if main_asked < max_main else "end"
    if kind == "question" and main_asked >= max_main:
        kind = "end"
    if kind == "end":
        # Only use the model's own closing words; never read a leftover question as a goodbye.
        return finish(text if (llm_said_end and text) else CLOSING_TEXT)
    if not text:
        raise llm.LLMError("The AI returned an empty question. Please retry.")
    if kind == "followup" and current:
        return _create_question(dbs, sess, text, kind, current.main_index, len(qs), section or current.section, elapsed)
    return _create_question(dbs, sess, text, kind, main_asked + 1, len(qs), section, elapsed)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=Health)
async def health() -> Health:
    ff = audio.ffmpeg_available()
    stt_st = stt_service.status()
    key_ok = bool(settings.resolved_llm_key) or (settings.llm_provider == "openai" and bool(settings.llm_base_url))
    llm_st = {
        "provider": settings.llm_provider,
        "model": settings.resolved_llm_model,
        "base_url": settings.llm_base_url or None,
        "ready": key_ok,
        "detail": "API key set" if key_ok else f"API key for {settings.llm_provider} missing in backend/.env",
    }
    missing = []
    if not ff:
        missing.append("ffmpeg not installed")
    if not stt_st["ready"]:
        missing.append(f"Speech: {stt_st['detail']}")
    if not key_ok:
        missing.append(f"LLM: {llm_st['detail']}")
    return Health(ok=not missing, ffmpeg=ff, stt=stt_st, llm=llm_st, missing=missing)


@app.post("/api/sessions", response_model=SessionCreated)
async def create_session(body: SessionCreate) -> SessionCreated:
    with db.get_db() as dbs:
        if body.retry_of:
            src = load_session(dbs, body.retry_of)
            mains = [{"text": q.text, "section": q.section} for q in db.questions_for(dbs, src.id) if q.kind == "question"]
            if not mains:
                raise HTTPException(400, "That interview has no questions to retry.")
            sess = db.InterviewSession(role=src.role, level=src.level, interview_type=src.interview_type,
                                       num_questions=len(mains), resume_text=src.resume_text, jd_text=src.jd_text, notes_text=src.notes_text,
                                       duration_minutes=src.duration_minutes or 30, plan_json=src.plan_json,
                                       scripted_questions=json.dumps(mains))
        else:
            minutes = body.duration_minutes
            sess = db.InterviewSession(role=body.role.strip() or "Software Engineer", level=body.level,
                                       interview_type=body.interview_type, duration_minutes=minutes,
                                       num_questions=max_main_questions(minutes), resume_text=_clean(body.resume_text),
                                       jd_text=_clean(body.jd_text), notes_text=_clean(body.notes_text))
        dbs.add(sess)
        dbs.commit()
        dbs.refresh(sess)
        try:
            if not sess.plan_json:
                sess.plan_json = json.dumps(await make_plan(sess))
                dbs.add(sess)
                dbs.commit()
            nxt = await ask_llm_next(dbs, sess)
        except Exception:
            dbs.delete(sess)
            dbs.commit()
            raise
        log.info("event=session_created id=%s role=%r level=%s type=%s minutes=%s retry_of=%s",
                 sess.id, sess.role, sess.level, sess.interview_type, sess.duration_minutes, body.retry_of)
        assert nxt.question is not None
        sections = [{"key": c["key"], "name": c["name"]} for c in prompts.COMPETENCIES]
        return SessionCreated(session_id=sess.id, duration_minutes=sess.duration_minutes,
                              sections=sections, question=nxt.question)


@app.post("/api/sessions/{session_id}/start", response_model=StartResult)
async def start_session(session_id: int) -> StartResult:
    """Starts the interview clock (called when the first question is shown)."""
    with db.get_db() as dbs:
        sess = load_session(dbs, session_id)
        if sess.started_at is None:
            sess.started_at = datetime.now(timezone.utc)
            dbs.add(sess)
            dbs.commit()
            dbs.refresh(sess)
        return StartResult(started_at=iso(sess.started_at), duration_minutes=sess.duration_minutes)


@app.post("/api/sessions/{session_id}/answer", response_model=AnswerResult)
async def submit_answer(
    session_id: int,
    question_id: int = Form(...),
    skipped: bool = Form(False),
    end_after: bool = Form(False),
    audio_file: Optional[UploadFile] = File(None, alias="audio"),
) -> AnswerResult:
    with db.get_db() as dbs:
        sess = load_session(dbs, session_id)
        q = dbs.get(db.Question, question_id)
        if not q or q.session_id != session_id:
            raise HTTPException(404, "Question not found in this session.")
        qs = db.questions_for(dbs, session_id)

        # Idempotent retry: if this answer was already processed, replay the result.
        existing = db.answers_for(dbs, session_id).get(question_id)
        if existing is not None:
            later = [x for x in qs if x.position > q.position]
            nxt = (NextStep(type=later[0].kind, text=later[0].text, question=q_out(later[0], sess))
                   if later else (await ask_llm_next(dbs, sess, end_now=end_after or sess.status == "finished")))
            return AnswerResult(question_id=question_id, skipped=existing.skipped,
                                verbatim_tokens=existing.tokens, stats=existing.stats, next=nxt)
        if qs and qs[-1].id != question_id:
            raise HTTPException(409, "This isn't the current question any more.")

        if skipped or audio_file is None:
            ans = db.Answer(session_id=session_id, question_id=question_id, skipped=True, stt_provider=settings.stt_provider)
            tokens, stats = [], {}
        else:
            folder = settings.audio_dir / str(session_id)
            src = folder / f"{question_id}{_ext_for(audio_file)}"
            await _save_upload(audio_file, src)
            wav = folder / f"{question_id}.16k.wav"
            result = await analyse_audio(src, wav)
            tokens, stats = result["tokens"], result["stats"]
            ans = db.Answer(
                session_id=session_id, question_id=question_id, audio_path=str(src),
                duration=result["duration"], thinking_time=result["thinking_time"],
                words_json=json.dumps(result["words"]), tokens_json=json.dumps(tokens),
                stats_json=json.dumps(stats), stt_provider=settings.stt_provider,
            )
            log.info("event=answer_processed session=%s question=%s words=%d fillers=%s pauses=%s",
                     session_id, question_id, stats.get("word_count"), stats["fillers"]["total"], stats["pauses"]["count"])
        dbs.add(ans)
        dbs.commit()

        nxt = await ask_llm_next(dbs, sess, end_now=end_after)
        return AnswerResult(question_id=question_id, skipped=ans.skipped, verbatim_tokens=tokens, stats=stats, next=nxt)


def recompute_answers(dbs, session_id: int) -> int:
    """Re-run pause detection, disfluency tagging and fluency stats on every saved answer."""
    n = 0
    for a in db.answers_for(dbs, session_id).values():
        if a.skipped:
            continue
        words = json.loads(a.words_json or "[]")
        wav = _wav_path(a)
        p = audio.detect_pauses(wav, words)
        tokens = disfluency.build_timeline(words, p["pauses"])
        stats = fluency.answer_stats(tokens, p["thinking_time"], p["duration"] or a.duration)
        a.tokens_json, a.stats_json, a.thinking_time = json.dumps(tokens), json.dumps(stats), p["thinking_time"]
        if p["duration"]:
            a.duration = p["duration"]
        dbs.add(a)
        n += 1
    dbs.commit()
    log.info("event=answers_recomputed session=%s answers=%d", session_id, n)
    return n


@app.post("/api/sessions/{session_id}/feedback", response_model=SessionDetail)
async def make_feedback(session_id: int, regenerate: bool = False) -> SessionDetail:
    with db.get_db() as dbs:
        sess = load_session(dbs, session_id)
        existing = db.report_for(dbs, session_id)
        if existing and not regenerate and json.loads(existing.report_json).get("version", 1) >= 5:
            return session_detail(dbs, sess)

        qs = db.questions_for(dbs, session_id)
        if regenerate:
            # Full re-assessment with the CURRENT code: rebuild pauses, disfluency tags and stats from the saved
            # recording + STT words (speech-to-text itself is not re-run), then the old report is replaced below.
            recompute_answers(dbs, session_id)
        answers = db.answers_for(dbs, session_id)
        per_answer, q_payload = [], []
        for q in qs:
            a = answers.get(q.id)
            item = {"id": q.id, "number": q.position + 1, "kind": q.kind, "section": q.section, "text": q.text,
                    "skipped": bool(a and a.skipped), "verbatim": None, "stats": None}
            if a and not a.skipped:
                item["verbatim"] = disfluency.to_plain_verbatim(a.tokens)
                item["stats"] = a.stats
                if a.stats:
                    per_answer.append({"question_id": q.id, "index": q.position + 1, "stats": a.stats})
            q_payload.append(item)
        if not per_answer:
            raise HTTPException(400, "No spoken answers were recorded, so there's nothing to give feedback on yet.")

        if not sess.plan_json:  # sessions from before plans existed
            sess.plan_json = json.dumps(await make_plan(sess))
            dbs.add(sess)
            dbs.commit()
        # Fixed rubric: every candidate is scored against the same standard expectations.
        plan = prompts.standard_plan()
        report_model = settings.resolved_report_model

        s_stats = fluency.session_stats(per_answer)
        setup = {"role": sess.role, "level": sess.level, "interview_type": sess.interview_type,
                 "minutes": sess.duration_minutes}
        log.info("calling LLM")
        raw = await llm.chat(prompts.REPORT_SYSTEM,
                             [{"role": "user", "content": prompts.report_user_message(setup, plan, q_payload, s_stats, ctx_of(sess))}],
                             # Long (60–90 min) interviews need a much bigger report budget and more time.
                             json_mode=True, max_tokens=min(60000, 24000 + 1500 * len(q_payload)),
                             # Gemini 3 models are tuned for their default temperature; others run at 0 for stability.
                             temperature=None if report_model.startswith("gemini-3") else 0,
                             model=report_model, thinking=settings.report_thinking_level or None,
                             timeout=240 + 10 * len(q_payload))


        log.info("event=report_raw session=%s model=%s", session_id, report_model)
        
        try:
            report = scoring.build_report(raw, plan, q_payload, s_stats)
        except ValueError as e:
            raise llm.LLMError("The AI returned an invalid report. Please try again.") from e
        report["generated_with"] = {"llm": settings.llm_provider, "model": report_model,
                                    "stt": stt_service.status()["name"]}
        report["assessed_at"] = datetime.now(timezone.utc).isoformat()

        await build_recording(sess, qs, answers)

        if existing:
            dbs.delete(existing)
            dbs.commit()
        dbs.add(db.Report(session_id=session_id, overall_score=report["overall_score"],
                          fluency_score=s_stats.get("fluency_score"), report_json=json.dumps(report)))
        sess.status = "finished"
        dbs.add(sess)
        dbs.commit()
        log.info("event=report_created session=%s overall=%s bands=%s", session_id, report["overall_score"],
                 {c["key"]: c["score"] for c in report["competencies"]})
        return session_detail(dbs, sess)


@app.get("/api/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    with db.get_db() as dbs:
        out = []
        for s, r in db.list_sessions(dbs):
            version = json.loads(r.report_json).get("version", 1) if r else None
            out.append(SessionSummary(
                id=s.id, created_at=iso(s.created_at), role=s.role, level=s.level, interview_type=s.interview_type,
                duration_minutes=s.duration_minutes or 30, status=s.status,
                overall_score=r.overall_score if r else None, fluency_score=r.fluency_score if r else None,
                report_version=version))
        return out


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: int) -> SessionDetail:
    with db.get_db() as dbs:
        return session_detail(dbs, load_session(dbs, session_id))


@app.get("/api/sessions/{session_id}/recording")
async def get_recording(session_id: int):
    """The full interview: every answer back to back (supports seeking)."""
    with db.get_db() as dbs:
        sess = load_session(dbs, session_id)
        path = _recording_file(session_id)
        if path is None:
            path = await build_recording(sess, db.questions_for(dbs, session_id), db.answers_for(dbs, session_id))
    if path is None:
        raise HTTPException(404, "No recording for this interview.")
    return FileResponse(path, media_type="audio/mp4" if path.suffix == ".m4a" else "audio/wav")


@app.get("/api/audio/{session_id}/{question_id}")
async def get_audio(session_id: int, question_id: int):
    with db.get_db() as dbs:
        a = db.answers_for(dbs, session_id).get(question_id)
    if not a or not a.audio_path or not Path(a.audio_path).exists():
        raise HTTPException(404, "No recording found for this answer.")
    path = Path(a.audio_path)
    media = {".webm": "audio/webm", ".mp4": "audio/mp4", ".ogg": "audio/ogg", ".wav": "audio/wav"}.get(path.suffix, "application/octet-stream")
    return FileResponse(path, media_type=media)


@app.post("/api/mic-check", response_model=MicCheckResult)
async def mic_check(audio_file: UploadFile = File(..., alias="audio")) -> MicCheckResult:
    work = settings.tmp_dir / f"mic-{uuid.uuid4().hex}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        src = work / f"clip{_ext_for(audio_file)}"
        await _save_upload(audio_file, src)
        result = await analyse_audio(src, work / "clip.wav")
        tokens = result["tokens"]
        fillers = [t["text"] for t in tokens if t["kind"] == "filler"]
        # The test phrase is "Umm, I'm ready" — we specifically want a hesitation sound.
        sound = any(disfluency.FILLER_SOUND_RE.match(t["norm"] or "") for t in tokens if t["kind"] == "filler")
        return MicCheckResult(text=disfluency.to_plain_verbatim(tokens), tokens=tokens,
                              filler_detected=sound, fillers=fillers, provider=stt_service.status()["name"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
