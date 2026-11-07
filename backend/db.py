"""
db.py — SQLite storage via SQLModel.

Tables
  interview_session  one mock interview (setup + status)
  question           every question asked (main or follow-up), in order
  answer             one per answered/skipped question: audio path, verbatim tokens, stats
  report             the final feedback report JSON
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from .config import get_settings

_engine = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InterviewSession(SQLModel, table=True):
    __tablename__ = "interview_session"
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
    role: str
    level: str
    interview_type: str
    num_questions: int
    resume_text: Optional[str] = None
    jd_text: Optional[str] = None         # job description
    notes_text: Optional[str] = None      # misc notes: company background, focus areas …
    # JSON list of question texts when this is a "retry same questions" session.
    scripted_questions: Optional[str] = None
    status: str = "active"  # active | finished
    duration_minutes: int = 30            # time box for the screen
    plan_json: Optional[str] = None       # interview plan: sections + expectations
    started_at: Optional[datetime] = None # when the candidate actually began (after mic check)


class Question(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="interview_session.id", index=True)
    position: int                 # order in the conversation (0, 1, 2 …)
    main_index: int               # which main question this belongs to (1-based)
    kind: str = "question"        # question | followup
    section: Optional[str] = None # competency key this question targets
    text: str
    created_at: datetime = Field(default_factory=utcnow)


class Answer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="interview_session.id", index=True)
    question_id: int = Field(foreign_key="question.id", index=True, unique=True)
    audio_path: Optional[str] = None
    skipped: bool = False
    duration: float = 0.0
    thinking_time: float = 0.0
    words_json: str = "[]"        # raw STT words
    tokens_json: str = "[]"       # verbatim timeline with disfluency labels
    stats_json: str = "{}"
    stt_provider: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def tokens(self) -> list[dict[str, Any]]:
        return json.loads(self.tokens_json or "[]")

    @property
    def stats(self) -> dict[str, Any]:
        return json.loads(self.stats_json or "{}")


class Report(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="interview_session.id", index=True, unique=True)
    created_at: datetime = Field(default_factory=utcnow)
    overall_score: Optional[float] = None
    fluency_score: Optional[float] = None
    report_json: str = "{}"


# ---------------------------------------------------------------------------

def engine():
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_engine(f"sqlite:///{s.db_path}", connect_args={"check_same_thread": False})
    return _engine


# Columns added after the first release: (table, column, SQL type). Added in place on startup
# so existing interviews are kept.
_MIGRATIONS = [
    ("interview_session", "duration_minutes", "INTEGER NOT NULL DEFAULT 30"),
    ("interview_session", "plan_json", "VARCHAR"),
    ("interview_session", "started_at", "DATETIME"),
    ("question", "section", "VARCHAR"),
    ("interview_session", "jd_text", "VARCHAR"),
    ("interview_session", "notes_text", "VARCHAR"),
]


def init_db() -> None:
    SQLModel.metadata.create_all(engine())
    with engine().begin() as conn:
        for table, col, typ in _MIGRATIONS:
            cols = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def get_db() -> Session:
    return Session(engine())


# ----- queries -----

def questions_for(db: Session, session_id: int) -> list[Question]:
    return list(db.exec(select(Question).where(Question.session_id == session_id).order_by(Question.position)))


def answers_for(db: Session, session_id: int) -> dict[int, Answer]:
    return {a.question_id: a for a in db.exec(select(Answer).where(Answer.session_id == session_id))}


def report_for(db: Session, session_id: int) -> Optional[Report]:
    return db.exec(select(Report).where(Report.session_id == session_id)).first()


def list_sessions(db: Session) -> list[tuple[InterviewSession, Optional[Report]]]:
    sessions = list(db.exec(select(InterviewSession).order_by(InterviewSession.created_at.desc())))
    reports = {r.session_id: r for r in db.exec(select(Report))}
    return [(s, reports.get(s.id)) for s in sessions]
