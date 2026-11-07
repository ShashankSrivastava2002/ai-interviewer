"""
config.py — loads all settings from backend/.env (and the process environment).

API keys live ONLY here, on the server. The frontend never sees them.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

# Report (scoring) model per provider when REPORT_LLM_MODEL is empty: the largest available.
DEFAULT_REPORT_MODELS = {
    "gemini": "gemini-3.1-pro-preview",
    "anthropic": "claude-opus-5-5",
}

# Sensible default model per LLM provider, used when LLM_MODEL is left empty.
DEFAULT_LLM_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-2.5-flash",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- LLM ----------
    llm_provider: Literal["openai", "anthropic", "gemini"] = "openai"
    llm_model: str = ""
    # For OpenAI-compatible servers: Ollama (http://localhost:11434/v1),
    # Groq (https://api.groq.com/openai/v1), vLLM (http://host:8000/v1) ...
    llm_base_url: str = ""
    llm_api_key: str = ""          # generic key for any OpenAI-compatible server
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_timeout_seconds: float = 120.0
    # Large model for the feedback report (scores are taken as final). Empty = provider default below.
    report_llm_model: str = ""

    # ---------- Speech-to-text ----------
    stt_provider: Literal["local", "deepgram", "assemblyai", "openai"] = "local"

    # Local engine: CrisperWhisper 2.0 (default, best for verbatim) or faster-whisper.
    local_stt_engine: Literal["crisperwhisper", "faster-whisper"] = "crisperwhisper"
    # CrisperWhisper 2.0 size: large | turbo | medium | small (or a HF id / local path)
    crisperwhisper_model: str = "large"
    # auto | transformers | ct2   (ct2 = CTranslate2, fastest on NVIDIA/Linux)
    crisperwhisper_backend: str = "transformers"
    # faster-whisper model name (only used when LOCAL_STT_ENGINE=faster-whisper)
    whisper_model: str = "large-v3"
    # auto | cpu | cuda
    device: str = "auto"
    # Empty = automatic (float16 on GPU, int8 on CPU for CTranslate2, float32 on CPU for PyTorch)
    compute_type: str = ""
    stt_language: str = "en"

    deepgram_api_key: str = ""
    assemblyai_api_key: str = ""
    openai_stt_model: str = "whisper-1"
    openai_stt_base_url: str = "https://api.openai.com/v1"

    # ---------- App ----------
    data_dir: Path = PROJECT_DIR / "data"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ----- derived helpers -----
    @property
    def resolved_llm_model(self) -> str:
        return self.llm_model or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def resolved_report_model(self) -> str:
        """Model that writes and scores the feedback report (a large model; its scores are final)."""
        return self.report_llm_model or DEFAULT_REPORT_MODELS.get(self.llm_provider) or self.resolved_llm_model

    @property
    def resolved_llm_key(self) -> str:
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return self.llm_api_key or self.openai_api_key

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "interviews.db"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.audio_dir, s.tmp_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s


def setup_logging(level: str = "INFO") -> None:
    """Structured-ish key=value logging that stays readable in a terminal."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Quiet down chatty libraries.
    for noisy in ("httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
