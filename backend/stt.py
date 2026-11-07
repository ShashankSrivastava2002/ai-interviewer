"""
stt.py — verbatim speech-to-text with word-level timestamps.

Every provider returns the same shape:
    [{"word": str, "start": float, "end": float, "confidence": float | None}]

Providers (STT_PROVIDER in .env):
  local       CrisperWhisper 2.0 (default; LOCAL_STT_ENGINE=crisperwhisper)
              or faster-whisper (LOCAL_STT_ENGINE=faster-whisper)
  deepgram    nova-3 with filler_words=true
  assemblyai  disfluencies=true
  openai      /audio/transcriptions verbose_json + word timestamps

Why CrisperWhisper 2.0? It is trained to transcribe *verbatim* — fillers
("[UM]", "[UH]"), stutters ("th- the"), repetitions and false starts are kept
instead of being smoothed away — and gives ~30 ms word timings. Its weights are
under nyra health's Non-Commercial Research License.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger("stt")

# Nudges Whisper-family models (which love to "clean up" speech) towards verbatim output.
VERBATIM_PROMPT = "Umm, so, uh, I-I think, like, hmm... you know, er, I mean, uh, basically—"


class STTError(RuntimeError):
    """Raised with a human-friendly message when transcription fails."""


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(setting: str) -> str:
    if setting and setting != "auto":
        return setting
    return "cuda" if _cuda_available() else "cpu"


# ---------------------------------------------------------------------------
# Local engines (loaded once, at startup, in a background thread)
# ---------------------------------------------------------------------------

class LocalEngine:
    """Holds the loaded local model. Inference is serialised with a lock."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = settings.local_stt_engine
        self.model: Any = None
        self.status = "not_loaded"      # not_loaded | loading | ready | error
        self.error = ""
        self.device = ""
        self.loaded = threading.Event()
        self._lock = threading.Lock()

    # ----- loading -----
    def load(self) -> None:
        self.status = "loading"
        t0 = time.perf_counter()
        try:
            self.device = resolve_device(self.settings.device)
            if self.engine == "crisperwhisper":
                self._load_crisperwhisper()
            else:
                self._load_faster_whisper()
            self.status = "ready"
            log.info("event=stt_model_loaded engine=%s device=%s seconds=%.1f",
                     self.engine, self.device, time.perf_counter() - t0)
        except Exception as e:  # keep the server up; /api/health explains
            self.status = "error"
            self.error = _friendly_load_error(self.engine, e)
            log.exception("event=stt_model_load_failed engine=%s", self.engine)
        finally:
            self.loaded.set()

    def _load_crisperwhisper(self) -> None:
        if importlib.util.find_spec("crisperwhisper") is None:
            raise ImportError("crisperwhisper")
        from crisperwhisper import CrisperWhisperModel  # type: ignore

        s = self.settings
        backend = s.crisperwhisper_backend or "auto"
        compute = s.compute_type
        if not compute:
            if self.device == "cuda":
                compute = "float16"
            elif backend == "ct2":
                compute = "int8"            # CPU + CTranslate2 -> int8
            else:
                compute = "float32"         # CPU + PyTorch -> float32 is the safe choice
        kwargs = {"backend": backend, "device": self.device, "compute_type": compute}
        log.info("event=stt_loading engine=crisperwhisper model=%s %s", s.crisperwhisper_model, kwargs)
        try:
            self.model = CrisperWhisperModel(s.crisperwhisper_model, **kwargs)
        except TypeError:
            # Older/newer package versions may not accept every keyword.
            self.model = CrisperWhisperModel(s.crisperwhisper_model, backend=backend)

    def _load_faster_whisper(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        compute = self.settings.compute_type or ("float16" if self.device == "cuda" else "int8")
        log.info("event=stt_loading engine=faster-whisper model=%s device=%s compute=%s",
                 self.settings.whisper_model, self.device, compute)
        self.model = WhisperModel(self.settings.whisper_model, device=self.device, compute_type=compute)

    # ----- inference -----
    def transcribe(self, wav_path: Path) -> list[dict[str, Any]]:
        if not self.loaded.wait(timeout=1800):
            raise STTError("The speech model is still loading. Please wait a moment and retry.")
        if self.status != "ready":
            raise STTError(self.error or "The local speech model failed to load.")
        with self._lock:
            if self.engine == "crisperwhisper":
                return self._run_crisperwhisper(wav_path)
            return self._run_faster_whisper(wav_path)

    def _run_crisperwhisper(self, wav_path: Path) -> list[dict[str, Any]]:
        lang = self.settings.stt_language
        try:
            result = self.model.transcribe(
                str(wav_path), language=lang, mode="verbatim",
                word_timestamps=True, word_confidence=True,
            )
        except TypeError:
            result = self.model.transcribe(str(wav_path), language=lang, mode="verbatim", word_timestamps=True)
        words = []
        for w in getattr(result, "words", None) or []:
            text = str(getattr(w, "word", "")).strip()
            if not text:
                continue
            words.append({
                "word": text,
                "start": float(w.start),
                "end": float(w.end),
                "confidence": _maybe_float(getattr(w, "probability", None)),
            })
        return words

    def _run_faster_whisper(self, wav_path: Path) -> list[dict[str, Any]]:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.settings.stt_language,
            word_timestamps=True,
            initial_prompt=VERBATIM_PROMPT,
            vad_filter=False,                    # keep every pause
            condition_on_previous_text=False,    # less "clean-up" drift
            beam_size=5,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append({"word": text, "start": float(w.start), "end": float(w.end),
                                  "confidence": _maybe_float(w.probability)})
        return words


def _maybe_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _friendly_load_error(engine: str, e: Exception) -> str:
    if isinstance(e, ImportError) and str(e) in ("crisperwhisper", "No module named 'faster_whisper'"):
        if engine == "crisperwhisper":
            return ('CrisperWhisper 2.0 is not installed. Run: pip install "crisperwhisper[transformers]>=2.0.4" '
                    '(or "crisperwhisper[ct2]>=2.0.4" on an NVIDIA/Linux machine).')
        return "faster-whisper is not installed. Run: pip install faster-whisper"
    return f"Couldn't load the local speech model ({engine}): {e}"


# ---------------------------------------------------------------------------
# Cloud providers
# ---------------------------------------------------------------------------

async def _deepgram(wav_path: Path, s: Settings) -> list[dict[str, Any]]:
    if not s.deepgram_api_key:
        raise STTError("DEEPGRAM_API_KEY is missing in backend/.env.")
    params = {
        "model": "nova-3", "filler_words": "true", "punctuate": "true",
        "smart_format": "false", "language": s.stt_language,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/listen", params=params,
            headers={"Authorization": f"Token {s.deepgram_api_key}", "Content-Type": "audio/wav"},
            content=wav_path.read_bytes(),
        )
    _raise_for(r, "Deepgram")
    alt = r.json()["results"]["channels"][0]["alternatives"][0]
    return [
        {"word": w.get("punctuated_word") or w["word"], "start": float(w["start"]),
         "end": float(w["end"]), "confidence": _maybe_float(w.get("confidence"))}
        for w in alt.get("words", [])
    ]


async def _assemblyai(wav_path: Path, s: Settings) -> list[dict[str, Any]]:
    if not s.assemblyai_api_key:
        raise STTError("ASSEMBLYAI_API_KEY is missing in backend/.env.")
    headers = {"authorization": s.assemblyai_api_key}
    base = "https://api.assemblyai.com/v2"
    async with httpx.AsyncClient(timeout=120) as client:
        up = await client.post(f"{base}/upload", headers=headers, content=wav_path.read_bytes())
        _raise_for(up, "AssemblyAI upload")
        job = await client.post(f"{base}/transcript", headers=headers, json={
            "audio_url": up.json()["upload_url"],
            "disfluencies": True,
            "language_code": s.stt_language,
            "punctuate": True,
            "format_text": False,
        })
        _raise_for(job, "AssemblyAI")
        tid = job.json()["id"]
        for _ in range(240):  # up to ~4 minutes
            await asyncio.sleep(1.0)
            res = await client.get(f"{base}/transcript/{tid}", headers=headers)
            _raise_for(res, "AssemblyAI")
            data = res.json()
            if data["status"] == "completed":
                return [
                    {"word": w["text"], "start": w["start"] / 1000.0, "end": w["end"] / 1000.0,
                     "confidence": _maybe_float(w.get("confidence"))}
                    for w in data.get("words") or []
                ]
            if data["status"] == "error":
                raise STTError(f"AssemblyAI couldn't transcribe the audio: {data.get('error')}")
    raise STTError("AssemblyAI took too long to respond. Please retry.")


async def _openai(wav_path: Path, s: Settings) -> list[dict[str, Any]]:
    if not s.openai_api_key:
        raise STTError("OPENAI_API_KEY is missing in backend/.env.")
    data = [
        ("model", s.openai_stt_model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("prompt", VERBATIM_PROMPT),
        ("language", s.stt_language),
    ]
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"{s.openai_stt_base_url.rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {s.openai_api_key}"},
            data=data,
            files={"file": (wav_path.name, wav_path.read_bytes(), "audio/wav")},
        )
    _raise_for(r, "OpenAI transcription")
    return [
        {"word": w["word"], "start": float(w["start"]), "end": float(w["end"]), "confidence": None}
        for w in r.json().get("words") or []
    ]


def _raise_for(r: httpx.Response, who: str) -> None:
    if r.status_code < 400:
        return
    log.error("event=stt_http_error provider=%s status=%s body=%r", who, r.status_code, r.text[:300])
    if r.status_code in (401, 403):
        raise STTError(f"{who} rejected the API key. Check it in backend/.env.")
    if r.status_code == 429:
        raise STTError(f"{who} is rate-limiting us. Wait a few seconds and press Retry.")
    raise STTError(f"{who} returned an error ({r.status_code}). Please retry.")


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class STTService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.stt_provider
        self.local = LocalEngine(settings) if self.provider == "local" else None

    async def start(self) -> None:
        """Kick off local model loading without blocking server start-up."""
        if self.local is not None:
            asyncio.get_running_loop().run_in_executor(None, self.local.load)

    def status(self) -> dict[str, Any]:
        s = self.settings
        if self.provider == "local":
            name = (f"CrisperWhisper 2.0 ({s.crisperwhisper_model})" if s.local_stt_engine == "crisperwhisper"
                    else f"faster-whisper ({s.whisper_model})")
            st = self.local.status if self.local else "error"
            detail = {"ready": "Loaded on " + (self.local.device if self.local else "?"),
                      "loading": "Loading model… (first run downloads weights)",
                      "not_loaded": "Not loaded yet",
                      "error": self.local.error if self.local else ""}[st]
            return {"provider": "local", "name": name, "ready": st == "ready", "state": st, "detail": detail}
        key = {"deepgram": s.deepgram_api_key, "assemblyai": s.assemblyai_api_key, "openai": s.openai_api_key}[self.provider]
        names = {"deepgram": "Deepgram nova-3", "assemblyai": "AssemblyAI", "openai": f"OpenAI {s.openai_stt_model}"}
        return {
            "provider": self.provider, "name": names[self.provider], "ready": bool(key),
            "state": "ready" if key else "error",
            "detail": "API key set" if key else f"{self.provider.upper()}_API_KEY missing in backend/.env",
        }

    async def transcribe(self, wav_path: Path) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        if self.provider == "local":
            assert self.local is not None
            words = await asyncio.to_thread(self.local.transcribe, wav_path)
        elif self.provider == "deepgram":
            words = await _deepgram(wav_path, self.settings)
        elif self.provider == "assemblyai":
            words = await _assemblyai(wav_path, self.settings)
        else:
            words = await _openai(wav_path, self.settings)
        words.sort(key=lambda w: w["start"])
        log.info("event=stt_done provider=%s words=%d seconds=%.2f", self.provider, len(words), time.perf_counter() - t0)
        return words
