"""
audio.py — ffmpeg conversion and pause detection.

Pauses are found two ways and then combined (union):
  1. Voice activity detection on the 16 kHz WAV (webrtcvad, aggressiveness 2,
     30 ms frames) — or an energy-based numpy fallback if webrtcvad is missing.
  2. Gaps between consecutive word timestamps from the speech recogniser.

Every silence >= 0.5 s *inside the answer* (between the first and last word) is
reported and categorised:
    short      0.5 – 1.5 s
    long       1.5 – 3 s
    very_long  > 3 s
The silence before the first word is reported separately as "thinking time".
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import wave
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("audio")

SAMPLE_RATE = 16_000
FRAME_MS = 30
MIN_PAUSE_S = 0.5

try:  # webrtcvad-wheels installs the same "webrtcvad" module
    import webrtcvad  # type: ignore

    _HAS_VAD = True
except Exception:  # pragma: no cover - depends on the environment
    webrtcvad = None
    _HAS_VAD = False


class AudioError(RuntimeError):
    """Raised with a human-friendly message when audio can't be processed."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


async def convert_to_wav(src: Path, dst: Path) -> Path:
    """Convert any browser recording (webm/opus, mp4/aac, ogg) to 16 kHz mono 16-bit WAV."""
    if not ffmpeg_available():
        raise AudioError("ffmpeg is not installed or not on PATH. Install it (see README) and restart the server.")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
        str(dst),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dst.exists():
        msg = err.decode(errors="ignore").strip()[-400:]
        log.error("event=ffmpeg_failed src=%s detail=%r", src, msg)
        raise AudioError("Couldn't read that recording (it may be empty or corrupted). Please try answering again.")
    return dst


async def encode_aac(src: Path, dst: Path) -> Path:
    """Compress a WAV to AAC/.m4a (used for the full-session recording)."""
    if not ffmpeg_available():
        raise AudioError("ffmpeg is not installed.")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(dst),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dst.exists():
        log.error("event=aac_encode_failed detail=%r", err.decode(errors="ignore")[-300:])
        raise AudioError("Couldn't encode the session recording.")
    return dst


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (int16 samples, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16), sr


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate() or 1)


# ---------------------------------------------------------------------------
# Voice activity
# ---------------------------------------------------------------------------

def speech_flags(samples: np.ndarray, sr: int = SAMPLE_RATE) -> list[bool]:
    """One bool per 30 ms frame: True = speech."""
    frame_len = int(sr * FRAME_MS / 1000)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return []
    if _HAS_VAD and sr in (8000, 16000, 32000, 48000):
        vad = webrtcvad.Vad(2)
        raw = samples[: n_frames * frame_len].tobytes()
        step = frame_len * 2  # bytes per frame (16-bit)
        return [vad.is_speech(raw[i * step:(i + 1) * step], sr) for i in range(n_frames)]
    return _energy_flags(samples, frame_len, n_frames)


def _energy_flags(samples: np.ndarray, frame_len: int, n_frames: int) -> list[bool]:
    """Fallback VAD: a frame is speech if its RMS is well above the noise floor."""
    frames = samples[: n_frames * frame_len].astype(np.float32).reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    noise_floor = np.percentile(rms, 15)
    peak = np.percentile(rms, 95)
    threshold = max(noise_floor * 2.5, noise_floor + 0.12 * (peak - noise_floor), 120.0)
    return [bool(v > threshold) for v in rms]


def _silences_from_flags(flags: list[bool], min_len: float = MIN_PAUSE_S) -> list[tuple[float, float]]:
    """Runs of non-speech frames at least min_len seconds long."""
    out: list[tuple[float, float]] = []
    frame_s = FRAME_MS / 1000
    run_start = None
    for i, is_speech in enumerate(flags + [True]):
        if not is_speech and run_start is None:
            run_start = i
        elif is_speech and run_start is not None:
            s, e = run_start * frame_s, i * frame_s
            if e - s >= min_len:
                out.append((s, e))
            run_start = None
    return out


def _first_speech_time(flags: list[bool]) -> float | None:
    """First moment with ~150 ms of sustained speech (ignores single clicks)."""
    need = 5
    for i in range(len(flags) - need + 1):
        if all(flags[i:i + need]):
            return i * FRAME_MS / 1000
    return None


# ---------------------------------------------------------------------------
# Combining both methods
# ---------------------------------------------------------------------------

def categorize(duration: float) -> str:
    if duration > 3.0:
        return "very_long"
    if duration >= 1.5:
        return "long"
    return "short"


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_pauses(wav_path: Path | None, words: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Returns {
      pauses: [{start, end, duration, category, source}],
      thinking_time: seconds from recording start to first word,
      duration: total audio seconds,
      vad: "webrtcvad" | "energy" | "none"
    }
    """
    flags: list[bool] = []
    duration = 0.0
    if wav_path is not None and wav_path.exists():
        samples, sr = read_wav(wav_path)
        duration = len(samples) / float(sr or SAMPLE_RATE)
        flags = speech_flags(samples, sr)

    vad_silences = _silences_from_flags(flags) if flags else []

    words = sorted(words, key=lambda w: w["start"])
    gap_silences = [
        (a["end"], b["start"])
        for a, b in zip(words, words[1:])
        if b["start"] - a["end"] >= MIN_PAUSE_S
    ]

    # Thinking time: recording start -> first recognised word (VAD as fallback).
    if words:
        thinking = float(words[0]["start"])
        speech_start, speech_end = float(words[0]["start"]), float(words[-1]["end"])
    else:
        first = _first_speech_time(flags)
        thinking = float(first) if first is not None else float(duration)
        speech_start, speech_end = thinking, thinking

    # Union of both methods, clipped to the spoken part of the answer.
    combined = _merge_intervals(vad_silences + gap_silences)
    # A recognised word is never silence: cut word spans out of every interval
    # (VAD can miss a soft "um" that the recogniser still heard).
    pieces: list[tuple[float, float]] = []
    for s, e in combined:
        segs = [(max(s, speech_start), min(e, speech_end))]
        for w in words:
            nxt = []
            for a, b in segs:
                if w["end"] <= a or w["start"] >= b:
                    nxt.append((a, b))
                    continue
                if w["start"] > a:
                    nxt.append((a, w["start"]))
                if w["end"] < b:
                    nxt.append((w["end"], b))
            segs = nxt
        pieces.extend(segs)

    pauses = []
    for s, e in pieces:
        if e - s < MIN_PAUSE_S:
            continue
        d = e - s
        in_vad = any(vs <= s + 0.1 and ve >= e - 0.1 for vs, ve in vad_silences)
        in_gap = any(gs <= s + 0.1 and ge >= e - 0.1 for gs, ge in gap_silences)
        pauses.append({
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(d, 2),
            "category": categorize(d),
            "source": "both" if (in_vad and in_gap) else ("vad" if in_vad else "words"),
        })

    return {
        "pauses": pauses,
        "thinking_time": round(thinking, 2),
        "duration": round(duration, 2),
        "vad": ("webrtcvad" if _HAS_VAD else "energy") if flags else "none",
    }
