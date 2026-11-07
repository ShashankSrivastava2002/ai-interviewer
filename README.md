# Interview Room — AI voice interviewer with verbatim feedback

An AI interviewer asks you questions out loud, records your spoken answers, and keeps a
**fully verbatim transcript**: every *um*, *uh*, stammer (*I-I-I*), repetition (*the the*),
false start (*I was going— I went*) and pause. Your feedback report uses those
markers, with fluency statistics computed in Python and content judged by an LLM.

```
ai-interviewer/
  backend/    FastAPI app: STT, pause detection, disfluency tagging, LLM, SQLite
  frontend/   plain HTML/CSS/JS (no frameworks), served by FastAPI at "/"
  data/       created on first run: recordings + interviews.db
```

## 1. Install ffmpeg

The backend uses ffmpeg to convert browser recordings (webm/opus) to 16 kHz mono WAV.

| OS | Command |
|---|---|
| **macOS** | `brew install ffmpeg` |
| **Windows** | `winget install Gyan.FFmpeg` (or `choco install ffmpeg`), then open a new terminal |
| **Ubuntu/Debian** | `sudo apt update && sudo apt install ffmpeg` |
| **Fedora** | `sudo dnf install ffmpeg` |

Check it with `ffmpeg -version`.

## 2. Create a virtual environment and install

Use Python 3.10–3.12.

```bash
cd ai-interviewer
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r backend/requirements.txt
```

The requirements include **CrisperWhisper 2.0** with its PyTorch backend
(`crisperwhisper[transformers]>=2.0.4`). This pulls in PyTorch, which is a large download.
If you only plan to use a cloud speech provider, you can remove the
`crisperwhisper` and `faster-whisper` lines first.

On an **NVIDIA GPU under Linux**, the CTranslate2 backend is faster:

```bash
pip install "crisperwhisper[ct2,convert]>=2.0.4"
# then in backend/.env:  CRISPERWHISPER_BACKEND=ct2
```

## 3. Configure

```bash
cp backend/.env.example backend/.env      # Windows: copy backend\.env.example backend\.env
```

Edit `backend/.env`. You need two things:

1. **An LLM** for questions and feedback. Set `LLM_PROVIDER` to `openai`, `anthropic`
   or `gemini` and add the matching key.
   - Local and free: install [Ollama](https://ollama.com), run `ollama pull llama3.1`,
     then set `LLM_PROVIDER=openai`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.1`.
   - Groq: `LLM_PROVIDER=openai`, `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=...`.
2. **Verbatim speech-to-text**, set with `STT_PROVIDER`:

| `STT_PROVIDER` | What it uses | Notes |
|---|---|---|
| `local` (default) | **CrisperWhisper 2.0** (`LOCAL_STT_ENGINE=crisperwhisper`) | Trained to transcribe verbatim. Emits `[UM]`/`[UH]`, stutters and false starts, with ~30 ms word timings. Pick a size with `CRISPERWHISPER_MODEL`: `large`, `turbo`, `medium` or `small`. |
| `local` | faster-whisper (`LOCAL_STT_ENGINE=faster-whisper`) | Uses a verbatim `initial_prompt`. Whisper still tends to drop some fillers. |
| `deepgram` | nova-3, `filler_words=true` | Fast and good at fillers. Needs `DEEPGRAM_API_KEY`. |
| `assemblyai` | `disfluencies=true` | Needs `ASSEMBLYAI_API_KEY`. |
| `openai` | `/audio/transcriptions`, word timestamps, verbatim prompt | Needs `OPENAI_API_KEY`. Usually drops more fillers than the others. |

> **License note:** CrisperWhisper 2.0 weights are released under nyra health's
> **Non-Commercial Research License**. That is fine for personal practice. For commercial
> use, get a license from nyra health or switch to Deepgram or AssemblyAI.

The local model is loaded **once, at startup**, in the background. The first run downloads the
weights from Hugging Face. The status chip in the app shows "Loading speech model…" until it
is ready. It uses the GPU if one is available and the CPU otherwise (int8 for CTranslate2,
float32 for PyTorch). Set `DEVICE`/`COMPUTE_TYPE` to override this. On a CPU, `turbo` or `small`
keeps answers snappy.

## 4. Run

From the `ai-interviewer/` folder:

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000**.

> 🎙️ Browsers allow microphone access only on **localhost or https**. Use
> `http://localhost:8000`, not your LAN IP. To use a phone, put the app behind https
> (for example with a tunnel such as `cloudflared` or `ngrok`).

The status chip at the top right calls `/api/health`. Click it to see exactly what is missing
(ffmpeg, the speech model or the LLM key).

## 5. Tests

```bash
pytest backend/tests -q
```

The tests cover the disfluency rules (fillers, stammers, repetitions, false starts), pause
categories, the verbatim guarantee (the transcript is never cleaned), and fluency statistics.

---

## The screen report

The flow follows the published design of recruiter-style AI screens (e.g. HackerRank's AI interviewer):

1. **Plan:** from the role, level and resume/JD, a planner writes **expectations** for four competency sections:
   - Role Alignment & Scope
   - Technical Background & Experience
   - Impact & Ownership
   - Communication & Clarity
2. **Interview:** a time-boxed voice screen (default 30 min). Questions are numbered, follow-ups included.
   - The interviewer covers every section.
   - When an answer is vague, it probes with up to 2 follow-ups per main question.
   - It never comments on delivery.
3. **Scoring:** each expectation is scored **3 Met, 2 Partially met, 1 Not met, 0 Not assessed**.
   - Every score must quote your **exact words**. `scoring.py` checks each quote against the transcript.
   - A quote that can't be found is dropped. An expectation with no valid quote becomes *Not Assessed*.
4. **Numbers, all computed in Python (strict):**
   - Each expectation earns **Met 100 · Partially met 45 · Not met 10 · No evidence 0**.
     "No evidence" counts as zero, because if you didn't show it, you get no credit for it.
   - **Substance gates:** a level only counts if a cited answer is long enough: Met needs 40+ content words and
     Partially met needs 20+. Otherwise the level is lowered. Question ratings are capped too: under 15 words is Weak,
     under 35 is Moderate at most.
   - **Competency score:** the average of its expectations. A section the interview never reached scores 0 ("Not reached").
   - **Communication & Clarity:** rubric × (0.5 + 0.5 × delivery/100). Fluent speech keeps the full score; heavy fillers
     and pauses can halve it. Delivery never *adds* points. Near-empty answers count as 0 delivery.
   - Expectations the interview never asked about are "Not probed" and excluded; asked-but-unanswered ones score 0.
   - **Overall:** the plain average of all four competencies. Bands: Weak < 30 ≤ Moderate < 50 ≤ Good < 75 ≤ Excellent.
   - Optional `LLM_REPORT_MODEL` in `.env` uses a stronger model just for grading (e.g. `gemini-2.5-pro`).
5. **Report:**
   - Overall score with its band
   - The four competency scores
   - Overall Performance
   - What You Are Doing Well / Top Things You Can Improve
   - Full-session recording, with speed control and question chapters
   - Detailed Breakdown, one tab per competency:
     - improvements and strengths, each with "Seen in: Question N" links
     - the evidence behind the score
     - every question with your verbatim answer, a rating, "Why this needs improvement" and a "Strong Answer"
   - Exports to PDF, CSV, verbatim .txt and JSON.

## How it works

```
browser (MediaRecorder webm/opus)
   └─ POST /api/sessions/{id}/answer
        1. save  data/audio/{session}/{question}.webm
        2. ffmpeg → 16 kHz mono WAV
        3. verbatim STT with word timestamps          (stt.py)
        4. pauses: webrtcvad ∪ word-timestamp gaps     (audio.py)
        5. one timeline + disfluency labels            (disfluency.py)
        6. stats                                       (fluency.py) → SQLite (db.py)
        7. LLM decides: follow-up (max 1) / next question / end
```

**Pauses.** Silences of at least 0.5 s between the first and last word are reported as
*short* (0.5–1.5 s), *long* (1.5–3 s) or *very long* (> 3 s). The silence before the first
word is *thinking time*. Recording never stops on silence, so pauses are kept.

**Fluency statistics** (Python, never the LLM):

- filler total, rate per minute and top fillers
- stammer, repetition and false-start counts
- pause count, longest, average and % silent
- average thinking time
- words per minute, with fillers and pauses excluded

Each answer also gets a 0–10 fluency score with transparent penalties (see `fluency.fluency_score`):

| Penalty | Amount | Cap |
|---|---|---|
| Fillers above 2 per minute | −0.35 per filler/min | −3 |
| Stammers, repetitions and false starts | −0.5 per event/min | −2.5 |
| Silence above 15% | −0.06 per % | −2 |
| Pauses of 1.5 s or longer | −0.3 each | −1.5 |
| Pace outside 120–160 WPM | −0.03 per WPM | −1.5 |

**Disfluency rules** (`disfluency.py`) are heuristics:

- **Filler sounds:** *um, uh, er, hmm* … are always fillers.
- **Context-dependent words:** *like, so, you know, I mean, kind of, sort of, right?* count
  as fillers only in filler positions. For example, *"I like Python"* is not a filler, but
  *"it was, like, slow"* is.
- **Repetitions:** for back-to-back repeats, the extra copies are tagged.
- **False starts:** phrases cut off with *—* and truncated words such as *wh-* are tagged.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | ffmpeg / STT / LLM status |
| POST | `/api/sessions` | plan + first question (`duration_minutes`; `retry_of` re-runs the same questions) |
| POST | `/api/sessions/{id}/start` | start the interview clock |
| GET | `/api/sessions/{id}/recording` | full-session recording (all answers, with chapters in the session JSON) |
| POST | `/api/sessions/{id}/answer` | multipart `audio` + `question_id` (or `skipped=true`) → verbatim tokens + next step |
| POST | `/api/sessions/{id}/feedback` | build and store the report (`?regenerate=true` to redo it) |
| GET | `/api/sessions` | history |
| GET | `/api/sessions/{id}` | full session, transcript and report |
| GET | `/api/audio/{session_id}/{question_id}` | recorded answer (supports seeking) |
| POST | `/api/mic-check` | transcribe a short clip and report whether a filler was captured |

## Troubleshooting

- **"Your speech engine is removing fillers"** in the mic check: switch to `STT_PROVIDER=local`
  with CrisperWhisper 2.0, or to Deepgram.
- **An upload failed mid-interview:** the answer stays in the browser tab. Press **Retry upload**,
  or **Save recording** to keep the file.
- **Slow transcription on CPU:** use `CRISPERWHISPER_MODEL=turbo` or `small`.
- **The LLM returns odd output:** the backend retries once on malformed JSON and backs off
  on rate limits. Small local models sometimes struggle with the long feedback JSON.
  Try a larger model.
