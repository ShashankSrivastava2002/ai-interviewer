/* =====================================================================
   Interview Room — frontend (no frameworks)

   Screens:  #/setup → #/mic → #/interview → #/loading → #/report/:id
             #/history
   The browser never sees API keys: everything goes through /api/*.
   ===================================================================== */
"use strict";

// ---------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** fetch wrapper that turns server errors into friendly Error messages. */
async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new Error("Can't reach the server. Is it still running?");
  }
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    let msg = `Something went wrong (HTTP ${res.status}).`;
    if (data && data.detail) {
      msg = typeof data.detail === "string" ? data.detail : data.detail.map((d) => d.msg).join("; ");
    }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

/** localStorage that never throws (private mode, blocked storage …). */
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
};

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

function fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec));
  return `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
}

function fmtDate(iso) {
  try { return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }); } catch { return iso; }
}

const TYPE_LABEL = { technical: "Technical", behavioural: "HR & Behavioural", mixed: "Mixed" };

// ---------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------
const state = {
  health: null,
  createPromise: null,   // pending POST /api/sessions (started while the user does the mic check)
  sessionId: null,
  question: null,
  total: 0,

  // audio
  stream: null,
  audioCtx: null,
  analyser: null,
  level: 0,              // 0..1 smoothed loudness
  rms: 0,
  noiseFloor: store.get("noiseFloor", 0.008),
  mime: "",

  // interviewer voice
  voiceURI: store.get("voiceURI", null),
  rate: store.get("rate", 1),
  muted: store.get("muted", false),

  // interview flow
  phase: "idle",         // idle | speaking | listening | uploading | error | done
  recorder: null,
  chunks: [],
  askToken: 0,
  pending: null,         // answer waiting to upload (kept until it succeeds)
  ending: false,
  timerStart: 0,
  timerId: null,
  pauseCount: 0,
  lastVoiceAt: 0,
  hasSpoken: false,
  inSilence: false,

  detailCache: {},       // session id -> detail with report
};

// ---------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------
function show(name) {
  $$(".screen").forEach((s) => (s.hidden = s.dataset.screen !== name));
  const tab = name === "history" || name === "report" ? "history" : "setup";
  $$("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === tab));
  document.body.dataset.screen = name;
  window.scrollTo(0, 0);
}

async function route() {
  const hash = location.hash || "#/setup";
  const [, name, arg] = hash.split("/");
  switch (name) {
    case "mic":
      if (!state.createPromise) { location.hash = "#/setup"; return; }
      show("mic");
      initMicScreen();
      break;
    case "interview":
      if (!state.sessionId) { location.hash = "#/setup"; return; }
      show("interview");
      break;
    case "loading":
      show("loading");
      break;
    case "report":
      show("report");
      await openReport(Number(arg));
      break;
    case "history":
      show("history");
      await loadHistory();
      break;
    default:
      show("setup");
  }
}
window.addEventListener("hashchange", route);

window.addEventListener("beforeunload", (e) => {
  if (["listening", "uploading", "error"].includes(state.phase) || state.pending) {
    e.preventDefault();
    e.returnValue = "";
  }
});

// ---------------------------------------------------------------------
// Health chip
// ---------------------------------------------------------------------
async function loadHealth() {
  const chip = $("#health-chip");
  try {
    const h = await api("/api/health");
    state.health = h;
    chip.classList.toggle("ok", h.ok);
    chip.classList.toggle("bad", !h.ok);
    $(".label", chip).textContent = h.ok ? "AI ready ✓" : (h.stt.state === "loading" ? "Loading speech model…" : `Setup needed (${h.missing.length})`);
    renderHealthPanel(h);
    if (h.stt.state === "loading") setTimeout(loadHealth, 5000);
  } catch (e) {
    chip.classList.add("bad");
    $(".label", chip).textContent = "Server offline";
    $("#health-panel").innerHTML = `<b>Can't reach the server.</b> Start it with <code>uvicorn backend.main:app --reload</code>.`;
  }
}

function renderHealthPanel(h) {
  const row = (ok, label, detail) => `<li>${ok ? "✅" : "⚠️"} <b>${esc(label)}</b> — ${esc(detail)}</li>`;
  $("#health-panel").innerHTML = `
    <b>${h.ok ? "Everything is ready." : "A few things need attention:"}</b>
    <ul>
      ${row(h.ffmpeg, "ffmpeg", h.ffmpeg ? "installed" : "not found on PATH — see README")}
      ${row(h.stt.ready, "Speech-to-text: " + h.stt.name, h.stt.detail)}
      ${row(h.llm.ready, `LLM: ${h.llm.provider} · ${h.llm.model}`, h.llm.detail)}
    </ul>
    ${h.ok ? "" : '<p class="muted small">Edit <code>backend/.env</code>, then restart the server.</p>'}`;
}

$("#health-chip").addEventListener("click", () => {
  const p = $("#health-panel");
  p.hidden = !p.hidden;
});

// ---------------------------------------------------------------------
// SCREEN 1 — Setup
// ---------------------------------------------------------------------
function initSetup() {
  const saved = store.get("setup", null);
  const form = $("#setup-form");
  if (saved) {
    $("#role").value = saved.role || "";
    const lvl = form.querySelector(`input[name=level][value="${saved.level}"]`);
    if (lvl) lvl.checked = true;
    const typ = form.querySelector(`input[name=interview_type][value="${saved.interview_type}"]`);
    if (typ) typ.checked = true;
    $("#duration").value = saved.duration_minutes || 30;
    $("#resume").value = saved.resume_text || "";
    $("#jd").value = saved.jd_text || "";
    $("#notes").value = saved.notes_text || "";
  }
  syncChips();
  updateLength();

  $("#role-chips").addEventListener("click", (e) => {
    const b = e.target.closest("[data-role]");
    if (!b) return;
    $("#role").value = b.dataset.role;
    syncChips();
  });
  $("#role").addEventListener("input", syncChips);
  $("#duration").addEventListener("input", updateLength);

  // Resume / JD / notes: live character counts + optional .txt loading
  const counts = () => $$("[data-count-for]").forEach((el) => {
    const ta = $(`#${el.dataset.countFor}`);
    el.textContent = `${ta.value.length.toLocaleString()} / ${Number(ta.maxLength).toLocaleString()}`;
  });
  ["resume", "jd", "notes"].forEach((id) => $(`#${id}`).addEventListener("input", counts));
  counts();
  $$("[data-file-for]").forEach((btn) => {
    const target = btn.dataset.fileFor;
    const input = $(`#${target}-file`);
    btn.addEventListener("click", (e) => { e.preventDefault(); input.click(); });
    input.addEventListener("change", async () => {
      const f = input.files?.[0];
      if (!f) return;
      if (f.size > 200_000) { toast("That file is too large — paste the text instead."); return; }
      $(`#${target}`).value = (await f.text()).slice(0, Number($(`#${target}`).maxLength));
      counts();
      toast(`Loaded ${f.name}`);
      input.value = "";
    });
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const err = $("#setup-error");
    err.hidden = true;
    const setup = {
      role: $("#role").value.trim(),
      level: form.querySelector("input[name=level]:checked").value,
      interview_type: form.querySelector("input[name=interview_type]:checked").value,
      duration_minutes: Number($("#duration").value),
      resume_text: $("#resume").value.trim() || null,
      jd_text: $("#jd").value.trim() || null,
      notes_text: $("#notes").value.trim() || null,
    };
    if (!setup.role) {
      err.textContent = "Tell us which role you're practising for (or tap a suggestion).";
      err.hidden = false;
      $("#role").focus();
      return;
    }
    store.set("setup", setup);
    // Start generating the first question now, while the user does the mic check.
    beginCreate({ ...setup });
    location.hash = "#/mic";
  });
}

function syncChips() {
  const v = $("#role").value.trim();
  $$("#role-chips .chip").forEach((c) => c.classList.toggle("active", c.dataset.role === v));
}

function updateLength() {
  const m = Number($("#duration").value);
  const h = m >= 60 ? `${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ""}` : `${m} min`;
  $("#duration-output").textContent = `${h} · about ${Math.max(3, Math.round(m / 3.5))} questions + follow-ups`;
  $$(".dur-val").forEach((el) => (el.textContent = `${m} mins`));
}

function beginCreate(body) {
  state.createPromise = api("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.createBody = body;
  state.createPromise.catch(() => { /* handled when awaited */ });
}

// ---------------------------------------------------------------------
// Microphone + level meter (Web Audio API)
// ---------------------------------------------------------------------
function pickMime() {
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  for (const m of options) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

function extFor(mime) {
  if (mime.includes("mp4")) return "mp4";
  if (mime.includes("ogg")) return "ogg";
  return "webm";
}

async function ensureMic() {
  if (state.stream && state.stream.active) return state.stream;
  if (!window.isSecureContext) {
    throw new Error("Microphone access needs http://localhost or https. Open the app at http://localhost:8000.");
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser can't record audio. Try Chrome, Edge, Firefox or Safari.");
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      // Noise suppression can erase soft "um"s — keep it off for verbatim capture.
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: true, channelCount: 1 },
    });
  } catch (e) {
    if (e.name === "NotAllowedError") throw new Error("Microphone access was blocked. Click the lock icon in the address bar, allow the microphone, then try again.");
    if (e.name === "NotFoundError") throw new Error("No microphone found. Plug one in and try again.");
    throw new Error(`Couldn't start the microphone (${e.name || e.message}).`);
  }
  state.mime = pickMime();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  state.audioCtx = new Ctx();
  const src = state.audioCtx.createMediaStreamSource(state.stream);
  state.analyser = state.audioCtx.createAnalyser();
  state.analyser.fftSize = 1024;
  src.connect(state.analyser);
  levelLoop();
  return state.stream;
}

function releaseMic() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  try { state.audioCtx?.close(); } catch { /* ignore */ }
  state.audioCtx = null;
  state.analyser = null;
}

/** Converts RMS to a 0..1 level on a log scale (feels natural on a meter). */
function rmsToLevel(rms) {
  const db = 20 * Math.log10(rms + 1e-8);
  return clamp((db + 60) / 50, 0, 1);
}

function levelLoop() {
  if (!state.analyser) return;
  const buf = new Float32Array(state.analyser.fftSize);
  const tick = () => {
    if (!state.analyser) return;
    state.analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    state.rms = rms;
    state.level = state.level * 0.6 + rmsToLevel(rms) * 0.4;

    // Mic-check meter
    const meter = $("#mic-meter");
    if (meter && !$("#screen-mic").hidden) meter.style.width = `${state.level * 100}%`;

    // Interview orb + live pause counter
    if (state.phase === "listening") {
      const floorLevel = rmsToLevel(state.noiseFloor);
      const above = clamp((state.level - floorLevel) / (1 - floorLevel + 1e-6), 0, 1);
      $("#orb").style.setProperty("--level", above.toFixed(3));
      trackPauses(rms);
    } else {
      $("#orb").style.setProperty("--level", "0");
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function trackPauses(rms) {
  const now = performance.now();
  const threshold = Math.max(state.noiseFloor * 2.5, 0.006);
  if (rms > threshold) {
    state.lastVoiceAt = now;
    state.hasSpoken = true;
    state.inSilence = false;
  } else if (state.hasSpoken && !state.inSilence && now - state.lastVoiceAt > 500) {
    state.inSilence = true;
    state.pauseCount += 1;
    paintMyStatus();
  }
}

/** Records until stop() is called; resolves to a Blob. */
function createRecorder() {
  const rec = new MediaRecorder(state.stream, state.mime ? { mimeType: state.mime } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const done = new Promise((resolve) => {
    rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || state.mime || "audio/webm" }));
  });
  rec.start(1000); // 1 s slices: nothing is lost if something goes wrong mid-answer
  return { rec, done, startedAt: performance.now() };
}

// ---------------------------------------------------------------------
// Interviewer voice (speechSynthesis)
// ---------------------------------------------------------------------
const synth = window.speechSynthesis || null;

function voices() {
  if (!synth) return [];
  const all = synth.getVoices();
  const en = all.filter((v) => /^en/i.test(v.lang));
  return en.length ? en : all;
}

function populateVoices() {
  const sel = $("#voice-select");
  const list = voices();
  if (!synth || !list.length) {
    $("#voice-note").hidden = !!(synth && synth.getVoices().length);
    sel.innerHTML = `<option value="">Default voice</option>`;
    return;
  }
  $("#voice-note").hidden = true;
  sel.innerHTML = list
    .map((v) => `<option value="${esc(v.voiceURI)}">${esc(v.name)} (${esc(v.lang)})</option>`)
    .join("");
  const preferred = list.find((v) => v.voiceURI === state.voiceURI)
    || list.find((v) => /natural|neural|google|samantha|aria|jenny/i.test(v.name))
    || list[0];
  sel.value = preferred.voiceURI;
  state.voiceURI = preferred.voiceURI;
}
if (synth) synth.addEventListener?.("voiceschanged", populateVoices);

function speak(text) {
  return new Promise((resolve) => {
    if (!synth || state.muted || !text) { setTimeout(resolve, 350); return; }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const v = voices().find((x) => x.voiceURI === state.voiceURI);
    if (v) u.voice = v;
    u.rate = state.rate;
    let finished = false;
    const finish = () => { if (!finished) { finished = true; clearTimeout(fallback); resolve(); } };
    // Some browsers occasionally never fire onend — fall back to an estimate.
    const est = (text.split(/\s+/).length * 430) / state.rate + 2500;
    const fallback = setTimeout(finish, est);
    u.onend = finish;
    u.onerror = finish;
    synth.speak(u);
  });
}

// ---------------------------------------------------------------------
// SCREEN 2 — Mic check
// ---------------------------------------------------------------------
let micScreenReady = false;
function initMicScreen() {
  $("#mic-error").hidden = true;
  $("#rate").value = state.rate;
  $("#rate-output").textContent = `${Number(state.rate).toFixed(2).replace(/0$/, "")}×`;
  populateVoices();
  if (state.stream?.active) markPermission();
  else {
    // If the browser already remembers permission, switch the mic on straight away.
    navigator.permissions?.query({ name: "microphone" })
      .then((p) => { if (p.state === "granted") return ensureMic().then(markPermission); })
      .catch(() => { /* not supported (e.g. Firefox) — the button still works */ });
  }
  if (micScreenReady) return;
  micScreenReady = true;

  $("#mic-allow").addEventListener("click", async () => {
    const st = $("#mic-perm-status");
    st.textContent = "Waiting for permission…";
    try {
      await ensureMic();
      markPermission();
    } catch (e) {
      st.textContent = e.message;
    }
  });

  $("#quiet-btn").addEventListener("click", measureSilence);
  $("#say-btn").addEventListener("click", recordPhrase);

  $("#voice-select").addEventListener("change", (e) => {
    state.voiceURI = e.target.value;
    store.set("voiceURI", state.voiceURI);
  });
  $("#rate").addEventListener("input", (e) => {
    state.rate = Number(e.target.value);
    $("#rate-output").textContent = `${state.rate.toFixed(2).replace(/0$/, "")}×`;
    store.set("rate", state.rate);
  });
  $("#voice-preview").addEventListener("click", () => {
    const wasMuted = state.muted;
    state.muted = false;
    speak("Hi! I'll be your interviewer today. Take your time with each answer — pauses are completely fine.");
    state.muted = wasMuted;
  });

  $("#mic-back").addEventListener("click", () => (location.hash = "#/setup"));
  $("#mic-continue").addEventListener("click", startInterview);
}

function markPermission() {
  $("#step-permission").classList.add("done");
  $("#mic-perm-status").textContent = "Microphone on ✓";
  $("#mic-allow").hidden = true;
  $("#step-quiet").setAttribute("aria-disabled", "false");
  $("#step-say").setAttribute("aria-disabled", "false");
  $("#quiet-btn").disabled = false;
  $("#say-btn").disabled = false;
  $("#mic-continue").disabled = false;
}

async function measureSilence() {
  const btn = $("#quiet-btn");
  const st = $("#quiet-status");
  btn.disabled = true;
  const samples = [];
  const t0 = performance.now();
  while (performance.now() - t0 < 3000) {
    samples.push(state.rms);
    st.textContent = `Shh… ${Math.ceil((3000 - (performance.now() - t0)) / 1000)}`;
    await sleep(50);
  }
  samples.sort((a, b) => a - b);
  // 90th percentile of the quiet room = noise floor
  state.noiseFloor = Math.max(samples[Math.floor(samples.length * 0.9)] || 0.004, 0.002);
  store.set("noiseFloor", state.noiseFloor);
  $("#mic-floor").style.left = `${rmsToLevel(state.noiseFloor) * 100}%`;
  const quiet = state.noiseFloor < 0.02;
  st.textContent = quiet ? "Noise baseline saved ✓ Nice quiet room." : "Baseline saved ✓ It's a bit noisy — a quieter spot will help accuracy.";
  $("#step-quiet").classList.add("done");
  btn.disabled = false;
  btn.textContent = "Measure again";
}

async function recordPhrase() {
  const btn = $("#say-btn");
  const out = $("#say-result");
  if (btn.dataset.recording === "1") {
    btn.dataset.stop = "1";
    return;
  }
  btn.dataset.recording = "1";
  btn.dataset.stop = "";
  btn.textContent = "■ Stop";
  out.innerHTML = `<span class="muted">Listening… say “Umm, I'm ready for my interview”.</span>`;
  const r = createRecorder();
  const t0 = performance.now();
  while (performance.now() - t0 < 7000 && btn.dataset.stop !== "1") await sleep(100);
  r.rec.stop();
  const blob = await r.done;
  btn.dataset.recording = "";
  btn.textContent = "● Record again";
  btn.disabled = true;
  out.innerHTML = `<span class="muted">Transcribing verbatim…</span>`;
  try {
    const fd = new FormData();
    fd.append("audio", blob, `mic-check.${extFor(blob.type)}`);
    const res = await api("/api/mic-check", { method: "POST", body: fd });
    const transcript = `<div class="transcript compact">${renderTokens(res.tokens)}</div>`;
    out.innerHTML = res.filler_detected
      ? `${transcript}<p class="verdict ok">✅ Filler words are being captured. <span class="muted small">(${esc(res.provider)})</span></p>`
      : `${transcript}<p class="verdict warn">⚠️ No “umm” came through. If you did say it, your speech engine is removing fillers — switch <code>STT_PROVIDER</code> (local CrisperWhisper 2.0 or Deepgram work best). <span class="muted small">Engine: ${esc(res.provider)}</span></p>`;
    $("#step-say").classList.add("done");
  } catch (e) {
    out.innerHTML = `<p class="verdict warn">⚠️ ${esc(e.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------
// SCREEN 3 — Interview
// ---------------------------------------------------------------------
async function startInterview() {
  const btn = $("#mic-continue");
  const err = $("#mic-error");
  err.hidden = true;
  btn.disabled = true;
  btn.textContent = "Preparing your first question…";
  try {
    await ensureMic();
    const res = await state.createPromise;
    state.sessionId = res.session_id;
    state.total = res.question.total;
    state.duration = res.duration_minutes || 30;
    state.sections = res.sections || [];
    state.covered = new Set();
    state.ending = false;
    state.pending = null;
    state.phase = "idle";
    $("#upload-error").hidden = true;
    $("#conversation").innerHTML = "";
    location.hash = "#/interview";
    // Start the server-side clock (time-boxed screen); the local timer mirrors it.
    api(`/api/sessions/${state.sessionId}/start`, { method: "POST" }).catch(() => {});
    startTimer();
    setupLivePreview();
    askQuestion(res.question);
  } catch (e) {
    err.textContent = `${e.message} `;
    const retry = document.createElement("button");
    retry.className = "btn btn-sm btn-secondary";
    retry.textContent = "Try again";
    retry.onclick = () => { beginCreate(state.createBody); startInterview(); };
    err.appendChild(retry);
    err.hidden = false;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Begin interview <span aria-hidden="true">→</span>';
  }
}

function startTimer() {
  clearInterval(state.timerId);
  state.timerStart = Date.now();
  const total = (state.duration || 30) * 60;
  const paint = () => {
    const el = (Date.now() - state.timerStart) / 1000;
    const t = $("#timer");
    t.textContent = el < total ? `${fmtTime(total - el)} left` : `+${fmtTime(el - total)} over`;
    t.classList.toggle("warn", el > total - 120);
  };
  paint();
  state.timerId = setInterval(paint, 1000);
}

function setOrb(phase, label) {
  $("#orb").dataset.state = phase;
  $("#orb-label").textContent = label;
}

function setControls() {
  const p = state.phase;
  $("#done-btn").disabled = p !== "listening";
  $("#repeat-btn").disabled = p !== "listening";
  $("#skip-btn").disabled = !(p === "listening" || p === "speaking");
  const mb = $("#mute-btn");
  mb.setAttribute("aria-pressed", String(state.muted));
  $(".ctrl-icon", mb).textContent = state.muted ? "🔇" : "🔊";
}

function renderProgress(q) {
  $("#q-label").textContent = `Question ${q.number}`;
  $("#q-section").textContent = (state.sections || []).find((s) => s.key === q.section)?.name || "";
  if (q.section) state.covered?.add(q.section);
  // One dot per competency section: filled once the interviewer has covered it.
  $("#q-dots").innerHTML = (state.sections || []).map((s) =>
    `<i class="${s.key === q.section ? "current" : state.covered?.has(s.key) ? "done" : ""}" title="${esc(s.name)}"></i>`).join("");
}

async function askQuestion(q) {
  const token = ++state.askToken;
  state.question = q;
  renderProgress(q);
  $("#followup-tag").hidden = q.kind !== "followup";
  $("#question-text").textContent = q.text;
  addAiTurn(q.text, { number: q.number, followup: q.kind === "followup" });
  state.phase = "speaking";
  setOrb("speaking", state.muted ? "New question" : "Interviewer speaking…");
  setControls();
  await speak(q.text);
  if (token !== state.askToken || state.ending || state.phase !== "speaking") return;
  startListening();
}

function startListening() {
  state.chunks = [];
  state.current = createRecorder();
  state.pauseCount = 0;
  state.hasSpoken = false;
  state.inSilence = false;
  state.lastVoiceAt = performance.now();
  state.phase = "listening";
  startMyTurn(state.question.id);
  setOrb("listening", "Listening… press Done when you're finished");
  setControls();
  startLive();
}

async function stopListening() {
  stopLive();
  const cur = state.current;
  state.current = null;
  if (!cur) return null;
  if (cur.rec.state !== "inactive") cur.rec.stop();
  const blob = await cur.done;
  return { blob, seconds: (performance.now() - cur.startedAt) / 1000 };
}

async function onDone() {
  if (state.phase !== "listening") return;
  state.phase = "uploading";
  setControls();
  const r = await stopListening();
  await submitAnswer({ questionId: state.question.id, blob: r?.blob || null, skipped: !r?.blob, endAfter: false });
}

async function onSkip() {
  if (state.phase === "speaking") {
    state.askToken++;
    synth?.cancel();
  } else if (state.phase === "listening") {
    await stopListening(); // discard
  } else {
    return;
  }
  state.phase = "uploading";
  setControls();
  await submitAnswer({ questionId: state.question.id, blob: null, skipped: true, endAfter: false });
}

async function onRepeat() {
  if (state.phase !== "listening" || !state.current) return;
  // Pause the recording while the question is re-read, then carry on.
  state.phase = "speaking";
  setControls();
  stopLive();
  try { state.current.rec.pause(); } catch { /* ignore */ }
  setOrb("speaking", "Repeating the question…");
  const wasMuted = state.muted;
  state.muted = false;
  await speak(state.question.text);
  state.muted = wasMuted;
  if (state.ending || !state.current) return;
  try { state.current.rec.resume(); } catch { /* ignore */ }
  state.phase = "listening";
  state.lastVoiceAt = performance.now();
  setOrb("listening", "Listening… press Done when you're finished");
  setControls();
  startLive();
}

function onMute() {
  state.muted = !state.muted;
  store.set("muted", state.muted);
  if (state.muted) synth?.cancel();
  setControls();
  toast(state.muted ? "Interviewer voice muted — questions appear as text" : "Interviewer voice on");
}

async function submitAnswer(item) {
  state.pending = item;
  $("#upload-error").hidden = true;
  if (!item.skipped) setMyTurnState(item.questionId, "processing", "Transcribing word-for-word…");
  setOrb("thinking", item.skipped ? "Moving on…" : "Transcribing your answer…");
  const fd = new FormData();
  fd.append("question_id", String(item.questionId));
  fd.append("skipped", String(!!item.skipped));
  fd.append("end_after", String(!!item.endAfter));
  if (item.blob && !item.skipped) fd.append("audio", item.blob, `answer.${extFor(item.blob.type)}`);
  try {
    const res = await api(`/api/sessions/${state.sessionId}/answer`, { method: "POST", body: fd });
    state.pending = null;
    finishMyTurn(item.questionId, res);
    if (!res.skipped && res.verbatim_tokens.length) {
      setOrb("thinking", "Got it — here's your transcript");
      await sleep(1400);
    } else if (!res.skipped) {
      toast("No speech detected in that answer.");
    }
    await handleNext(res.next);
  } catch (e) {
    state.phase = "error";
    setControls();
    setOrb("idle", "Upload paused");
    $("#upload-error-text").textContent = `Couldn't process your answer: ${e.message}`;
    const dl = $("#download-blob");
    if (item.blob) {
      dl.hidden = false;
      dl.href = URL.createObjectURL(item.blob);
      dl.download = `answer-q${item.questionId}.${extFor(item.blob.type)}`;
    } else {
      dl.hidden = true;
    }
    $("#upload-error").hidden = false;
    setMyTurnState(item.questionId, "error", "Not uploaded yet — press Retry");
  }
}

async function handleNext(next) {
  if (state.ending || next.type === "end") {
    if (next.type === "end" && next.text && !state.ending) {
      $("#followup-tag").hidden = true;
      $("#question-text").textContent = next.text;
      addAiTurn(next.text, {});
      setOrb("speaking", "Wrapping up…");
      state.phase = "done";
      setControls();
      await speak(next.text);
    }
    finishInterview();
    return;
  }
  askQuestion(next.question);
}

async function onEnd() {
  const ok = await confirmDialog();
  if (!ok) return;
  state.ending = true;
  state.askToken++;
  synth?.cancel();
  if (state.phase === "listening") {
    state.phase = "uploading";
    setControls();
    const r = await stopListening();
    // Keep a partial answer only if there was real speech in it.
    if (r && r.blob && r.seconds > 2 && state.hasSpoken) {
      await submitAnswer({ questionId: state.question.id, blob: r.blob, skipped: false, endAfter: true });
      return; // handleNext() finishes the interview after the upload
    }
  }
  if (state.phase === "uploading" || state.phase === "error") {
    if (state.phase === "error") { state.pending = null; finishInterview(); }
    return; // in-flight upload will call handleNext → finish
  }
  finishInterview();
}

function finishInterview() {
  state.phase = "done";
  clearInterval(state.timerId);
  stopLive();
  synth?.cancel();
  releaseMic();
  location.hash = "#/loading";
  generateFeedback(state.sessionId);
}

function confirmDialog() {
  const dlg = $("#confirm-dialog");
  return new Promise((resolve) => {
    if (typeof dlg.showModal !== "function") { resolve(window.confirm("End the interview and get feedback?")); return; }
    dlg.returnValue = "";
    dlg.showModal();
    dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
  });
}

// Buttons + keyboard shortcuts
$("#done-btn").addEventListener("click", onDone);
$("#skip-btn").addEventListener("click", onSkip);
$("#repeat-btn").addEventListener("click", onRepeat);
$("#mute-btn").addEventListener("click", onMute);
$("#end-btn").addEventListener("click", onEnd);
$("#retry-upload").addEventListener("click", () => { if (state.pending) { state.phase = "uploading"; submitAnswer(state.pending); } });

document.addEventListener("keydown", (e) => {
  if ($("#screen-interview").hidden || $("#confirm-dialog").open) return;
  if (e.target.closest("input, textarea, select")) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === " " || k === "spacebar") { e.preventDefault(); onDone(); }
  else if (k === "r") onRepeat();
  else if (k === "s") onSkip();
  else if (k === "m") onMute();
  else if (k === "t") toggleTranscript();
});

$("#transcript-btn").addEventListener("click", () => toggleTranscript());
function toggleTranscript(force) {
  const panel = $("#transcript-panel");
  const showIt = force ?? panel.hidden;
  panel.hidden = !showIt;
  $("#screen-interview").classList.toggle("no-transcript", !showIt);
  $("#transcript-btn").setAttribute("aria-pressed", String(showIt));
  store.set("showTranscript", showIt);
}
toggleTranscript(store.get("showTranscript", true));

// ---------------------------------------------------------------------
// Conversation transcript (Interviewer / You turns, verbatim)
// ---------------------------------------------------------------------
function stamp() {
  return state.timerStart ? fmtTime((Date.now() - state.timerStart) / 1000) : "00:00";
}

function scrollConversation() {
  const c = $("#conversation");
  c.scrollTop = c.scrollHeight;
}

function addAiTurn(text, { number, followup } = {}) {
  const li = document.createElement("li");
  li.className = "turn ai";
  li.innerHTML = `<div class="turn-meta"><b>Interviewer</b>${number ? `<span>Q${number}${followup ? " · follow-up" : ""}</span>` : ""}<time>${stamp()}</time></div>
    <p class="turn-text">${esc(text)}</p>`;
  $("#conversation").appendChild(li);
  scrollConversation();
}

function myTurn(qid) { return $(`#conversation .turn.me[data-qid="${qid}"]`); }

function startMyTurn(qid) {
  let li = myTurn(qid);
  if (!li) {
    li = document.createElement("li");
    li.className = "turn me";
    li.dataset.qid = qid;
    li.innerHTML = `<div class="turn-meta"><b>You</b><span class="turn-status"></span><time>${stamp()}</time></div>
      <div class="turn-body"><p class="live-cap" aria-live="polite"></p></div>`;
    $("#conversation").appendChild(li);
  }
  li.dataset.state = "listening";
  state.liveBase = $(".live-cap", li)?.textContent || "";
  paintMyStatus();
  scrollConversation();
}

function paintMyStatus() {
  const li = state.question && myTurn(state.question.id);
  if (!li || li.dataset.state !== "listening") return;
  const p = state.pauseCount;
  const cap = $(".live-cap", li);
  const captionsOn = SR && $("#live-toggle").checked;
  if (cap && !cap.textContent) cap.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>${captionsOn ? "" : ' <span class="muted small">Recording your answer…</span>'}`;
  $(".turn-status", li).innerHTML = `<span class="rec-dot small"></span> Speaking${p ? ` · ⏸ ${p} pause${p === 1 ? "" : "s"}` : ""}`;
}

function setMyTurnState(qid, st, label) {
  const li = myTurn(qid);
  if (!li) return;
  li.dataset.state = st;
  $(".turn-status", li).textContent = label;
}

function finishMyTurn(qid, res) {
  const li = myTurn(qid) || (startMyTurn(qid), myTurn(qid));
  li.dataset.state = res.skipped ? "skipped" : "done";
  const body = $(".turn-body", li);
  if (res.skipped) {
    $(".turn-status", li).textContent = "Skipped";
    body.innerHTML = `<p class="muted">Skipped this question.</p>`;
  } else if (!res.verbatim_tokens.length) {
    $(".turn-status", li).textContent = "No speech detected";
    body.innerHTML = `<p class="muted">We couldn't hear anything in that answer.</p>`;
  } else {
    const st = res.stats || {};
    $(".turn-status", li).textContent = "Verbatim";
    const chips = st.fillers ? [
      `${st.fillers.total} filler${st.fillers.total === 1 ? "" : "s"}`,
      st.stammers ? `${st.stammers} stammer${st.stammers === 1 ? "" : "s"}` : "",
      st.repetitions ? `${st.repetitions} repeat${st.repetitions === 1 ? "" : "s"}` : "",
      `${st.pauses.count} pause${st.pauses.count === 1 ? "" : "s"}`,
      st.wpm ? `${st.wpm} WPM` : "",
    ].filter(Boolean) : [];
    body.innerHTML = `<div class="transcript">${renderTokens(res.verbatim_tokens)}</div>
      ${chips.length ? `<div class="turn-stats">${chips.map((c) => `<span>${esc(c)}</span>`).join("")}</div>` : ""}`;
  }
  scrollConversation();
}

// ---------------------------------------------------------------------
// Optional live preview (Web Speech API — approximate, not verbatim)
// ---------------------------------------------------------------------
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let live = null;

function setupLivePreview() {
  const toggle = $("#live-toggle");
  if (!SR) {
    $("#live-switch").hidden = true;
    $("#live-note").textContent = "Your exact word-for-word transcript appears here after you press Done.";
    return;
  }
  // On by default where supported; the exact transcript always replaces it after Done.
  toggle.checked = store.get("livePreview", true);
  toggle.onchange = () => {
    store.set("livePreview", toggle.checked);
    if (toggle.checked && state.phase === "listening") startLive();
    else stopLive();
  };
}

function startLive() {
  if (!SR || !$("#live-toggle").checked) return;
  stopLive();
  const rec = new SR();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "en-US";
  let finalText = state.liveBase ? state.liveBase + " " : "";
  rec.onresult = (ev) => {
    let interim = "";
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const r = ev.results[i];
      if (r.isFinal) finalText += r[0].transcript + " ";
      else interim += r[0].transcript;
    }
    const li = state.question && myTurn(state.question.id);
    const el = li && $(".live-cap", li);
    if (!el) return;
    el.textContent = (finalText + interim).trim();
    state.liveBase = finalText.trim();
    scrollConversation();
  };
  rec.onend = () => { if (live === rec && state.phase === "listening") { try { rec.start(); } catch { /* ignore */ } } };
  rec.onerror = () => { /* preview is best-effort */ };
  try { rec.start(); live = rec; } catch { live = null; }
}

function stopLive() {
  const rec = live;
  live = null;
  try { rec?.stop(); } catch { /* ignore */ }
}

// ---------------------------------------------------------------------
// Feedback generation
// ---------------------------------------------------------------------
async function generateFeedback(id, regenerate = false) {
  show("loading");
  $("#loading-error").hidden = true;
  $("#loading-actions").hidden = true;
  try {
    const detail = await api(`/api/sessions/${id}/feedback${regenerate ? "?regenerate=true" : ""}`, { method: "POST" });
    state.detailCache[id] = detail;
    state.sessionId = null;
    if (location.hash === `#/report/${id}`) route();
    else location.hash = `#/report/${id}`;
  } catch (e) {
    $("#loading-error").textContent = e.message;
    $("#loading-error").hidden = false;
    $("#loading-actions").hidden = false;
    $("#loading-retry").hidden = e.status === 400;
    $("#loading-retry").onclick = () => generateFeedback(id, regenerate);
  }
}

// ---------------------------------------------------------------------
// Transcript rendering
// ---------------------------------------------------------------------
const KIND_TITLE = {
  filler: "Filler", stammer: "Stammer", repetition: "Repetition", false_start: "False start", event: "Non-speech sound",
};

function renderTokens(tokens) {
  return (tokens || []).map((t) => {
    if (t.kind === "pause") {
      return `<span class="pause-pill ${esc(t.category)}" data-s="${t.start}" data-e="${t.end}" title="Pause ${t.duration}s (${esc(t.category.replace("_", " "))})">⏸ ${Number(t.duration).toFixed(1)}s</span>`;
    }
    const cls = ["tok"];
    if (t.kind !== "word") cls.push(t.kind);
    if (t.confidence != null && t.confidence < 0.5) cls.push("low-conf");
    let clean = "";
    if (t.kind === "stammer") {
      const raw = t.text.trim();
      clean = raw.endsWith("-") ? "" : raw.split(/[-‐]/).pop();
    }
    const title = KIND_TITLE[t.kind] ? ` title="${KIND_TITLE[t.kind]} · ${Number(t.start).toFixed(1)}s"` : ` title="${Number(t.start).toFixed(1)}s"`;
    const cleanAttr = t.kind === "stammer" ? ` data-clean="${esc(clean)}" data-raw="${esc(t.text)}"` : "";
    return `<span class="${cls.join(" ")}" data-s="${t.start}" data-e="${t.end}"${title}${cleanAttr}>${esc(t.text)}</span>`;
  }).join(" ");
}

/** Verbatim text with explicit markers — mirrors backend disfluency.to_marked_text(). */
function markedText(tokens) {
  const out = [];
  let i = 0;
  while (i < tokens.length) {
    const t = tokens[i];
    let run = [t];
    if (t.group) {
      while (i + run.length < tokens.length && tokens[i + run.length].group === t.group) run.push(tokens[i + run.length]);
    }
    const text = run.map((x) => x.text).join(" ");
    switch (t.kind) {
      case "pause": out.push(`[pause ${Number(t.duration).toFixed(1)}s]`); break;
      case "filler": out.push(`[${text}]`); break;
      case "stammer": out.push(`[stammer: ${text}]`); break;
      case "repetition": out.push(`[repetition: ${text}]`); break;
      case "false_start": out.push(`[false start: ${text}]`); break;
      default: out.push(text);
    }
    i += run.length;
  }
  return out.join(" ");
}

function legendHTML() {
  return `<div class="legend" aria-label="Legend">
    <span><span class="tok filler">um</span> filler</span>
    <span><span class="tok stammer">I-I</span> stammer</span>
    <span><span class="tok repetition">the</span> repetition</span>
    <span><span class="tok false_start">I was—</span> false start</span>
    <span><span class="pause-pill">⏸ 0.8s</span><span class="pause-pill long">⏸ 2.1s</span><span class="pause-pill very_long">⏸ 3.6s</span> pauses</span>
  </div>`;
}

// ---------------------------------------------------------------------
// Charts (inline SVG)
// ---------------------------------------------------------------------
function gaugeSVG(score) {
  const s = clamp(Number(score) || 0, 0, 10);
  const color = s >= 7.5 ? "var(--good)" : s >= 5 ? "var(--accent)" : "var(--warn)";
  return `<svg class="gauge" viewBox="0 0 200 120" role="img" aria-label="Overall score ${s} out of 10">
    <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="var(--surface-2)" stroke-width="16" stroke-linecap="round"/>
    <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round"
      pathLength="100" stroke-dasharray="${s * 10} 100"/>
    <text x="100" y="92" text-anchor="middle" font-family="Instrument Serif, serif" font-size="44" fill="var(--text)">${s.toFixed(1)}</text>
    <text x="100" y="114" text-anchor="middle" font-size="12" fill="var(--muted)">out of 10</text>
  </svg>`;
}

/**
 * series: [{values:[number|null], cls:"overall"|"", name}]
 * labels: x-axis labels
 */
function lineChart(labels, series, { height = 190, yMax = 10 } = {}) {
  const W = 640, H = height, L = 30, R = 14, T = 12, B = 28;
  const n = labels.length;
  const x = (i) => (n <= 1 ? L + (W - L - R) / 2 : L + (i * (W - L - R)) / (n - 1));
  const y = (v) => T + (1 - v / yMax) * (H - T - B);
  let g = `<g class="grid">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1].map((f) => f * yMax)) {
    g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  }
  g += `</g>`;
  const step = Math.max(1, Math.ceil(n / 8));
  const xl = labels.map((lab, i) => (i % step === 0 || i === n - 1)
    ? `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">${esc(lab)}</text>` : "").join("");
  const lines = series.map((s) => {
    const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v), v])).filter(Boolean);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
    const dots = pts.map((p) => `<circle class="pt ${s.cls}" cx="${p[0]}" cy="${p[1]}" r="4"><title>${esc(s.name)}: ${p[2]}</title></circle>`).join("");
    return `<path class="line ${s.cls}" d="${d}"/>${dots}`;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Line chart">${g}${xl}${lines}</svg></div>`;
}

// ---------------------------------------------------------------------
// SCREEN 4 — Report  (Overall /100 · 4 competencies · detailed breakdown)
// ---------------------------------------------------------------------
const REPORT_TITLE = {
  technical: "Technical Screen Feedback",
  behavioural: "HR & Behavioural Screen Feedback",
  mixed: "Screening Interview Feedback",
};
const bandCls = (b) => `b-${String(b || "none").toLowerCase()}`;

async function openReport(id) {
  const root = $("#report-root");
  let detail = state.detailCache[id];
  if (!detail || !detail.report) {
    root.innerHTML = `<p class="muted">Loading…</p>`;
    try {
      detail = await api(`/api/sessions/${id}`);
      state.detailCache[id] = detail;
    } catch (e) {
      root.innerHTML = `<p class="form-error">${esc(e.message)}</p>`;
      return;
    }
  }
  if (!detail.report || detail.report.version !== 2) {
    const legacy = !!detail.report;
    root.innerHTML = `
      <p class="eyebrow">${esc(detail.role)}</p>
      <h2 class="title">${legacy ? "This report uses the old format" : "No report yet"}</h2>
      <p class="muted">${legacy
        ? "Regenerate it to get the new breakdown: overall score /100, four competencies, evidence and strong answers."
        : "This interview was stopped before feedback was generated."}</p>
      <div class="row-gap"><button class="btn btn-primary" id="gen-now">${legacy ? "Regenerate report" : "Generate feedback"}</button><a class="btn btn-ghost" href="#/history">Back</a></div>`;
    $("#gen-now").onclick = () => generateFeedback(id, legacy);
    return;
  }
  renderReport(detail);
}

function statTile(label, value, sub = "", cls = "") {
  return `<div class="stat ${cls}"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div>${sub ? `<div class="stat-sub">${esc(sub)}</div>` : ""}</div>`;
}

const ICON = {
  check: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="var(--ok-soft)"/><path d="M5 8.2l2 2 4-4.2" fill="none" stroke="var(--ok)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  up: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="var(--up-soft)"/><path d="M8 11V5.2M5.6 7.4L8 5l2.4 2.4" fill="none" stroke="var(--up)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  play: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6.5 4.8v10.4L15 10z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
  pause: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5v10M13 5v10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
  transcript: '<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="4" width="11" height="9" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M6 13v3l3-3M17 7v6a2 2 0 0 1-2 2" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M6 7.5h5M6 10h3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
  download: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3.5v9M6.5 9.2L10 12.7l3.5-3.5M4 15.5h12" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  back: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M13 8H3.5M7.5 4L3.5 8l4 4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

function scoreHTML(score, band, size = "") {
  return `<div class="score ${size}">
    <span class="score-num">${score ?? "—"}</span><span class="score-den">/100</span>
    <span class="band ${bandCls(band)}">${esc(band || "Not assessed")}</span>
  </div>`;
}

/** Semicircle gauge: Weak 0–30 · Moderate 30–50 · Good 50–75 · Excellent 75–100, needle at the score. */
function gaugeHTML(score, band) {
  const cx = 160, cy = 160, r = 118, w = 30;
  const pt = (v, rad = r) => {
    const t = Math.PI * (1 - v / 100);
    return [cx + rad * Math.cos(t), cy - rad * Math.sin(t)];
  };
  const arc = (v1, v2, rad = r) => {
    const [x1, y1] = pt(v1, rad), [x2, y2] = pt(v2, rad);
    return `M${x1.toFixed(1)} ${y1.toFixed(1)} A${rad} ${rad} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
  };
  const segs = [["Weak", 0, 30, "#e0565b", "#e9875f"], ["Moderate", 30, 50, "#e6a46a", "#e8d07c"],
                ["Good", 50, 75, "#d7dd78", "#a7cf6c"], ["Excellent", 75, 100, "#93c66b", "#55a35a"]];
  const gap = 1.1;
  const defs = segs.map(([n, a, b, c1, c2]) => {
    const [x1, y1] = pt(a), [x2, y2] = pt(b);
    return `<linearGradient id="g-${n}" gradientUnits="userSpaceOnUse" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient>`;
  }).join("");
  const arcs = segs.map(([n, a, b]) =>
    `<path d="${arc(a + (a ? gap : 0), b - (b < 100 ? gap : 0))}" stroke="url(#g-${n})" stroke-width="${w}" fill="none" stroke-linecap="${a === 0 || b === 100 ? "round" : "butt"}"/>`).join("");
  const labels = segs.map(([n, a, b]) =>
    `<path id="lp-${n}" d="${arc(a, b, r - w / 2 - 12)}" fill="none"/><text class="g-label"><textPath href="#lp-${n}" startOffset="50%" text-anchor="middle">${n}</textPath></text>`).join("");
  let needle = "";
  if (score != null) {
    const [x1, y1] = pt(clamp(score, 0.5, 99.5), r - w / 2 - 4), [x2, y2] = pt(clamp(score, 0.5, 99.5), r + w / 2 + 4);
    needle = `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--surface)" stroke-width="9" stroke-linecap="round"/>
      <line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--text)" stroke-width="5" stroke-linecap="round"/>`;
  }
  return `<div class="gauge-wrap">
    <svg class="gauge2" viewBox="0 0 320 178" role="img" aria-label="Overall score ${score ?? "not available"} out of 100, ${esc(band || "")}">
      <defs>${defs}</defs>${arcs}${labels}${needle}
    </svg>
    <div class="gauge-center"><div><span class="g-num">${score ?? "—"}</span><span class="g-den">/100</span></div>
      <span class="band ${bandCls(band)}">${esc(band || "Not assessed")}</span></div>
  </div>`;
}

function seenIn(list) {
  if (!list || !list.length) return "";
  return `<div class="seen-in"><span>Seen in:</span>${list.map((n) => `<button type="button" class="q-chip" data-goto="${n}">Question ${n}</button>`).join("")}</div>`;
}

function itemsHTML(items, cls) {
  return (items || []).map((it) => `
    <div class="fb-item ${cls}">
      <span class="fb-ico">${cls === "good" ? ICON.check : ICON.up}</span>
      <div>
        <h4>${esc(it.title)}</h4>
        ${it.detail ? `<p>${esc(it.detail)}</p>` : ""}
        ${seenIn(it.seen_in)}
      </div>
    </div>`).join("");
}

function iconList(items, kind) {
  return `<ul class="ico-list">${(items || []).map((t) => `<li><span class="fb-ico">${kind === "good" ? ICON.check : ICON.up}</span><span>${esc(t)}</span></li>`).join("")}</ul>`;
}

function evidenceHTML(c) {
  if (!c.expectations?.length) return "";
  const lv = (e) => `<span class="lvl lvl-${e.level}">${esc(e.label)}</span>`;
  return `<details class="evidence">
    <summary>How this score was calculated</summary>
    <p class="muted small">Each expectation is scored 3 Met · 2 Partially met · 1 Not met · 0 Not assessed, and every score
      must quote your exact words. Score = average of assessed levels ÷ 3 × 100${c.key === "communication" && c.delivery_score != null
        ? `, blended 70/30 with your measured delivery score (rubric ${c.rubric_score ?? "—"} · delivery ${c.delivery_score})` : ""}.</p>
    <ul class="exp-list">${c.expectations.map((e) => `
      <li>
        <div class="exp-row"><span>${esc(e.text)}</span>${lv(e)}</div>
        ${(e.evidence || []).map((ev) => `<blockquote>“${esc(ev.quote)}” <button type="button" class="q-chip" data-goto="${ev.question}">Question ${ev.question}</button></blockquote>`).join("")}
      </li>`).join("")}</ul>
  </details>`;
}

function deliveryHTML(d) {
  const ss = d.report.session_stats || {};
  const top = ss.fillers?.top || [];
  const maxTop = Math.max(1, ...top.map((t) => t.count));
  const topBars = top.length
    ? top.map((t) => `<div class="hbar"><span>${esc(t.word)}</span><div class="bar-track"><div class="bar-fill" style="width:${(t.count / maxTop) * 100}%"></div></div><b>${t.count}</b></div>`).join("")
    : `<p class="muted">No fillers detected.</p>`;
  const wpm = ss.wpm || 0;
  const [lo, hi] = ss.ideal_wpm || [120, 160];
  const pos = (v) => clamp((v - 60) / (220 - 60), 0, 1) * 100;
  const answered = d.questions.filter((q) => q.answer && !q.answer.skipped && q.answer.stats?.fluency_score != null);
  const trend = answered.length > 1
    ? lineChart(answered.map((q) => `Q${q.number}`), [{ name: "Delivery", cls: "", values: answered.map((q) => q.answer.stats.fluency_score) }])
    : "";
  return `
    <h3 class="sub-title">Delivery, measured from your audio</h3>
    <div class="grid-3">
      ${statTile("Fillers", ss.fillers?.total ?? 0, `${ss.fillers?.per_minute ?? 0} per minute`, "filler")}
      ${statTile("Stammers", ss.stammers ?? 0, "", "stammer")}
      ${statTile("Repetitions", ss.repetitions ?? 0, "", "repetition")}
      ${statTile("False starts", ss.false_starts ?? 0, "", "false_start")}
      ${statTile("Pauses", ss.pauses?.count ?? 0, `longest ${ss.pauses?.longest ?? 0}s · avg ${ss.pauses?.average ?? 0}s`)}
      ${statTile("Silent time", `${ss.pauses?.silent_pct ?? 0}%`, "of speaking time")}
      ${statTile("Thinking time", `${ss.avg_thinking_time ?? 0}s`, "avg before first word")}
      ${statTile("Pace", `${wpm} WPM`, `ideal ${lo}–${hi}, fillers excluded`)}
      ${statTile("Delivery score", ss.fluency_score != null ? `${Math.round(ss.fluency_score * 10)}/100` : "—", "computed, not AI-judged")}
    </div>
    <div class="grid-2" style="margin-top:12px">
      <div class="card"><h4 class="card-title">Top fillers</h4>${topBars}</div>
      <div class="card"><h4 class="card-title">Speaking pace</h4>
        <div class="wpm-track"><div class="wpm-band" style="left:${pos(lo)}%;width:${pos(hi) - pos(lo)}%"></div>${wpm ? `<div class="wpm-marker" style="left:${pos(wpm)}%"></div>` : ""}</div>
        <div class="wpm-scale"><span>60</span><span>ideal ${lo}–${hi}</span><span>220</span></div>
      </div>
    </div>
    ${trend ? `<div class="card" style="margin-top:12px"><h4 class="card-title">Delivery across questions (/10)</h4>${trend}</div>` : ""}`;
}

function questionCardHTML(d, qc) {
  const q = d.questions.find((x) => x.number === qc.question);
  if (!q) return "";
  const a = q.answer;
  const st = a?.stats || {};
  const good = qc.rating === "Good" || qc.rating === "Excellent";
  const statLine = a && st.fillers ? `<p class="muted small">${st.fillers.total} fillers · ${st.stammers} stammers · ${st.repetitions} repetitions · ${st.false_starts} false starts · ${st.pauses.count} pauses (longest ${st.pauses.longest}s) · ${st.wpm} WPM · ${st.thinking_time}s before first word</p>` : "";
  return `
  <article class="qa" id="qa-${q.number}" data-qid="${q.id}">
    <div class="qa-head">
      <span class="qnum">Q${q.number}</span>
      <p class="qa-q">${q.kind === "followup" ? '<span class="tag">Follow-up</span> ' : ""}${esc(q.text)}</p>
    </div>
    <div class="qa-row"><span class="qa-label">Your Answer:</span><span class="band ${bandCls(qc.rating)}">${esc(qc.rating)}</span></div>
    ${a?.has_audio ? `<audio class="qa-audio" controls preload="none" src="/api/audio/${d.id}/${q.id}"></audio>` : ""}
    <div class="toolbar">${legendHTML()}<label class="switch"><input type="checkbox" class="clean-toggle" /> <span>Show clean version</span></label></div>
    <div class="transcript">${a?.tokens?.length ? renderTokens(a.tokens) : '<span class="muted">No speech detected.</span>'}</div>
    ${statLine}
    ${qc.verdict ? `<p class="qa-label">${good ? "Why this worked:" : "Why this needs improvement:"}</p><p>${esc(qc.verdict)}</p>` : ""}
    ${qc.strong_answer ? `<p class="qa-label">Strong Answer:</p><p class="strong-answer">${esc(qc.strong_answer)}</p>` : ""}
  </article>`;
}

function renderReport(d) {
  const r = d.report;
  const mins = Math.max(1, Math.round((d.session_seconds || d.duration_minutes * 60) / 60));
  const comps = r.competencies || [];
  const rec = d.recording;
  const date = new Date(d.created_at).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });

  $("#report-root").innerHTML = `
    <a class="back-link no-print" href="#/history">${ICON.back} Back</a>

    <article class="card report-main">
      <h1 class="report-title">${esc(REPORT_TITLE[d.interview_type] || "Interview Feedback")}</h1>
      <p class="report-meta">${esc(date)}<i>•</i>${mins} min session<i>•</i>${esc(d.role)}</p>

      <div class="score-grid">
        <div class="score-box overall-box">
          <h3>Overall Score</h3>
          ${gaugeHTML(r.overall_score, r.overall_band)}
        </div>
        <div class="comp-tiles">
          ${comps.map((c) => `<button type="button" class="score-box comp-tile" data-tab="${c.key}">
              <h3>${esc(c.name)}</h3>
              <div class="tile-foot"><span><b>${c.score ?? "—"}</b> <small>/100</small></span><span class="band ${bandCls(c.band)}">${esc(c.band || "Not assessed")}</span></div>
            </button>`).join("")}
        </div>
      </div>

      <h2 class="h-sec">Overall Performance</h2>
      <p class="prose">${esc(r.overall_summary)}</p>

      <div class="grid-2 lists">
        <div><h2 class="h-sec">What You Are Doing Well</h2>${iconList(r.doing_well, "good")}</div>
        <div><h2 class="h-sec">Top Things You Can Improve</h2>${iconList(r.improve, "up")}</div>
      </div>
    </article>

    ${rec ? `
    <div class="card player no-print">
      <span class="player-label">Interview Recording</span>
      <span class="vsep" aria-hidden="true"></span>
      <button type="button" class="icon-btn" id="pl-play" aria-label="Play">${ICON.play}</button>
      <input type="range" id="pl-seek" class="pl-seek" min="0" max="1000" value="0" aria-label="Seek" />
      <span class="pl-time" id="pl-time">00:00 / 00:00</span>
      <select id="session-speed" class="pl-speed" aria-label="Playback speed">${[0.75, 1, 1.25, 1.5, 2].map((v) => `<option value="${v}" ${v === 1 ? "selected" : ""}>${v}x</option>`).join("")}</select>
      <button type="button" class="icon-btn" id="pl-transcript" aria-pressed="true" title="Show transcript">${ICON.transcript}</button>
      <a class="icon-btn" href="${esc(rec.url)}" download="interview-${d.id}.m4a" title="Download recording">${ICON.download}</a>
      <audio id="session-audio" preload="metadata" src="${esc(rec.url)}"></audio>
    </div>
    <div class="card synced-card no-print" id="synced-card">
      <div class="synced-head">
        <div class="chapters">${(rec.chapters || []).map((c) => `<button type="button" class="q-chip" data-seek="${c.start}">Q${c.question} · ${fmtTime(c.start)}</button>`).join("")}</div>
        <label class="switch"><input type="checkbox" id="follow-toggle" checked /> <span>Follow along</span></label>
      </div>
      <ol id="synced" class="conversation synced-list">${syncedTranscriptHTML(d)}</ol>
      <p class="muted small">Your answers are played back to back; the interviewer's voice isn't recorded. Click any word or timestamp to jump there.</p>
    </div>` : ""}

    <h2 class="h-page">Detailed Breakdown</h2>
    <div class="card breakdown">
      <div class="tabs no-print" role="tablist">
        ${comps.map((c, i) => `<button type="button" role="tab" class="tab" data-tab="${c.key}" aria-selected="${i === 0}">${esc(c.name)}</button>`).join("")}
      </div>
      ${comps.map((c, i) => `
      <div class="tab-panel" role="tabpanel" data-panel="${c.key}" ${i === 0 ? "" : "hidden"}>
        <div class="panel-head"><h3 class="panel-title">${esc(c.name)}</h3>
          <div class="panel-score"><b>${c.score ?? "—"}</b><small>/100</small><span class="band ${bandCls(c.band)}">${esc(c.band || "Not assessed")}</span></div></div>
        <p class="prose">${esc(c.description)} ${esc(c.assessment)}</p>
        ${c.improvements?.length ? `<h4 class="h-sub">Where You Can Improve</h4>${itemsHTML(c.improvements, "improve")}` : ""}
        ${c.strengths?.length ? `<h4 class="h-sub">What You Are Doing Well</h4>${itemsHTML(c.strengths, "good")}` : ""}
        ${evidenceHTML(c)}
        ${c.key === "communication" ? deliveryHTML(d) : ""}
        ${(c.questions || []).map((qc) => questionCardHTML(d, qc)).join("")}
      </div>`).join("")}
    </div>

    <div class="actions">
      <button class="btn btn-primary" id="act-retry">Retry same questions</button>
      <a class="btn btn-secondary" href="#/setup">New interview</a>
      <button class="btn btn-ghost" id="act-pdf">Download PDF</button>
      <button class="btn btn-ghost" id="act-csv">Download CSV</button>
      <button class="btn btn-ghost" id="act-txt">Download verbatim .txt</button>
      <button class="btn btn-ghost" id="act-json">Download JSON</button>
    </div>
    <p class="muted small" style="margin-top:12px">Generated with ${esc(r.generated_with?.model || "")} · speech: ${esc(r.generated_with?.stt || "")}. Scores are computed from evidence-backed expectation levels; quotes that couldn't be found in your transcript were discarded.</p>
  `;

  wireReport(d);
  wirePlayer();
}

/** Custom one-line player (play · seek · time · speed · transcript · download). */
function wirePlayer() {
  const au = $("#session-audio");
  if (!au) return;
  const play = $("#pl-play"), seek = $("#pl-seek"), time = $("#pl-time");
  const paint = () => {
    const dur = Number.isFinite(au.duration) ? au.duration : 0;
    time.textContent = `${fmtTime(au.currentTime)} / ${fmtTime(dur)}`;
    if (!seek.matches(":active")) seek.value = dur ? Math.round((au.currentTime / dur) * 1000) : 0;
    seek.style.setProperty("--p", `${seek.value / 10}%`);
    play.innerHTML = au.paused ? ICON.play : ICON.pause;
    play.setAttribute("aria-label", au.paused ? "Play" : "Pause");
  };
  ["timeupdate", "loadedmetadata", "play", "pause", "ended", "durationchange"].forEach((ev) => au.addEventListener(ev, paint));
  play.addEventListener("click", () => (au.paused ? au.play().catch(() => {}) : au.pause()));
  seek.addEventListener("input", () => {
    if (Number.isFinite(au.duration)) au.currentTime = (seek.value / 1000) * au.duration;
    seek.style.setProperty("--p", `${seek.value / 10}%`);
  });
  $("#pl-transcript").addEventListener("click", (e) => {
    const card = $("#synced-card");
    card.hidden = !card.hidden;
    e.currentTarget.setAttribute("aria-pressed", String(!card.hidden));
  });
  paint();
}

function syncedTranscriptHTML(d) {
  const starts = Object.fromEntries((d.recording?.chapters || []).map((c) => [c.question_id, c.start]));
  return d.questions.map((q) => {
    const a = q.answer;
    const off = starts[q.id];
    const ts = off != null ? `<button type="button" class="ts" data-seek="${off}">${fmtTime(off)}</button>` : "";
    const ai = `<li class="turn ai"><div class="turn-meta"><b>Interviewer</b><span>Q${q.number}${q.kind === "followup" ? " · follow-up" : ""}</span>${ts}</div><p class="turn-text">${esc(q.text)}</p></li>`;
    let me;
    if (a && !a.skipped && a.tokens?.length && off != null) {
      me = `<li class="turn me" data-state="done"><div class="turn-meta"><b>You</b><span>Verbatim</span></div>
        <div class="transcript synced" data-offset="${off}">${renderTokens(a.tokens)}</div></li>`;
    } else {
      me = `<li class="turn me" data-state="skipped"><div class="turn-meta"><b>You</b></div><p class="muted">${a?.skipped ? "Skipped" : "No answer recorded"}</p></li>`;
    }
    return ai + me;
  }).join("");
}

function wireSyncedTranscript() {
  const au = $("#session-audio");
  const list = $("#synced");
  if (!au || !list) return;
  const els = [];
  $$(".synced", list).forEach((tr) => {
    const off = Number(tr.dataset.offset);
    $$("[data-s]", tr).forEach((el) => els.push({ el, s: off + Number(el.dataset.s), e: off + Number(el.dataset.e) }));
  });
  list.addEventListener("click", (e) => {
    const w = e.target.closest(".synced [data-s]");
    if (!w) return;
    au.currentTime = Math.max(0, Number(w.closest(".synced").dataset.offset) + Number(w.dataset.s) - 0.05);
    au.play().catch(() => {});
  });
  let last = null;
  au.addEventListener("timeupdate", () => {
    const t = au.currentTime;
    const hit = els.find((x) => t >= x.s && t < x.e + 0.15);
    if (hit?.el === last) return;
    last?.classList.remove("playing");
    if (!hit) { last = null; return; }
    hit.el.classList.add("playing");
    last = hit.el;
    if ($("#follow-toggle")?.checked && !au.paused) {
      const top = hit.el.offsetTop - list.clientHeight / 3; // list is position:relative
      list.scrollTo({ top, behavior: "smooth" });
    }
  });
}

function activateTab(key) {
  $$(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === key)));
  $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.panel !== key));
}

function wireReport(d) {
  const root = $("#report-root");
  root.addEventListener("click", (e) => {
    const tab = e.target.closest("[data-tab]");
    if (tab) {
      activateTab(tab.dataset.tab);
      if (tab.classList.contains("comp-tile")) $(".tabs").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    const go = e.target.closest("[data-goto]");
    if (go) {
      const card = $(`#qa-${go.dataset.goto}`);
      if (!card) return;
      activateTab(card.closest(".tab-panel").dataset.panel);
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.remove("flash-hl");
      void card.offsetWidth;
      card.classList.add("flash-hl");
      return;
    }
    const seek = e.target.closest("[data-seek]");
    if (seek) {
      const au = $("#session-audio");
      au.currentTime = Number(seek.dataset.seek);
      au.play().catch(() => {});
    }
  });
  wireSyncedTranscript();
  $("#session-speed")?.addEventListener("change", (e) => ($("#session-audio").playbackRate = Number(e.target.value)));

  wireTranscripts(".qa");
  $("#act-retry").onclick = () => { beginCreate({ retry_of: d.id }); location.hash = "#/mic"; };
  $("#act-pdf").onclick = () => window.print();
  $("#act-csv").onclick = () => download(`interview-${d.id}.csv`, reportCsv(d), "text/csv");
  $("#act-txt").onclick = () => download(`interview-${d.id}-verbatim.txt`, verbatimTxt(d), "text/plain");
  $("#act-json").onclick = () => download(`interview-${d.id}.json`, JSON.stringify(d, null, 2), "application/json");
}

/** Click a word → seek that answer's audio; highlight the word being played. */
function wireTranscripts(selector) {
  $$(selector).forEach((card) => {
    const audio = $("audio", card);
    const tr = $(".transcript", card);
    if (!tr) return;
    const els = $$("[data-s]", tr).map((el) => ({ el, s: Number(el.dataset.s), e: Number(el.dataset.e) }));
    tr.addEventListener("click", (e) => {
      const el = e.target.closest("[data-s]");
      if (!el || !audio) return;
      audio.currentTime = Math.max(0, Number(el.dataset.s) - 0.05);
      audio.play().catch(() => {});
    });
    if (audio) {
      let lastEl = null;
      audio.addEventListener("timeupdate", () => {
        const t = audio.currentTime;
        const hit = els.find((x) => t >= x.s && t < x.e);
        if (hit?.el !== lastEl) {
          lastEl?.classList.remove("playing");
          hit?.el.classList.add("playing");
          lastEl = hit?.el || null;
        }
      });
    }
    const toggle = $(".clean-toggle", card);
    toggle?.addEventListener("change", () => {
      tr.classList.toggle("clean", toggle.checked);
      $$(".tok.stammer", tr).forEach((s) => {
        s.textContent = toggle.checked ? s.dataset.clean : s.dataset.raw;
        s.style.display = toggle.checked && !s.dataset.clean ? "none" : "";
      });
    });
  });
}

function verbatimTxt(d) {
  const lines = [
    `${REPORT_TITLE[d.interview_type] || "Interview"} #${d.id} — ${d.role} (${d.level})`,
    fmtDate(d.created_at),
    "",
    "Markers: [um] filler · [stammer: I-I] · [repetition: the] · [false start: I was—] · [pause 2.4s]",
    "",
  ];
  d.questions.forEach((q) => {
    lines.push(`Question ${q.number}${q.kind === "followup" ? " (follow-up)" : ""}: ${q.text}`);
    if (!q.answer) lines.push("Answer: (not answered)");
    else if (q.answer.skipped) lines.push("Answer: (skipped)");
    else lines.push(`Answer: ${markedText(q.answer.tokens) || "(no speech detected)"}`);
    lines.push("");
  });
  return lines.join("\n");
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function reportCsv(d) {
  const r = d.report;
  const rows = [["section", "item", "score_100", "band", "detail"]];
  rows.push(["overall", "Overall Score", r.overall_score, r.overall_band, r.overall_summary]);
  (r.competencies || []).forEach((c) => {
    rows.push(["competency", c.name, c.score, c.band, c.assessment]);
    (c.expectations || []).forEach((e) => rows.push(["expectation", `${c.name}: ${e.text}`, "", e.label,
      (e.evidence || []).map((ev) => `Q${ev.question}: "${ev.quote}"`).join(" | ")]));
  });
  const ratingOf = {};
  (r.competencies || []).forEach((c) => (c.questions || []).forEach((qc) => (ratingOf[qc.question] = [c.name, qc.rating])));
  d.questions.forEach((q) => {
    const [comp, rating] = ratingOf[q.number] || ["", ""];
    const ans = !q.answer ? "(not answered)" : q.answer.skipped ? "(skipped)" : markedText(q.answer.tokens);
    rows.push(["question", `Question ${q.number}: ${q.text}`, "", rating, `[${comp}] ${ans}`]);
  });
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Print every tab and question.
window.addEventListener("beforeprint", () => $$(".tab-panel").forEach((p) => (p.dataset.wasHidden = p.hidden, p.hidden = false)));
window.addEventListener("afterprint", () => $$(".tab-panel").forEach((p) => (p.hidden = p.dataset.wasHidden === "true")));

// ---------------------------------------------------------------------
// History
// ---------------------------------------------------------------------
/** Overall on a 0–100 scale (older reports were /10). */
const overall100 = (s) => (s.overall_score == null ? null : s.report_version === 2 ? s.overall_score : Math.round(s.overall_score * 10));

async function loadHistory() {
  const list = $("#history-list");
  list.innerHTML = `<li class="muted">Loading…</li>`;
  try {
    const sessions = await api("/api/sessions");
    $("#history-empty").hidden = sessions.length > 0;
    list.innerHTML = sessions.map((s) => {
      const o = overall100(s);
      return `
      <li class="history-item">
        <a href="#/report/${s.id}">
          <div>
            <div class="h-role">${esc(s.role)}</div>
            <div class="h-meta">${esc(fmtDate(s.created_at))} · ${esc(s.level)} · ${esc(TYPE_LABEL[s.interview_type] || s.interview_type)} · ${s.duration_minutes} min</div>
          </div>
          ${o != null ? `<span class="score-badge" title="Overall">${o}/100</span>` : `<span class="score-badge none">${s.status === "active" ? "Unfinished" : "No report"}</span>`}
          ${s.fluency_score != null ? `<span class="score-badge flu" title="Delivery">Delivery ${Math.round(s.fluency_score * 10)}</span>` : "<span></span>"}
        </a>
      </li>`;
    }).join("");

    const scored = sessions.filter((s) => s.overall_score != null).reverse(); // oldest → newest
    $("#history-chart-card").hidden = scored.length < 1;
    if (scored.length) {
      $("#history-chart").innerHTML = lineChart(
        scored.map((s) => new Date(s.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })),
        [
          { name: "Overall", cls: "overall", values: scored.map(overall100) },
          { name: "Delivery", cls: "", values: scored.map((s) => (s.fluency_score == null ? null : Math.round(s.fluency_score * 10))) },
        ],
        { yMax: 100 },
      );
    }
  } catch (e) {
    list.innerHTML = `<li class="form-error">${esc(e.message)}</li>`;
  }
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------
initSetup();
loadHealth();
setControls();
route();
