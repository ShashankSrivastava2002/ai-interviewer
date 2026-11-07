"""
disfluency.py — turns STT words + detected pauses into one verbatim timeline and
labels every token as:

    word | filler | stammer | repetition | false_start | pause | event

("event" covers non-speech vocal tags some engines emit, e.g. CrisperWhisper's
"[laughter]". It is kept so the transcript stays exactly as recognised.)

The transcript is NEVER cleaned here: every token the recogniser produced stays in
the output, in order, with its original text. We only *annotate* it.

This module is pure Python (no project imports) so it is easy to unit-test.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Hesitation sounds: um, umm, uh, uhh, er, erm, ah, hmm, mm, mhm, eh ...
FILLER_SOUND_RE = re.compile(
    r"^(u+m+|u+h+m*|e+r+m*|e+r+|a+h+|h+m+|m{2,}|m+h+m+|e+h+|uh-?huh)$"
)

# Words that are (almost always) used as fillers in spoken interview answers.
ALWAYS_FILLER_WORDS = {"basically", "actually", "literally"}

# Back-to-back doubles that are usually grammatical ("I know that that works").
ALLOWED_DOUBLES = {"that", "had"}

# Previous words after which "like" is a real verb/preposition, not a filler.
LIKE_NOT_FILLER_PREV = {
    "would", "i'd", "you'd", "we'd", "they'd", "he'd", "she'd", "don't", "didn't",
    "doesn't", "to", "really", "also", "feel", "felt", "looks", "look", "looked",
    "seems", "seem", "something", "things", "anything", "nothing", "more", "much",
    "i", "we", "you", "they", "just", "exactly", "not", "is", "it's", "are", "be",
}
# Previous words after which "like" is typically a filler ("I was like ...").
LIKE_FILLER_PREV = {"and", "but", "so", "was", "were", "totally", "literally"}

YOU_KNOW_NOT_FILLER_NEXT = {
    "that", "what", "how", "why", "where", "when", "who", "whether", "if", "the",
    "a", "an", "it", "him", "her", "them", "about", "it's", "this", "those", "these",
    "me", "us", "my", "your", "our", "their", "his",
}
YOU_KNOW_NOT_FILLER_PREV = {"do", "did", "don't", "didn't", "if", "would", "could", "you", "as"}

I_MEAN_NOT_FILLER_NEXT = {"it", "that", "to", "what", "this", "is", "was", "them", "him", "her", "you"}
I_MEAN_NOT_FILLER_PREV = {"what", "do", "did", "don't"}

KIND_OF_NOT_FILLER_PREV = {
    "a", "the", "what", "this", "that", "some", "any", "one", "which", "every",
    "each", "no", "same", "different", "these", "those", "my", "your", "our",
}

SO_NOT_FILLER_NEXT = {"far", "much", "many", "long", "that"}

_STRIP_CHARS = " \t\"'“”‘’.,!?;:()…"
_DASHES = "-‐–—"

TOKEN_KINDS = ("word", "filler", "stammer", "repetition", "false_start", "pause", "event")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def norm(text: str) -> str:
    """Lower-case and strip surrounding punctuation and dashes (keeps inner '-' / "'")."""
    t = text.lower().strip().strip(_STRIP_CHARS).strip(_DASHES).strip(_STRIP_CHARS)
    return t.replace("’", "'")


def _ends_sentence(raw: str) -> bool:
    return raw.rstrip("\"')”’ ").endswith((".", "?", "!", "…"))


def _ends_clause(raw: str) -> bool:
    return raw.rstrip("\"')”’ ").endswith((",", ";", ":"))


def _ends_with_cut(raw: str) -> bool:
    """Word abandoned with a long dash: 'going—', 'going--', 'going –'."""
    r = raw.rstrip()
    return r.endswith(("—", "–", "--"))


def _is_truncated_fragment(raw: str) -> bool:
    """A word fragment ending in a single hyphen, e.g. 'th-' or 'wh-'."""
    r = raw.strip().strip(_STRIP_CHARS)
    return len(r) >= 2 and r.endswith("-") and not r.endswith("--") and r[:-1].replace("'", "").isalpha()


def is_stammer_word(raw: str) -> bool:
    """
    True for hyphenated syllable repeats inside one token:
        'I-I-I', 'th-the', 'w-we', 'b-b-but'
    but not real hyphenated words like 'well-known' or 'follow-up'.
    """
    s = raw.lower().strip().strip(_STRIP_CHARS)
    parts = [p for p in re.split(r"[-‐]", s)]
    if len(parts) < 2 or any(not p for p in parts):
        return False
    last = parts[-1]
    if not all(p.replace("'", "").isalpha() for p in parts):
        return False
    for p in parts[:-1]:
        if not last.startswith(p):
            return False
        if len(p) > 3 and p != last:
            return False
    return True


def _filler_sound(n: str) -> bool:
    return bool(FILLER_SOUND_RE.match(n))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_timeline(words: list[dict[str, Any]], pauses: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    words:  [{word, start, end, confidence}] from stt.py (seconds)
    pauses: [{start, end, duration, category}] from audio.py
    returns ordered tokens: [{text, norm, kind, start, end, confidence, group?, ...}]
    """
    pauses = sorted(pauses or [], key=lambda p: p["start"])
    toks = _initial_tokens(words)
    _mark_pause_context(toks, pauses)
    _mark_stammers_and_fragments(toks)
    toks = _mark_fillers(toks)
    _mark_false_starts(toks)
    _mark_repetitions(toks)

    # Merge pauses into the timeline.
    out: list[dict[str, Any]] = []
    for t in toks:
        t.pop("_pause_before", None)
        t.pop("_pause_after", None)
        out.append(t)
    for p in pauses:
        out.append({
            "text": "",
            "norm": "",
            "kind": "pause",
            "start": round(float(p["start"]), 3),
            "end": round(float(p["end"]), 3),
            "duration": round(float(p.get("duration", p["end"] - p["start"])), 2),
            "category": p.get("category", ""),
        })
    # Sort by time; on ties a pause goes after the word that ends where it starts.
    out.sort(key=lambda t: (t["start"], 1 if t["kind"] == "pause" else 0))
    return out


def _initial_tokens(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    toks: list[dict[str, Any]] = []
    for w in words:
        raw = str(w.get("word", "")).strip()
        if not raw:
            continue
        tok: dict[str, Any] = {
            "text": raw,
            "norm": norm(raw),
            "kind": "word",
            "start": round(float(w.get("start", 0.0)), 3),
            "end": round(float(w.get("end", 0.0)), 3),
            "confidence": None if w.get("confidence") is None else round(float(w["confidence"]), 3),
        }
        # Bracketed tags (CrisperWhisper: "[UM]", "[UH]", "[laughter]").
        m = re.match(r"^\[(.+?)\]([.,!?]*)$", raw)
        if m:
            inner = m.group(1).strip().lower()
            if _filler_sound(inner):
                tok.update(text=inner + m.group(2), norm=inner, kind="filler")
            else:
                tok.update(norm="", kind="event")
        toks.append(tok)
    return toks


def _mark_pause_context(toks: list[dict[str, Any]], pauses: list[dict[str, Any]]) -> None:
    """Record the length of any detected pause just before/after each token."""
    for t in toks:
        t["_pause_before"] = 0.0
        t["_pause_after"] = 0.0
    for p in pauses:
        dur = p["end"] - p["start"]
        # The token right before the pause is the last one ending at/just after its start.
        before = [i for i, t in enumerate(toks) if t["end"] <= p["start"] + 0.25]
        if before:
            i = before[-1]
            toks[i]["_pause_after"] = max(toks[i]["_pause_after"], dur)
            if i + 1 < len(toks):
                toks[i + 1]["_pause_before"] = max(toks[i + 1]["_pause_before"], dur)
        elif toks:
            toks[0]["_pause_before"] = max(toks[0]["_pause_before"], dur)


def _mark_stammers_and_fragments(toks: list[dict[str, Any]]) -> None:
    for i, t in enumerate(toks):
        if t["kind"] != "word":
            continue
        raw = t["text"]
        if is_stammer_word(raw):
            t["kind"] = "stammer"
            continue
        if _is_truncated_fragment(raw):
            frag = raw.strip().strip(_STRIP_CHARS)[:-1].lower()
            nxt = toks[i + 1]["norm"] if i + 1 < len(toks) else ""
            # "th- the" -> stammer;  "wh- I think" -> abandoned (false start)
            if nxt and nxt.startswith(frag):
                t["kind"] = "stammer"
            else:
                t["kind"] = "false_start"
                t["group"] = f"fs{i}"


def _is_sentence_start(toks: list[dict[str, Any]], i: int) -> bool:
    j = i - 1
    while j >= 0 and toks[j]["kind"] == "event":
        j -= 1
    if j < 0:
        return True
    prev = toks[j]
    return (
        _ends_sentence(prev["text"])
        or prev["kind"] == "filler"
        or toks[i]["_pause_before"] >= 1.0
    )


def _merge(toks: list[dict[str, Any]], i: int, n: int, kind: str) -> dict[str, Any]:
    """Merge n tokens starting at i into one token (used for multi-word fillers)."""
    parts = toks[i:i + n]
    confs = [p["confidence"] for p in parts if p["confidence"] is not None]
    return {
        "text": " ".join(p["text"] for p in parts),
        "norm": " ".join(p["norm"] for p in parts),
        "kind": kind,
        "start": parts[0]["start"],
        "end": parts[-1]["end"],
        "confidence": round(min(confs), 3) if confs else None,
        "_pause_before": parts[0]["_pause_before"],
        "_pause_after": parts[-1]["_pause_after"],
    }


def _mark_fillers(toks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    n_toks = len(toks)
    while i < n_toks:
        t = toks[i]
        if t["kind"] != "word":
            out.append(t)
            i += 1
            continue
        w = t["norm"]
        nxt = toks[i + 1] if i + 1 < n_toks else None
        nxt2 = toks[i + 2] if i + 2 < n_toks else None
        prev = out[-1] if out else None
        prev_norm = prev["norm"] if prev else ""

        # ---- two-word fillers ----
        if nxt is not None and nxt["kind"] == "word":
            pair = f"{w} {nxt['norm']}"
            after = nxt2["norm"] if nxt2 is not None else ""
            comma_after = _ends_clause(nxt["text"]) or _ends_sentence(nxt["text"]) or nxt["_pause_after"] >= 0.5
            is_filler = False
            if pair == "you know":
                is_filler = prev_norm not in YOU_KNOW_NOT_FILLER_PREV and (
                    comma_after or not after or after not in YOU_KNOW_NOT_FILLER_NEXT
                )
            elif pair == "i mean":
                is_filler = prev_norm not in I_MEAN_NOT_FILLER_PREV and (
                    comma_after or not after or after not in I_MEAN_NOT_FILLER_NEXT
                )
            elif pair in ("kind of", "sort of"):
                is_filler = prev_norm not in KIND_OF_NOT_FILLER_PREV and after not in ("thing", "things")
            if is_filler:
                t["kind"] = nxt["kind"] = "filler"  # so later context checks see it
                out.append(_merge(toks, i, 2, "filler"))
                i += 2
                continue

        # ---- single-word fillers ----
        is_filler = False
        if _filler_sound(w) or w in ALWAYS_FILLER_WORDS:
            is_filler = True
        elif w == "so":
            nxt_norm = nxt["norm"] if nxt else ""
            is_filler = (_is_sentence_start(toks, i) and nxt_norm not in SO_NOT_FILLER_NEXT) or _ends_clause(t["text"])
        elif w == "like":
            pause_near = t["_pause_before"] >= 0.5 or t["_pause_after"] >= 0.5
            comma_near = _ends_clause(t["text"]) or (prev is not None and _ends_clause(prev["text"]))
            if prev_norm in LIKE_NOT_FILLER_PREV and not comma_near:
                is_filler = False
            else:
                is_filler = (
                    _is_sentence_start(toks, i)
                    or comma_near
                    or pause_near
                    or (prev is not None and prev["kind"] == "filler")
                    or prev_norm in LIKE_FILLER_PREV
                )
        elif w == "right":
            is_filler = t["text"].rstrip().endswith("?") or (_is_sentence_start(toks, i) and _ends_clause(t["text"]))
        if is_filler:
            t["kind"] = "filler"
        out.append(t)
        i += 1
    return out


def _mark_false_starts(toks: list[dict[str, Any]]) -> None:
    """A phrase cut off with a long dash ('I was going— I went') is a false start."""
    for i, t in enumerate(toks):
        if t["kind"] not in ("word",) or not _ends_with_cut(t["text"]):
            continue
        group = f"fs{i}"
        j = i
        # Walk back to the start of the clause (max 6 words).
        while j - 1 >= 0 and i - (j - 1) < 6:
            p = toks[j - 1]
            if p["kind"] != "word" or _ends_sentence(p["text"]) or _ends_clause(p["text"]) or _ends_with_cut(p["text"]):
                break
            if t["_pause_before"] >= 1.0 and j == i:
                break
            j -= 1
        for k in range(j, i + 1):
            toks[k]["kind"] = "false_start"
            toks[k]["group"] = group


def _mark_repetitions(toks: list[dict[str, Any]]) -> None:
    """
    Back-to-back repeats of 1–4 word phrases ('the the', 'I think I think').
    Fillers/pauses in between are ignored ('the, um, the' is a repetition).
    The extra copies are marked as repetition; the first copy stays a word.
    """
    seq = [i for i, t in enumerate(toks) if t["kind"] == "word" and t["norm"]]
    i = 0
    rep_id = 0
    while i < len(seq):
        found = False
        for n in (4, 3, 2, 1):
            if i + 2 * n > len(seq):
                continue
            a = [toks[k]["norm"] for k in seq[i:i + n]]
            b = [toks[k]["norm"] for k in seq[i + n:i + 2 * n]]
            if a != b or (n == 1 and a[0] in ALLOWED_DOUBLES):
                continue
            # The two copies must be close in time (no long gap / topic change).
            if toks[seq[i + n]]["start"] - toks[seq[i + n - 1]]["end"] > 3.0:
                continue
            rep_id += 1
            k = i + n
            while k + n <= len(seq) and [toks[x]["norm"] for x in seq[k:k + n]] == a:
                for x in seq[k:k + n]:
                    toks[x]["kind"] = "repetition"
                    toks[x]["group"] = f"rep{rep_id}"
                k += n
            i = k
            found = True
            break
        if not found:
            i += 1


# ---------------------------------------------------------------------------
# Text renderings
# ---------------------------------------------------------------------------

def _groups(tokens: list[dict[str, Any]]):
    """Yield runs of tokens; consecutive tokens with the same group form one run."""
    run: list[dict[str, Any]] = []
    for t in tokens:
        g = t.get("group")
        if run and g and run[-1].get("group") == g:
            run.append(t)
            continue
        if run:
            yield run
        run = [t]
    if run:
        yield run


def to_marked_text(tokens: list[dict[str, Any]]) -> str:
    """
    Verbatim text with explicit markers, used for the LLM and the .txt download:
      [um] I [stammer: I-I-I] think [pause 2.4s] the [repetition: the] ...
    """
    parts: list[str] = []
    for run in _groups(tokens):
        kind = run[0]["kind"]
        text = " ".join(t["text"] for t in run)
        if kind == "pause":
            parts.append(f"[pause {run[0]['duration']:.1f}s]")
        elif kind == "filler":
            parts.append(f"[{text}]")
        elif kind == "stammer":
            parts.append(f"[stammer: {text}]")
        elif kind == "repetition":
            parts.append(f"[repetition: {text}]")
        elif kind == "false_start":
            parts.append(f"[false start: {text}]")
        else:
            parts.append(text)
    return " ".join(parts).strip()


def to_plain_verbatim(tokens: list[dict[str, Any]]) -> str:
    """Verbatim words exactly as spoken, pauses shown as '…'."""
    parts = []
    for t in tokens:
        if t["kind"] == "pause":
            parts.append("…")
        else:
            parts.append(t["text"])
    return " ".join(parts).strip()


def to_clean_text(tokens: list[dict[str, Any]]) -> str:
    """'What you meant' view: drops fillers, pauses, repetitions and false starts."""
    parts = []
    for t in tokens:
        k = t["kind"]
        if k in ("pause", "filler", "repetition", "false_start", "event"):
            continue
        if k == "stammer":
            # 'I-I-I' -> 'I' ; 'th-' -> (dropped, the next word completes it)
            raw = t["text"].strip()
            if raw.endswith("-"):
                continue
            parts.append(re.split(r"[-‐]", raw)[-1])
            continue
        parts.append(t["text"])
    return " ".join(parts).strip()


def has_filler(tokens: list[dict[str, Any]]) -> bool:
    return any(t["kind"] == "filler" for t in tokens)
