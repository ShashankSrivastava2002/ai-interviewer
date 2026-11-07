"""
Unit tests for the disfluency rules, pause handling and fluency stats.

Run from the project root:  pytest backend/tests -q
"""
from backend import disfluency as d
from backend import fluency
from backend.audio import categorize, detect_pauses


def W(text: str, t0: float = None):
    """Build STT-style words from a string; each word lasts 0.3 s with 0.1 s gaps."""
    words, t = [], 0.0 if t0 is None else t0
    for w in text.split():
        words.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2), "confidence": 0.9})
        t += 0.4
    return words


def kinds(tokens):
    return [(t["text"], t["kind"]) for t in tokens]


def kind_of(tokens, text):
    return [t["kind"] for t in tokens if t["text"] == text]


# ---------------- fillers ----------------

def test_hesitation_sounds_are_fillers():
    toks = d.build_timeline(W("um I uhh think er it hmm works umm"))
    for f in ("um", "uhh", "er", "hmm", "umm"):
        assert kind_of(toks, f) == ["filler"], f
    assert kind_of(toks, "think") == ["word"]


def test_crisperwhisper_tags_become_fillers_and_events():
    toks = d.build_timeline(W("[UM] so we [UH] need [laughter] it"))
    assert toks[0]["kind"] == "filler" and toks[0]["text"] == "um"
    assert any(t["text"] == "uh" and t["kind"] == "filler" for t in toks)
    assert any(t["text"] == "[laughter]" and t["kind"] == "event" for t in toks)


def test_you_know_and_i_mean_as_fillers():
    toks = d.build_timeline(W("it was, you know, hard. I mean, we shipped"))
    assert any(t["text"] == "you know," and t["kind"] == "filler" for t in toks)
    assert any(t["text"] == "I mean," and t["kind"] == "filler" for t in toks)


def test_you_know_not_filler_when_literal():
    toks = d.build_timeline(W("do you know what a mutex is"))
    assert all(t["kind"] == "word" for t in toks)


def test_like_filler_vs_verb():
    toks = d.build_timeline(W("I like Python."))
    assert kind_of(toks, "like") == ["word"]
    toks = d.build_timeline(W("and it was, like, really slow"))
    assert kind_of(toks, "like,") == ["filler"]


def test_sentence_initial_so_is_filler_but_not_so_far():
    toks = d.build_timeline(W("So I built it. So far it works"))
    assert kind_of(toks, "So") == ["filler", "word"]


def test_mid_sentence_so_is_word():
    toks = d.build_timeline(W("it was so fast"))
    assert kind_of(toks, "so") == ["word"]


def test_basically_actually_kind_of_right():
    toks = d.build_timeline(W("basically it's kind of slow, right?"))
    assert kind_of(toks, "basically") == ["filler"]
    assert any(t["text"] == "kind of" and t["kind"] == "filler" for t in toks)
    assert kind_of(toks, "right?") == ["filler"]
    toks = d.build_timeline(W("a kind of cache"))
    assert all(t["kind"] == "word" for t in toks)


# ---------------- stammers ----------------

def test_hyphenated_stammers():
    for w in ("I-I-I", "th-the", "w-we", "b-b-but"):
        assert d.is_stammer_word(w), w
    for w in ("well-known", "follow-up", "e-mail", "co-op"):
        assert not d.is_stammer_word(w), w
    toks = d.build_timeline(W("I-I-I think th-the answer"))
    assert kind_of(toks, "I-I-I") == ["stammer"]
    assert kind_of(toks, "th-the") == ["stammer"]


def test_fragment_followed_by_completion_is_stammer():
    toks = d.build_timeline(W("we th- then deployed"))
    assert kind_of(toks, "th-") == ["stammer"]


# ---------------- repetitions ----------------

def test_single_word_repetition():
    toks = d.build_timeline(W("the the system"))
    assert kinds(toks)[:2] == [("the", "word"), ("the", "repetition")]


def test_phrase_repetition_and_fillers_between():
    toks = d.build_timeline(W("I think I think it works"))
    assert [t["kind"] for t in toks][:4] == ["word", "word", "repetition", "repetition"]
    toks = d.build_timeline(W("the um the cache"))
    assert [t["kind"] for t in toks] == ["word", "filler", "repetition", "word"]


def test_triple_repetition_and_allowed_doubles():
    toks = d.build_timeline(W("we we we"))
    assert [t["kind"] for t in toks] == ["word", "repetition", "repetition"]
    toks = d.build_timeline(W("I know that that works"))
    assert all(t["kind"] == "word" for t in toks)


# ---------------- false starts ----------------

def test_em_dash_false_start():
    toks = d.build_timeline(W("I was going— I went to Berlin"))
    assert [t["kind"] for t in toks][:3] == ["false_start"] * 3
    assert kind_of(toks, "went") == ["word"]


def test_truncated_word_false_start():
    toks = d.build_timeline(W("we used wh- I mean Redis"))
    assert kind_of(toks, "wh-") == ["false_start"]


# ---------------- verbatim guarantee ----------------

def test_transcript_is_never_cleaned():
    text = "um so I-I think the the uh you know cache was— is fast"
    toks = d.build_timeline(W(text))
    spoken = " ".join(t["text"] for t in toks if t["kind"] != "pause")
    assert spoken == text


def test_pauses_are_inserted_in_order():
    words = W("I think") + W("it works", t0=3.0)
    toks = d.build_timeline(words, [{"start": 0.7, "end": 3.0, "duration": 2.3, "category": "long"}])
    assert [t["kind"] for t in toks] == ["word", "word", "pause", "word", "word"]
    assert "[pause 2.3s]" in d.to_marked_text(toks)


def test_marked_and_clean_text():
    toks = d.build_timeline(W("um I-I think the the cache"))
    assert d.to_marked_text(toks) == "[um] [stammer: I-I] think the [repetition: the] cache"
    assert d.to_clean_text(toks) == "I think the cache"


# ---------------- pauses + stats ----------------

def test_pause_categories():
    assert categorize(0.7) == "short"
    assert categorize(2.0) == "long"
    assert categorize(3.5) == "very_long"


def test_word_gap_pauses_without_audio():
    words = W("I think", t0=1.2) + W("so yes", t0=4.0)
    p = detect_pauses(None, words)
    assert p["thinking_time"] == 1.2
    assert len(p["pauses"]) == 1 and p["pauses"][0]["category"] == "long"


def test_answer_stats_counts():
    words = W("um I think the the uh cache", t0=1.0)
    p = detect_pauses(None, words)
    toks = d.build_timeline(words, p["pauses"])
    s = fluency.answer_stats(toks, p["thinking_time"], 5.0)
    assert s["fillers"]["total"] == 2
    assert s["repetitions"] == 1
    assert s["thinking_time"] == 1.0
    assert s["word_count"] == 5
    assert 0 <= s["fluency_score"] <= 10
