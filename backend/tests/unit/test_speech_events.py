"""Speech channel scoring: density, emphasis and honest empty cases."""
import json

import pytest

from src.media import speech_events as se


def words(spec):
    """``spec`` is (start_s, end_s, text) triples in seconds."""
    return [
        {"start": int(start * 1000), "end": int(end * 1000), "text": text}
        for start, end, text in spec
    ]


def dense_run(start_s, count, per_second=3.0, text="слово"):
    return words(
        [(start_s + i / per_second, start_s + (i + 1) / per_second, text) for i in range(count)]
    )


def test_a_silent_stretch_produces_no_candidate():
    transcript = dense_run(0, 60)
    scored = se.window_scores(transcript, window_seconds=20, hop_seconds=4, start_s=100, end_s=200)
    assert scored == []


def test_dense_speech_outscores_a_sparse_one():
    transcript = dense_run(0, 60, per_second=3.0) + dense_run(20, 6, per_second=0.5)
    scored = se.window_scores(transcript, window_seconds=20, hop_seconds=4, start_s=0, end_s=20)
    assert scored
    assert all(candidate["words_per_second"] > 2.0 for candidate in scored)


def test_emphasis_raises_the_score_of_the_same_word_count():
    calm = words([(i, i + 0.3, "слово") for i in range(10)])
    excited = words([(i, i + 0.3, "слово!") for i in range(10)])
    calm_score = se.window_scores(calm, window_seconds=20, hop_seconds=20)[0]["score"]
    excited_score = se.window_scores(excited, window_seconds=20, hop_seconds=20)[0]["score"]
    assert excited_score > calm_score


def test_ranked_windows_do_not_overlap():
    scored = [
        {"start_s": 0, "end_s": 20, "score": 90},
        {"start_s": 10, "end_s": 30, "score": 80},
        {"start_s": 20, "end_s": 40, "score": 70},
    ]
    chosen = se.rank_windows(scored, max_candidates=3)
    assert [item["start_s"] for item in chosen] == [0, 20]
    assert [item["rank"] for item in chosen] == [1, 2]


def test_bad_words_are_skipped_instead_of_crashing():
    transcript = words([(i, i + 0.3, "слово") for i in range(10)]) + [{"text": "no timing"}, {}]
    scored = se.window_scores(transcript, window_seconds=20, hop_seconds=20)
    assert scored and scored[0]["words"] == 10


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        se.window_scores([], window_seconds=0, hop_seconds=1)


def test_reversed_span_is_refused():
    with pytest.raises(ValueError):
        se.window_scores(dense_run(0, 10), start_s=100, end_s=50)


def test_analyze_transcript_marks_the_signal_kind():
    payload = {"words": dense_run(0, 40)}
    report = se.analyze_transcript(payload, max_candidates=2, source_id="demo")
    assert report["mode"] == "speech_events"
    assert report["source_id"] == "demo"
    # 40 words at 3/s cover 13.3s, so the span yields exactly one window.
    assert len(report["candidates"]) == 1
    assert all(c["event_type"] == "speech_scene" for c in report["candidates"])
    assert "Text signals only" in report["candidates"][0]["reason"]


def test_cli_writes_a_report(tmp_path):
    transcript = tmp_path / "t.transcript_cache.json"
    transcript.write_text(json.dumps({"words": dense_run(0, 40)}), encoding="utf-8")
    out = tmp_path / "out.json"
    assert se.main([str(transcript), "--out", str(out), "--max-candidates", "3"]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["candidates"] and len(payload["candidates"]) <= 3
