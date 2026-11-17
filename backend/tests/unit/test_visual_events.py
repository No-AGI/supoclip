"""Signal maths and candidate assembly for speech-independent moment detection.

These tests use synthetic frames and PCM, so they need no video file and no
network: the ffmpeg passes themselves are exercised by the manual run recorded
in docs/visual-events-prototype-2026-11-17.md.
"""
import numpy as np
import pytest

from src.media import visual_events as ve


def test_raw_frames_decode_to_thumbnails():
    raw = bytes(range(256)) * (ve.THUMB_BYTES * 2 // 256 + 1)
    raw = raw[: ve.THUMB_BYTES * 2]
    frames = ve.frames_from_raw(raw)
    assert frames.shape == (2, ve.THUMB_HEIGHT, ve.THUMB_WIDTH)


def test_partial_frame_is_dropped_not_misaligned():
    raw = bytes(ve.THUMB_BYTES + 10)
    frames = ve.frames_from_raw(raw)
    assert len(frames) == 1


def test_frame_difference_keeps_the_time_axis():
    still = np.full((2, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), 10, dtype=np.uint8)
    moved = np.full((2, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), 200, dtype=np.uint8)
    differences = ve.frame_difference(np.concatenate([still, moved]))
    assert len(differences) == 4
    assert differences[0] == 0.0  # no previous frame
    assert differences[1] == 0.0  # 10 -> 10
    assert differences[2] > 0.5  # 10 -> 200 is a large change
    assert differences[3] == 0.0  # 200 -> 200


def test_a_constant_signal_is_not_scaled_into_a_peak():
    assert ve.normalize(np.full(20, -26.0)).max() == 0.0


def test_a_hard_cut_stands_out_from_steady_motion():
    steady = np.full(12, 0.10)
    steady[6] = 0.80
    spikes = ve.cut_spikes(steady)
    assert spikes[6] > 0.5
    assert spikes[3] == 0.0


def test_slow_drift_is_not_a_cut():
    ramp = np.linspace(0.10, 0.30, 12)
    assert ve.cut_spikes(ramp).max() == 0.0


def test_audio_envelope_separates_silence_from_sound():
    sample_rate = ve.AUDIO_SAMPLE_RATE
    silence = np.zeros(sample_rate * 2, dtype=np.int16)
    # Rising amplitude so the envelope has a real dynamic range to rank.
    ramp = np.linspace(0.05, 1.0, sample_rate * 2)
    tone = (np.sin(np.linspace(0, 400 * np.pi, sample_rate * 2)) * 10000 * ramp).astype(np.int16)
    times_quiet, quiet = ve.audio_envelope(silence)
    times_loud, loud = ve.audio_envelope(tone)
    assert len(times_quiet) == len(times_loud) == 4
    assert quiet.max() == 0.0
    assert loud[-1] > loud[0]
    assert loud.min() < 0.2 and loud.max() > 0.8


def test_windows_do_not_overlap_and_respect_duration():
    times = np.arange(0, 120, 2.0)
    scores = np.zeros(len(times))
    scores[[5, 6, 30, 45]] = [90, 85, 80, 70]
    windows = ve.build_windows(
        times, scores, source_duration=120.0, max_candidates=4, target_duration=20.0
    )
    assert len(windows) >= 2
    for window in windows:
        assert 8.0 <= window["end_s"] - window["start_s"] <= 45.0
        assert 0.0 <= window["start_s"] < window["end_s"] <= 120.0
    ordered = sorted(windows, key=lambda item: item["start_s"])
    for first, second in zip(ordered, ordered[1:]):
        assert first["end_s"] <= second["start_s"]


def test_window_snaps_to_a_nearby_cut():
    times = np.arange(0, 100, 1.0)
    scores = np.zeros(len(times))
    scores[50] = 99.0
    windows = ve.build_windows(
        times, scores, source_duration=100.0, cuts=np.array([41.0]), target_duration=20.0
    )
    assert windows[0]["start_s"] == 41.0


def test_zero_scores_produce_no_candidates():
    times = np.arange(0, 60, 2.0)
    assert ve.build_windows(times, np.zeros(len(times)), source_duration=60.0) == []


def test_event_labels_describe_signals_only():
    assert ve.classify_event(0.1, 0.1, 0.9, 0.0) == "audio_peak"
    assert ve.classify_event(0.9, 0.1, 0.1, 0.0) == "motion_burst"
    assert ve.classify_event(0.2, 0.9, 0.6, 0.0) == "montage_cut"
    assert ve.classify_event(0.1, 0.1, 0.1, 1.0) == "speech_scene"


def test_speech_windows_shadow_the_picture():
    motion = 0.9
    with_speech = ve.score_window(motion, 0.1, 0.1, 1.0, "speech_scene")
    without = ve.score_window(motion, 0.1, 0.1, 0.0, "motion_burst")
    # A transcript-covered window ranks below the same picture with no speech,
    # so the audio-only pass does not simply re-find what the transcript found.
    assert with_speech < without


def test_evidence_frames_stay_inside_the_window():
    window = {"start_s": 10.0, "end_s": 30.0, "peak_s": 19.0}
    frames = ve.evidence_times(window)
    assert frames and all(10.0 <= value <= 30.0 for value in frames)
    assert len(set(frames)) == len(frames)


def test_a_held_slate_is_detected_as_a_card():
    held = np.full((ve.THUMB_HEIGHT, ve.THUMB_WIDTH), 40, dtype=np.uint8)
    frames = np.repeat(held[np.newaxis, :, :], 20, axis=0)
    assert ve.static_share(frames) == 1.0
    assert ve.is_card(1.0, 0.0) is True


def test_moving_footage_is_not_a_card():
    frames = np.zeros((20, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    for index in range(20):
        frames[index] = (index * 7) % 256
    assert ve.static_share(frames) == 0.0
    assert ve.is_card(0.0, 0.0) is False


def test_black_frames_are_flat():
    black = np.zeros((10, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    assert ve.flat_share(black) == 1.0
    assert ve.is_card(0.0, 1.0) is True


def test_a_card_is_ranked_below_real_footage(monkeypatch, tmp_path):
    """An end slate gets a hard cut and a loud hit; it must not outrank action."""
    times = np.arange(0, 200, 10.0)
    tier = ve.ScanTier(
        times=times,
        motion=np.full(len(times), 0.9),
        cuts=np.zeros(len(times)),
        audio=np.full(len(times), 0.9),
        source_duration=200.0,
        detail={},
    )
    monkeypatch.setattr(ve, "probe_duration", lambda path: 200.0)
    monkeypatch.setattr(ve, "_coarse_scan", lambda path, duration, speech: tier)

    def fake_refine(path, window):
        card = window["start_s"] >= 150
        return {
            "motion_rel": 0.4,
            "cut_peak": 0.9,
            "static_share": 0.9 if card else 0.0,
            "flat_share": 0.0,
            "fps": 8.0,
            "frames": 40,
        }

    monkeypatch.setattr(ve, "_refine_window", fake_refine)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"stub")
    report = ve.analyze_source(source, max_candidates=4, target_duration=20.0)
    cards = [item for item in report["candidates"] if item["event_type"] == "static_card"]
    assert cards, "the held window must be labelled"
    assert cards[0]["score"] < cards[0]["score_raw"]
    assert "held frame" in cards[0]["reason"]


def test_report_contract_fields(monkeypatch, tmp_path):
    times = np.arange(0, 60, 2.0)
    tier = ve.ScanTier(
        times=times,
        motion=np.linspace(0.1, 0.9, len(times)),
        cuts=np.zeros(len(times)),
        audio=np.linspace(0.9, 0.1, len(times)),
        source_duration=60.0,
        detail={"frames": len(times)},
    )
    monkeypatch.setattr(ve, "probe_duration", lambda path: 60.0)
    monkeypatch.setattr(ve, "_coarse_scan", lambda path, duration, speech: tier)
    monkeypatch.setattr(ve, "_refine_window", lambda path, window: {"motion": 0.5, "cut": 0.1, "fps": 8.0})
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a real video")
    report = ve.analyze_source(source, source_id="fixture-1", max_candidates=2)
    assert report["schema_status"] == ve.SCHEMA_STATUS
    assert report["mode"] == ve.MODE
    assert report["candidates"]
    candidate = report["candidates"][0]
    for key in (
        "start_s", "end_s", "event_type", "reason", "score", "evidence_frames_s",
        "rights_status", "caption_review", "review_status", "publish_status",
        "audio_lang", "subtitle_lang", "source_type",
    ):
        assert key in candidate, key
    assert candidate["score_kind"] == "heuristic_rank"
    assert "not evidence" in report["disclaimer"]


def test_missing_source_is_an_error(tmp_path):
    source = tmp_path / "absent.mp4"
    try:
        ve.analyze_source(source)
    except FileNotFoundError:
        return
    raise AssertionError("missing source must raise")


def test_motion_centroid_finds_the_side_motion_is_on():
    frames = np.zeros((6, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    frames[:, :, :] = 20
    # Only the right third changes between frames 3 and 4.
    frames[4:, :, int(ve.THUMB_WIDTH * 2 / 3):] = 220
    centres = ve.motion_centroid(frames)
    assert len(centres) == 6
    assert centres[0] == 0.5  # no previous frame
    assert centres[4] > 0.6


def test_centroid_is_smoothed_across_neighbours():
    centres = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    smoothed = ve.smoothed_centroid(centres, window=3)
    assert float(smoothed.std()) < float(centres.std())


def test_crop_window_keeps_the_centre_inside_a_wide_source():
    for centre in (0.0, 0.5, 1.0):
        box = ve.crop_window_for_centroid(centre, 1920, 1080)
        assert box["height"] == 1080
        assert box["width"] == 608  # 1080 * 9/16 = 607.5, rounded to even
        assert box["x"] >= 0 and box["x"] + box["width"] <= 1920
        assert box["y"] >= 0


def test_crop_window_is_centred_for_the_middle():
    box = ve.crop_window_for_centroid(0.5, 1920, 1080)
    assert abs(box["x"] - (1920 - box["width"]) / 2) <= 1


def test_crop_window_handles_a_tall_source():
    box = ve.crop_window_for_centroid(0.5, 1080, 1920)
    assert box["width"] == 1080
    assert box["height"] <= 1920


def test_signatures_find_a_shared_frame():
    frames = np.zeros((6, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    frames[:, :, :20] = 220
    repeated = np.zeros((6, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    repeated[:, :, 40:] = 60
    repeated[3] = frames[0]  # same shot re-used at another time
    assert ve.windows_share_footage(
        ve.frame_signatures(frames), ve.frame_signatures(repeated)
    )


def test_different_footage_is_not_a_duplicate():
    left = np.zeros((6, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    left[:, :, :20] = 220
    right = np.zeros((6, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    right[:, :, 40:] = 220
    assert not ve.windows_share_footage(
        ve.frame_signatures(left), ve.frame_signatures(right)
    )


def test_a_mean_picture_would_have_confused_two_dark_scenes():
    """Regression for a rejected design: whole-window means were meaningless."""
    first = np.zeros((10, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    first[:, :, :20] = 90
    second = np.zeros((10, ve.THUMB_HEIGHT, ve.THUMB_WIDTH), dtype=np.uint8)
    second[:, :, 20:40] = 92
    signatures = ve.frame_signatures(first)
    means_differ = abs(float(first.mean()) - float(second.mean()))
    assert means_differ < 10  # nearly the same average brightness ...
    assert not ve.windows_share_footage(signatures, ve.frame_signatures(second))


def test_duplicate_content_is_dropped_from_the_report(monkeypatch, tmp_path):
    """A trailer reuses one shot; two windows on it must not both ship."""
    times = np.arange(0, 120, 10.0)
    tier = ve.ScanTier(
        times=times,
        motion=np.full(len(times), 0.9),
        cuts=np.zeros(len(times)),
        audio=np.full(len(times), 0.8),
        source_duration=120.0,
        detail={},
    )
    monkeypatch.setattr(ve, "probe_duration", lambda path: 120.0)
    monkeypatch.setattr(ve, "_coarse_scan", lambda path, duration, speech: tier)

    reused = np.zeros((4, ve.SIGNATURE_BLOCKS[0] * ve.SIGNATURE_BLOCKS[1]), dtype=float)
    other = np.full((4, ve.SIGNATURE_BLOCKS[0] * ve.SIGNATURE_BLOCKS[1]), 0.9, dtype=float)

    def fake_refine(path, window):
        signature = reused if window["start_s"] < 60 else other
        return {
            "motion_rel": 0.3,
            "cut_peak": 0.5,
            "static_share": 0.0,
            "flat_share": 0.0,
            "signature": signature,
            "fps": 8.0,
            "frames": 20,
        }

    monkeypatch.setattr(ve, "_refine_window", fake_refine)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"stub")
    report = ve.analyze_source(source, max_candidates=4, target_duration=20.0)
    assert report["scan"]["duplicates_dropped"] >= 1
    assert 1 <= len(report["candidates"]) < 4
    assert "_signature" not in report["candidates"][0]


def test_window_limits_candidates_to_the_requested_span():
    """A loud hour must not crowd out the slice the caller asked about."""
    times = np.arange(0.0, 1200.0, 1.0)
    scores = np.full_like(times, 4.0)
    scores[:400] = 99.0  # the loud part, outside the requested window
    scores[650] = 9.0
    windows = ve.build_windows(times, scores, source_duration=1200.0, max_candidates=5)
    assert windows[0]["start_s"] < 420

    keep = (times >= 600.0) & (times <= 900.0)
    inside = ve.build_windows(
        times[keep], scores[keep], source_duration=1200.0, max_candidates=5
    )
    assert inside
    for window in inside:
        # A window is centred on its peak, so it may start up to half a target
        # duration before the span begins.
        assert 600.0 - ve.DEFAULT_TARGET_DURATION / 2 <= window["start_s"] <= 900.0


def test_window_must_not_be_reversed(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not a real video")
    with pytest.raises(ValueError):
        ve.analyze_source(source, window=(600.0, 300.0))
