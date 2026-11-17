"""Speech-independent moment detection for gameplay video.

SupoClip's existing pipeline selects moments from the transcript. A game stream
also contains long stretches with no useful speech (chases, animation details,
crashes, environmental shots), and those are invisible to a transcript-driven
selector. This module scans the picture and the audio envelope locally, with no
paid API and no new dependency (ffmpeg + numpy only), and returns ranked
candidate windows with evidence frames for a human to review.

Cost model — two tiers, because a full decode of a long stream is expensive:

* Tier A (coarse) reads **keyframes only** and a 2 kHz audio envelope. On a
  535 s 1080p source this is ~0.7 s of CPU for the video pass and ~3 s for the
  audio, versus ~50 s for a full decode, so a two-hour stream stays cheap.
* Tier B (refine) re-reads only the shortlisted seconds at 8 fps to place the
  peak inside the window precisely.

A score from this module is a **ranking hint, never proof that an event
happened**: audio peaks are not evidence of a game mechanic, and frame
difference cannot name what it saw. Every candidate therefore carries the
signals that produced it plus frames a reviewer must actually look at.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# Frames are reduced to a tiny grey thumbnail: motion and cut detection only
# need relative change, and this keeps a two-hour scan small.
THUMB_WIDTH = 64
THUMB_HEIGHT = 36
THUMB_BYTES = THUMB_WIDTH * THUMB_HEIGHT

# Tier B sampling rate for shortlisted windows.
REFINE_FPS = 8.0
AUDIO_SAMPLE_RATE = 2000
AUDIO_WINDOW_SECONDS = 0.5

# Candidate defaults: a Reels-length window, never shorter than a readable beat.
DEFAULT_TARGET_DURATION = 20.0
MIN_CANDIDATE_DURATION = 8.0
MAX_CANDIDATE_DURATION = 45.0

# A cut must stand out from its neighbourhood to count as a shot change rather
# than one more frame of fast movement.
CUT_SPIKE_RATIO = 2.5
CUT_FLOOR = 0.06

# Documented ranking weights. The labels describe what the signals are, not what
# happened in the game.
EVENT_WEIGHTS = {
    "montage_cut": 1.0,
    "motion_burst": 0.9,
    "audio_peak": 0.55,
    "quiet_detail": 0.45,
    "speech_scene": 0.35,
}

SCHEMA_STATUS = "PROPOSED_EXAMPLE_NOT_EXISTING_API"
MODE = "gaming_visual_highlights"

# A held frame (end slate, black card, loading screen) produces a hard cut and a
# loud music hit, so it scores well on the raw signals while showing nothing.
# These thresholds demote it instead of quietly ranking it first.
STATIC_DIFF_EPS = 0.004
STATIC_SHARE_DEMOTE = 0.35
FLAT_FRAME_STD = 3.0
FLAT_SHARE_DEMOTE = 0.25
CARD_DEMOTION = 0.35


@dataclass
class ScanTier:
    """One sampling of the source: frame times and the signals read with them."""

    times: np.ndarray
    motion: np.ndarray
    cuts: np.ndarray
    audio: np.ndarray
    mode: str = "coarse"
    source_duration: float = 0.0
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Signal maths (pure functions, unit-tested without ffmpeg)
# --------------------------------------------------------------------------


def frames_from_raw(buffer: bytes, width: int = THUMB_WIDTH, height: int = THUMB_HEIGHT) -> np.ndarray:
    """Decode ffmpeg's raw grey output into (frame, height, width) uint8."""
    frame_bytes = width * height
    usable = len(buffer) - (len(buffer) % frame_bytes)
    if usable <= 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    return np.frombuffer(buffer[:usable], dtype=np.uint8).reshape(-1, height, width)


def frame_difference(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference from the previous thumbnail, scaled to 0..1.

    Entry ``i`` is the change into frame ``i``, so index 0 is always 0 (there is
    no earlier frame) and the array keeps the same length as ``times``.
    """
    if len(frames) < 2:
        return np.zeros(len(frames), dtype=float)
    data = frames.astype(np.int16)
    diffs = np.abs(np.diff(data, axis=0)).mean(axis=(1, 2)) / 255.0
    return np.concatenate([[0.0], diffs])


def cut_spikes(differences: np.ndarray, window: int = 5) -> np.ndarray:
    """How far each difference stands out from its neighbourhood, in 0..1.

    A hard cut shows a single large difference surrounded by small ones; fast
    motion keeps a raised floor over many frames, which the running median
    removes.
    """
    if len(differences) == 0:
        return np.zeros(0, dtype=float)
    spikes = np.zeros(len(differences), dtype=float)
    for index, value in enumerate(differences):
        low = max(0, index - window)
        high = min(len(differences), index + window + 1)
        neighbours = np.delete(differences[low:high], index - low)
        baseline = float(np.median(neighbours)) if len(neighbours) else 0.0
        if value < CUT_FLOOR or value < baseline * CUT_SPIKE_RATIO:
            continue
        spikes[index] = min(1.0, (value - baseline) / max(value, 1e-6))
    return spikes


def audio_envelope(samples: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE, window_seconds: float = AUDIO_WINDOW_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, relative loudness 0..1) for s16 mono PCM.

    Loudness is RMS converted to dB and mapped onto the observed dynamic range,
    so a quiet stream and a loud one are compared against themselves.
    """
    window = max(1, int(sample_rate * window_seconds))
    count = len(samples) // window
    if count == 0:
        return np.zeros(0), np.zeros(0)
    blocks = samples[: count * window].astype(np.float32).reshape(count, window) / 32768.0
    rms = np.sqrt((blocks**2).mean(axis=1))
    levels = 20.0 * np.log10(np.maximum(rms, 1e-6))
    times = (np.arange(count) + 0.5) * window_seconds
    return times, normalize(levels)


def normalize(values: np.ndarray, low_percentile: float = 5.0, high_percentile: float = 95.0) -> np.ndarray:
    """Robust 0..1 scaling that ignores a few extreme samples.

    A signal that never varies (silence, a locked-off shot, a steady music bed)
    carries no ranking information, so it maps to zeros instead of being scaled
    up into a false peak by percentile interpolation.
    """
    if len(values) == 0:
        return np.zeros(0, dtype=float)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    spread = high - low
    if spread < 1e-9 or spread < 1e-6 * max(1.0, abs(high), abs(low)):
        return np.zeros(len(values), dtype=float)
    return np.clip((values - low) / spread, 0.0, 1.0)


def resample_to(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Sample ``values`` (given at ``times``) onto ``target_times``."""
    if len(times) == 0 or len(values) == 0 or len(target_times) == 0:
        return np.zeros(len(target_times), dtype=float)
    return np.interp(target_times, times, values)


def speech_mask(times: np.ndarray, speech_windows: Sequence[tuple[float, float]]) -> np.ndarray:
    """1.0 where a transcript word is audible, so speech scenes stay separable."""
    mask = np.zeros(len(times), dtype=float)
    for start, end in speech_windows:
        mask[(times >= start) & (times <= end)] = 1.0
    return mask


def classify_event(motion: float, cut: float, audio: float, speech: float) -> str:
    """Name the dominant signal of a window. Describes signals, not the game."""
    if speech >= 0.5:
        return "speech_scene"
    if cut >= 0.5 and audio >= 0.35:
        return "montage_cut"
    if motion >= 0.5:
        return "motion_burst"
    if audio >= 0.5:
        return "audio_peak"
    return "quiet_detail"


def score_window(motion: float, cut: float, audio: float, speech: float, event_type: str) -> float:
    """0..100 heuristic rank from the four normalised signals."""
    strength = 0.45 * motion + 0.30 * cut + 0.25 * audio
    return round(100.0 * min(1.0, strength * EVENT_WEIGHTS.get(event_type, 0.5)), 1)


def static_share(frames: np.ndarray) -> float:
    """Share of the window held on one picture (identical consecutive frames).

    End slates, cards and loading screens sit still while the music swells, and
    frame difference alone cannot tell them from a real scene.
    """
    if len(frames) < 2:
        return 0.0
    differences = frame_difference(frames)
    held = differences[1:] <= STATIC_DIFF_EPS
    longest = 0
    current = 0
    for flag in held:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return round(longest / max(1, len(differences) - 1), 3)


def flat_share(frames: np.ndarray, threshold: float = FLAT_FRAME_STD) -> float:
    """Share of frames with almost no spatial detail (black or solid colour)."""
    if len(frames) == 0:
        return 0.0
    spreads = frames.reshape(len(frames), -1).std(axis=1)
    return round(float((spreads < threshold).mean()), 3)


def is_card(static: float, flat: float) -> bool:
    """True when a window is mostly a held card or black rather than footage."""
    return static >= STATIC_SHARE_DEMOTE or flat >= FLAT_SHARE_DEMOTE


def motion_centroid(frames: np.ndarray) -> np.ndarray:
    """Horizontal centre (0..1) of what changed in each frame pair.

    For gameplay the interesting thing is usually where the movement is, not
    where a face is. Returns 0.5 for pairs with no measurable change so a still
    moment does not drag the crop to an edge.
    """
    if len(frames) < 2:
        return np.full(len(frames), 0.5, dtype=float)
    data = frames.astype(np.int16)
    deltas = np.abs(np.diff(data, axis=0)).mean(axis=1)  # (frames-1, width)
    columns = np.arange(deltas.shape[1], dtype=float)
    centres = np.full(len(frames), 0.5, dtype=float)
    for index, row in enumerate(deltas, start=1):
        total = float(row.sum())
        if total <= 1e-6:
            continue
        centres[index] = float((row * columns).sum() / total) / max(1, deltas.shape[1] - 1)
    return centres


def smoothed_centroid(centres: np.ndarray, window: int = 9) -> np.ndarray:
    """Median-filtered centroid, so the crop does not twitch frame to frame."""
    if len(centres) == 0:
        return centres
    half = max(1, window // 2)
    smoothed = np.empty_like(centres)
    for index in range(len(centres)):
        low = max(0, index - half)
        high = min(len(centres), index + half + 1)
        smoothed[index] = float(np.median(centres[low:high]))
    return smoothed


def crop_window_for_centroid(
    centre: float, width: int, height: int, target_ratio: float = 9 / 16
) -> dict:
    """9:16 crop rectangle that keeps ``centre`` inside the frame.

    When the source is wider than 9:16 the crop keeps full height and slides
    horizontally; for a taller source it keeps full width instead.
    """
    if width <= 0 or height <= 0:
        return {"x": 0, "y": 0, "width": width, "height": height}
    if width / height > target_ratio:
        crop_h = height
        crop_w = max(2, int(round(height * target_ratio)))
    else:
        crop_w = width
        crop_h = max(2, int(round(width / target_ratio)))
    centre = float(min(1.0, max(0.0, centre)))
    x = int(round(centre * width - crop_w / 2))
    x = max(0, min(width - crop_w, x))
    y = max(0, min(height - crop_h, int((height - crop_h) / 2)))
    # Even offsets keep every encoder happy.
    return {
        "x": x - (x % 2),
        "y": y - (y % 2),
        "width": crop_w - (crop_w % 2),
        "height": crop_h - (crop_h % 2),
    }


def build_windows(
    times: np.ndarray,
    scores: np.ndarray,
    *,
    source_duration: float,
    cuts: Optional[np.ndarray] = None,
    max_candidates: int = 8,
    target_duration: float = DEFAULT_TARGET_DURATION,
    min_duration: float = MIN_CANDIDATE_DURATION,
    max_duration: float = MAX_CANDIDATE_DURATION,
) -> list[dict]:
    """Turn a per-sample score series into ranked, non-overlapping windows."""
    if len(times) == 0 or len(scores) == 0:
        return []
    target = float(np.clip(target_duration, min_duration, max_duration))
    cut_times = list(np.asarray(cuts, dtype=float)) if cuts is not None else []

    order = np.argsort(scores)[::-1]
    chosen: list[dict] = []
    for index in order:
        if len(chosen) >= max_candidates:
            break
        score = float(scores[index])
        if score <= 0:
            continue
        peak = float(times[index])
        start = max(0.0, peak - target / 2.0)
        end = min(float(source_duration or peak + target / 2.0), start + target)
        start = max(0.0, end - target)
        # Snap to a nearby shot boundary so a window does not open mid-cut.
        near = [value for value in cut_times if abs(value - start) <= 2.0]
        if near:
            start = max(0.0, min(near, key=lambda value: abs(value - start)))
            end = min(float(source_duration or start + target), start + target)
        if end - start < min_duration:
            continue
        if any(start < other["end_s"] and end > other["start_s"] for other in chosen):
            continue
        chosen.append(
            {
                "start_s": round(start, 2),
                "end_s": round(end, 2),
                "peak_s": round(peak, 2),
                "score": score,
                "sample_index": int(index),
            }
        )
    chosen.sort(key=lambda item: item["start_s"])
    return chosen


def evidence_times(window: dict) -> list[float]:
    """Frames a reviewer should look at for this window, deduplicated."""
    start, end = window["start_s"], window["end_s"]
    span = max(0.1, end - start)
    raw = [start + 0.05 * span, window["peak_s"], start + 0.55 * span, end - 0.05 * span]
    result: list[float] = []
    for value in raw:
        rounded = round(max(start, min(end, value)), 2)
        if not any(abs(rounded - seen) < 0.4 for seen in result):
            result.append(rounded)
    return result


# --------------------------------------------------------------------------
# ffmpeg probing (local, no network)
# --------------------------------------------------------------------------


def _run(command: Sequence[str], *, capture: bool = True) -> bytes:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            f"ffmpeg failed ({result.returncode}): {message[-1] if message else 'no output'}"
        )
    return result.stdout or b""


def probe_size(path: Path) -> tuple[int, int]:
    """(width, height) of the first video stream, or (0, 0) if unreadable."""
    output = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ]
    )
    parts = output.decode().strip().split(",")[:2]
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def probe_duration(path: Path) -> float:
    output = _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ]
    )
    try:
        return float(output.decode().strip())
    except ValueError:
        return 0.0


def keyframe_times(path: Path) -> np.ndarray:
    output = _run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
            "-of", "csv=p=0", str(path),
        ]
    )
    values = []
    for line in output.decode("utf-8", "replace").splitlines():
        line = line.strip().strip(",")
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.asarray(sorted(set(values)), dtype=float)


def thumbnails(path: Path, *, fps: Optional[float] = None, keyframes_only: bool = False, start: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Decode tiny grey frames, either keyframe-only or at a fixed rate."""
    command = ["ffmpeg", "-v", "error"]
    if start:
        command += ["-ss", f"{start:.3f}"]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    if keyframes_only:
        command += ["-skip_frame", "nokey"]
    command += ["-i", str(path)]
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    filters += [f"scale={THUMB_WIDTH}:{THUMB_HEIGHT}", "format=gray"]
    command += ["-vsync", "0", "-vf", ",".join(filters), "-f", "rawvideo", "-"]
    return frames_from_raw(_run(command))


def audio_levels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    output = _run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vn",
            "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "s16le", "-",
        ]
    )
    samples = np.frombuffer(output, dtype=np.int16)
    return audio_envelope(samples)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _coarse_scan(path: Path, duration: float, speech_windows: Sequence[tuple[float, float]]) -> ScanTier:
    times = keyframe_times(path)
    frames = thumbnails(path, keyframes_only=True)
    audio_times, audio_values = audio_levels(path)
    if len(times) != len(frames):
        # Keyframe probing and decoding disagree (rare, but a cut or odd stream
        # can trigger it): fall back to evenly spaced times instead of pairing
        # signals with the wrong second.
        if len(frames) < 2:
            return ScanTier(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), "coarse", duration, {"reason": "no keyframes"})
        times = np.linspace(0.0, duration or float(len(frames) - 1), len(frames))
    differences = frame_difference(frames)
    motion = normalize(differences)
    cuts = cut_spikes(differences)
    audio = resample_to(audio_times, audio_values, times)
    return ScanTier(
        times=times,
        motion=motion,
        cuts=cuts,
        audio=audio,
        mode="coarse",
        source_duration=duration,
        detail={
            "frames": int(len(frames)),
            "audio_windows": int(len(audio_times)),
            "speech": speech_mask(times, speech_windows).mean() if len(times) else 0.0,
        },
    )


def _refine_window(path: Path, window: dict) -> dict:
    """Re-measure one window at a higher frame rate to place the peak.

    These numbers describe the window relative to itself (how much movement and
    how sharp the hardest cut is inside it); they are not on the coarse global
    scale, so they inform the reviewer rather than re-rank the list.
    """
    start, end = window["start_s"], window["end_s"]
    frames = thumbnails(path, fps=REFINE_FPS, start=start, duration=max(0.5, end - start))
    if len(frames) < 3:
        return {"frames": int(len(frames)), "fps": 0.0}
    differences = frame_difference(frames)
    return {
        "motion_rel": float(normalize(differences).mean()),
        "cut_peak": float(cut_spikes(differences).max()),
        "static_share": static_share(frames),
        "flat_share": flat_share(frames),
        "fps": REFINE_FPS,
        "frames": int(len(frames)),
    }


def analyze_source(
    path: Path,
    *,
    source_id: Optional[str] = None,
    source_url: Optional[str] = None,
    source_type: str = "authorized_fixture",
    rights_status: str = "verify_before_use",
    speech_windows: Sequence[tuple[float, float]] = (),
    max_candidates: int = 8,
    target_duration: float = DEFAULT_TARGET_DURATION,
    refine: bool = True,
) -> dict:
    """Scan one source and return ranked candidates plus their evidence."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    duration = probe_duration(path)
    tier = _coarse_scan(path, duration, speech_windows)
    if len(tier.times) == 0:
        return {
            "schema_status": SCHEMA_STATUS,
            "mode": MODE,
            "source_id": source_id or path.stem,
            "source_url": source_url,
            "source_type": source_type,
            "rights_status": rights_status,
            "duration_s": round(duration, 2),
            "scan": {"reason": "no readable video frames", **tier.detail},
            "candidates": [],
        }

    speech = speech_mask(tier.times, speech_windows)
    scores = np.zeros(len(tier.times), dtype=float)
    for index in range(len(tier.times)):
        event = classify_event(
            float(tier.motion[index]),
            float(tier.cuts[index]),
            float(tier.audio[index]),
            float(speech[index]),
        )
        scores[index] = score_window(
            float(tier.motion[index]),
            float(tier.cuts[index]),
            float(tier.audio[index]),
            float(speech[index]),
            event,
        )

    cut_times = tier.times[tier.cuts >= 0.5]
    windows = build_windows(
        tier.times,
        scores,
        source_duration=duration,
        cuts=cut_times,
        max_candidates=max_candidates,
        target_duration=target_duration,
    )

    candidates = []
    for window in windows:
        index = window["sample_index"]
        motion = float(tier.motion[index])
        cut = float(tier.cuts[index])
        audio = float(tier.audio[index])
        spoke = float(speech[index])
        event = classify_event(motion, cut, audio, spoke)
        refined = _refine_window(path, window) if refine else {}
        raw_score = window["score"]
        score = raw_score
        card = is_card(refined.get("static_share", 0.0), refined.get("flat_share", 0.0))
        if card:
            # A slate or black card is not a moment: keep it in the report for
            # transparency, but rank it below real footage.
            event = "static_card"
            score = round(score * CARD_DEMOTION, 1)
        candidates.append(
            {
                "schema_status": SCHEMA_STATUS,
                "source_id": source_id or path.stem,
                "source_url": source_url,
                "source_type": source_type,
                "rights_status": rights_status,
                "start_s": window["start_s"],
                "end_s": window["end_s"],
                "duration_s": round(window["end_s"] - window["start_s"], 2),
                "mode": MODE,
                "event_type": event,
                "reason": (
                    f"{event}: motion {motion:.2f}, cut {cut:.2f}, audio {audio:.2f}"
                    + (
                        f", inside the window cut peak {refined['cut_peak']:.2f}"
                        if refined.get("cut_peak") is not None
                        else ""
                    )
                    + (
                        f"; {refined['static_share']:.0%} of the window is a held frame"
                        if card
                        else ""
                    )
                    + ". Signals only; the event itself is not verified."
                ),
                "score": score,
                "score_raw": raw_score,
                "score_kind": "heuristic_rank",
                "evidence_frames_s": evidence_times(window),
                "signals": {
                    "motion": round(motion, 3),
                    "cut": round(cut, 3),
                    "audio": round(audio, 3),
                    "speech": round(spoke, 3),
                    "refined": {
                        key: round(value, 3) if isinstance(value, float) else value
                        for key, value in refined.items()
                    },
                },
                "audio_lang": None,
                "subtitle_lang": None,
                "caption_review": "pending",
                "review_status": "pending",
                "publish_status": "not_published",
            }
        )

    # Rank by the final score (after any card demotion), then present the list in
    # reading order so a reviewer can walk the source front to back.
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    for position, candidate in enumerate(ranked, start=1):
        candidate["rank"] = position
    candidates = sorted(ranked, key=lambda item: item["start_s"])

    return {
        "schema_status": SCHEMA_STATUS,
        "mode": MODE,
        "source_id": source_id or path.stem,
        "source_url": source_url,
        "source_type": source_type,
        "rights_status": rights_status,
        "duration_s": round(duration, 2),
        "scan": {
            "tier_a": "keyframes",
            **tier.detail,
            "cut_count": int((tier.cuts >= 0.5).sum()),
            "speech_share": round(float(speech.mean()) if len(speech) else 0.0, 3),
        },
        "weights": dict(EVENT_WEIGHTS),
        "disclaimer": (
            "Scores rank candidates for review. They are not evidence that a game "
            "event happened; audio peaks are not mechanics and frame difference "
            "cannot name what it saw. Look at the evidence frames."
        ),
        "candidates": candidates,
    }


def _parse_windows(values: Iterable[str]) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for value in values or ():
        try:
            start, end = value.replace("-", ":", 1).split(":", 1)
            windows.append((float(start), float(end)))
        except ValueError:
            raise SystemExit(f"Unreadable speech window {value!r}; expected START:END seconds")
    return windows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Local speech-independent moment scan")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--source-id")
    parser.add_argument("--source-url")
    parser.add_argument("--source-type", default="authorized_fixture")
    parser.add_argument("--rights-status", default="verify_before_use")
    parser.add_argument("--speech", action="append", default=[], metavar="START:END")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--target-duration", type=float, default=DEFAULT_TARGET_DURATION)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args(argv)

    report = analyze_source(
        args.source,
        source_id=args.source_id,
        source_url=args.source_url,
        source_type=args.source_type,
        rights_status=args.rights_status,
        speech_windows=_parse_windows(args.speech),
        max_candidates=args.max_candidates,
        target_duration=args.target_duration,
        refine=not args.no_refine,
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out} ({len(report['candidates'])} candidates)")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
