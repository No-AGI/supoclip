"""Speech-driven moment candidates: the second channel of the stream scan.

The visual scan measures motion, cuts and loudness, so a quiet cutscene or a
calm conversation scores near zero even when it is the most watchable part of a
stream. This module reads the word-timed transcript instead and ranks windows by
how much is being said, how emphatic it is, and whether the speaker reacts.

It is deliberately dumb and local: no model, no network, no opinion about the
game. It exists so the visual channel can be compared with a text channel on the
same labelled intervals, and so a stream with speech is not scored as if it were
a silent montage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_HOP_SECONDS = 4.0
# A window with almost no words is not a moment of speech even if it is loud.
MIN_WORDS = 4
EXCITED = re.compile(r"[!?]")
# Long words are usually names and swearing in Russian streams, both of which
# mark emphasis rather than prose.
LONG_WORD = 6


def _word_seconds(word: dict) -> tuple[float, float]:
    return float(word["start"]) / 1000.0, float(word["end"]) / 1000.0


def window_scores(
    words: Sequence[dict],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
) -> list[dict]:
    """Score every overlapping window of the transcript by speech activity.

    ``score`` is a 0..1 mix of words per second, emphatic punctuation and long
    words. Windows with fewer than ``MIN_WORDS`` words are dropped, so silence
    and the gaps between scenes never become candidates.
    """
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window and hop must be positive")
    timed = []
    for word in words:
        try:
            begin, finish = _word_seconds(word)
        except (KeyError, TypeError, ValueError):
            continue
        if finish < begin:
            continue
        timed.append((begin, finish, str(word.get("text", ""))))
    if not timed:
        return []

    first = start_s if start_s is not None else timed[0][0]
    last = end_s if end_s is not None else timed[-1][1]
    if last <= first:
        raise ValueError("end must be after start")

    results = []
    position = first
    # The last window is clipped to the end of the span instead of being
    # dropped: a short transcript must still produce its one honest candidate.
    while position < last - 1e-9:
        begin = position
        finish = min(position + window_seconds, last)
        inside = [item for item in timed if item[0] < finish and item[1] > begin]
        position += hop_seconds
        if len(inside) < MIN_WORDS:
            continue
        span = max(1e-6, finish - begin)
        words_per_second = len(inside) / span
        excited = sum(1 for item in inside if EXCITED.search(item[2]))
        long_words = sum(1 for item in inside if len(item[2]) >= LONG_WORD)
        # 2.5 words/s is already continuous commentary in Russian, so it saturates
        # the density term instead of rewarding a shout for being a shout.
        density = min(1.0, words_per_second / 2.5)
        emphasis = min(1.0, (excited + long_words * 0.5) / max(1.0, len(inside) * 0.6))
        score = round(100.0 * (0.6 * density + 0.4 * emphasis), 1)
        results.append(
            {
                "start_s": round(begin, 2),
                "end_s": round(finish, 2),
                "score": score,
                "words": len(inside),
                "words_per_second": round(words_per_second, 2),
                "excited_words": excited,
                "speech_s": round(sum(min(item[1], finish) - max(item[0], begin) for item in inside), 2),
            }
        )
    return results


def rank_windows(scored: Iterable[dict], *, max_candidates: int = 8) -> list[dict]:
    """Take the best non-overlapping windows, best first."""
    chosen: list[dict] = []
    for candidate in sorted(scored, key=lambda item: item["score"], reverse=True):
        if len(chosen) >= max_candidates:
            break
        if any(
            candidate["start_s"] < other["end_s"] and candidate["end_s"] > other["start_s"]
            for other in chosen
        ):
            continue
        chosen.append(dict(candidate))
    for rank, candidate in enumerate(chosen, start=1):
        candidate["rank"] = rank
    return chosen


def analyze_transcript(
    payload: dict,
    *,
    max_candidates: int = 8,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    source_id: Optional[str] = None,
) -> dict:
    """Candidates for one transcript cache file (the pipeline's own format)."""
    words = payload.get("words") or []
    scored = window_scores(
        words,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        start_s=start_s,
        end_s=end_s,
    )
    candidates = rank_windows(scored, max_candidates=max_candidates)
    for candidate in candidates:
        candidate["event_type"] = "speech_scene"
        candidate["score_kind"] = "speech_heuristic_rank"
        candidate["reason"] = (
            f"speech_scene: {candidate['words']} words in {window_seconds:.0f}s "
            f"({candidate['words_per_second']} w/s), "
            f"{candidate['excited_words']} emphatic. Text signals only."
        )
    return {
        "schema_status": "prototype",
        "mode": "speech_events",
        "source_id": source_id,
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "words_available": len(words),
        "candidates": candidates,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Speech-driven moment candidates")
    parser.add_argument("transcript", type=Path, help=".transcript_cache.json with words")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    parser.add_argument("--window", metavar="START:END")
    args = parser.parse_args(argv)

    payload = json.loads(args.transcript.read_text(encoding="utf-8"))
    start_s = end_s = None
    if args.window:
        try:
            start_text, end_text = args.window.split(":", 1)
            start_s, end_s = float(start_text), float(end_text)
        except ValueError:
            parser.error("--window takes START:END in seconds")
        if end_s <= start_s or start_s < 0:
            parser.error("--window needs START < END and START >= 0")

    report = analyze_transcript(
        payload,
        max_candidates=args.max_candidates,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        start_s=start_s,
        end_s=end_s,
        source_id=args.source_id or args.transcript.stem,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(report['candidates'])} candidates)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
