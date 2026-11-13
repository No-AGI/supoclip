"""Build SubRip subtitles from finalized caption groups and their timings.

This module deliberately does not derive captions from a transcript or translation.
Callers provide the exact text and timing for each final cue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CaptionGroup:
    """One final caption cue on the exported clip timeline, in seconds."""

    text: str
    start: float
    end: float


def _field(group: CaptionGroup | Mapping[str, Any], name: str) -> Any:
    if isinstance(group, CaptionGroup):
        return getattr(group, name)
    if isinstance(group, Mapping):
        try:
            return group[name]
        except KeyError as exc:
            raise ValueError(f"Caption group is missing {name!r}") from exc
    raise TypeError("Caption groups must be CaptionGroup values or mappings")


def _milliseconds(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Caption {field} must be a finite number")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Caption {field} must be a finite number") from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Caption {field} must be finite and non-negative")
    return round(seconds * 1000)


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(caption_groups: Iterable[CaptionGroup | Mapping[str, Any]]) -> str:
    """Return UTF-8-ready SRT text for ordered final groups.

    Each group must supply ``text``, ``start`` and ``end``. Times are seconds
    from the exported clip's beginning. Gaps and text are kept as supplied;
    cue times are rounded to the nearest millisecond as required by SRT.
    """
    cues: list[tuple[int, int, str]] = []
    previous_end = 0
    for number, group in enumerate(caption_groups, start=1):
        text = _field(group, "text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Caption {number} text must be a non-empty string")
        start = _milliseconds(_field(group, "start"), "start")
        end = _milliseconds(_field(group, "end"), "end")
        if end <= start:
            raise ValueError(f"Caption {number} end must follow its start at millisecond precision")
        if cues and start < previous_end:
            raise ValueError(f"Caption {number} overlaps or precedes the previous caption")
        cues.append((start, end, text))
        previous_end = end

    return "\n\n".join(
        f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}"
        for index, (start, end, text) in enumerate(cues, start=1)
    ) + ("\n" if cues else "")
