"""Names shown to users when they download generated clips."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

MAX_FILENAME_CHARS = 120
_MAX_PRESET_SUFFIX_CHARS = 21  # underscore plus the bounded preset name
_INVALID = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
_SPACE = re.compile(r"\s+")
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE)


def _part(value: Any, limit: int) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _SPACE.sub(" ", _INVALID.sub(" ", text)).strip(" .")
    if not any(character.isalnum() for character in text):
        return ""
    return _whole_words(text, limit)


def _whole_words(text: str, limit: int) -> str:
    """Keep complete words only, even when the limit cuts through one."""
    kept: list[str] = []
    used = 0
    for word in text.split():
        extra = len(word) + (1 if kept else 0)
        if used + extra > limit:
            break
        kept.append(word)
        used += extra
    return " ".join(kept)


def clip_download_filename(
    clip: dict[str, Any],
    source_title: str | None = None,
    *,
    position: int | None = None,
    preset: str | None = None,
) -> str:
    """Build a bounded, cross-platform filename without changing stored paths.

    ``position`` is the one-based display number within a task. ``clip_order``
    is the stable fallback because repository results are ordered by it.
    """
    if position is None:
        order = clip.get("clip_order")
        position = order if isinstance(order, int) and order >= 1 else 1
    number = max(1, position)
    hook = _part(clip.get("hook_title"), 64) or "Clip"
    source = _part(source_title, 56)
    preset_label = _part(preset, 20)
    suffix = f"_{preset_label}" if preset_label else ""
    prefix = f"clip_{number:02d} - "
    parts = [hook]
    if source:
        parts.append(source)
    # Reserve the full preset allowance for every form so adding a preset never
    # changes the readable base shared with the ZIP and original download.
    ending = f"{'_' * _MAX_PRESET_SUFFIX_CHARS}.mp4"
    room = MAX_FILENAME_CHARS - len(prefix) - len(ending)
    label = _whole_words(" - ".join(parts), room).rstrip(" -") or "Clip"
    stem = f"{prefix}{label}{suffix}"
    if _RESERVED.match(stem):
        stem = f"_{stem}"
    return f"{stem}.mp4"


def clip_position(clip: dict[str, Any], clips: list[dict[str, Any]]) -> int:
    """Return the one-based task position for a clip, with a safe fallback."""
    for index, candidate in enumerate(clips, start=1):
        if candidate.get("id") == clip.get("id"):
            return index
    order = clip.get("clip_order")
    if isinstance(order, int) and order >= 1:
        return order
    return 1
