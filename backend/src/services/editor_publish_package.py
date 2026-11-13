"""Build a publish ZIP from one completed editor export job snapshot."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..clip_archive import ArchiveEntry
from ..editor_document import EditDocument, caption_chunk_starts, mapped_words
from ..media.srt import build_srt
from ..media_storage import MediaStorage


def _srt(document: EditDocument) -> tuple[str | None, str | None]:
    if not document.captions.enabled:
        return None, "captions_disabled"
    words = mapped_words(document)
    if not words:
        return None, "no_caption_words"
    starts = caption_chunk_starts(words, document.captions.wordsPerLine, document.translation)
    groups = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(words)
        group = words[start:stop]
        text = " ".join(word["text"] for word in group)
        groups.append({
            "text": text.upper() if document.captions.uppercase else text,
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })
    try:
        content = build_srt(groups)
    except ValueError:
        return None, "invalid_caption_timings"
    return (content, None) if content else (None, "no_caption_groups")


def _export_source(session, storage: MediaStorage, job_id: str) -> tuple[str, int]:
    name = f"export-{job_id}.mp4"
    local = session.directory / name
    # The worker removes its local copy after publishing the completed job.
    # Prefer the durable object whenever storage is enabled to avoid a race.
    if storage.enabled:
        reference = storage.editor_reference(session.directory, name)
        size = storage.object_size(reference)
        if size is not None:
            return reference, size
        raise FileNotFoundError("Completed export media is unavailable")
    if local.is_file():
        return str(local), local.stat().st_size
    reference = storage.editor_reference(session.directory, name)
    if storage.bucket and not local.exists():
        size = storage.object_size(reference)
        if size is not None:
            return reference, size
    raise FileNotFoundError("Completed export media is unavailable")


def _extract_cover(source_ref: str, target: Path) -> bool:
    input_path = Path(source_ref)
    # A remote multi-gigabyte export must not be staged locally just to make
    # an optional cover image. The MP4 itself remains streamed into the ZIP.
    if not input_path.is_file():
        return False
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(input_path), "-frames:v", "1", "-q:v", "2", str(target)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and target.is_file() and target.stat().st_size > 0


def prepare_publish_package(
    session,
    storage: MediaStorage,
    *,
    task_id: str,
    clip_id: str,
    job_id: str,
    job_request: dict[str, Any],
    filename: str,
    source_title: str | None = None,
    clip_title: str | None = None,
    social_caption: str | None = None,
    social_hashtags: str | None = None,
) -> tuple[tempfile.TemporaryDirectory, list[ArchiveEntry], dict[str, Any]]:
    """Prepare sidecar files and entries; caller owns the temporary directory."""
    document = EditDocument.model_validate(job_request["document"])
    export_ref, export_size = _export_source(session, storage, job_id)
    temporary = tempfile.TemporaryDirectory(prefix=f"publish-{job_id}-", dir=session.directory)
    scratch = Path(temporary.name)
    entries = [ArchiveEntry(filename, export_ref, export_size)]
    missing: list[dict[str, str]] = []
    try:
        for name, content, reason in (
            ("social_caption.txt", social_caption, "no_caption_saved_for_export_job"),
            ("social_hashtags.txt", social_hashtags, "no_hashtags_supplied"),
        ):
            if content is None or not content.strip():
                missing.append({"file": name, "reason": reason})
            else:
                sidecar = scratch / name
                sidecar.write_text(content, encoding="utf-8", newline="")
                entries.append(ArchiveEntry(name, str(sidecar), sidecar.stat().st_size))

        subtitles, subtitle_error = _srt(document)
        if subtitles is None:
            missing.append({"file": "captions.srt", "reason": subtitle_error or "unavailable"})
        else:
            subtitle_path = scratch / "captions.srt"
            subtitle_path.write_text(subtitles, encoding="utf-8")
            entries.append(ArchiveEntry("captions.srt", str(subtitle_path), subtitle_path.stat().st_size))

        cover_path = scratch / "cover.jpg"
        try:
            has_cover = _extract_cover(export_ref, cover_path)
        except (OSError, subprocess.SubprocessError, RuntimeError):
            has_cover = False
        if has_cover:
            entries.append(ArchiveEntry("cover.jpg", str(cover_path), cover_path.stat().st_size))
        else:
            reason = "export_not_local_for_cover" if not Path(export_ref).is_file() else "frame_extraction_failed"
            missing.append({"file": "cover.jpg", "reason": reason})

        existing = {entry.name for entry in entries}
        manifest = {
            "format": "supoclip-publish-package-v1",
            "task_id": task_id,
            "clip_id": clip_id,
            "export_job_id": job_id,
            "preset": job_request.get("preset"),
            "draft_revision": job_request.get("draft_revision"),
            "source_title": source_title,
            "title": clip_title,
            "metadata_missing": [
                field for field, value in (("source_title", source_title), ("title", clip_title))
                if not value
            ],
            "files": [
                {"file": filename, "status": "included"},
                *[{"file": name, "status": "included"} for name in ("captions.srt", "cover.jpg", "social_caption.txt", "social_hashtags.txt") if name in existing],
                *[{"file": item["file"], "status": "missing", "reason": item["reason"]} for item in missing],
                {"file": "manifest.json", "status": "included"},
            ],
            "missing": missing,
        }
        manifest_path = scratch / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entries.append(ArchiveEntry("manifest.json", str(manifest_path), manifest_path.stat().st_size))
        return temporary, entries, manifest
    except BaseException:
        temporary.cleanup()
        raise
