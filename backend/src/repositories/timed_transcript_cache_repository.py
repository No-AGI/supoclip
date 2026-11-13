"""User-owned cache for reusable, word-timed stable-source transcripts."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..media.common import TRANSCRIPT_CACHE_SCHEMA_VERSION


def timed_transcript_provenance(config: Any, processing_mode: str) -> dict[str, Any]:
    provider = getattr(config, "transcription_provider", "assemblyai")
    if provider == "whisper":
        model = getattr(config, "whisper_model", "base")
    else:
        model = getattr(config, "assembly_ai_speech_model", "")
        if not model:
            model = (
                getattr(config, "fast_mode_transcript_model", "nano")
                if processing_mode == "fast"
                else "best"
            )
    return {
        "version": 1,
        "transcript_schema": TRANSCRIPT_CACHE_SCHEMA_VERSION,
        "provider": provider,
        "model": model or "default",
        "language": getattr(config, "transcription_language", "") or "",
    }


def file_content_fingerprint(path: Path) -> str:
    """Fingerprint exact downloaded bytes; URL identity alone can go stale."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_timed_payload(payload: Any) -> bool:
    if (
        not isinstance(payload, dict)
        or payload.get("version") != TRANSCRIPT_CACHE_SCHEMA_VERSION
        or not isinstance(payload.get("text"), str)
        or not payload["text"].strip()
        or not isinstance(payload.get("words"), list)
        or not payload["words"]
    ):
        return False
    previous_start = -1
    for word in payload["words"]:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str) or not word["text"].strip():
            return False
        start, end = word.get("start"), word.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or start < previous_start
        ):
            return False
        previous_start = start
    return True


class TimedTranscriptCacheRepository:
    @staticmethod
    async def get(
        db: AsyncSession,
        user_id: str,
        source_fingerprint: str,
        provenance: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        result = await db.execute(
            text("""
            SELECT transcript_data FROM user_timed_transcript_cache
            WHERE user_id = :user_id AND source_fingerprint = :source_fingerprint
              AND provenance = CAST(:provenance AS jsonb)
            """),
            {
                "user_id": user_id,
                "source_fingerprint": source_fingerprint,
                "provenance": json.dumps(provenance, sort_keys=True),
            },
        )
        payload = result.scalar_one_or_none()
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not _valid_timed_payload(payload):
            return None
        return payload

    @staticmethod
    async def save(
        db: AsyncSession,
        user_id: str,
        source_fingerprint: str,
        provenance: dict[str, Any],
        transcript_data: dict[str, Any],
    ) -> None:
        if not _valid_timed_payload(transcript_data):
            return
        await db.execute(
            text("""
            INSERT INTO user_timed_transcript_cache
                (user_id, source_fingerprint, provenance, transcript_data, updated_at)
            VALUES (
                :user_id, :source_fingerprint, CAST(:provenance AS jsonb),
                CAST(:transcript_data AS jsonb), CURRENT_TIMESTAMP
            )
            ON CONFLICT (user_id, source_fingerprint, provenance)
            DO UPDATE SET transcript_data = EXCLUDED.transcript_data, updated_at = CURRENT_TIMESTAMP
            """),
            {
                "user_id": user_id,
                "source_fingerprint": source_fingerprint,
                "provenance": json.dumps(provenance, sort_keys=True),
                "transcript_data": json.dumps(transcript_data),
            },
        )
        await db.commit()
