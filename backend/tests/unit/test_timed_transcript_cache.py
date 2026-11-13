from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.timed_transcript_cache_repository import (
    TimedTranscriptCacheRepository,
    file_content_fingerprint,
    timed_transcript_provenance,
)
from src.media.common import TRANSCRIPT_CACHE_SCHEMA_VERSION
from src.services.video_service import (
    _analysis_cache_for_source,
    _checkpoint_source_mismatch,
    _select_transcript_pair,
    _transcript_cache_for_source,
)


def test_provenance_changes_with_provider_language_and_model():
    config = SimpleNamespace(
        transcription_provider="assemblyai",
        assembly_ai_speech_model="universal-2",
        fast_mode_transcript_model="nano",
        transcription_language="en",
        whisper_model="base",
    )
    assembly = timed_transcript_provenance(config, "balanced")
    assert timed_transcript_provenance(config, "fast") == assembly
    config.assembly_ai_speech_model = ""
    fast_default = timed_transcript_provenance(config, "fast")
    assert fast_default["model"] == "nano"
    assert fast_default != assembly
    config.transcription_provider = "whisper"
    assert timed_transcript_provenance(config, "balanced") != assembly
    config.transcription_provider = "assemblyai"
    config.transcription_language = "fr"
    assert timed_transcript_provenance(config, "balanced") != assembly


def test_fast_mode_uses_assembly_model_override_when_configured():
    config = SimpleNamespace(
        transcription_provider="assemblyai",
        assembly_ai_speech_model="universal-2",
        fast_mode_transcript_model="nano",
        transcription_language="",
    )
    assert timed_transcript_provenance(config, "fast")["model"] == "universal-2"


def test_content_fingerprint_changes_when_source_bytes_change(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"version one")
    first = file_content_fingerprint(source)
    assert file_content_fingerprint(source) == first
    source.write_bytes(b"version two")
    assert file_content_fingerprint(source) != first


def test_checkpoint_text_does_not_take_timing_data_from_another_cache_entry():
    unrelated_timing = {
        "text": "different source words",
        "words": [{"text": "different", "start": 0, "end": 100}],
    }
    selected = _select_transcript_pair(
        "checkpoint text",
        None,
        unrelated_timing,
        "shared cached text",
    )
    assert selected == ("checkpoint text", None, False)


def test_timed_cache_uses_its_own_text_and_words_as_a_pair():
    timed_payload = {
        "text": "paired words",
        "words": [{"text": "paired", "start": 0, "end": 100}],
    }
    selected = _select_transcript_pair(None, None, timed_payload, "other text")
    assert selected == ("paired words", timed_payload, True)


def test_legacy_checkpoint_is_preserved_and_fingerprinted_mismatch_is_detected():
    assert not _checkpoint_source_mismatch("youtube", "legacy text", None, "new-hash")
    assert _checkpoint_source_mismatch("youtube", "saved text", "old-hash", "new-hash")
    assert _checkpoint_source_mismatch("twitch", "saved text", "old-hash", "new-hash")


def test_content_addressed_youtube_source_does_not_use_unproven_url_analysis():
    assert _analysis_cache_for_source("youtube", "new-content-hash", '{"old":true}') is None
    assert _analysis_cache_for_source("twitch", None, '{"legacy":true}') == '{"legacy":true}'
    assert _transcript_cache_for_source("youtube", "new-content-hash", "old text") is None
    assert _transcript_cache_for_source("youtube", "new-content-hash", "old text") is None


@pytest.mark.asyncio
async def test_cache_lookup_is_scoped_to_user_and_provenance_and_requires_word_timings():
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = {
        "version": TRANSCRIPT_CACHE_SCHEMA_VERSION,
        "words": [{"text": "hello", "start": 0, "end": 100}],
        "text": "hello",
    }
    db.execute.return_value = result
    provenance = {"provider": "assemblyai", "language": "en", "version": 1}

    payload = await TimedTranscriptCacheRepository.get(
        db, "user-a", "fingerprint", provenance
    )

    assert payload["words"]
    parameters = db.execute.await_args.args[1]
    assert parameters["user_id"] == "user-a"
    assert parameters["provenance"] == (
        '{"language": "en", "provider": "assemblyai", "version": 1}'
    )

    result.scalar_one_or_none.return_value = {
        "version": TRANSCRIPT_CACHE_SCHEMA_VERSION,
        "words": [],
        "text": "no timings",
    }
    assert (
        await TimedTranscriptCacheRepository.get(
            db, "user-b", "fingerprint", provenance
        )
        is None
    )
    assert db.execute.await_args.args[1]["user_id"] == "user-b"


@pytest.mark.asyncio
async def test_cache_save_refuses_transcripts_without_timing_words():
    db = AsyncMock()
    await TimedTranscriptCacheRepository.save(
        db,
        "user-a",
        "fingerprint",
        {"version": 1},
        {
            "version": TRANSCRIPT_CACHE_SCHEMA_VERSION,
            "text": "plain text",
            "words": [],
        },
    )
    db.execute.assert_not_awaited()

@pytest.mark.asyncio
async def test_cache_save_rejects_invalid_or_unordered_word_timing():
    db = AsyncMock()
    await TimedTranscriptCacheRepository.save(
        db,
        "user-a",
        "fingerprint",
        {"version": 1},
        {
            "version": TRANSCRIPT_CACHE_SCHEMA_VERSION,
            "text": "hello world",
            "words": [
                {"text": "hello", "start": 500, "end": 700},
                {"text": "world", "start": 100, "end": 200},
            ],
        },
    )
    db.execute.assert_not_awaited()
