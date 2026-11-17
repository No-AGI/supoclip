"""Bilingual streams must not be forced into one language, and a near-empty
transcript must say so instead of exporting a silent starter clip."""
from pathlib import Path
from types import SimpleNamespace

from src.config import Config
from src.media import transcription
from src.repositories.timed_transcript_cache_repository import timed_transcript_provenance


def test_language_list_drops_redirects_and_duplicates(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_CODE_SWITCHING_LANGUAGES", "ru, en-US, auto, ru, zzz")
    assert Config().transcription_code_switching_languages == ["ru", "en"]


def test_language_list_is_empty_by_default(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTION_CODE_SWITCHING_LANGUAGES", raising=False)
    assert Config().transcription_code_switching_languages == []


def test_code_switching_options_request_both_languages():
    config = SimpleNamespace(transcription_code_switching_languages=["ru", "en"])
    options = transcription._assemblyai_language_detection_options(config)
    assert options.code_switching is True
    assert options.expected_languages == ["ru", "en"]
    # A confident wrong single language must not fail the paid job.
    assert options.fallback_language == "ru"
    assert options.on_low_language_confidence == "fallback"


def test_single_language_path_sends_no_options():
    config = SimpleNamespace(transcription_code_switching_languages=[])
    assert transcription._assemblyai_language_detection_options(config) is None
    assert transcription._assemblyai_language_detection_options(SimpleNamespace()) is None


def test_thin_transcript_for_long_audio_is_reported():
    words = [{"start": 52_082, "end": 66_548}]
    warning = transcription._transcript_coverage_warning(535.0, 217, words)
    assert warning is not None
    assert "217 chars for 535s" in warning
    assert "speech between 52s and 67s" in warning


def test_plausible_or_short_audio_is_not_reported():
    assert transcription._transcript_coverage_warning(535.0, 8000, []) is None
    assert transcription._transcript_coverage_warning(60.0, 217, []) is None
    assert transcription._transcript_coverage_warning(None, 217, []) is None


def test_provenance_identity_is_unchanged_without_code_switching():
    config = SimpleNamespace(
        transcription_provider="assemblyai",
        assembly_ai_speech_model="",
        transcription_language="",
        transcription_code_switching_languages=[],
    )
    provenance = timed_transcript_provenance(config, "fast")
    assert "code_switching_languages" not in provenance
    assert provenance["language"] == ""


def test_provenance_separates_bilingual_caches():
    config = SimpleNamespace(
        transcription_provider="assemblyai",
        assembly_ai_speech_model="",
        transcription_language="",
        transcription_code_switching_languages=["ru", "en"],
    )
    provenance = timed_transcript_provenance(config, "fast")
    assert provenance["code_switching_languages"] == "ru,en"


def test_default_request_keeps_single_language_detection(monkeypatch):
    """Turning the feature off must not change the submitted configuration."""
    captured = {}

    class FakeTranscriber:
        def __init__(self, client=None):
            self.client = client

    def fake_submit(transcriber, media_path, config_obj, budget):
        captured["config"] = config_obj
        return None

    monkeypatch.setattr(transcription.aai, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: path)
    monkeypatch.setattr(
        transcription, "_submit_and_wait_for_assemblyai_transcript", fake_submit
    )
    runtime_config = SimpleNamespace(
        assembly_ai_api_key="test-key",
        assembly_ai_speech_model="",
        assembly_ai_request_timeout_seconds=60,
        assembly_ai_http_timeout_seconds=60,
        transcription_language="",
        transcription_code_switching_languages=[],
    )
    try:
        transcription._get_transcript_with_assemblyai(
            Path("audio.mp3"), "universal", runtime_config
        )
    except RuntimeError:
        pass  # the stubbed submit returns no transcript; only the config matters
    config_obj = captured["config"]
    assert config_obj.language_detection is True
    assert config_obj.language_detection_options is None
