"""Reproduce the legacy retry billing boundary without provider requests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import httpx
import pytest
from src.media import transcription


def test_uncheckpointed_retry_submits_twice(monkeypatch):
    """Documents why callers must supply a durable provider-job checkpoint."""
    client = httpx.Client(base_url='https://api.assemblyai.com', transport=httpx.MockTransport(lambda _: httpx.Response(403)))
    transcriber = SimpleNamespace(upload_file=Mock(return_value='audio'), submit=Mock(return_value=SimpleNamespace(id='job', _client=SimpleNamespace(http_client=client))))
    try:
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                transcription._submit_and_wait_for_assemblyai_transcript(transcriber, Path('audio.mp3'), None, 10)
    finally:
        client.close()
    assert transcriber.submit.call_count == 2
