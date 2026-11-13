"""Provider jobs survive process boundaries via task-owned durable state."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.media import transcription
from tests.unit.test_assemblyai_polling import response


def test_restart_after_poll_failure_does_not_submit_again(monkeypatch, tmp_path):
    calls = []
    fail_poll = True
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith('/upload'):
            return httpx.Response(200, json={'upload_url': 'https://example.test/audio'})
        if request.method == 'POST':
            return response('queued')
        if fail_poll:
            return httpx.Response(403)
        return response('completed')
    media = tmp_path / 'audio.mp3'
    media.write_bytes(b'fake audio')
    state = {}
    def saved(job):
        state.clear()
        state.update(job)
    monkeypatch.setattr(transcription.time, 'sleep', lambda _: None)
    config = transcription.aai.TranscriptionConfig(speech_models=['universal-2'])
    def run():
        client = transcription.aai.Client(settings=transcription.aai.Settings(api_key='test'))
        client.http_client.close()
        client._http_client = httpx.Client(base_url='https://api.assemblyai.com', transport=httpx.MockTransport(handler))
        try:
            return transcription._submit_and_wait_for_assemblyai_transcript(
                transcription.aai.Transcriber(client=client), media, config, 10,
                job_state=dict(state), job_callback=saved,
            )
        finally:
            client.http_client.close()
    with pytest.raises(httpx.HTTPStatusError):
        run()
    fail_poll = False
    assert run().status == 'completed'
    assert sum(method == 'POST' and path.endswith('/transcript') for method, path in calls) == 1
    assert sum(path.endswith('/upload') for _, path in calls) == 1


def test_ambiguous_submission_blocks_automatic_repayment(monkeypatch, tmp_path):
    state = {}
    transcriber = SimpleNamespace(upload_file=Mock(return_value='audio'),
                                  submit=Mock(side_effect=httpx.ReadTimeout('lost response')))
    def saved(job):
        state.update(job)
    for attempt in range(2):
        with pytest.raises((httpx.ReadTimeout, RuntimeError)):
            transcription._submit_and_wait_for_assemblyai_transcript(
                transcriber, tmp_path / 'audio.mp3', None, 10,
                job_state=dict(state), job_callback=saved,
            )
    transcriber.submit.assert_called_once()
    assert state['state'] == 'submitting'


def test_checkpoint_must_commit_before_paid_submission(tmp_path):
    transcriber = SimpleNamespace(upload_file=Mock(return_value='audio'), submit=Mock())
    def unavailable(_):
        raise OSError('checkpoint unavailable')
    with pytest.raises(OSError):
        transcription._submit_and_wait_for_assemblyai_transcript(
            transcriber, tmp_path / 'audio.mp3', None, 10, job_state={}, job_callback=unavailable,
        )
    transcriber.submit.assert_not_called()
