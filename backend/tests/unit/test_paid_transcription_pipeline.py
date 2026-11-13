"""Exercise async-to-thread durable paid job bridge through the real pipeline."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.media import transcription
from src.services.video_service import VideoService
from tests.unit.test_processing_checkpoints import pipeline


@pytest.mark.asyncio
async def test_provider_job_checkpoint_survives_pipeline_restart(pipeline, monkeypatch):
    p = pipeline
    attempt = 0
    def provider(path, *_args):
        nonlocal attempt
        attempt += 1
        job, save = transcription.assemblyai_job_context.get()
        if attempt == 1:
            assert job is None
            save({'state': 'submitting'})
            save({'state': 'submitted', 'id': 'existing-job'})
            raise TimeoutError('poll failed')
        assert job['id'] == 'existing-job'
        assert job['source_fingerprint']
        Path(path).with_suffix('.transcript_cache.json').write_text(json.dumps(p.words))
        return 'Hello'
    async def generate(path, **kwargs):
        return await asyncio.to_thread(provider, path)
    monkeypatch.setattr(VideoService, 'generate_transcript', generate)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(p.run(), 5)
    assert p.state['checkpoint']['transcription_job']['id'] == 'existing-job'
    p.state['fail_sections'] = False
    await asyncio.wait_for(p.run(), 5)
    assert attempt == 2
    assert p.state['checkpoint']['transcript'] == 'Hello'


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['source', 'settings'])
async def test_pending_job_identity_cannot_be_reused_for_other_input(pipeline, monkeypatch, change):
    p = pipeline
    def provider(*_):
        _, save = transcription.assemblyai_job_context.get()
        save({'state': 'submitted', 'id': 'existing-job'})
        raise TimeoutError('poll failed')
    async def generate(path, **kwargs):
        return await asyncio.to_thread(provider, path)
    monkeypatch.setattr(VideoService, 'generate_transcript', generate)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(p.run(), 5)
    if change == 'source':
        p.source.write_bytes(b'different source')
    else:
        p.config.assembly_ai_speech_model = 'different-model'
    new_transcript = AsyncMock()
    monkeypatch.setattr(VideoService, 'generate_transcript', new_transcript)
    with pytest.raises(ValueError, match='transcription settings changed|source or transcription settings changed'):
        await asyncio.wait_for(p.run(), 5)
    new_transcript.assert_not_awaited()
