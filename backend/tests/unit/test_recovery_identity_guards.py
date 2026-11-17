"""Recovery must not lose paid job identity when result inputs change."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.task_checkpoint_repository import TaskCheckpointRepository
from src.services.video_service import VideoService
from tests.unit.test_processing_checkpoints import pipeline
from tests.unit.test_task_service_workflows import _workflow_service, SERIES_SEGMENTS, SERIES_PLAN


@pytest.mark.asyncio
async def test_changed_checkpoint_key_keeps_paid_submission_visible():
    db = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.first.return_value = {
        "input_key": "original", "payload": {"transcription_job": {"state": "submitting"}},
    }
    db.execute.return_value = result
    with pytest.raises(ValueError, match="paid transcription"):
        await TaskCheckpointRepository.get(db, "task", "changed")


@pytest.mark.asyncio
async def test_completed_checkpoint_rejects_changed_model(pipeline):
    p = pipeline
    p.state['fail_sections'] = False
    await p.run()
    p.config.assembly_ai_speech_model = 'different-model'
    with pytest.raises(ValueError, match='transcription settings changed'):
        await p.run()
    p.transcription.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['upload', 'twitch', 'kick'])
async def test_completed_checkpoint_rejects_changed_non_youtube_bytes(pipeline, monkeypatch, kind):
    from src.services import video_service as module
    p = pipeline
    p.state['fail_sections'] = False
    monkeypatch.setattr(module, 'materialize_stored_upload', AsyncMock(return_value=p.source))
    monkeypatch.setattr(module, 'upload_needs_remote_copy', AsyncMock(return_value=False))
    monkeypatch.setattr(module, 'download_stream_audio', lambda *a, **kw: p.source)
    monkeypatch.setattr(VideoService, '_select_highlight_segments', AsyncMock(return_value=(None, [], 1, True)))
    async def run():
        return await VideoService.process_video_complete(
            'upload://source.mp4' if kind == 'upload' else 'https://example.test/vod', kind,
            checkpoint=p.state['checkpoint'], checkpoint_callback=lambda patch: save(patch),
        )
    async def save(patch):
        p.state['checkpoint'].update(patch)
    # Use a synchronous failure for the synchronous sidecar hook.
    def stopped(*a):
        raise RuntimeError('after transcript')
    monkeypatch.setattr(module, 'persist_cache_sidecar', stopped)
    with pytest.raises(RuntimeError, match='after transcript'):
        await run()
    p.source.write_bytes(b'changed bytes')
    p.transcription.reset_mock()
    async def changed():
        raise ValueError('source changed')
    with pytest.raises(ValueError, match='source.*changed'):
        await VideoService.process_video_complete(
            'upload://source.mp4' if kind == 'upload' else 'https://example.test/vod', kind,
            checkpoint=p.state['checkpoint'], checkpoint_callback=save, source_identity_changed=changed,
        )
    p.transcription.assert_not_awaited()


@pytest.mark.asyncio
async def test_youtube_workflow_has_safe_planning_cache():
    service, _ = _workflow_service('series', SERIES_SEGMENTS, SERIES_PLAN)
    await service.process_task('task-1', 'https://youtu.be/abcdefghijk', 'youtube')
    cache = service.video_service.process_video_complete.await_args.kwargs['plan_cache']
    service.cache_repo.upsert_cache.reset_mock()
    assert cache is not None
    assert await cache.load('unproven-url-key') is None
    await cache.store('unproven-url-key', 'outline')
    service.cache_repo.upsert_cache.assert_not_awaited()
