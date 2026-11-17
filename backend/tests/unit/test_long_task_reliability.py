"""Failure boundaries for long jobs; no media, network or external stores."""
import copy
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.media import transcription
from src.services.video_service import VideoService
from tests.unit.test_task_service import build_task_service
from tests.unit.test_task_service_workflows import _workflow_service, _run, SERIES_SEGMENTS, SERIES_PLAN


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_interrupted_audio_is_not_reused(monkeypatch, tmp_path, failure):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    calls = []
    monkeypatch.setattr(transcription, "select_audio_stream_index", lambda _: 0)

    def ffmpeg(command, **kwargs):
        output = Path(command[-1])
        calls.append(output)
        output.write_bytes(b"partial" if len(calls) == 1 else b"complete")
        if len(calls) == 1:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 900)
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(transcription, "run_ffmpeg_command", ffmpeg)
    if failure == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            transcription._prepare_audio_for_transcription(video)
    else:
        assert transcription._prepare_audio_for_transcription(video) == video
    audio = transcription._prepare_audio_for_transcription(video)
    assert len(calls) == 2
    assert audio.read_bytes() == b"complete"
    assert transcription._prepare_audio_for_transcription(video) == audio
    assert len(calls) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.mp4", "source.transcription.mp3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["progress", "metadata", "email"])
async def test_post_completion_failure_never_discards_outputs(stage):
    service, events = _workflow_service("series", SERIES_SEGMENTS, copy.deepcopy(SERIES_PLAN))
    service.checkpoint_repo.forget_completed_positions = AsyncMock()

    async def progress(percent, *_args):
        if percent == 100 and stage == "progress":
            raise RuntimeError("notification unavailable")

    async def metadata(*_args, **kwargs):
        if kwargs.get("completed_at") and stage == "metadata":
            raise RuntimeError("metadata unavailable")

    service.task_repo.update_task_runtime_metadata.side_effect = metadata
    service._send_completion_notification_if_needed = AsyncMock(
        side_effect=RuntimeError("email unavailable") if stage == "email" else None
    )
    result = await _run(service, progress_callback=progress)
    assert result["clips_count"] == 2
    assert ("delete_clips",) not in events
    assert [e[1] for e in events if e[0] == "status"][-1] == "completed"


@pytest.mark.asyncio
async def test_changed_source_with_outputs_stops_before_new_paid_work():
    service = build_task_service()
    service.clip_repo.get_clips_by_task.return_value = [{"id": "old", "clip_order": 1}]
    service.checkpoint_repo.get.return_value = {"completed_positions": [1]}

    async def pipeline(**kwargs):
        await kwargs["source_identity_changed"]()
        pytest.fail("Must stop before mixing new source and saved clips")

    service.video_service.process_video_complete.side_effect = pipeline
    with pytest.raises(ValueError, match="source.*changed"):
        await service.process_task("task-1", "https://youtu.be/abcdefghijk", "youtube")
    service.clip_repo.delete_clips_by_task.assert_not_awaited()
    service.video_service.create_single_clip.assert_not_awaited()
    service.checkpoint_repo.save.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred_at", ["admission", "upload"])
async def test_worker_deferral_preserves_profile(monkeypatch, deferred_at):
    from src.workers import tasks, video_admission
    from src import database, runtime_settings
    db = SimpleNamespace(commit=AsyncMock())

    @asynccontextmanager
    async def session():
        yield db

    monkeypatch.setattr(database, "AsyncSessionLocal", session)
    monkeypatch.setattr(runtime_settings, "load_runtime_settings_cache", AsyncMock())
    admission = AsyncMock(return_value="deferred" if deferred_at == "admission" else "run")
    upload = AsyncMock(return_value="deferred")
    monkeypatch.setattr(video_admission, "admit_video_job", admission)
    monkeypatch.setattr(video_admission, "hold_for_upload", upload)
    redis = SimpleNamespace(publish=AsyncMock(), setex=AsyncMock())
    await tasks.process_video_task({"redis": redis}, "task", "upload://x", "upload", "user",
                                   clip_profile="gaming_stream")
    arguments = (admission if deferred_at == "admission" else upload).await_args.kwargs["job_args"]
    assert arguments[-1] == "gaming_stream"
    assert len(arguments) == 14
