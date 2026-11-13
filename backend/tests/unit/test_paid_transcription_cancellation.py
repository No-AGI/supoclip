"""Cancelling the async caller must not abandon a thread's saved provider ID."""
import asyncio
import threading

import pytest

from src.utils.async_helpers import run_in_thread


@pytest.mark.asyncio
async def test_cancellation_waits_for_provider_state_to_be_saved():
    started = threading.Event()
    release = threading.Event()
    saved = {}

    def provider():
        started.set()
        assert release.wait(5), 'test did not release provider thread'
        saved.update(state='submitted', id='existing-job')

    # Worker jobs own a media workspace; thread cancellation is shielded there.
    from src.media_storage import _workspace
    token = _workspace.set(object())
    try:
        task = asyncio.create_task(run_in_thread(provider))
    finally:
        _workspace.reset(token)
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert saved == {'state': 'submitted', 'id': 'existing-job'}
