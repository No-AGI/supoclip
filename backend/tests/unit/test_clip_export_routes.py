from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from src.api.routes import clip_exports
from src.database import get_db
from src.repositories.clip_export_repository import ClipExportRepository


@pytest.mark.asyncio
async def test_s3_export_status_and_download_use_persisted_filename(monkeypatch):
    task_id, clip_id, export_id = "task", "clip", "export"
    persisted_name = "clip_02 - Original Hook - Source_tiktok.mp4"
    row = {
        "id": export_id,
        "task_id": task_id,
        "clip_id": clip_id,
        "status": "succeeded",
        "file_path": "s3://bucket/export.mp4",
        "download_filename": persisted_name,
        "preset": "tiktok",
    }
    task = {"id": task_id, "user_id": "owner", "source_title": "Renamed Source"}
    clip = {"id": clip_id, "task_id": task_id, "file_path": "s3://bucket/clip.mp4"}
    service = SimpleNamespace(clip_repo=SimpleNamespace(
        get_clip_by_id=AsyncMock(return_value=clip),
    ))
    db = SimpleNamespace(close=AsyncMock())
    repository_rows = [row, row, row]

    async def get_row(*_args, **_kwargs):
        return repository_rows.pop(0)

    monkeypatch.setattr(clip_exports, "TaskService", lambda _db: service)
    monkeypatch.setattr(clip_exports, "_require_task_reader", AsyncMock(return_value=task))
    monkeypatch.setattr(
        ClipExportRepository, "get_for_task_clip", staticmethod(get_row)
    )
    monkeypatch.setattr(
        clip_exports,
        "present_export",
        AsyncMock(return_value={
            "id": export_id,
            "status": "succeeded",
            "download_filename": persisted_name,
            "download_url": None,
        }),
    )
    monkeypatch.setattr(
        clip_exports.MediaStorage,
        "signed_url",
        lambda _self, _reference, filename, attachment=False: f"https://storage.test/{filename}",
    )
    monkeypatch.setattr(clip_exports, "capture_clip_export", lambda *_args, **_kwargs: None)

    app = FastAPI()
    app.include_router(clip_exports.router)
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        status = await client.get(f"/tasks/{task_id}/clips/{clip_id}/exports/{export_id}")
        download = await client.get(
            f"/tasks/{task_id}/clips/{clip_id}/exports/{export_id}/file",
            follow_redirects=False,
        )

    assert status.status_code == 200
    assert status.json()["download_filename"] == persisted_name
    assert download.status_code == 307
    from urllib.parse import quote
    assert download.headers["location"] == f"https://storage.test/{quote(persisted_name)}"
