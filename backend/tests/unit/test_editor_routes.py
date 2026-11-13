from types import SimpleNamespace
from unittest.mock import AsyncMock
import io
import json
import zipfile

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from src.api.routes import editor
from src.database import get_db
from src.editor_document import EditDocument, atomic_json, editor_dir
from src.editor_state import editor_key


@pytest.fixture
async def client(tmp_path, monkeypatch, editor_store):
    task = {"id": "task", "user_id": "owner", "status": "completed"}
    clip = {"id": "clip", "task_id": "task", "clip_order": 1, "filename": "clip.mp4", "file_path": str(tmp_path / "clip.mp4")}
    service = SimpleNamespace(
        task_repo=SimpleNamespace(get_task_by_id=AsyncMock(return_value=task)),
        clip_repo=SimpleNamespace(
            get_clip_by_id=AsyncMock(return_value=clip),
            get_clips_by_task=AsyncMock(return_value=[clip]),
        ),
        config=SimpleNamespace(temp_dir=str(tmp_path)),
    )
    monkeypatch.setattr(editor, "TaskService", lambda db: service)
    monkeypatch.setattr(
        "src.api.routes.tasks._get_user_id_from_headers",
        AsyncMock(return_value="owner"),
    )
    monkeypatch.setattr(editor.JobQueue, "enqueue_job", AsyncMock(return_value="job"))

    db = SimpleNamespace(close=AsyncMock())
    app = FastAPI()
    app.include_router(editor.router)
    app.dependency_overrides[get_db] = lambda: db
    directory = editor_dir(str(tmp_path), "task", "clip", "clip.mp4")
    doc = EditDocument.model_validate(
        {"segments": [{"id": "one", "start": 0, "end": 2}]}
    ).model_dump()
    editor_store.put_ready("task", "clip", "clip.mp4", {"duration": 2}, doc)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http, service, doc, directory


async def test_drafts_require_task_and_clip_ownership(client, monkeypatch):
    http, service, doc, directory = client
    monkeypatch.setattr(
        "src.api.routes.tasks._get_user_id_from_headers",
        AsyncMock(return_value="outsider"),
    )
    assert (await http.get("/tasks/task/clips/clip/editor")).status_code == 403
    assert (
        await http.patch(
            "/tasks/task/clips/clip/editor",
            json={"basis": "clip.mp4", "revision": 0, "document": doc},
        )
    ).status_code == 403
    monkeypatch.setattr(
        "src.api.routes.tasks._get_user_id_from_headers",
        AsyncMock(return_value="owner"),
    )
    service.clip_repo.get_clip_by_id.return_value = {
        "id": "clip",
        "task_id": "other",
        "filename": "clip.mp4",
    }
    assert (await http.get("/tasks/task/clips/clip/editor")).status_code == 404


async def test_revision_and_basis_prevent_stale_writes(client):
    http, service, doc, directory = client
    payload = {"basis": "clip.mp4", "revision": 0, "document": doc}
    assert (await http.patch("/tasks/task/clips/clip/editor", json=payload)).json()[
        "revision"
    ] == 1
    assert (
        await http.patch("/tasks/task/clips/clip/editor", json=payload)
    ).status_code == 409
    payload["basis"] = "previous.mp4"
    assert (
        await http.patch("/tasks/task/clips/clip/editor", json=payload)
    ).status_code == 409
    payload.update(basis="clip.mp4", revision=1, document={**doc, "volume": -1})
    assert (
        await http.patch("/tasks/task/clips/clip/editor", json=payload)
    ).status_code == 400


async def test_export_snapshot_cancel_and_private_download(client, editor_store):
    http, service, doc, directory = client
    response = await http.post(
        "/tasks/task/clips/clip/editor/exports",
        json={"basis": "clip.mp4", "document": doc, "preset": "reels"},
    )
    assert response.status_code == 202
    job = response.json()["id"]
    # The job carries the document it will render; no file on this machine does.
    stored = editor_store.jobs_by_id[job]
    assert stored["request"] == {"document": doc, "preset": "reels", "draft_revision": 0}
    assert not directory.exists() or not list(directory.glob("*.json"))
    queued = editor.JobQueue.enqueue_job.await_args
    assert queued.args == ("export_editor", "task", "clip", "clip.mp4", job)
    assert queued.kwargs == {"_job_id": f"editor-job:{job}"}
    assert (
        await http.get(f"/tasks/task/clips/clip/editor/exports/{job}/file")
    ).status_code == 409
    cancelled = await http.delete(f"/tasks/task/clips/clip/editor/exports/{job}")
    assert cancelled.status_code == 200 and cancelled.json() == {"status": "cancelled"}
    assert stored["status"] == "cancelled" and stored["cancel_requested"]
    # A worker that receives the queue job afterwards finds nothing to claim.
    assert await editor_store.start_job(job) is None
    assert (
        await http.get("/tasks/task/clips/clip/editor/media/draft.json")
    ).status_code == 404
    assert (
        await http.get("/tasks/task/clips/clip/editor/exports/not-a-job")
    ).status_code == 404


async def test_publish_package_pairs_srt_with_completed_job_snapshot(client, editor_store, monkeypatch):
    http, service, doc, directory = client
    document = {
        **doc,
        "words": [
            {"id": "w1", "start": 0.1, "end": 0.4, "text": "Saved"},
            {"id": "w2", "start": 0.5, "end": 0.8, "text": "revision."},
        ],
    }
    saved = await http.patch(
        "/tasks/task/clips/clip/editor",
        json={"basis": "clip.mp4", "revision": 0, "document": document},
    )
    assert saved.status_code == 200
    created = await http.post(
        "/tasks/task/clips/clip/editor/exports",
        json={"basis": "clip.mp4", "document": document, "preset": "reels"},
    )
    job_id = created.json()["id"]
    stored = editor_store.jobs_by_id[job_id]
    assert stored["request"]["draft_revision"] == 1
    claim = (await editor_store.start_job(job_id))["claim"]
    await editor_store.finish_job(job_id, claim, "completed")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"export-{job_id}.mp4").write_bytes(b"exact export")

    # A later saved draft must not change the subtitle paired with this job.
    later = {**document, "words": [{"id": "w3", "start": 0.2, "end": 0.7, "text": "Later."}]}
    assert (await http.patch(
        "/tasks/task/clips/clip/editor",
        json={"basis": "clip.mp4", "revision": 1, "document": later},
    )).status_code == 200

    def cover(source, target):
        target.write_bytes(b"cover from exact job")
        return True

    monkeypatch.setattr("src.services.editor_publish_package._extract_cover", cover)
    response = await http.get(f"/tasks/task/clips/clip/editor/exports/{job_id}/publish.zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "clip_01 - Clip_editor.mp4", "captions.srt", "cover.jpg", "manifest.json",
        }
        assert archive.read("clip_01 - Clip_editor.mp4") == b"exact export"
        assert "Saved revision." in archive.read("captions.srt").decode()
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["draft_revision"] == 1
        assert {item["file"] for item in manifest["missing"]} == {
            "social_caption.txt", "social_hashtags.txt",
        }


async def test_publish_package_rejects_export_without_saved_revision(client, editor_store):
    http, service, doc, directory = client
    job_id = "d" * 32
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.create_job(key, job_id, "export", {"document": doc, "preset": "reels"})
    claim = (await editor_store.start_job(job_id))["claim"]
    await editor_store.finish_job(job_id, claim, "completed")
    response = await http.get(f"/tasks/task/clips/clip/editor/exports/{job_id}/publish.zip")
    assert response.status_code == 409


async def test_publish_package_post_includes_user_supplied_social_text(client, editor_store, monkeypatch):
    http, service, doc, directory = client
    job_id = "e" * 32
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.create_job(key, job_id, "export", {
        "document": doc, "preset": "reels", "draft_revision": 0,
    })
    claim = (await editor_store.start_job(job_id))["claim"]
    await editor_store.finish_job(job_id, claim, "completed")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"export-{job_id}.mp4").write_bytes(b"exact export")
    monkeypatch.setattr("src.services.editor_publish_package._extract_cover", lambda source, target: False)

    response = await http.post(
        f"/tasks/task/clips/clip/editor/exports/{job_id}/publish.zip",
        data={"caption": "Editable caption", "hashtags": "#Keep  #These\nexactly"},
    )
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("social_caption.txt").decode() == "Editable caption"
        assert archive.read("social_hashtags.txt").decode() == "#Keep  #These\nexactly"
        manifest = json.loads(archive.read("manifest.json"))
        assert not {item["file"] for item in manifest["missing"]} & {
            "social_caption.txt", "social_hashtags.txt"
        }


async def test_publish_package_post_rejects_oversized_social_text(client, editor_store):
    http, service, doc, directory = client
    job_id = "f" * 32
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.create_job(key, job_id, "export", {
        "document": doc, "preset": "reels", "draft_revision": 0,
    })
    claim = (await editor_store.start_job(job_id))["claim"]
    await editor_store.finish_job(job_id, claim, "completed")
    response = await http.post(
        f"/tasks/task/clips/clip/editor/exports/{job_id}/publish.zip",
        data={"caption": "x" * 2201},
    )
    assert response.status_code == 422


async def test_publish_package_post_marks_blank_social_text_missing(client, editor_store):
    http, service, doc, directory = client
    job_id = "a" * 32
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.create_job(key, job_id, "export", {
        "document": doc, "preset": "reels", "draft_revision": 0,
    })
    claim = (await editor_store.start_job(job_id))["claim"]
    await editor_store.finish_job(job_id, claim, "completed")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"export-{job_id}.mp4").write_bytes(b"exact export")

    response = await http.post(
        f"/tasks/task/clips/clip/editor/exports/{job_id}/publish.zip",
        data={"caption": "  \n", "hashtags": ""},
    )
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "social_caption.txt" not in archive.namelist()
        assert "social_hashtags.txt" not in archive.namelist()
        missing = {item["file"] for item in json.loads(archive.read("manifest.json"))["missing"]}
        assert {"social_caption.txt", "social_hashtags.txt"} <= missing


async def test_srt_download_uses_only_the_saved_draft_and_phrase_groups(client):
    http, _, doc, _ = client
    translated = {
        **doc,
        "words": [
            {"id": "t1", "start": 0.1, "end": 0.3, "text": "Добрый"},
            {"id": "t2", "start": 0.3, "end": 0.55, "text": "день!"},
            {"id": "t3", "start": 1.8, "end": 2, "text": "Пока."},
        ],
        "captions": {"wordsPerLine": 4},
        "translation": {
            "sourceLanguage": "en",
            "targetLanguage": "ru",
            "phrases": [
                {"id": "p1", "start": 0.1, "end": 0.55, "wordIds": ["t1", "t2"]},
                {"id": "p2", "start": 1.8, "end": 2, "wordIds": ["t3"]},
            ],
        },
    }
    saved = await http.patch(
        "/tasks/task/clips/clip/editor",
        json={"basis": "clip.mp4", "revision": 0, "document": translated},
    )
    assert saved.status_code == 200
    response = await http.get("/tasks/task/clips/clip/editor/captions.srt")
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith(".srt")
    assert response.text == (
        "1\n00:00:00,100 --> 00:00:00,550\nДобрый день!\n\n"
        "2\n00:00:01,800 --> 00:00:02,000\nПока.\n"
    )


async def test_srt_download_rejects_disabled_or_missing_saved_captions(client):
    http, _, doc, _ = client
    no_words = await http.get("/tasks/task/clips/clip/editor/captions.srt")
    assert no_words.status_code == 409
    disabled = {
        **doc,
        "words": [{"id": "w", "start": 0.1, "end": 0.5, "text": "Hello."}],
        "captions": {"enabled": False},
    }
    response = await http.patch(
        "/tasks/task/clips/clip/editor",
        json={"basis": "clip.mp4", "revision": 0, "document": disabled},
    )
    assert response.status_code == 200
    assert (await http.get("/tasks/task/clips/clip/editor/captions.srt")).status_code == 409


async def test_srt_download_requires_clip_ownership(client, monkeypatch):
    http, _, _, _ = client
    monkeypatch.setattr(
        "src.api.routes.tasks._get_user_id_from_headers",
        AsyncMock(return_value="outsider"),
    )
    assert (await http.get("/tasks/task/clips/clip/editor/captions.srt")).status_code == 403


async def test_failed_enqueue_is_retryable(client, monkeypatch):
    http, service, doc, directory = client
    monkeypatch.setattr(
        editor.JobQueue, "enqueue_job", AsyncMock(side_effect=ConnectionError())
    )
    response = await http.post(
        "/tasks/task/clips/clip/editor/exports",
        json={"basis": "clip.mp4", "document": doc},
    )
    assert response.status_code == 503
    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    assert state["jobs"][0]["status"] == "failed"


async def test_autosave_does_not_depend_on_object_storage_or_local_files(client, monkeypatch, editor_store):
    http, service, doc, directory = client

    def unavailable(*args, **kwargs):
        raise AssertionError("Saving a draft must not touch media storage")

    monkeypatch.setattr(editor.MediaStorage, "client", property(unavailable))
    payload = {"basis": "clip.mp4", "revision": 0, "document": doc}
    saved = await http.patch("/tasks/task/clips/clip/editor", json=payload)
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    payload["revision"] = 1
    assert (await http.patch("/tasks/task/clips/clip/editor", json=payload)).json()["revision"] == 2
    assert not directory.exists() or not list(directory.glob("*.json"))


async def test_a_session_written_as_files_is_imported_once_with_its_jobs(client, editor_store):
    """Drafts saved by the file-based editor must open after the upgrade."""
    http, service, doc, directory = client
    editor_store.sessions.clear()
    edited = {**doc, "muted": True}
    atomic_json(directory / "draft.json", {"revision": 7, "document": edited})
    atomic_json(directory / "original.json", doc)
    atomic_json(directory / "asset.json", {"status": "ready", "duration": 2})
    job = "a" * 32
    atomic_json(directory / f"job-{job}.json", {"status": "completed", "progress": 100})
    atomic_json(directory / f"request-{job}.json", {"document": doc, "preset": "tiktok"})

    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    assert state["status"] == "ready"
    assert state["draft"]["revision"] == 7 and state["draft"]["document"]["muted"] is True
    assert state["original"]["muted"] is False
    assert state["jobs"] == [{"id": job, "status": "completed", "progress": 100}]

    # From here the stored session is the truth; the old files are ignored.
    payload = {"basis": "clip.mp4", "revision": 7, "document": doc}
    assert (await http.patch("/tasks/task/clips/clip/editor", json=payload)).json()["revision"] == 8
    atomic_json(directory / "draft.json", {"revision": 99, "document": edited})
    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    assert state["draft"]["revision"] == 8 and state["draft"]["document"]["muted"] is False


async def test_only_one_request_starts_a_preparation(client, editor_store):
    http, service, doc, directory = client
    editor_store.sessions.clear()
    first = await http.post("/tasks/task/clips/clip/editor/prepare")
    second = await http.post("/tasks/task/clips/clip/editor/prepare")
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json() == {"status": "preparing"}
    assert editor.JobQueue.enqueue_job.await_count == 1
    assert (await http.get("/tasks/task/clips/clip/editor")).json()["status"] == "preparing"


async def test_a_preparation_whose_worker_died_can_be_retried(client, editor_store):
    from src.editor_state import STALE_PREPARE_SECONDS

    http, service, doc, directory = client
    editor_store.sessions.clear()
    await http.post("/tasks/task/clips/clip/editor/prepare")
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.begin_prepare(key, "task", "clip", "clip.mp4")
    assert (await http.get("/tasks/task/clips/clip/editor")).json()["status"] == "preparing"
    # The worker stopped heartbeating a while ago.
    editor_store.sessions[key]["updated"] -= STALE_PREPARE_SECONDS + 1
    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    assert state["status"] == "failed" and "interrupted" in state["error"]
    assert (await http.post("/tasks/task/clips/clip/editor/prepare")).json() == {"status": "preparing"}
    assert editor.JobQueue.enqueue_job.await_count == 2


async def test_a_render_whose_worker_died_frees_its_slot(client, editor_store):
    from src.editor_state import STALE_RENDER_SECONDS

    http, service, doc, directory = client
    body = {"basis": "clip.mp4", "document": doc}
    jobs = [(await http.post("/tasks/task/clips/clip/editor/exports", json=body)).json()["id"] for _ in range(3)]
    assert (await http.post("/tasks/task/clips/clip/editor/exports", json=body)).status_code == 429
    assert await editor_store.start_job(jobs[0]) is not None
    editor_store.jobs_by_id[jobs[0]]["updated"] -= STALE_RENDER_SECONDS + 1
    status = (await http.get(f"/tasks/task/clips/clip/editor/exports/{jobs[0]}")).json()
    assert status["status"] == "failed" and "stopped" in status["error"]
    assert (await http.post("/tasks/task/clips/clip/editor/exports", json=body)).status_code == 202


async def test_local_preview_and_export_survive_storage_outage(client, monkeypatch, editor_store):
    from botocore.exceptions import EndpointConnectionError
    http, service, doc, directory = client
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "clean.mp4").write_bytes(b"local preview")
    job = "f" * 32
    service.clip_repo.get_clips_by_task.return_value = [
        {"id": "deleted-before-this-clip"},
        service.clip_repo.get_clip_by_id.return_value,
    ]
    key = editor_key("task", "clip", "clip.mp4")
    await editor_store.create_job(key, job, "export", {"document": doc, "preset": "tiktok"})
    claim = (await editor_store.start_job(job))["claim"]
    await editor_store.finish_job(job, claim, "completed")
    (directory / f"export-{job}.mp4").write_bytes(b"local export")
    monkeypatch.setattr(editor.MediaStorage, "enabled", True, raising=False)
    monkeypatch.setattr(editor.MediaStorage, "editor_reference", lambda self, directory, name="": f"s3://bucket/media/editor/{name}")
    monkeypatch.setattr(editor.MediaStorage, "exists", lambda *a: (_ for _ in ()).throw(EndpointConnectionError(endpoint_url="https://storage.test")))
    preview = await http.get("/tasks/task/clips/clip/editor/media/clean.mp4")
    export = await http.get(f"/tasks/task/clips/clip/editor/exports/{job}/file")
    assert preview.status_code == export.status_code == 200
    assert preview.content == b"local preview"
    assert export.content == b"local export"
    assert export.headers["content-disposition"] == "attachment; filename*=utf-8''clip_02%20-%20Clip_editor.mp4"


async def test_old_drafts_regain_template_animation_and_hook_title(client, editor_store):
    http, service, doc, directory = client
    service.task_repo.get_task_by_id.return_value = {
        "id": "task",
        "user_id": "owner",
        "status": "completed",
        "caption_template": "hormozi",
    }
    service.clip_repo.get_clip_by_id.return_value = {
        **service.clip_repo.get_clip_by_id.return_value,
        "hook_title": "Nobody tells you this",
    }
    # A draft saved before the editor kept the template and hook title.
    old = {**doc, "captions": {k: v for k, v in doc["captions"].items() if k not in {"template", "karaoke", "uppercase"}}}
    old.pop("hookTitle")
    editor_store.put_ready("task", "clip", "clip.mp4", {"duration": 2}, old, revision=3)

    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    captions = state["draft"]["document"]["captions"]
    assert captions["template"] == "hormozi"
    assert captions["karaoke"] is True and captions["uppercase"] is True
    assert state["draft"]["document"]["hookTitle"] == "Nobody tells you this"
    assert state["draft"]["revision"] == 3
    assert state["captionLooks"]["hormozi"]["wordBox"] is True

    # Drafts that already chose these values keep them.
    current = {**doc, "hookTitle": "", "captions": {**doc["captions"], "karaoke": False}}
    editor_store.put_ready("task", "clip", "clip.mp4", {"duration": 2}, current, revision=4)
    state = (await http.get("/tasks/task/clips/clip/editor")).json()
    assert state["draft"]["document"]["hookTitle"] == ""
    assert state["draft"]["document"]["captions"]["karaoke"] is False
