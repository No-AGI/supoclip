from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes import social


@pytest.mark.parametrize("target_language", [None, "Russian"])
async def test_caption_route_passes_target_language(monkeypatch, target_language):
    payload = {"task_id": "task-1", "clip_id": "clip-1", "provider": "youtube"}
    if target_language is not None:
        payload["target_language"] = target_language
    monkeypatch.setattr(social, "_json_body", AsyncMock(return_value=payload))
    monkeypatch.setattr(social, "_allowed_user_id", AsyncMock(return_value="user-1"))
    service = type("Service", (), {"generate_caption": AsyncMock(return_value={"caption": "Draft"})})()
    monkeypatch.setattr(social, "_service", lambda request, db: service)

    result = await social.generate_caption(request=object(), db=object())

    assert result == {"caption": "Draft"}
    service.generate_caption.assert_awaited_once_with(
        "user-1", task_id="task-1", clip_id="clip-1", provider="youtube",
        target_language=target_language,
    )


async def test_caption_route_rejects_unsupported_target_language(monkeypatch):
    monkeypatch.setattr(social, "_json_body", AsyncMock(return_value={
        "task_id": "task-1", "clip_id": "clip-1", "provider": "youtube",
        "target_language": "French",
    }))
    monkeypatch.setattr(social, "_allowed_user_id", AsyncMock(return_value="user-1"))

    with pytest.raises(HTTPException) as error:
        await social.generate_caption(request=object(), db=object())

    assert error.value.status_code == 422
