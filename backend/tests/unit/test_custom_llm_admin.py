import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes import admin


@pytest.mark.asyncio
async def test_add_connection_stores_secret_only_in_encrypted_setting(monkeypatch):
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    monkeypatch.setattr(admin, "get_config", lambda: SimpleNamespace())
    monkeypatch.setattr(admin, "_lock_custom_registry", AsyncMock(return_value=[]))
    save = AsyncMock()
    monkeypatch.setattr(admin, "_save_setting", save)
    payload = admin.CustomConnectionInput(name="Example", base_url="https://example.com/v1", model="model", api_key="private-token")
    result = await admin.create_custom_connection(None, payload, None)
    assert "private-token" not in str(result)
    assert result["id"].startswith("custom-")
    assert result["base_url"] == payload.base_url
    assert save.await_args.args[1] == "LLM_CUSTOM_CONNECTIONS"
    assert json.loads(save.await_args.args[2])[0]["api_key"] == "private-token"


@pytest.mark.asyncio
async def test_add_connection_rejects_unsafe_before_save(monkeypatch):
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    monkeypatch.setattr(admin, "get_config", lambda: SimpleNamespace())
    lock = AsyncMock()
    monkeypatch.setattr(admin, "_lock_custom_registry", lock)
    payload = admin.CustomConnectionInput(name="Local", base_url="http://169.254.169.254/", model="m", api_key="private-token")
    with pytest.raises(HTTPException) as exc:
        await admin.create_custom_connection(None, payload, None)
    assert exc.value.status_code == 400
    lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_refuses_connection_in_chain(monkeypatch):
    item = {"id": "custom-" + "a" * 32, "name": "one", "base_url": "https://example.com/v1", "model": "m", "api_key": "secret"}
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    monkeypatch.setattr(admin, "get_config", lambda: SimpleNamespace(llm_chain=lambda: (item["id"] + ":m",)))
    monkeypatch.setattr(admin, "_lock_custom_registry", AsyncMock(return_value=[item]))
    monkeypatch.setattr(admin, "_current_chain", lambda: (item["id"] + ":m",))
    with pytest.raises(HTTPException) as exc:
        await admin.delete_custom_connection(item["id"], None, None)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_custom_model_probe_never_returns_provider_echo(monkeypatch):
    provider = "custom-" + "a" * 32
    secret = "a-private-key"
    config = SimpleNamespace(provider_keys=lambda _: [secret], custom_connection=lambda _: {"api_key": secret})
    monkeypatch.setattr(admin, "get_config", lambda: config)
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    monkeypatch.setattr(admin, "run_with_fallback", AsyncMock(return_value=SimpleNamespace(value=SimpleNamespace(output=secret), attempts=())))
    response = await admin.test_llm_model(None, admin.LlmModelTest(model=provider + ":m"), None)
    assert response["answer"] == "ok"
    assert secret not in str(response)


@pytest.mark.asyncio
async def test_chain_validation_happens_under_same_registry_lock(monkeypatch):
    order = []
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    monkeypatch.setattr(admin, "get_config", lambda: SimpleNamespace(llm="openai:primary"))
    async def lock(_db):
        order.append("lock")
    def validate(value):
        assert order == ["lock"]
        order.append("validate")
        return value
    async def save(*_args):
        order.append("save")
    monkeypatch.setattr(admin, "_lock_custom_registry", lock)
    monkeypatch.setattr(admin, "_validate_model_name", validate)
    monkeypatch.setattr(admin, "_save_setting", save)
    monkeypatch.setattr(admin, "_current_chain", lambda: ("openai:primary", "openai:backup"))
    await admin.update_llm_chain(None, admin.LlmChainUpdate(fallbacks=["openai:backup"]), None)
    assert order == ["lock", "validate", "save"]


@pytest.mark.asyncio
async def test_delete_and_chain_update_are_serialized(monkeypatch):
    item = {"id": "custom-" + "a" * 32, "name": "one", "base_url": "https://example.com/v1", "model": "m", "api_key": "secret"}
    state = {"connections": [item], "chain": ("openai:primary",)}
    lock = asyncio.Lock()
    deleted_has_lock = asyncio.Event()
    release_delete = asyncio.Event()
    monkeypatch.setattr(admin, "require_admin_user", AsyncMock(return_value="admin"))
    cfg = SimpleNamespace(llm="openai:primary", llm_chain=lambda: state["chain"], custom_connection=lambda value: next((x for x in state["connections"] if x["id"] == value), None))
    monkeypatch.setattr(admin, "get_config", lambda: cfg)
    monkeypatch.setattr(admin, "_current_chain", lambda: state["chain"])
    async def locked(_db):
        await lock.acquire()
        if asyncio.current_task().get_name() == "delete":
            deleted_has_lock.set()
            await release_delete.wait()
        return list(state["connections"])
    async def save(_db, setting, value, _user):
        if setting == "LLM_CUSTOM_CONNECTIONS":
            state["connections"] = json.loads(value)
        else:
            state["chain"] = ("openai:primary", *value.split(","))
        lock.release()
    monkeypatch.setattr(admin, "_lock_custom_registry", locked)
    monkeypatch.setattr(admin, "_save_setting", save)
    deleting = asyncio.create_task(admin.delete_custom_connection(item["id"], None, None), name="delete")
    await deleted_has_lock.wait()
    updating = asyncio.create_task(admin.update_llm_chain(None, admin.LlmChainUpdate(fallbacks=[item["id"] + ":m"]), None))
    await asyncio.sleep(0)
    assert not updating.done()
    release_delete.set()
    await deleting
    with pytest.raises(HTTPException) as exc:
        await updating
    assert exc.value.status_code == 400
    assert state["chain"] == ("openai:primary",)
