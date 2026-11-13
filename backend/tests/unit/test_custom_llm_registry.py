import json
import pytest
import asyncio
from types import SimpleNamespace

from src.custom_llm_connections import parse_connections, connection_provider, model_target
from src.llm_fallback import ModelUnavailableError, run_with_fallback
from src import ai
from src.config import Config


def test_registry_parsing_ignores_malformed_and_keeps_distinct_connections():
    entries = [
        {"id": "custom-11111111111111111111111111111111", "name": "one", "base_url": "https://one.example/v1", "model": "m", "api_key": "secret1"},
        {"id": "custom-22222222222222222222222222222222", "name": "two", "base_url": "https://two.example/v1", "model": "m", "api_key": "secret2"},
        {"id": "bad", "api_key": "secret3"},
    ]
    result = parse_connections(json.dumps(entries))
    assert len(result) == 2
    assert result[0]["api_key"] != result[1]["api_key"]
    assert connection_provider(result[0]["id"])
    assert model_target(result[0]) == f'{result[0]["id"]}:m'


def test_invalid_registry_is_empty():
    assert parse_connections("not json") == []
    assert parse_connections('{}') == []
    assert not connection_provider("openai")


def test_config_only_returns_key_for_its_connection():
    one = "custom-" + "a" * 32
    two = "custom-" + "b" * 32
    cfg = Config()
    cfg.llm_custom_connections = json.dumps([
        {"id": one, "name": "one", "base_url": "https://one.example/v1", "model": "m", "api_key": "first-key"},
        {"id": two, "name": "two", "base_url": "https://two.example/v1", "model": "m", "api_key": "second-key"},
    ])
    assert cfg.provider_keys(one) == ["first-key"]
    assert cfg.provider_keys(two) == ["second-key"]
    assert cfg.provider_keys("custom-" + "c" * 32) == []


@pytest.mark.asyncio
async def test_custom_provider_never_exposes_provider_response_in_error_or_logs(caplog):
    secret = "secret-which-must-not-appear"
    async def run(_agent):
        raise RuntimeError("provider echoed " + secret)
    with pytest.raises(ModelUnavailableError) as error:
        await run_with_fallback(lambda *_: object(), run, chain=["custom-11111111111111111111111111111111:model"], keys_for=lambda _: [secret])
    assert secret not in str(error.value)
    assert secret not in caplog.text
    assert all(secret not in attempt.error for attempt in error.value.attempts)


def test_custom_builder_binds_key_and_endpoint_to_one_connection(monkeypatch):
    provider = "custom-" + "a" * 32
    connection = {"id": provider, "name": "test", "model": "m", "base_url": "https://example.com/v1", "api_key": "key-A"}
    config = SimpleNamespace(llm="openai:primary", custom_connection=lambda name: connection if name == provider else None)
    captured = {}
    monkeypatch.setattr(ai, "OpenAIProvider", lambda **kwargs: captured.update(kwargs) or object())
    monkeypatch.setattr(ai, "OpenAIModel", lambda model, provider: SimpleNamespace(name=model, provider=provider))
    result = ai._build_transcript_model(config, provider + ":m", "key-A")
    assert result.name == "m"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["api_key"] == "key-A"
    with pytest.raises(RuntimeError):
        ai._build_transcript_model(config, provider + ":m", "another-key")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "timeout", "cancel"])
async def test_custom_client_closed_after_each_attempt(outcome):
    import asyncio
    from unittest.mock import AsyncMock
    client = SimpleNamespace(aclose=AsyncMock())
    agent = SimpleNamespace(model=SimpleNamespace(_supoclip_http_client=client))
    async def run(_agent):
        if outcome == "error":
            raise RuntimeError("error")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "timeout":
            await asyncio.sleep(1)
        return "ok"
    if outcome == "success":
        await run_with_fallback(lambda *_: agent, run, chain=["custom-" + "a" * 32 + ":m"], keys_for=lambda _: ["key"], attempt_timeout=.01, total_timeout=.02)
    else:
        with pytest.raises(BaseException):
            await run_with_fallback(lambda *_: agent, run, chain=["custom-" + "a" * 32 + ":m"], keys_for=lambda _: ["key"], attempt_timeout=.01, total_timeout=.02)
    client.aclose.assert_awaited_once()


def test_workflow_custom_agent_is_not_cached(monkeypatch):
    from src.workflows import llm
    custom = "custom-" + "a" * 32 + ":m"
    cfg = SimpleNamespace(openai_api_key=None, openrouter_api_key=None, openrouter_fallback_model=None, google_api_key=None, anthropic_api_key=None, ollama_base_url=None, ollama_api_key=None, llm_custom_connections="", as_runtime_settings=lambda: {})
    monkeypatch.setattr(llm, "get_config", lambda: cfg)
    monkeypatch.setattr(llm, "apply_settings_to_process_env", lambda *_: None)
    monkeypatch.setattr(llm.ai, "_get_missing_llm_key_error", lambda *_: None)
    monkeypatch.setattr(llm.ai, "_model_for", lambda *_: object())
    monkeypatch.setattr(llm, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    first = llm._build_agent_for(str, "prompt", custom, "key")
    second = llm._build_agent_for(str, "prompt", custom, "key")
    assert first is not second
