"""A free model whose context window is 8192 rejects a request that asks for
8192 output tokens before it generates anything, so the output cap has to be
tunable instead of hard-coded."""
from src.config import Config


def test_max_tokens_defaults_to_the_paid_model_value(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    assert Config().llm_max_tokens == 8192


def test_max_tokens_can_be_lowered_for_small_free_models(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    assert Config().llm_max_tokens == 2048


def test_max_tokens_ignores_a_value_that_is_not_a_number(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "lots")
    assert Config().llm_max_tokens == 8192
