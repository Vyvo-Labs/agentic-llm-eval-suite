from __future__ import annotations

from pathlib import Path

from llm_eval_suite.config import EvalConfig
from llm_eval_suite.providers import LLMEndpoint
from llm_eval_suite.runner import _cache_request_fingerprint, _resolve_reasoning_effort_for_model


def _config() -> EvalConfig:
    return EvalConfig(
        env_path=Path(".env"),
        env={},
        provider_presets=["openrouter"],
        max_models_per_provider=8,
        preset_models={},
        datasets=[],
        output_dir=Path("reports"),
        cache_enabled=True,
        cache_dir=Path(".cache"),
        timeout_s=30,
        concurrency=1,
        max_completion_tokens=256,
        reasoning_effort=None,
        temperature=None,
        judge_provider="openai",
        judge_model="gpt-5-mini",
        judge_base_url=None,
        judge_api_key=None,
        judge_timeout_s=30,
        judge_max_completion_tokens=256,
        judge_reasoning_effort=None,
    )


def test_cache_fingerprint_changes_with_api_key() -> None:
    config = _config()
    messages = [{"role": "user", "content": "ping"}]
    endpoint_a = LLMEndpoint(
        provider="openrouter",
        configured_model="moonshotai/kimi-k2.5",
        request_model="moonshotai/kimi-k2.5",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key="sk-or-v1-a",
    )
    endpoint_b = LLMEndpoint(
        provider="openrouter",
        configured_model="moonshotai/kimi-k2.5",
        request_model="moonshotai/kimi-k2.5",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key="sk-or-v1-b",
    )

    fingerprint_a = _cache_request_fingerprint(
        endpoint=endpoint_a,
        messages=messages,
        config=config,
        reasoning_effort=config.reasoning_effort,
    )
    fingerprint_b = _cache_request_fingerprint(
        endpoint=endpoint_b,
        messages=messages,
        config=config,
        reasoning_effort=config.reasoning_effort,
    )

    assert fingerprint_a["credentials_sha256"] != fingerprint_b["credentials_sha256"]
    assert fingerprint_a != fingerprint_b


def test_cache_fingerprint_changes_with_base_url() -> None:
    config = _config()
    messages = [{"role": "user", "content": "ping"}]
    endpoint_a = LLMEndpoint(
        provider="openai",
        configured_model="gpt-5-mini",
        request_model="gpt-5-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        api_key="sk-test",
    )
    endpoint_b = LLMEndpoint(
        provider="openai",
        configured_model="gpt-5-mini",
        request_model="gpt-5-mini",
        base_url="https://custom.example/v1",
        api_key_env="OPENAI_API_KEY",
        api_key="sk-test",
    )

    fingerprint_a = _cache_request_fingerprint(
        endpoint=endpoint_a,
        messages=messages,
        config=config,
        reasoning_effort=config.reasoning_effort,
    )
    fingerprint_b = _cache_request_fingerprint(
        endpoint=endpoint_b,
        messages=messages,
        config=config,
        reasoning_effort=config.reasoning_effort,
    )

    assert fingerprint_a["base_url"] is None
    assert fingerprint_b["base_url"] == "https://custom.example/v1"
    assert fingerprint_a != fingerprint_b


def test_reasoning_effort_override_for_none_alias_models() -> None:
    endpoint = LLMEndpoint(
        provider="openrouter",
        configured_model="openai/gpt-5.2-none",
        request_model="openai/gpt-5.2",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key="sk-or-v1-a",
    )
    resolved = _resolve_reasoning_effort_for_model(
        endpoint=endpoint,
        configured_reasoning_effort="medium",
    )
    assert resolved == "none"


def test_reasoning_effort_override_keeps_config_for_normal_models() -> None:
    endpoint = LLMEndpoint(
        provider="openrouter",
        configured_model="anthropic/claude-sonnet-4.5",
        request_model="anthropic/claude-sonnet-4.5",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key="sk-or-v1-a",
    )
    resolved = _resolve_reasoning_effort_for_model(
        endpoint=endpoint,
        configured_reasoning_effort="medium",
    )
    assert resolved == "medium"
