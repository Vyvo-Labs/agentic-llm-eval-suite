from pathlib import Path

from llm_eval_suite.config import EvalConfig
from llm_eval_suite.providers import normalize_llm_model_for_provider, resolve_candidate_models


def _config() -> EvalConfig:
    return EvalConfig(
        env_path=Path(".env"),
        env={"OPENAI_API_KEY": "sk-test"},
        provider_presets=["openai"],
        max_models_per_provider=2,
        preset_models={"openai": ["openai/gpt-5-mini", "gpt-5"]},
        datasets=[],
        output_dir=Path("reports"),
        cache_enabled=False,
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


def test_model_normalization_strips_provider_prefix() -> None:
    assert normalize_llm_model_for_provider("openrouter", "openrouter/foo/bar") == "foo/bar"


def test_model_normalization_maps_none_alias_to_base_model() -> None:
    assert normalize_llm_model_for_provider("openrouter", "openai/gpt-5.2-none") == "openai/gpt-5.2"
    assert normalize_llm_model_for_provider("openai", "openai-5.2-none") == "gpt-5.2"


def test_model_normalization_supports_reasoning_suffix_tags() -> None:
    assert normalize_llm_model_for_provider("openrouter", "openai/gpt-5-mini/minimal") == "openai/gpt-5-mini"
    assert normalize_llm_model_for_provider("openrouter", "openrouter/openai/gpt-5.2/none") == "openai/gpt-5.2"


def test_resolve_candidate_models_uses_provider_credentials() -> None:
    resolved = resolve_candidate_models(_config())
    assert not resolved.warnings
    assert len(resolved.candidates) == 2
    assert resolved.candidates[0].request_model == "gpt-5-mini"


def test_openrouter_defaults_include_expected_models() -> None:
    config = EvalConfig(
        env_path=Path(".env"),
        env={"OPENROUTER_API_KEY": "or-test"},
        provider_presets=["openrouter"],
        max_models_per_provider=20,
        preset_models={},
        datasets=[],
        output_dir=Path("reports"),
        cache_enabled=False,
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

    resolved = resolve_candidate_models(config)
    assert not resolved.warnings
    assert [item.configured_model for item in resolved.candidates] == [
        "moonshotai/kimi-k2.5",
        "z-ai/glm-5",
        "z-ai/glm-4.7",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.5",
        "openai/gpt-5-mini/minimal",
        "openai/gpt-5.2",
        "openai/gpt-5.2/none",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "z-ai/glm-4.7-flash",
        "minimax/minimax-m2.5",
        "minimax/minimax-m2-her",
        "stepfun/step-3.5-flash",
        "xiaomi/mimo-v2-flash",
        "nvidia/nemotron-3-nano-30b-a3b",
        "meituan/longcat-flash-chat",
    ]


def test_openai_defaults_include_custom_models() -> None:
    config = EvalConfig(
        env_path=Path(".env"),
        env={"OPENAI_API_KEY": "sk-test"},
        provider_presets=["openai"],
        max_models_per_provider=10,
        preset_models={},
        datasets=[],
        output_dir=Path("reports"),
        cache_enabled=False,
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

    resolved = resolve_candidate_models(config)
    assert not resolved.warnings
    assert [item.configured_model for item in resolved.candidates] == [
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini/minimal",
        "gpt-5.2/none",
    ]
    assert [item.request_model for item in resolved.candidates] == [
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5.2",
    ]
