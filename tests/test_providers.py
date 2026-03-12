from pathlib import Path

from llm_eval_suite.config import EvalConfig
from llm_eval_suite.providers import (
    extra_body_for_model,
    normalize_llm_model_for_provider,
    resolve_candidate_models,
    resolve_judge_endpoint,
)


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


def test_qwen_27b_extra_body_disables_thinking() -> None:
    assert extra_body_for_model("Qwen/Qwen3.5-27B") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert extra_body_for_model("qwen/qwen3.5-27b") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert extra_body_for_model("openrouter/Qwen/Qwen3.5-27B") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert extra_body_for_model("gpt-5-mini") is None


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
        max_models_per_provider=30,
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
        "z-ai/glm-5",
        "google/gemini-3.1-flash-lite-preview",
        "qwen/qwen3.5-35b-a3b",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "Qwen/Qwen3.5-27B",
        "deepseek/deepseek-v3.2",
        "anthropic/claude-haiku-4.5",
        "openai/gpt-5-mini/minimal",
        "openai/gpt-5.3-chat",
        "openai/gpt-5.2",
        "openai/gpt-5.2/none",
        "openai/gpt-4.1-mini",
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
        "Qwen/Qwen3.5-27B",
    ]
    assert [item.request_model for item in resolved.candidates] == [
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5.2",
        "Qwen/Qwen3.5-27B",
    ]


def test_openai_custom_chat_completions_base_url_is_normalized_for_candidates() -> None:
    config = EvalConfig(
        env_path=Path(".env"),
        env={
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "http://localhost:8000/v1/chat/completions",
        },
        provider_presets=["openai"],
        max_models_per_provider=1,
        preset_models={"openai": ["gpt-5-mini"]},
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
    assert len(resolved.candidates) == 1
    assert resolved.candidates[0].base_url == "http://localhost:8000/v1"
    assert resolved.candidates[0].request_model == "gpt-5-mini"


def test_openai_custom_chat_completions_base_url_is_normalized_for_judge() -> None:
    config = EvalConfig(
        env_path=Path(".env"),
        env={"OPENAI_API_KEY": "sk-test"},
        provider_presets=["openai"],
        max_models_per_provider=1,
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
        judge_base_url="http://localhost:8000/v1/chat/completions",
        judge_api_key=None,
        judge_timeout_s=30,
        judge_max_completion_tokens=256,
        judge_reasoning_effort=None,
    )

    endpoint, error = resolve_judge_endpoint(config)
    assert error is None
    assert endpoint is not None
    assert endpoint.base_url == "http://localhost:8000/v1"
    assert endpoint.request_model == "gpt-5-mini"


def test_openrouter_defaults_do_not_include_nvidia_nemotron_30b_a3b() -> None:
    config = EvalConfig(
        env_path=Path(".env"),
        env={"OPENROUTER_API_KEY": "or-test"},
        provider_presets=["openrouter"],
        max_models_per_provider=50,
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

    nemotron_candidates = [
        item for item in resolved.candidates if item.configured_model == "nvidia/nemotron-3-nano-30b-a3b"
    ]
    assert len(nemotron_candidates) == 0
