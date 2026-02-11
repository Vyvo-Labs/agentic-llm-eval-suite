from pathlib import Path

from llm_eval_suite.config import load_config

_ENV_KEYS = [
    "EVAL_PROVIDER_PRESET",
    "EVAL_PRESET_MODELS_OPENAI",
    "EVAL_PRESET_MODELS_GROQ",
    "EVAL_PRESET_MODELS_OPENROUTER",
    "EVAL_PRESET_MODELS_FIREWORKS",
    "EVAL_PRESET_MODELS_TOGETHER",
    "EVAL_PRESET_MODELS_CEREBRAS",
    "EVAL_REASONING_EFFORT",
    "EVAL_JUDGE_REASONING_EFFORT",
    "EVAL_LEGACY_MODEL_PIN",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "TOOL_MESSAGE_MODEL",
    "LLM_REASONING_EFFORT",
]


def _clear_relevant_env(monkeypatch: object) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_config_supports_legacy_anthropic_provider_alias_without_model_pin(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=anthropic",
                "LLM_MODEL=claude-4-5-haiku",
                "TOOL_MESSAGE_MODEL=claude-4-5-haiku",
                "LLM_REASONING_EFFORT=none",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(dotenv_path)

    assert config.provider_presets == ["openrouter"]
    assert "openrouter" not in config.preset_models
    assert config.reasoning_effort == "none"
    assert config.judge_reasoning_effort == "none"


def test_load_config_legacy_model_pin_can_restrict_candidates(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=anthropic",
                "LLM_MODEL=claude-4-5-haiku",
                "TOOL_MESSAGE_MODEL=claude-4-5-haiku",
                "EVAL_LEGACY_MODEL_PIN=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(dotenv_path)

    assert config.provider_presets == ["openrouter"]
    assert config.preset_models["openrouter"] == ["anthropic/claude-haiku-4.5"]


def test_load_config_legacy_includes_distinct_tool_message_model_when_pinned(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=anthropic",
                "LLM_MODEL=claude-4-5-haiku",
                "TOOL_MESSAGE_MODEL=claude-4-5-sonnet",
                "EVAL_LEGACY_MODEL_PIN=true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(dotenv_path)

    assert config.provider_presets == ["openrouter"]
    assert config.preset_models["openrouter"] == [
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
    ]


def test_explicit_eval_preset_models_override_legacy_alias(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=anthropic",
                "LLM_MODEL=claude-4-5-haiku",
                "TOOL_MESSAGE_MODEL=claude-4-5-sonnet",
                "EVAL_LEGACY_MODEL_PIN=true",
                "EVAL_PRESET_MODELS_OPENROUTER=anthropic/claude-opus-4.5",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(dotenv_path)

    assert config.provider_presets == ["openrouter"]
    assert config.preset_models["openrouter"] == ["anthropic/claude-opus-4.5"]


def test_load_config_defaults_to_all_models_per_provider_limit_of_20(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    config = load_config(dotenv_path)
    assert config.max_models_per_provider == 20


def test_load_config_defaults_judge_to_gpt_5_2_xhigh(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    _clear_relevant_env(monkeypatch)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("", encoding="utf-8")

    config = load_config(dotenv_path)
    assert config.judge_provider == "openai"
    assert config.judge_model == "gpt-5.2"
    assert config.judge_reasoning_effort == "xhigh"
