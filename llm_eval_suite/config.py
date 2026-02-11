from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDERS: tuple[str, ...] = (
    "openai",
    "groq",
    "openrouter",
    "fireworks",
    "together",
    "cerebras",
)


@dataclass(slots=True)
class EvalConfig:
    env_path: Path
    env: dict[str, str]
    provider_presets: list[str]
    max_models_per_provider: int
    preset_models: dict[str, list[str]]
    datasets: list[Path]
    output_dir: Path
    cache_enabled: bool
    cache_dir: Path
    timeout_s: float
    concurrency: int
    max_completion_tokens: int
    reasoning_effort: str | None
    temperature: float | None
    judge_provider: str
    judge_model: str
    judge_base_url: str | None
    judge_api_key: str | None
    judge_timeout_s: float
    judge_max_completion_tokens: int
    judge_reasoning_effort: str | None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "env_path": str(self.env_path),
            "provider_presets": list(self.provider_presets),
            "max_models_per_provider": self.max_models_per_provider,
            "datasets": [str(path) for path in self.datasets],
            "output_dir": str(self.output_dir),
            "cache_enabled": self.cache_enabled,
            "cache_dir": str(self.cache_dir),
            "timeout_s": self.timeout_s,
            "concurrency": self.concurrency,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "judge_timeout_s": self.judge_timeout_s,
            "judge_max_completion_tokens": self.judge_max_completion_tokens,
            "judge_reasoning_effort": self.judge_reasoning_effort,
        }


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


def _parse_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _parse_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_csv(value: str | None, *, lower: bool = False) -> list[str]:
    if value is None:
        return []
    parts = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if lower:
        return [part.lower() for part in parts]
    return parts


def _map_legacy_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    if normalized == "anthropic":
        # Anthropic is routed through OpenRouter in this suite.
        return "openrouter"
    return normalized


def _normalize_legacy_model_for_provider(
    *,
    resolved_provider: str,
    legacy_provider: str | None,
    model: str,
) -> str:
    normalized = model.strip()
    if not normalized:
        return normalized

    if resolved_provider != "openrouter":
        return normalized

    if "/" in normalized:
        return normalized

    legacy = (legacy_provider or "").strip().lower()
    lowered = normalized.lower()
    if legacy == "anthropic":
        anthropic_aliases = {
            "claude-4-5-haiku": "anthropic/claude-haiku-4.5",
            "claude-4-5-sonnet": "anthropic/claude-sonnet-4.5",
            "claude-4-5-opus": "anthropic/claude-opus-4.5",
        }
        return anthropic_aliases.get(lowered, f"anthropic/{normalized}")

    return normalized


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[7:].strip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None

    if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
        return key, value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].strip()

    return key, value


def load_dotenv_values(dotenv_path: Path) -> dict[str, str]:
    if not dotenv_path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        parsed_line = _parse_env_line(raw_line)
        if parsed_line is None:
            continue
        key, value = parsed_line
        parsed[key] = value
    return parsed


def _resolve_env(dotenv_path: Path | None = None) -> tuple[Path, dict[str, str]]:
    repo_root = Path.cwd()
    env_path = dotenv_path or (repo_root / ".env")

    dotenv_values = load_dotenv_values(env_path)
    merged = dict(dotenv_values)
    # Process env has precedence over .env
    for key, value in os.environ.items():
        merged[key] = value

    return env_path, merged


def _resolve_dataset_paths(env: dict[str, str]) -> list[Path]:
    dataset_csv = env.get("EVAL_DATASETS")
    default_paths = [Path("datasets/single_turn.yaml"), Path("datasets/multi_turn.yaml")]
    if dataset_csv is None:
        return default_paths

    datasets = [Path(item) for item in _parse_csv(dataset_csv)]
    return datasets or default_paths


def _resolve_provider_presets(env: dict[str, str]) -> list[str]:
    explicit = _parse_optional_str(env.get("EVAL_PROVIDER_PRESET"))
    if explicit is None:
        legacy_provider = _map_legacy_provider(_parse_optional_str(env.get("LLM_PROVIDER")))
        if legacy_provider in PROVIDERS:
            return [legacy_provider]

    raw = _parse_csv(env.get("EVAL_PROVIDER_PRESET", "openrouter"), lower=True)
    if not raw:
        return ["all"]
    if "all" in raw:
        return ["all"]

    selected = [provider for provider in raw if provider in PROVIDERS]
    return selected or ["all"]


def _resolve_preset_models(env: dict[str, str]) -> dict[str, list[str]]:
    models: dict[str, list[str]] = {}
    for provider in PROVIDERS:
        key = f"EVAL_PRESET_MODELS_{provider.upper()}"
        parsed = _parse_csv(env.get(key))
        if parsed:
            models[provider] = parsed

    use_legacy_model_pin = _parse_bool(env.get("EVAL_LEGACY_MODEL_PIN"), False)
    legacy_provider_raw = _parse_optional_str(env.get("LLM_PROVIDER"))
    legacy_provider = _map_legacy_provider(legacy_provider_raw)
    legacy_llm_model = _parse_optional_str(env.get("LLM_MODEL"))
    legacy_tool_model = _parse_optional_str(env.get("TOOL_MESSAGE_MODEL"))

    if use_legacy_model_pin and legacy_provider in PROVIDERS and (legacy_llm_model or legacy_tool_model):
        explicit_key = f"EVAL_PRESET_MODELS_{legacy_provider.upper()}"
        if not _parse_csv(env.get(explicit_key)):
            legacy_models = [item for item in [legacy_llm_model, legacy_tool_model] if item]
            normalized_models: list[str] = []
            seen: set[str] = set()
            for model in legacy_models:
                normalized = _normalize_legacy_model_for_provider(
                    resolved_provider=legacy_provider,
                    legacy_provider=legacy_provider_raw,
                    model=model,
                )
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                normalized_models.append(normalized)
            if normalized_models:
                models[legacy_provider] = normalized_models
    return models


def load_config(dotenv_path: Path | None = None) -> EvalConfig:
    env_path, env = _resolve_env(dotenv_path)

    return EvalConfig(
        env_path=env_path,
        env=env,
        provider_presets=_resolve_provider_presets(env),
        max_models_per_provider=max(1, _parse_int(env.get("EVAL_MAX_MODELS_PER_PROVIDER"), 20)),
        preset_models=_resolve_preset_models(env),
        datasets=_resolve_dataset_paths(env),
        output_dir=Path(env.get("EVAL_OUTPUT_DIR", "reports")),
        cache_enabled=_parse_bool(env.get("EVAL_CACHE_ENABLED"), True),
        cache_dir=Path(env.get("EVAL_CACHE_DIR", ".cache/llm_eval_suite")),
        timeout_s=max(1.0, _parse_float(env.get("EVAL_TIMEOUT_S"), 90.0)),
        concurrency=max(1, _parse_int(env.get("EVAL_CONCURRENCY"), 4)),
        max_completion_tokens=max(1, _parse_int(env.get("EVAL_MAX_COMPLETION_TOKENS"), 512)),
        reasoning_effort=_parse_optional_str(env.get("EVAL_REASONING_EFFORT") or env.get("LLM_REASONING_EFFORT")),
        temperature=_parse_optional_float(env.get("EVAL_TEMPERATURE")),
        judge_provider=(env.get("EVAL_JUDGE_PROVIDER") or "openai").strip().lower(),
        judge_model=(env.get("EVAL_JUDGE_MODEL") or "gpt-5.2").strip(),
        judge_base_url=_parse_optional_str(env.get("EVAL_JUDGE_BASE_URL")),
        judge_api_key=_parse_optional_str(env.get("EVAL_JUDGE_API_KEY")),
        judge_timeout_s=max(1.0, _parse_float(env.get("EVAL_JUDGE_TIMEOUT_S"), 90.0)),
        judge_max_completion_tokens=max(1, _parse_int(env.get("EVAL_JUDGE_MAX_COMPLETION_TOKENS"), 400)),
        judge_reasoning_effort=_parse_optional_str(
            env.get("EVAL_JUDGE_REASONING_EFFORT") or env.get("LLM_REASONING_EFFORT") or "xhigh"
        ),
    )


def parse_csv_arg(value: str | None) -> list[str]:
    return _parse_csv(value, lower=False)
