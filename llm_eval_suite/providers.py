from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import EvalConfig, PROVIDERS

_PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "together": "https://api.together.xyz/v1",
    "cerebras": "https://api.cerebras.ai/v1",
}

_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

_MODEL_ALIAS_TO_BASE: dict[str, str] = {
    "gpt-5.2-none": "gpt-5.2",
    "openai-5.2-none": "gpt-5.2",
    "gpt-5-mini-minimal": "gpt-5-mini",
    "openai-5-mini-minimal": "gpt-5-mini",
}

_DEFAULT_PRESET_MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-5-mini",
        "gpt-5",
        "gpt-5.2",
        "openai-5-mini-minimal",
        "openai-5.2-none",
    ],
    "groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    # OpenRouter defaults bias toward current open models.
    "openrouter": [
        "moonshotai/kimi-k2.5",
        "z-ai/glm-5",
        "z-ai/glm-4.7",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.1",
        "anthropic/claude-opus-4.5",
        "openai/gpt-5-mini",
        "openai/gpt-5",
        "openai/gpt-5.2",
        "openai/gpt-5.2-none",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "z-ai/glm-4.7-flash",
        "minimax/minimax-m2-her",
        "stepfun/step-3.5-flash",
        "xiaomi/mimo-v2-flash",
        "nvidia/nemotron-3-nano-30b-a3b",
        "meituan/longcat-flash-chat",
    ],
    "fireworks": ["accounts/fireworks/models/llama-v3p1-8b-instruct"],
    "together": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
    "cerebras": ["llama3.1-8b"],
}


@dataclass(slots=True)
class LLMEndpoint:
    provider: str
    configured_model: str
    request_model: str
    base_url: str | None
    api_key_env: str
    api_key: str

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.configured_model}"


@dataclass(slots=True)
class ResolvedModels:
    candidates: list[LLMEndpoint]
    warnings: list[str]


def normalize_llm_model_for_provider(provider: str, model: str) -> str:
    normalized_model = (model or "").strip()
    if not normalized_model:
        return normalized_model

    normalized_provider = (provider or "").strip().lower()
    if normalized_provider:
        prefix = f"{normalized_provider}/"
        if normalized_model.lower().startswith(prefix):
            normalized_model = normalized_model[len(prefix) :]

    lowered_model = normalized_model.lower()
    if "/" in normalized_model:
        vendor, tail = normalized_model.split("/", 1)
        mapped_tail = _MODEL_ALIAS_TO_BASE.get(tail.lower())
        if mapped_tail:
            return f"{vendor}/{mapped_tail}"
        return normalized_model

    mapped_model = _MODEL_ALIAS_TO_BASE.get(lowered_model)
    if mapped_model:
        if normalized_provider == "openrouter":
            return f"openai/{mapped_model}"
        return mapped_model

    return normalized_model


def _provider_base_url(provider: str, env: dict[str, str]) -> str | None:
    if provider == "openai":
        raw = (env.get("OPENAI_BASE_URL") or "").strip()
        return raw or None
    return _PROVIDER_BASE_URLS[provider]


def _resolve_provider_models(config: EvalConfig, provider: str) -> list[str]:
    configured = config.preset_models.get(provider)
    if configured:
        models = configured
    else:
        models = _DEFAULT_PRESET_MODELS.get(provider, [])

    deduped: list[str] = []
    seen: set[str] = set()
    for model in models:
        normalized = model.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)

    return deduped[: config.max_models_per_provider]


def _selected_providers(config: EvalConfig, provider_filters: Iterable[str] | None = None) -> list[str]:
    filters = [item.strip().lower() for item in (provider_filters or []) if item.strip()]
    if filters:
        return [provider for provider in filters if provider in PROVIDERS]

    if "all" in config.provider_presets:
        return list(PROVIDERS)

    return [provider for provider in config.provider_presets if provider in PROVIDERS]


def resolve_candidate_models(config: EvalConfig, provider_filters: Iterable[str] | None = None) -> ResolvedModels:
    warnings: list[str] = []
    candidates: list[LLMEndpoint] = []

    for provider in _selected_providers(config, provider_filters):
        api_key_env = _PROVIDER_API_KEY_ENV[provider]
        api_key = (config.env.get(api_key_env) or "").strip()
        if not api_key:
            warnings.append(
                f"Skipping provider '{provider}': missing required credential '{api_key_env}'."
            )
            continue

        models = _resolve_provider_models(config, provider)
        if not models:
            warnings.append(f"Skipping provider '{provider}': no models configured for preset.")
            continue

        base_url = _provider_base_url(provider, config.env)

        for model in models:
            candidates.append(
                LLMEndpoint(
                    provider=provider,
                    configured_model=model,
                    request_model=normalize_llm_model_for_provider(provider, model),
                    base_url=base_url,
                    api_key_env=api_key_env,
                    api_key=api_key,
                )
            )

    return ResolvedModels(candidates=candidates, warnings=warnings)


def resolve_judge_endpoint(config: EvalConfig) -> tuple[LLMEndpoint | None, str | None]:
    provider = (config.judge_provider or "openai").strip().lower()
    if provider not in PROVIDERS:
        return None, f"Unsupported EVAL_JUDGE_PROVIDER '{provider}'."

    if not config.judge_model.strip():
        return None, "EVAL_JUDGE_MODEL cannot be empty."

    api_key_env = _PROVIDER_API_KEY_ENV[provider]
    api_key = (config.judge_api_key or "").strip() or (config.env.get(api_key_env) or "").strip()
    if not api_key:
        return None, (
            "Judge model unavailable: missing API key. "
            f"Set EVAL_JUDGE_API_KEY or {api_key_env}."
        )

    if provider == "openai":
        base_url = (config.judge_base_url or "").strip() or (config.env.get("OPENAI_BASE_URL") or "").strip() or None
    else:
        base_url = _PROVIDER_BASE_URLS[provider]

    endpoint = LLMEndpoint(
        provider=provider,
        configured_model=config.judge_model,
        request_model=normalize_llm_model_for_provider(provider, config.judge_model),
        base_url=base_url,
        api_key_env=api_key_env,
        api_key=api_key,
    )
    return endpoint, None
