from __future__ import annotations

from dataclasses import fields
import threading
from pathlib import Path
from types import SimpleNamespace

import llm_eval_suite.runner as runner_mod
from llm_eval_suite.config import EvalConfig
from llm_eval_suite.judge import JudgeRuntimeConfig, LLMJudge
from llm_eval_suite.models import (
    CaseInput,
    CaseWeights,
    EvalCase,
    ExpectedChecks,
    InferenceResult,
    JudgeRubric,
    UsageStats,
)
from llm_eval_suite.providers import LLMEndpoint
from llm_eval_suite.runner import (
    _build_request_attempts,
    _call_model_once,
    _extract_usage,
    _model_case_worker,
    _stream_output_requires_retry,
)


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


def _endpoint() -> LLMEndpoint:
    return LLMEndpoint(
        provider="openai",
        configured_model="gpt-5-mini",
        request_model="gpt-5-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        api_key="sk-test",
    )


def _case() -> EvalCase:
    return EvalCase(
        id="c1",
        name="Cache usage case",
        type="single_turn",
        category="general",
        tags=[],
        input=CaseInput(user="hello"),
        expected=ExpectedChecks(),
        judge_rubric=JudgeRubric(criteria=[]),
        weights=CaseWeights(),
    )


def test_usage_stats_matches_external_agent_usage_contract_keys() -> None:
    field_names = {field.name for field in fields(UsageStats)}
    assert field_names == {
        "llm_prompt_tokens",
        "llm_prompt_cached_tokens",
        "llm_input_audio_tokens",
        "llm_completion_tokens",
        "llm_output_audio_tokens",
        "tts_characters_count",
        "tts_audio_duration",
        "stt_audio_duration",
    }


def test_extract_usage_maps_openai_usage_dict_with_nested_details() -> None:
    stats = _extract_usage(
        {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 12, "audio_tokens": 3},
            "completion_tokens_details": {"audio_tokens": 2},
            "tts_characters_count": 100,
            "tts_audio_duration": 1.5,
            "stt_audio_duration": 2.25,
        }
    )

    assert stats.llm_prompt_tokens == 120
    assert stats.llm_prompt_cached_tokens == 12
    assert stats.llm_input_audio_tokens == 3
    assert stats.llm_completion_tokens == 40
    assert stats.llm_output_audio_tokens == 2
    assert stats.tts_characters_count == 100
    assert stats.tts_audio_duration == 1.5
    assert stats.stt_audio_duration == 2.25


def test_extract_usage_supports_external_usage_summary_envelope() -> None:
    stats = _extract_usage(
        {
            "success": True,
            "summary": {
                "llm_prompt_tokens": 21,
                "llm_prompt_cached_tokens": 4,
                "llm_input_audio_tokens": 1,
                "llm_completion_tokens": 8,
                "llm_output_audio_tokens": 0,
                "tts_characters_count": 300,
                "tts_audio_duration": 2.4,
                "stt_audio_duration": 1.1,
            },
        }
    )

    assert stats.llm_prompt_tokens == 21
    assert stats.llm_prompt_cached_tokens == 4
    assert stats.llm_input_audio_tokens == 1
    assert stats.llm_completion_tokens == 8
    assert stats.llm_output_audio_tokens == 0
    assert stats.tts_characters_count == 300
    assert stats.tts_audio_duration == 2.4
    assert stats.stt_audio_duration == 1.1


def test_extract_usage_supports_external_usage_summary_object_envelope() -> None:
    stats = _extract_usage(
        SimpleNamespace(
            success=True,
            summary=SimpleNamespace(
                llm_prompt_tokens=31,
                llm_prompt_cached_tokens=5,
                llm_input_audio_tokens=2,
                llm_completion_tokens=9,
                llm_output_audio_tokens=1,
                tts_characters_count=410,
                tts_audio_duration=3.3,
                stt_audio_duration=1.7,
            ),
        )
    )

    assert stats.llm_prompt_tokens == 31
    assert stats.llm_prompt_cached_tokens == 5
    assert stats.llm_input_audio_tokens == 2
    assert stats.llm_completion_tokens == 9
    assert stats.llm_output_audio_tokens == 1
    assert stats.tts_characters_count == 410
    assert stats.tts_audio_duration == 3.3
    assert stats.stt_audio_duration == 1.7


def test_extract_usage_handles_external_usage_summary_failure_envelope() -> None:
    stats = _extract_usage({"success": False, "error": "metrics unavailable", "summary": None})
    assert stats == UsageStats()


def test_extract_usage_failure_envelope_ignores_summary_payload() -> None:
    stats = _extract_usage(
        {
            "success": False,
            "error": "metrics unavailable",
            "summary": {"llm_prompt_tokens": 999, "llm_completion_tokens": 999},
        }
    )
    assert stats == UsageStats()


def test_extract_usage_supports_object_payload_and_numeric_normalization() -> None:
    stats = _extract_usage(
        SimpleNamespace(
            prompt_tokens=10.9,
            completion_tokens=7.2,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5.9, audio_tokens=1.2),
            completion_tokens_details=SimpleNamespace(audio_tokens=2.7),
            tts_characters_count=80.8,
            tts_audio_duration=3,
            stt_audio_duration=4.5,
        )
    )

    assert stats.llm_prompt_tokens == 10
    assert stats.llm_prompt_cached_tokens == 5
    assert stats.llm_input_audio_tokens == 1
    assert stats.llm_completion_tokens == 7
    assert stats.llm_output_audio_tokens == 2
    assert stats.tts_characters_count == 80
    assert stats.tts_audio_duration == 3.0
    assert stats.stt_audio_duration == 4.5


def test_extract_usage_clamps_negative_metrics_to_zero() -> None:
    stats = _extract_usage(
        {
            "llm_prompt_tokens": -1,
            "llm_prompt_cached_tokens": -2,
            "llm_input_audio_tokens": -3,
            "llm_completion_tokens": -4,
            "llm_output_audio_tokens": -5,
            "tts_characters_count": -6,
            "tts_audio_duration": -7.1,
            "stt_audio_duration": -8.2,
        }
    )

    assert stats.llm_prompt_tokens == 0
    assert stats.llm_prompt_cached_tokens == 0
    assert stats.llm_input_audio_tokens == 0
    assert stats.llm_completion_tokens == 0
    assert stats.llm_output_audio_tokens == 0
    assert stats.tts_characters_count == 0
    assert stats.tts_audio_duration == 0.0
    assert stats.stt_audio_duration == 0.0


def test_extract_usage_defaults_to_zero_when_missing() -> None:
    stats = _extract_usage(None)
    assert stats.llm_prompt_tokens == 0
    assert stats.llm_prompt_cached_tokens == 0
    assert stats.llm_input_audio_tokens == 0
    assert stats.llm_completion_tokens == 0
    assert stats.llm_output_audio_tokens == 0
    assert stats.tts_characters_count == 0
    assert stats.tts_audio_duration == 0.0
    assert stats.stt_audio_duration == 0.0


def test_model_case_worker_cache_hit_uses_new_usage_schema_fields() -> None:
    class _FakeCache:
        def key_for(self, _: object) -> str:
            return "cache-key"

        def get(self, _: str) -> dict[str, object]:
            return {
                "output_text": "cached answer",
                "ttft_s": 0.1,
                "total_latency_s": 0.3,
                "tokens_per_s": 66.0,
                "usage": {
                    "llm_prompt_tokens": 101,
                    "llm_prompt_cached_tokens": 11,
                    "llm_input_audio_tokens": 2,
                    "llm_completion_tokens": 20,
                    "llm_output_audio_tokens": 1,
                    "tts_characters_count": 230,
                    "tts_audio_duration": 4.2,
                    "stt_audio_duration": 0.9,
                },
            }

        def put(self, _: str, __: dict[str, object]) -> None:
            raise AssertionError("put should not be called on cache hit")

    judge = LLMJudge(endpoint=None, config=JudgeRuntimeConfig(timeout_s=1.0, max_completion_tokens=10, reasoning_effort=None))
    result = _model_case_worker(
        endpoint=_endpoint(),
        case=_case(),
        cache=_FakeCache(),
        config=_config(),
        judge=judge,
        lock=threading.Lock(),
    )

    assert result.inference.cache_hit is True
    assert result.inference.usage.llm_prompt_tokens == 101
    assert result.inference.usage.llm_prompt_cached_tokens == 11
    assert result.inference.usage.llm_input_audio_tokens == 2
    assert result.inference.usage.llm_completion_tokens == 20
    assert result.inference.usage.llm_output_audio_tokens == 1
    assert result.inference.usage.tts_characters_count == 230
    assert result.inference.usage.tts_audio_duration == 4.2
    assert result.inference.usage.stt_audio_duration == 0.9


def test_call_model_once_computes_throughput_from_llm_completion_tokens(
    monkeypatch: object,
) -> None:
    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if kwargs.get("stream"):
                raise RuntimeError("stream unsupported")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage={"llm_completion_tokens": 25},
            )

    fake_completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=fake_completions)

    monkeypatch.setattr(runner_mod, "OpenAI", _FakeOpenAI)

    result = _call_model_once(
        endpoint=_endpoint(),
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=5.0,
        max_completion_tokens=32,
        reasoning_effort=None,
        temperature=None,
    )

    assert result.output_text == "ok"
    assert result.usage.llm_completion_tokens == 25
    assert result.tokens_per_s is not None
    assert result.tokens_per_s > 0
    assert fake_completions.calls[0]["stream"] is True


def test_stream_output_requires_retry_for_empty_or_unclosed_fence() -> None:
    assert _stream_output_requires_retry("") is True
    assert _stream_output_requires_retry("```json\n{\"result\": 1}\n") is True
    assert _stream_output_requires_retry("{\"result\": 1}") is False


def test_call_model_once_retries_non_stream_when_stream_returns_empty(
    monkeypatch: object,
) -> None:
    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if kwargs.get("stream"):
                return [SimpleNamespace(choices=[])]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{\"result\":323}"))],
                usage={"completion_tokens": 9},
            )

    fake_completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=fake_completions)

    monkeypatch.setattr(runner_mod, "OpenAI", _FakeOpenAI)

    result = _call_model_once(
        endpoint=_endpoint(),
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=5.0,
        max_completion_tokens=32,
        reasoning_effort=None,
        temperature=None,
    )

    assert len(fake_completions.calls) == 2
    assert fake_completions.calls[0]["stream"] is True
    assert "stream" not in fake_completions.calls[1]
    assert result.output_text == "{\"result\":323}"
    assert result.usage.llm_completion_tokens == 9


def test_call_model_once_falls_back_when_stream_exceeds_timeout(
    monkeypatch: object,
) -> None:
    class _FakeStream:
        def __iter__(self) -> "_FakeStream":
            return self

        def __next__(self) -> object:
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))],
                usage=None,
            )

        def close(self) -> None:
            return None

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if kwargs.get("stream"):
                return _FakeStream()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="fallback output"))],
                usage={"completion_tokens": 6},
            )

    fake_completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=fake_completions)

    perf_values = iter([0.0, 6.0, 7.0, 7.4])
    monkeypatch.setattr(runner_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(runner_mod.time, "perf_counter", lambda: next(perf_values))

    result = _call_model_once(
        endpoint=_endpoint(),
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=5.0,
        max_completion_tokens=32,
        reasoning_effort=None,
        temperature=None,
    )

    assert len(fake_completions.calls) == 2
    assert fake_completions.calls[0]["stream"] is True
    assert "stream" not in fake_completions.calls[1]
    assert result.output_text == "fallback output"
    assert result.usage.llm_completion_tokens == 6


def test_model_case_worker_refreshes_unusable_cached_output(monkeypatch: object) -> None:
    class _FakeCache:
        def __init__(self) -> None:
            self.put_payload: dict[str, object] | None = None

        def key_for(self, _: object) -> str:
            return "cache-key"

        def get(self, _: str) -> dict[str, object]:
            return {
                "output_text": "",
                "ttft_s": 0.0,
                "total_latency_s": 0.0,
                "tokens_per_s": None,
                "usage": {},
            }

        def put(self, _: str, payload: dict[str, object]) -> None:
            self.put_payload = payload

    def _fake_call_model_once(**_: object) -> InferenceResult:
        return InferenceResult(
            output_text="fresh output",
            ttft_s=0.1,
            total_latency_s=0.2,
            tokens_per_s=10.0,
            usage=UsageStats(llm_completion_tokens=2),
            cache_hit=False,
            error=None,
        )

    monkeypatch.setattr(runner_mod, "_call_model_once", _fake_call_model_once)

    cache = _FakeCache()
    judge = LLMJudge(endpoint=None, config=JudgeRuntimeConfig(timeout_s=1.0, max_completion_tokens=10, reasoning_effort=None))
    result = _model_case_worker(
        endpoint=_endpoint(),
        case=_case(),
        cache=cache,
        config=_config(),
        judge=judge,
        lock=threading.Lock(),
    )

    assert result.inference.cache_hit is False
    assert result.inference.output_text == "fresh output"
    assert cache.put_payload is not None
    assert cache.put_payload["output_text"] == "fresh output"


def test_call_model_once_retries_when_non_stream_response_is_empty(
    monkeypatch: object,
) -> None:
    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if kwargs.get("stream"):
                raise RuntimeError("stream unsupported")
            if "max_completion_tokens" in kwargs:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                    usage={"completion_tokens": 1},
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="fallback output"))],
                usage={"completion_tokens": 4},
            )

    fake_completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=fake_completions)

    monkeypatch.setattr(runner_mod, "OpenAI", _FakeOpenAI)

    result = _call_model_once(
        endpoint=_endpoint(),
        messages=[{"role": "user", "content": "hello"}],
        timeout_s=5.0,
        max_completion_tokens=32,
        reasoning_effort=None,
        temperature=None,
    )

    assert result.output_text == "fallback output"
    assert result.usage.llm_completion_tokens == 4
    assert len(fake_completions.calls) >= 3


def test_build_request_attempts_includes_boosted_max_tokens_fallback() -> None:
    attempts = _build_request_attempts(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "timeout": 5,
            "max_completion_tokens": 512,
            "reasoning_effort": "none",
        }
    )
    max_tokens_values = [a.get("max_tokens") for a in attempts if "max_tokens" in a]
    assert 512 in max_tokens_values
    assert 1024 in max_tokens_values
    assert 2048 in max_tokens_values
