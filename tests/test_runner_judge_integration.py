from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import llm_eval_suite.judge as judge_mod
import llm_eval_suite.runner as runner_mod
from llm_eval_suite.config import EvalConfig
from llm_eval_suite.models import InferenceResult, UsageStats
from llm_eval_suite.providers import LLMEndpoint
from llm_eval_suite.runner import RunOptions, run_benchmark


def _write_dataset(path: Path) -> Path:
    dataset = path / "judge_cases.yaml"
    dataset.write_text(
        """
schema_version: 1
cases:
  - id: "judge_case"
    name: "Judge-required case"
    type: "single_turn"
    category: "integration"
    tags: ["judge"]
    input:
      user: "say hello"
    expected:
      must_include: ["required-token"]
    judge_rubric:
      criteria:
        - "Response quality is acceptable."
      force: false
    weights:
      deterministic: 0.5
      judge: 0.5
""".strip(),
        encoding="utf-8",
    )
    return dataset


def _base_config(tmp_path: Path, *, env: dict[str, str]) -> EvalConfig:
    return EvalConfig(
        env_path=tmp_path / ".env",
        env=env,
        provider_presets=["openrouter"],
        max_models_per_provider=1,
        preset_models={"openrouter": ["z-ai/glm-4.7"]},
        datasets=[_write_dataset(tmp_path)],
        output_dir=tmp_path / "reports",
        cache_enabled=False,
        cache_dir=tmp_path / ".cache",
        timeout_s=30,
        concurrency=1,
        max_completion_tokens=128,
        reasoning_effort=None,
        temperature=None,
        judge_provider="openai",
        judge_model="gpt-5-mini",
        judge_base_url=None,
        judge_api_key=None,
        judge_timeout_s=30,
        judge_max_completion_tokens=128,
        judge_reasoning_effort=None,
    )


def _options() -> RunOptions:
    return RunOptions(provider_filters=[], model_filters=[])


def _stub_inference(**_: object) -> InferenceResult:
    return InferenceResult(
        output_text="hello",
        ttft_s=0.1,
        total_latency_s=0.2,
        tokens_per_s=10.0,
        usage=UsageStats(llm_completion_tokens=2),
        cache_hit=False,
        error=None,
    )


def test_run_benchmark_marks_judge_unavailable_when_no_judge_key(monkeypatch: object, tmp_path: Path) -> None:
    config = _base_config(tmp_path, env={"OPENROUTER_API_KEY": "or-test"})
    monkeypatch.setattr(runner_mod, "_call_model_once", _stub_inference)

    results = run_benchmark(config, _options())

    assert len(results.case_results) == 1
    case = results.case_results[0]
    assert case.judge is not None
    assert case.judge.flags == ["judge_unavailable"]
    assert case.judge.error == "judge endpoint unavailable"
    assert any("Judge model unavailable: missing API key." in warning for warning in results.warnings)


def test_run_benchmark_marks_judge_request_error_on_judge_failure(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    config = _base_config(
        tmp_path,
        env={"OPENROUTER_API_KEY": "or-test", "OPENAI_API_KEY": "sk-test"},
    )
    monkeypatch.setattr(runner_mod, "_call_model_once", _stub_inference)

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            pass

    monkeypatch.setattr(judge_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(judge_mod.LLMJudge, "_request_judgment", lambda *_args, **_kwargs: ("", "judge down"))

    results = run_benchmark(config, _options())

    assert len(results.case_results) == 1
    case = results.case_results[0]
    assert case.judge is not None
    assert case.judge.flags == ["judge_request_error"]
    assert case.judge.error == "judge down"
    # With judge unavailable and deterministic miss, combined score falls back to deterministic score.
    assert case.final_score == 0.0
