from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import llm_eval_suite.judge as judge_mod
from llm_eval_suite.judge import JudgeRuntimeConfig, LLMJudge
from llm_eval_suite.models import (
    CaseInput,
    CaseWeights,
    DeterministicScore,
    EvalCase,
    ExpectedChecks,
    JudgeRubric,
)
from llm_eval_suite.providers import LLMEndpoint


def _case(*, criteria: list[str], force: bool = False) -> EvalCase:
    return EvalCase(
        id="judge-case",
        name="Judge case",
        type="single_turn",
        category="general",
        tags=[],
        input=CaseInput(user="hello"),
        expected=ExpectedChecks(),
        judge_rubric=JudgeRubric(criteria=criteria, force=force),
        weights=CaseWeights(),
    )


def _deterministic(score: float | None) -> DeterministicScore:
    passed = None if score is None else score >= 1.0
    return DeterministicScore(score=score, checks={}, passed=passed, confidence=0)


def _judge() -> LLMJudge:
    endpoint = LLMEndpoint(
        provider="openai",
        configured_model="gpt-5-mini",
        request_model="gpt-5-mini",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        api_key="sk-test",
    )
    return LLMJudge(
        endpoint=endpoint,
        config=JudgeRuntimeConfig(timeout_s=5.0, max_completion_tokens=128, reasoning_effort="medium"),
    )


def test_should_run_matrix() -> None:
    assert LLMJudge.should_run(_case(criteria=[]), _deterministic(0.2)) is False
    assert LLMJudge.should_run(_case(criteria=["quality"], force=True), _deterministic(1.0)) is True
    assert LLMJudge.should_run(_case(criteria=["quality"]), _deterministic(None)) is True
    assert LLMJudge.should_run(_case(criteria=["quality"]), _deterministic(1.0)) is False
    assert LLMJudge.should_run(_case(criteria=["quality"]), _deterministic(0.8)) is True


def test_extract_json_payload_from_wrapped_response() -> None:
    payload = LLMJudge._extract_json_payload(
        'Judge output:\n{"final_score": 0.42, "flags": [], "rationale": "ok", "criterion_scores": []}\nDone.'
    )
    assert payload is not None
    assert payload["final_score"] == 0.42


def test_parse_judge_response_clamps_out_of_range_final_score() -> None:
    judge = _judge()
    result = judge._parse_judge_response(
        json.dumps(
            {
                "final_score": 1.7,
                "flags": ["a"],
                "rationale": "ok",
                "criterion_scores": [],
            }
        )
    )

    assert result.final_score == 1.0
    assert result.flags == ["a"]
    assert result.rationale == "ok"


def test_parse_judge_response_derives_score_from_criteria() -> None:
    judge = _judge()
    result = judge._parse_judge_response(
        json.dumps(
            {
                "flags": [],
                "rationale": "criteria only",
                "criterion_scores": [
                    {"criterion": "accuracy", "score": 3, "max_score": 5, "reason": "mostly correct"},
                    {"criterion": "safety", "score": 10, "max_score": 5, "reason": "excellent"},
                    {"criterion": "", "score": 5, "max_score": 5, "reason": "ignored"},
                ],
            }
        )
    )

    assert result.final_score == pytest.approx(0.8)
    assert len(result.criterion_scores) == 2


def test_parse_judge_response_returns_parse_error_for_non_json() -> None:
    judge = _judge()
    result = judge._parse_judge_response("not-json")

    assert result.final_score is None
    assert result.flags == ["judge_parse_error"]
    assert result.error == "Unable to parse judge response as JSON."


def test_request_judgment_falls_back_from_reasoning_and_legacy_tokens() -> None:
    judge = _judge()

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise RuntimeError("reasoning unsupported")
            if "max_completion_tokens" in kwargs:
                raise RuntimeError("max_completion_tokens unsupported")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"final_score":0.6,"flags":[],"rationale":"ok","criterion_scores":[]}'
                        )
                    )
                ]
            )

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response_text, error = judge._request_judgment(client=fake_client, prompt="judge prompt")

    assert error is None
    assert '"final_score":0.6' in response_text
    assert len(completions.calls) == 3
    assert "reasoning_effort" in completions.calls[0]
    assert "max_completion_tokens" in completions.calls[0]
    assert "reasoning_effort" not in completions.calls[1]
    assert "max_completion_tokens" in completions.calls[1]
    assert "max_completion_tokens" not in completions.calls[2]
    assert "max_tokens" in completions.calls[2]


def test_evaluate_returns_unavailable_when_judge_endpoint_missing() -> None:
    judge = LLMJudge(
        endpoint=None,
        config=JudgeRuntimeConfig(timeout_s=5.0, max_completion_tokens=128, reasoning_effort=None),
    )
    result = judge.evaluate(
        case=_case(criteria=["quality"]),
        conversation_messages=[{"role": "user", "content": "hello"}],
        candidate_response="answer",
    )

    assert result.final_score is None
    assert result.flags == ["judge_unavailable"]
    assert result.error == "judge endpoint unavailable"


def test_evaluate_returns_request_error_when_request_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = _judge()

    class _FakeOpenAI:
        def __init__(self, **_: object) -> None:
            pass

    monkeypatch.setattr(judge_mod, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(judge, "_request_judgment", lambda **_: ("", "boom"))

    result = judge.evaluate(
        case=_case(criteria=["quality"]),
        conversation_messages=[{"role": "user", "content": "hello"}],
        candidate_response="answer",
    )

    assert result.final_score is None
    assert result.flags == ["judge_request_error"]
    assert result.error == "boom"


def test_request_judgment_retries_transient_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = _judge()

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                raise RuntimeError("APIConnectionError: Connection error.")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"final_score":0.9,"flags":[],"rationale":"ok","criterion_scores":[]}'
                        )
                    )
                ]
            )

    monkeypatch.setattr(judge_mod.time, "sleep", lambda _value: None)

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    response_text, error = judge._request_judgment(client=fake_client, prompt="judge prompt")

    assert error is None
    assert '"final_score":0.9' in response_text
    assert len(completions.calls) == 2


def test_request_judgment_skips_empty_response_and_uses_next_attempt() -> None:
    judge = _judge()

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs: object) -> object:
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"final_score":0.7,"flags":[],"rationale":"ok","criterion_scores":[]}'
                        )
                    )
                ]
            )

    completions = _FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    response_text, error = judge._request_judgment(client=fake_client, prompt="judge prompt")

    assert error is None
    assert '"final_score":0.7' in response_text
    assert len(completions.calls) == 2
    assert "reasoning_effort" in completions.calls[0]
    assert "reasoning_effort" not in completions.calls[1]
