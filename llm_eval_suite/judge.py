from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .models import (
    DeterministicScore,
    EvalCase,
    JudgeCriterionScore,
    JudgeScore,
)
from .providers import LLMEndpoint


@dataclass(slots=True)
class JudgeRuntimeConfig:
    timeout_s: float
    max_completion_tokens: int
    reasoning_effort: str | None
    max_retries: int = 2
    retry_backoff_s: float = 0.4


class LLMJudge:
    def __init__(self, endpoint: LLMEndpoint | None, config: JudgeRuntimeConfig) -> None:
        self._endpoint = endpoint
        self._config = config

    @staticmethod
    def should_run(case: EvalCase, deterministic: DeterministicScore) -> bool:
        if not case.judge_rubric.criteria:
            return False
        if case.judge_rubric.force:
            return True
        if deterministic.score is None:
            return True
        return deterministic.score < 1.0

    def evaluate(
        self,
        *,
        case: EvalCase,
        conversation_messages: list[dict[str, str]],
        candidate_response: str,
    ) -> JudgeScore:
        if self._endpoint is None:
            return JudgeScore(
                final_score=None,
                criterion_scores=[],
                flags=["judge_unavailable"],
                rationale="Dedicated judge model is not configured.",
                error="judge endpoint unavailable",
            )

        prompt = self._build_prompt(
            case=case,
            conversation_messages=conversation_messages,
            candidate_response=candidate_response,
        )

        client = OpenAI(api_key=self._endpoint.api_key, base_url=self._endpoint.base_url)
        response_text, error = self._request_judgment(client=client, prompt=prompt)
        if error is not None:
            return JudgeScore(
                final_score=None,
                criterion_scores=[],
                flags=["judge_request_error"],
                rationale="Judge request failed.",
                error=error,
            )

        parsed = self._parse_judge_response(response_text)
        return parsed

    def _request_judgment(self, *, client: OpenAI, prompt: str) -> tuple[str, str | None]:
        params: dict[str, Any] = {
            "model": self._endpoint.request_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an impartial LLM benchmark judge. "
                        "Return JSON only, with no extra prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "timeout": self._config.timeout_s,
            "max_completion_tokens": self._config.max_completion_tokens,
        }
        if self._config.reasoning_effort:
            params["reasoning_effort"] = self._config.reasoning_effort

        attempts = self._build_attempts(params)

        last_error: str | None = None
        max_retries = max(0, int(self._config.max_retries))
        retry_backoff_s = max(0.0, float(self._config.retry_backoff_s))
        for attempt in attempts:
            for retry_idx in range(max_retries + 1):
                try:
                    response = client.chat.completions.create(**attempt)
                    response_text = self._extract_response_text(response).strip()
                    if response_text:
                        return response_text, None
                    last_error = "Empty judge response."
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = f"{type(exc).__name__}: {exc}"
                    if retry_idx >= max_retries or not self._is_transient_error(last_error):
                        break
                    if retry_backoff_s > 0:
                        time.sleep(retry_backoff_s * (2**retry_idx))

        return "", (last_error or "Unknown judge request error")

    def _build_attempts(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Tolerate OpenAI-compatible endpoints with partial parameter support.
        attempts: list[dict[str, Any]] = [dict(params)]

        without_reasoning = dict(params)
        without_reasoning.pop("reasoning_effort", None)
        attempts.append(without_reasoning)

        if "max_completion_tokens" in params:
            legacy_tokens = dict(without_reasoning)
            legacy_tokens.pop("max_completion_tokens", None)
            legacy_tokens["max_tokens"] = self._config.max_completion_tokens
            attempts.append(legacy_tokens)

            boosted_tokens = dict(legacy_tokens)
            if isinstance(boosted_tokens.get("max_tokens"), int):
                current = boosted_tokens["max_tokens"]
                boosted_tokens["max_tokens"] = min(max(current * 2, 1024), 2048)
                attempts.append(boosted_tokens)
                if boosted_tokens["max_tokens"] < 2048:
                    maxed_tokens = dict(boosted_tokens)
                    maxed_tokens["max_tokens"] = 2048
                    attempts.append(maxed_tokens)

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for attempt in attempts:
            key = json.dumps(attempt, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(attempt)
        return deduped

    @staticmethod
    def _is_transient_error(error_text: str) -> bool:
        lowered = error_text.lower()
        transient_markers = (
            "connection",
            "timeout",
            "timed out",
            "temporarily",
            "rate limit",
            "429",
            "service unavailable",
            "503",
            "gateway",
            "reset by peer",
        )
        return any(marker in lowered for marker in transient_markers)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        try:
            message = response.choices[0].message
            content = message.content
        except Exception:  # noqa: BLE001
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                text_part = None
                if isinstance(item, dict):
                    text_part = item.get("text")
                else:
                    text_part = getattr(item, "text", None)
                if isinstance(text_part, str):
                    fragments.append(text_part)
            return "".join(fragments)

        return str(content)

    @staticmethod
    def _build_prompt(
        *,
        case: EvalCase,
        conversation_messages: list[dict[str, str]],
        candidate_response: str,
    ) -> str:
        rubric_lines = "\n".join(f"- {criterion}" for criterion in case.judge_rubric.criteria)
        conversation = json.dumps(conversation_messages, ensure_ascii=False, indent=2)
        expected = {
            "exact": case.expected.exact,
            "regex": case.expected.regex,
            "must_include": case.expected.must_include,
            "json_valid": case.expected.json_valid,
        }

        return (
            "Score the candidate response using the rubric.\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            '  "final_score": <float 0..1>,\n'
            '  "flags": [<string>],\n'
            '  "rationale": <string>,\n'
            '  "criterion_scores": [\n'
            "    {\n"
            '      "criterion": <string>,\n'
            '      "score": <float>,\n'
            '      "max_score": <float>,\n'
            '      "reason": <string>\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Case id: {case.id}\n"
            f"Case name: {case.name}\n"
            f"Case type: {case.type}\n"
            f"Expected checks: {json.dumps(expected, ensure_ascii=False)}\n"
            f"Rubric criteria:\n{rubric_lines or '- (none provided)'}\n\n"
            f"Conversation:\n{conversation}\n\n"
            f"Candidate response:\n{candidate_response}\n"
        )

    def _parse_judge_response(self, response_text: str) -> JudgeScore:
        payload = self._extract_json_payload(response_text)
        if payload is None:
            return JudgeScore(
                final_score=None,
                criterion_scores=[],
                flags=["judge_parse_error"],
                rationale="Judge returned non-JSON output.",
                error="Unable to parse judge response as JSON.",
            )

        criterion_scores = self._parse_criterion_scores(payload)

        final_score_raw = payload.get("final_score")
        final_score: float | None
        if isinstance(final_score_raw, (float, int)):
            final_score = max(0.0, min(1.0, float(final_score_raw)))
        else:
            final_score = self._compute_score_from_criteria(criterion_scores)

        flags_raw = payload.get("flags", [])
        flags = [str(flag) for flag in flags_raw if isinstance(flag, (str, int, float))]

        rationale_raw = payload.get("rationale", "")
        rationale = rationale_raw if isinstance(rationale_raw, str) else json.dumps(rationale_raw)

        return JudgeScore(
            final_score=final_score,
            criterion_scores=criterion_scores,
            flags=flags,
            rationale=rationale,
        )

    @staticmethod
    def _extract_json_payload(response_text: str) -> dict[str, Any] | None:
        if not response_text.strip():
            return None

        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Extract first JSON object if wrapped in prose.
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    @staticmethod
    def _parse_criterion_scores(payload: dict[str, Any]) -> list[JudgeCriterionScore]:
        raw_scores = payload.get("criterion_scores")
        if not isinstance(raw_scores, list):
            return []

        parsed: list[JudgeCriterionScore] = []
        for item in raw_scores:
            if not isinstance(item, dict):
                continue

            criterion = str(item.get("criterion", "")).strip()
            score_raw = item.get("score")
            max_raw = item.get("max_score", 5)
            reason = str(item.get("reason", "")).strip()

            if not criterion:
                continue
            if not isinstance(score_raw, (int, float)):
                continue
            if not isinstance(max_raw, (int, float)):
                continue
            max_score = float(max_raw)
            if max_score <= 0:
                continue

            parsed.append(
                JudgeCriterionScore(
                    criterion=criterion,
                    score=float(score_raw),
                    max_score=max_score,
                    reason=reason,
                )
            )

        return parsed

    @staticmethod
    def _compute_score_from_criteria(criteria: list[JudgeCriterionScore]) -> float | None:
        if not criteria:
            return None

        ratios: list[float] = []
        for criterion in criteria:
            ratios.append(max(0.0, min(1.0, criterion.score / criterion.max_score)))
        return sum(ratios) / len(ratios)
