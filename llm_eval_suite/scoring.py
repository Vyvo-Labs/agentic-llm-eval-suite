from __future__ import annotations

import json
import re
from statistics import mean

from .models import DeterministicScore, EvalCase

_UNICODE_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _safe_regex_search(pattern: str, text: str) -> bool:
    try:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE) is not None:
            return True

        normalized_text = text.translate(_UNICODE_PUNCT_TRANSLATION)
        if normalized_text != text:
            return re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE) is not None
        return False
    except re.error:
        return False


def _contains_valid_json_payload(output_text: str) -> bool:
    text = output_text.strip()
    if not text:
        return False

    candidates: list[str] = [text]

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())

    bracketed = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if bracketed:
        candidates.append(bracketed.group(1).strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return True
        except json.JSONDecodeError:
            continue

    return False


def evaluate_deterministic(case: EvalCase, output_text: str) -> DeterministicScore:
    checks: dict[str, bool] = {}
    normalized_output = _normalize_text(output_text)

    for index, exact in enumerate(case.expected.exact):
        checks[f"exact[{index}]"] = normalized_output == _normalize_text(exact)

    for index, pattern in enumerate(case.expected.regex):
        checks[f"regex[{index}]"] = _safe_regex_search(pattern, output_text)

    lowered_output = output_text.lower()
    for index, token in enumerate(case.expected.must_include):
        checks[f"must_include[{index}]"] = token.lower() in lowered_output

    if case.expected.json_valid:
        checks["json_valid"] = _contains_valid_json_payload(output_text)

    if not checks:
        return DeterministicScore(score=None, checks=checks, passed=None, confidence=0)

    score = mean(1.0 if passed else 0.0 for passed in checks.values())
    passed = all(checks.values())
    return DeterministicScore(score=score, checks=checks, passed=passed, confidence=len(checks))


def combine_scores(
    deterministic_score: DeterministicScore,
    judge_score: float | None,
    *,
    deterministic_weight: float,
    judge_weight: float,
) -> float:
    det_value = deterministic_score.score

    if det_value is None and judge_score is None:
        return 0.0
    if judge_score is None:
        return max(0.0, min(1.0, float(det_value or 0.0)))
    if det_value is None:
        return max(0.0, min(1.0, float(judge_score)))

    det_weight = max(0.0, deterministic_weight)
    llm_weight = max(0.0, judge_weight)
    if det_weight == 0 and llm_weight == 0:
        return max(0.0, min(1.0, float(det_value)))

    combined = (det_value * det_weight + judge_score * llm_weight) / (det_weight + llm_weight)
    return max(0.0, min(1.0, float(combined)))
