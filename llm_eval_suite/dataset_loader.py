from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import CaseInput, CaseWeights, EvalCase, ExpectedChecks, JudgeRubric


class DatasetValidationError(ValueError):
    pass


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DatasetValidationError(f"'{field_name}' must be a list of strings.")

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DatasetValidationError(f"'{field_name}' must contain only strings.")
        stripped = item.strip()
        if stripped:
            parsed.append(stripped)
    return parsed


def _normalize_input(input_payload: dict[str, Any], *, case_id: str) -> CaseInput:
    if not isinstance(input_payload, dict):
        raise DatasetValidationError(f"Case '{case_id}': 'input' must be an object.")

    system = input_payload.get("system")
    user = input_payload.get("user")
    messages = input_payload.get("messages")

    if system is not None and not isinstance(system, str):
        raise DatasetValidationError(f"Case '{case_id}': 'input.system' must be a string.")
    if user is not None and not isinstance(user, str):
        raise DatasetValidationError(f"Case '{case_id}': 'input.user' must be a string.")

    parsed_messages: list[dict[str, str]] = []
    if messages is not None:
        if not isinstance(messages, list):
            raise DatasetValidationError(f"Case '{case_id}': 'input.messages' must be a list.")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise DatasetValidationError(
                    f"Case '{case_id}': 'input.messages[{index}]' must be an object."
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise DatasetValidationError(
                    f"Case '{case_id}': each message must include string 'role' and 'content'."
                )
            parsed_messages.append({"role": role.strip(), "content": content})

    case_input = CaseInput(system=system, user=user, messages=parsed_messages)
    messages_out = case_input.to_messages()
    if not messages_out:
        raise DatasetValidationError(
            f"Case '{case_id}': at least one input message is required (user or messages)."
        )
    return case_input


def _normalize_expected(payload: Any, *, case_id: str) -> ExpectedChecks:
    if payload is None:
        return ExpectedChecks()
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Case '{case_id}': 'expected' must be an object.")

    exact = _as_str_list(payload.get("exact"), field_name=f"{case_id}.expected.exact")
    regex = _as_str_list(payload.get("regex"), field_name=f"{case_id}.expected.regex")
    must_include = _as_str_list(payload.get("must_include"), field_name=f"{case_id}.expected.must_include")

    json_valid_raw = payload.get("json_valid", False)
    if not isinstance(json_valid_raw, bool):
        raise DatasetValidationError(f"Case '{case_id}': 'expected.json_valid' must be boolean.")

    return ExpectedChecks(exact=exact, regex=regex, must_include=must_include, json_valid=json_valid_raw)


def _normalize_rubric(payload: Any, *, case_id: str) -> JudgeRubric:
    if payload is None:
        return JudgeRubric()
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Case '{case_id}': 'judge_rubric' must be an object.")

    criteria = _as_str_list(payload.get("criteria"), field_name=f"{case_id}.judge_rubric.criteria")

    scale_min = payload.get("scale_min", 0)
    scale_max = payload.get("scale_max", 5)
    force = payload.get("force", False)

    if not isinstance(scale_min, int) or not isinstance(scale_max, int):
        raise DatasetValidationError(f"Case '{case_id}': rubric scale bounds must be integers.")
    if scale_max <= scale_min:
        raise DatasetValidationError(f"Case '{case_id}': rubric scale_max must be greater than scale_min.")
    if not isinstance(force, bool):
        raise DatasetValidationError(f"Case '{case_id}': 'judge_rubric.force' must be boolean.")

    return JudgeRubric(criteria=criteria, scale_min=scale_min, scale_max=scale_max, force=force)


def _normalize_weights(payload: Any, *, case_id: str) -> CaseWeights:
    if payload is None:
        return CaseWeights()
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Case '{case_id}': 'weights' must be an object.")

    deterministic = payload.get("deterministic", 0.5)
    judge = payload.get("judge", 0.5)

    if not isinstance(deterministic, (int, float)) or not isinstance(judge, (int, float)):
        raise DatasetValidationError(f"Case '{case_id}': weights must be numeric.")

    deterministic_f = float(deterministic)
    judge_f = float(judge)
    if deterministic_f < 0 or judge_f < 0:
        raise DatasetValidationError(f"Case '{case_id}': weights cannot be negative.")

    return CaseWeights(deterministic=deterministic_f, judge=judge_f)


def _parse_case(raw_case: dict[str, Any], *, path: Path) -> EvalCase:
    case_id = raw_case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetValidationError(f"{path}: each case requires non-empty string 'id'.")
    case_id = case_id.strip()

    name = raw_case.get("name")
    if not isinstance(name, str) or not name.strip():
        raise DatasetValidationError(f"{path}: case '{case_id}' requires non-empty string 'name'.")

    case_type = raw_case.get("type", "single_turn")
    if case_type not in {"single_turn", "multi_turn"}:
        raise DatasetValidationError(f"{path}: case '{case_id}' has invalid type '{case_type}'.")

    category_raw = raw_case.get("category", "general")
    if not isinstance(category_raw, str) or not category_raw.strip():
        raise DatasetValidationError(f"{path}: case '{case_id}' has invalid category.")
    category = category_raw.strip()

    tags = _as_str_list(raw_case.get("tags", []), field_name=f"{case_id}.tags")
    case_input = _normalize_input(raw_case.get("input", {}), case_id=case_id)
    expected = _normalize_expected(raw_case.get("expected"), case_id=case_id)
    rubric = _normalize_rubric(raw_case.get("judge_rubric"), case_id=case_id)
    weights = _normalize_weights(raw_case.get("weights"), case_id=case_id)

    metadata_raw = raw_case.get("metadata", {})
    if metadata_raw is None:
        metadata_raw = {}
    if not isinstance(metadata_raw, dict):
        raise DatasetValidationError(f"Case '{case_id}': 'metadata' must be an object.")

    return EvalCase(
        id=case_id,
        name=name.strip(),
        type=case_type,
        category=category,
        tags=tags,
        input=case_input,
        expected=expected,
        judge_rubric=rubric,
        weights=weights,
        metadata=dict(metadata_raw),
    )


def load_cases(paths: list[Path]) -> list[EvalCase]:
    all_cases: list[EvalCase] = []
    seen_case_ids: set[str] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            continue

        if isinstance(payload, dict):
            schema_version = payload.get("schema_version", 1)
            if schema_version != 1:
                raise DatasetValidationError(f"{path}: unsupported schema_version '{schema_version}'.")
            cases_payload = payload.get("cases", [])
        elif isinstance(payload, list):
            schema_version = 1
            cases_payload = payload
        else:
            raise DatasetValidationError(f"{path}: dataset root must be object or list.")

        if not isinstance(cases_payload, list):
            raise DatasetValidationError(f"{path}: 'cases' must be a list.")

        for raw_case in cases_payload:
            if not isinstance(raw_case, dict):
                raise DatasetValidationError(f"{path}: each case entry must be an object.")
            parsed = _parse_case(raw_case, path=path)
            if parsed.id in seen_case_ids:
                raise DatasetValidationError(f"Duplicate case id detected: '{parsed.id}'.")
            seen_case_ids.add(parsed.id)
            all_cases.append(parsed)

    return all_cases

