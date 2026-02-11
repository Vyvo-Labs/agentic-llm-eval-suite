from pathlib import Path

import pytest

from llm_eval_suite.dataset_loader import DatasetValidationError, load_cases

PROJECT_DATASETS = [Path("datasets/single_turn.yaml"), Path("datasets/multi_turn.yaml")]
NEW_CASE_IDS = {
    "st_extract_contact_card",
    "st_extract_tasks_json",
    "st_extract_trip_prefs",
    "st_extract_medical_flags",
    "st_extract_finance_limits",
    "st_transform_rename_fields",
    "st_transform_filter_fields",
    "st_transform_sort_events",
    "st_transform_units",
    "st_classify_intent_json",
    "st_safety_phishing_refusal",
    "st_safety_malware_refusal",
    "st_safety_self_harm_referral",
    "st_safety_weapon_refusal",
    "st_safety_fraud_refusal",
    "st_safety_doxxing_refusal",
    "st_format_one_sentence",
    "st_format_no_markdown",
    "st_format_three_words",
    "st_contract_tool_call_weather",
    "st_contract_tool_call_search",
    "st_contract_tool_call_file_lookup",
    "st_reasoning_date_math_json",
    "st_reasoning_compare_table_json",
    "mt_preference_update_latest_wins",
    "mt_name_correction_persistence",
    "mt_timezone_memory",
    "mt_language_switch_persistence",
    "mt_contradiction_resolution",
    "mt_task_status_update",
    "mt_budget_constraint_memory",
    "mt_contact_preference_memory",
    "mt_system_priority_over_user",
    "mt_refuse_after_context_setup",
    "mt_style_then_content_conflict",
    "mt_format_lock_json",
    "mt_followup_with_new_constraint",
    "mt_policy_override_attempt",
    "mt_multi_fact_recall",
    "mt_preference_ordering_json",
    "mt_clarify_before_action",
    "mt_empathetic_deescalation",
    "mt_uncertainty_hallucination_guard",
    "mt_tradeoff_reasoning",
    "mt_plan_revision_with_feedback",
    "mt_tool_intent_schema_routing",
    "mt_policy_cot_boundary",
    "mt_partial_info_followup_question",
    "mt_multi_objective_prioritization",
    "mt_safety_boundary_persistence",
    "mt_context_switch_return",
    "mt_error_acknowledge_and_recover",
}

CHAT_INTELLIGENCE_CASE_IDS = {
    "mt_clarify_before_action",
    "mt_empathetic_deescalation",
    "mt_uncertainty_hallucination_guard",
    "mt_tradeoff_reasoning",
    "mt_plan_revision_with_feedback",
    "mt_tool_intent_schema_routing",
    "mt_policy_cot_boundary",
    "mt_partial_info_followup_question",
    "mt_multi_objective_prioritization",
    "mt_safety_boundary_persistence",
    "mt_context_switch_return",
    "mt_error_acknowledge_and_recover",
}


def test_load_cases_schema_v1(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.yaml"
    dataset.write_text(
        """
schema_version: 1
cases:
  - id: c1
    name: Case 1
    type: single_turn
    category: general
    tags: [a]
    input:
      user: hello
    expected:
      must_include: [hello]
""".strip(),
        encoding="utf-8",
    )

    cases = load_cases([dataset])
    assert len(cases) == 1
    assert cases[0].id == "c1"
    assert cases[0].input.to_messages()[0]["content"] == "hello"


def test_duplicate_case_ids_raises(tmp_path: Path) -> None:
    dataset_a = tmp_path / "a.yaml"
    dataset_b = tmp_path / "b.yaml"
    payload = """
schema_version: 1
cases:
  - id: dup
    name: One
    input:
      user: hi
""".strip()
    dataset_a.write_text(payload, encoding="utf-8")
    dataset_b.write_text(payload, encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_cases([dataset_a, dataset_b])


def test_project_datasets_have_expanded_case_count() -> None:
    cases = load_cases(PROJECT_DATASETS)
    assert len(cases) >= 57


def test_new_cases_exist_and_have_minimum_deterministic_signal() -> None:
    cases = load_cases(PROJECT_DATASETS)
    by_id = {case.id: case for case in cases}

    missing_ids = [case_id for case_id in NEW_CASE_IDS if case_id not in by_id]
    assert not missing_ids

    def _deterministic_check_count(case: object) -> int:
        expected = case.expected
        return (
            len(expected.exact)
            + len(expected.regex)
            + len(expected.must_include)
            + (1 if expected.json_valid else 0)
        )

    for case_id in NEW_CASE_IDS:
        assert _deterministic_check_count(by_id[case_id]) >= 2


def test_new_cases_have_balanced_judge_coverage() -> None:
    cases = load_cases(PROJECT_DATASETS)
    selected = [case for case in cases if case.id in NEW_CASE_IDS]

    assert len(selected) == 52
    force_count = sum(1 for case in selected if case.judge_rubric.force)
    assert force_count >= 30

    deterministic_heavy = sum(
        1 for case in selected if case.weights.deterministic > case.weights.judge
    )
    judge_heavy = sum(1 for case in selected if case.weights.judge > case.weights.deterministic)
    balanced_or_judge = len(selected) - deterministic_heavy

    assert deterministic_heavy >= 25
    assert judge_heavy >= 18
    assert balanced_or_judge >= 20


def test_chat_intelligence_cases_are_multi_turn_and_judge_heavy() -> None:
    cases = load_cases(PROJECT_DATASETS)
    by_id = {case.id: case for case in cases}

    selected = [by_id[case_id] for case_id in CHAT_INTELLIGENCE_CASE_IDS]
    assert len(selected) == 12
    assert all(case.type == "multi_turn" for case in selected)
    assert sum(1 for case in selected if case.judge_rubric.force) >= 10
    assert sum(1 for case in selected if case.weights.judge >= 0.6) >= 10

    def _deterministic_check_count(case: object) -> int:
        expected = case.expected
        return (
            len(expected.exact)
            + len(expected.regex)
            + len(expected.must_include)
            + (1 if expected.json_valid else 0)
        )

    assert all(_deterministic_check_count(case) >= 2 for case in selected)
