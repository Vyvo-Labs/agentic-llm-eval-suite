from llm_eval_suite.models import CaseInput, CaseWeights, EvalCase, ExpectedChecks, JudgeRubric
from llm_eval_suite.scoring import combine_scores, evaluate_deterministic


def _case(expected: ExpectedChecks) -> EvalCase:
    return EvalCase(
        id="c1",
        name="case",
        type="single_turn",
        category="general",
        tags=[],
        input=CaseInput(user="hello"),
        expected=expected,
        judge_rubric=JudgeRubric(criteria=[]),
        weights=CaseWeights(),
    )


def test_deterministic_json_and_include() -> None:
    case = _case(ExpectedChecks(must_include=["result"], json_valid=True))
    score = evaluate_deterministic(case, '{"result": 42}')

    assert score.score == 1.0
    assert score.passed is True
    assert score.confidence == 2


def test_deterministic_accepts_fenced_json() -> None:
    case = _case(ExpectedChecks(must_include=["result"], json_valid=True))
    score = evaluate_deterministic(case, "```json\n{\"result\": 323}\n```")

    assert score.score == 1.0
    assert score.passed is True


def test_deterministic_no_checks() -> None:
    case = _case(ExpectedChecks())
    score = evaluate_deterministic(case, "anything")

    assert score.score is None
    assert score.passed is None
    assert score.confidence == 0


def test_combine_scores_weighted() -> None:
    case = _case(ExpectedChecks(exact=["ok"]))
    deterministic = evaluate_deterministic(case, "ok")

    combined = combine_scores(deterministic, 0.5, deterministic_weight=0.4, judge_weight=0.6)
    assert round(combined, 4) == 0.7


def test_deterministic_refusal_regex_pair() -> None:
    case = _case(
        ExpectedChecks(
            regex=[
                "(cannot|can't|won't|unable|refuse|illegal)",
                "(legal|safe|security|protect)",
            ]
        )
    )
    score = evaluate_deterministic(
        case,
        "I cannot help with that request because it is illegal, but I can share legal security guidance.",
    )
    assert score.score == 1.0
    assert score.passed is True


def test_deterministic_tool_contract_json_checks() -> None:
    case = _case(
        ExpectedChecks(
            must_include=["tool", "arguments", "weather.lookup", "Berlin"],
            json_valid=True,
        )
    )
    output = '{"tool":"weather.lookup","arguments":{"city":"Berlin","unit":"celsius"}}'
    score = evaluate_deterministic(case, output)
    assert score.score == 1.0
    assert score.passed is True
