from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CaseType = Literal["single_turn", "multi_turn"]


@dataclass(slots=True)
class CaseInput:
    system: str | None = None
    user: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_messages(self) -> list[dict[str, str]]:
        if self.messages:
            return [dict(message) for message in self.messages]

        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if self.user:
            messages.append({"role": "user", "content": self.user})
        return messages


@dataclass(slots=True)
class ExpectedChecks:
    exact: list[str] = field(default_factory=list)
    regex: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    json_valid: bool = False


@dataclass(slots=True)
class JudgeRubric:
    criteria: list[str] = field(default_factory=list)
    scale_min: int = 0
    scale_max: int = 5
    force: bool = False


@dataclass(slots=True)
class CaseWeights:
    deterministic: float = 0.5
    judge: float = 0.5


@dataclass(slots=True)
class EvalCase:
    id: str
    name: str
    type: CaseType
    category: str
    tags: list[str]
    input: CaseInput
    expected: ExpectedChecks
    judge_rubric: JudgeRubric
    weights: CaseWeights
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UsageStats:
    llm_prompt_tokens: int = 0
    llm_prompt_cached_tokens: int = 0
    llm_input_audio_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_output_audio_tokens: int = 0
    tts_characters_count: int = 0
    tts_audio_duration: float = 0.0
    stt_audio_duration: float = 0.0


@dataclass(slots=True)
class InferenceResult:
    output_text: str
    ttft_s: float | None
    total_latency_s: float
    tokens_per_s: float | None
    usage: UsageStats = field(default_factory=UsageStats)
    cache_hit: bool = False
    error: str | None = None


@dataclass(slots=True)
class DeterministicScore:
    score: float | None
    checks: dict[str, bool]
    passed: bool | None
    confidence: int


@dataclass(slots=True)
class JudgeCriterionScore:
    criterion: str
    score: float
    max_score: float
    reason: str


@dataclass(slots=True)
class JudgeScore:
    final_score: float | None
    criterion_scores: list[JudgeCriterionScore]
    flags: list[str]
    rationale: str
    error: str | None = None


@dataclass(slots=True)
class CaseResult:
    model_name: str
    provider: str
    case_id: str
    case_name: str
    category: str
    tags: list[str]
    inference: InferenceResult
    deterministic: DeterministicScore
    judge: JudgeScore | None
    final_score: float
    passed: bool


@dataclass(slots=True)
class ModelSummary:
    model_name: str
    provider: str
    case_count: int
    error_count: int
    deterministic_score_avg: float | None
    judge_score_avg: float | None
    final_score_avg: float
    pass_rate: float
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    tokens_per_s_p50: float | None
    tokens_per_s_p95: float | None


@dataclass(slots=True)
class RunResults:
    run_id: str
    started_at: str
    finished_at: str
    config: dict[str, Any]
    git_sha: str | None
    datasets: list[str]
    model_summaries: list[ModelSummary]
    case_results: list[CaseResult]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
