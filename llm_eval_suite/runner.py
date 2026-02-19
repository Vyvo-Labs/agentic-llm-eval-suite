from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from .cache import ResponseCache
from .config import EvalConfig
from .dataset_loader import load_cases
from .judge import JudgeRuntimeConfig, LLMJudge
from .models import (
    CaseResult,
    InferenceResult,
    ModelSummary,
    RunResults,
    UsageStats,
)
from .providers import (
    LLMEndpoint,
    ResolvedModels,
    resolve_candidate_models,
    resolve_judge_endpoint,
    split_model_reasoning_tag,
)
from .report import write_history_report, write_reports
from .scoring import combine_scores, evaluate_deterministic

try:
    import resource
except Exception:  # pragma: no cover - unavailable on some platforms (e.g., Windows)
    resource = None  # type: ignore[assignment]

PASS_THRESHOLD = 0.8
_TRANSIENT_ERROR_HINTS: tuple[str, ...] = (
    "429",
    "rate limit",
    "rate-limited",
    "temporarily rate-limited",
    "timeout",
    "timed out",
    "connection",
    "service unavailable",
    "overloaded",
    "try again",
    "retry",
)
_REQUEST_RETRY_ATTEMPTS = 3
_REQUEST_RETRY_BASE_BACKOFF_S = 0.6
_FD_RESERVE = 64
_FD_BUDGET_PER_WORKER = 4


@dataclass(slots=True)
class RunOptions:
    provider_filters: list[str]
    model_filters: list[str]
    datasets: list[Path] | None = None
    tags: set[str] | None = None
    categories: set[str] | None = None
    case_ids: set[str] | None = None
    max_cases: int | None = None
    concurrency: int | None = None
    timeout_s: float | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    cache_enabled: bool | None = None
    output_dir: Path | None = None


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _print_progress(*, completed: int, total: int, started_perf: float) -> None:
    if total <= 0:
        return
    elapsed = max(0.0, time.perf_counter() - started_perf)
    rate = (completed / elapsed) if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    eta_s = (remaining / rate) if rate > 0 else 0.0
    pct = (completed / total) * 100.0
    print(
        (
            f"[progress] {completed}/{total} ({pct:.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_s)}"
        ),
        flush=True,
    )


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_sha() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    return value or None


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _extract_usage(raw_usage: Any) -> UsageStats:
    if raw_usage is None:
        return UsageStats()

    def _get(value: Any, key: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def _as_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        if isinstance(value, float):
            return max(0, int(value))
        return 0

    def _as_float(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        return 0.0

    summary_payload = _get(raw_usage, "summary")
    if summary_payload is not None:
        success = _get(raw_usage, "success")
        if success is False:
            return UsageStats()
        raw_usage = summary_payload

    prompt_details = _get(raw_usage, "prompt_tokens_details", {}) or {}
    completion_details = _get(raw_usage, "completion_tokens_details", {}) or {}

    return UsageStats(
        llm_prompt_tokens=_as_int(_get(raw_usage, "llm_prompt_tokens", _get(raw_usage, "prompt_tokens", 0))),
        llm_prompt_cached_tokens=_as_int(
            _get(
                raw_usage,
                "llm_prompt_cached_tokens",
                _get(prompt_details, "cached_tokens", 0),
            )
        ),
        llm_input_audio_tokens=_as_int(
            _get(
                raw_usage,
                "llm_input_audio_tokens",
                _get(prompt_details, "audio_tokens", 0),
            )
        ),
        llm_completion_tokens=_as_int(
            _get(raw_usage, "llm_completion_tokens", _get(raw_usage, "completion_tokens", 0))
        ),
        llm_output_audio_tokens=_as_int(
            _get(
                raw_usage,
                "llm_output_audio_tokens",
                _get(completion_details, "audio_tokens", 0),
            )
        ),
        tts_characters_count=_as_int(_get(raw_usage, "tts_characters_count", 0)),
        tts_audio_duration=_as_float(_get(raw_usage, "tts_audio_duration", 0.0)),
        stt_audio_duration=_as_float(_get(raw_usage, "stt_audio_duration", 0.0)),
    )


def _extract_text_payload(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
            else:
                text_value = getattr(item, "text", None)
            if isinstance(text_value, str):
                fragments.append(text_value)
        return "".join(fragments)
    return str(content)


def _build_request_attempts(base: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = [dict(base)]

    if "reasoning_effort" in base:
        no_reasoning = dict(base)
        no_reasoning.pop("reasoning_effort", None)
        attempts.append(no_reasoning)

    if "max_completion_tokens" in base:
        legacy = dict(base)
        max_tokens_value = legacy.pop("max_completion_tokens")
        legacy["max_tokens"] = max_tokens_value
        attempts.append(legacy)

        if "reasoning_effort" in legacy:
            legacy_no_reasoning = dict(legacy)
            legacy_no_reasoning.pop("reasoning_effort", None)
            attempts.append(legacy_no_reasoning)

            boosted_tokens = dict(legacy_no_reasoning)
            boosted_value = boosted_tokens.get("max_tokens")
            if isinstance(boosted_value, int) and boosted_value > 0:
                boosted_tokens["max_tokens"] = min(max(boosted_value * 2, 1024), 2048)
                attempts.append(boosted_tokens)

                if boosted_tokens["max_tokens"] < 2048:
                    boosted_tokens_max = dict(boosted_tokens)
                    boosted_tokens_max["max_tokens"] = 2048
                    attempts.append(boosted_tokens_max)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt in attempts:
        key = json.dumps(attempt, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(attempt)
    return deduped


def _stream_output_requires_retry(output_text: str) -> bool:
    stripped = output_text.strip()
    if not stripped:
        return True
    # Some providers end stream early and leave an unclosed fenced block.
    if stripped.count("```") % 2 == 1:
        return True
    return False


def _is_transient_request_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _TRANSIENT_ERROR_HINTS)


def _chat_create_with_retries(
    *,
    client: OpenAI,
    request_payload: dict[str, Any],
    max_attempts: int = _REQUEST_RETRY_ATTEMPTS,
) -> Any:
    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None

    for attempt_index in range(attempts):
        try:
            return client.chat.completions.create(**request_payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            is_last_attempt = attempt_index >= (attempts - 1)
            if is_last_attempt or not _is_transient_request_error(last_error):
                raise last_error

            sleep_s = min(3.0, _REQUEST_RETRY_BASE_BACKOFF_S * (2**attempt_index))
            time.sleep(sleep_s)

    if last_error is not None:
        raise last_error
    raise RuntimeError("request failed without explicit error")


def _close_client(client: Any) -> None:
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:  # noqa: BLE001
            pass


def _fd_soft_limit() -> int | None:
    if resource is None:
        return None
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(soft_limit, int):
        return None
    if soft_limit <= 0 or soft_limit >= 10_000_000:
        return None
    return soft_limit


def _resolve_worker_concurrency(requested_concurrency: int) -> tuple[int, str | None]:
    requested = max(1, int(requested_concurrency))
    soft_limit = _fd_soft_limit()
    if soft_limit is None:
        return requested, None

    safe_max = max(1, (soft_limit - _FD_RESERVE) // _FD_BUDGET_PER_WORKER)
    if requested <= safe_max:
        return requested, None

    warning = (
        f"Capping concurrency from {requested} to {safe_max} due to open-file soft limit ({soft_limit}). "
        "Increase `ulimit -n` to allow higher concurrency."
    )
    return safe_max, warning


def _call_model_once(
    *,
    endpoint: LLMEndpoint,
    messages: list[dict[str, str]],
    timeout_s: float,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    temperature: float | None,
) -> InferenceResult:
    try:
        client = OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url)
    except Exception as exc:  # noqa: BLE001
        return InferenceResult(
            output_text="",
            ttft_s=None,
            total_latency_s=0.0,
            tokens_per_s=None,
            usage=UsageStats(),
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        request_base: dict[str, Any] = {
            "model": endpoint.request_model,
            "messages": messages,
            "timeout": timeout_s,
            "max_completion_tokens": max_completion_tokens,
        }
        if reasoning_effort:
            request_base["reasoning_effort"] = reasoning_effort
        if temperature is not None:
            request_base["temperature"] = temperature

        stream_attempt = dict(request_base)
        stream_attempt["stream"] = True
        stream_attempt["stream_options"] = {"include_usage": True}

        start = time.perf_counter()
        fragments: list[str] = []
        ttft_s: float | None = None
        usage = UsageStats()

        stream_error: str | None = None
        stream = None
        try:
            stream = _chat_create_with_retries(client=client, request_payload=stream_attempt)
            for chunk in stream:
                now = time.perf_counter()
                if now - start > timeout_s:
                    raise TimeoutError(f"stream response exceeded timeout_s={timeout_s}")
                raw_chunk_usage = getattr(chunk, "usage", None)
                if raw_chunk_usage is not None:
                    usage = _extract_usage(raw_chunk_usage)

                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                text_piece = _extract_text_payload(content)
                if text_piece:
                    fragments.append(text_piece)
                    if ttft_s is None:
                        ttft_s = now - start

            total_latency_s = time.perf_counter() - start
            output_text = "".join(fragments).strip()

            if _stream_output_requires_retry(output_text):
                stream_error = "stream output incomplete; retrying non-stream request"
                raise RuntimeError(stream_error)

            completion_tokens = usage.llm_completion_tokens
            tokens_per_s = (
                (completion_tokens / total_latency_s)
                if completion_tokens > 0 and total_latency_s > 0
                else None
            )

            if ttft_s is None:
                ttft_s = total_latency_s

            return InferenceResult(
                output_text=output_text,
                ttft_s=ttft_s,
                total_latency_s=total_latency_s,
                tokens_per_s=tokens_per_s,
                usage=usage,
            )
        except Exception as stream_exc:  # noqa: BLE001
            # Fall back to non-stream request for providers that do not support stream semantics.
            stream_error = f"{type(stream_exc).__name__}: {stream_exc}"
        finally:
            close_fn = getattr(stream, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:  # noqa: BLE001
                    pass

        attempts = _build_request_attempts(request_base)
        last_error = stream_error
        for attempt in attempts:
            try:
                started = time.perf_counter()
                response = _chat_create_with_retries(client=client, request_payload=attempt)
                total_latency_s = time.perf_counter() - started

                message = response.choices[0].message
                output_text = _extract_text_payload(message.content).strip()
                if not output_text:
                    last_error = "Empty model response."
                    continue
                usage = _extract_usage(getattr(response, "usage", None))

                completion_tokens = usage.llm_completion_tokens
                tokens_per_s = (
                    (completion_tokens / total_latency_s)
                    if completion_tokens > 0 and total_latency_s > 0
                    else None
                )

                return InferenceResult(
                    output_text=output_text,
                    ttft_s=total_latency_s,
                    total_latency_s=total_latency_s,
                    tokens_per_s=tokens_per_s,
                    usage=usage,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"

        return InferenceResult(
            output_text="",
            ttft_s=None,
            total_latency_s=time.perf_counter() - start,
            tokens_per_s=None,
            usage=UsageStats(),
            error=last_error,
        )
    finally:
        _close_client(client)


def _filter_cases(cases: list[Any], options: RunOptions) -> list[Any]:
    selected = list(cases)

    if options.tags:
        selected = [
            case
            for case in selected
            if options.tags.intersection({tag.lower() for tag in case.tags})
        ]

    if options.categories:
        selected = [
            case for case in selected if case.category.lower() in options.categories
        ]

    if options.case_ids:
        selected = [case for case in selected if case.id in options.case_ids]

    if options.max_cases is not None:
        selected = selected[: options.max_cases]

    return selected


def _filter_models(models: list[LLMEndpoint], filters: list[str]) -> list[LLMEndpoint]:
    if not filters:
        return models

    normalized_filters = [item.strip().lower() for item in filters if item.strip()]
    selected: list[LLMEndpoint] = []

    for model in models:
        display = model.display_name.lower()
        configured = model.configured_model.lower()
        if any(fragment in display or fragment in configured for fragment in normalized_filters):
            selected.append(model)

    return selected


def _aggregate_model_summaries(case_results: list[CaseResult]) -> list[ModelSummary]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in case_results:
        grouped.setdefault(result.model_name, []).append(result)

    summaries: list[ModelSummary] = []

    for model_name, rows in grouped.items():
        provider = rows[0].provider if rows else ""
        reasoning_efforts = sorted(
            {
                value.strip()
                for value in (row.reasoning_effort for row in rows)
                if value is not None and value.strip()
            }
        )
        reasoning_effort = (
            reasoning_efforts[0]
            if len(reasoning_efforts) == 1
            else ", ".join(reasoning_efforts)
            if reasoning_efforts
            else None
        )

        deterministic_values = [row.deterministic.score for row in rows if row.deterministic.score is not None]
        judge_values = [
            row.judge.final_score
            for row in rows
            if row.judge is not None and row.judge.final_score is not None
        ]
        final_values = [row.final_score for row in rows]
        pass_values = [1.0 if row.passed else 0.0 for row in rows]

        ttft_values = [row.inference.ttft_s for row in rows if row.inference.ttft_s is not None]
        latency_values = [row.inference.total_latency_s for row in rows if row.inference.total_latency_s is not None]
        tps_values = [row.inference.tokens_per_s for row in rows if row.inference.tokens_per_s is not None]

        error_count = sum(1 for row in rows if row.inference.error)

        summaries.append(
            ModelSummary(
                model_name=model_name,
                provider=provider,
                reasoning_effort=reasoning_effort,
                case_count=len(rows),
                error_count=error_count,
                deterministic_score_avg=_mean([float(v) for v in deterministic_values]),
                judge_score_avg=_mean([float(v) for v in judge_values]),
                final_score_avg=float(mean(final_values)) if final_values else 0.0,
                pass_rate=float(mean(pass_values)) if pass_values else 0.0,
                ttft_p50_s=_percentile([float(v) for v in ttft_values], 0.5),
                ttft_p95_s=_percentile([float(v) for v in ttft_values], 0.95),
                latency_p50_s=_percentile([float(v) for v in latency_values], 0.5),
                latency_p95_s=_percentile([float(v) for v in latency_values], 0.95),
                tokens_per_s_p50=_percentile([float(v) for v in tps_values], 0.5),
                tokens_per_s_p95=_percentile([float(v) for v in tps_values], 0.95),
            )
        )

    summaries.sort(key=lambda summary: summary.final_score_avg, reverse=True)
    return summaries


def _resolve_run_dir(base_output_dir: Path) -> tuple[str, Path]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = timestamp
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _resolve_reasoning_effort_for_model(
    *,
    endpoint: LLMEndpoint,
    configured_reasoning_effort: str | None,
) -> str | None:
    _, tagged_reasoning_effort = split_model_reasoning_tag(endpoint.configured_model)
    if tagged_reasoning_effort is not None:
        return tagged_reasoning_effort

    model_tail = endpoint.configured_model.strip().lower().split("/")[-1]
    legacy_reasoning_aliases: dict[str, str] = {
        "gpt-5.2-none": "none",
        "openai-5.2-none": "none",
        "gpt-5-mini-minimal": "minimal",
        "openai-5-mini-minimal": "minimal",
    }
    if model_tail in legacy_reasoning_aliases:
        return legacy_reasoning_aliases[model_tail]
    return configured_reasoning_effort


def _cache_request_fingerprint(
    *,
    endpoint: LLMEndpoint,
    messages: list[dict[str, str]],
    config: EvalConfig,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    credential_fingerprint = hashlib.sha256(endpoint.api_key.encode("utf-8")).hexdigest()
    return {
        "provider": endpoint.provider,
        "model": endpoint.configured_model,
        "request_model": endpoint.request_model,
        "base_url": endpoint.base_url,
        "credentials_sha256": credential_fingerprint,
        "messages": messages,
        "max_completion_tokens": config.max_completion_tokens,
        "reasoning_effort": reasoning_effort,
        "temperature": config.temperature,
    }


def _model_case_worker(
    *,
    endpoint: LLMEndpoint,
    case: Any,
    cache: ResponseCache | None,
    config: EvalConfig,
    judge: LLMJudge,
    lock: threading.Lock,
) -> CaseResult:
    messages = case.input.to_messages()
    resolved_reasoning_effort = _resolve_reasoning_effort_for_model(
        endpoint=endpoint,
        configured_reasoning_effort=config.reasoning_effort,
    )

    request_fingerprint = _cache_request_fingerprint(
        endpoint=endpoint,
        messages=messages,
        config=config,
        reasoning_effort=resolved_reasoning_effort,
    )

    inference: InferenceResult
    cache_key = cache.key_for(request_fingerprint) if cache is not None else None

    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None and _stream_output_requires_retry(str(cached.get("output_text", ""))):
            cached = None

        if cached is not None:
            usage_payload = cached.get("usage", {})
            inference = InferenceResult(
                output_text=str(cached.get("output_text", "")),
                ttft_s=cached.get("ttft_s"),
                total_latency_s=float(cached.get("total_latency_s", 0.0)),
                tokens_per_s=cached.get("tokens_per_s"),
                usage=_extract_usage(usage_payload),
                cache_hit=True,
                error=cached.get("error"),
            )
        else:
            inference = _call_model_once(
                endpoint=endpoint,
                messages=messages,
                timeout_s=config.timeout_s,
                max_completion_tokens=config.max_completion_tokens,
                reasoning_effort=resolved_reasoning_effort,
                temperature=config.temperature,
            )
            lock.acquire()
            try:
                cache.put(
                    cache_key,
                    {
                        "output_text": inference.output_text,
                        "ttft_s": inference.ttft_s,
                        "total_latency_s": inference.total_latency_s,
                        "tokens_per_s": inference.tokens_per_s,
                        "usage": asdict(inference.usage),
                        "error": inference.error,
                    },
                )
            finally:
                lock.release()
    else:
        inference = _call_model_once(
            endpoint=endpoint,
            messages=messages,
            timeout_s=config.timeout_s,
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=resolved_reasoning_effort,
            temperature=config.temperature,
        )

    deterministic = evaluate_deterministic(case, inference.output_text)
    judge_score = None

    if LLMJudge.should_run(case, deterministic):
        judge_score = judge.evaluate(
            case=case,
            conversation_messages=messages,
            candidate_response=inference.output_text,
        )

    combined_score = combine_scores(
        deterministic,
        judge_score.final_score if judge_score else None,
        deterministic_weight=case.weights.deterministic,
        judge_weight=case.weights.judge,
    )

    return CaseResult(
        model_name=endpoint.display_name,
        provider=endpoint.provider,
        reasoning_effort=resolved_reasoning_effort,
        case_id=case.id,
        case_name=case.name,
        category=case.category,
        tags=list(case.tags),
        inference=inference,
        deterministic=deterministic,
        judge=judge_score,
        final_score=combined_score,
        passed=combined_score >= PASS_THRESHOLD,
    )


def run_benchmark(config: EvalConfig, options: RunOptions) -> RunResults:
    dataset_paths = options.datasets if options.datasets is not None else config.datasets
    loaded_cases = load_cases(dataset_paths)

    filtered_cases = _filter_cases(loaded_cases, options)

    resolved_models: ResolvedModels = resolve_candidate_models(config, options.provider_filters)
    model_candidates = _filter_models(resolved_models.candidates, options.model_filters)

    warnings = list(resolved_models.warnings)

    judge_endpoint, judge_error = resolve_judge_endpoint(config)
    if judge_error is not None:
        warnings.append(judge_error)

    judge = LLMJudge(
        judge_endpoint,
        JudgeRuntimeConfig(
            timeout_s=config.judge_timeout_s,
            max_completion_tokens=config.judge_max_completion_tokens,
            reasoning_effort=config.judge_reasoning_effort,
        ),
    )

    cache_enabled = config.cache_enabled if options.cache_enabled is None else options.cache_enabled
    cache = ResponseCache(config.cache_dir) if cache_enabled else None
    cache_lock = threading.Lock()

    started_at = _utc_now_iso()

    case_results: list[CaseResult] = []

    tasks: list[tuple[LLMEndpoint, Any]] = []
    for endpoint in model_candidates:
        for case in filtered_cases:
            tasks.append((endpoint, case))

    total_tasks = len(tasks)
    progress_every = max(1, min(25, total_tasks // 20)) if total_tasks else 1
    started_perf = time.perf_counter()

    requested_concurrency = options.concurrency or config.concurrency
    concurrency, concurrency_warning = _resolve_worker_concurrency(requested_concurrency)
    if concurrency_warning is not None:
        warnings.append(concurrency_warning)

    if tasks:
        if concurrency <= 1:
            for index, (endpoint, case) in enumerate(tasks, start=1):
                case_results.append(
                    _model_case_worker(
                        endpoint=endpoint,
                        case=case,
                        cache=cache,
                        config=config,
                        judge=judge,
                        lock=cache_lock,
                    )
                )
                if index == total_tasks or index % progress_every == 0:
                    _print_progress(completed=index, total=total_tasks, started_perf=started_perf)
        else:
            ordered: list[CaseResult | None] = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                future_to_idx = {
                    pool.submit(
                        _model_case_worker,
                        endpoint=endpoint,
                        case=case,
                        cache=cache,
                        config=config,
                        judge=judge,
                        lock=cache_lock,
                    ): index
                    for index, (endpoint, case) in enumerate(tasks)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    ordered[idx] = future.result()
                    completed += 1
                    if completed == total_tasks or completed % progress_every == 0:
                        _print_progress(completed=completed, total=total_tasks, started_perf=started_perf)

            case_results = [item for item in ordered if item is not None]

    model_summaries = _aggregate_model_summaries(case_results)

    finished_at = _utc_now_iso()

    return RunResults(
        run_id="",
        started_at=started_at,
        finished_at=finished_at,
        config=config.safe_summary(),
        git_sha=_git_sha(),
        datasets=[str(path) for path in dataset_paths],
        model_summaries=model_summaries,
        case_results=case_results,
        warnings=warnings,
    )


def persist_run_results(results: RunResults, output_dir: Path) -> Path:
    run_id, run_dir = _resolve_run_dir(output_dir)
    results.run_id = run_id

    results_path = run_dir / "results.json"
    markdown_path = run_dir / "leaderboard.md"
    html_path = run_dir / "leaderboard.html"
    raw_responses_path = run_dir / "raw_responses.jsonl"

    payload = results.to_dict()
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(payload, markdown_path, html_path)

    with raw_responses_path.open("w", encoding="utf-8") as handle:
        for case_result in results.case_results:
            handle.write(
                json.dumps(
                    {
                        "model_name": case_result.model_name,
                        "provider": case_result.provider,
                        "case_id": case_result.case_id,
                        "case_name": case_result.case_name,
                        "response_text": case_result.inference.output_text,
                        "error": case_result.inference.error,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    write_history_report(output_dir)

    return run_dir
