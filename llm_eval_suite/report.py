from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _format_num(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean_value = _mean(values)
    if mean_value is None:
        return None
    variance = sum((item - mean_value) ** 2 for item in values) / len(values)
    return math.sqrt(variance)


def _sort_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12f}"


def _format_count(value: float | None) -> str:
    if value is None:
        return "-"

    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))

    text = f"{value:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text


def _normalize_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_reasoning_effort(value: Any) -> str:
    return _normalize_reasoning_effort(value) or "-"


def _winner_model_names(model_summaries: list[dict[str, Any]]) -> tuple[list[str], float | None]:
    scored_rows: list[tuple[str, float]] = []
    for summary in model_summaries:
        model_name = str(summary.get("model_name", "")).strip()
        score = _as_float(summary.get("final_score_avg"))
        if not model_name or score is None:
            continue
        scored_rows.append((model_name, score))

    if not scored_rows:
        return [], None

    top_score = max(score for _, score in scored_rows)
    eps = 1e-9
    winners = [model_name for model_name, score in scored_rows if abs(score - top_score) <= eps]
    return winners, top_score


def _winner_label(winner_models: list[str]) -> str:
    if not winner_models:
        return "-"
    if len(winner_models) == 1:
        return winner_models[0]
    return f"TIE ({len(winner_models)}): " + ", ".join(winner_models)


def _is_page_asset_export_enabled() -> bool:
    raw = (os.getenv("EVAL_EXPORT_PAGE_ASSETS") or "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


def _resolve_playwright_command() -> list[str] | None:
    if shutil.which("playwright"):
        return ["playwright"]
    if shutil.which("npx"):
        return ["npx", "--yes", "playwright"]
    return None


def _page_assets_dir_for_html(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}_assets"


def _export_html_to_pdf(html_output_path: Path, pdf_output_path: Path) -> bool:
    playwright_command = _resolve_playwright_command()
    if not playwright_command:
        return False

    pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *playwright_command,
        "pdf",
        html_output_path.resolve().as_uri(),
        str(pdf_output_path),
        "--paper-format",
        "A4",
        "--wait-for-timeout",
        "1200",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return pdf_output_path.exists()


def _export_page_assets(html_output_path: Path) -> None:
    if not _is_page_asset_export_enabled():
        return

    if not _resolve_playwright_command():
        return

    assets_dir = _page_assets_dir_for_html(html_output_path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    pdf_target = assets_dir / f"{html_output_path.stem}.pdf"

    rendered_any = _export_html_to_pdf(html_output_path, pdf_target)

    playwright_command = _resolve_playwright_command()
    if playwright_command:
        png_target = assets_dir / f"{html_output_path.stem}.png"
        page_uri = html_output_path.resolve().as_uri()
        command = [
            *playwright_command,
            "screenshot",
            page_uri,
            str(png_target),
            "--full-page",
            "--wait-for-timeout",
            "1200",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Rendering is best-effort and should not block report generation.
            pass
        else:
            if png_target.exists():
                rendered_any = True

    if not rendered_any:
        try:
            assets_dir.rmdir()
        except OSError:
            pass


def _infer_provider_label(summary: dict[str, Any]) -> str:
    provider = str(summary.get("provider", "")).strip().lower()
    if provider:
        return provider

    model_name = str(summary.get("model_name", "")).strip().lower()
    if "/" in model_name:
        prefix = model_name.split("/", 1)[0].strip()
        if prefix:
            return prefix
    return "unknown"


def _sorted_model_summaries(results: dict[str, Any]) -> list[dict[str, Any]]:
    model_summaries = list(results.get("model_summaries", []))
    model_summaries.sort(
        key=lambda item: _as_float(item.get("final_score_avg")) or 0.0,
        reverse=True,
    )
    return model_summaries


def _sorted_failed_cases(results: dict[str, Any]) -> list[dict[str, Any]]:
    failed_cases = [
        case
        for case in results.get("case_results", [])
        if not bool(case.get("passed"))
    ]
    failed_cases.sort(key=lambda item: _as_float(item.get("final_score")) or 0.0)
    return failed_cases


def _case_error_text(case: dict[str, Any]) -> str | None:
    inference = case.get("inference", {})
    if not isinstance(inference, dict):
        return None
    raw = inference.get("error")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _split_failed_cases(results: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failed_cases = _sorted_failed_cases(results)
    quality_failures: list[dict[str, Any]] = []
    execution_errors: list[dict[str, Any]] = []

    for case in failed_cases:
        if _case_error_text(case):
            execution_errors.append(case)
        else:
            quality_failures.append(case)

    return quality_failures, execution_errors


def _derive_day_key(*, started_at: str, run_dir_name: str) -> str:
    if len(started_at) >= 10 and started_at[4:5] == "-" and started_at[7:8] == "-":
        return started_at[:10]
    if len(run_dir_name) >= 8 and run_dir_name[:8].isdigit():
        raw = run_dir_name[:8]
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return "unknown"


def _load_history_entries(reports_root: Path) -> list[dict[str, Any]]:
    if not reports_root.exists():
        return []

    entries: list[dict[str, Any]] = []
    for child in sorted(reports_root.iterdir()):
        if not child.is_dir():
            continue

        results_path = child / "results.json"
        if not results_path.exists():
            continue

        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue

        model_summaries = _sorted_model_summaries(payload)
        normalized_summaries: list[dict[str, Any]] = []
        for summary in model_summaries:
            model_name = str(summary.get("model_name", "")).strip()
            if not model_name:
                continue
            normalized_summaries.append(
                {
                    "model_name": model_name,
                    "provider": _infer_provider_label(summary),
                    "reasoning_effort": _normalize_reasoning_effort(summary.get("reasoning_effort")),
                    "final_score_avg": _as_float(summary.get("final_score_avg")),
                    "pass_rate": _as_float(summary.get("pass_rate")),
                    "deterministic_score_avg": _as_float(summary.get("deterministic_score_avg")),
                    "judge_score_avg": _as_float(summary.get("judge_score_avg")),
                    "latency_p50_s": _as_float(summary.get("latency_p50_s")),
                    "ttft_p50_s": _as_float(summary.get("ttft_p50_s")),
                    "tokens_per_s_p50": _as_float(summary.get("tokens_per_s_p50")),
                    "error_count": _as_float(summary.get("error_count")),
                }
            )

        winner_models, winner_score = _winner_model_names(normalized_summaries)
        winner_model = winner_models[0] if winner_models else "-"
        winner_share = (1.0 / len(winner_models)) if winner_models else 0.0
        winner_tie = len(winner_models) > 1
        winner_label = _winner_label(winner_models)
        started_at = str(payload.get("started_at", ""))
        finished_at = str(payload.get("finished_at", ""))
        run_id = str(payload.get("run_id", "")) or child.name
        day_key = _derive_day_key(started_at=started_at, run_dir_name=child.name)

        entries.append(
            {
                "run_dir_name": child.name,
                "run_id": run_id,
                "day_key": day_key,
                "started_at": started_at,
                "finished_at": finished_at,
                "winner_model": winner_model,
                "winner_models": winner_models,
                "winner_label": winner_label,
                "winner_score": winner_score,
                "winner_share": winner_share,
                "winner_tie": winner_tie,
                "case_count": len(payload.get("case_results", [])),
                "model_count": len(normalized_summaries),
                "warning_count": len(payload.get("warnings", [])),
                "model_summaries": normalized_summaries,
            }
        )

    entries.sort(
        key=lambda item: (
            str(item.get("started_at", "")),
            str(item.get("run_dir_name", "")),
        ),
        reverse=True,
    )
    return entries


def _build_history_metrics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    report_count = len(entries)
    winner_total_score = sum((entry.get("winner_score") or 0.0) for entry in entries)
    mean_winner_score = (winner_total_score / report_count) if report_count else None
    tie_report_count = sum(1 for entry in entries if bool(entry.get("winner_tie")))

    winner_by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        winner_models = [str(item) for item in entry.get("winner_models", []) if str(item).strip()]
        winner_share = _as_float(entry.get("winner_share")) or 0.0
        winner_score = _as_float(entry.get("winner_score"))
        if winner_score is None or not winner_models or winner_share <= 0:
            continue
        is_outright = len(winner_models) == 1
        for model_name in winner_models:
            state = winner_by_model.setdefault(
                model_name,
                {"model_name": model_name, "win_credits": 0.0, "outright_wins": 0, "total_score": 0.0},
            )
            state["win_credits"] += winner_share
            if is_outright:
                state["outright_wins"] += 1
            state["total_score"] += (winner_score * winner_share)

    winner_rows: list[dict[str, Any]] = []
    for row in winner_by_model.values():
        wins = float(row["win_credits"])
        outright_wins = int(row["outright_wins"])
        tie_win_credits = max(0.0, wins - float(outright_wins))
        total_score = float(row["total_score"])
        winner_rows.append(
            {
                "model_name": row["model_name"],
                "wins": wins,
                "outright_wins": outright_wins,
                "tie_win_credits": tie_win_credits,
                "total_score": total_score,
                "mean_when_winner": (total_score / wins) if wins else None,
                "mean_per_report": (total_score / report_count) if report_count else None,
            }
        )
    winner_rows.sort(
        key=lambda item: (
            _as_float(item.get("mean_per_report")) or 0.0,
            _as_float(item.get("wins")) or 0.0,
            int(item.get("outright_wins") or 0),
        ),
        reverse=True,
    )

    historical_by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        winner_models = {str(item) for item in entry.get("winner_models", []) if str(item).strip()}
        winner_share = _as_float(entry.get("winner_share")) or 0.0
        is_outright = len(winner_models) == 1
        model_summaries = entry.get("model_summaries", [])
        for summary in model_summaries:
            model_name = str(summary.get("model_name", "-"))
            state = historical_by_model.setdefault(
                model_name,
                {
                    "model_name": model_name,
                    "reports_seen": 0,
                    "win_credits": 0.0,
                    "outright_wins": 0,
                    "reasoning_efforts": set(),
                    "final_scores": [],
                    "pass_rates": [],
                    "deterministic_scores": [],
                    "judge_scores": [],
                    "latency_p50_values": [],
                    "ttft_p50_values": [],
                    "tokens_per_s_p50_values": [],
                    "error_total": 0,
                },
            )
            state["reports_seen"] += 1
            if model_name in winner_models and winner_share > 0:
                state["win_credits"] += winner_share
                if is_outright:
                    state["outright_wins"] += 1

            final_score = _as_float(summary.get("final_score_avg"))
            pass_rate = _as_float(summary.get("pass_rate"))
            deterministic_score = _as_float(summary.get("deterministic_score_avg"))
            judge_score = _as_float(summary.get("judge_score_avg"))
            latency_p50 = _as_float(summary.get("latency_p50_s"))
            ttft_p50 = _as_float(summary.get("ttft_p50_s"))
            tokens_per_s_p50 = _as_float(summary.get("tokens_per_s_p50"))
            error_count = _as_float(summary.get("error_count"))
            reasoning_effort = _normalize_reasoning_effort(summary.get("reasoning_effort"))

            if final_score is not None:
                state["final_scores"].append(final_score)
            if pass_rate is not None:
                state["pass_rates"].append(pass_rate)
            if deterministic_score is not None:
                state["deterministic_scores"].append(deterministic_score)
            if judge_score is not None:
                state["judge_scores"].append(judge_score)
            if latency_p50 is not None:
                state["latency_p50_values"].append(latency_p50)
            if ttft_p50 is not None:
                state["ttft_p50_values"].append(ttft_p50)
            if tokens_per_s_p50 is not None:
                state["tokens_per_s_p50_values"].append(tokens_per_s_p50)
            if error_count is not None:
                state["error_total"] += int(error_count)
            if reasoning_effort is not None:
                state["reasoning_efforts"].add(reasoning_effort)

    historical_leaderboard_rows: list[dict[str, Any]] = []
    for state in historical_by_model.values():
        reports_seen = int(state["reports_seen"])
        wins = float(state["win_credits"])
        outright_wins = int(state["outright_wins"])
        tie_win_credits = max(0.0, wins - float(outright_wins))
        final_scores = [float(item) for item in state["final_scores"]]
        reasoning_efforts = sorted(str(item) for item in state["reasoning_efforts"])
        total_final_score = sum(final_scores)
        historical_leaderboard_rows.append(
            {
                "model_name": state["model_name"],
                "reports_seen": reports_seen,
                "wins": wins,
                "outright_wins": outright_wins,
                "tie_win_credits": tie_win_credits,
                "reasoning_effort": (
                    reasoning_efforts[0]
                    if len(reasoning_efforts) == 1
                    else ", ".join(reasoning_efforts)
                    if reasoning_efforts
                    else None
                ),
                "win_rate_global": (wins / report_count) if report_count else None,
                "win_rate_seen": (wins / reports_seen) if reports_seen else None,
                "total_final_score": total_final_score,
                "mean_final_score": _mean(final_scores),
                "median_final_score": _median(final_scores),
                "mean_pass_rate": _mean([float(item) for item in state["pass_rates"]]),
                "mean_deterministic_score": _mean([float(item) for item in state["deterministic_scores"]]),
                "mean_judge_score": _mean([float(item) for item in state["judge_scores"]]),
                "mean_latency_p50_s": _mean([float(item) for item in state["latency_p50_values"]]),
                "mean_ttft_p50_s": _mean([float(item) for item in state["ttft_p50_values"]]),
                "mean_tokens_per_s_p50": _mean([float(item) for item in state["tokens_per_s_p50_values"]]),
                "mean_error_count": ((state["error_total"] / reports_seen) if reports_seen else None),
            }
        )
    historical_leaderboard_rows.sort(
        key=lambda item: (
            _as_float(item.get("mean_final_score")) or 0.0,
            _as_float(item.get("win_rate_global")) or 0.0,
            _as_float(item.get("wins")) or 0.0,
            int(item.get("outright_wins") or 0),
        ),
        reverse=True,
    )

    historical_by_provider: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model_summaries = list(entry.get("model_summaries", []))
        provider_rows_in_run: dict[str, list[dict[str, Any]]] = {}

        winner_models = {str(item) for item in entry.get("winner_models", []) if str(item).strip()}
        winner_share = _as_float(entry.get("winner_share")) or 0.0
        is_outright = len(winner_models) == 1
        winner_provider_credits: dict[str, float] = {}
        model_provider_map: dict[str, str] = {}

        for summary in model_summaries:
            model_name = str(summary.get("model_name", "-"))
            provider = _infer_provider_label(summary)
            model_provider_map[model_name] = provider

            provider_rows_in_run.setdefault(provider, []).append(summary)

            state = historical_by_provider.setdefault(
                provider,
                {
                    "provider": provider,
                    "reports_seen": 0,
                    "model_rows": 0,
                    "win_credits": 0.0,
                    "outright_wins": 0,
                    "final_scores": [],
                    "pass_rates": [],
                    "deterministic_scores": [],
                    "judge_scores": [],
                    "latency_p50_values": [],
                    "ttft_p50_values": [],
                    "tokens_per_s_p50_values": [],
                    "best_final_per_report_values": [],
                    "error_total": 0,
                },
            )
            state["model_rows"] += 1

            final_score = _as_float(summary.get("final_score_avg"))
            pass_rate = _as_float(summary.get("pass_rate"))
            deterministic_score = _as_float(summary.get("deterministic_score_avg"))
            judge_score = _as_float(summary.get("judge_score_avg"))
            latency_p50 = _as_float(summary.get("latency_p50_s"))
            ttft_p50 = _as_float(summary.get("ttft_p50_s"))
            tokens_per_s_p50 = _as_float(summary.get("tokens_per_s_p50"))
            error_count = _as_float(summary.get("error_count"))

            if final_score is not None:
                state["final_scores"].append(final_score)
            if pass_rate is not None:
                state["pass_rates"].append(pass_rate)
            if deterministic_score is not None:
                state["deterministic_scores"].append(deterministic_score)
            if judge_score is not None:
                state["judge_scores"].append(judge_score)
            if latency_p50 is not None:
                state["latency_p50_values"].append(latency_p50)
            if ttft_p50 is not None:
                state["ttft_p50_values"].append(ttft_p50)
            if tokens_per_s_p50 is not None:
                state["tokens_per_s_p50_values"].append(tokens_per_s_p50)
            if error_count is not None:
                state["error_total"] += int(error_count)

        for provider, rows in provider_rows_in_run.items():
            state = historical_by_provider[provider]
            state["reports_seen"] += 1

            best_final_candidates = [
                value
                for value in (_as_float(item.get("final_score_avg")) for item in rows)
                if value is not None
            ]
            if best_final_candidates:
                state["best_final_per_report_values"].append(max(best_final_candidates))

        for winner_model in winner_models:
            provider = model_provider_map.get(winner_model, "")
            if not provider and "/" in winner_model:
                provider = winner_model.split("/", 1)[0].strip().lower()
            if not provider:
                continue
            winner_provider_credits[provider] = winner_provider_credits.get(provider, 0.0) + winner_share

        for provider, credit in winner_provider_credits.items():
            if provider not in historical_by_provider or credit <= 0:
                continue
            historical_by_provider[provider]["win_credits"] += credit
            if is_outright and abs(credit - 1.0) <= 1e-9:
                historical_by_provider[provider]["outright_wins"] += 1

    provider_rows: list[dict[str, Any]] = []
    for state in historical_by_provider.values():
        reports_seen = int(state["reports_seen"])
        wins = float(state["win_credits"])
        outright_wins = int(state["outright_wins"])
        tie_win_credits = max(0.0, wins - float(outright_wins))
        model_rows = int(state["model_rows"])
        provider_rows.append(
            {
                "provider": state["provider"],
                "reports_seen": reports_seen,
                "models_seen_total": model_rows,
                "wins": wins,
                "outright_wins": outright_wins,
                "tie_win_credits": tie_win_credits,
                "win_rate_global": (wins / report_count) if report_count else None,
                "win_rate_seen": (wins / reports_seen) if reports_seen else None,
                "mean_final_score": _mean([float(item) for item in state["final_scores"]]),
                "mean_best_final_per_report": _mean(
                    [float(item) for item in state["best_final_per_report_values"]]
                ),
                "mean_pass_rate": _mean([float(item) for item in state["pass_rates"]]),
                "mean_deterministic_score": _mean(
                    [float(item) for item in state["deterministic_scores"]]
                ),
                "mean_judge_score": _mean([float(item) for item in state["judge_scores"]]),
                "mean_latency_p50_s": _mean([float(item) for item in state["latency_p50_values"]]),
                "mean_ttft_p50_s": _mean([float(item) for item in state["ttft_p50_values"]]),
                "mean_tokens_per_s_p50": _mean(
                    [float(item) for item in state["tokens_per_s_p50_values"]]
                ),
                "mean_error_count_per_model": ((state["error_total"] / model_rows) if model_rows else None),
            }
        )
    provider_rows.sort(
        key=lambda item: (
            _as_float(item.get("mean_best_final_per_report")) or 0.0,
            _as_float(item.get("mean_final_score")) or 0.0,
            _as_float(item.get("win_rate_global")) or 0.0,
            _as_float(item.get("wins")) or 0.0,
            int(item.get("outright_wins") or 0),
        ),
        reverse=True,
    )

    grouped_days: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped_days.setdefault(str(entry.get("day_key", "unknown")), []).append(entry)
    day_rows = sorted(grouped_days.items(), key=lambda item: item[0], reverse=True)

    return {
        "report_count": report_count,
        "day_count": len(day_rows),
        "tie_report_count": tie_report_count,
        "winner_total_score": winner_total_score,
        "mean_winner_score": mean_winner_score,
        "winner_model_count": len(winner_rows),
        "winner_rows": winner_rows,
        "historical_leaderboard_rows": historical_leaderboard_rows,
        "provider_row_count": len(provider_rows),
        "provider_rows": provider_rows,
        "day_rows": day_rows,
    }


def _history_entry_sort_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (
        str(entry.get("started_at", "")),
        str(entry.get("run_dir_name", "")),
    )


def _mean_metric_for_entry(entry: dict[str, Any], metric_key: str) -> float | None:
    model_summaries = entry.get("model_summaries", [])
    if not isinstance(model_summaries, list):
        return None

    values = [
        value
        for value in (_as_float(summary.get(metric_key)) for summary in model_summaries if isinstance(summary, dict))
        if value is not None
    ]
    return _mean(values)


def _build_run_metric_series(entries: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for entry in sorted(entries, key=_history_entry_sort_key):
        value = _mean_metric_for_entry(entry, metric_key)
        if value is None:
            continue

        started_at = str(entry.get("started_at", "")).strip()
        run_id = str(entry.get("run_id", "")).strip() or str(entry.get("run_dir_name", "")).strip() or "-"
        day_key = str(entry.get("day_key", "")).strip()
        short_label = (
            day_key
            if day_key and day_key != "unknown"
            else started_at[:10]
            if len(started_at) >= 10
            else run_id
        )
        label = started_at or run_id
        series.append(
            {
                "label": label,
                "short_label": short_label,
                "value": float(value),
            }
        )
    return series


def _build_model_metric_series(model_summaries: list[dict[str, Any]], metric_key: str) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for index, summary in enumerate(model_summaries, start=1):
        value = _as_float(summary.get(metric_key))
        if value is None:
            continue
        model_name = str(summary.get("model_name", "-"))
        series.append(
            {
                "label": model_name,
                "short_label": f"#{index}",
                "value": float(value),
            }
        )
    return series


def _render_metric_trend_chart(
    *,
    series: list[dict[str, Any]],
    title: str,
    subtitle: str,
    formatter: Any,
    higher_is_better: bool,
) -> str:
    cleaned_points: list[tuple[str, str, float]] = []
    for point in series:
        value = _as_float(point.get("value"))
        if value is None:
            continue
        label = str(point.get("label", "")).strip() or "-"
        short_label = str(point.get("short_label", "")).strip() or label
        cleaned_points.append((label, short_label, float(value)))

    if not cleaned_points:
        return (
            f'<section class="chart trend-chart"><h3>{html.escape(title)}</h3>'
            f'<p class="muted">{html.escape(subtitle)}</p>'
            '<p class="muted">No chart data available.</p></section>'
        )

    values = [value for _, _, value in cleaned_points]
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value

    bars: list[str] = []
    for index, (label, _, value) in enumerate(cleaned_points):
        normalized = 1.0 if span <= 1e-12 else (value - min_value) / span
        visual_ratio = normalized if higher_is_better else (1.0 - normalized)
        height = 16.0 + (visual_ratio * 84.0)
        tooltip = f"{label}: {formatter(value)}"
        class_name = "trend-bar latest" if index == len(cleaned_points) - 1 else "trend-bar"
        bars.append(
            f'<span class="{class_name}" style="height:{height:.2f}%;" title="{html.escape(tooltip)}"></span>'
        )

    first_label = cleaned_points[0][1]
    latest_label = cleaned_points[-1][1]
    latest_value = cleaned_points[-1][2]
    delta_text = "-"
    if len(cleaned_points) > 1:
        delta = latest_value - cleaned_points[0][2]
        delta_formatted = _format_num(delta)
        if delta > 0:
            delta_formatted = f"+{delta_formatted}"
        delta_text = delta_formatted

    direction_text = "higher is better" if higher_is_better else "lower is better"
    return "".join(
        [
            f'<section class="chart trend-chart"><h3>{html.escape(title)}</h3>',
            f'<p class="muted">{html.escape(subtitle)}</p>',
            f'<div class="trend-bars">{"".join(bars)}</div>',
            '<div class="trend-meta">',
            f'<span>Latest: <strong>{html.escape(formatter(latest_value))}</strong></span>',
            f'<span>Range: {html.escape(formatter(min_value))} to {html.escape(formatter(max_value))}</span>',
            f"<span>Delta: {html.escape(delta_text)}</span>",
            "</div>",
            (
                f'<p class="muted trend-window">Window: {html.escape(first_label)} to '
                f'{html.escape(latest_label)} ({len(cleaned_points)} points, {direction_text}).</p>'
            ),
            "</section>",
        ]
    )


def _render_sortable_tables_script_lines() -> list[str]:
    return [
        "<script>",
        "(function () {",
        "  function normalize(raw) {",
        "    if (raw === null || raw === undefined) return '';",
        "    return String(raw).trim();",
        "  }",
        "  function asNumber(raw) {",
        "    var text = normalize(raw).replace(/[%,$]/g, '');",
        "    if (!text || text === '-') return null;",
        "    var value = Number(text);",
        "    return Number.isFinite(value) ? value : null;",
        "  }",
        "  function asDate(raw) {",
        "    var text = normalize(raw);",
        "    if (!text || text === '-') return null;",
        "    var value = Date.parse(text);",
        "    return Number.isFinite(value) ? value : null;",
        "  }",
        "  function cellValue(row, index, type) {",
        "    var cell = row.children[index];",
        "    if (!cell) return null;",
        "    var raw = cell.getAttribute('data-sort-value');",
        "    if (raw === null) raw = cell.textContent;",
        "    if (type === 'number' || type === 'percent') return asNumber(raw);",
        "    if (type === 'date') return asDate(raw);",
        "    var text = normalize(raw);",
        "    return text ? text.toLowerCase() : null;",
        "  }",
        "  function setHeaderState(headers, active, order) {",
        "    headers.forEach(function (header) {",
        "      header.classList.remove('sorted-asc', 'sorted-desc');",
        "      header.dataset.sortOrder = 'none';",
        "      header.setAttribute('aria-sort', 'none');",
        "    });",
        "    active.classList.add(order === 'asc' ? 'sorted-asc' : 'sorted-desc');",
        "    active.dataset.sortOrder = order;",
        "    active.setAttribute('aria-sort', order === 'asc' ? 'ascending' : 'descending');",
        "  }",
        "  function updateRankColumn(tbody) {",
        "    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));",
        "    rows.forEach(function (row, index) {",
        "      var rankCell = row.querySelector('td.col-rank');",
        "      if (!rankCell) return;",
        "      var rank = index + 1;",
        "      rankCell.textContent = String(rank);",
        "      rankCell.setAttribute('data-sort-value', String(rank));",
        "    });",
        "  }",
        "  function sortTable(table, headers, index, order) {",
        "    var tbody = table.querySelector('tbody');",
        "    if (!tbody) return;",
        "    var header = headers[index];",
        "    var type = header.dataset.sortType || 'text';",
        "    var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr')).map(function (row, i) {",
        "      return { row: row, originalIndex: i };",
        "    });",
        "    rows.sort(function (a, b) {",
        "      var aValue = cellValue(a.row, index, type);",
        "      var bValue = cellValue(b.row, index, type);",
        "      if (aValue === null && bValue === null) return a.originalIndex - b.originalIndex;",
        "      if (aValue === null) return 1;",
        "      if (bValue === null) return -1;",
        "      if (aValue < bValue) return order === 'asc' ? -1 : 1;",
        "      if (aValue > bValue) return order === 'asc' ? 1 : -1;",
        "      return a.originalIndex - b.originalIndex;",
        "    });",
        "    rows.forEach(function (entry) { tbody.appendChild(entry.row); });",
        "    updateRankColumn(tbody);",
        "    setHeaderState(headers, header, order);",
        "  }",
        "  function toggleSort(table, headers, index) {",
        "    var header = headers[index];",
        "    var nextOrder = header.dataset.sortOrder === 'asc' ? 'desc' : 'asc';",
        "    sortTable(table, headers, index, nextOrder);",
        "  }",
        "  document.querySelectorAll('table.sortable').forEach(function (table) {",
        "    var headers = Array.prototype.slice.call(table.querySelectorAll('thead th'));",
        "    headers.forEach(function (header, index) {",
        "      if (!header.dataset.sortType) return;",
        "      header.classList.add('sortable-header');",
        "      header.dataset.sortOrder = 'none';",
        "      header.setAttribute('aria-sort', 'none');",
        "      header.setAttribute('role', 'button');",
        "      header.setAttribute('tabindex', '0');",
        "      header.setAttribute('title', 'Sort rows by this column');",
        "      header.addEventListener('click', function () {",
        "        toggleSort(table, headers, index);",
        "      });",
        "      header.addEventListener('keydown', function (event) {",
        "        if (event.key === 'Enter' || event.key === ' ') {",
        "          event.preventDefault();",
        "          toggleSort(table, headers, index);",
        "        }",
        "      });",
        "    });",
        "    var defaultColumn = Number(table.dataset.defaultSortColumn || '-1');",
        "    var defaultOrder = table.dataset.defaultSortOrder === 'asc' ? 'asc' : 'desc';",
        "    if (Number.isInteger(defaultColumn) && defaultColumn >= 0 && defaultColumn < headers.length) {",
        "      if (headers[defaultColumn].dataset.sortType) {",
        "        sortTable(table, headers, defaultColumn, defaultOrder);",
        "      }",
        "    }",
        "  });",
        "})();",
        "</script>",
    ]


def render_history_html(reports_root: Path) -> str:
    entries = _load_history_entries(reports_root)
    metrics = _build_history_metrics(entries)
    report_count = int(metrics["report_count"])
    day_count = int(metrics["day_count"])
    tie_report_count = int(metrics["tie_report_count"])
    winner_total_score = float(metrics["winner_total_score"])
    mean_winner_score = _as_float(metrics["mean_winner_score"])
    winner_model_count = int(metrics["winner_model_count"])
    winner_rows = list(metrics["winner_rows"])
    historical_leaderboard_rows = list(metrics["historical_leaderboard_rows"])
    provider_row_count = int(metrics["provider_row_count"])
    provider_rows = list(metrics["provider_rows"])
    day_rows = list(metrics["day_rows"])
    quality_trend_series = _build_run_metric_series(entries, "final_score_avg")
    latency_trend_series = _build_run_metric_series(entries, "latency_p50_s")
    tps_trend_series = _build_run_metric_series(entries, "tokens_per_s_p50")

    lines: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>LLM Eval Reports History</title>",
        "<style>",
        ":root { color-scheme: light; }",
        "body { margin: 0; font-family: 'Avenir Next', Avenir, 'Segoe UI', sans-serif; background: #f5f7fb; color: #1f2937; }",
        ".page { max-width: 1200px; margin: 0 auto; padding: 24px; }",
        ".hero { background: linear-gradient(120deg, #0f172a, #1d4ed8); color: #f8fafc; border-radius: 18px; padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25); }",
        ".hero h1 { margin: 0; font-size: 2rem; }",
        ".hero p { margin: 10px 0 0; opacity: 0.94; }",
        ".meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }",
        ".meta-card { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 10px 12px; }",
        ".section { margin-top: 22px; background: #ffffff; border-radius: 14px; padding: 18px; box-shadow: 0 5px 16px rgba(2, 6, 23, 0.08); }",
        "h2 { margin: 0 0 12px 0; font-size: 1.25rem; color: #111827; }",
        "h3 { margin: 0 0 10px 0; font-size: 1.05rem; }",
        ".muted { color: #6b7280; margin: 0 0 8px 0; }",
        ".table-wrap { overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 12px; background: #ffffff; }",
        "table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 0.93rem; }",
        "th, td { text-align: left; border-bottom: 1px solid #e5e7eb; padding: 10px 8px; vertical-align: top; white-space: nowrap; }",
        "th { background: #f8fafc; color: #111827; }",
        ".mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        ".sticky-col { position: sticky; left: 0; background: #ffffff; z-index: 2; }",
        "th.sticky-col { background: #f8fafc; z-index: 3; }",
        ".col-rank { min-width: 56px; }",
        ".col-model { min-width: 320px; }",
        ".chart-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }",
        ".chart { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fbfdff; }",
        ".chart-row { margin: 8px 0; }",
        ".chart-row-top { display: flex; justify-content: space-between; gap: 12px; font-size: 0.9rem; }",
        ".bar { margin-top: 6px; height: 10px; border-radius: 999px; background: #e5e7eb; overflow: hidden; }",
        ".fill { display: block; height: 100%; background: linear-gradient(90deg, #22c55e, #2563eb); }",
        ".trend-bars { display: flex; gap: 3px; align-items: flex-end; height: 126px; padding: 8px; border: 1px solid #dbeafe; border-radius: 10px; background: linear-gradient(180deg, #f8fbff, #eff6ff); overflow: hidden; }",
        ".trend-bar { flex: 1; min-width: 2px; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #60a5fa, #2563eb); opacity: 0.78; }",
        ".trend-bar.latest { background: linear-gradient(180deg, #34d399, #059669); opacity: 1; }",
        ".trend-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 0.82rem; color: #334155; }",
        ".trend-window { margin-top: 6px; font-size: 0.8rem; }",
        ".sortable th { user-select: none; }",
        ".sortable th.sortable-header { cursor: pointer; position: relative; padding-right: 18px; }",
        ".sortable th.sortable-header::after { content: '↕'; position: absolute; right: 6px; color: #94a3b8; font-size: 0.8rem; }",
        ".sortable th.sorted-asc::after { content: '↑'; color: #2563eb; }",
        ".sortable th.sorted-desc::after { content: '↓'; color: #2563eb; }",
        ".day-block { margin-top: 16px; }",
        ".day-title { margin: 0 0 8px 0; font-size: 1rem; color: #1f2937; }",
        "a { color: #1d4ed8; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "@media (max-width: 640px) { .page { padding: 12px; } .hero h1 { font-size: 1.55rem; } table { font-size: 0.84rem; } }",
        "</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="hero">',
        "<h1>LLM Eval Reports History</h1>",
        "<p>Inspect benchmark runs day by day and compare top-score trends.</p>",
        '<div class="meta-grid">',
        f'<div class="meta-card"><strong>Report Count</strong><div class="mono">{report_count}</div></div>',
        f'<div class="meta-card"><strong>Day Count</strong><div class="mono">{day_count}</div></div>',
        f'<div class="meta-card"><strong>Tied Reports</strong><div class="mono">{tie_report_count}</div></div>',
        (
            '<div class="meta-card"><strong>Mean Winner Score</strong>'
            f'<div class="mono">{_format_num(mean_winner_score)} '
            f'(total={_format_num(winner_total_score)} / reports={report_count})</div></div>'
        ),
        f'<div class="meta-card"><strong>Distinct Winner Models</strong><div class="mono">{winner_model_count}</div></div>',
        f'<div class="meta-card"><strong>Distinct Providers</strong><div class="mono">{provider_row_count}</div></div>',
        "</div>",
        "</section>",
        '<section class="section">',
        "<h2>Historical Leaderboard</h2>",
        '<p class="muted">Aggregated model metrics across all reports (not only wins).</p>',
        '<p class="muted">`Win Credits` split ties evenly across tied top models in each run.</p>',
        '<p class="muted">Click any column header to sort ascending/descending.</p>',
    ]

    if historical_leaderboard_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                '<table class="sortable" data-default-sort-column="6" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th class="sticky-col col-model" data-sort-type="text">Model</th>'
                '<th data-sort-type="text">Reasoning</th>'
                '<th data-sort-type="number">Reports</th><th data-sort-type="number">Win Credits</th><th data-sort-type="percent">Global Win Rate (credits/report_count)</th>'
                '<th data-sort-type="number">Mean Final</th><th data-sort-type="number">Median Final</th><th data-sort-type="percent">Mean Pass Rate</th><th data-sort-type="number">Mean Deterministic</th>'
                '<th data-sort-type="number">Mean Judge</th><th data-sort-type="number">Mean Latency p50</th><th data-sort-type="number">Mean TTFT p50</th>'
                '<th data-sort-type="number">Mean Tokens/s p50</th><th data-sort-type="number">Mean Errors/Report</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(historical_leaderboard_rows, start=1):
            reports_seen = int(row.get("reports_seen") or 0)
            wins = _as_float(row.get("wins"))
            win_rate_global = _as_float(row.get("win_rate_global"))
            mean_final_score = _as_float(row.get("mean_final_score"))
            median_final_score = _as_float(row.get("median_final_score"))
            mean_pass_rate = _as_float(row.get("mean_pass_rate"))
            mean_deterministic_score = _as_float(row.get("mean_deterministic_score"))
            mean_judge_score = _as_float(row.get("mean_judge_score"))
            mean_latency_p50_s = _as_float(row.get("mean_latency_p50_s"))
            mean_ttft_p50_s = _as_float(row.get("mean_ttft_p50_s"))
            mean_tokens_per_s_p50 = _as_float(row.get("mean_tokens_per_s_p50"))
            mean_error_count = _as_float(row.get("mean_error_count"))
            reasoning_effort = _format_reasoning_effort(row.get("reasoning_effort"))
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono sticky-col col-model">{html.escape(str(row.get("model_name", "-")))}</td>'
                f'<td class="mono" data-sort-value="{html.escape(reasoning_effort)}">{html.escape(reasoning_effort)}</td>'
                f'<td data-sort-value="{reports_seen}">{reports_seen}</td>'
                f'<td data-sort-value="{_sort_value(wins)}">{_format_count(wins)}</td>'
                f'<td data-sort-value="{_sort_value(win_rate_global)}">{_format_pct(win_rate_global)}</td>'
                f'<td data-sort-value="{_sort_value(mean_final_score)}">{_format_num(mean_final_score)}</td>'
                f'<td data-sort-value="{_sort_value(median_final_score)}">{_format_num(median_final_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_pass_rate)}">{_format_pct(mean_pass_rate)}</td>'
                f'<td data-sort-value="{_sort_value(mean_deterministic_score)}">{_format_num(mean_deterministic_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_judge_score)}">{_format_num(mean_judge_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_latency_p50_s)}">{_format_num(mean_latency_p50_s)}</td>'
                f'<td data-sort-value="{_sort_value(mean_ttft_p50_s)}">{_format_num(mean_ttft_p50_s)}</td>'
                f'<td data-sort-value="{_sort_value(mean_tokens_per_s_p50)}">{_format_num(mean_tokens_per_s_p50)}</td>'
                f'<td data-sort-value="{_sort_value(mean_error_count)}">{_format_num(mean_error_count)}</td>'
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])
    else:
        lines.append('<p class="muted">No completed report data found.</p>')
    lines.append("</section>")

    lines.extend(
        [
            '<section class="section">',
            "<h2>Provider Cross-Compare</h2>",
            '<p class="muted">Cross-report comparison aggregated by provider.</p>',
            '<p class="muted">`Win Credits` split ties evenly across tied top models in each run.</p>',
            '<p class="muted">Click any column header to sort ascending/descending.</p>',
        ]
    )

    if provider_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                '<table class="sortable" data-default-sort-column="7" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th class="sticky-col col-model" data-sort-type="text">Provider</th>'
                '<th data-sort-type="number">Reports</th><th data-sort-type="number">Model Rows</th><th data-sort-type="number">Win Credits</th><th data-sort-type="percent">Global Win Rate (credits/report_count)</th>'
                '<th data-sort-type="number">Mean Final</th><th data-sort-type="number">Mean Best Final/Report</th><th data-sort-type="percent">Mean Pass Rate</th>'
                '<th data-sort-type="number">Mean Deterministic</th><th data-sort-type="number">Mean Judge</th><th data-sort-type="number">Mean Latency p50</th>'
                '<th data-sort-type="number">Mean TTFT p50</th><th data-sort-type="number">Mean Tokens/s p50</th><th data-sort-type="number">Mean Errors/Model Row</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(provider_rows, start=1):
            reports_seen = int(row.get("reports_seen") or 0)
            models_seen_total = int(row.get("models_seen_total") or 0)
            wins = _as_float(row.get("wins"))
            win_rate_global = _as_float(row.get("win_rate_global"))
            mean_final_score = _as_float(row.get("mean_final_score"))
            mean_best_final_per_report = _as_float(row.get("mean_best_final_per_report"))
            mean_pass_rate = _as_float(row.get("mean_pass_rate"))
            mean_deterministic_score = _as_float(row.get("mean_deterministic_score"))
            mean_judge_score = _as_float(row.get("mean_judge_score"))
            mean_latency_p50_s = _as_float(row.get("mean_latency_p50_s"))
            mean_ttft_p50_s = _as_float(row.get("mean_ttft_p50_s"))
            mean_tokens_per_s_p50 = _as_float(row.get("mean_tokens_per_s_p50"))
            mean_error_count_per_model = _as_float(row.get("mean_error_count_per_model"))

            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono sticky-col col-model">{html.escape(str(row.get("provider", "-")))}</td>'
                f'<td data-sort-value="{reports_seen}">{reports_seen}</td>'
                f'<td data-sort-value="{models_seen_total}">{models_seen_total}</td>'
                f'<td data-sort-value="{_sort_value(wins)}">{_format_count(wins)}</td>'
                f'<td data-sort-value="{_sort_value(win_rate_global)}">{_format_pct(win_rate_global)}</td>'
                f'<td data-sort-value="{_sort_value(mean_final_score)}">{_format_num(mean_final_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_best_final_per_report)}">{_format_num(mean_best_final_per_report)}</td>'
                f'<td data-sort-value="{_sort_value(mean_pass_rate)}">{_format_pct(mean_pass_rate)}</td>'
                f'<td data-sort-value="{_sort_value(mean_deterministic_score)}">{_format_num(mean_deterministic_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_judge_score)}">{_format_num(mean_judge_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_latency_p50_s)}">{_format_num(mean_latency_p50_s)}</td>'
                f'<td data-sort-value="{_sort_value(mean_ttft_p50_s)}">{_format_num(mean_ttft_p50_s)}</td>'
                f'<td data-sort-value="{_sort_value(mean_tokens_per_s_p50)}">{_format_num(mean_tokens_per_s_p50)}</td>'
                f'<td data-sort-value="{_sort_value(mean_error_count_per_model)}">{_format_num(mean_error_count_per_model)}</td>'
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])
    else:
        lines.append('<p class="muted">No provider comparison data found.</p>')
    lines.append("</section>")

    quality_trend_chart = _render_metric_trend_chart(
        series=quality_trend_series,
        title="Mean Quality Trend",
        subtitle="Per-run mean of model final scores (quality).",
        formatter=_format_num,
        higher_is_better=True,
    )
    latency_trend_chart = _render_metric_trend_chart(
        series=latency_trend_series,
        title="Mean Latency Trend (p50 seconds)",
        subtitle="Per-run mean latency across models.",
        formatter=_format_num,
        higher_is_better=False,
    )
    tps_trend_chart = _render_metric_trend_chart(
        series=tps_trend_series,
        title="Mean TPS Trend (tokens/s p50)",
        subtitle="Per-run mean throughput across models.",
        formatter=_format_num,
        higher_is_better=True,
    )
    lines.extend(
        [
            '<section class="section">',
            "<h2>Leaderboard Trend Graphs</h2>",
            '<p class="muted">Leaderboard-style run history for mean quality, latency, and TPS.</p>',
            '<div class="chart-grid">',
            quality_trend_chart,
            latency_trend_chart,
            tps_trend_chart,
            "</div>",
            "</section>",
        ]
    )

    chart_rows = historical_leaderboard_rows[:12]
    score_chart = _render_metric_chart_rows(
        model_summaries=chart_rows,
        metric_key="mean_final_score",
        title="Historical Mean Final Score",
        subtitle="Top models by cross-run average final score.",
        formatter=_format_num,
    )
    win_rate_chart = _render_metric_chart_rows(
        model_summaries=chart_rows,
        metric_key="win_rate_global",
        title="Global Win Rate",
        subtitle="win_credits/report_count across all reports.",
        formatter=_format_pct,
    )
    pass_rate_chart = _render_metric_chart_rows(
        model_summaries=chart_rows,
        metric_key="mean_pass_rate",
        title="Historical Mean Pass Rate",
        subtitle="Average pass-rate share across reports.",
        formatter=_format_pct,
    )
    lines.extend(
        [
            '<section class="section">',
            "<h2>Leaderboard Charts</h2>",
            '<p class="muted">Visual comparison for the top 12 models by historical mean final score.</p>',
            '<div class="chart-grid">',
            score_chart,
            win_rate_chart,
            pass_rate_chart,
            "</div>",
            "</section>",
        ]
    )

    lines.extend(
        [
            '<section class="section">',
            "<h2>Winner Model Means</h2>",
            '<p class="muted">`mean_per_report = credited_winner_score_total/report_count` across all reports.</p>',
            '<p class="muted">`Total Winner Score` and `Win Credits` split ties evenly across tied top models.</p>',
            '<p class="muted">Click any column header to sort ascending/descending.</p>',
        ]
    )

    if winner_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                '<table class="sortable" data-default-sort-column="4" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th class="sticky-col col-model" data-sort-type="text">Winner Model</th><th data-sort-type="number">Win Credits</th><th data-sort-type="number">Total Winner Score</th>'
                '<th data-sort-type="number">Mean Per Report (total/report_count)</th><th data-sort-type="number">Mean When Winner (total/wins)</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(winner_rows, start=1):
            wins = _as_float(row.get("wins"))
            total_score = _as_float(row.get("total_score"))
            mean_per_report = _as_float(row.get("mean_per_report"))
            mean_when_winner = _as_float(row.get("mean_when_winner"))
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono sticky-col col-model">{html.escape(str(row.get("model_name", "-")))}</td>'
                f'<td data-sort-value="{_sort_value(wins)}">{_format_count(wins)}</td>'
                f'<td data-sort-value="{_sort_value(total_score)}">{_format_num(total_score)}</td>'
                f'<td data-sort-value="{_sort_value(mean_per_report)}">{_format_num(mean_per_report)}</td>'
                f'<td data-sort-value="{_sort_value(mean_when_winner)}">{_format_num(mean_when_winner)}</td>'
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])
    else:
        lines.append('<p class="muted">No completed report data found.</p>')
    lines.append("</section>")

    lines.extend(
        [
            '<section class="section">',
            "<h2>Day-by-Day Runs</h2>",
            '<p class="muted">Each run links to its detailed `leaderboard.html`.</p>',
            '<p class="muted">Click any column header to sort ascending/descending.</p>',
        ]
    )

    if not day_rows:
        lines.append('<p class="muted">No run directories found under this reports root.</p>')
    else:
        for day_key, rows in day_rows:
            day_total = sum((_as_float(item.get("winner_score")) or 0.0) for item in rows)
            day_mean = (day_total / len(rows)) if rows else None
            lines.extend(
                [
                    '<div class="day-block">',
                    f'<h3 class="day-title">{html.escape(day_key)} '
                    f'<span class="muted">({len(rows)} runs, mean winner={_format_num(day_mean)})</span></h3>',
                    '<div class="table-wrap">',
                    '<table class="sortable" data-default-sort-column="1" data-default-sort-order="desc">',
                    "<thead><tr>"
                    '<th class="sticky-col col-model" data-sort-type="text">Run</th><th data-sort-type="date">Started</th><th data-sort-type="date">Finished</th><th data-sort-type="text">Winner</th><th data-sort-type="number">Winner Score</th>'
                    '<th data-sort-type="number">Models</th><th data-sort-type="number">Cases</th><th data-sort-type="number">Warnings</th><th data-sort-type="text">Open</th>'
                    "</tr></thead>",
                    "<tbody>",
                ]
            )
            for entry in rows:
                run_dir_name = str(entry.get("run_dir_name", "-"))
                run_id = str(entry.get("run_id", "-"))
                started_at = str(entry.get("started_at", "-"))
                finished_at = str(entry.get("finished_at", "-"))
                winner_model = str(entry.get("winner_label", entry.get("winner_model", "-")))
                winner_score = _format_num(_as_float(entry.get("winner_score")))
                model_count = int(entry.get("model_count") or 0)
                case_count = int(entry.get("case_count") or 0)
                warning_count = int(entry.get("warning_count") or 0)
                link = f"{html.escape(run_dir_name)}/leaderboard.html"
                lines.append(
                    "<tr>"
                    f'<td class="mono sticky-col col-model">{html.escape(run_id)}</td>'
                    f'<td class="mono" data-sort-value="{html.escape(started_at)}">{html.escape(started_at)}</td>'
                    f'<td class="mono" data-sort-value="{html.escape(finished_at)}">{html.escape(finished_at)}</td>'
                    f'<td class="mono">{html.escape(winner_model)}</td>'
                    f'<td data-sort-value="{_sort_value(_as_float(entry.get("winner_score")))}">{winner_score}</td>'
                    f'<td data-sort-value="{model_count}">{model_count}</td>'
                    f'<td data-sort-value="{case_count}">{case_count}</td>'
                    f'<td data-sort-value="{warning_count}">{warning_count}</td>'
                    f'<td><a href="{link}">Open</a></td>'
                    "</tr>"
                )
            lines.extend(["</tbody>", "</table>", "</div>", "</div>"])
    lines.extend(
        [
            "</section>",
        ]
    )
    lines.extend(_render_sortable_tables_script_lines())
    lines.extend(["</main>", "</body>", "</html>"])
    return "\n".join(lines)


def render_leaderboard_markdown(results: dict[str, Any]) -> str:
    run_id = results.get("run_id", "unknown")
    started_at = results.get("started_at", "unknown")
    finished_at = results.get("finished_at", "unknown")
    datasets = results.get("datasets", [])

    model_summaries = _sorted_model_summaries(results)

    lines: list[str] = []
    lines.append(f"# LLM Eval Leaderboard ({run_id})")
    lines.append("")
    lines.append(f"- Started: `{started_at}`")
    lines.append(f"- Finished: `{finished_at}`")
    lines.append(f"- Datasets: `{', '.join(datasets) if datasets else '-'}`")
    lines.append("")

    if not model_summaries:
        lines.append("No model results were generated.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Rank | Model | Reasoning | Final | Deterministic | Judge | Pass Rate | "
        "TTFT p50/p95 (s) | Latency p50/p95 (s) | Tokens/s p50/p95 | Errors |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---|---|---|---:|")

    for index, summary in enumerate(model_summaries, start=1):
        lines.append(
            "| "
            f"{index}"
            " | "
            f"`{summary.get('model_name', '-')}`"
            " | "
            f"{_format_reasoning_effort(summary.get('reasoning_effort'))}"
            " | "
            f"{_format_num(summary.get('final_score_avg'))}"
            " | "
            f"{_format_num(summary.get('deterministic_score_avg'))}"
            " | "
            f"{_format_num(summary.get('judge_score_avg'))}"
            " | "
            f"{_format_pct(summary.get('pass_rate'))}"
            " | "
            f"{_format_num(summary.get('ttft_p50_s'))}/{_format_num(summary.get('ttft_p95_s'))}"
            " | "
            f"{_format_num(summary.get('latency_p50_s'))}/{_format_num(summary.get('latency_p95_s'))}"
            " | "
            f"{_format_num(summary.get('tokens_per_s_p50'))}/{_format_num(summary.get('tokens_per_s_p95'))}"
            " | "
            f"{summary.get('error_count', 0)}"
            " |"
        )

    quality_failures, execution_errors = _split_failed_cases(results)

    if quality_failures:
        lines.append("")
        lines.append("## Notable Failed Cases")
        lines.append("")
        for case in quality_failures[:10]:
            inference = case.get("inference", {})
            error = inference.get("error")
            lines.append(
                "- "
                f"`{case.get('case_id')}` on `{case.get('model_name')}`: "
                f"score={_format_num(case.get('final_score'))}"
                + (f" error={error}" if error else "")
            )

    if execution_errors:
        lines.append("")
        lines.append("## Execution Errors")
        lines.append("")
        lines.append(
            "_These failures include transport/provider issues (e.g., rate limits or empty responses)._"
        )
        lines.append("")
        for case in execution_errors[:10]:
            lines.append(
                "- "
                f"`{case.get('case_id')}` on `{case.get('model_name')}`: "
                f"score={_format_num(case.get('final_score'))} "
                f"error={_case_error_text(case)}"
            )

    warnings = results.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def _render_pass_fail_pie_svg(pass_rate: float | None) -> str:
    normalized_pass_rate = _clamp(pass_rate if pass_rate is not None else 0.0, 0.0, 1.0)
    radius = 44.0
    center = 56.0
    circumference = 2.0 * math.pi * radius
    pass_dash = circumference * normalized_pass_rate
    pass_label = _format_pct(normalized_pass_rate)

    return (
        '<svg class="pie-chart" viewBox="0 0 112 112" role="img" aria-label="Pass fail pie chart">'
        f'<circle class="pie-fail" cx="{center}" cy="{center}" r="{radius}"></circle>'
        f'<circle class="pie-pass" cx="{center}" cy="{center}" r="{radius}" '
        f'stroke-dasharray="{pass_dash:.3f} {circumference:.3f}"></circle>'
        f'<text class="pie-label" x="{center}" y="{center}">{html.escape(pass_label)}</text>'
        "</svg>"
    )


def _render_metric_chart_rows(
    *,
    model_summaries: list[dict[str, Any]],
    metric_key: str,
    title: str,
    subtitle: str,
    formatter: Any,
) -> str:
    rows: list[tuple[str, float]] = []
    for summary in model_summaries:
        value = _as_float(summary.get(metric_key))
        if value is None:
            continue
        rows.append((str(summary.get("model_name", "-")), value))

    if not rows:
        return (
            f'<section class="chart"><h3>{html.escape(title)}</h3>'
            f'<p class="muted">{html.escape(subtitle)}</p>'
            '<p class="muted">No chart data available.</p></section>'
        )

    max_value = max(value for _, value in rows)
    safe_denominator = max_value if max_value > 0 else 1.0

    line_items: list[str] = [
        f'<section class="chart"><h3>{html.escape(title)}</h3>',
        f'<p class="muted">{html.escape(subtitle)}</p>',
    ]
    for model_name, value in rows:
        width = _clamp((value / safe_denominator) * 100.0, 0.0, 100.0)
        line_items.append('<div class="chart-row">')
        line_items.append(
            '<div class="chart-row-top">'
            f"<span>{html.escape(model_name)}</span>"
            f"<span>{html.escape(formatter(value))}</span>"
            "</div>"
        )
        line_items.append(
            f'<div class="bar"><span class="fill" style="width:{width:.2f}%"></span></div>'
        )
        line_items.append("</div>")
    line_items.append("</section>")

    return "".join(line_items)


def render_leaderboard_html(results: dict[str, Any]) -> str:
    run_id = str(results.get("run_id", "unknown"))
    started_at = str(results.get("started_at", "unknown"))
    finished_at = str(results.get("finished_at", "unknown"))
    datasets = [str(item) for item in results.get("datasets", [])]
    dataset_text = ", ".join(datasets) if datasets else "-"

    model_summaries = _sorted_model_summaries(results)
    quality_failures, execution_errors = _split_failed_cases(results)
    warnings = [str(item) for item in results.get("warnings", [])]

    top_model = model_summaries[0] if model_summaries else None
    top_model_name = str(top_model.get("model_name", "-")) if top_model else "-"
    top_model_score = _format_num(_as_float(top_model.get("final_score_avg")) if top_model else None)
    quality_rank_series = _build_model_metric_series(model_summaries, "final_score_avg")
    latency_rank_series = _build_model_metric_series(model_summaries, "latency_p50_s")
    tps_rank_series = _build_model_metric_series(model_summaries, "tokens_per_s_p50")

    lines: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>LLM Eval Leaderboard {html.escape(run_id)}</title>",
        "<style>",
        ":root { color-scheme: light; }",
        "body { margin: 0; font-family: 'Avenir Next', Avenir, 'Segoe UI', sans-serif; background: #f5f7fb; color: #1f2937; }",
        ".page { max-width: 1200px; margin: 0 auto; padding: 24px; }",
        ".hero { background: linear-gradient(120deg, #0f172a, #1d4ed8); color: #f8fafc; border-radius: 18px; padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25); }",
        ".hero h1 { margin: 0; font-size: 2rem; }",
        ".hero p { margin: 10px 0 0; opacity: 0.94; }",
        ".meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 16px; }",
        ".meta-card { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; padding: 10px 12px; }",
        ".section { margin-top: 22px; background: #ffffff; border-radius: 14px; padding: 18px; box-shadow: 0 5px 16px rgba(2, 6, 23, 0.08); }",
        "h2 { margin: 0 0 12px 0; font-size: 1.25rem; color: #111827; }",
        "h3 { margin: 0 0 8px 0; font-size: 1.02rem; }",
        ".muted { color: #6b7280; margin: 0 0 8px 0; }",
        "table { width: 100%; border-collapse: collapse; font-size: 0.93rem; }",
        "th, td { text-align: left; border-bottom: 1px solid #e5e7eb; padding: 10px 8px; vertical-align: top; }",
        "th { background: #f8fafc; color: #111827; }",
        ".sortable th { user-select: none; }",
        ".sortable th.sortable-header { cursor: pointer; position: relative; padding-right: 18px; }",
        ".sortable th.sortable-header::after { content: '↕'; position: absolute; right: 6px; color: #94a3b8; font-size: 0.8rem; }",
        ".sortable th.sorted-asc::after { content: '↑'; color: #2563eb; }",
        ".sortable th.sorted-desc::after { content: '↓'; color: #2563eb; }",
        ".mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        ".pie-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }",
        ".pie-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; background: #fbfdff; }",
        ".pie-card h3 { margin-bottom: 10px; }",
        ".pie-wrap { display: flex; justify-content: center; margin-bottom: 8px; }",
        ".pie-chart { width: 112px; height: 112px; }",
        ".pie-chart circle { fill: none; stroke-width: 18; transform-origin: 50% 50%; }",
        ".pie-fail { stroke: #fecaca; }",
        ".pie-pass { stroke: #22c55e; transform: rotate(-90deg); stroke-linecap: round; }",
        ".pie-label { font-size: 11px; font-weight: 700; text-anchor: middle; dominant-baseline: middle; fill: #1f2937; }",
        ".chart-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }",
        ".chart { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fbfdff; }",
        ".chart-row { margin: 8px 0; }",
        ".chart-row-top { display: flex; justify-content: space-between; gap: 12px; font-size: 0.9rem; }",
        ".bar { margin-top: 6px; height: 10px; border-radius: 999px; background: #e5e7eb; overflow: hidden; }",
        ".fill { display: block; height: 100%; background: linear-gradient(90deg, #22c55e, #2563eb); }",
        ".trend-bars { display: flex; gap: 3px; align-items: flex-end; height: 126px; padding: 8px; border: 1px solid #dbeafe; border-radius: 10px; background: linear-gradient(180deg, #f8fbff, #eff6ff); overflow: hidden; }",
        ".trend-bar { flex: 1; min-width: 2px; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #60a5fa, #2563eb); opacity: 0.78; }",
        ".trend-bar.latest { background: linear-gradient(180deg, #34d399, #059669); opacity: 1; }",
        ".trend-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 0.82rem; color: #334155; }",
        ".trend-window { margin-top: 6px; font-size: 0.8rem; }",
        ".explain-list { margin: 0; padding-left: 20px; }",
        ".warning-list, .failed-list { margin: 0; padding-left: 20px; }",
        "@media (max-width: 640px) { .page { padding: 12px; } .hero h1 { font-size: 1.55rem; } table { font-size: 0.84rem; } }",
        "</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="hero">',
        f"<h1>LLM Eval Leaderboard ({html.escape(run_id)})</h1>",
        "<p>Detailed HTML output with leaderboard, pies, charts, and explanation.</p>",
        '<div class="meta-grid">',
        f'<div class="meta-card"><strong>Started</strong><div class="mono">{html.escape(started_at)}</div></div>',
        f'<div class="meta-card"><strong>Finished</strong><div class="mono">{html.escape(finished_at)}</div></div>',
        f'<div class="meta-card"><strong>Datasets</strong><div class="mono">{html.escape(dataset_text)}</div></div>',
        f'<div class="meta-card"><strong>Top Model</strong><div class="mono">{html.escape(top_model_name)} ({html.escape(top_model_score)})</div></div>',
        "</div>",
        "</section>",
        '<section class="section">',
        "<h2>Leaderboard</h2>",
    ]

    if not model_summaries:
        lines.extend(
            [
                '<p class="muted">No model results were generated.</p>',
                "</section>",
            ]
        )
    else:
        lines.extend(
            [
                '<p class="muted">Click any column header to sort ascending/descending.</p>',
                '<table class="sortable" data-default-sort-column="3" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th data-sort-type="text">Model</th>'
                '<th data-sort-type="text">Reasoning</th><th data-sort-type="number">Final</th><th data-sort-type="number">Deterministic</th><th data-sort-type="number">Judge</th>'
                '<th data-sort-type="percent">Pass Rate</th><th data-sort-type="number">TTFT p50/p95</th><th data-sort-type="number">Latency p50/p95</th>'
                '<th data-sort-type="number">Tokens/s p50/p95</th><th data-sort-type="number">Errors</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )

        for rank, summary in enumerate(model_summaries, start=1):
            final_score = _as_float(summary.get("final_score_avg"))
            deterministic_score = _as_float(summary.get("deterministic_score_avg"))
            judge_score = _as_float(summary.get("judge_score_avg"))
            pass_rate = _as_float(summary.get("pass_rate"))
            ttft_p50 = _as_float(summary.get("ttft_p50_s"))
            ttft_p95 = _as_float(summary.get("ttft_p95_s"))
            latency_p50 = _as_float(summary.get("latency_p50_s"))
            latency_p95 = _as_float(summary.get("latency_p95_s"))
            tokens_per_s_p50 = _as_float(summary.get("tokens_per_s_p50"))
            tokens_per_s_p95 = _as_float(summary.get("tokens_per_s_p95"))
            error_count = int(_as_float(summary.get("error_count")) or 0)
            reasoning_effort = _format_reasoning_effort(summary.get("reasoning_effort"))
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono">{html.escape(str(summary.get("model_name", "-")))}</td>'
                f'<td class="mono" data-sort-value="{html.escape(reasoning_effort)}">{html.escape(reasoning_effort)}</td>'
                f'<td data-sort-value="{_sort_value(final_score)}">{_format_num(final_score)}</td>'
                f'<td data-sort-value="{_sort_value(deterministic_score)}">{_format_num(deterministic_score)}</td>'
                f'<td data-sort-value="{_sort_value(judge_score)}">{_format_num(judge_score)}</td>'
                f'<td data-sort-value="{_sort_value(pass_rate)}">{_format_pct(pass_rate)}</td>'
                f'<td data-sort-value="{_sort_value(ttft_p50)}">{_format_num(ttft_p50)}/{_format_num(ttft_p95)}</td>'
                f'<td data-sort-value="{_sort_value(latency_p50)}">{_format_num(latency_p50)}/{_format_num(latency_p95)}</td>'
                f'<td data-sort-value="{_sort_value(tokens_per_s_p50)}">{_format_num(tokens_per_s_p50)}/{_format_num(tokens_per_s_p95)}</td>'
                f'<td data-sort-value="{error_count}">{error_count}</td>'
                "</tr>"
            )

        lines.extend(["</tbody>", "</table>", "</section>"])

        lines.extend(
            [
                '<section class="section">',
                "<h2>Pie Charts</h2>",
                '<p class="muted">Each pie shows pass vs fail share across evaluated cases per model.</p>',
                '<div class="pie-grid">',
            ]
        )
        for summary in model_summaries:
            model_name = str(summary.get("model_name", "-"))
            pass_rate = _as_float(summary.get("pass_rate"))
            normalized_pass = _clamp(pass_rate if pass_rate is not None else 0.0, 0.0, 1.0)
            fail_rate = 1.0 - normalized_pass
            lines.extend(
                [
                    '<article class="pie-card">',
                    f"<h3>{html.escape(model_name)}</h3>",
                    '<div class="pie-wrap">',
                    _render_pass_fail_pie_svg(pass_rate),
                    "</div>",
                    f"<p class=\"muted\">Pass: {_format_pct(normalized_pass)} | Fail: {_format_pct(fail_rate)}</p>",
                    "</article>",
                ]
            )
        lines.extend(["</div>", "</section>"])

        score_chart = _render_metric_chart_rows(
            model_summaries=model_summaries,
            metric_key="final_score_avg",
            title="Final Score Chart",
            subtitle="Higher is better.",
            formatter=_format_num,
        )
        pass_chart = _render_metric_chart_rows(
            model_summaries=model_summaries,
            metric_key="pass_rate",
            title="Pass Rate Chart",
            subtitle="Share of cases above threshold.",
            formatter=_format_pct,
        )
        latency_chart = _render_metric_chart_rows(
            model_summaries=model_summaries,
            metric_key="latency_p50_s",
            title="Latency Chart (p50 seconds)",
            subtitle="Lower is better. Bar length is relative to the slowest model.",
            formatter=_format_num,
        )
        lines.extend(
            [
                '<section class="section">',
                "<h2>Performance Charts</h2>",
                '<div class="chart-grid">',
                score_chart,
                pass_chart,
                latency_chart,
                "</div>",
                "</section>",
            ]
        )

        quality_trend_chart = _render_metric_trend_chart(
            series=quality_rank_series,
            title="Mean Quality (Final Score Avg)",
            subtitle="Models ordered by leaderboard rank for this run.",
            formatter=_format_num,
            higher_is_better=True,
        )
        latency_trend_chart = _render_metric_trend_chart(
            series=latency_rank_series,
            title="Mean Latency (p50 seconds)",
            subtitle="Models ordered by leaderboard rank for this run.",
            formatter=_format_num,
            higher_is_better=False,
        )
        tps_trend_chart = _render_metric_trend_chart(
            series=tps_rank_series,
            title="Mean TPS (tokens/s p50)",
            subtitle="Models ordered by leaderboard rank for this run.",
            formatter=_format_num,
            higher_is_better=True,
        )
        lines.extend(
            [
                '<section class="section">',
                "<h2>Leaderboard Trend Graphs</h2>",
                '<p class="muted">Leaderboard-style bars for mean quality, latency, and TPS.</p>',
                '<div class="chart-grid">',
                quality_trend_chart,
                latency_trend_chart,
                tps_trend_chart,
                "</div>",
                "</section>",
            ]
        )

    lines.extend(
        [
            '<section class="section">',
            "<h2>Explanation</h2>",
            '<ul class="explain-list">',
            "<li><strong>Final</strong>: weighted combination of deterministic checks and judge score.</li>",
            "<li><strong>Deterministic</strong>: exact/regex/include/json checks against expected output.</li>",
            "<li><strong>Judge</strong>: optional rubric-based LLM scoring when enabled for a case.</li>",
            "<li><strong>Reasoning</strong>: resolved reasoning effort applied to each model request (if set).</li>",
            "<li><strong>Pass rate</strong>: fraction of cases with final score at or above the suite threshold.</li>",
            "<li><strong>TTFT / Latency / Tokens per second</strong>: response performance distribution metrics.</li>",
            "</ul>",
            "</section>",
        ]
    )

    if quality_failures:
        lines.extend(
            [
                '<section class="section">',
                "<h2>Notable Failed Cases</h2>",
                '<ul class="failed-list">',
            ]
        )
        for case in quality_failures[:10]:
            inference = case.get("inference", {})
            error = inference.get("error")
            details = (
                f"{html.escape(str(case.get('case_id', '-')))} on "
                f"{html.escape(str(case.get('model_name', '-')))}: "
                f"score={html.escape(_format_num(_as_float(case.get('final_score'))))}"
            )
            if error:
                details += f" | error={html.escape(str(error))}"
            lines.append(f"<li><span class=\"mono\">{details}</span></li>")
        lines.extend(["</ul>", "</section>"])

    if execution_errors:
        lines.extend(
            [
                '<section class="section">',
                "<h2>Execution Errors</h2>",
                '<p class="muted">These failures include transport/provider issues (e.g., rate limits or empty responses).</p>',
                '<ul class="failed-list">',
            ]
        )
        for case in execution_errors[:10]:
            details = (
                f"{html.escape(str(case.get('case_id', '-')))} on "
                f"{html.escape(str(case.get('model_name', '-')))}: "
                f"score={html.escape(_format_num(_as_float(case.get('final_score'))))}"
                f" | error={html.escape(str(_case_error_text(case) or '-'))}"
            )
            lines.append(f"<li><span class=\"mono\">{details}</span></li>")
        lines.extend(["</ul>", "</section>"])

    if warnings:
        lines.extend(
            [
                '<section class="section">',
                "<h2>Warnings</h2>",
                '<ul class="warning-list">',
            ]
        )
        for warning in warnings:
            lines.append(f"<li>{html.escape(warning)}</li>")
        lines.extend(["</ul>", "</section>"])

    lines.extend(_render_sortable_tables_script_lines())
    lines.extend(["</main>", "</body>", "</html>"])
    return "\n".join(lines)


def _is_same_run_entry(
    entry: dict[str, Any],
    *,
    results: dict[str, Any],
    current_run_dir_name: str | None,
) -> bool:
    entry_run_id = str(entry.get("run_id", "")).strip()
    target_run_id = str(results.get("run_id", "")).strip()
    if entry_run_id and target_run_id and entry_run_id == target_run_id:
        return True

    if current_run_dir_name:
        run_dir_name = str(entry.get("run_dir_name", "")).strip()
        if run_dir_name and run_dir_name == current_run_dir_name:
            return True

    entry_started_at = str(entry.get("started_at", "")).strip()
    target_started_at = str(results.get("started_at", "")).strip()
    if entry_started_at and target_started_at and entry_started_at == target_started_at:
        entry_finished_at = str(entry.get("finished_at", "")).strip()
        target_finished_at = str(results.get("finished_at", "")).strip()
        if not target_finished_at or entry_finished_at == target_finished_at:
            return True

    return False


def _build_detailed_history_context(
    *,
    results: dict[str, Any],
    reports_root: Path | None,
    current_run_dir_name: str | None,
    max_prior_runs: int | None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "reports_root": str(reports_root) if reports_root is not None else None,
        "prior_entries": [],
        "prior_rows": [],
        "prior_run_count": 0,
        "prior_score_count": 0,
        "models_with_prior_count": 0,
        "mean_winner_score": None,
    }
    if reports_root is None:
        return context

    entries = _load_history_entries(reports_root)
    if not entries:
        return context

    prior_entries = [
        entry
        for entry in entries
        if not _is_same_run_entry(
            entry,
            results=results,
            current_run_dir_name=current_run_dir_name,
        )
    ]

    if max_prior_runs is not None:
        max_runs = max(0, int(max_prior_runs))
        if max_runs == 0:
            prior_entries = []
        elif len(prior_entries) > max_runs:
            prior_entries = prior_entries[:max_runs]

    context["prior_entries"] = prior_entries
    context["prior_run_count"] = len(prior_entries)
    if not prior_entries:
        return context

    prior_scores_by_model: dict[str, list[float]] = {}
    for entry in prior_entries:
        for summary in entry.get("model_summaries", []):
            model_name = str(summary.get("model_name", "")).strip()
            score = _as_float(summary.get("final_score_avg"))
            if not model_name or score is None:
                continue
            prior_scores_by_model.setdefault(model_name, []).append(score)

    winner_scores = [
        score
        for score in (_as_float(item.get("winner_score")) for item in prior_entries)
        if score is not None
    ]
    context["mean_winner_score"] = _mean(winner_scores)
    context["prior_score_count"] = sum(len(values) for values in prior_scores_by_model.values())

    prior_rows: list[dict[str, Any]] = []
    models_with_prior_count = 0
    for summary in _sorted_model_summaries(results):
        model_name = str(summary.get("model_name", "")).strip()
        if not model_name:
            continue
        current_final = _as_float(summary.get("final_score_avg"))
        prior_scores = [float(item) for item in prior_scores_by_model.get(model_name, [])]
        prior_samples = len(prior_scores)
        if prior_samples > 0:
            models_with_prior_count += 1
        prior_mean = _mean(prior_scores)
        prior_stddev = _stddev(prior_scores)
        prior_ci95 = (
            (1.96 * prior_stddev / math.sqrt(prior_samples))
            if prior_stddev is not None and prior_samples > 1
            else None
        )

        prior_rows.append(
            {
                "model_name": model_name,
                "current_final_score": current_final,
                "prior_samples": prior_samples,
                "prior_mean_score": prior_mean,
                "prior_stddev_score": prior_stddev,
                "prior_ci95_score": prior_ci95,
                "prior_median_score": _median(prior_scores),
                "prior_min_score": min(prior_scores) if prior_scores else None,
                "prior_max_score": max(prior_scores) if prior_scores else None,
                "latest_prior_score": prior_scores[0] if prior_scores else None,
                "delta_vs_prior_mean": (
                    (current_final - prior_mean)
                    if current_final is not None and prior_mean is not None
                    else None
                ),
            }
        )

    context["prior_rows"] = prior_rows
    context["models_with_prior_count"] = models_with_prior_count
    return context


def render_detailed_report_html(
    results: dict[str, Any],
    include_raw_output: bool = False,
    *,
    reports_root: Path | None = None,
    current_run_dir_name: str | None = None,
    max_prior_runs: int | None = 20,
) -> str:
    run_id = str(results.get("run_id", "unknown"))
    started_at = str(results.get("started_at", "unknown"))
    finished_at = str(results.get("finished_at", "unknown"))
    datasets = [str(item) for item in results.get("datasets", [])]
    dataset_text = ", ".join(datasets) if datasets else "-"

    model_summaries = _sorted_model_summaries(results)
    case_results = list(results.get("case_results", []))
    warnings = [str(item) for item in results.get("warnings", [])]
    quality_failures, execution_errors = _split_failed_cases(results)
    winner_models, winner_score = _winner_model_names(model_summaries)
    winner_label = _winner_label(winner_models)

    total_cases = len(case_results)
    pass_count = sum(1 for case in case_results if bool(case.get("passed")))
    fail_count = max(0, total_cases - pass_count)
    pass_rate = (pass_count / total_cases) if total_cases else None
    execution_error_count = len(execution_errors)
    history_context = _build_detailed_history_context(
        results=results,
        reports_root=reports_root,
        current_run_dir_name=current_run_dir_name,
        max_prior_runs=max_prior_runs,
    )
    prior_entries = list(history_context.get("prior_entries", []))
    prior_rows = list(history_context.get("prior_rows", []))
    prior_run_count = int(history_context.get("prior_run_count") or 0)
    prior_score_count = int(history_context.get("prior_score_count") or 0)
    models_with_prior_count = int(history_context.get("models_with_prior_count") or 0)
    mean_prior_winner_score = _as_float(history_context.get("mean_winner_score"))
    reports_root_text = str(history_context.get("reports_root") or "-")
    trend_entries = list(prior_entries)
    trend_entries.append(
        {
            "run_id": run_id,
            "run_dir_name": current_run_dir_name or run_id,
            "started_at": started_at,
            "day_key": _derive_day_key(
                started_at=started_at,
                run_dir_name=current_run_dir_name or run_id,
            ),
            "model_summaries": model_summaries,
        }
    )
    quality_trend_series = _build_run_metric_series(trend_entries, "final_score_avg")
    latency_trend_series = _build_run_metric_series(trend_entries, "latency_p50_s")
    tps_trend_series = _build_run_metric_series(trend_entries, "tokens_per_s_p50")

    model_failure_rows: list[tuple[str, int, int]] = []
    failure_by_model: dict[str, dict[str, int]] = {}
    for case in case_results:
        model_name = str(case.get("model_name", "-"))
        bucket = failure_by_model.setdefault(model_name, {"failed": 0, "errors": 0})
        if not bool(case.get("passed")):
            bucket["failed"] += 1
            if _case_error_text(case):
                bucket["errors"] += 1
    for model_name, counts in failure_by_model.items():
        model_failure_rows.append((model_name, counts["failed"], counts["errors"]))
    model_failure_rows.sort(key=lambda row: (row[1], row[2], row[0]), reverse=True)

    ordered_cases = sorted(
        case_results,
        key=lambda item: (
            bool(item.get("passed")),
            _as_float(item.get("final_score")) if _as_float(item.get("final_score")) is not None else 1.0,
            str(item.get("case_id", "")),
            str(item.get("model_name", "")),
        ),
    )

    lines: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>LLM Eval Detailed Report {html.escape(run_id)}</title>",
        "<style>",
        ":root { color-scheme: light; }",
        "body { margin: 0; font-family: 'Avenir Next', Avenir, 'Segoe UI', sans-serif; background: #f5f7fb; color: #1f2937; }",
        ".page { max-width: 1150px; margin: 0 auto; padding: 22px; }",
        ".hero { background: linear-gradient(120deg, #0f172a, #1d4ed8); color: #f8fafc; border-radius: 16px; padding: 22px; box-shadow: 0 10px 28px rgba(15, 23, 42, 0.2); }",
        ".hero h1 { margin: 0; font-size: 1.9rem; }",
        ".hero p { margin: 8px 0 0; opacity: 0.92; }",
        ".meta-grid { margin-top: 14px; display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }",
        ".meta-card { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); border-radius: 11px; padding: 9px 11px; }",
        ".section { margin-top: 18px; background: #ffffff; border-radius: 12px; border: 1px solid #e5e7eb; padding: 16px; }",
        "h2 { margin: 0 0 10px 0; font-size: 1.2rem; }",
        "h3 { margin: 0 0 8px 0; font-size: 1.02rem; }",
        ".muted { margin: 0 0 8px 0; color: #6b7280; }",
        ".mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }",
        ".kpi-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }",
        ".kpi { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }",
        ".kpi strong { display: block; margin-bottom: 4px; }",
        ".chart-grid { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }",
        ".chart { border: 1px solid #e5e7eb; border-radius: 10px; background: #fbfdff; padding: 10px; }",
        ".trend-bars { display: flex; gap: 3px; align-items: flex-end; height: 124px; padding: 8px; border: 1px solid #dbeafe; border-radius: 10px; background: linear-gradient(180deg, #f8fbff, #eff6ff); overflow: hidden; }",
        ".trend-bar { flex: 1; min-width: 2px; border-radius: 4px 4px 0 0; background: linear-gradient(180deg, #60a5fa, #2563eb); opacity: 0.78; }",
        ".trend-bar.latest { background: linear-gradient(180deg, #34d399, #059669); opacity: 1; }",
        ".trend-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 0.8rem; color: #334155; }",
        ".trend-window { margin-top: 6px; font-size: 0.78rem; }",
        ".table-wrap { overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 10px; }",
        "table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }",
        "th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; vertical-align: top; }",
        "th { background: #f8fafc; }",
        ".case-card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 11px; margin-bottom: 10px; background: #fcfdff; }",
        ".case-meta { margin: 0 0 8px 0; font-size: 0.88rem; color: #374151; }",
        ".case-error { color: #b91c1c; }",
        ".positive { color: #166534; }",
        ".negative { color: #b91c1c; }",
        ".small { font-size: 0.82rem; }",
        ".output { margin: 0; padding: 9px; border-radius: 8px; border: 1px solid #d1d5db; background: #0f172a; color: #e5e7eb; white-space: pre-wrap; word-break: break-word; font-size: 0.82rem; }",
        ".page-break { break-before: page; page-break-before: always; }",
        "@media print { body { background: #fff; } .page { max-width: none; padding: 12mm; } .section { box-shadow: none; } }",
        "</style>",
        "</head>",
        "<body>",
        '<main class="page">',
        '<section class="hero">',
        f"<h1>LLM Eval Detailed Report ({html.escape(run_id)})</h1>",
        "<p>Executive summary plus technical appendix for leadership review.</p>",
        '<div class="meta-grid">',
        f'<div class="meta-card"><strong>Started</strong><div class="mono">{html.escape(started_at)}</div></div>',
        f'<div class="meta-card"><strong>Finished</strong><div class="mono">{html.escape(finished_at)}</div></div>',
        f'<div class="meta-card"><strong>Datasets</strong><div class="mono">{html.escape(dataset_text)}</div></div>',
        f'<div class="meta-card"><strong>Winner</strong><div class="mono">{html.escape(winner_label)} ({html.escape(_format_num(winner_score))})</div></div>',
        "</div>",
        "</section>",
        '<section class="section">',
        "<h2>Executive Summary</h2>",
        "<p class=\"muted\">Primary decision metrics for this benchmark run.</p>",
        '<div class="kpi-grid">',
        f'<div class="kpi"><strong>Model Count</strong><div class="mono">{len(model_summaries)}</div></div>',
        f'<div class="kpi"><strong>Case Count</strong><div class="mono">{total_cases}</div></div>',
        f'<div class="kpi"><strong>Pass Rate</strong><div class="mono">{html.escape(_format_pct(pass_rate))}</div></div>',
        f'<div class="kpi"><strong>Failed Cases</strong><div class="mono">{fail_count}</div></div>',
        f'<div class="kpi"><strong>Execution Errors</strong><div class="mono">{execution_error_count}</div></div>',
        f'<div class="kpi"><strong>Warnings</strong><div class="mono">{len(warnings)}</div></div>',
        "</div>",
        "</section>",
    ]

    lines.extend(
        [
            '<section class="section">',
            "<h2>Historical Reliability</h2>",
            "<p class=\"muted\">Current scores compared against prior runs to reduce single-run noise.</p>",
            '<div class="kpi-grid">',
            f'<div class="kpi"><strong>Prior Runs Used</strong><div class="mono">{prior_run_count}</div></div>',
            f'<div class="kpi"><strong>Prior Model Scores</strong><div class="mono">{prior_score_count}</div></div>',
            f'<div class="kpi"><strong>Models With Prior Data</strong><div class="mono">{models_with_prior_count}/{len(model_summaries)}</div></div>',
            f'<div class="kpi"><strong>Mean Prior Winner Score</strong><div class="mono">{html.escape(_format_num(mean_prior_winner_score))}</div></div>',
            "</div>",
            f'<p class="muted small">Reports root: <span class="mono">{html.escape(reports_root_text)}</span></p>',
        ]
    )

    quality_trend_chart = _render_metric_trend_chart(
        series=quality_trend_series,
        title="Mean Quality Trend",
        subtitle="Mean final score per run (prior runs plus current).",
        formatter=_format_num,
        higher_is_better=True,
    )
    latency_trend_chart = _render_metric_trend_chart(
        series=latency_trend_series,
        title="Mean Latency Trend (p50 seconds)",
        subtitle="Mean latency per run (prior runs plus current).",
        formatter=_format_num,
        higher_is_better=False,
    )
    tps_trend_chart = _render_metric_trend_chart(
        series=tps_trend_series,
        title="Mean TPS Trend (tokens/s p50)",
        subtitle="Mean throughput per run (prior runs plus current).",
        formatter=_format_num,
        higher_is_better=True,
    )
    lines.extend(
        [
            "<h3>Leaderboard Trend Graphs</h3>",
            '<p class="muted small">Trend bars for mean quality, latency, and TPS.</p>',
            '<div class="chart-grid">',
            quality_trend_chart,
            latency_trend_chart,
            tps_trend_chart,
            "</div>",
        ]
    )

    if prior_rows and prior_run_count > 0:
        lines.extend(
            [
                "<h3>Current vs Prior Score Distribution</h3>",
                '<div class="table-wrap">',
                "<table>",
                "<thead><tr>"
                "<th>Rank</th><th>Model</th><th>Current Final</th><th>Prior Samples</th><th>Prior Mean</th>"
                "<th>Prior StdDev</th><th>95% CI</th><th>Prior Median</th><th>Prior Min/Max</th><th>Latest Prior</th>"
                "<th>Delta vs Prior Mean</th>"
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(prior_rows, start=1):
            delta = _as_float(row.get("delta_vs_prior_mean"))
            delta_text = _format_num(delta)
            if delta is not None and delta > 0:
                delta_text = f"+{delta_text}"
            delta_class = (
                "positive"
                if delta is not None and delta > 0
                else "negative"
                if delta is not None and delta < 0
                else ""
            )
            lines.append(
                "<tr>"
                f"<td>{rank}</td>"
                f'<td class="mono">{html.escape(str(row.get("model_name", "-")))}</td>'
                f"<td>{html.escape(_format_num(_as_float(row.get('current_final_score'))))}</td>"
                f"<td>{int(row.get('prior_samples') or 0)}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('prior_mean_score'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('prior_stddev_score'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('prior_ci95_score'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('prior_median_score'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('prior_min_score'))))}/"
                f"{html.escape(_format_num(_as_float(row.get('prior_max_score'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(row.get('latest_prior_score'))))}</td>"
                f'<td class="{delta_class}">{html.escape(delta_text)}</td>'
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])

        lines.extend(
            [
                "<h3>Recent Prior Runs</h3>",
                '<div class="table-wrap">',
                "<table>",
                "<thead><tr>"
                "<th>Run</th><th>Started</th><th>Winner</th><th>Winner Score</th><th>Models</th><th>Cases</th>"
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for entry in prior_entries[:15]:
            lines.append(
                "<tr>"
                f'<td class="mono">{html.escape(str(entry.get("run_id", "-")))}</td>'
                f'<td class="mono">{html.escape(str(entry.get("started_at", "-")))}</td>'
                f'<td class="mono">{html.escape(str(entry.get("winner_label", entry.get("winner_model", "-"))))}</td>'
                f"<td>{html.escape(_format_num(_as_float(entry.get('winner_score'))))}</td>"
                f"<td>{int(entry.get('model_count') or 0)}</td>"
                f"<td>{int(entry.get('case_count') or 0)}</td>"
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])
    else:
        lines.append('<p class="muted">No prior run data found for historical score comparison.</p>')

    lines.extend(
        [
            "</section>",
            '<section class="section">',
            "<h2>Leaderboard</h2>",
            "<p class=\"muted\">Model ranking by final score with quality and performance metrics.</p>",
        ]
    )

    if not model_summaries:
        lines.append('<p class="muted">No model results were generated.</p>')
    else:
        lines.extend(
            [
                '<div class="table-wrap">',
                "<table>",
                "<thead><tr>"
                "<th>Rank</th><th>Model</th><th>Reasoning</th><th>Final</th><th>Deterministic</th><th>Judge</th>"
                "<th>Pass Rate</th><th>TTFT p50/p95</th><th>Latency p50/p95</th><th>Tokens/s p50/p95</th><th>Errors</th>"
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, summary in enumerate(model_summaries, start=1):
            lines.append(
                "<tr>"
                f"<td>{rank}</td>"
                f'<td class="mono">{html.escape(str(summary.get("model_name", "-")))}</td>'
                f'<td class="mono">{html.escape(_format_reasoning_effort(summary.get("reasoning_effort")))}</td>'
                f"<td>{html.escape(_format_num(_as_float(summary.get('final_score_avg'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(summary.get('deterministic_score_avg'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(summary.get('judge_score_avg'))))}</td>"
                f"<td>{html.escape(_format_pct(_as_float(summary.get('pass_rate'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(summary.get('ttft_p50_s'))))}/"
                f"{html.escape(_format_num(_as_float(summary.get('ttft_p95_s'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(summary.get('latency_p50_s'))))}/"
                f"{html.escape(_format_num(_as_float(summary.get('latency_p95_s'))))}</td>"
                f"<td>{html.escape(_format_num(_as_float(summary.get('tokens_per_s_p50'))))}/"
                f"{html.escape(_format_num(_as_float(summary.get('tokens_per_s_p95'))))}</td>"
                f"<td>{html.escape(_format_count(_as_float(summary.get('error_count'))))}</td>"
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])

    lines.extend(
        [
            "</section>",
            '<section class="section page-break">',
            "<h2>Failure Analysis</h2>",
            "<p class=\"muted\">Highest-risk misses and transport/provider errors.</p>",
            "<h3>Model Failure Breakdown</h3>",
        ]
    )

    if model_failure_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                "<table>",
                "<thead><tr><th>Model</th><th>Failed Cases</th><th>Execution Errors</th></tr></thead>",
                "<tbody>",
            ]
        )
        for model_name, failed_count, error_count in model_failure_rows:
            lines.append(
                "<tr>"
                f'<td class="mono">{html.escape(model_name)}</td>'
                f"<td>{failed_count}</td>"
                f"<td>{error_count}</td>"
                "</tr>"
            )
        lines.extend(["</tbody>", "</table>", "</div>"])
    else:
        lines.append('<p class="muted">No failures detected.</p>')

    lines.extend(["<h3>Notable Failed Cases</h3>"])
    if quality_failures:
        lines.extend(["<ul>"])
        for case in quality_failures[:20]:
            lines.append(
                "<li class=\"mono\">"
                f"{html.escape(str(case.get('case_id', '-')))} on "
                f"{html.escape(str(case.get('model_name', '-')))} | "
                f"score={html.escape(_format_num(_as_float(case.get('final_score'))))}"
                "</li>"
            )
        lines.append("</ul>")
    else:
        lines.append('<p class="muted">No quality-only failed cases.</p>')

    lines.extend(["<h3>Execution Errors</h3>"])
    if execution_errors:
        lines.extend(["<ul>"])
        for case in execution_errors[:20]:
            lines.append(
                "<li class=\"mono\">"
                f"{html.escape(str(case.get('case_id', '-')))} on "
                f"{html.escape(str(case.get('model_name', '-')))} | "
                f"score={html.escape(_format_num(_as_float(case.get('final_score'))))} | "
                f"error={html.escape(str(_case_error_text(case) or '-'))}"
                "</li>"
            )
        lines.append("</ul>")
    else:
        lines.append('<p class="muted">No execution errors detected.</p>')
    lines.append("</section>")

    lines.extend(
        [
            '<section class="section page-break">',
            "<h2>Case-Level Appendix</h2>",
            "<p class=\"muted\">Full case outcomes across all evaluated model-case pairs.</p>",
        ]
    )
    if include_raw_output:
        lines.append('<p class="muted">Raw model outputs are included for every case.</p>')
    else:
        lines.append('<p class="muted">Raw model outputs are omitted (enable via `--include-raw-output`).</p>')

    if not ordered_cases:
        lines.append('<p class="muted">No case-level rows available.</p>')
    else:
        for case in ordered_cases:
            inference = case.get("inference", {})
            if not isinstance(inference, dict):
                inference = {}
            judge = case.get("judge")
            judge_flags_text = "-"
            if isinstance(judge, dict):
                raw_flags = judge.get("flags")
                if isinstance(raw_flags, list) and raw_flags:
                    judge_flags_text = ", ".join(str(item) for item in raw_flags)

            lines.extend(
                [
                    '<article class="case-card">',
                    f"<h3>{html.escape(str(case.get('case_id', '-')))} "
                    f"on {html.escape(str(case.get('model_name', '-')))}</h3>",
                    (
                        '<p class="case-meta">'
                        f"Name: {html.escape(str(case.get('case_name', '-')))} | "
                        f"Category: {html.escape(str(case.get('category', '-')))} | "
                        f"Tags: {html.escape(', '.join(str(item) for item in case.get('tags', []) or [])) or '-'} | "
                        f"Final: {html.escape(_format_num(_as_float(case.get('final_score'))))} | "
                        f"Passed: {html.escape(str(bool(case.get('passed'))))} | "
                        f"Judge Flags: {html.escape(judge_flags_text)}"
                        "</p>"
                    ),
                ]
            )

            error_text = _case_error_text(case)
            if error_text:
                lines.append(f'<p class="case-meta case-error mono">Error: {html.escape(error_text)}</p>')

            if include_raw_output:
                lines.append(f'<pre class="output mono">{html.escape(str(inference.get("output_text", "")))}</pre>')

            lines.append("</article>")

    if warnings:
        lines.extend(
            [
                '<section class="section page-break">',
                "<h2>Warnings</h2>",
                "<ul>",
            ]
        )
        for warning in warnings:
            lines.append(f"<li>{html.escape(warning)}</li>")
        lines.extend(["</ul>", "</section>"])

    lines.extend(["</main>", "</body>", "</html>"])
    return "\n".join(lines)


def write_detailed_pdf_report(
    results: dict[str, Any],
    output_path: Path,
    *,
    include_raw_output: bool = False,
    reports_root: Path | None = None,
    current_run_dir_name: str | None = None,
    max_prior_runs: int | None = 20,
) -> bool:
    resolved_reports_root = reports_root
    if resolved_reports_root is None:
        run_dir_candidate = output_path.parent
        if (run_dir_candidate / "results.json").exists():
            resolved_reports_root = run_dir_candidate.parent

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_detailed_report_html(
        results,
        include_raw_output=include_raw_output,
        reports_root=resolved_reports_root,
        current_run_dir_name=current_run_dir_name,
        max_prior_runs=max_prior_runs,
    )

    temp_html_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            prefix="llm_eval_detail_",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(html_content)
            temp_html_path = Path(handle.name)

        if temp_html_path is None:
            return False
        return _export_html_to_pdf(temp_html_path, output_path)
    finally:
        if temp_html_path is not None:
            try:
                temp_html_path.unlink(missing_ok=True)
            except OSError:
                pass


def write_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_leaderboard_markdown(results), encoding="utf-8")


def write_html_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_leaderboard_html(results), encoding="utf-8")
    _export_page_assets(output_path)


def write_reports(results: dict[str, Any], markdown_output_path: Path, html_output_path: Path) -> None:
    write_markdown_report(results, markdown_output_path)
    write_html_report(results, html_output_path)


def write_history_report(reports_root: Path, output_path: Path | None = None) -> Path:
    target = output_path or (reports_root / "history.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_history_html(reports_root), encoding="utf-8")
    _export_page_assets(target)
    return target


def regenerate_reports_from_json(
    results_json_path: Path,
    markdown_output_path: Path | None = None,
    html_output_path: Path | None = None,
) -> tuple[Path, Path]:
    payload = json.loads(results_json_path.read_text(encoding="utf-8"))
    markdown_target = markdown_output_path or (results_json_path.parent / "leaderboard.md")
    html_target = html_output_path or (results_json_path.parent / "leaderboard.html")
    write_reports(payload, markdown_target, html_target)
    return markdown_target, html_target


def regenerate_report_from_json(results_json_path: Path, output_path: Path | None = None) -> Path:
    markdown_target, _ = regenerate_reports_from_json(
        results_json_path,
        markdown_output_path=output_path,
        html_output_path=None,
    )
    return markdown_target
