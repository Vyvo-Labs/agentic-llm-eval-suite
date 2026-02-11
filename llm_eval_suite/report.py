from __future__ import annotations

import html
import json
import math
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


def _sort_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.12f}"


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

        winner = normalized_summaries[0] if normalized_summaries else {}
        winner_model = str(winner.get("model_name", "-")) if winner else "-"
        winner_score = _as_float(winner.get("final_score_avg")) if winner else None
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
                "winner_score": winner_score,
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

    winner_by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model_name = str(entry.get("winner_model", "-"))
        winner_score = _as_float(entry.get("winner_score"))
        if winner_score is None:
            continue
        state = winner_by_model.setdefault(
            model_name,
            {"model_name": model_name, "wins": 0, "total_score": 0.0},
        )
        state["wins"] += 1
        state["total_score"] += winner_score

    winner_rows: list[dict[str, Any]] = []
    for row in winner_by_model.values():
        wins = int(row["wins"])
        total_score = float(row["total_score"])
        winner_rows.append(
            {
                "model_name": row["model_name"],
                "wins": wins,
                "total_score": total_score,
                "mean_when_winner": (total_score / wins) if wins else None,
                "mean_per_report": (total_score / report_count) if report_count else None,
            }
        )
    winner_rows.sort(
        key=lambda item: (_as_float(item.get("mean_per_report")) or 0.0, int(item.get("wins") or 0)),
        reverse=True,
    )

    historical_by_model: dict[str, dict[str, Any]] = {}
    for entry in entries:
        winner_model = str(entry.get("winner_model", "-"))
        model_summaries = entry.get("model_summaries", [])
        for summary in model_summaries:
            model_name = str(summary.get("model_name", "-"))
            state = historical_by_model.setdefault(
                model_name,
                {
                    "model_name": model_name,
                    "reports_seen": 0,
                    "wins": 0,
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
            if model_name == winner_model:
                state["wins"] += 1

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

    historical_leaderboard_rows: list[dict[str, Any]] = []
    for state in historical_by_model.values():
        reports_seen = int(state["reports_seen"])
        wins = int(state["wins"])
        final_scores = [float(item) for item in state["final_scores"]]
        total_final_score = sum(final_scores)
        historical_leaderboard_rows.append(
            {
                "model_name": state["model_name"],
                "reports_seen": reports_seen,
                "wins": wins,
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
            int(item.get("wins") or 0),
        ),
        reverse=True,
    )

    historical_by_provider: dict[str, dict[str, Any]] = {}
    for entry in entries:
        model_summaries = list(entry.get("model_summaries", []))
        provider_rows_in_run: dict[str, list[dict[str, Any]]] = {}

        winner_model = str(entry.get("winner_model", "-"))
        winner_provider = ""

        for summary in model_summaries:
            model_name = str(summary.get("model_name", "-"))
            provider = _infer_provider_label(summary)
            if model_name == winner_model and not winner_provider:
                winner_provider = provider

            provider_rows_in_run.setdefault(provider, []).append(summary)

            state = historical_by_provider.setdefault(
                provider,
                {
                    "provider": provider,
                    "reports_seen": 0,
                    "model_rows": 0,
                    "wins": 0,
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

        if not winner_provider and "/" in winner_model:
            winner_provider = winner_model.split("/", 1)[0].strip().lower()

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

        if winner_provider and winner_provider in historical_by_provider:
            historical_by_provider[winner_provider]["wins"] += 1

    provider_rows: list[dict[str, Any]] = []
    for state in historical_by_provider.values():
        reports_seen = int(state["reports_seen"])
        wins = int(state["wins"])
        model_rows = int(state["model_rows"])
        provider_rows.append(
            {
                "provider": state["provider"],
                "reports_seen": reports_seen,
                "models_seen_total": model_rows,
                "wins": wins,
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
            int(item.get("wins") or 0),
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
        "winner_total_score": winner_total_score,
        "mean_winner_score": mean_winner_score,
        "winner_model_count": len(winner_rows),
        "winner_rows": winner_rows,
        "historical_leaderboard_rows": historical_leaderboard_rows,
        "provider_row_count": len(provider_rows),
        "provider_rows": provider_rows,
        "day_rows": day_rows,
    }


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
    winner_total_score = float(metrics["winner_total_score"])
    mean_winner_score = _as_float(metrics["mean_winner_score"])
    winner_model_count = int(metrics["winner_model_count"])
    winner_rows = list(metrics["winner_rows"])
    historical_leaderboard_rows = list(metrics["historical_leaderboard_rows"])
    provider_row_count = int(metrics["provider_row_count"])
    provider_rows = list(metrics["provider_rows"])
    day_rows = list(metrics["day_rows"])

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
        "<p>Inspect benchmark runs day by day and compare winner trends.</p>",
        '<div class="meta-grid">',
        f'<div class="meta-card"><strong>Report Count</strong><div class="mono">{report_count}</div></div>',
        f'<div class="meta-card"><strong>Day Count</strong><div class="mono">{day_count}</div></div>',
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
        '<p class="muted">Click any column header to sort ascending/descending.</p>',
    ]

    if historical_leaderboard_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                '<table class="sortable" data-default-sort-column="5" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th class="sticky-col col-model" data-sort-type="text">Model</th>'
                '<th data-sort-type="number">Reports</th><th data-sort-type="number">Wins</th><th data-sort-type="percent">Win Rate (wins/report_count)</th>'
                '<th data-sort-type="number">Mean Final</th><th data-sort-type="number">Median Final</th><th data-sort-type="percent">Mean Pass Rate</th><th data-sort-type="number">Mean Deterministic</th>'
                '<th data-sort-type="number">Mean Judge</th><th data-sort-type="number">Mean Latency p50</th><th data-sort-type="number">Mean TTFT p50</th>'
                '<th data-sort-type="number">Mean Tokens/s p50</th><th data-sort-type="number">Mean Errors/Report</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(historical_leaderboard_rows, start=1):
            reports_seen = int(row.get("reports_seen") or 0)
            wins = int(row.get("wins") or 0)
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
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono sticky-col col-model">{html.escape(str(row.get("model_name", "-")))}</td>'
                f'<td data-sort-value="{reports_seen}">{reports_seen}</td>'
                f'<td data-sort-value="{wins}">{wins}</td>'
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
                '<th data-sort-type="number">Reports</th><th data-sort-type="number">Model Rows</th><th data-sort-type="number">Wins</th><th data-sort-type="percent">Win Rate (wins/report_count)</th>'
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
            wins = int(row.get("wins") or 0)
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
                f'<td data-sort-value="{wins}">{wins}</td>'
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
        subtitle="wins/report_count across all reports.",
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
        '<p class="muted">`mean_per_report = winner_total_score/report_count` across all reports.</p>',
        '<p class="muted">Click any column header to sort ascending/descending.</p>',
        ]
    )

    if winner_rows:
        lines.extend(
            [
                '<div class="table-wrap">',
                '<table class="sortable" data-default-sort-column="4" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th class="sticky-col col-model" data-sort-type="text">Winner Model</th><th data-sort-type="number">Wins</th><th data-sort-type="number">Total Winner Score</th>'
                '<th data-sort-type="number">Mean Per Report (total/report_count)</th><th data-sort-type="number">Mean When Winner (total/wins)</th>'
                "</tr></thead>",
                "<tbody>",
            ]
        )
        for rank, row in enumerate(winner_rows, start=1):
            wins = int(row.get("wins", 0))
            total_score = _as_float(row.get("total_score"))
            mean_per_report = _as_float(row.get("mean_per_report"))
            mean_when_winner = _as_float(row.get("mean_when_winner"))
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono sticky-col col-model">{html.escape(str(row.get("model_name", "-")))}</td>'
                f'<td data-sort-value="{wins}">{wins}</td>'
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
                winner_model = str(entry.get("winner_model", "-"))
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
        "| Rank | Model | Final | Deterministic | Judge | Pass Rate | "
        "TTFT p50/p95 (s) | Latency p50/p95 (s) | Tokens/s p50/p95 | Errors |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---|---|---|---:|")

    for index, summary in enumerate(model_summaries, start=1):
        lines.append(
            "| "
            f"{index}"
            " | "
            f"`{summary.get('model_name', '-')}`"
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
                '<table class="sortable" data-default-sort-column="2" data-default-sort-order="desc">',
                "<thead><tr>"
                '<th class="col-rank" data-sort-type="number">Rank</th><th data-sort-type="text">Model</th>'
                '<th data-sort-type="number">Final</th><th data-sort-type="number">Deterministic</th><th data-sort-type="number">Judge</th>'
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
            lines.append(
                "<tr>"
                f'<td class="col-rank" data-sort-value="{rank}">{rank}</td>'
                f'<td class="mono">{html.escape(str(summary.get("model_name", "-")))}</td>'
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

    lines.extend(
        [
            '<section class="section">',
            "<h2>Explanation</h2>",
            '<ul class="explain-list">',
            "<li><strong>Final</strong>: weighted combination of deterministic checks and judge score.</li>",
            "<li><strong>Deterministic</strong>: exact/regex/include/json checks against expected output.</li>",
            "<li><strong>Judge</strong>: optional rubric-based LLM scoring when enabled for a case.</li>",
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


def write_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_leaderboard_markdown(results), encoding="utf-8")


def write_html_report(results: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_leaderboard_html(results), encoding="utf-8")


def write_reports(results: dict[str, Any], markdown_output_path: Path, html_output_path: Path) -> None:
    write_markdown_report(results, markdown_output_path)
    write_html_report(results, html_output_path)


def write_history_report(reports_root: Path, output_path: Path | None = None) -> Path:
    target = output_path or (reports_root / "history.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_history_html(reports_root), encoding="utf-8")
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
