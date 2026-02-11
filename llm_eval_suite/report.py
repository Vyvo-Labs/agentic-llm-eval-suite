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

    failed_cases = _sorted_failed_cases(results)

    if failed_cases:
        lines.append("")
        lines.append("## Notable Failed Cases")
        lines.append("")
        for case in failed_cases[:10]:
            inference = case.get("inference", {})
            error = inference.get("error")
            lines.append(
                "- "
                f"`{case.get('case_id')}` on `{case.get('model_name')}`: "
                f"score={_format_num(case.get('final_score'))}"
                + (f" error={error}" if error else "")
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
    failed_cases = _sorted_failed_cases(results)
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
                "<table>",
                "<thead><tr>"
                "<th>Rank</th><th>Model</th><th>Final</th><th>Deterministic</th><th>Judge</th>"
                "<th>Pass Rate</th><th>TTFT p50/p95</th><th>Latency p50/p95</th>"
                "<th>Tokens/s p50/p95</th><th>Errors</th>"
                "</tr></thead>",
                "<tbody>",
            ]
        )

        for rank, summary in enumerate(model_summaries, start=1):
            lines.append(
                "<tr>"
                f"<td>{rank}</td>"
                f'<td class="mono">{html.escape(str(summary.get("model_name", "-")))}</td>'
                f"<td>{_format_num(_as_float(summary.get('final_score_avg')))}</td>"
                f"<td>{_format_num(_as_float(summary.get('deterministic_score_avg')))}</td>"
                f"<td>{_format_num(_as_float(summary.get('judge_score_avg')))}</td>"
                f"<td>{_format_pct(_as_float(summary.get('pass_rate')))}</td>"
                f"<td>{_format_num(_as_float(summary.get('ttft_p50_s')))}/{_format_num(_as_float(summary.get('ttft_p95_s')))}</td>"
                f"<td>{_format_num(_as_float(summary.get('latency_p50_s')))}/{_format_num(_as_float(summary.get('latency_p95_s')))}</td>"
                f"<td>{_format_num(_as_float(summary.get('tokens_per_s_p50')))}/{_format_num(_as_float(summary.get('tokens_per_s_p95')))}</td>"
                f"<td>{int(_as_float(summary.get('error_count')) or 0)}</td>"
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

    if failed_cases:
        lines.extend(
            [
                '<section class="section">',
                "<h2>Notable Failed Cases</h2>",
                '<ul class="failed-list">',
            ]
        )
        for case in failed_cases[:10]:
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
