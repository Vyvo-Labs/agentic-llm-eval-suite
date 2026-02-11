from __future__ import annotations

import json
import subprocess
from pathlib import Path

from llm_eval_suite.report import (
    regenerate_report_from_json,
    regenerate_reports_from_json,
    render_history_html,
    render_leaderboard_html,
    render_leaderboard_markdown,
    write_history_report,
    write_html_report,
)


def _sample_results() -> dict[str, object]:
    return {
        "run_id": "r1",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:10Z",
        "datasets": ["datasets/single_turn.yaml"],
        "model_summaries": [
            {
                "model_name": "openrouter/qwen",
                "final_score_avg": 0.68,
                "deterministic_score_avg": 0.55,
                "judge_score_avg": 0.76,
                "pass_rate": 0.4,
                "ttft_p50_s": 0.5,
                "ttft_p95_s": 1.1,
                "latency_p50_s": 1.8,
                "latency_p95_s": 2.7,
                "tokens_per_s_p50": 22.0,
                "tokens_per_s_p95": 15.0,
                "error_count": 1,
            },
            {
                "model_name": "openai/gpt-5-mini",
                "final_score_avg": 0.88,
                "deterministic_score_avg": 0.81,
                "judge_score_avg": 0.91,
                "pass_rate": 0.9,
                "ttft_p50_s": 0.2,
                "ttft_p95_s": 0.3,
                "latency_p50_s": 1.0,
                "latency_p95_s": 1.6,
                "tokens_per_s_p50": 45.0,
                "tokens_per_s_p95": 39.0,
                "error_count": 0,
            },
        ],
        "case_results": [
            {
                "case_id": "case-low",
                "model_name": "openrouter/qwen",
                "final_score": 0.1,
                "passed": False,
                "inference": {"error": "TimeoutError"},
            },
            {
                "case_id": "case-mid",
                "model_name": "openrouter/qwen",
                "final_score": 0.4,
                "passed": False,
                "inference": {"error": None},
            },
            {
                "case_id": "case-high",
                "model_name": "openai/gpt-5-mini",
                "final_score": 0.95,
                "passed": True,
                "inference": {"error": None},
            },
        ],
        "warnings": ["Judge endpoint unavailable for one provider"],
    }


def test_render_leaderboard_markdown_detailed_sections() -> None:
    markdown = render_leaderboard_markdown(_sample_results())

    assert "LLM Eval Leaderboard (r1)" in markdown
    assert markdown.index("openai/gpt-5-mini") < markdown.index("openrouter/qwen")
    assert "## Notable Failed Cases" in markdown
    assert "`case-mid` on `openrouter/qwen`" in markdown
    assert "## Execution Errors" in markdown
    assert "`case-low` on `openrouter/qwen`" in markdown
    assert "error=TimeoutError" in markdown
    assert "## Warnings" in markdown
    assert "Judge endpoint unavailable for one provider" in markdown


def test_render_leaderboard_html_includes_leaderboard_pies_charts_and_explanation() -> None:
    html_output = render_leaderboard_html(_sample_results())

    assert "<h2>Leaderboard</h2>" in html_output
    assert "Click any column header to sort ascending/descending." in html_output
    assert 'table class="sortable" data-default-sort-column="2" data-default-sort-order="desc"' in html_output
    assert "Sort rows by this column" in html_output
    assert "<h2>Pie Charts</h2>" in html_output
    assert "class=\"pie-chart\"" in html_output
    assert "<h2>Performance Charts</h2>" in html_output
    assert "Final Score Chart" in html_output
    assert "Pass Rate Chart" in html_output
    assert "Latency Chart (p50 seconds)" in html_output
    assert "<h2>Explanation</h2>" in html_output
    assert "Notable Failed Cases" in html_output
    assert "Execution Errors" in html_output
    assert "case-low" in html_output
    assert "TimeoutError" in html_output
    assert "Judge endpoint unavailable for one provider" in html_output


def test_report_regeneration_writes_markdown_and_html(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_sample_results()), encoding="utf-8")

    markdown_path, html_path = regenerate_reports_from_json(results_path)

    assert markdown_path == tmp_path / "leaderboard.md"
    assert html_path == tmp_path / "leaderboard.html"
    assert "LLM Eval Leaderboard" in markdown_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html_text
    assert "<h2>Leaderboard</h2>" in html_text

    legacy_path = regenerate_report_from_json(results_path)
    assert legacy_path == markdown_path


def test_write_html_report_handles_empty_models(tmp_path: Path) -> None:
    payload = {
        "run_id": "r-empty",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "datasets": [],
        "model_summaries": [],
        "case_results": [],
        "warnings": [],
    }

    html_path = tmp_path / "leaderboard.html"
    write_html_report(payload, html_path)
    html_text = html_path.read_text(encoding="utf-8")

    assert "No model results were generated." in html_text
    assert "<h2>Explanation</h2>" in html_text


def test_write_html_report_writes_pdf_and_png_assets_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVAL_EXPORT_PAGE_ASSETS", "1")
    monkeypatch.setattr("llm_eval_suite.report.shutil.which", lambda binary: "/usr/bin/playwright")

    commands: list[list[str]] = []

    def _fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_text("asset", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("llm_eval_suite.report.subprocess.run", _fake_run)

    html_path = tmp_path / "leaderboard.html"
    write_html_report(_sample_results(), html_path)

    assets_dir = tmp_path / "leaderboard_assets"
    assert (assets_dir / "leaderboard.pdf").exists()
    assert (assets_dir / "leaderboard.png").exists()
    assert len(commands) == 2
    assert commands[0][1] == "pdf"
    assert commands[1][1] == "screenshot"


def test_write_history_report_writes_pdf_and_png_assets_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EVAL_EXPORT_PAGE_ASSETS", "1")
    monkeypatch.setattr("llm_eval_suite.report.shutil.which", lambda binary: "/usr/bin/playwright")

    def _fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[3]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[3]).write_text("asset", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("llm_eval_suite.report.subprocess.run", _fake_run)

    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    output_path = write_history_report(reports_root)

    assert output_path == reports_root / "history.html"
    assets_dir = reports_root / "history_assets"
    assert (assets_dir / "history.pdf").exists()
    assert (assets_dir / "history.png").exists()


def test_write_history_report_groups_runs_day_by_day_and_computes_winner_mean(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    run_a = reports_root / "20260210T120000Z"
    run_b = reports_root / "20260210T180000Z"
    run_c = reports_root / "20260211T090000Z"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    run_c.mkdir(parents=True)

    payload_a = {
        "run_id": "20260210T120000Z",
        "started_at": "2026-02-10T12:00:00Z",
        "finished_at": "2026-02-10T12:05:00Z",
        "datasets": [],
        "model_summaries": [
            {"model_name": "openai/gpt-5.2", "final_score_avg": 0.90},
            {"model_name": "anthropic/claude-sonnet-4.5", "final_score_avg": 0.85},
        ],
        "case_results": [{}, {}],
        "warnings": [],
    }
    payload_b = {
        "run_id": "20260210T180000Z",
        "started_at": "2026-02-10T18:00:00Z",
        "finished_at": "2026-02-10T18:06:00Z",
        "datasets": [],
        "model_summaries": [
            {"model_name": "anthropic/claude-sonnet-4.5", "final_score_avg": 0.80},
            {"model_name": "openai/gpt-5.2", "final_score_avg": 0.79},
        ],
        "case_results": [{}, {}, {}],
        "warnings": ["warn-1"],
    }
    payload_c = {
        "run_id": "20260211T090000Z",
        "started_at": "2026-02-11T09:00:00Z",
        "finished_at": "2026-02-11T09:04:00Z",
        "datasets": [],
        "model_summaries": [
            {"model_name": "openai/gpt-5.2", "final_score_avg": 1.0},
            {"model_name": "anthropic/claude-sonnet-4.5", "final_score_avg": 0.95},
        ],
        "case_results": [{}],
        "warnings": [],
    }
    (run_a / "results.json").write_text(json.dumps(payload_a), encoding="utf-8")
    (run_b / "results.json").write_text(json.dumps(payload_b), encoding="utf-8")
    (run_c / "results.json").write_text(json.dumps(payload_c), encoding="utf-8")

    output_path = write_history_report(reports_root)
    html_text = output_path.read_text(encoding="utf-8")

    assert output_path == reports_root / "history.html"
    assert "LLM Eval Reports History" in html_text
    assert "Historical Leaderboard" in html_text
    assert "Provider Cross-Compare" in html_text
    assert "Leaderboard Charts" in html_text
    assert "Historical Mean Final Score" in html_text
    assert "Global Win Rate" in html_text
    assert "Historical Mean Pass Rate" in html_text
    assert "Click any column header to sort ascending/descending." in html_text
    assert "updateRankColumn" in html_text
    assert "Mean Final" in html_text
    assert "Mean Best Final/Report" in html_text
    assert "Mean Pass Rate" in html_text
    assert "Day-by-Day Runs" in html_text
    assert "2026-02-10" in html_text
    assert "2026-02-11" in html_text
    assert "Mean Winner Score" in html_text
    assert "Distinct Providers" in html_text
    assert "66.7%" in html_text
    assert "0.897" in html_text
    assert "total=2.700 / reports=3" in html_text
    assert "openai/gpt-5.2" in html_text
    assert "anthropic/claude-sonnet-4.5" in html_text
    assert "20260210T120000Z/leaderboard.html" in html_text


def test_render_history_html_handles_missing_reports_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    html_text = render_history_html(missing_root)
    assert "Historical Leaderboard" in html_text
    assert "Provider Cross-Compare" in html_text
    assert "Leaderboard Charts" in html_text
    assert "No chart data available." in html_text
    assert "No completed report data found." in html_text
    assert "No provider comparison data found." in html_text
    assert "No run directories found under this reports root." in html_text
