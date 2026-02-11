from __future__ import annotations

import json
from pathlib import Path

from llm_eval_suite.report import (
    regenerate_report_from_json,
    regenerate_reports_from_json,
    render_leaderboard_html,
    render_leaderboard_markdown,
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
    assert "`case-low` on `openrouter/qwen`" in markdown
    assert "error=TimeoutError" in markdown
    assert "## Warnings" in markdown
    assert "Judge endpoint unavailable for one provider" in markdown


def test_render_leaderboard_html_includes_leaderboard_pies_charts_and_explanation() -> None:
    html_output = render_leaderboard_html(_sample_results())

    assert "<h2>Leaderboard</h2>" in html_output
    assert "<h2>Pie Charts</h2>" in html_output
    assert "class=\"pie-chart\"" in html_output
    assert "<h2>Performance Charts</h2>" in html_output
    assert "Final Score Chart" in html_output
    assert "Pass Rate Chart" in html_output
    assert "Latency Chart (p50 seconds)" in html_output
    assert "<h2>Explanation</h2>" in html_output
    assert "Notable Failed Cases" in html_output
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
