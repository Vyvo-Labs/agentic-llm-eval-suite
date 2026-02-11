from __future__ import annotations

from pathlib import Path

from llm_eval_suite.models import RunResults
from llm_eval_suite.runner import persist_run_results


def test_persist_run_results_writes_html_and_markdown_reports(tmp_path: Path) -> None:
    results = RunResults(
        run_id="",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        config={},
        git_sha=None,
        datasets=[],
        model_summaries=[],
        case_results=[],
        warnings=[],
    )

    run_dir = persist_run_results(results, tmp_path)

    markdown_path = run_dir / "leaderboard.md"
    html_path = run_dir / "leaderboard.html"
    results_path = run_dir / "results.json"
    raw_path = run_dir / "raw_responses.jsonl"

    assert results_path.exists()
    assert markdown_path.exists()
    assert html_path.exists()
    assert raw_path.exists()
    assert "LLM Eval Leaderboard" in markdown_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in html_content
    assert "<h2>Leaderboard</h2>" in html_content
