from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import llm_eval_suite.cli as cli
from llm_eval_suite.models import RunResults


def _base_run_args() -> argparse.Namespace:
    return argparse.Namespace(
        provider=None,
        model=None,
        dataset=None,
        tags=None,
        categories=None,
        case_id=None,
        max_cases=None,
        concurrency=None,
        timeout_s=None,
        max_completion_tokens=None,
        reasoning_effort=None,
        output_dir=None,
        detailed_pdf=True,
        detailed_pdf_output=None,
        include_raw_output=False,
        cache=False,
        no_cache=False,
    )


def test_run_parser_enables_detailed_pdf_by_default() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["run"])
    assert args.detailed_pdf is True


def test_run_parser_allows_disabling_detailed_pdf() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["run", "--no-detailed-pdf"])
    assert args.detailed_pdf is False


def test_report_command_generates_detailed_pdf_with_default_output(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:01:00Z",
                "datasets": [],
                "model_summaries": [],
                "case_results": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "regenerate_reports_from_json",
        lambda *_args, **_kwargs: (tmp_path / "leaderboard.md", tmp_path / "leaderboard.html"),
    )

    captured: dict[str, object] = {}

    def _fake_write(payload: dict[str, object], output_path: Path, *, include_raw_output: bool = False) -> bool:
        captured["payload"] = payload
        captured["output_path"] = output_path
        captured["include_raw_output"] = include_raw_output
        return True

    monkeypatch.setattr(cli, "write_detailed_pdf_report", _fake_write)

    args = argparse.Namespace(
        input=results_path,
        output=None,
        html_output=None,
        detailed_pdf=True,
        detailed_pdf_output=None,
        include_raw_output=True,
    )
    exit_code = cli._report_command(args)

    assert exit_code == 0
    assert captured["output_path"] == tmp_path / "leaderboard_detailed.pdf"
    assert captured["include_raw_output"] is True


def test_report_command_detailed_pdf_fail_soft(tmp_path: Path, monkeypatch) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:01:00Z",
                "datasets": [],
                "model_summaries": [],
                "case_results": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "regenerate_reports_from_json",
        lambda *_args, **_kwargs: (tmp_path / "leaderboard.md", tmp_path / "leaderboard.html"),
    )
    monkeypatch.setattr(cli, "write_detailed_pdf_report", lambda *_args, **_kwargs: False)

    args = argparse.Namespace(
        input=results_path,
        output=None,
        html_output=None,
        detailed_pdf=True,
        detailed_pdf_output=tmp_path / "custom.pdf",
        include_raw_output=False,
    )
    exit_code = cli._report_command(args)

    assert exit_code == 0


def test_run_command_generates_detailed_pdf_with_default_output(tmp_path: Path, monkeypatch) -> None:
    config = SimpleNamespace(
        concurrency=1,
        timeout_s=30.0,
        max_completion_tokens=256,
        reasoning_effort=None,
        output_dir=tmp_path / "reports",
    )
    monkeypatch.setattr(cli, "load_config", lambda: config)
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda *_args, **_kwargs: RunResults(
            run_id="",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            config={},
            git_sha=None,
            datasets=[],
            model_summaries=[],
            case_results=[],
            warnings=[],
        ),
    )

    run_dir = tmp_path / "reports" / "r1"
    monkeypatch.setattr(cli, "persist_run_results", lambda *_args, **_kwargs: run_dir)

    captured: dict[str, object] = {}

    def _fake_write(_payload: dict[str, object], output_path: Path, *, include_raw_output: bool = False) -> bool:
        captured["output_path"] = output_path
        captured["include_raw_output"] = include_raw_output
        return True

    monkeypatch.setattr(cli, "write_detailed_pdf_report", _fake_write)

    args = _base_run_args()
    exit_code = cli._run_command(args)

    assert exit_code == 0
    assert captured["output_path"] == run_dir / "leaderboard_detailed.pdf"
    assert captured["include_raw_output"] is False
