from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, parse_csv_arg
from .providers import resolve_candidate_models
from .report import regenerate_reports_from_json, write_history_report
from .runner import RunOptions, persist_run_results, run_benchmark


def _split_repeated(values: list[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        parsed.extend(parse_csv_arg(value))
    return parsed


def _page_assets_dir(html_path: Path) -> Path:
    return html_path.parent / f"{html_path.stem}_assets"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Env-driven LLM benchmark + judge suite")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run benchmark suite")
    run_parser.add_argument("--provider", action="append", help="Provider filter (can repeat or be comma-separated)")
    run_parser.add_argument("--model", action="append", help="Model substring filter (can repeat or be comma-separated)")
    run_parser.add_argument("--dataset", action="append", help="Dataset path (can repeat)")
    run_parser.add_argument("--tags", help="Filter cases by tags (comma-separated)")
    run_parser.add_argument("--categories", help="Filter cases by categories (comma-separated)")
    run_parser.add_argument("--case-id", action="append", help="Include only case ids (can repeat)")
    run_parser.add_argument("--max-cases", type=int, help="Limit cases per run")
    run_parser.add_argument("--concurrency", type=int, help="Worker concurrency")
    run_parser.add_argument("--timeout-s", type=float, help="Model request timeout in seconds")
    run_parser.add_argument("--max-completion-tokens", type=int, help="Max completion tokens per response")
    run_parser.add_argument("--reasoning-effort", help="Override reasoning effort")
    run_parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    run_parser.add_argument("--cache", action="store_true", help="Force-enable cache")
    run_parser.add_argument("--no-cache", action="store_true", help="Disable cache")

    list_parser = subparsers.add_parser("list-models", help="List resolved model matrix")
    list_parser.add_argument("--provider", action="append", help="Provider filter (can repeat or be comma-separated)")

    report_parser = subparsers.add_parser("report", help="Regenerate markdown + HTML reports from results.json")
    report_parser.add_argument("--input", type=Path, required=True, help="Path to results.json")
    report_parser.add_argument("--output", type=Path, help="Optional markdown output path")
    report_parser.add_argument("--html-output", type=Path, help="Optional HTML output path")

    history_parser = subparsers.add_parser("history", help="Generate day-by-day reports dashboard HTML")
    history_parser.add_argument("--reports-dir", type=Path, help="Reports root directory (default: EVAL_OUTPUT_DIR)")
    history_parser.add_argument("--output", type=Path, help="Optional output path (default: <reports-dir>/history.html)")

    return parser


def _apply_run_overrides(config, args: argparse.Namespace) -> None:
    if args.concurrency is not None:
        config.concurrency = max(1, int(args.concurrency))
    if args.timeout_s is not None:
        config.timeout_s = max(1.0, float(args.timeout_s))
    if args.max_completion_tokens is not None:
        config.max_completion_tokens = max(1, int(args.max_completion_tokens))
    if args.reasoning_effort is not None:
        config.reasoning_effort = args.reasoning_effort.strip() or None
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    if args.cache and args.no_cache:
        raise ValueError("--cache and --no-cache cannot be used together")


def _run_command(args: argparse.Namespace) -> int:
    config = load_config()
    _apply_run_overrides(config, args)

    provider_filters = [item.lower() for item in _split_repeated(args.provider)]
    model_filters = _split_repeated(args.model)

    datasets = [Path(path) for path in args.dataset] if args.dataset else None
    tags = {tag.strip().lower() for tag in parse_csv_arg(args.tags)} if args.tags else None
    categories = {item.strip().lower() for item in parse_csv_arg(args.categories)} if args.categories else None
    case_ids = set(_split_repeated(args.case_id)) if args.case_id else None

    cache_enabled: bool | None = None
    if args.cache:
        cache_enabled = True
    elif args.no_cache:
        cache_enabled = False

    options = RunOptions(
        provider_filters=provider_filters,
        model_filters=model_filters,
        datasets=datasets,
        tags=tags,
        categories=categories,
        case_ids=case_ids,
        max_cases=args.max_cases,
        cache_enabled=cache_enabled,
    )

    results = run_benchmark(config, options)
    run_dir = persist_run_results(results, config.output_dir)

    print(f"Benchmark run completed: {run_dir}")
    print(f"Reports: {run_dir / 'leaderboard.md'} and {run_dir / 'leaderboard.html'}")
    leaderboard_assets = _page_assets_dir(run_dir / "leaderboard.html")
    history_assets = _page_assets_dir(config.output_dir / "history.html")
    if leaderboard_assets.exists():
        print(f"Leaderboard page assets: {leaderboard_assets}")
    if history_assets.exists():
        print(f"History page assets: {history_assets}")
    print(f"Model count: {len(results.model_summaries)}")
    print(f"Case results: {len(results.case_results)}")
    if results.warnings:
        print("Warnings:")
        for warning in results.warnings:
            print(f"- {warning}")

    return 0


def _list_models_command(args: argparse.Namespace) -> int:
    config = load_config()
    provider_filters = [item.lower() for item in _split_repeated(args.provider)]

    resolved = resolve_candidate_models(config, provider_filters)

    for warning in resolved.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not resolved.candidates:
        print("No candidates resolved.")
        return 1

    for candidate in resolved.candidates:
        base_url = candidate.base_url or "https://api.openai.com/v1 (default)"
        print(
            f"{candidate.display_name} -> request_model={candidate.request_model} "
            f"base_url={base_url} api_key_env={candidate.api_key_env}"
        )

    return 0


def _report_command(args: argparse.Namespace) -> int:
    markdown_output, html_output = regenerate_reports_from_json(
        args.input,
        markdown_output_path=args.output,
        html_output_path=args.html_output,
    )
    print(f"Wrote markdown report: {markdown_output}")
    print(f"Wrote HTML report: {html_output}")
    assets_dir = _page_assets_dir(html_output)
    if assets_dir.exists():
        print(f"Wrote page assets: {assets_dir}")
    return 0


def _history_command(args: argparse.Namespace) -> int:
    config = load_config()
    reports_root = args.reports_dir or config.output_dir
    history_output = write_history_report(reports_root, args.output)
    print(f"Wrote history dashboard: {history_output}")
    assets_dir = _page_assets_dir(history_output)
    if assets_dir.exists():
        print(f"Wrote page assets: {assets_dir}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "list-models":
            return _list_models_command(args)
        if args.command == "report":
            return _report_command(args)
        if args.command == "history":
            return _history_command(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
