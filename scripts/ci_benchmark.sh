#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set; skipping live benchmark run."
  exit 0
fi

: "${EVAL_CI_MAX_CASES:=2}"

uv run eval-suite run \
  --provider openai \
  --max-cases "${EVAL_CI_MAX_CASES}" \
  --concurrency 1 \
  --output-dir reports

latest_report_dir="$(find reports -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

echo "Latest benchmark report: ${latest_report_dir}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "latest_report_dir=${latest_report_dir}" >> "${GITHUB_OUTPUT}"
fi
