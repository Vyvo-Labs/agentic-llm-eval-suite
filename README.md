# llm-eval-suite

Env-driven LLM benchmark + hybrid judge suite.

This project benchmarks candidate models using local YAML datasets and computes
scores from deterministic checks plus optional LLM-judge rubric scoring.

## Key behavior

- Loads `.env` from repo root (process env overrides file values).
- Uses OpenAI-compatible provider mapping aligned with `external_repo` conventions:
  - `openai`, `groq`, `openrouter`, `fireworks`, `together`, `cerebras`
- Candidate model matrix comes from provider presets.
- Judge model is dedicated and independently configurable.
- Output artifacts:
  - `results.json`
  - `leaderboard.md`
  - `leaderboard.html`
  - `raw_responses.jsonl`
  - `reports/history.html` (day-by-day dashboard across runs with historical leaderboard + winner metrics)

## Install

```bash
uv sync
```

## Commands

```bash
# Run full benchmark (default datasets)
uv run eval-suite run

# Run filtered benchmark
uv run eval-suite run --provider openai --model gpt-5 --max-cases 10

# List resolved candidate models from env/presets
uv run eval-suite list-models

# Rebuild markdown + HTML reports from prior run
uv run eval-suite report --input reports/<run_id>/results.json

# Rebuild day-by-day dashboard across reports/
uv run eval-suite history

# Optional custom output paths
uv run eval-suite report --input reports/<run_id>/results.json --output reports/custom.md --html-output reports/custom.html
```

## Dataset format

Datasets live in `datasets/*.yaml` and use schema version 1:

```yaml
schema_version: 1
cases:
  - id: "example"
    name: "Example case"
    type: "single_turn" # or multi_turn
    category: "general"
    tags: ["tag1"]
    input:
      user: "Prompt"
      # or messages: [{role, content}, ...]
    expected:
      exact: ["optional exact match"]
      regex: ["optional regex"]
      must_include: ["required phrase"]
      json_valid: false
    judge_rubric:
      criteria:
        - "Criterion text"
      force: false
    weights:
      deterministic: 0.5
      judge: 0.5
```

## Important env vars

### Candidate matrix

- `EVAL_PROVIDER_PRESET` (`all` or comma list of providers, default: `openrouter`)
- `EVAL_MAX_MODELS_PER_PROVIDER` (default: `20`)
- `EVAL_PRESET_MODELS_OPENAI` (optional comma list override)
- `EVAL_PRESET_MODELS_GROQ`
- `EVAL_PRESET_MODELS_OPENROUTER`
- `EVAL_PRESET_MODELS_FIREWORKS`
- `EVAL_PRESET_MODELS_TOGETHER`
- `EVAL_PRESET_MODELS_CEREBRAS`

Legacy agent env aliases (auto-mapped when `EVAL_*` values are not set):

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-4-5-haiku
TOOL_MESSAGE_MODEL=claude-4-5-haiku
LLM_REASONING_EFFORT=none
```

Notes:
- `LLM_PROVIDER=anthropic` maps to `openrouter` for this suite.
- `LLM_MODEL` and `TOOL_MESSAGE_MODEL` do not narrow candidates by default.
- Set `EVAL_LEGACY_MODEL_PIN=true` to use `LLM_MODEL` + `TOOL_MESSAGE_MODEL` as a pinned candidate list.

Default OpenAI preset models:
- `gpt-5-mini`
- `gpt-5`
- `gpt-5.2`
- `openai-5-mini-minimal`
- `openai-5.2-none`

Default OpenRouter preset models:
- `moonshotai/kimi-k2.5`
- `z-ai/glm-5`
- `z-ai/glm-4.7`
- `anthropic/claude-haiku-4.5`
- `anthropic/claude-sonnet-4.5`
- `anthropic/claude-opus-4.1`
- `anthropic/claude-opus-4.5`
- `openai/gpt-5-mini`
- `openai/gpt-5`
- `openai/gpt-5.2`
- `openai/gpt-5.2-none`
- `openai/gpt-4.1`
- `openai/gpt-4.1-mini`
- `z-ai/glm-4.7-flash`
- `minimax/minimax-m2-her`
- `stepfun/step-3.5-flash`
- `meituan/longcat-flash-chat`

### Runtime

- `EVAL_CONCURRENCY` (default: `4`)
- `EVAL_TIMEOUT_S` (default: `90`)
- `EVAL_MAX_COMPLETION_TOKENS` (default: `512`)
- `EVAL_REASONING_EFFORT`
- `EVAL_TEMPERATURE`
- For `gpt-5.2`, set `EVAL_REASONING_EFFORT=none` for minimal-reasoning runs.
- `openai-5.2-none` and `openai/gpt-5.2-none` are aliases for `gpt-5.2` with `reasoning_effort=none`.
- `EVAL_CACHE_ENABLED` (`true` by default)
- `EVAL_CACHE_DIR` (default: `.cache/llm_eval_suite`)
- `EVAL_OUTPUT_DIR` (default: `reports`)

### Judge model

- `EVAL_JUDGE_PROVIDER` (default: `openai`)
- `EVAL_JUDGE_MODEL` (default: `gpt-5.2`)
- `EVAL_JUDGE_BASE_URL` (optional)
- `EVAL_JUDGE_API_KEY` (optional override; otherwise provider key is used)
- `EVAL_JUDGE_TIMEOUT_S` (default: `90`)
- `EVAL_JUDGE_MAX_COMPLETION_TOKENS` (default: `400`)
- `EVAL_JUDGE_REASONING_EFFORT` (default: `xhigh`)

### Provider keys

- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `FIREWORKS_API_KEY`
- `TOGETHER_API_KEY`
- `CEREBRAS_API_KEY`

## Notes

- Secrets are never written to output artifacts.
- Providers missing required API keys are skipped with warnings.
- CI can consume `results.json` while keeping benchmark reporting non-blocking.

## CI (non-blocking)

- Workflow: `.github/workflows/llm-eval-report.yml`
- Script: `scripts/ci_benchmark.sh`
- Behavior:
  - Uses `OPENAI_API_KEY` from GitHub secrets when available.
  - Runs a small benchmark sample (`EVAL_CI_MAX_CASES`, default `2`).
  - Uploads `reports/` artifacts.
  - Does not fail the pipeline (`continue-on-error: true`).
