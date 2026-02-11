from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from llm_eval_suite.runner import _extract_usage

_EXTERNAL_ROOT = Path("external_repo/vai-livekit-next-migrations")
_EXTERNAL_USAGE_MODEL = _EXTERNAL_ROOT / "src/vai/agent/models/rpc/usage.py"
_EXTERNAL_RPC_DOC = _EXTERNAL_ROOT / "docs/rpc.md"

_EXPECTED_USAGE_FIELDS = {
    "llm_prompt_tokens",
    "llm_prompt_cached_tokens",
    "llm_input_audio_tokens",
    "llm_completion_tokens",
    "llm_output_audio_tokens",
    "tts_characters_count",
    "tts_audio_duration",
    "stt_audio_duration",
}


def _require_external_repo() -> None:
    if not _EXTERNAL_USAGE_MODEL.exists() or not _EXTERNAL_RPC_DOC.exists():
        pytest.skip("external_repo usage contract files are not available in this environment")


def test_external_usage_model_fields_match_expected_contract() -> None:
    _require_external_repo()
    source = _EXTERNAL_USAGE_MODEL.read_text(encoding="utf-8")

    # Capture typed metric field definitions declared in UsageSummaryMetrics.
    declared_fields = set(
        re.findall(r"^\s+([a-z_]+)\s*:\s*(?:int|float)\s*=\s*Field\(", source, flags=re.MULTILINE)
    )

    assert declared_fields == _EXPECTED_USAGE_FIELDS


def test_extract_usage_parses_usage_summary_payload_from_external_rpc_docs() -> None:
    _require_external_repo()
    markdown = _EXTERNAL_RPC_DOC.read_text(encoding="utf-8")
    match = re.search(
        r"### `usage_summary`.*?```json\s*(\{.*?\})\s*```",
        markdown,
        flags=re.DOTALL,
    )
    assert match is not None, "Could not find usage_summary JSON example in external RPC docs."

    payload = json.loads(match.group(1))
    stats = _extract_usage(payload)

    assert stats.llm_prompt_tokens == 0
    assert stats.llm_prompt_cached_tokens == 0
    assert stats.llm_input_audio_tokens == 0
    assert stats.llm_completion_tokens == 0
    assert stats.llm_output_audio_tokens == 0
    assert stats.tts_characters_count == 0
    assert stats.tts_audio_duration == 0.0
    assert stats.stt_audio_duration == 0.0
