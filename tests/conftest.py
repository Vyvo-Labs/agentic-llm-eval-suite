from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_page_asset_export_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_EXPORT_PAGE_ASSETS", "0")
