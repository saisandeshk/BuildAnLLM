"""Unit tests for demo mode helpers."""

import pytest

from backend.app.core.demo import block_if_demo, is_demo_mode
from fastapi import HTTPException


@pytest.mark.unit
@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_is_demo_mode_enabled(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DEMO_MODE", value)
    assert is_demo_mode() is True


@pytest.mark.unit
def test_is_demo_mode_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert is_demo_mode() is False


@pytest.mark.unit
def test_block_if_demo_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    with pytest.raises(HTTPException):
        block_if_demo()

