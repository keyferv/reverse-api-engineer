"""Tests for cursor_auth.get_cursor_api_key."""

from pathlib import Path

import pytest

from reverse_api.cursor_auth import get_cursor_api_key


def test_get_cursor_api_key_prefers_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURSOR_API_KEY_FILE", raising=False)
    monkeypatch.setenv("CURSOR_API_KEY", "  env-key  ")
    assert get_cursor_api_key() == "env-key"


def test_get_cursor_api_key_file_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    key_file = tmp_path / "key.txt"
    key_file.write_text("# comment\n\nfile-key\n")
    monkeypatch.setenv("CURSOR_API_KEY_FILE", str(key_file))
    assert get_cursor_api_key() == "file-key"


def test_get_cursor_api_key_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("from-file")
    monkeypatch.setenv("CURSOR_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("CURSOR_API_KEY", "from-env")
    assert get_cursor_api_key() == "from-env"
