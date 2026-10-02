"""Resolution of the light-curve cache directory."""

from __future__ import annotations

from pathlib import Path

from ..paths import LIGHTCURVE_CACHE_ENV, lightcurve_cache_dir


def test_explicit_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(tmp_path / "from_env"))
    chosen = tmp_path / "explicit"
    assert lightcurve_cache_dir(chosen) == chosen


def test_env_var_used_when_no_argument(tmp_path, monkeypatch):
    expected = tmp_path / "from_env"
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(expected))
    assert lightcurve_cache_dir() == expected


def test_falls_back_to_package_relative_default(monkeypatch):
    monkeypatch.delenv(LIGHTCURVE_CACHE_ENV, raising=False)
    assert lightcurve_cache_dir().name == "lightkurve"


def test_missing_directory_is_created_with_parents(tmp_path):
    nested = tmp_path / "a" / "b" / "lightkurve"
    assert not nested.exists()
    resolved = lightcurve_cache_dir(nested)
    assert resolved.is_dir()


def test_accepts_a_string_path(tmp_path):
    resolved = lightcurve_cache_dir(str(tmp_path / "as_string"))
    assert isinstance(resolved, Path)
    assert resolved.is_dir()
