"""Resolution of the lightkurve download directory (a pure resolver)."""

from __future__ import annotations

from pathlib import Path

import lightkurve as lk

from ..paths import LIGHTCURVE_CACHE_ENV, lightcurve_cache_dir


def test_explicit_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(tmp_path / "from_env"))
    chosen = tmp_path / "explicit"
    assert lightcurve_cache_dir(chosen) == chosen


def test_env_var_used_when_no_argument(tmp_path, monkeypatch):
    expected = tmp_path / "from_env"
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(expected))
    assert lightcurve_cache_dir() == expected


def test_returns_none_and_creates_nothing_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(LIGHTCURVE_CACHE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert lightcurve_cache_dir() is None
    assert list(tmp_path.iterdir()) == []


def test_empty_env_var_is_treated_as_unset(tmp_path, monkeypatch):
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, "")
    monkeypatch.chdir(tmp_path)
    assert lightcurve_cache_dir() is None
    assert list(tmp_path.iterdir()) == []


def test_missing_directory_is_created_with_parents(tmp_path):
    nested = tmp_path / "a" / "b" / "lightkurve"
    assert not nested.exists()
    lightcurve_cache_dir(nested)
    assert nested.is_dir()


def test_accepts_a_string_path(tmp_path):
    resolved = lightcurve_cache_dir(str(tmp_path / "as_string"))
    assert isinstance(resolved, Path)
    assert resolved.is_dir()


def test_does_not_change_lightkurves_global_cache_dir(tmp_path, monkeypatch):
    """The point of the per-call design: no process-wide state is touched."""
    before = lk.conf.cache_dir
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(tmp_path / "from_env"))
    lightcurve_cache_dir(tmp_path / "explicit")
    lightcurve_cache_dir()
    assert lk.conf.cache_dir == before
