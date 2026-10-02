"""Resolution and redirection of lightkurve's download cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..paths import LIGHTCURVE_CACHE_ENV, lightcurve_cache_dir


@pytest.fixture(autouse=True)
def _restore_lightkurve_cache():
    """Keep one test's redirection from leaking into the next."""
    import lightkurve as lk

    original = lk.conf.cache_dir
    yield
    lk.conf.cache_dir = original


def test_explicit_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(tmp_path / "from_env"))
    chosen = tmp_path / "explicit"
    assert lightcurve_cache_dir(chosen) == chosen


def test_env_var_used_when_no_argument(tmp_path, monkeypatch):
    expected = tmp_path / "from_env"
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(expected))
    assert lightcurve_cache_dir() == expected


def test_falls_back_to_lightkurves_own_default(monkeypatch):
    """No package-relative default: a pip-installed library must not write into site-packages."""
    from lightkurve.config import get_cache_dir

    monkeypatch.delenv(LIGHTCURVE_CACHE_ENV, raising=False)
    assert lightcurve_cache_dir() == Path(get_cache_dir())


def test_missing_directory_is_created_with_parents(tmp_path):
    nested = tmp_path / "a" / "b" / "lightkurve"
    assert not nested.exists()
    lightcurve_cache_dir(nested)
    assert nested.is_dir()


def test_accepts_a_string_path(tmp_path):
    resolved = lightcurve_cache_dir(str(tmp_path / "as_string"))
    assert isinstance(resolved, Path)
    assert resolved.is_dir()


def test_redirects_lightkurves_download_cache(tmp_path):
    """Assigning lk.conf.cache_dir is the only thing that moves downloads."""
    import lightkurve as lk

    chosen = tmp_path / "cache"
    lightcurve_cache_dir(chosen)
    assert Path(lk.conf.cache_dir) == chosen


def test_env_var_reaches_lightkurve_not_just_the_return_value(tmp_path, monkeypatch):
    """The mounted-volume case: returning the right path is not enough."""
    import lightkurve as lk

    expected = tmp_path / "from_env"
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(expected))
    lightcurve_cache_dir()
    assert Path(lk.conf.cache_dir) == expected
