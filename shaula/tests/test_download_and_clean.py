"""download_and_clean_lightcurve against a fake lightkurve search."""

from __future__ import annotations

import contextlib
import io

import lightkurve as lk
import numpy as np
import pytest

from .. import download_and_clean as dac
from ..paths import LIGHTCURVE_CACHE_ENV


class _FakeLC:
    time = np.arange(3.0)

    def stitch(self):
        return self

    def remove_nans(self):
        return self

    def normalize(self):
        return self

    def remove_outliers(self, **kwargs):
        return self


class _FakeResult:
    def __init__(self, record):
        self.record = record

    def __len__(self):
        return 1

    def download(self, **kwargs):
        self.record["download"] = kwargs
        return _FakeLC()

    def download_all(self, **kwargs):
        self.record["download_all"] = kwargs
        return _FakeLC()


@pytest.fixture
def record(monkeypatch):
    rec = {}
    monkeypatch.setattr(lk, "search_lightcurve", lambda *a, **k: _FakeResult(rec))
    monkeypatch.delenv(LIGHTCURVE_CACHE_ENV, raising=False)
    return rec


def _run(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return dac.download_and_clean_lightcurve(*args, **kwargs)


@pytest.mark.parametrize("use_all", [False, True])
def test_explicit_dir_is_passed_per_call(record, tmp_path, use_all):
    _run("Kepler-186", "Kepler", all=use_all, cache_dir=tmp_path / "x")
    call = record["download_all" if use_all else "download"]
    assert call["download_dir"] == str(tmp_path / "x")


@pytest.mark.parametrize("use_all", [False, True])
def test_env_dir_is_passed_per_call(record, tmp_path, monkeypatch, use_all):
    monkeypatch.setenv(LIGHTCURVE_CACHE_ENV, str(tmp_path / "env"))
    _run("Kepler-186", "Kepler", all=use_all)
    call = record["download_all" if use_all else "download"]
    assert call["download_dir"] == str(tmp_path / "env")


@pytest.mark.parametrize("use_all", [False, True])
def test_no_dir_passes_none(record, use_all):
    _run("Kepler-186", "Kepler", all=use_all)
    call = record["download_all" if use_all else "download"]
    assert call["download_dir"] is None


def test_global_lightkurve_config_is_untouched(record, tmp_path):
    before = lk.conf.cache_dir
    _run("Kepler-186", "Kepler", cache_dir=tmp_path / "x")
    assert lk.conf.cache_dir == before
