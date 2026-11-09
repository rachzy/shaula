"""download_and_clean_lightcurve against a fake lightkurve search."""

from __future__ import annotations

import contextlib
import io

import lightkurve as lk
import numpy as np
import pytest
from astropy.table import Table

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


class _TableResult:
    """A search result that is only its table, like lightkurve's SearchResult."""

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return _TableResult(self.table[key])


def _products(*rows):
    """Rows of (target_name, author, exptime seconds, distance arcsec)."""
    names, authors, seconds, distance = zip(*rows) if rows else ([], [], [], [])
    return _TableResult(
        Table(
            {
                "target_name": list(names),
                "author": list(authors),
                "exptime": list(seconds),
                "distance": list(distance),
            }
        )
    )


@pytest.fixture
def searched(monkeypatch):
    """Serve one canned result and record how the search was called."""
    state = {"result": _products(), "calls": []}

    def fake_search(target, **kwargs):
        state["calls"].append(kwargs)
        return state["result"]

    monkeypatch.setattr(lk, "search_lightcurve", fake_search)
    return state


def test_tess_default_asks_for_every_default_author_in_one_query(searched):
    dac.search_default_product("TIC 1", "TESS")
    assert searched["calls"] == [
        {"mission": "TESS", "author": ["SPOC", "TESS-SPOC", "QLP"]}
    ]


def test_tess_default_keeps_only_two_minute_spoc_when_there_is_any(searched):
    searched["result"] = _products(
        ("1", "SPOC", 120.0, 0.0),
        ("1", "SPOC", 20.0, 0.0),
        ("1", "TESS-SPOC", 1800.0, 0.0),
        ("1", "QLP", 600.0, 0.0),
        ("1", "SPOC", 120.0, 0.0),
    )
    found = dac.search_default_product("TIC 1", "TESS")
    assert (found.author, found.exptime) == ("SPOC", "short")
    assert list(found.result.table["exptime"]) == [120.0, 120.0]
    assert set(found.result.table["author"]) == {"SPOC"}


def test_tess_default_falls_back_to_tess_spoc_at_any_exposure(searched):
    searched["result"] = _products(
        ("1", "QLP", 1800.0, 0.0),
        ("1", "TESS-SPOC", 1800.0, 0.0),
        ("1", "TESS-SPOC", 600.0, 0.0),
        ("1", "SPOC", 20.0, 0.0),
    )
    found = dac.search_default_product("TIC 1", "TESS")
    assert (found.author, found.exptime) == ("TESS-SPOC", None)
    assert list(found.result.table["exptime"]) == [1800.0, 600.0]


def test_tess_default_falls_back_to_qlp_last(searched):
    searched["result"] = _products(("1", "QLP", 200.0, 0.0), ("1", "QLP", 600.0, 0.0))
    found = dac.search_default_product("TIC 1", "TESS")
    assert (found.author, found.exptime) == ("QLP", None)
    assert len(found.result) == 2


def test_tess_default_with_nothing_usable_is_empty(searched):
    found = dac.search_default_product("TIC 1", "TESS")
    assert len(found.result) == 0
    searched["result"] = _products(("1", "SPOC", 20.0, 0.0))
    assert len(dac.search_default_product("TIC 1", "TESS").result) == 0


def test_cone_search_keeps_the_nearest_star_not_the_best_product(searched):
    searched["result"] = _products(
        ("2", "SPOC", 120.0, 3.5),
        ("1", "QLP", 1800.0, 0.2),
        ("1", "QLP", 600.0, 0.2),
    )
    found = dac.search_default_product("position", "TESS", radius=4.0)
    assert set(found.result.table["target_name"]) == {"1"}
    assert found.author == "QLP"
    assert searched["calls"][0]["radius"] == 4.0


def test_explicit_tess_author_is_searched_as_given_at_any_exposure(searched):
    found = dac.search_default_product("TIC 1", "TESS", author="QLP")
    assert searched["calls"] == [{"mission": "TESS", "author": "QLP"}]
    assert (found.author, found.exptime) == ("QLP", None)


def test_explicit_exposure_keeps_the_default_author(searched):
    dac.search_default_product("TIC 1", "TESS", exptime=20)
    assert searched["calls"] == [{"mission": "TESS", "author": "SPOC", "exptime": 20}]


def test_kepler_default_is_unchanged(searched):
    found = dac.search_default_product("KIC 1", "Kepler")
    assert searched["calls"] == [
        {"mission": "Kepler", "author": "Kepler", "exptime": "long"}
    ]
    assert (found.author, found.exptime) == ("Kepler", "long")


def _light_curve(cadence_minutes, days=2.0, start=0.0):
    time = start + np.arange(0.0, days, cadence_minutes / 1440.0)
    return lk.LightCurve(time=time, flux=np.ones(time.size))


def _cadence_minutes(lc):
    return float(np.median(np.diff(lc.time.value)) * 1440.0)


def test_mixed_cadences_are_binned_to_the_coarsest():
    mixed = lk.LightCurveCollection(
        [_light_curve(30.0), _light_curve(10.0, start=10.0), _light_curve(200 / 60, start=20.0)]
    )
    binned = dac.bin_to_common_cadence(mixed)
    assert [round(_cadence_minutes(lc)) for lc in binned] == [30, 30, 30]
    assert len(binned[0]) == len(mixed[0])
    assert len(binned[1]) == pytest.approx(len(mixed[1]) / 3, abs=1)


def test_a_single_cadence_set_is_returned_untouched():
    same = lk.LightCurveCollection([_light_curve(2.0), _light_curve(2.0, start=5.0)])
    assert dac.bin_to_common_cadence(same) is same


class _FfiResult:
    def __init__(self, table, collection):
        self.table = table
        self.collection = collection

    def __len__(self):
        return len(self.table)

    def __getitem__(self, key):
        return _FfiResult(self.table[key], self.collection)

    def download_all(self, **kwargs):
        return self.collection


def test_stitched_ffi_fallback_comes_out_at_one_cadence(monkeypatch):
    table = _products(("1", "TESS-SPOC", 1800.0, 0.0), ("1", "TESS-SPOC", 600.0, 0.0)).table
    collection = lk.LightCurveCollection(
        [_light_curve(30.0), _light_curve(10.0, start=10.0)]
    )
    monkeypatch.setattr(
        lk, "search_lightcurve", lambda *a, **k: _FfiResult(table, collection)
    )
    lc = _run("TIC 1", "TESS", all=True)
    steps = np.diff(lc.time.value) * 1440.0
    assert np.all(steps[steps < 60.0] == pytest.approx(30.0, abs=0.01))
