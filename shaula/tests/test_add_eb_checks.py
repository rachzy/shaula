"""The backfill script that adds the eclipsing-binary checks to feature CSVs."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..scripts import add_eb_checks
from ..sesmes import EB_CHECK_COLUMNS

PERIOD = 5.0
EPOCH = 1.3
DURATION = 0.2


def _lightcurve(seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 200.0, 0.0204)
    flux = 1.0 + rng.normal(0.0, 2e-4, time.size)
    dt = np.mod(time - EPOCH + 0.5 * PERIOD, PERIOD) - 0.5 * PERIOD
    flux[np.abs(dt) < 0.5 * DURATION] -= 2e-3
    return SimpleNamespace(
        time=SimpleNamespace(value=time), flux=SimpleNamespace(value=flux)
    )


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


ROW = {
    "period_days": PERIOD,
    "t0": EPOCH,
    "duration_days": DURATION,
    "max_mes": 0.1 + 0.2,  # a float that only round-trips with care
    "detection_status": "accepted",
}


def test_star_of_reads_the_file_name(tmp_path):
    assert add_eb_checks.star_of(tmp_path / "KIC11904151_inverted_20261020.csv") == (
        "KIC",
        11904151,
    )
    assert add_eb_checks.star_of(tmp_path / "TIC123_20261020.csv") == ("TIC", 123)
    assert add_eb_checks.star_of(tmp_path / "Kepler-90_20261020.csv") is None


def test_fill_csv_appends_the_columns_and_keeps_existing_values(tmp_path):
    path = tmp_path / "KIC1_20261025.csv"
    _write_csv(path, [ROW])
    before = list(csv.reader(path.open()))
    assert add_eb_checks.fill_csv(path, _lightcurve()) == 1
    after = list(csv.reader(path.open()))
    assert after[0][: len(before[0])] == before[0]
    assert after[0][len(before[0]) :] == list(EB_CHECK_COLUMNS)
    assert after[1][: len(before[1])] == before[1]
    row = pd.read_csv(path).iloc[0]
    assert row["secondary_scan_trials"] > 0
    assert np.isfinite(row["even_odd_depth_diff_sigma"])


def test_rows_gap_the_other_accepted_candidates(monkeypatch):
    seen = []

    def fake_measure(time, flux, period, t0, duration, gap_mask=None):
        seen.append((period, int(gap_mask.sum())))
        return dict.fromkeys(EB_CHECK_COLUMNS, np.nan)

    monkeypatch.setattr(add_eb_checks, "measure_eb_checks", fake_measure)
    lc = _lightcurve()
    rows = [
        ROW,
        {**ROW, "period_days": 7.3, "t0": 2.0},
        {**ROW, "period_days": 11.1, "t0": 3.0, "detection_status": "provisional"},
    ]
    add_eb_checks.eb_checks_for_rows(lc.time.value, lc.flux.value, rows)
    gaps = dict(seen)
    # The provisional row never masks anything, but sees both accepted ones.
    assert 0 < gaps[PERIOD] < gaps[11.1]
    assert 0 < gaps[7.3] < gaps[11.1]


def test_needs_filling_skips_empty_and_filled_csvs(tmp_path):
    empty = tmp_path / "KIC1_inverted_20261025.csv"
    empty.write_text("\n")
    assert not add_eb_checks.needs_filling(empty, overwrite=True)
    filled = tmp_path / "KIC1_20261025.csv"
    _write_csv(filled, [{**ROW, **dict.fromkeys(EB_CHECK_COLUMNS, 1.0)}])
    assert not add_eb_checks.needs_filling(filled, overwrite=False)
    assert add_eb_checks.needs_filling(filled, overwrite=True)
    fresh = tmp_path / "KIC2_20261025.csv"
    _write_csv(fresh, [ROW])
    assert add_eb_checks.needs_filling(fresh, overwrite=False)


def test_main_fills_a_directory_and_flips_inverted_csvs(tmp_path, monkeypatch):
    normal = tmp_path / "KIC1_20261025.csv"
    inverted = tmp_path / "KIC1_inverted_20261025.csv"
    _write_csv(normal, [ROW])
    _write_csv(inverted, [ROW])
    loads, flips = [], []

    def fake_load(prefix, number, cache_dir):
        loads.append((prefix, number))
        return _lightcurve()

    def fake_invert(lc, prefix, number, reference_dir):
        flips.append((prefix, number))
        return lc

    monkeypatch.setattr(add_eb_checks, "load_lightcurve", fake_load)
    monkeypatch.setattr(add_eb_checks, "inverted_lightcurve", fake_invert)
    assert add_eb_checks.main([str(tmp_path)]) == 0
    assert loads == [("KIC", 1)]  # one download serves both CSVs
    assert flips == [("KIC", 1)]
    for path in (normal, inverted):
        assert set(EB_CHECK_COLUMNS) <= set(pd.read_csv(path).columns)
    # A second run finds nothing to do.
    assert add_eb_checks.main([str(tmp_path)]) == 0
    assert loads == [("KIC", 1)]
