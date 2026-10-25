import lightkurve as lk
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from ..scripts import add_stellar_density
from ..utils.stellar_density import (
    STELLAR_DENSITY_COLUMNS,
    catalogue_stellar_density,
    stellar_density_features,
    transit_stellar_density,
)

SUN_DENSITY = 1.41  # g/cm^3
SUN_LOGG = 4.438
EARTH_PERIOD = 365.25
EARTH_A_OVER_RSUN = 215.03
EARTH_DEPTH = (1 / 109.1) ** 2


def central_duration(period, a_over_rstar, depth):
    """T14 of a b = 0 transit, from sin(pi T / P) = (1 + k) / (a/R*)."""
    return period / np.pi * np.arcsin((1 + np.sqrt(depth)) / a_over_rstar)


def test_the_sun_has_its_density():
    assert catalogue_stellar_density(SUN_LOGG, 1.0) == pytest.approx(SUN_DENSITY, rel=0.01)


def test_earths_transit_gives_the_suns_density():
    duration = central_duration(EARTH_PERIOD, EARTH_A_OVER_RSUN, EARTH_DEPTH)
    rho = transit_stellar_density(EARTH_PERIOD, duration, EARTH_DEPTH)
    assert rho == pytest.approx(SUN_DENSITY, rel=0.01)


def test_a_planet_of_its_star_has_a_ratio_of_zero():
    duration = central_duration(EARTH_PERIOD, EARTH_A_OVER_RSUN, EARTH_DEPTH)
    feats = stellar_density_features(EARTH_PERIOD, duration, EARTH_DEPTH, SUN_LOGG, 1.0)
    assert set(feats) == set(STELLAR_DENSITY_COLUMNS)
    assert feats["log_rho_ratio"] == pytest.approx(0.0, abs=0.01)


def test_a_transit_too_short_for_the_star_reads_too_dense():
    duration = central_duration(EARTH_PERIOD, EARTH_A_OVER_RSUN, EARTH_DEPTH)
    feats = stellar_density_features(EARTH_PERIOD, duration / 2, EARTH_DEPTH, SUN_LOGG, 1.0)
    # rho goes as T^-3, so halving the duration adds log10(8).
    assert feats["log_rho_ratio"] == pytest.approx(np.log10(8), abs=0.01)


@pytest.mark.parametrize(
    "period, duration, depth",
    [
        (None, 0.1, 1e-3),
        (np.nan, 0.1, 1e-3),
        (-1.0, 0.1, 1e-3),
        (10.0, 0.0, 1e-3),
        (10.0, 5.0, 1e-3),  # half the period: no orbit gives that
        (10.0, 0.1, 0.0),
        (10.0, 0.1, -1e-3),  # a bump, as on an inverted light curve
        (10.0, 0.1, 1.0),
        (10.0, 0.1, np.inf),
    ],
)
def test_impossible_transits_give_nan(period, duration, depth):
    assert np.isnan(transit_stellar_density(period, duration, depth))
    feats = stellar_density_features(period, duration, depth, SUN_LOGG, 1.0)
    assert np.isnan(feats["log_rho_ratio"])


@pytest.mark.parametrize(
    "logg, radius",
    [(None, 1.0), (SUN_LOGG, None), (np.nan, 1.0), (SUN_LOGG, 0.0), ("", 1.0)],
)
def test_a_missing_catalogue_value_gives_nan_never_inf(logg, radius):
    feats = stellar_density_features(10.0, 0.1, 1e-3, logg, radius)
    assert np.isfinite(feats["rho_star_transit"])
    assert np.isnan(catalogue_stellar_density(logg, radius))
    assert np.isnan(feats["log_rho_ratio"])


def test_lightcurve_rows_carry_the_density(monkeypatch):
    from .. import extract_feats

    duration = central_duration(EARTH_PERIOD, EARTH_A_OVER_RSUN, EARTH_DEPTH)
    monkeypatch.setattr(
        extract_feats,
        "extract_features_from_arrays",
        lambda *a, **k: [
            {
                "period_days": EARTH_PERIOD,
                "duration_days": duration,
                "depth_mean_per_transit": EARTH_DEPTH,
            }
        ],
    )
    time = np.arange(0.0, 10.0, 0.02)
    lc = lk.LightCurve(time=time, flux=np.ones_like(time))
    lc.meta.update({"TEFF": 5772.0, "LOGG": SUN_LOGG, "RADIUS": 1.0})
    row = extract_feats.extract_features_from_lightcurve(lc)[0]
    assert row["log_rho_ratio"] == pytest.approx(0.0, abs=0.01)


def write_kepler_file(directory, kepid, logg, radius):
    header = fits.Header({"OBJECT": f"KIC {kepid}", "LOGG": logg, "RADIUS": radius})
    path = directory / f"kplr{kepid:09d}-2009350155506_llc.fits"
    fits.PrimaryHDU(header=header).writeto(path)
    return path


def test_script_fills_every_csv_from_the_cached_header(tmp_path):
    cache = tmp_path / "cache" / "mastDownload" / "Kepler" / "kplr000000042_lc_Q111"
    cache.mkdir(parents=True)
    write_kepler_file(cache, 42, SUN_LOGG, 1.0)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    duration = central_duration(EARTH_PERIOD, EARTH_A_OVER_RSUN, EARTH_DEPTH)
    rows = pd.DataFrame(
        {
            "period_days": [EARTH_PERIOD, 10.0],
            "duration_days": [duration, 0.1],
            "depth_mean_per_transit": [EARTH_DEPTH, -1e-3],
            "candidate_label": ["CONFIRMED", "FALSE-POSITIVE"],
        }
    )
    rows.to_csv(extracted / "KIC42_20261020.csv", index=False)
    rows.to_csv(extracted / "KIC42_inverted_20261020.csv", index=False)
    (extracted / "KIC42_inverted_20261021.csv").write_text("\n")

    status = add_stellar_density.main(
        [str(extracted), "--cache-dir", str(tmp_path / "cache"), "--no-download"]
    )

    assert status == 0
    for name in ("KIC42_20261020.csv", "KIC42_inverted_20261020.csv"):
        out = pd.read_csv(extracted / name)
        assert list(out.columns) == list(rows.columns) + list(STELLAR_DENSITY_COLUMNS)
        pd.testing.assert_frame_equal(out[rows.columns], rows)
        assert out["log_rho_ratio"][0] == pytest.approx(0.0, abs=0.01)
        assert np.isnan(out["log_rho_ratio"][1])
    assert (extracted / "KIC42_inverted_20261021.csv").read_text() == "\n"


def test_script_reports_a_star_it_cannot_find(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    pd.DataFrame({"period_days": [10.0]}).to_csv(extracted / "KIC7_20261020.csv", index=False)
    status = add_stellar_density.main(
        [str(extracted), "--cache-dir", str(tmp_path), "--no-download"]
    )
    assert status == 1
    assert "log_rho_ratio" not in pd.read_csv(extracted / "KIC7_20261020.csv")


def test_folds_keep_a_star_together(tmp_path):
    for kepid in range(1, 15):
        for suffix in ("", "_inverted"):
            (tmp_path / f"KIC{kepid}{suffix}_20261020.csv").write_text("a\n1\n")
    folds, skipped = add_stellar_density.split_into_folds(tmp_path, 7)
    assert skipped == []
    assert [len(fold) for fold in folds] == [2] * 7
    assert all(len(paths) == 2 for fold in folds for _star, paths in fold)
