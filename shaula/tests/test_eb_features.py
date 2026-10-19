"""Secondary-eclipse and V-shape features: the eclipsing-binary discriminants."""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pytest

from ..cdpp import duration_matched_statistics
from ..extract_feats import extract_features_from_arrays
from ..sesmes import secondary_eclipse
from ..utils import compute_vshape

CADENCE_DAYS = 0.0204  # Kepler long cadence
PERIOD = 5.0
EPOCH = 1.3
DURATION = 0.2
NOISE = 2e-4


def _trapezoid(dt, total, ingress):
    half = 0.5 * total
    if ingress <= 0:
        return (np.abs(dt) < half).astype(float)
    return np.clip((half - np.abs(dt)) / ingress, 0.0, 1.0)


def _light_curve(
    seed,
    depth=2e-3,
    ingress_fraction=0.1,
    secondary_depth=0.0,
    period=PERIOD,
    span=200.0,
):
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, CADENCE_DAYS)
    flux = 1.0 + rng.normal(0.0, NOISE, time.size)
    dt = np.mod(time - EPOCH + 0.5 * period, period) - 0.5 * period
    flux -= depth * _trapezoid(dt, DURATION, ingress_fraction * 0.5 * DURATION)
    dt_secondary = np.mod(time - EPOCH, period) - 0.5 * period
    flux -= secondary_depth * _trapezoid(dt_secondary, DURATION, 0.05 * DURATION)
    transit_mask = np.abs(dt) < 0.5 * DURATION
    return time, flux, transit_mask


def _statistics(time, flux, transit_mask):
    return duration_matched_statistics(
        time,
        flux,
        duration_hours=DURATION * 24.0,
        cadence_hours=CADENCE_DAYS * 24.0,
        exclude_mask=transit_mask,
    )


def test_secondary_eclipse_recovers_an_injected_secondary():
    time, flux, mask = _light_curve(1, secondary_depth=8e-4)
    result = secondary_eclipse(_statistics(time, flux, mask), PERIOD, EPOCH)
    assert result["depth"] == pytest.approx(8e-4, rel=0.1)
    assert result["snr"] > 30
    assert result["n_events"] >= 35


def test_secondary_eclipse_is_centred_on_zero_without_a_secondary():
    # The old 5th-percentile depth reported ~1.6 sigma of noise for every
    # candidate and an SNR that grew with point count.
    snrs = []
    for seed in range(20):
        time, flux, mask = _light_curve(seed)
        snrs.append(secondary_eclipse(_statistics(time, flux, mask), PERIOD, EPOCH)["snr"])
    snrs = np.asarray(snrs)
    assert abs(np.mean(snrs)) < 0.8
    assert np.max(np.abs(snrs)) < 5.0


def test_secondary_eclipse_handles_missing_inputs():
    assert np.isnan(secondary_eclipse(None, PERIOD, EPOCH)["snr"])
    time, flux, mask = _light_curve(0)
    stats = _statistics(time, flux, mask)
    assert np.isnan(secondary_eclipse(stats, np.nan, EPOCH)["depth"])


@pytest.mark.parametrize(
    ("ingress_fraction", "low", "high"),
    [(0.0, 0.0, 0.15), (0.5, 0.35, 0.65), (1.0, 0.85, 1.0)],
)
def test_vshape_measures_the_ingress_fraction(ingress_fraction, low, high):
    time, flux, _ = _light_curve(3, ingress_fraction=ingress_fraction)
    value = compute_vshape(
        time, flux, PERIOD, EPOCH, DURATION, cadence_days=CADENCE_DAYS
    )
    assert low <= value <= high


def test_vshape_tolerates_an_overestimated_duration():
    # Extracted T14 runs long; a fixed-width fit would read the excess as ingress.
    time, flux, _ = _light_curve(4, ingress_fraction=0.1)
    value = compute_vshape(
        time, flux, PERIOD, EPOCH, 1.25 * DURATION, cadence_days=CADENCE_DAYS
    )
    assert value < 0.25


def test_vshape_is_finite_at_long_periods():
    # 200 phase bins made a 300 d transit narrower than one bin, so the old
    # metric was NaN for almost every long-period candidate.
    period = 300.0
    time, flux, _ = _light_curve(5, period=period, span=1400.0)
    value = compute_vshape(
        time, flux, period, EPOCH, DURATION, cadence_days=CADENCE_DAYS
    )
    assert np.isfinite(value)
    assert value < 0.3


def test_vshape_rejects_unusable_inputs():
    time, flux, _ = _light_curve(6)
    assert np.isnan(compute_vshape(time, flux, np.nan, EPOCH, DURATION))
    assert np.isnan(compute_vshape(time, np.full_like(flux, np.nan), PERIOD, EPOCH, DURATION))


def _extract(time, flux, mask):
    bls_info = {"best_period": PERIOD, "t0": EPOCH, "best_duration": DURATION}
    with (
        patch(
            "shaula.extract_feats.detrend_with_bls_mask",
            return_value=(flux, np.ones_like(flux), mask, bls_info),
        ),
        patch("shaula.extract_feats.MAX_TRANSIT_CANDIDATES", 1),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return extract_features_from_arrays(time, flux)[0]


def test_extracted_secondary_columns_share_one_definition():
    row = _extract(*_light_curve(7, secondary_depth=8e-4))
    snr = row["secondary_depth_snr"]
    assert snr > 30
    assert row["secondary_depth"] == pytest.approx(8e-4, rel=0.1)
    assert row["secondary_depth_snr_log"] == pytest.approx(np.log1p(snr))
    assert row["secondary_depth_snr_capped"] == pytest.approx(min(snr, 100.0))
    assert row["vshape_metric"] < 0.25


def test_extracted_features_separate_a_planet_from_a_v_shaped_eclipse():
    planet = _extract(*_light_curve(8, ingress_fraction=0.1))
    grazing = _extract(*_light_curve(8, ingress_fraction=1.0))
    assert planet["vshape_metric"] < 0.25 < 0.85 < grazing["vshape_metric"]
    assert abs(planet["secondary_depth_snr"]) < 4.0
