"""Secondary-eclipse and V-shape features: the eclipsing-binary discriminants."""

from __future__ import annotations

import contextlib
import io
from unittest.mock import patch

import numpy as np
import pytest

from ..cdpp import duration_matched_statistics
from ..detrend_and_period import detrend_with_bls_mask
from ..extract_feats import extract_features_from_arrays, measure_eb_checks
from ..sesmes import (
    EB_CHECK_COLUMNS,
    SECONDARY_SCAN_EXCLUSION_DURATIONS,
    eb_check_features,
    odd_even_depth_significance,
    secondary_any_phase,
    secondary_eclipse,
)
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


def _alternating_light_curve(seed, odd_depth, even_depth, period=PERIOD, span=200.0):
    """Every other event shallower: a binary found at half its period."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, CADENCE_DAYS)
    flux = 1.0 + rng.normal(0.0, NOISE, time.size)
    cycle = np.floor((time - EPOCH) / period + 0.5)
    dt = time - (EPOCH + cycle * period)
    depth = np.where(np.mod(cycle, 2) == 1, odd_depth, even_depth)
    flux -= depth * _trapezoid(dt, DURATION, 0.05 * DURATION)
    return time, flux, np.abs(dt) < 0.5 * DURATION


def _eccentric_light_curve(seed, secondary_phase, secondary_depth):
    time, flux, mask = _light_curve(seed)
    dt = np.mod(time - EPOCH - secondary_phase * PERIOD + 0.5 * PERIOD, PERIOD)
    flux -= secondary_depth * _trapezoid(dt - 0.5 * PERIOD, DURATION, 0.05 * DURATION)
    return time, flux, mask


def test_odd_even_significance_flags_alternating_depths():
    time, flux, mask = _alternating_light_curve(10, 1.6e-3, 2.0e-3)
    sigma = odd_even_depth_significance(_statistics(time, flux, mask), PERIOD, EPOCH)
    assert sigma > 10


def test_odd_even_significance_is_small_for_equal_depths():
    sigmas = []
    for seed in range(10):
        time, flux, mask = _alternating_light_curve(seed, 2e-3, 2e-3)
        sigmas.append(
            odd_even_depth_significance(_statistics(time, flux, mask), PERIOD, EPOCH)
        )
    assert min(sigmas) >= 0
    assert np.median(sigmas) < 1.5
    assert max(sigmas) < 4.0


def test_odd_even_significance_needs_two_events_per_half():
    # Events at 1.3, 61.3 and 121.3 d: two even, one odd.
    time, flux, mask = _light_curve(11, period=60.0, span=150.0)
    stats = _statistics(time, flux, mask)
    assert np.isnan(odd_even_depth_significance(stats, 60.0, EPOCH))
    assert np.isnan(odd_even_depth_significance(None, PERIOD, EPOCH))


def test_secondary_any_phase_finds_an_eccentric_secondary():
    time, flux, mask = _eccentric_light_curve(12, 0.3, 8e-4)
    stats = _statistics(time, flux, mask)
    scan = secondary_any_phase(stats, PERIOD, EPOCH, DURATION)
    assert scan["snr"] > 30
    assert scan["phase"] == pytest.approx(0.3, abs=0.01)
    assert scan["depth_ratio"] == pytest.approx(0.4, rel=0.15)
    # The fixed phase-0.5 test cannot see it.
    assert abs(secondary_eclipse(stats, PERIOD, EPOCH)["snr"]) < 5


def test_secondary_any_phase_stays_low_without_a_secondary():
    snrs = []
    for seed in range(10):
        time, flux, mask = _light_curve(seed)
        scan = secondary_any_phase(_statistics(time, flux, mask), PERIOD, EPOCH)
        snrs.append(scan["snr"])
        # The primary itself is never scanned.
        exclusion = SECONDARY_SCAN_EXCLUSION_DURATIONS * DURATION / PERIOD
        assert exclusion - 1e-9 <= scan["phase"] <= 1 - exclusion + 1e-9
    assert max(snrs) < 5.5


def test_secondary_any_phase_counts_its_trials():
    time, flux, mask = _light_curve(13)
    scan = secondary_any_phase(_statistics(time, flux, mask), PERIOD, EPOCH)
    expected = np.arange(
        3 * DURATION, PERIOD - 3 * DURATION + 1e-9, 0.25 * DURATION
    ).size
    assert scan["trials"] == expected
    # Nothing left to scan once the exclusion covers the orbit.
    short = secondary_any_phase(_statistics(time, flux, mask), 1.0, EPOCH, DURATION)
    assert short["trials"] == 0 and np.isnan(short["snr"])


def test_eb_check_features_are_nan_when_not_computable():
    values = eb_check_features(None, PERIOD, EPOCH, DURATION)
    assert set(values) == set(EB_CHECK_COLUMNS)
    assert values["secondary_scan_trials"] == 0
    for column in EB_CHECK_COLUMNS:
        if column != "secondary_scan_trials":
            assert np.isnan(values[column])


def test_extracted_rows_carry_the_eb_checks():
    row = _extract(*_alternating_light_curve(14, 1.6e-3, 2.0e-3))
    assert row["even_odd_depth_diff_sigma"] > 10
    row = _extract(*_eccentric_light_curve(15, 0.3, 8e-4))
    assert row["secondary_max_snr_any_phase"] > 30
    assert row["secondary_phase"] == pytest.approx(0.3, abs=0.01)


def test_measure_eb_checks_matches_the_pipeline_columns():
    time, flux, _ = _eccentric_light_curve(16, 0.3, 8e-4)
    with contextlib.redirect_stdout(io.StringIO()):
        values = measure_eb_checks(time, flux, PERIOD, EPOCH, DURATION)
    assert values["secondary_max_snr_any_phase"] > 30
    assert values["secondary_phase"] == pytest.approx(0.3, abs=0.01)
    assert values["even_odd_depth_diff_sigma"] < 4.0


def test_measure_eb_checks_rejects_unusable_ephemerides():
    time, flux, _ = _light_curve(17)
    values = measure_eb_checks(time, flux, np.nan, EPOCH, DURATION)
    assert np.isnan(values["even_odd_depth_diff_sigma"])
    assert values["secondary_scan_trials"] == 0


def test_fixed_ephemeris_keeps_the_hint():
    time, flux, _ = _light_curve(18)
    hint = {"period": PERIOD * 1.0004, "t0": EPOCH + 0.01, "duration": 0.9 * DURATION}
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, info = detrend_with_bls_mask(
            time, flux, candidate_hint=hint, refine_duration=False,
            dealias=False, fixed_ephemeris=True,
        )
    assert info["best_period"] == hint["period"]
    assert info["t0"] == hint["t0"]
    assert info["best_duration"] == hint["duration"]
    with pytest.raises(ValueError):
        detrend_with_bls_mask(time, flux, fixed_ephemeris=True)
