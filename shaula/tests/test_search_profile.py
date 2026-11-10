"""Per-mission search profiles: event count, period baseline and binning."""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from .. import api, detrend_and_period, extract_feats
from ..api import extract
from ..detrend_and_period import (
    STAGE1_BIN_MIN_POINTS,
    detrend_with_bls_mask,
    observed_baseline_days,
    search_baseline_days,
)
from ..extract_feats import (
    _candidate_passes_mes,
    _harmonic_trials,
    bin_for_features,
    extract_features_from_arrays,
)
from ..search_profile import (
    DEFAULT_PROFILE,
    KEPLER_LONG_CADENCE_DAYS,
    TESS_PROFILE,
    SearchProfile,
    profile_for_mission,
)


def _two_sectors(cadence=0.02, gap=700.0):
    """Two 27-day stretches of data ``gap`` days apart."""
    first = np.arange(0.0, 27.0, cadence)
    return np.concatenate([first, first + gap])


class _StopAfterGrid(Exception):
    pass


def _captured_search(monkeypatch, time, **kwargs):
    """Run the search up to its global BLS and report what it was given."""
    captured = {}

    def fake_bls(bls_time, bls_flux):
        captured.setdefault("sizes", []).append(int(np.size(bls_time)))

        def power(periods, durations, **_):
            captured["max_period"] = float(np.max(periods))
            raise _StopAfterGrid

        return SimpleNamespace(power=power)

    monkeypatch.setattr(detrend_and_period, "BoxLeastSquares", fake_bls)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_StopAfterGrid):
        detrend_with_bls_mask(time, np.ones(time.size), **kwargs)
    return captured


def test_default_profile_is_the_kepler_rule():
    assert SearchProfile(3, "span", None) == DEFAULT_PROFILE
    for mission in ("Kepler", "K2", "kepler", None, "unknown"):
        assert profile_for_mission(mission) is DEFAULT_PROFILE


def test_tess_profile_takes_two_transits_over_observed_time():
    assert profile_for_mission(" tess ") is TESS_PROFILE
    assert TESS_PROFILE.min_transit_events == 2
    assert TESS_PROFILE.period_baseline == "observed"


def test_profile_rejects_unusable_settings():
    with pytest.raises(ValueError, match="at least 2"):
        SearchProfile(min_transit_events=1)
    with pytest.raises(ValueError, match="period_baseline"):
        SearchProfile(period_baseline="sectors")


def test_observed_baseline_leaves_gaps_out():
    time = _two_sectors()
    assert search_baseline_days(time, "span") == pytest.approx(727.0, abs=0.05)
    assert observed_baseline_days(time) == pytest.approx(54.0, abs=0.1)
    assert search_baseline_days(time, "observed") == observed_baseline_days(time)


def test_observed_baseline_bridges_cadence_sized_holes():
    time = np.arange(0.0, 27.0, 0.02)
    time = np.delete(time, slice(100, 110))  # 0.2 d of flagged cadences
    assert observed_baseline_days(time) == pytest.approx(27.0, abs=0.05)


def test_observed_baseline_of_almost_nothing_is_zero():
    assert observed_baseline_days(np.array([1.0])) == 0.0


def test_span_rule_searches_the_gap_and_observed_rule_does_not(monkeypatch):
    time = _two_sectors()
    by_span = _captured_search(monkeypatch, time)
    assert by_span["max_period"] == pytest.approx(727.0 / 3.0, abs=0.1)

    by_observed = _captured_search(
        monkeypatch, time, min_transit_events=2, period_baseline="observed"
    )
    assert by_observed["max_period"] == pytest.approx(54.0 / 2.0, abs=0.1)


def test_two_transit_rule_alone_reaches_half_the_span(monkeypatch):
    time = np.arange(0.0, 27.0, 0.02)
    captured = _captured_search(monkeypatch, time, min_transit_events=2)
    assert captured["max_period"] == pytest.approx(13.5, abs=0.05)


def test_long_fast_cadence_series_is_binned_for_the_global_search(monkeypatch):
    cadence = 2.0 / 1440.0
    time = np.arange(0.0, 100.0, cadence)
    assert time.size > STAGE1_BIN_MIN_POINTS
    captured = _captured_search(monkeypatch, time)
    full, stage1 = captured["sizes"][0], captured["sizes"][-1]
    assert full == time.size
    # 0.01 d shortest duration / 3 = 4.8 min bins on 2 min data.
    assert stage1 == pytest.approx(time.size / 2.4, rel=0.02)


def test_long_cadence_series_is_never_binned(monkeypatch):
    time = np.arange(0.0, 1460.0, 0.0204)
    assert time.size > STAGE1_BIN_MIN_POINTS
    captured = _captured_search(monkeypatch, time)
    assert captured["sizes"] == [time.size]


def test_gate_counts_events_against_the_profile():
    features = {"max_mes": 9.0}
    two_events = {"n_mes_events": 2}
    assert not _candidate_passes_mes(features, two_events)
    assert _candidate_passes_mes(features, two_events, min_events=2)
    assert not _candidate_passes_mes(features, {"n_mes_events": 1}, min_events=2)


def test_harmonic_trials_respect_the_event_count(monkeypatch):
    monkeypatch.setattr(
        extract_feats,
        "_score_harmonic",
        lambda statistics, period, epoch, duration, reference: (10.0, epoch, 2),
    )
    statistics = {"time": np.array([0.0])}
    # P = 6 d over a 27 d baseline: 2P = 12 d holds two transits, not three.
    kepler = {t[1] for t in _harmonic_trials(statistics, 6.0, 1.0, 0.1, 27.0)}
    tess = {t[1] for t in _harmonic_trials(statistics, 6.0, 1.0, 0.1, 27.0, 2)}
    assert 12.0 not in kepler
    assert 12.0 in tess
    assert 18.0 not in tess


@pytest.fixture
def recorded_profile(monkeypatch):
    calls = {}

    def fake_download(*args, **kwargs):
        return SimpleNamespace(meta={})

    def fake_features(lc, **kwargs):
        calls["search_profile"] = kwargs.get("search_profile")
        return []

    monkeypatch.setattr(api, "download_and_clean_lightcurve", fake_download)
    monkeypatch.setattr(api, "extract_features_from_lightcurve", fake_features)
    return calls


def test_extract_searches_tess_with_the_tess_profile(recorded_profile):
    extract("TIC 259377017", "tess")
    assert recorded_profile["search_profile"] is TESS_PROFILE


def test_extract_keeps_the_kepler_rule_for_kepler_and_k2(recorded_profile):
    for mission in ("Kepler", "K2"):
        extract("Kepler-11", mission)
        assert recorded_profile["search_profile"] is DEFAULT_PROFILE


def _one_sector_with_two_transits():
    rng = np.random.default_rng(7)
    time = np.arange(0.0, 26.0, 30.0 / 1440.0)
    flux = 1.0 + rng.normal(0.0, 3e-4, time.size)
    period, epoch, duration = 11.0, 4.0, 0.15
    in_transit = np.abs((time - epoch + 0.5 * period) % period - 0.5 * period) < (
        0.5 * duration
    )
    flux[in_transit] -= 0.004
    return time, flux, period


def test_two_transit_planet_in_one_sector_needs_the_tess_profile():
    time, flux, period = _one_sector_with_two_transits()
    with contextlib.redirect_stdout(io.StringIO()):
        kepler_rows = extract_features_from_arrays(time, flux)
        tess_rows = extract_features_from_arrays(
            time, flux, search_profile=TESS_PROFILE
        )

    def found(rows):
        return any(
            row["detection_status"] == "accepted"
            and abs(row["period_days"] / period - 1.0) < 0.01
            for row in rows
        )

    assert not found(kepler_rows)
    assert found(tess_rows)


def _two_minute_series(days=6.0):
    return np.arange(0.0, days, 2.0 / 1440.0)


def test_tess_features_are_measured_at_kepler_long_cadence():
    assert DEFAULT_PROFILE.feature_cadence_days is None
    assert TESS_PROFILE.feature_cadence_days == pytest.approx(29.4244 / 1440.0)
    assert TESS_PROFILE.feature_cadence_days == KEPLER_LONG_CADENCE_DAYS
    with pytest.raises(ValueError, match="feature_cadence_days"):
        SearchProfile(feature_cadence_days=0.0)


def test_binning_averages_to_an_even_grid():
    time = _two_minute_series()
    flux = 1.0 + 1e-3 * np.sin(time)
    gap = np.zeros(time.size, dtype=bool)
    binned_time, binned_flux, binned_gap = bin_for_features(
        time, flux, gap, KEPLER_LONG_CADENCE_DAYS
    )
    assert np.diff(binned_time) == pytest.approx(KEPLER_LONG_CADENCE_DAYS)
    assert binned_time.size == pytest.approx(time.size / 14.7, rel=0.02)
    assert not binned_gap.any()
    assert binned_flux == pytest.approx(1.0 + 1e-3 * np.sin(binned_time), abs=2e-5)


def test_binning_cuts_white_noise_by_the_root_of_the_bin_size():
    rng = np.random.default_rng(3)
    time = _two_minute_series(days=27.0)
    flux = 1.0 + rng.normal(0.0, 1e-3, time.size)
    _, binned_flux, _ = bin_for_features(
        time, flux, np.zeros(time.size, dtype=bool), KEPLER_LONG_CADENCE_DAYS
    )
    assert np.std(binned_flux) == pytest.approx(1e-3 / np.sqrt(14.7), rel=0.1)


def test_a_series_already_at_long_cadence_is_not_rebinned():
    for minutes in (29.4244, 30.0):
        time = np.arange(0.0, 10.0, minutes / 1440.0)
        gap = np.zeros(time.size, dtype=bool)
        assert bin_for_features(time, np.ones(time.size), gap, KEPLER_LONG_CADENCE_DAYS) is None


def test_ten_minute_ffi_data_is_rebinned():
    time = np.arange(0.0, 10.0, 10.0 / 1440.0)
    gap = np.zeros(time.size, dtype=bool)
    binned = bin_for_features(time, np.ones(time.size), gap, KEPLER_LONG_CADENCE_DAYS)
    assert binned is not None
    assert binned[0].size == pytest.approx(time.size / 2.94, rel=0.02)


def test_gapped_cadences_stay_out_of_a_bin_and_gap_it_when_none_are_left():
    time = _two_minute_series(days=1.0)
    flux = np.ones(time.size)
    gap = np.zeros(time.size, dtype=bool)
    # A dip already claimed by an accepted candidate, 3 h wide.
    claimed = (time > 0.4) & (time < 0.525)
    flux[claimed] = 0.99
    gap[claimed] = True
    _, binned_flux, binned_gap = bin_for_features(
        time, flux, gap, KEPLER_LONG_CADENCE_DAYS
    )
    assert binned_gap.sum() >= 5
    assert binned_flux[binned_gap] == pytest.approx(0.99)
    # Bins straddling the edge of the dip average their unclaimed cadences only.
    assert binned_flux[~binned_gap] == pytest.approx(1.0)


def test_thin_bins_at_the_edge_of_a_data_gap_are_dropped():
    time = _two_minute_series(days=1.0)
    # Leave 4 minutes of data stranded after a half-day hole.
    time = np.concatenate([time[time < 0.4], [0.9003, 0.9017]])
    flux = np.ones(time.size)
    binned_time, _, _ = bin_for_features(
        time, flux, np.zeros(time.size, dtype=bool), KEPLER_LONG_CADENCE_DAYS
    )
    assert binned_time.max() < 0.45


def test_tess_rows_report_the_feature_cadence_and_kepler_rows_the_native_one():
    rng = np.random.default_rng(11)
    time = np.arange(0.0, 12.0, 2.0 / 1440.0)
    flux = 1.0 + rng.normal(0.0, 1e-3, time.size)
    period, epoch, duration = 2.5, 1.0, 0.12
    in_transit = np.abs((time - epoch + 0.5 * period) % period - 0.5 * period) < (
        0.5 * duration
    )
    flux[in_transit] -= 0.004
    with contextlib.redirect_stdout(io.StringIO()):
        native, info, _ = extract_feats._extract_single_candidate_from_arrays(
            time, flux
        )
        binned, binned_info, _ = extract_feats._extract_single_candidate_from_arrays(
            time, flux, search_profile=TESS_PROFILE
        )
    assert native["cadence_hours"] == pytest.approx(2.0 / 60.0, rel=1e-3)
    assert binned["cadence_hours"] == pytest.approx(29.4244 / 60.0, rel=1e-3)
    assert binned["period_days"] == pytest.approx(period, rel=2e-3)
    assert binned["npts_transit_median"] < native["npts_transit_median"] / 5
    assert binned["depth_mean_per_transit"] == pytest.approx(0.004, rel=0.25)
    assert binned["max_mes"] > 7.1
    # The caller's deep-event sweep and masks still get the native series.
    assert binned_info["flux_detrended"].shape == time.shape
    assert binned_info["mask_transit"].shape == time.shape
    assert info["flux_detrended"].shape == time.shape


def _short_transit_series(duration_hours, period=0.948, depth=0.004):
    rng = np.random.default_rng(1)
    time = np.arange(0.0, 27.0, 2.0 / 1440.0)
    flux = 1.0 + rng.normal(0.0, 1e-3, time.size)
    phase = (time - 1.0 + 0.5 * period) % period - 0.5 * period
    flux[np.abs(phase) < duration_hours / 48.0] -= depth
    return time, flux


def test_transit_shorter_than_the_feature_bins_keeps_native_statistics():
    # TOI-736.02: 0.56 h transits on 29.4-minute bins read MES NaN, no events.
    time, flux = _short_transit_series(0.6)
    with contextlib.redirect_stdout(io.StringIO()):
        feats, info, _ = extract_feats._extract_single_candidate_from_arrays(
            time, flux, search_profile=TESS_PROFILE
        )
    assert feats["duration_days"] < (
        extract_feats.SHORT_TRANSIT_FEATURE_BINS * KEPLER_LONG_CADENCE_DAYS
    )
    assert feats["period_days"] == pytest.approx(0.948, rel=2e-3)
    assert feats["max_mes"] > 7.1
    assert info["n_mes_events"] >= 20
    assert np.isfinite(feats["even_odd_depth_diff_sigma"])
    assert extract_feats._candidate_passes_mes(feats, info, min_events=2)
    # The cadence-dependent features still come from the binned series.
    assert feats["cadence_hours"] == pytest.approx(29.4244 / 60.0, rel=1e-3)
    assert info["sesmes_statistics"]["time"].size == time.size


def test_long_transit_statistics_stay_on_the_feature_series():
    time, flux = _short_transit_series(3.0, period=4.1)
    with contextlib.redirect_stdout(io.StringIO()):
        feats, info, _ = extract_feats._extract_single_candidate_from_arrays(
            time, flux, search_profile=TESS_PROFILE
        )
    assert feats["max_mes"] > 7.1
    assert info["sesmes_statistics"]["time"].size < time.size / 5


def test_gate_rejects_a_brightening():
    passing = {"max_mes": 20.0, "depth_mean_per_transit": 1e-3}
    assert extract_feats._candidate_passes_mes(passing)
    assert not extract_feats._candidate_passes_mes(
        {"max_mes": 20.0, "depth_mean_per_transit": -7e-3}
    )
    assert not extract_feats._candidate_passes_mes(
        {"max_mes": 20.0, "depth_mean_per_transit": 0.0}
    )
    # An unmeasurable depth is not evidence against the candidate.
    assert extract_feats._candidate_passes_mes(
        {"max_mes": 20.0, "depth_mean_per_transit": float("nan")}
    )
    assert extract_feats._candidate_passes_mes({"max_mes": 20.0})


def test_ingress_egress_asymmetry_is_nan_when_no_point_falls_in_transit():
    from shaula.utils.compute_ingress_egress_asymmetry import (
        compute_ingress_egress_asymmetry,
    )

    # Period commensurate with the cadence, epoch between samples: no cadence
    # ever lands within 1.5 x T14 of mid-transit.
    time = np.arange(0.0, 20.0, 0.02)
    flux = np.ones(time.size)
    value = compute_ingress_egress_asymmetry(time, flux, 2.0, 0.01, 0.001)
    assert np.isnan(value)
