"""The public extract() entrypoint and its progress contract."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from .. import api
from ..api import MAX_QUERY_LENGTH, STAGES, ExtractionResult, ProgressEvent, extract
from ..extract_feats import extract_features_from_arrays


class _FakeLightCurve:
    """Stands in for a lightkurve LightCurve; extract() only passes it through."""

    meta: ClassVar[dict] = {}


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the two heavy calls with recorders."""
    calls = {}

    def fake_download(target, mission, sigma_upper=5.0, all=False, **kwargs):
        calls["download"] = {
            "target": target,
            "mission": mission,
            "sigma_upper": sigma_upper,
            "all": all,
            "author": kwargs.get("author"),
            "exptime": kwargs.get("exptime"),
            "cache_dir": kwargs.get("cache_dir"),
        }
        return _FakeLightCurve()

    def fake_features(lc, **kwargs):
        calls["features"] = kwargs
        progress = kwargs.get("progress")
        if progress is not None:
            progress(ProgressEvent(stage="period_search", message="candidate 1"))
        return [{"period_days": 3.5, "MES": 12.0}]

    monkeypatch.setattr(api, "download_and_clean_lightcurve", fake_download)
    monkeypatch.setattr(api, "extract_features_from_lightcurve", fake_features)
    return calls


def test_returns_features_and_version(stubbed):
    result = extract("Kepler-11", "Kepler")
    assert isinstance(result, ExtractionResult)
    assert result.features == [{"period_days": 3.5, "MES": 12.0}]
    assert result.target == "Kepler-11"
    assert result.mission == "Kepler"
    assert result.shaula_version


def test_works_without_a_progress_callback(stubbed):
    result = extract("Kepler-11", "Kepler", progress=None)
    assert len(result.features) == 1


def test_emits_known_stages_in_order_ending_with_done(stubbed):
    seen: list[ProgressEvent] = []
    extract("Kepler-11", "Kepler", progress=seen.append)

    assert seen, "extract() emitted no progress events"
    assert all(e.stage in STAGES for e in seen)
    assert seen[-1].stage == "done"

    order = [STAGES.index(e.stage) for e in seen]
    assert order == sorted(order), f"stages went backwards: {[e.stage for e in seen]}"


def test_every_declared_stage_is_actually_emitted(stubbed):
    """STAGES is a contract: a stage that never fires is a lie to the caller."""
    seen: list[ProgressEvent] = []
    extract("Kepler-11", "Kepler", progress=seen.append)

    emitted = {event.stage for event in seen}
    never_fired = set(STAGES) - emitted
    assert never_fired == set(), f"declared but never emitted: {sorted(never_fired)}"


def test_callback_exception_propagates_unchanged(stubbed):
    class Cancelled(Exception):
        pass

    def cancel(event: ProgressEvent) -> None:
        raise Cancelled(event.stage)

    with pytest.raises(Cancelled):
        extract("Kepler-11", "Kepler", progress=cancel)


def test_download_failure_surfaces_with_its_message(monkeypatch):
    def failing_download(*args, **kwargs):
        raise ValueError("No Kepler light curves found for 'Nope'")

    monkeypatch.setattr(api, "download_and_clean_lightcurve", failing_download)

    with pytest.raises(ValueError, match="No Kepler light curves found"):
        extract("Nope", "Kepler")


def test_passes_parameters_through_to_the_downloader(stubbed, monkeypatch, tmp_path):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    extract(
        "Kepler-11",
        "TESS",
        sigma_clip=3.0,
        download_all=True,
        author="QLP",
        exptime=120,
        use_tls=True,
        mask_eclipses=True,
        cache_dir=tmp_path,
    )
    assert stubbed["download"]["mission"] == "TESS"
    assert stubbed["download"]["sigma_upper"] == 3.0
    assert stubbed["download"]["all"] is True
    assert stubbed["download"]["author"] == "QLP"
    assert stubbed["download"]["exptime"] == 120
    assert stubbed["download"]["cache_dir"] == tmp_path
    assert stubbed["features"]["use_tls"] is True
    assert stubbed["features"]["mask_eclipses"] is True


def test_period_search_events_fire_during_extraction(monkeypatch):
    """The callback must fire inside extraction, not only at its boundaries."""

    def fake_download(*args, **kwargs):
        return _FakeLightCurve()

    def fake_features(lc, progress=None, **kwargs):
        for index in range(3):
            progress(
                ProgressEvent(
                    stage="period_search",
                    message=f"candidate {index}",
                )
            )
        return [{"period_days": 1.0}]

    monkeypatch.setattr(api, "download_and_clean_lightcurve", fake_download)
    monkeypatch.setattr(api, "extract_features_from_lightcurve", fake_features)

    seen: list[ProgressEvent] = []
    extract("Kepler-11", "Kepler", progress=seen.append)

    searching = [e for e in seen if e.stage == "period_search"]
    assert len(searching) >= 3


def test_real_candidate_loop_emits_period_search_events():
    """Exercise the real hook in extract_features_from_arrays.

    Mirrors the pattern in test_detection_statistics: the BLS detrend is
    stubbed so the real candidate loop runs in well under a second.
    """
    cadence = 0.5 / 24.0
    time = np.arange(0.0, 60.0, cadence)
    flux = np.ones_like(time)
    period, epoch, duration = 5.0, 2.0, 4.0 / 24.0
    phase = np.mod(time - epoch + 0.5 * period, period) - 0.5 * period
    mask = np.abs(phase) <= duration / 2.0
    flux[mask] -= 2e-3
    bls_info = {"best_period": period, "t0": epoch, "best_duration": duration}

    seen: list[ProgressEvent] = []
    with (
        patch(
            "shaula.extract_feats.detrend_with_bls_mask",
            return_value=(flux, np.ones_like(flux), mask, bls_info),
        ),
        patch("shaula.extract_feats.MAX_TRANSIT_CANDIDATES", 1),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        extract_features_from_arrays(time, flux, progress=seen.append)

    searching = [e for e in seen if e.stage == "period_search"]
    assert searching, "no period_search event from the real candidate loop"
    assert "1" in searching[0].message


def test_planet_letter_is_stripped_before_the_downloader(stubbed):
    extract("Kepler-186f", "Kepler")
    assert stubbed["download"]["target"] == "Kepler-186"


def test_mission_is_canonicalised_for_the_downloader(stubbed):
    result = extract("TOI-178", "tess")
    assert stubbed["download"]["mission"] == "TESS"
    assert result.mission == "TESS"


def test_unsupported_mission_is_rejected_before_downloading(stubbed):
    with pytest.raises(ValueError, match="Hubble"):
        extract("Kepler-11", "Hubble")
    assert "download" not in stubbed


def test_overlong_query_is_rejected_before_any_download(stubbed):
    with pytest.raises(ValueError, match=str(MAX_QUERY_LENGTH)):
        extract("x" * (MAX_QUERY_LENGTH + 1), "Kepler")
    assert "download" not in stubbed


def test_query_at_the_limit_is_accepted(stubbed):
    extract("x" * MAX_QUERY_LENGTH, "Kepler")
    assert "download" in stubbed


def test_use_tls_without_the_extra_fails_before_downloading(stubbed, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    with pytest.raises(RuntimeError, match="tls"):
        extract("Kepler-11", "Kepler", use_tls=True)
    assert "download" not in stubbed


def test_tls_check_is_skipped_when_use_tls_is_false(stubbed, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("find_spec must not be called")

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert extract("Kepler-11", "Kepler", use_tls=False).features


def test_use_tls_proceeds_when_the_extra_is_installed(stubbed, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    assert extract("Kepler-11", "Kepler", use_tls=True).features
    assert "download" in stubbed
