"""KOI reference tables, the training target list and inverted light curves."""

from __future__ import annotations

from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
import pytest

from ..scripts.build_koi_targets import (
    GROUP_CONFIRMED_HOST,
    GROUP_KOI_FALSE_POSITIVE,
    build_targets,
    false_positive_kepids,
)
from ..scripts.build_koi_targets import (
    main as build_targets_main,
)
from ..scripts.extract_and_compare import (
    invert_lightcurve,
    known_signal_mask,
    output_stem,
    remove_known_signals,
)
from ..scripts.extract_multiple_stars import (
    Star,
    build_argv,
    has_existing_output,
    load_targets,
)
from ..scripts.get_literature_data import koi_rows_to_reference
from ..utils.label_candidates import (
    label_all_false_positive,
    label_candidate_rows,
)


def _koi(kepid, kepoi, period, disposition, kepler_name=None, **flags):
    return {
        "kepid": kepid,
        "kepoi_name": kepoi,
        "kepler_name": kepler_name,
        "koi_disposition": disposition,
        "koi_period": period,
        "koi_time0bk": 130.0,
        "koi_duration": 3.0,
        "koi_depth": 1000.0,
        "koi_prad": 2.0,
        "koi_fpflag_nt": flags.get("nt", 0),
        "koi_fpflag_ss": flags.get("ss", 0),
        "koi_fpflag_co": flags.get("co", 0),
        "koi_fpflag_ec": flags.get("ec", 0),
    }


KOI = pd.DataFrame(
    [
        # A confirmed host that also has a false positive KOI.
        _koi(100, "K00010.01", 4.0, "CONFIRMED", "Kepler-99 b"),
        _koi(100, "K00010.02", 9.0, "FALSE POSITIVE", ss=1),
        # Eclipsing-binary false positive.
        _koi(200, "K00020.01", 2.5, "FALSE POSITIVE", ss=1, co=1),
        # Not-transit-like false positive next to an open candidate.
        _koi(300, "K00030.01", 7.0, "FALSE POSITIVE", nt=1),
        _koi(300, "K00030.02", 15.0, "CANDIDATE"),
        # Rejected on pixel data only: invisible in the light curve.
        _koi(400, "K00040.01", 3.3, "FALSE POSITIVE", co=1),
        # Rejected by follow-up, with no Robovetter flag.
        _koi(500, "K00050.01", 6.6, "FALSE POSITIVE"),
    ]
)


def _reference(kepid):
    return koi_rows_to_reference(KOI[KOI["kepid"] == kepid])


# --- reference tables -------------------------------------------------------


def test_reference_keeps_every_koi_with_disposition_and_flags():
    ref = _reference(100)
    assert list(ref["target"]) == ["Kepler-99b", "K00010.02"]
    assert list(ref["disposition"]) == ["CONFIRMED", "FALSE POSITIVE"]
    assert list(ref["fp_flags"]) == ["", "ss"]
    assert _reference(200)["fp_flags"].iloc[0] == "ss+co"


# --- labelling --------------------------------------------------------------


def _label(period, kepid):
    return label_candidate_rows([{"period_days": period}], _reference(kepid))[0]


def test_a_row_on_a_flagged_false_positive_koi_is_a_false_positive():
    row = _label(2.5, 200)
    assert row["candidate_label"] == "FALSE-POSITIVE"
    assert row["matched_target"] == "K00020.01"
    assert row["fp_source"] == "koi:ss+co"


def test_an_alias_of_an_eclipsing_binary_is_still_a_false_positive():
    row = _label(1.25, 200)
    assert row["candidate_label"] == "FALSE-POSITIVE"
    assert row["matched_period_ratio"] == "P/2"


def test_a_planet_on_the_same_star_stays_confirmed():
    row = _label(4.0, 100)
    assert row["candidate_label"] == "CONFIRMED"
    assert row["fp_source"] == ""


def test_an_open_candidate_is_unknown():
    assert _label(15.0, 300)["candidate_label"] == "UNKNOWN"


@pytest.mark.parametrize("kepid, period", [(400, 3.3), (500, 6.6)])
def test_a_false_positive_invisible_in_the_light_curve_is_unknown(kepid, period):
    row = _label(period, kepid)
    assert row["candidate_label"] == "UNKNOWN"
    assert row["fp_source"] == ""


def test_an_unmatched_row_is_a_false_positive_from_the_pipeline():
    row = _label(13.7, 300)
    assert row["candidate_label"] == "FALSE-POSITIVE"
    assert row["fp_source"] == "unmatched"


def test_legacy_confirmed_csvs_label_as_before():
    confirmed = pd.DataFrame({"target": ["Kepler-99b"], "period_days": [4.0]})
    rows = label_candidate_rows(
        [{"period_days": 4.0}, {"period_days": 5.5}], confirmed
    )
    assert [r["candidate_label"] for r in rows] == ["CONFIRMED", "FALSE-POSITIVE"]
    assert [r["fp_source"] for r in rows] == ["", "unmatched"]


def test_inverted_rows_are_all_false_positives():
    rows = label_all_false_positive([{"period_days": 3.0}, {"period_days": 8.0}])
    assert {r["candidate_label"] for r in rows} == {"FALSE-POSITIVE"}
    assert {r["fp_source"] for r in rows} == {"inverted"}


# --- target list ------------------------------------------------------------


def test_only_light_curve_visible_false_positives_qualify_a_star():
    assert false_positive_kepids(KOI) == [100, 200, 300]


def test_build_targets_writes_references_and_skips_hosts_in_the_fp_sample(tmp_path):
    targets = build_targets(
        KOI, hosts=["Kepler-99"], fp_limit=None, seed=0, koi_dir=tmp_path
    )
    assert list(targets["target"]) == ["KIC 100", "KIC 200", "KIC 300"]
    assert list(targets["group"]) == [
        GROUP_CONFIRMED_HOST,
        GROUP_KOI_FALSE_POSITIVE,
        GROUP_KOI_FALSE_POSITIVE,
    ]
    assert targets["name"].iloc[0] == "Kepler-99"
    for path in targets["reference"]:
        assert Path(path).is_file()
    assert len(pd.read_csv(tmp_path / "KIC300-koi.csv")) == 2


def test_fp_sampling_is_seeded(tmp_path):
    first = build_targets(KOI, hosts=[], fp_limit=2, seed=7, koi_dir=tmp_path)
    second = build_targets(KOI, hosts=[], fp_limit=2, seed=7, koi_dir=tmp_path)
    assert len(first) == 2
    assert list(first["target"]) == list(second["target"])


def test_main_writes_a_target_list_the_batch_can_read(tmp_path):
    confirmed_dir = tmp_path / "confirmed"
    confirmed_dir.mkdir()
    (confirmed_dir / "Kepler-99-confirmed.csv").write_text("target,period_days\n")
    out = tmp_path / "targets.csv"
    code = build_targets_main(
        [
            "--fp-limit", "-1",
            "--confirmed-dir", str(confirmed_dir),
            "--koi-dir", str(tmp_path / "koi"),
            "--out", str(out),
        ],
        fetch=lambda: KOI,
    )
    assert code == 0
    stars = load_targets(out)
    assert [s.name for s in stars] == ["KIC 100", "KIC 200", "KIC 300"]
    assert stars[0].reference.endswith("KIC100-koi.csv")


# --- batch wiring -------------------------------------------------------------


def test_build_argv_passes_the_reference_and_invert(tmp_path):
    entry = Star("KIC 200", "koi_fp", reference="data/koi/KIC200-koi.csv")
    argv = build_argv(
        entry, "KIC 200", tmp_path, tmp_path, False, True, [], invert=True
    )
    assert argv[argv.index("--reference") + 1] == "data/koi/KIC200-koi.csv"
    assert "--invert" in argv


def test_existing_output_tells_inverted_runs_apart(tmp_path):
    (tmp_path / "KIC200_20261019.csv").write_text("x\n")
    assert has_existing_output("KIC 200", tmp_path)
    assert not has_existing_output("KIC 200", tmp_path, inverted=True)
    (tmp_path / "KIC200_inverted_20261019.csv").write_text("x\n")
    assert has_existing_output("KIC 200", tmp_path, inverted=True)


def test_output_stem_has_no_spaces():
    assert output_stem("KIC 200") == "KIC200"
    assert output_stem("Kepler-7", inverted=True) == "Kepler-7_inverted"


# --- inversion ----------------------------------------------------------------


def test_inversion_turns_a_transit_into_a_brightening():
    time = np.arange(0.0, 10.0, 0.02)
    flux = np.ones_like(time)
    flux[(time % 2.0) < 0.1] -= 0.01
    lc = lk.LightCurve(time=time, flux=flux)
    inverted = invert_lightcurve(lc)
    assert np.nanmax(inverted.flux.value) == pytest.approx(1.01)
    assert np.nanmin(inverted.flux.value) == pytest.approx(1.0)
    # The original is untouched.
    assert np.nanmin(lc.flux.value) == pytest.approx(0.99)


def test_known_signals_are_cut_at_transit_and_phase_half_in_mission_time():
    time = np.arange(0.0, 20.0, 0.02)
    # t0 in full BJD, as the reference tables store it; 102 BKJD = 2 mod 4.
    reference = pd.DataFrame(
        {"period_days": [4.0], "t0": [2454833.0 + 102.0], "duration_days": [0.1]}
    )
    mask = known_signal_mask(time, reference)
    assert mask[np.argmin(np.abs(time - 6.0))]
    assert mask[np.argmin(np.abs(time - 4.0))]  # phase 0.5
    assert not mask[np.argmin(np.abs(time - 5.0))]
    assert np.all(np.abs(time[mask] - np.round(time[mask] / 2.0) * 2.0) < 0.15 + 1e-9)


def test_rows_without_an_ephemeris_cut_nothing():
    time = np.arange(0.0, 10.0, 0.02)
    reference = pd.DataFrame({"period_days": [np.nan], "t0": [1.0], "duration_days": [0.1]})
    lc = lk.LightCurve(time=time, flux=np.ones_like(time))
    kept, removed = remove_known_signals(lc, reference)
    assert removed == 0
    assert len(kept) == len(lc)


# --- stellar radius -----------------------------------------------------------


@pytest.mark.parametrize("radius, expected", [(None, None), (1.0, 109.1 * 0.01)])
def test_planet_radius_tolerates_a_missing_stellar_radius(monkeypatch, radius, expected):
    # KIC stars with no catalogue radius carry RADIUS=None, which crashed the
    # radius step after a whole search.
    from .. import extract_feats

    monkeypatch.setattr(
        extract_feats,
        "extract_features_from_arrays",
        lambda *a, **k: [{"depth_mean_per_transit": 1e-4, "planet_radius_rearth": np.nan}],
    )
    time = np.arange(0.0, 10.0, 0.02)
    lc = lk.LightCurve(time=time, flux=np.ones_like(time))
    lc.meta["RADIUS"] = radius
    row = extract_feats.extract_features_from_lightcurve(lc)[0]
    if expected is None:
        assert np.isnan(row["planet_radius_rearth"])
    else:
        assert row["planet_radius_rearth"] == pytest.approx(expected)


def test_target_lists_store_repo_relative_references(tmp_path):
    from ..scripts.build_koi_targets import REPO_ROOT, portable_path

    inside = REPO_ROOT / "data" / "koi" / "KIC100-koi.csv"
    assert portable_path(inside) == "data/koi/KIC100-koi.csv"
    assert portable_path(tmp_path / "x.csv") == str(tmp_path / "x.csv")

    targets = tmp_path / "targets.csv"
    targets.write_text(
        "target,mission,group,name,reference\n"
        "KIC 100,Kepler,koi_fp,K1,data/koi/KIC100-koi.csv\n"
    )
    assert load_targets(targets)[0].reference == str(inside)
