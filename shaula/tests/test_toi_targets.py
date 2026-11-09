"""TOI reference tables, the TESS target list and their labels."""

from __future__ import annotations

import pandas as pd
import pytest

from ..scripts.build_toi_targets import (
    GROUP_CONFIRMED_HOST,
    GROUP_TOI_FALSE_POSITIVE,
    build_targets,
    false_positive_tids,
    planet_host_tids,
    toi_name,
    toi_rows_to_reference,
)
from ..scripts.build_toi_targets import main as build_targets_main
from ..scripts.extract_multiple_stars import load_targets
from ..scripts.get_literature_data import LiteratureDataError
from ..utils.ephemeris import align_catalog_epoch
from ..utils.label_candidates import label_candidate_rows


def _toi(tid, toi, period, disposition, **extra):
    row = {
        "tid": tid,
        "toi": toi,
        "tfopwg_disp": disposition,
        "pl_orbper": period,
        "pl_tranmid": 2459000.5,
        "pl_trandurh": 2.4,
        "pl_trandep": 1500.0,
        "pl_rade": 2.0,
        "st_rad": 0.5,
        "st_tmag": 10.0,
    }
    row.update(extra)
    return row


# 100: two planets and a candidate. 200: one planet, one false positive.
# 300: a false alarm. 400: a blended false positive. 500: a candidate only.
TOI_TABLE = pd.DataFrame(
    [
        _toi(100, 270.01, 5.66, "CP"),
        _toi(100, 270.02, 11.38, "KP"),
        _toi(100, 270.03, 3.36, "PC"),
        _toi(200, 500.01, 4.0, "CP"),
        _toi(200, 500.02, 9.0, "FP"),
        _toi(300, 600.01, 7.0, "FA"),
        _toi(400, 700.01, 2.5, "FP"),
        _toi(500, 800.01, 6.0, "APC"),
    ]
)


def _reference(tid):
    return toi_rows_to_reference(TOI_TABLE[TOI_TABLE["tid"] == tid])


def _label(period, tid):
    rows = [{"period_days": period, "t0": 1.0, "duration_days": 0.1}]
    return label_candidate_rows(rows, _reference(tid))[0]


def test_toi_numbers_become_names():
    assert toi_name(270.01) == "TOI-270.01"
    assert toi_name("1234.1") == "TOI-1234.10"


def test_reference_maps_dispositions_and_units():
    reference = _reference(100)
    assert list(reference["target"]) == ["TOI-270.03", "TOI-270.01", "TOI-270.02"]
    assert list(reference["disposition"]) == ["CANDIDATE", "CONFIRMED", "CONFIRMED"]
    assert set(reference["catalog"]) == {"toi"}
    planet = reference.iloc[1]
    assert planet["duration_days"] == pytest.approx(0.1)
    assert planet["depth_mean_per_transit"] == pytest.approx(0.0015)
    assert planet["planet_radius_rjup"] == pytest.approx(2.0 / 11.209)
    assert planet["t0"] == 2459000.5


def test_reference_flags_false_alarms_and_false_positives():
    assert _reference(300).iloc[0][["disposition", "fp_flags"]].tolist() == [
        "FALSE POSITIVE",
        "fa",
    ]
    assert _reference(400).iloc[0][["disposition", "fp_flags"]].tolist() == [
        "FALSE POSITIVE",
        "fp",
    ]


def test_single_transit_toi_pairs_with_nothing():
    reference = toi_rows_to_reference(pd.DataFrame([_toi(900, 950.01, 0.0, "PC")]))
    assert pd.isna(reference.iloc[0]["period_days"])


def test_unknown_disposition_is_left_blank():
    reference = toi_rows_to_reference(pd.DataFrame([_toi(900, 950.01, 3.0, None)]))
    assert reference.iloc[0]["disposition"] == ""
    row = label_candidate_rows([{"period_days": 3.0}], reference)[0]
    assert row["candidate_label"] == "CONFIRMED"  # same as a legacy confirmed CSV


def test_empty_table_gives_an_empty_reference():
    assert toi_rows_to_reference(TOI_TABLE.iloc[0:0]).empty


def test_a_row_on_a_confirmed_or_known_planet_is_confirmed():
    for period, name in ((5.66, "TOI-270.01"), (11.38, "TOI-270.02")):
        row = _label(period, 100)
        assert row["candidate_label"] == "CONFIRMED"
        assert row["matched_target"] == name
        assert row["fp_source"] == ""


def test_a_row_on_a_planet_candidate_is_unknown():
    assert _label(3.36, 100)["candidate_label"] == "UNKNOWN"
    assert _label(6.0, 500)["candidate_label"] == "UNKNOWN"


def test_a_row_on_a_false_alarm_is_a_false_positive_from_the_toi_table():
    row = _label(7.0, 300)
    assert row["candidate_label"] == "FALSE-POSITIVE"
    assert row["fp_source"] == "toi:fa"


def test_a_row_on_a_plain_false_positive_is_unknown():
    row = _label(2.5, 400)
    assert row["candidate_label"] == "UNKNOWN"
    assert row["matched_target"] == "TOI-700.01"
    assert row["fp_source"] == ""


def test_an_unmatched_row_is_a_false_positive_from_the_pipeline():
    row = _label(1.2345, 100)
    assert row["candidate_label"] == "FALSE-POSITIVE"
    assert row["fp_source"] == "unmatched"


def test_toi_epochs_align_to_btjd():
    aligned = align_catalog_epoch(2459000.5, 2001.0, 5.0, target="TOI-270.01")
    assert aligned == pytest.approx(2000.5)


def test_host_and_false_positive_stars_are_told_apart():
    assert planet_host_tids(TOI_TABLE) == [100, 200]
    assert planet_host_tids(TOI_TABLE, min_planets=2) == [100]
    # 200 has a false positive but hosts a planet, so it is a host only.
    assert false_positive_tids(TOI_TABLE) == [300, 400]


def test_build_targets_writes_one_reference_per_star(tmp_path):
    targets = build_targets(
        TOI_TABLE,
        host_limit=None,
        fp_limit=None,
        min_planets=1,
        seed=0,
        toi_dir=tmp_path,
    )
    assert list(targets["target"]) == ["TIC 100", "TIC 200", "TIC 300", "TIC 400"]
    assert set(targets["mission"]) == {"TESS"}
    assert list(targets["group"]) == [GROUP_CONFIRMED_HOST] * 2 + [
        GROUP_TOI_FALSE_POSITIVE
    ] * 2
    assert targets.iloc[0]["name"] == "TOI-270.03, TOI-270.01, TOI-270.02"
    written = pd.read_csv(tmp_path / "TIC200-toi.csv")
    assert list(written["target"]) == ["TOI-500.01", "TOI-500.02"]


def test_sampling_is_seeded_and_limited(tmp_path):
    def run(seed):
        return build_targets(
            TOI_TABLE,
            host_limit=1,
            fp_limit=1,
            min_planets=1,
            seed=seed,
            toi_dir=tmp_path,
        )

    assert run(3).equals(run(3))
    assert list(run(3)["group"]) == [GROUP_CONFIRMED_HOST, GROUP_TOI_FALSE_POSITIVE]
    assert build_targets(
        TOI_TABLE, host_limit=0, fp_limit=0, min_planets=1, seed=0, toi_dir=tmp_path
    ).empty


def test_main_writes_a_target_list_the_batch_can_read(tmp_path):
    out = tmp_path / "targets" / "tess_targets.csv"
    code = build_targets_main(
        ["--host-limit", "-1", "--fp-limit", "-1", "--toi-dir", str(tmp_path / "toi"),
         "--out", str(out)],
        fetch=lambda: TOI_TABLE,
    )
    assert code == 0
    stars = load_targets(out)
    assert [star.name for star in stars] == ["TIC 100", "TIC 200", "TIC 300", "TIC 400"]
    assert {star.mission for star in stars} == {"TESS"}
    assert all(star.reference.endswith("-toi.csv") for star in stars)


def test_main_reports_a_failed_query(tmp_path, capsys):
    def failing():
        raise LiteratureDataError("Failed to query the TOI table: boom")

    code = build_targets_main(["--out", str(tmp_path / "t.csv")], fetch=failing)
    assert code == 1
    assert "boom" in capsys.readouterr().err
