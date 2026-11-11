"""Second look at rows labelled FALSE-POSITIVE only because nothing matched."""

from __future__ import annotations

import pandas as pd

from ..scripts.relabel_unmatched import (
    REASON_ALIAS,
    REASON_NEARBY,
    REASON_VIABLE,
    is_planet_viable,
    relabel_directory,
    relabel_unmatched_rows,
)


def _row(period, **extra):
    row = {
        "period_days": period,
        "depth_mean_per_transit": 0.001,
        "max_mes": 8.0,
        "max_ses": 5.0,
        "detection_status": "accepted",
        "log_rho_ratio": 0.1,
        "planet_radius_rjup": 0.3,
        "even_odd_depth_diff_sigma": 0.5,
        "secondary_depth_snr": 0.2,
        "candidate_label": "FALSE-POSITIVE",
        "matched_target": None,
        "matched_period_ratio": None,
        "fp_source": "unmatched",
    }
    row.update(extra)
    return row


def _reference(*rows):
    return pd.DataFrame(
        [
            {
                "target": target,
                "period_days": period,
                "depth_mean_per_transit": depth,
                "disposition": disposition,
                "fp_flags": flags,
                "catalog": "toi",
            }
            for target, period, depth, disposition, flags in rows
        ]
    )


def _relabel(rows, reference, inverted_best_mes=float("nan")):
    return relabel_unmatched_rows(pd.DataFrame(rows), reference, inverted_best_mes)


def test_weak_unmatched_row_stays_false_positive():
    out = _relabel([_row(7.3)], _reference(("TOI-1.01", 3.1, 0.001, "CONFIRMED", "")))
    assert out.loc[0, "candidate_label"] == "FALSE-POSITIVE"
    assert out.loc[0, "fp_source"] == "unmatched"
    assert out.loc[0, "relabel_reason"] == ""


def test_alias_of_unrecovered_planet_becomes_confirmed():
    out = _relabel([_row(4.0)], _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", "")))
    assert out.loc[0, "candidate_label"] == "CONFIRMED"
    assert out.loc[0, "matched_target"] == "TOI-1.01"
    assert abs(out.loc[0, "matched_period_ratio"] - 2 / 3) < 1e-9
    assert out.loc[0, "relabel_reason"] == REASON_ALIAS
    assert out.loc[0, "candidate_label_original"] == "FALSE-POSITIVE"


def test_alias_of_recovered_planet_becomes_unknown():
    found = _row(6.0, candidate_label="CONFIRMED", matched_target="TOI-1.01", fp_source=None)
    out = _relabel([found, _row(12.0)], _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", "")))
    assert list(out["candidate_label"]) == ["CONFIRMED", "UNKNOWN"]
    assert out.loc[1, "relabel_reason"] == REASON_ALIAS


def test_only_one_alias_row_claims_an_unrecovered_planet():
    out = _relabel(
        [_row(3.0), _row(12.0)], _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", ""))
    )
    assert list(out["candidate_label"]) == ["CONFIRMED", "UNKNOWN"]


def test_alias_follows_false_alarm_and_plain_false_positive():
    reference = _reference(
        ("TOI-1.01", 6.0, 0.001, "FALSE POSITIVE", "fa"),
        ("TOI-2.01", 10.0, 0.001, "FALSE POSITIVE", "fp"),
    )
    out = _relabel([_row(3.0), _row(20.0)], reference)
    assert out.loc[0, "candidate_label"] == "FALSE-POSITIVE"
    assert out.loc[0, "fp_source"] == "toi:fa"
    assert out.loc[1, "candidate_label"] == "UNKNOWN"


def test_loose_ratio_is_not_an_alias():
    # 4/5 of the period: too easy to hit by chance.
    out = _relabel([_row(4.8)], _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", "")))
    assert out.loc[0, "relabel_reason"] == ""


def test_nearby_period_with_similar_depth_is_the_same_eclipse():
    # TIC 2041563029: TOI-1427.01 at 12.81 d was found at 13.58 d.
    reference = _reference(("TOI-1427.01", 12.81, 0.027, "FALSE POSITIVE", "fp"))
    out = _relabel([_row(13.58, depth_mean_per_transit=0.032, max_mes=389.0)], reference)
    assert out.loc[0, "candidate_label"] == "UNKNOWN"
    assert out.loc[0, "relabel_reason"] == REASON_NEARBY


def test_nearby_period_needs_a_similar_depth_and_an_unmatched_reference():
    reference = _reference(("TOI-1.01", 12.81, 0.027, "FALSE POSITIVE", "fp"))
    shallow = _relabel([_row(13.58, depth_mean_per_transit=0.001)], reference)
    assert shallow.loc[0, "relabel_reason"] == ""
    found = _row(12.81, candidate_label="UNKNOWN", matched_target="TOI-1.01", fp_source=None)
    both = _relabel([found, _row(13.58, depth_mean_per_transit=0.03)], reference)
    assert both.loc[1, "relabel_reason"] == ""


def test_viable_row_becomes_unknown():
    out = _relabel([_row(7.3, max_mes=14.0, max_ses=6.0)], _reference())
    assert out.loc[0, "candidate_label"] == "UNKNOWN"
    assert out.loc[0, "relabel_reason"] == REASON_VIABLE


def test_viability_tests_each_reject():
    strong = _row(7.3, max_mes=14.0, max_ses=6.0)
    assert is_planet_viable(strong)
    assert is_planet_viable({**strong, "log_rho_ratio": float("nan")})
    assert not is_planet_viable({**strong, "max_mes": 9.0})
    assert not is_planet_viable({**strong, "detection_status": "provisional"})
    assert not is_planet_viable({**strong, "max_ses": 13.0})  # one event carries it
    assert not is_planet_viable({**strong, "log_rho_ratio": -1.4})
    assert not is_planet_viable({**strong, "depth_mean_per_transit": 0.03})
    assert not is_planet_viable({**strong, "even_odd_depth_diff_sigma": 4.0})
    assert not is_planet_viable(strong, inverted_best_mes=12.0)
    assert is_planet_viable(strong, inverted_best_mes=9.0)


def test_other_labels_pass_through():
    rows = [
        _row(6.0, candidate_label="CONFIRMED", matched_target="TOI-1.01", fp_source=None),
        _row(3.0, fp_source="inverted", max_mes=20.0, max_ses=6.0),
    ]
    out = _relabel(rows, _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", "")))
    assert list(out["candidate_label"]) == ["CONFIRMED", "FALSE-POSITIVE"]
    assert list(out["relabel_reason"]) == ["", ""]


def test_relabel_directory_writes_copies_and_keeps_originals(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    reference = tmp_path / "TIC1-toi.csv"
    _reference(("TOI-1.01", 6.0, 0.001, "CONFIRMED", "")).to_csv(reference, index=False)
    original = source / "TIC1_20261110.csv"
    pd.DataFrame([_row(4.0), _row(7.3, max_mes=14.0, max_ses=6.0)]).to_csv(original, index=False)
    before = original.read_bytes()
    # The inverted pass reached MES 12, so the 14.0 row is not above the star's noise.
    inverted = pd.DataFrame([_row(9.9, max_mes=12.0)])
    inverted.to_csv(source / "TIC1_inverted_20261110.csv", index=False)
    (source / "TIC2_20261110.csv").write_text("\n")  # a star with no detections
    targets = pd.DataFrame(
        [
            {"target": "TIC 1", "reference": str(reference)},
            {"target": "TIC 2", "reference": ""},
            {"target": "TIC 3", "reference": ""},  # not run yet
        ]
    )

    out_dir = tmp_path / "relabelled"
    changes = relabel_directory(source, targets, out_dir)

    assert original.read_bytes() == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["TIC1_20261110.csv", "TIC2_20261110.csv"]
    written = pd.read_csv(out_dir / "TIC1_20261110.csv")
    assert list(written["candidate_label"]) == ["CONFIRMED", "FALSE-POSITIVE"]
    assert list(changes["relabel_reason"]) == [REASON_ALIAS]
