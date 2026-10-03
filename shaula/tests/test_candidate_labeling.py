"""Labelling extracted candidates against the confirmed catalog."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ..utils.compare_extracted_confirmed import (
    _detections_only,
    compare_extracted_confirmed,
)
from ..utils.label_candidates import (
    label_candidate_rows,
    summarize_labels,
)

CONFIRMED = pd.DataFrame(
    {
        "target": ["Kepler-186b", "Kepler-186f"],
        "period_days": [3.886795001, 129.945392],
        "duration_days": [0.073225, 0.241083],
    }
)


class CandidateLabelTests(unittest.TestCase):
    def test_a_matching_period_is_confirmed_and_names_its_planet(self):
        rows = label_candidate_rows([{"period_days": 3.8867953}], CONFIRMED)

        self.assertEqual(rows[0]["candidate_label"], "CONFIRMED")
        self.assertEqual(rows[0]["matched_target"], "Kepler-186b")
        self.assertEqual(rows[0]["matched_period_ratio"], "direct")

    def test_an_unmatched_period_is_a_false_positive(self):
        rows = label_candidate_rows([{"period_days": 68.0}], CONFIRMED)

        self.assertEqual(rows[0]["candidate_label"], "FALSE-POSITIVE")
        self.assertEqual(rows[0]["matched_target"], "")

    def test_an_integer_alias_is_confirmed_and_records_the_ratio(self):
        rows = label_candidate_rows([{"period_days": 64.9727}], CONFIRMED)

        self.assertEqual(rows[0]["candidate_label"], "CONFIRMED")
        self.assertEqual(rows[0]["matched_target"], "Kepler-186f")
        self.assertEqual(rows[0]["matched_period_ratio"], "P/2")

    def test_every_row_is_unknown_without_a_catalog(self):
        rows = label_candidate_rows(
            [{"period_days": 3.8867953}, {"period_days": 68.0}], None
        )

        self.assertEqual(
            [row["candidate_label"] for row in rows], ["UNKNOWN", "UNKNOWN"]
        )

    def test_a_planet_is_claimed_once_so_the_runner_up_is_a_false_positive(self):
        rows = label_candidate_rows(
            [{"period_days": 3.886795}, {"period_days": 3.8871}], CONFIRMED
        )

        # Both sit inside the 2% window; only the closer one is the planet.
        self.assertEqual(rows[0]["candidate_label"], "CONFIRMED")
        self.assertEqual(rows[1]["candidate_label"], "FALSE-POSITIVE")

    def test_row_order_and_existing_features_survive_labelling(self):
        original = [
            {"period_days": 68.0, "max_mes": 3.0},
            {"period_days": 3.8867953, "max_mes": 30.0},
        ]

        rows = label_candidate_rows(original, CONFIRMED)

        self.assertEqual([row["period_days"] for row in rows], [68.0, 3.8867953])
        self.assertEqual([row["max_mes"] for row in rows], [3.0, 30.0])
        # The caller's dicts are not mutated.
        self.assertNotIn("candidate_label", original[0])

    def test_a_csv_path_is_accepted_in_place_of_a_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Kepler-186-confirmed.csv"
            CONFIRMED.to_csv(path, index=False)

            rows = label_candidate_rows([{"period_days": 3.8867953}], path)

        self.assertEqual(rows[0]["candidate_label"], "CONFIRMED")

    def test_a_missing_catalog_path_degrades_to_unknown(self):
        rows = label_candidate_rows([{"period_days": 3.88}], "/no/such/file.csv")

        self.assertEqual(rows[0]["candidate_label"], "UNKNOWN")

    def test_summary_tallies_each_label(self):
        rows = label_candidate_rows(
            [{"period_days": 3.8867953}, {"period_days": 68.0}], CONFIRMED
        )

        self.assertEqual(summarize_labels(rows), "CONFIRMED=1, FALSE-POSITIVE=1")


class DetectionOnlyComparisonTests(unittest.TestCase):
    def test_false_candidates_are_excluded_from_the_comparison(self):
        rows = pd.DataFrame(
            {
                "period_days": [3.886795, 68.0],
                "detection_status": ["accepted", "rejected"],
            }
        )

        self.assertEqual(list(_detections_only(rows)["period_days"]), [3.886795])

    def test_a_csv_without_the_column_is_left_alone(self):
        rows = pd.DataFrame({"period_days": [3.886795, 68.0]})

        self.assertEqual(len(_detections_only(rows)), 2)

    def test_report_grades_detections_only_but_can_be_told_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            confirmed_path = Path(tmp) / "Kepler-186-confirmed.csv"
            CONFIRMED.to_csv(confirmed_path, index=False)
            extracted_path = Path(tmp) / "Kepler-186_extracted.csv"
            pd.DataFrame(
                {
                    "period_days": [3.886795, 129.945392, 68.0],
                    "duration_days": [0.0732, 0.2411, 0.4],
                    "detection_status": ["accepted", "accepted", "rejected"],
                }
            ).to_csv(extracted_path, index=False)

            with contextlib.redirect_stdout(io.StringIO()):
                default = compare_extracted_confirmed(extracted_path, confirmed_path)
                everything = compare_extracted_confirmed(
                    extracted_path, confirmed_path, detections_only=False
                )

        kinds = [m["kind"] for m in default.attrs["matches"]]
        self.assertNotIn("extra", kinds)
        self.assertIn("extra", [m["kind"] for m in everything.attrs["matches"]])


if __name__ == "__main__":
    unittest.main()
