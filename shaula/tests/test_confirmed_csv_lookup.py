"""Resolving a host star to its confirmed-catalog CSV."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ..utils.compare_extracted_confirmed import (
    _is_confirmed_stem_for,
    find_confirmed_csv,
)


class ConfirmedStemMatchTests(unittest.TestCase):
    def test_the_star_name_alone_is_not_a_confirmed_table(self):
        self.assertFalse(_is_confirmed_stem_for("Kepler-42", "Kepler-42"))

    def test_separators_other_than_a_dash_are_accepted(self):
        for stem in ("Kepler-42-confirmed", "Kepler-42_confirmed", "Kepler-42.confirmed"):
            with self.subTest(stem=stem):
                self.assertTrue(_is_confirmed_stem_for(stem, "Kepler-42"))

    def test_the_historical_typo_still_matches(self):
        self.assertTrue(_is_confirmed_stem_for("Kepler-42_confimed", "Kepler-42"))

    def test_matching_ignores_case(self):
        self.assertTrue(_is_confirmed_stem_for("kepler-42-CONFIRMED", "Kepler-42"))

    def test_a_longer_star_name_is_not_a_match(self):
        # Kepler-42 is a prefix of Kepler-421, but they are different stars.
        self.assertFalse(_is_confirmed_stem_for("Kepler-421-confirmed", "Kepler-42"))
        self.assertFalse(_is_confirmed_stem_for("Kepler-138-confirmed", "Kepler-13"))


class FindConfirmedCsvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.confirmed_dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str) -> Path:
        path = self.confirmed_dir / name
        path.write_text("target,period_days\nKepler-999b,1.0\n", encoding="utf-8")
        return path

    def test_the_exact_file_is_returned(self):
        expected = self.write("Kepler-42-confirmed.csv")

        found = find_confirmed_csv("Kepler-42", self.confirmed_dir, auto_fetch=False)

        self.assertEqual(found, expected)

    def test_a_planet_letter_resolves_to_its_host_star(self):
        expected = self.write("Kepler-42-confirmed.csv")

        found = find_confirmed_csv("Kepler-42b", self.confirmed_dir, auto_fetch=False)

        self.assertEqual(found, expected)

    def test_a_neighbouring_star_is_never_substituted(self):
        # The failure this guards: Kepler-42's candidates were graded against
        # Kepler-421's single 704-day planet, so every real planet came out
        # FALSE-POSITIVE. Missing is the honest answer; a neighbour is not.
        self.write("Kepler-421-confirmed.csv")
        self.write("Kepler-422-confirmed.csv")

        found = find_confirmed_csv("Kepler-42", self.confirmed_dir, auto_fetch=False)

        self.assertIsNone(found)

    def test_the_historical_typo_filename_still_resolves(self):
        expected = self.write("Kepler-42-confimed.csv")

        found = find_confirmed_csv("Kepler-42", self.confirmed_dir, auto_fetch=False)

        self.assertEqual(found, expected)

    def test_an_alternative_spelling_resolves(self):
        expected = self.write("Kepler-42_confirmed_v2.csv")

        found = find_confirmed_csv("Kepler-42", self.confirmed_dir, auto_fetch=False)

        self.assertEqual(found, expected)

    def test_a_missing_star_returns_none_without_fetching(self):
        self.write("Kepler-186-confirmed.csv")

        found = find_confirmed_csv("Kepler-42", self.confirmed_dir, auto_fetch=False)

        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
