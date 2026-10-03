"""Regression tests for the batch labeled-star extraction runner."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from collections import Counter
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from ..scripts.extract_multiple_stars import (
    STARS,
    Star,
    _extract_one,
    build_argv,
    has_existing_output,
    parse_args,
    run_batch,
)


class HasExistingOutputTests(unittest.TestCase):
    def test_no_matching_csv_is_not_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(has_existing_output("Kepler-7", Path(tmp)))

    def test_a_dated_csv_for_the_star_counts_as_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "Kepler-7_20260101.csv").write_text("target\n")
            self.assertTrue(has_existing_output("Kepler-7", out_dir))

    def test_a_csv_for_a_different_star_does_not_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "Kepler-70_20260101.csv").write_text("target\n")
            # "Kepler-7" must not match "Kepler-70_..." via a loose prefix check.
            self.assertFalse(has_existing_output("Kepler-7", out_dir))


class StarListTests(unittest.TestCase):
    """The batch list is the deliverable here, so guard its shape."""

    def test_no_duplicate_targets(self):
        names = [host.name for host in STARS]
        dupes = [name for name, count in Counter(names).items() if count > 1]
        self.assertEqual(dupes, [])

    def test_every_group_has_at_least_three_stars(self):
        counts = Counter(host.group for host in STARS)
        thin = {group: n for group, n in counts.items() if n < 3}
        self.assertEqual(thin, {}, f"groups with fewer than 3 stars: {thin}")

    def test_every_entry_names_a_mission(self):
        # The TESS block is currently commented out pending TESS support in
        # get_literature_data, so this deliberately does not require TESS to
        # be present -- only that whatever is enabled carries a mission.
        for host in STARS:
            self.assertTrue(host.mission.strip(), f"{host.name} has no mission")

    def test_tess_targets_are_not_named_like_kepler_targets(self):
        for host in STARS:
            if host.mission == "TESS":
                self.assertFalse(
                    host.name.lower().startswith("kepler"),
                    f"{host.name} is tagged TESS but named as a Kepler target",
                )


class RunBatchTests(unittest.TestCase):
    def _dirs(self, tmp):
        out_dir = Path(tmp) / "out"
        confirmed_dir = Path(tmp) / "confirmed"
        out_dir.mkdir()
        confirmed_dir.mkdir()
        return out_dir, confirmed_dir

    def test_a_star_with_existing_output_is_skipped_without_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)
            (out_dir / "Kepler-7_20260101.csv").write_text("target\n")

            with patch(
                "shaula.scripts.extract_multiple_stars.extract_and_compare.main"
            ) as mock_main:
                succeeded, skipped, failed = run_batch(
                    [Star("Kepler-7", "test")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                )

            mock_main.assert_not_called()
            self.assertEqual((succeeded, skipped, failed), ([], ["Kepler-7"], []))

    def test_a_successful_extraction_passes_the_expected_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)

            with patch(
                "shaula.scripts.extract_multiple_stars.extract_and_compare.main"
            ) as mock_main:
                mock_main.return_value = 0
                succeeded, skipped, failed = run_batch(
                    [Star("Kepler-8", "test")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                )

            (argv,), _kwargs = mock_main.call_args
            self.assertEqual(argv[0], "Kepler-8")
            self.assertIn("--include-false-candidates", argv)
            self.assertIn("--label-output-candidates", argv)
            self.assertEqual(argv[argv.index("--out-dir") + 1], str(out_dir))
            self.assertEqual(argv[argv.index("--confirmed-dir") + 1], str(confirmed_dir))
            self.assertEqual(argv[argv.index("--mission") + 1], "Kepler")

            self.assertEqual((succeeded, skipped, failed), (["Kepler-8"], [], []))

    def test_the_per_star_mission_reaches_the_extractor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)

            with patch(
                "shaula.scripts.extract_multiple_stars.extract_and_compare.main"
            ) as mock_main:
                mock_main.return_value = 0
                run_batch(
                    [Star("TOI-178", "tess", mission="TESS")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                )

            (argv,), _kwargs = mock_main.call_args
            self.assertEqual(argv[argv.index("--mission") + 1], "TESS")

    def test_a_failing_star_is_marked_failed_and_the_batch_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)

            def fake_main(argv):
                if argv[0] == "Kepler-9":
                    raise RuntimeError("boom")
                return 0

            with patch(
                "shaula.scripts.extract_multiple_stars.extract_and_compare.main",
                side_effect=fake_main,
            ) as mock_main:
                succeeded, skipped, failed = run_batch(
                    [Star("Kepler-9", "test"), Star("Kepler-10", "test")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                )

            self.assertEqual(mock_main.call_count, 2)
            self.assertEqual((succeeded, skipped, failed), (["Kepler-10"], [], ["Kepler-9"]))

    def test_planet_letter_targets_are_normalized_to_the_host_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)

            with patch(
                "shaula.scripts.extract_multiple_stars.extract_and_compare.main"
            ) as mock_main:
                mock_main.return_value = 0
                succeeded, _skipped, _failed = run_batch(
                    [Star("Kepler-8b", "test")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                )

            (argv,), _kwargs = mock_main.call_args
            self.assertEqual(argv[0], "Kepler-8")
            self.assertEqual(succeeded, ["Kepler-8"])


class WorkerTests(unittest.TestCase):
    """The parallel worker must swallow failures and capture output."""

    def test_a_worker_captures_stdout_instead_of_printing_it(self):
        def noisy(argv):
            print("chatter from the extractor")
            return 0

        buffer = io.StringIO()
        with patch(
            "shaula.scripts.extract_multiple_stars.extract_and_compare.main",
            side_effect=noisy,
        ), contextlib.redirect_stdout(buffer):
            star, ok, output = _extract_one((Star("Kepler-8", "test"), "Kepler-8", []))

        self.assertTrue(ok)
        self.assertIn("chatter from the extractor", output)
        # The parent replays the block later; nothing may leak out mid-run.
        self.assertEqual(buffer.getvalue(), "")

    def test_a_worker_reports_failure_instead_of_raising(self):
        with patch(
            "shaula.scripts.extract_multiple_stars.extract_and_compare.main",
            side_effect=RuntimeError("boom"),
        ):
            star, ok, output = _extract_one((Star("Kepler-9", "test"), "Kepler-9", []))

        self.assertEqual(star, "Kepler-9")
        self.assertFalse(ok)
        self.assertIn("boom", output)


class ParallelRunBatchTests(unittest.TestCase):
    """workers > 1 must bookkeep exactly like the serial path."""

    def _dirs(self, tmp):
        out_dir = Path(tmp) / "out"
        confirmed_dir = Path(tmp) / "confirmed"
        out_dir.mkdir()
        confirmed_dir.mkdir()
        return out_dir, confirmed_dir

    def test_results_match_the_serial_path(self):
        entries = [
            Star("Kepler-8", "test"),
            Star("Kepler-9", "test"),
            Star("Kepler-10", "test"),
        ]

        def fake_extract_one(job):
            _entry, star, _argv = job
            return star, star != "Kepler-9", ""

        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)
            # Patch the worker itself: a real pool would need to pickle the
            # mock, and the scheduling is what's under test here, not the
            # extraction.
            with patch(
                "shaula.scripts.extract_multiple_stars._extract_one",
                side_effect=fake_extract_one,
            ), patch(
                "shaula.scripts.extract_multiple_stars.ProcessPoolExecutor",
                _InlineExecutor,
            ):
                succeeded, skipped, failed = run_batch(
                    entries,
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                    workers=3,
                )

        self.assertEqual(sorted(succeeded), ["Kepler-10", "Kepler-8"])
        self.assertEqual(skipped, [])
        self.assertEqual(failed, ["Kepler-9"])

    def test_already_extracted_stars_are_skipped_before_any_worker_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir, confirmed_dir = self._dirs(tmp)
            (out_dir / "Kepler-8_20260101.csv").write_text("target\n")

            with patch(
                "shaula.scripts.extract_multiple_stars._extract_one"
            ) as mock_worker, patch(
                "shaula.scripts.extract_multiple_stars.ProcessPoolExecutor", _InlineExecutor
            ):
                succeeded, skipped, failed = run_batch(
                    [Star("Kepler-8", "test")],
                    include_false_candidates=True,
                    label_output_candidates=True,
                    out_dir=out_dir,
                    confirmed_dir=confirmed_dir,
                    workers=3,
                )

            mock_worker.assert_not_called()
            self.assertEqual((succeeded, skipped, failed), ([], ["Kepler-8"], []))


class _InlineExecutor:
    """ProcessPoolExecutor stand-in that runs jobs inline, in order.

    Returns real ``Future`` objects, already resolved, so the production
    ``as_completed`` loop is exercised unchanged.
    """

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, arg):
        future = Future()
        try:
            future.set_result(fn(arg))
        except Exception as exc:  # noqa: BLE001 - mirrors pool behaviour
            future.set_exception(exc)
        return future


class ArgParsingTests(unittest.TestCase):
    def test_workers_defaults_to_serial(self):
        args = parse_args(
            ["--include-false-candidates", "--label-output-candidates"]
        )
        self.assertEqual(args.workers, 1)

    def test_workers_is_configurable(self):
        args = parse_args(
            [
                "--workers",
                "3",
                "--no-include-false-candidates",
                "--no-label-output-candidates",
            ]
        )
        self.assertEqual(args.workers, 3)
        self.assertFalse(args.include_false_candidates)
        self.assertFalse(args.label_output_candidates)

    def test_candidate_options_must_be_selected(self):
        with self.assertRaises(SystemExit):
            parse_args([])


class BuildArgvTests(unittest.TestCase):
    def test_selected_candidate_flags_are_forwarded(self):
        argv = build_argv(
            Star("TOI-178", "tess", mission="TESS"),
            "TOI-178",
            Path("/out"),
            Path("/confirmed"),
            True,
            True,
            [],
        )
        self.assertEqual(argv[0], "TOI-178")
        self.assertIn("--include-false-candidates", argv)
        self.assertIn("--label-output-candidates", argv)
        self.assertEqual(argv[argv.index("--mission") + 1], "TESS")

    def test_unlabelled_stars_do_not_receive_the_labelling_flag(self):
        argv = build_argv(
            Star("TOI-178", "tess", mission="TESS"),
            "TOI-178",
            Path("/out"),
            Path("/confirmed"),
            True,
            False,
            [],
        )
        self.assertIn("--include-false-candidates", argv)
        self.assertNotIn("--label-output-candidates", argv)


if __name__ == "__main__":
    unittest.main()
