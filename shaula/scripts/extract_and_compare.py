#!/usr/bin/env python3
"""Extract candidate features for a star and compare against confirmed values.

Usage (from the repo root with the project venv active)::

    python -m shaula.scripts.extract_and_compare Kepler-5
    python -m shaula.scripts.extract_and_compare HAT-P-7 --mission Kepler --download-all

Pass ``--include-false-candidates --label-output-candidates`` to build training
data: the CSV then also carries the peaks the search measured and declined, each
row labelled CONFIRMED / FALSE-POSITIVE / UNKNOWN against the confirmed catalog.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from ..download_and_clean import download_and_clean_lightcurve
from ..extract_feats import extract_features_from_lightcurve
from ..save import save_features
from ..utils.compare_extracted_confirmed import (
    compare_extracted_confirmed,
    find_confirmed_csv,
)
from ..utils.target_names import host_star_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a stellar light curve, extract candidate rows, and compare "
            "against data/confirmed/{star}-confirmed.csv when available."
        )
    )
    parser.add_argument(
        "target",
        help="LightKurve host-star name, e.g. Kepler-5",
    )
    parser.add_argument(
        "--mission",
        default="Kepler",
        help="Mission name passed to lightkurve (default: Kepler)",
    )
    parser.add_argument(
        "--author",
        help="Lightkurve product author (default: Kepler/K2/SPOC by mission)",
    )
    parser.add_argument(
        "--exptime",
        help="Exposure class or seconds (default: long for Kepler/K2, short for TESS)",
    )
    parser.add_argument(
        "--sigma-upper",
        type=float,
        default=5.0,
        help="Upper sigma for cleaning (default: 5.0)",
    )
    parser.add_argument(
        "--download-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stitch all available light curves (default: true)",
    )
    parser.add_argument(
        "--refine-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine BLS duration (default: true)",
    )
    parser.add_argument(
        "--use-tls",
        action="store_true",
        help="Use TLS instead of BLS where supported",
    )
    parser.add_argument(
        "--mask-eclipses",
        action="store_true",
        help="Mask eclipses in the light curve",
    )
    parser.add_argument(
        "--include-false-candidates",
        action="store_true",
        help=(
            "Also write the candidates the search measured and rejected, marked "
            "by detection_status. They mask nothing, so the detections are "
            "unchanged. Useful for building negative ML training examples."
        ),
    )
    parser.add_argument(
        "--label-output-candidates",
        action="store_true",
        help=(
            "Add a candidate_label column: CONFIRMED when the row's period pairs "
            "with a confirmed planet for this star, FALSE-POSITIVE when it pairs "
            "with none, UNKNOWN when no confirmed CSV is available"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "extracted",
        help="Directory for the extracted features CSV (default: data/extracted)",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "confirmed",
        help="Directory of confirmed CSVs (default: data/confirmed)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce extraction verbosity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet
    star = host_star_name(args.target)

    print(f"\n=== Extract & compare: {star} ({args.mission}) ===\n")

    # Resolved before the extraction rather than after it: labelling needs the
    # catalog while the rows are still in memory, and a star with no confirmed
    # data is worth reporting before a long download and search rather than
    # after one.
    confirmed_path = find_confirmed_csv(star, args.confirmed_dir)
    if confirmed_path is None:
        note = (
            " Candidates will be labelled UNKNOWN."
            if args.label_output_candidates
            else ""
        )
        print(
            f"No confirmed CSV for {star} in {args.confirmed_dir} "
            f"(expected `{star}-confirmed.csv`).{note}\n"
        )

    lc = download_and_clean_lightcurve(
        target=star,
        mission=args.mission,
        sigma_upper=args.sigma_upper,
        all=args.download_all,
        verbose=verbose,
        author=args.author,
        exptime=args.exptime,
    )

    feats = extract_features_from_lightcurve(
        lc,
        verbose=verbose,
        refine_duration=args.refine_duration,
        use_tls=args.use_tls,
        mask_eclipses=args.mask_eclipses,
        include_false_candidates=args.include_false_candidates,
        label_output_candidates=args.label_output_candidates,
        confirmed_rows=confirmed_path,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = args.out_dir / f"{star}_{datetime.now().strftime('%Y%m%d')}.csv"
    save_features(feats, star, str(extracted_path), verbose=verbose)
    print(f"\nExtracted features saved to: {extracted_path}")

    if confirmed_path is None:
        print("\nNo confirmed catalog for this star; skipping comparison.")
        return 0

    print(f"\nComparing against confirmed catalog: {confirmed_path}\n")
    compare_extracted_confirmed(extracted_path, confirmed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
