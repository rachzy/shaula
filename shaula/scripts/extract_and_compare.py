#!/usr/bin/env python3
"""Extract candidate features for a star and compare against confirmed values.

Usage (from the repo root with the project venv active)::

    python -m shaula.scripts.extract_and_compare Kepler-5
    python -m shaula.scripts.extract_and_compare HAT-P-7 --mission Kepler --download-all

Pass ``--include-false-candidates --label-output-candidates`` to build training
data: the CSV then also carries the peaks the search measured and declined, each
row labelled CONFIRMED / FALSE-POSITIVE / UNKNOWN against the confirmed catalog.

``--reference`` points at a KOI reference table (see ``build_koi_targets``)
instead of looking one up by star name. ``--invert`` flips the cleaned light
curve before the search, so real transits become brightenings that cannot be
detected: every row found is a false alarm, labelled FALSE-POSITIVE with
``fp_source=inverted``, and is written to ``<star>_inverted_<date>.csv``.
With ``--reference``, the transits and phase-0.5 windows of every KOI in the
table are cut out before the flip, so the search cannot latch onto the edges of
a flipped real signal.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..download_and_clean import download_and_clean_lightcurve
from ..extract_feats import extract_features_from_lightcurve
from ..save import save_features
from ..utils.compare_extracted_confirmed import (
    compare_extracted_confirmed,
    find_confirmed_csv,
)
from ..utils.ephemeris import MISSION_TIME_OFFSETS
from ..utils.label_candidates import (
    FP_SOURCE_INVERTED,
    label_all_false_positive,
    summarize_labels,
)
from ..utils.target_names import host_star_name
from .network import install_default_request_timeout, with_retries

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        "--reference",
        type=Path,
        help=(
            "Reference table to label against (a confirmed CSV or a KOI table "
            "from build_koi_targets); skips the lookup in --confirmed-dir"
        ),
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help=(
            "Flip the cleaned light curve before the search and label every "
            "row it finds FALSE-POSITIVE (fp_source=inverted)"
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


def output_stem(star: str, *, inverted: bool = False) -> str:
    """File-name stem for a star's output: no spaces, so ``KIC 123`` -> ``KIC123``."""
    stem = star.replace(" ", "")
    return f"{stem}_inverted" if inverted else stem


def invert_lightcurve(lc):
    """Mirror the flux about its median, turning every dip into a brightening.

    Done after cleaning, which clips upward outliers only: flares are already
    gone and cannot turn into fake dips, while the real transits survive the
    cleaning and are flipped out of reach of the search.
    """
    inverted = lc.copy()
    flux = inverted.flux
    inverted.flux = 2.0 * np.nanmedian(flux) - flux
    return inverted


# Full width of the window cut around each known transit and its phase-0.5
# point, in units of T14. Flipping a deep transit leaves a dip-like residual on
# either side of the brightening once the trend is fitted around it (Kepler-7b
# inverted gave MES 486 at 2/5 of its period), so the cut is wider than the
# 1.5 x T14 the search itself masks.
KNOWN_SIGNAL_MASK_WIDTH = 3.0
# Used when a KOI has no duration.
DEFAULT_KNOWN_SIGNAL_DURATION_DAYS = 0.25


def known_signal_mask(time, reference: pd.DataFrame, mission: str = "Kepler"):
    """Cadences within the transit or phase-0.5 window of any reference row.

    Phase 0.5 is cut too because an eclipsing binary's secondary is not a KOI
    of its own but flips into a brightening all the same. Catalogue epochs in
    full BJD are moved to the light curve's mission time base.
    """
    time = np.asarray(time, dtype=float)
    mask = np.zeros(time.size, dtype=bool)
    offset = MISSION_TIME_OFFSETS.get(str(mission).lower(), 0.0)
    for _, row in reference.iterrows():
        period = float(row.get("period_days", np.nan))
        epoch = float(row.get("t0", np.nan))
        if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)):
            continue
        if abs(epoch) > 1_000_000.0:
            epoch -= offset
        duration = float(row.get("duration_days", np.nan))
        if not (np.isfinite(duration) and duration > 0):
            duration = DEFAULT_KNOWN_SIGNAL_DURATION_DAYS
        half_width = min(0.5 * KNOWN_SIGNAL_MASK_WIDTH * duration, 0.2 * period)
        for phase_epoch in (epoch, epoch + 0.5 * period):
            dt = np.mod(time - phase_epoch + 0.5 * period, period) - 0.5 * period
            mask |= np.abs(dt) < half_width
    return mask


def remove_known_signals(lc, reference: pd.DataFrame, mission: str = "Kepler"):
    """Drop the cadences ``known_signal_mask`` selects; returns (lc, n_removed)."""
    mask = known_signal_mask(lc.time.value, reference, mission)
    return lc[~mask], int(mask.sum())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet
    star = host_star_name(args.target)

    print(f"\n=== Extract & compare: {star} ({args.mission}) ===\n")

    # Resolved before the extraction rather than after it: labelling needs the
    # catalog while the rows are still in memory, and a star with no confirmed
    # data is worth reporting before a long download and search rather than
    # after one.
    if args.invert:
        confirmed_path = None
    elif args.reference is not None:
        confirmed_path = args.reference if args.reference.is_file() else None
        if confirmed_path is None:
            print(f"Reference table {args.reference} not found.\n")
    else:
        confirmed_path = find_confirmed_csv(star, args.confirmed_dir)
    if confirmed_path is None and not args.invert:
        note = (
            " Candidates will be labelled UNKNOWN."
            if args.label_output_candidates
            else ""
        )
        print(
            f"No confirmed CSV for {star} in {args.confirmed_dir} "
            f"(expected `{star}-confirmed.csv`).{note}\n"
        )

    install_default_request_timeout()
    lc = with_retries(
        lambda: download_and_clean_lightcurve(
            target=star,
            mission=args.mission,
            sigma_upper=args.sigma_upper,
            all=args.download_all,
            verbose=verbose,
            author=args.author,
            exptime=args.exptime,
        )
    )
    if args.invert:
        if args.reference is not None and args.reference.is_file():
            lc, removed = remove_known_signals(
                lc, pd.read_csv(args.reference), args.mission
            )
            print(
                f"Removed {removed} cadences around the known signals in "
                f"{args.reference} before inverting.\n"
            )
        lc = invert_lightcurve(lc)
        print("Light curve inverted: every detection below is a false alarm.\n")

    feats = extract_features_from_lightcurve(
        lc,
        verbose=verbose,
        refine_duration=args.refine_duration,
        use_tls=args.use_tls,
        mask_eclipses=args.mask_eclipses,
        include_false_candidates=args.include_false_candidates,
        label_output_candidates=args.label_output_candidates and not args.invert,
        confirmed_rows=confirmed_path,
    )
    if args.invert and args.label_output_candidates:
        feats = label_all_false_positive(feats, FP_SOURCE_INVERTED)
        print(f"Candidate labels: {summarize_labels(feats)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(star, inverted=args.invert)
    extracted_path = args.out_dir / f"{stem}_{datetime.now().strftime('%Y%m%d')}.csv"
    save_features(feats, star, str(extracted_path), verbose=verbose)
    print(f"\nExtracted features saved to: {extracted_path}")

    if args.invert:
        return 0
    if confirmed_path is None:
        print("\nNo confirmed catalog for this star; skipping comparison.")
        return 0
    if not feats:
        print("\nNo candidates extracted; skipping comparison.")
        return 0

    print(f"\nComparing against confirmed catalog: {confirmed_path}\n")
    compare_extracted_confirmed(extracted_path, confirmed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
