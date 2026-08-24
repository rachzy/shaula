#!/usr/bin/env python3
"""Run extract_and_compare over a diverse batch of stars for ML training data.

Runs :mod:`extract_and_compare` for each entry in ``STARS`` with
the caller-selected false-candidate and output-labelling options. This supports
both labelled training data and unlabelled feature rows. Stars are processed one
after another: a star that already has an extracted CSV in the output directory
is skipped, and a star whose extraction raises is marked failed and the batch
moves straight on to the next one rather than crashing.

``STARS`` is deliberately spread across the axes a transit classifier can
overfit on -- system multiplicity, transit depth, orbital period, host spectral
type, stellar multiplicity, and mission -- so the negative and positive rows
don't all come from the same corner of parameter space. Every Kepler entry was
checked against the NASA Exoplanet Archive KOI cumulative table and every TESS
entry against the SPOC short-cadence products lightkurve serves, so each target
here does resolve.

Labels come from ``data/confirmed/``; ``find_confirmed_csv`` already auto-fetches
a missing Kepler CSV from the NASA Exoplanet Archive, so no pre-seeding is needed
for the Kepler entries. That fetch is Kepler-only, so labelled TESS rows below
currently receive UNKNOWN until ``get_literature_data`` grows TESS support.

Usage (from ``src/scripts`` with the project venv active)::

    python extract_multiple_stars.py \\
        --include-false-candidates --label-output-candidates

Edit the ``STARS`` list below to change which stars are processed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
REPO_ROOT = SRC_DIR.parent

# Allow `python extract_multiple_stars.py` from src/scripts without
# installing the package.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import extract_and_compare  # noqa: E402
from utils.target_names import host_star_name  # noqa: E402


@dataclass(frozen=True)
class Star:
    """One batch target, tagged with the diversity group it was chosen for."""

    name: str
    group: str
    mission: str = "Kepler"


# Every star below is annotated with the archive values it was selected on:
# `n` = confirmed planets, `depth` in ppm, `P` in days, `Teff` in K,
# `R*` in solar radii, `Kp`/`V` magnitude, `MES` = koi_max_mult_ev.
STARS: list[Star] = [
    # -- High multiplicity: many true transits in a single light curve --------
    # Only two Kepler hosts have >=6 confirmed KOI planets, and both are
    # already extracted, so the rest of this group is the 5-planet tier.
    Star("Kepler-90", "high-multiplicity"),   # n=7, 7.0-331.6 d, the deepest KOI stack
    Star("Kepler-11", "high-multiplicity"),   # n=6, 10.3-118.4 d, coplanar low-density
    Star("Kepler-444", "high-multiplicity"),  # n=5, 3.6-9.7 d, Kp=8.72, triple star
    Star("Kepler-80", "high-multiplicity"),   # n=5, 1.0-9.5 d, resonant chain
    Star("Kepler-102", "high-multiplicity"),  # n=5, 5.3-27.5 d, Kp=11.49
    Star("Kepler-62", "high-multiplicity"),   # n=5, 5.7-267.3 d, wide period spread
    Star("Kepler-33", "high-multiplicity"),   # n=5, 5.7-41.0 d
    Star("Kepler-20", "high-multiplicity"),   # n=5, 3.7-77.6 d, mixed radii
    # -- Mixed-scale systems: huge depth dynamic range inside one star --------
    Star("Kepler-37", "mixed-scale"),   # n=4, depth 12.2-610.4, Kp=9.71, sub-Mercury
    Star("Kepler-89", "mixed-scale"),   # n=4, depth 126.3-5612.8, 43x range
    Star("Kepler-167", "mixed-scale"),  # n=4, 4.4-1071.2 d, cold Jupiter + 3 small
    # -- Deep hot Jupiters: saturated, unambiguous positives ------------------
    Star("Kepler-45", "deep-hot-jupiter"),  # depth 36912, P=2.46 d, M dwarf host
    Star("Kepler-17", "deep-hot-jupiter"),  # depth 20824, P=1.49 d, spot crossings
    Star("Kepler-71", "deep-hot-jupiter"),  # depth 21341, P=3.91 d
    # -- Shallow / near-threshold: the hard positives a model must not miss ---
    Star("Kepler-408", "near-threshold"),   # depth 30.7, Kp=8.77, brightest host here
    Star("Kepler-1972", "near-threshold"),  # n=2, depth 17.1-22.2, MES 9.78
    Star("Kepler-452", "near-threshold"),   # depth 189.9, P=384.8 d, MES 7.60
    # -- Long period / few transits: sparse-event regime ----------------------
    Star("Kepler-421", "long-period"),   # P=704.96 d
    Star("Kepler-1704", "long-period"),  # P=988.88 d
    Star("Kepler-849", "long-period"),   # P=394.62 d
    # -- Cool M dwarf hosts: small stars, deep transits, red noise ------------
    Star("Kepler-42", "m-dwarf"),   # n=3, Teff=3292, R*=0.148, ultra-short periods
    Star("Kepler-296", "m-dwarf"),  # n=5, Teff=3526, binary M dwarf pair
    Star("Kepler-32", "m-dwarf"),   # n=5, Teff=3731
    Star("Kepler-138", "m-dwarf"),  # n=3, Teff=3846, Kp=12.92, TTV-rich
    # -- Evolved / hot hosts: strong stellar variability, granulation ---------
    Star("Kepler-91", "evolved-host"),  # R*=6.22 red giant, Teff=4604
    Star("Kepler-56", "evolved-host"),  # n=2, R*=4.36 subgiant, misaligned
    Star("Kepler-432", "evolved-host"), # R*=4.16 giant
    Star("Kepler-13", "evolved-host"),  # Teff=9107 A star, Kp=9.96, depth 4591
    # -- Stellar multiples: the classic false-positive breeding ground --------
    # Blended companion light dilutes depth, so these stress exactly the
    # signals an eclipsing-binary false positive imitates.
    Star("Kepler-908", "binary-host"),  # triple, depth 107.5, Kp=11.52
    Star("Kepler-411", "binary-host"),  # n=3, binary, 3.0-58.0 d
    Star("Kepler-14", "binary-host"),   # close binary, diluted hot Jupiter, P=6.79 d
    Star("Kepler-25", "binary-host"),   # n=2, binary, Kp=10.73, TTV pair
    Star("Kepler-68", "binary-host"),   # n=2, binary, depth 55.0-346.5, Kp=10.00
    # TODO: Implement a `pscomppars` based query for `get_literature_data` to properly fetch TESS Data
    # # (Kepler-444 and Kepler-296 above are also multiple-star systems.)
    # # -- TESS: different cadence, bandpass, and systematics -------------------
    # Star("TOI-178", "tess", mission="TESS"),     # n=6, 1.9-20.7 d resonant chain
    # Star("HD 191939", "tess", mission="TESS"),   # n=6, 8.9-38.4 d
    # Star("TOI-1136", "tess", mission="TESS"),    # n=6, 4.2-39.5 d, young/active
    # Star("L 98-59", "tess", mission="TESS"),     # n=5, Teff=3415 M dwarf
    # Star("HD 23472", "tess", mission="TESS"),    # n=5, Teff=4684, sub-Earths
    # Star("LTT 1445 A", "tess", mission="TESS"),  # n=2, triple M dwarf system
    # # NB: pi Men is HD 39091 / TIC 261136679 -- don't re-add it under those names.
    # Star("pi Men", "tess", mission="TESS"),      # n=3, V=5.65, naked-eye bright host
    # Star("TOI-700", "tess", mission="TESS"),     # M dwarf, habitable-zone planets
    # # Bulk TESS population: all V<11.5 multi-planet hosts, chosen for volume
    # # rather than for covering a new axis. Every one was checked to return
    # # SPOC short-cadence products.
    # Star("HD 108236", "tess", mission="TESS"),  # n=5, V=9.25
    # Star("TOI-5624", "tess", mission="TESS"),   # n=5, V=10.81
    # Star("TOI-1203", "tess", mission="TESS"),   # n=4, V=8.59
    # Star("AU Mic", "tess", mission="TESS"),     # n=4, V=8.81, young active M dwarf
    # Star("TOI-2076", "tess", mission="TESS"),   # n=4, V=9.14, young
    # Star("TOI-561", "tess", mission="TESS"),    # n=4, V=10.25, old thick-disk host
    # Star("TOI-500", "tess", mission="TESS"),    # n=4, V=10.54
    # Star("HR 858", "tess", mission="TESS"),     # n=3, V=6.38
    # Star("HD 63433", "tess", mission="TESS"),   # n=3, V=6.92, young solar analog
    # Star("HD 22946", "tess", mission="TESS"),   # n=3, V=8.27
    # Star("TOI-431", "tess", mission="TESS"),    # n=3, V=9.12
    # Star("HD 28109", "tess", mission="TESS"),   # n=3, V=9.42, 44 sectors
    # Star("GJ 367", "tess", mission="TESS"),     # n=3, V=10.15, Teff=3535
    # Star("GJ 357", "tess", mission="TESS"),     # n=3, V=10.91, Teff=3505
    # Star("TOI-451", "tess", mission="TESS"),    # n=3, V=10.94, young cluster member
    # Star("TOI-125", "tess", mission="TESS"),    # n=3, V=11.02
    # Star("HD 86226", "tess", mission="TESS"),   # n=2, V=7.93
    # Star("HD 213885", "tess", mission="TESS"),  # n=2, V=7.95
]

OUT_DIR = REPO_ROOT / "data" / "extracted"
CONFIRMED_DIR = REPO_ROOT / "data" / "confirmed"


def has_existing_output(star: str, out_dir: Path) -> bool:
    """Return True if `out_dir` already holds an extracted CSV for `star`."""
    return any(out_dir.glob(f"{star}_*.csv"))


def build_argv(
    entry: Star,
    star: str,
    out_dir: Path,
    confirmed_dir: Path,
    include_false_candidates: bool,
    label_output_candidates: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the extract_and_compare CLI arguments for one star."""
    argv = [
        star,
        "--mission",
        entry.mission,
        "--out-dir",
        str(out_dir),
        "--confirmed-dir",
        str(confirmed_dir),
    ]
    if include_false_candidates:
        argv.append("--include-false-candidates")
    if label_output_candidates:
        argv.append("--label-output-candidates")
    return [*argv, *extra_args]


@contextlib.contextmanager
def _single_threaded_math():
    """Pin BLAS/OpenMP to one thread for the duration of a worker's run.

    A single extraction is ~98% CPU on one core, so the win comes from running
    several stars at once, not from threading inside one. Without this cap each
    worker could still spin up a full BLAS thread pool and the pools would
    fight over the same cores.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield  # threadpoolctl is optional; without it we just don't cap.
        return
    with threadpool_limits(limits=1):
        yield


def _extract_one(job: tuple[Star, str, list[str]]) -> tuple[str, bool, str]:
    """Run one star, capturing its output. Never raises.

    Returns ``(star, ok, captured_output)``. Output is captured rather than
    printed so that parallel workers don't interleave their logs into an
    unreadable mess; the parent replays each block when the star finishes.
    """
    _entry, star, argv = job
    buffer = io.StringIO()
    try:
        with _single_threaded_math(), contextlib.redirect_stdout(
            buffer
        ), contextlib.redirect_stderr(buffer):
            extract_and_compare.main(argv)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, not raised
        buffer.write(f"\n!!! Extraction failed for {star}: {exc!r} !!!\n")
        return star, False, buffer.getvalue()
    return star, True, buffer.getvalue()


def run_batch(
    stars: list[Star],
    *,
    include_false_candidates: bool,
    label_output_candidates: bool,
    out_dir: Path = OUT_DIR,
    confirmed_dir: Path = CONFIRMED_DIR,
    extra_args: list[str] | None = None,
    workers: int = 1,
) -> tuple[list[str], list[str], list[str]]:
    """Extract every star in `stars`, skipping or failing individually.

    With ``workers == 1`` stars run one after another and their output streams
    live. With ``workers > 1`` that many stars run in separate processes --
    the work is CPU-bound and effectively single-core per star, so processes
    (not threads) are what actually buys wall-clock time.

    Returns the (succeeded, skipped, failed) star names, each normalized to
    its host-star name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_args = extra_args or []

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    jobs: list[tuple[Star, str, list[str]]] = []

    # The skip check runs up front for every star, so a resumed batch costs
    # nothing and workers are only spawned for stars that actually need work.
    for entry in stars:
        star = host_star_name(entry.name)
        if has_existing_output(star, out_dir):
            print(f"=== Skipping {star}: existing output found in {out_dir} ===")
            skipped.append(star)
            continue
        jobs.append(
            (
                entry,
                star,
                build_argv(
                    entry,
                    star,
                    out_dir,
                    confirmed_dir,
                    include_false_candidates,
                    label_output_candidates,
                    extra_args,
                ),
            )
        )

    if not jobs:
        return succeeded, skipped, failed

    if workers <= 1:
        for entry, star, argv in jobs:
            print(f"\n=== [{entry.group}] {star} ({entry.mission}) ===")
            try:
                extract_and_compare.main(argv)
                succeeded.append(star)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the batch
                print(f"\n!!! Extraction failed for {star}: {exc!r} !!!\n")
                failed.append(star)
        return succeeded, skipped, failed

    print(f"\nRunning {len(jobs)} stars across {workers} worker processes.\n")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, job): job for job in jobs}
        for future in as_completed(futures):
            entry, star, _argv = futures[future]
            try:
                star, ok, output = future.result()
            except Exception as exc:  # noqa: BLE001 - worker died outright
                ok, output = False, f"!!! Worker for {star} died: {exc!r} !!!"
            done += 1
            status = "ok" if ok else "FAILED"
            print(f"\n=== [{done}/{len(jobs)}] {star} ({entry.group}) -> {status} ===")
            print(output, end="" if output.endswith("\n") else "\n")
            (succeeded if ok else failed).append(star)

    return succeeded, skipped, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the extraction pipeline over the STARS list defined "
            "in this module."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Stars to extract in parallel, in separate processes (default: 1). "
            "Each worker peaks near 350 MB and saturates about one core."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Directory for the extracted CSVs (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=CONFIRMED_DIR,
        help=f"Directory of confirmed CSVs (default: {CONFIRMED_DIR})",
    )
    parser.add_argument(
        "--include-false-candidates",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Include rejected search candidates in the output.",
    )
    parser.add_argument(
        "--label-output-candidates",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Add candidate labels to output rows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        print("--workers must be at least 1", file=sys.stderr)
        return 2

    succeeded, skipped, failed = run_batch(
        STARS,
        include_false_candidates=args.include_false_candidates,
        label_output_candidates=args.label_output_candidates,
        out_dir=args.out_dir,
        confirmed_dir=args.confirmed_dir,
        workers=args.workers,
    )

    print("\n=== Batch extraction complete ===")
    print(f"Succeeded: {len(succeeded)} {sorted(succeeded)}")
    print(f"Skipped:   {len(skipped)} {sorted(skipped)}")
    print(f"Failed:    {len(failed)} {sorted(failed)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
