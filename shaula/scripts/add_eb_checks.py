#!/usr/bin/env python3
"""Add the eclipsing-binary check columns to every feature CSV in a directory.

Feature CSVs written before ``sesmes.eb_check_features`` existed lack
``EB_CHECK_COLUMNS`` (the odd/even depth significance and the all-phase
secondary search). This script fills them with the same function the pipeline
now uses, through ``extract_feats.measure_eb_checks``: the star's light curve
is detrended around each row's ``period_days``, ``t0`` and ``duration_days``
and the SES/MES statistics are rebuilt, with no period search. The other
accepted rows of the same CSV are gapped, as they were in the search. Values
can differ slightly from a fresh full run, because the detrend is rebuilt.

The light curve is loaded the way ``extract_and_compare`` loads it (every
quarter or sector stitched, upward outliers clipped at 5 sigma), from the
lightkurve cache when it is there and downloaded otherwise. An
``_inverted_`` CSV gets the same cut around its star's reference KOIs
(``--reference-dir``, ``<star>-koi.csv``) and the same flip as the inverted
run that wrote it.

The star comes from the file name (``KIC<id>_...``, ``EPIC<id>_...`` or
``TIC<id>_...``), so a star's normal and inverted CSVs share one download.
Existing columns are left untouched; the new columns are appended, or
overwritten in place when ``--overwrite`` is given. CSVs that already have
every column are skipped otherwise, so an interrupted run can be restarted.
Empty CSVs are skipped.

Usage (from the repo root)::

    python -m shaula.scripts.add_eb_checks data/extracted --workers 7
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..extract_feats import (
    DETECTION_STATUS_ACCEPTED,
    TRANSIT_MASK_WIDTH,
    _periodic_mask,
    measure_eb_checks,
)
from ..paths import lightcurve_cache_dir
from ..sesmes import EB_CHECK_COLUMNS

STAR_FROM_FILENAME = re.compile(r"^(KIC|EPIC|TIC)(\d+)_")
MISSION_BY_PREFIX = {"KIC": "Kepler", "EPIC": "K2", "TIC": "TESS"}
INVERTED_MARKER = "_inverted_"


def star_of(path: Path) -> tuple[str, int] | None:
    """``("KIC", 11904151)`` for ``KIC11904151_inverted_20261020.csv``."""
    match = STAR_FROM_FILENAME.match(path.name)
    return (match.group(1), int(match.group(2))) if match else None


def load_lightcurve(prefix: str, number: int, cache_dir: Path | None):
    """The star's cleaned, stitched light curve, as the extraction run saw it."""
    from ..download_and_clean import download_and_clean_lightcurve
    from .network import install_default_request_timeout, with_retries

    install_default_request_timeout()
    return with_retries(
        lambda: download_and_clean_lightcurve(
            target=f"{prefix} {number}",
            mission=MISSION_BY_PREFIX[prefix],
            all=True,
            cache_dir=cache_dir,
        )
    )


def inverted_lightcurve(lc, prefix: str, number: int, reference_dir: Path | None):
    """The flipped light curve an inverted run searched, known signals cut."""
    from .extract_and_compare import invert_lightcurve, remove_known_signals

    if reference_dir is not None:
        reference = reference_dir / f"{prefix}{number}-koi.csv"
        if reference.is_file():
            lc, _ = remove_known_signals(
                lc, pd.read_csv(reference), MISSION_BY_PREFIX[prefix]
            )
    return invert_lightcurve(lc)


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def eb_checks_for_rows(time, flux, rows: list[dict]) -> list[dict]:
    """``EB_CHECK_COLUMNS`` for every row of one CSV on one light curve.

    Each row is measured with the transits of the CSV's other accepted rows
    gapped. The search gapped only the candidates accepted before it, but the
    CSV is sorted by period and no longer records that order, so all of them
    are; for a provisional row that is exactly what the search did.
    """
    accepted = [
        index
        for index, row in enumerate(rows)
        if row.get("detection_status", DETECTION_STATUS_ACCEPTED)
        == DETECTION_STATUS_ACCEPTED
    ]
    windows = {}
    for index in accepted:
        row = rows[index]
        period = _finite(row.get("period_days"))
        t0 = _finite(row.get("t0"))
        duration = _finite(row.get("duration_days"))
        if np.isfinite(period) and period > 0 and np.isfinite(t0) and duration > 0:
            windows[index] = _periodic_mask(
                time, period, t0, duration, TRANSIT_MASK_WIDTH
            )
    values = []
    for index, row in enumerate(rows):
        gap = np.zeros(np.shape(time), dtype=bool)
        for other, window in windows.items():
            if other != index:
                gap |= window
        values.append(
            measure_eb_checks(
                time,
                flux,
                _finite(row.get("period_days")),
                _finite(row.get("t0")),
                _finite(row.get("duration_days")),
                gap_mask=gap,
            )
        )
    return values


def needs_filling(path: Path, overwrite: bool) -> bool:
    """False for an empty CSV, or one already filled unless ``overwrite``."""
    if path.stat().st_size < 5:
        return False
    try:
        columns = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError:
        return False
    return overwrite or not all(column in columns for column in EB_CHECK_COLUMNS)


def fill_csv(path: Path, lc) -> int:
    """Write the EB check columns into one CSV; return the number of rows."""
    # round_trip keeps every existing value byte-for-byte when it is rewritten.
    df = pd.read_csv(path, float_precision="round_trip")
    if df.empty:
        return 0
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    # The pipeline prints every step; one line per star is enough here.
    with contextlib.redirect_stdout(io.StringIO()):
        values = eb_checks_for_rows(time, flux, df.to_dict("records"))
    for column in EB_CHECK_COLUMNS:
        df[column] = [v[column] for v in values]
    # Write beside the target and rename, so an interrupted run never leaves
    # a half-written CSV.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            df.to_csv(handle, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(df)


def process_fold(
    fold: list[tuple[tuple[str, int], list[str]]],
    cache_dir: str | None,
    reference_dir: str | None,
    overwrite: bool,
    label: str,
) -> dict:
    """Fill every CSV of the stars in one fold; return counts and failures."""
    from .extract_multiple_stars import _single_threaded_math

    with _single_threaded_math():
        return _process_fold(fold, cache_dir, reference_dir, overwrite, label)


def _process_fold(fold, cache_dir, reference_dir, overwrite, label):
    cache = Path(cache_dir) if cache_dir else None
    references = Path(reference_dir) if reference_dir else None
    summary = {"files": 0, "rows": 0, "skipped": 0, "failed": []}
    for done, ((prefix, number), names) in enumerate(fold, start=1):
        paths = [Path(name) for name in names]
        todo = [path for path in paths if needs_filling(path, overwrite)]
        summary["skipped"] += len(paths) - len(todo)
        if not todo:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                lc = load_lightcurve(prefix, number, cache)
        except Exception as exc:
            summary["failed"].append(f"{prefix}{number}: {exc!r}")
            print(f"[{label}] {prefix}{number}: FAILED {exc!r}", flush=True)
            continue
        rows = 0
        for path in todo:
            try:
                if INVERTED_MARKER in path.name:
                    with contextlib.redirect_stdout(io.StringIO()):
                        source = inverted_lightcurve(lc, prefix, number, references)
                else:
                    source = lc
                rows += fill_csv(path, source)
                summary["files"] += 1
            except Exception as exc:
                summary["failed"].append(f"{path.name}: {exc!r}")
                print(f"[{label}] {path.name}: FAILED {exc!r}", flush=True)
        summary["rows"] += rows
        print(
            f"[{label}] {done}/{len(fold)} {prefix}{number}: "
            f"{len(todo)} CSV(s), {rows} rows",
            flush=True,
        )
    return summary


def split_into_folds(directory: Path, folds: int):
    """Group the directory's CSVs by star, then deal the stars into folds."""
    by_star: dict[tuple[str, int], list[str]] = defaultdict(list)
    skipped = []
    for path in sorted(directory.glob("*.csv")):
        star = star_of(path)
        if star is None:
            skipped.append(path.name)
        else:
            by_star[star].append(str(path))
    stars = sorted(by_star.items())
    # Round-robin keeps the folds the same size to within one star.
    return [stars[i::folds] for i in range(folds)], skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="Directory of feature CSVs")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Split the stars into this many folds and fill them in parallel",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Light-curve download cache; defaults to SHAULA_LIGHTCURVE_CACHE, "
            "then lightkurve's own cache"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/koi"),
        help=(
            "Per-star KOI tables (<star>-koi.csv) whose signals the inverted "
            "runs cut before flipping (default: data/koi)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute CSVs that already have every EB check column",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")
    workers = max(1, args.workers)
    cache_dir = args.cache_dir or lightcurve_cache_dir()
    folds, skipped = split_into_folds(args.directory, workers)
    folds = [fold for fold in folds if fold]
    print(
        f"{sum(len(f) for f in folds)} stars in {len(folds)} fold(s); "
        f"light-curve cache: {cache_dir or 'lightkurve default'}",
        flush=True,
    )
    for name in skipped:
        print(f"Skipping {name}: no KIC/EPIC/TIC id in the file name")

    cache_arg = str(cache_dir) if cache_dir is not None else None
    reference_arg = str(args.reference_dir) if args.reference_dir else None
    summaries = []
    if len(folds) == 1:
        summaries.append(
            process_fold(folds[0], cache_arg, reference_arg, args.overwrite, "fold 1")
        )
    else:
        with ProcessPoolExecutor(max_workers=len(folds)) as pool:
            futures = [
                pool.submit(
                    process_fold,
                    fold,
                    cache_arg,
                    reference_arg,
                    args.overwrite,
                    f"fold {i}",
                )
                for i, fold in enumerate(folds, start=1)
            ]
            for future in as_completed(futures):
                summaries.append(future.result())

    failed = [item for s in summaries for item in s["failed"]]
    print(
        f"\nFilled {sum(s['rows'] for s in summaries)} rows in "
        f"{sum(s['files'] for s in summaries)} CSVs "
        f"({sum(s['skipped'] for s in summaries)} empty or already filled, "
        f"{len(failed)} failures)."
    )
    for item in failed:
        print(f"  FAILED {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
