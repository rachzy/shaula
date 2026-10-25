#!/usr/bin/env python3
"""Add the stellar-density columns to every feature CSV in a directory.

Feature CSVs written before ``utils/stellar_density`` existed lack
``rho_star_transit`` and ``log_rho_ratio``. This
script fills them with the same function the pipeline now uses, so old and new
rows agree.

The transit density needs only ``period_days``, ``duration_days`` and
``depth_mean_per_transit``, which every CSV has. The catalogue density also
needs the star's log g and radius, which older CSVs do not carry, so they are
read from the primary header of one of the star's light-curve files, exactly
as ``extract_features_from_lightcurve`` reads them from ``lc.meta``. Files are
looked up in the lightkurve cache first; a star with no cached file gets one
file downloaded (one quarter or sector, not the whole light curve) unless
``--no-download`` is given.

The star comes from the file name (``KIC<id>_...``, ``EPIC<id>_...`` or
``TIC<id>_...``), and a normal and an ``_inverted_`` CSV of the same star get
the same catalogue values. Existing columns are left untouched; the two
columns are appended, or overwritten in place when a CSV already has them.
Empty CSVs are skipped.

Usage (from the repo root)::

    python -m shaula.scripts.add_stellar_density data/extracted --workers 7
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..paths import lightcurve_cache_dir
from ..utils.stellar_density import STELLAR_DENSITY_COLUMNS, stellar_density_features
from ..utils.stellar_properties import stellar_parameters

STAR_FROM_FILENAME = re.compile(r"^(KIC|EPIC|TIC)(\d+)_")
MISSION_BY_PREFIX = {"KIC": "Kepler", "EPIC": "K2", "TIC": "TESS"}
# The id as it appears in the archive's light-curve file names.
FITS_ID_PATTERNS = {
    "KIC": re.compile(r"^kplr(\d{9})-\d+_llc\.fits$"),
    "EPIC": re.compile(r"^ktwo(\d{9})-c\d+_llc\.fits$"),
    "TIC": re.compile(r"^tess\d+-s\d+-(\d{16})-\d+-s_lc\.fits$"),
}


def star_of(path: Path) -> tuple[str, int] | None:
    """``("KIC", 11904151)`` for ``KIC11904151_inverted_20261020.csv``."""
    match = STAR_FROM_FILENAME.match(path.name)
    return (match.group(1), int(match.group(2))) if match else None


def default_cache_dir() -> Path | None:
    """``SHAULA_LIGHTCURVE_CACHE``, else lightkurve's own cache directory."""
    explicit = lightcurve_cache_dir()
    if explicit is not None:
        return explicit
    try:
        import lightkurve as lk

        return Path(lk.config.get_cache_dir())
    except Exception:
        return None


def index_cached_fits(cache_dir: Path | None) -> dict[tuple[str, int], Path]:
    """Map each star in the cache to one of its light-curve files."""
    found: dict[tuple[str, int], Path] = {}
    if cache_dir is None or not cache_dir.is_dir():
        return found
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            for prefix, pattern in FITS_ID_PATTERNS.items():
                match = pattern.match(name)
                if match:
                    found.setdefault((prefix, int(match.group(1))), Path(root) / name)
    return found


def header_star(path: Path) -> dict:
    """``stellar_parameters`` from a light-curve file's primary header."""
    from astropy.io import fits

    return stellar_parameters(fits.getheader(path, 0))


def downloaded_star(prefix: str, number: int, cache_dir: Path | None) -> dict:
    """Download one light-curve file of the star and read its header."""
    from ..download_and_clean import search_default_product

    search = search_default_product(
        f"{prefix} {number}", MISSION_BY_PREFIX[prefix]
    ).result
    if len(search) == 0:
        raise LookupError(f"no light curve found for {prefix} {number}")
    download_dir = str(cache_dir) if cache_dir is not None else None
    lc = search[0].download(download_dir=download_dir)
    return stellar_parameters(lc.meta)


def fill_csv(path: Path, star: dict) -> int:
    """Write the density columns into one CSV; return the number of rows."""
    # round_trip keeps every existing value byte-for-byte when it is rewritten.
    df = pd.read_csv(path, float_precision="round_trip")
    values = [
        stellar_density_features(
            row.get("period_days"),
            row.get("duration_days"),
            row.get("depth_mean_per_transit"),
            star["stellar_logg"],
            star["stellar_radius_rsun"],
        )
        for row in df.to_dict("records")
    ]
    for column in STELLAR_DENSITY_COLUMNS:
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
    download: bool,
    label: str,
) -> dict:
    """Fill every CSV of the stars in one fold; return counts and failures."""
    cache = Path(cache_dir) if cache_dir else None
    cached = index_cached_fits(cache)
    summary = {"files": 0, "rows": 0, "empty": 0, "downloaded": 0, "failed": []}
    for done, ((prefix, number), paths) in enumerate(fold, start=1):
        try:
            if (prefix, number) in cached:
                star = header_star(cached[(prefix, number)])
            elif download:
                star = downloaded_star(prefix, number, cache)
                summary["downloaded"] += 1
            else:
                raise LookupError("no cached light-curve file (--no-download)")
        except Exception as exc:
            summary["failed"].append(f"{prefix}{number}: {exc!r}")
            print(f"[{label}] {prefix}{number}: FAILED {exc!r}", flush=True)
            continue
        for name in paths:
            path = Path(name)
            if path.stat().st_size < 5:
                summary["empty"] += 1
                continue
            try:
                summary["rows"] += fill_csv(path, star)
                summary["files"] += 1
            except Exception as exc:
                summary["failed"].append(f"{path.name}: {exc!r}")
                print(f"[{label}] {path.name}: FAILED {exc!r}", flush=True)
        print(
            f"[{label}] {done}/{len(fold)} {prefix}{number}: "
            f"logg={star['stellar_logg']:.3f} R={star['stellar_radius_rsun']:.3f}",
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
            "Where to look for (and download) light-curve files; defaults to "
            "SHAULA_LIGHTCURVE_CACHE, then lightkurve's cache"
        ),
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Fail a star with no cached light-curve file instead of downloading one",
    )
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")
    workers = max(1, args.workers)
    cache_dir = args.cache_dir or default_cache_dir()
    folds, skipped = split_into_folds(args.directory, workers)
    folds = [fold for fold in folds if fold]
    print(
        f"{sum(len(f) for f in folds)} stars in {len(folds)} fold(s); "
        f"light-curve cache: {cache_dir}",
        flush=True,
    )
    for name in skipped:
        print(f"Skipping {name}: no KIC/EPIC/TIC id in the file name")

    cache_arg = str(cache_dir) if cache_dir is not None else None
    summaries = []
    if len(folds) == 1:
        summaries.append(process_fold(folds[0], cache_arg, args.download, "fold 1"))
    else:
        with ProcessPoolExecutor(max_workers=len(folds)) as pool:
            futures = [
                pool.submit(process_fold, fold, cache_arg, args.download, f"fold {i}")
                for i, fold in enumerate(folds, start=1)
            ]
            for future in as_completed(futures):
                summaries.append(future.result())

    failed = [item for s in summaries for item in s["failed"]]
    print(
        f"\nFilled {sum(s['rows'] for s in summaries)} rows in "
        f"{sum(s['files'] for s in summaries)} CSVs "
        f"({sum(s['empty'] for s in summaries)} empty skipped, "
        f"{sum(s['downloaded'] for s in summaries)} stars downloaded, "
        f"{len(failed)} failures)."
    )
    for item in failed:
        print(f"  FAILED {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
