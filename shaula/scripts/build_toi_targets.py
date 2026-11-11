#!/usr/bin/env python3
"""Build a TESS target list and per-star TOI reference tables.

The TESS counterpart of ``build_koi_targets``. One query of the NASA Exoplanet
Archive TOI table gives every TESS Object of Interest of every star. From it
this writes:

- ``data/toi/TIC<tid>-toi.csv`` for each selected star: every TOI of that
  star in the reference-table format ``label_candidate_rows`` reads (the
  confirmed-CSV columns plus ``disposition``, ``fp_flags`` and ``catalog``).
- a target list CSV (``target,mission,group,name,reference``) that
  ``extract_multiple_stars --targets`` runs over, plainly or with ``--invert``.

TFOPWG dispositions are mapped onto the reference vocabulary:

- ``CP`` (confirmed planet) and ``KP`` (known planet) -> ``CONFIRMED``
- ``PC`` and ``APC`` (planet candidate, ambiguous) -> ``CANDIDATE``
- ``FA`` (false alarm) -> ``FALSE POSITIVE`` with ``fp_flags`` ``fa``
- ``FP`` (false positive) -> ``FALSE POSITIVE`` with ``fp_flags`` ``fp``

The TOI table has no counterpart of the KOI ``nt``/``ss``/``co``/``ec``
flags. A false alarm is a signal that is not astrophysical, so ``fa`` labels
a matching row ``FALSE-POSITIVE`` like ``nt`` does. An ``FP`` does not say
whether the eclipse is on the target or on a neighbour blended into TESS's
21 arcsecond pixels, so a row matching one is ``UNKNOWN``, as a Kepler
centroid-only false positive is.

Two groups of stars are selected:

- ``confirmed_host``: stars with at least ``--min-planets`` ``CP``/``KP`` TOIs.
- ``toi_fp``: stars with at least one ``FA`` or ``FP`` TOI and no planet.

Both are seeded random samples when a limit is given. Targets are written as
``TIC <tid>``, which MAST resolves without a name lookup. The TOI table does
not say which stars have 2-minute SPOC light curves, so some targets may have
none; the extraction reports those and moves on.

Usage (from the repo root with the project venv active)::

    python -m shaula.scripts.build_toi_targets --host-limit 100 --fp-limit 100
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .build_koi_targets import TARGET_COLUMNS, portable_path
from .get_literature_data import (
    CONFIRMED_COLUMNS,
    EARTH_RADII_PER_JUPITER_RADIUS,
    PPM_PER_UNITY,
    REFERENCE_COLUMNS,
    TAP_SYNC_URL,
    LiteratureDataError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOI_DIR = REPO_ROOT / "data" / "toi"
DEFAULT_TARGETS_PATH = REPO_ROOT / "data" / "targets" / "tess_targets.csv"

TOI_SELECT_COLUMNS = (
    "tid",
    "toi",
    "tfopwg_disp",
    "pl_orbper",
    "pl_tranmid",
    "pl_trandurh",
    "pl_trandep",
    "pl_rade",
    "st_rad",
    "st_tmag",
)

# TFOPWG disposition -> (reference disposition, fp_flags).
TOI_DISPOSITIONS = {
    "CP": ("CONFIRMED", ""),
    "KP": ("CONFIRMED", ""),
    "PC": ("CANDIDATE", ""),
    "APC": ("CANDIDATE", ""),
    "FA": ("FALSE POSITIVE", "fa"),
    "FP": ("FALSE POSITIVE", "fp"),
}
PLANET_DISPOSITIONS = ("CP", "KP")
FALSE_DISPOSITIONS = ("FA", "FP")

CATALOG_TOI = "toi"
TOI_REFERENCE_COLUMNS = [*REFERENCE_COLUMNS, "catalog"]

GROUP_CONFIRMED_HOST = "confirmed_host"
GROUP_TOI_FALSE_POSITIVE = "toi_fp"


def fetch_toi_table(*, base_url: str = TAP_SYNC_URL) -> pd.DataFrame:
    """Every row of the TOI table, with the columns references need."""
    query = f"select {', '.join(TOI_SELECT_COLUMNS)} from toi"
    url = f"{base_url}?{urlencode({'query': query, 'format': 'csv'})}"
    try:
        return pd.read_csv(url)
    except Exception as exc:  # network error, bad response, malformed CSV, ...
        raise LiteratureDataError(f"Failed to query the TOI table: {exc}") from exc


def tic_target(tid: int) -> str:
    return f"TIC {int(tid)}"


def reference_filename(tid: int) -> str:
    return f"TIC{int(tid)}-toi.csv"


def toi_name(toi) -> str:
    """``270.01`` -> ``TOI-270.01`` (the archive stores the number alone)."""
    return f"TOI-{float(toi):.2f}"


def _disposition(raw: pd.DataFrame) -> pd.Series:
    return raw["tfopwg_disp"].fillna("").astype(str).str.strip().str.upper()


def toi_rows_to_reference(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert every TOI row of a star into the reference-table format.

    Periods of 0 (single-transit TOIs) become NaN so they pair with nothing.
    ``t0`` is the archive's full BJD; the comparison tooling moves it to BTJD.
    The TOI table has no radius ratio, impact parameter or MES, so those
    columns are NaN.
    """
    if raw.empty:
        return pd.DataFrame(columns=TOI_REFERENCE_COLUMNS)

    def number(column: str) -> pd.Series:
        return pd.to_numeric(raw.get(column), errors="coerce")

    out = pd.DataFrame(index=raw.index, columns=CONFIRMED_COLUMNS, dtype=float)
    out["target"] = raw["toi"].map(toi_name)
    period = number("pl_orbper")
    out["period_days"] = period.where(period > 0)
    out["t0"] = number("pl_tranmid")
    out["duration_hours"] = number("pl_trandurh")
    out["duration_days"] = out["duration_hours"] / 24.0
    out["depth_mean_per_transit"] = number("pl_trandep") / PPM_PER_UNITY
    out["planet_radius_rjup"] = number("pl_rade") / EARTH_RADII_PER_JUPITER_RADIUS
    out["stellar_radius"] = number("st_rad")

    mapped = _disposition(raw).map(TOI_DISPOSITIONS)
    out["disposition"] = mapped.map(lambda pair: pair[0] if isinstance(pair, tuple) else "")
    out["fp_flags"] = mapped.map(lambda pair: pair[1] if isinstance(pair, tuple) else "")
    out["catalog"] = CATALOG_TOI
    out = out[TOI_REFERENCE_COLUMNS]
    out = out.drop_duplicates(subset="target", keep="first")
    return out.sort_values("period_days", kind="stable").reset_index(drop=True)


def _sample(tids: list[int], limit: int | None, rng: np.random.Generator) -> list[int]:
    if limit is None or limit >= len(tids):
        return tids
    return sorted(rng.choice(tids, size=limit, replace=False).tolist())


def planet_host_tids(toi: pd.DataFrame, min_planets: int = 1) -> list[int]:
    """Stars with at least ``min_planets`` confirmed or known planet TOIs."""
    planets = toi[_disposition(toi).isin(PLANET_DISPOSITIONS)]
    counts = planets.groupby(planets["tid"].astype(int)).size()
    return sorted(counts[counts >= int(min_planets)].index.tolist())


def false_positive_tids(toi: pd.DataFrame) -> list[int]:
    """Stars with a false alarm or false positive TOI and no planet TOI."""
    disposition = _disposition(toi)
    tids = toi["tid"].astype(int)
    false = set(tids[disposition.isin(FALSE_DISPOSITIONS)])
    hosts = set(tids[disposition.isin(PLANET_DISPOSITIONS)])
    return sorted(false - hosts)


def build_targets(
    toi: pd.DataFrame,
    *,
    host_limit: int | None,
    fp_limit: int | None,
    min_planets: int,
    seed: int,
    toi_dir: Path,
) -> pd.DataFrame:
    """Write one reference CSV per selected star and return the target list."""
    toi_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    selected = [
        (tid, GROUP_CONFIRMED_HOST)
        for tid in _sample(planet_host_tids(toi, min_planets), host_limit, rng)
    ] + [
        (tid, GROUP_TOI_FALSE_POSITIVE)
        for tid in _sample(false_positive_tids(toi), fp_limit, rng)
    ]

    tids = toi["tid"].astype(int)
    targets = []
    for tid, group in selected:
        reference = toi_rows_to_reference(toi[tids == tid])
        path = toi_dir / reference_filename(tid)
        reference.to_csv(path, index=False)
        targets.append(
            {
                "target": tic_target(tid),
                "mission": "TESS",
                "group": group,
                "name": ", ".join(reference["target"].astype(str)),
                "reference": portable_path(path),
            }
        )
    return pd.DataFrame(targets, columns=TARGET_COLUMNS)


def _limit(value: int) -> int | None:
    return None if value < 0 else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--host-limit",
        type=int,
        default=100,
        help="Planet hosts to sample; 0 for none, -1 for all (default: 100)",
    )
    parser.add_argument(
        "--fp-limit",
        type=int,
        default=100,
        help="False alarm/positive stars to sample; 0 for none, -1 for all (default: 100)",
    )
    parser.add_argument(
        "--min-planets",
        type=int,
        default=1,
        help="Confirmed or known planet TOIs a host needs (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument("--toi-dir", type=Path, default=DEFAULT_TOI_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_TARGETS_PATH)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[], pd.DataFrame] = fetch_toi_table,
) -> int:
    args = parse_args(argv)
    try:
        toi = fetch()
    except LiteratureDataError as exc:
        print(exc, file=sys.stderr)
        return 1

    targets = build_targets(
        toi,
        host_limit=_limit(args.host_limit),
        fp_limit=_limit(args.fp_limit),
        min_planets=args.min_planets,
        seed=args.seed,
        toi_dir=args.toi_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out, index=False)

    counts = targets["group"].value_counts().to_dict()
    print(f"Wrote {len(targets)} targets to {args.out}: {counts}")
    print(f"Reference tables in {args.toi_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
