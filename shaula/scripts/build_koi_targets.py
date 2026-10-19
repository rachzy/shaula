#!/usr/bin/env python3
"""Build a training target list and per-star KOI reference tables.

One query of the NASA Exoplanet Archive KOI cumulative table gives every KOI
of every Kepler star. From it this writes:

- ``data/koi/KIC<kepid>-koi.csv`` for each selected star: every KOI of that
  star, whatever its disposition, in the confirmed-CSV format plus
  ``disposition`` and ``fp_flags``. ``label_candidate_rows`` uses it to tell a
  planet from a vetted false positive and from an undecided candidate.
- a target list CSV (``target,mission,group,name,reference``) that
  ``extract_multiple_stars --targets`` runs over.

Two groups of stars are selected:

- ``confirmed_host``: every host with a CSV in ``data/confirmed/``, so the
  positives are extracted with the same reference tables as the negatives.
- ``koi_fp``: a seeded random sample of stars carrying at least one false
  positive KOI flagged not transit-like (``nt``) or stellar eclipse (``ss``).
  False positives flagged only for a centroid offset or ephemeris match are
  invisible in the star's own light curve and do not qualify a star.

Targets are written as ``KIC <kepid>``, which MAST resolves without a name
lookup and which names exactly the star whose KOIs were selected.

Usage (from the repo root with the project venv active)::

    python -m shaula.scripts.build_koi_targets --fp-limit 300 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from ..utils.target_names import host_star_name
from .get_literature_data import (
    KOI_SELECT_COLUMNS,
    TAP_SYNC_URL,
    LiteratureDataError,
    koi_rows_to_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIRMED_DIR = REPO_ROOT / "data" / "confirmed"
DEFAULT_KOI_DIR = REPO_ROOT / "data" / "koi"
DEFAULT_TARGETS_PATH = REPO_ROOT / "data" / "targets" / "training_targets.csv"

GROUP_CONFIRMED_HOST = "confirmed_host"
GROUP_KOI_FALSE_POSITIVE = "koi_fp"
QUALIFYING_FP_FLAGS = ("koi_fpflag_nt", "koi_fpflag_ss")
TARGET_COLUMNS = ["target", "mission", "group", "name", "reference"]


def fetch_cumulative_table(*, base_url: str = TAP_SYNC_URL) -> pd.DataFrame:
    """Every row of the KOI cumulative table, with the columns references need."""
    query = f"select {', '.join(KOI_SELECT_COLUMNS)} from cumulative"
    url = f"{base_url}?{urlencode({'query': query, 'format': 'csv'})}"
    try:
        return pd.read_csv(url)
    except Exception as exc:  # network error, bad response, malformed CSV, ...
        raise LiteratureDataError(
            f"Failed to query the KOI cumulative table: {exc}"
        ) from exc


def kic_target(kepid: int) -> str:
    return f"KIC {int(kepid)}"


def reference_filename(kepid: int) -> str:
    return f"KIC{int(kepid)}-koi.csv"


def confirmed_host_names(confirmed_dir: Path) -> list[str]:
    """Host names that have a confirmed CSV, e.g. ``Kepler-7``."""
    if not confirmed_dir.is_dir():
        return []
    return sorted(
        path.name[: -len("-confirmed.csv")]
        for path in confirmed_dir.glob("*-confirmed.csv")
    )


def host_kepids(koi: pd.DataFrame, hosts: list[str]) -> dict[str, int]:
    """Map each host name to its kepid through the KOIs' Kepler names."""
    named = koi[koi["kepler_name"].notna()]
    star_of = named["kepler_name"].astype(str).str.strip().map(
        lambda name: host_star_name(name.replace(" ", ""))
    )
    by_star = dict(zip(star_of.str.casefold(), named["kepid"].astype(int)))
    return {host: by_star[host.casefold()] for host in hosts if host.casefold() in by_star}


def false_positive_kepids(koi: pd.DataFrame) -> list[int]:
    """Stars with at least one light-curve-visible false positive KOI."""
    is_fp = koi["koi_disposition"] == "FALSE POSITIVE"
    visible = np.zeros(len(koi), dtype=bool)
    for column in QUALIFYING_FP_FLAGS:
        visible |= pd.to_numeric(koi[column], errors="coerce").fillna(0).to_numpy() > 0
    return sorted(set(koi.loc[is_fp.to_numpy() & visible, "kepid"].astype(int)))


def build_targets(
    koi: pd.DataFrame,
    *,
    hosts: list[str],
    fp_limit: int | None,
    seed: int,
    koi_dir: Path,
    include_hosts: bool = True,
) -> pd.DataFrame:
    """Write one reference CSV per selected star and return the target list."""
    koi_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    host_ids = host_kepids(koi, hosts) if include_hosts else {}
    for name, kepid in sorted(host_ids.items(), key=lambda item: item[1]):
        rows.append(
            {"kepid": kepid, "group": GROUP_CONFIRMED_HOST, "name": name}
        )

    candidates = [k for k in false_positive_kepids(koi) if k not in set(host_ids.values())]
    if fp_limit is not None and fp_limit < len(candidates):
        rng = np.random.default_rng(seed)
        candidates = sorted(rng.choice(candidates, size=fp_limit, replace=False).tolist())
    for kepid in candidates:
        rows.append({"kepid": kepid, "group": GROUP_KOI_FALSE_POSITIVE, "name": ""})

    targets = []
    for row in rows:
        kepid = int(row["kepid"])
        reference = koi_rows_to_reference(koi[koi["kepid"] == kepid])
        path = koi_dir / reference_filename(kepid)
        reference.to_csv(path, index=False)
        name = row["name"] or ", ".join(reference["target"].astype(str))
        targets.append(
            {
                "target": kic_target(kepid),
                "mission": "Kepler",
                "group": row["group"],
                "name": name,
                "reference": str(path),
            }
        )
    return pd.DataFrame(targets, columns=TARGET_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fp-limit",
        type=int,
        default=300,
        help="KOI false-positive stars to sample; 0 for none, -1 for all (default: 300)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default: 0)")
    parser.add_argument(
        "--hosts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include every host with a confirmed CSV (default: true)",
    )
    parser.add_argument("--confirmed-dir", type=Path, default=DEFAULT_CONFIRMED_DIR)
    parser.add_argument("--koi-dir", type=Path, default=DEFAULT_KOI_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_TARGETS_PATH)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    fetch: Callable[[], pd.DataFrame] = fetch_cumulative_table,
) -> int:
    args = parse_args(argv)
    try:
        koi = fetch()
    except LiteratureDataError as exc:
        print(exc, file=sys.stderr)
        return 1

    hosts = confirmed_host_names(args.confirmed_dir)
    targets = build_targets(
        koi,
        hosts=hosts,
        fp_limit=None if args.fp_limit < 0 else args.fp_limit,
        seed=args.seed,
        koi_dir=args.koi_dir,
        include_hosts=args.hosts,
    )
    missing = sorted(set(hosts) - set(targets["name"])) if args.hosts else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out, index=False)

    counts = targets["group"].value_counts().to_dict()
    print(f"Wrote {len(targets)} targets to {args.out}: {counts}")
    print(f"Reference tables in {args.koi_dir}")
    if missing:
        print(f"Hosts with no KOI in the cumulative table, left out: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
