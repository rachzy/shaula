#!/usr/bin/env python3
"""Second look at rows labelled FALSE-POSITIVE only because no TOI/KOI matched.

``label_candidate_rows`` calls every row that pairs with no reference signal a
false positive (``fp_source`` ``unmatched``). Two kinds of those rows are not
safe negatives for training:

- **alias**: the row is a catalogued signal of the same star found at a
  harmonic the period matcher does not pair (a simple ratio such as 2/3), or
  at a nearby period with a similar depth when that signal has no row of its
  own. The row then follows the reference: a planet that no other row
  recovered becomes ``CONFIRMED``, a false alarm becomes ``FALSE-POSITIVE``
  with its catalogue ``fp_source``, and anything else (a duplicate of a planet
  already recovered, a plain false positive, an open candidate) becomes
  ``UNKNOWN``.
- **viable**: nothing in the light curve argues against a planet. The row
  becomes ``UNKNOWN``, because the catalogue cannot say it is false.

Everything else keeps its label. The original files are never changed: the
relabelled copies go to a second directory, with the first label kept in
``candidate_label_original`` and the reason in ``relabel_reason``.

Usage (from the repo root)::

    python -m shaula.scripts.relabel_unmatched data/extracted/tess-train-20261110 \
        --targets data/targets/tess_train_fold1.csv data/targets/tess_train_fold2.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ..utils.label_candidates import (
    CANDIDATE_LABEL_CONFIRMED,
    CANDIDATE_LABEL_UNKNOWN,
    FP_SOURCE_UNMATCHED,
    _reference_label,
)
from .build_koi_targets import portable_path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Harmonics the period matcher leaves unpaired. Kept to small integers and a
# tight tolerance: with ratios up to 5/4 and 2% about half the "aliases" on the
# first 94 TESS training stars were coincidences.
ALIAS_MAX_INTEGER = 3
ALIAS_PERIOD_TOLERANCE = 0.01
# TIC 2041563029: TOI-1427.01 (12.81 d, 2.7%) was found at 13.58 d with a 3.2%
# depth and MES 389. Same eclipse, period 6% off, so not a harmonic.
NEARBY_PERIOD_TOLERANCE = 0.10
NEARBY_DEPTH_FACTOR = 2.0

# A row is planet-viable only when it passes every test. On the first 181 TESS
# training stars these kept 20 of 23 recovered confirmed planets and passed
# inverted-pass rows (never planets) at the same rate as unmatched rows.
VIABLE_MIN_MES = 10.0
VIABLE_MAX_SINGLE_EVENT_SHARE = 0.85   # max_ses / max_mes: one event carries it
VIABLE_MAX_LOG_RHO_RATIO = 1.0         # transit density within 10x of the star's
VIABLE_INVERTED_MARGIN = 1.3           # above the star's own inverted-pass noise
EB_MIN_DEPTH = 0.02
EB_MIN_RADIUS_RJUP = 2.0
EB_MIN_ODD_EVEN_SIGMA = 3.0
EB_MIN_SECONDARY_SNR = 5.0

REASON_ALIAS = "alias"
REASON_NEARBY = "nearby_period"
REASON_VIABLE = "viable"


def _num(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _alias_ratio(period: float, reference_period: float) -> tuple[str, float] | None:
    """The simple ratio ``m/n`` with ``period ~ m/n x reference``, if any."""
    if not (period > 0 and reference_period > 0):
        return None
    for m in range(1, ALIAS_MAX_INTEGER + 1):
        for n in range(1, ALIAS_MAX_INTEGER + 1):
            if m == n or math.gcd(m, n) != 1:
                continue
            if abs(period / reference_period * n / m - 1.0) <= ALIAS_PERIOD_TOLERANCE:
                return f"{m}/{n}", period / reference_period
    return None


def _is_eb_like(row: dict) -> bool:
    return (
        _num(row.get("depth_mean_per_transit")) > EB_MIN_DEPTH
        or _num(row.get("planet_radius_rjup")) > EB_MIN_RADIUS_RJUP
        or _num(row.get("even_odd_depth_diff_sigma")) > EB_MIN_ODD_EVEN_SIGMA
        or _num(row.get("secondary_depth_snr")) > EB_MIN_SECONDARY_SNR
    )


def is_planet_viable(row: dict, inverted_best_mes: float = math.nan) -> bool:
    """True when no test argues against the row being a planet."""
    mes = _num(row.get("max_mes"))
    if row.get("detection_status") != "accepted" or not mes >= VIABLE_MIN_MES:
        return False
    if _num(row.get("max_ses")) / mes > VIABLE_MAX_SINGLE_EVENT_SHARE:
        return False
    if abs(_num(row.get("log_rho_ratio"))) >= VIABLE_MAX_LOG_RHO_RATIO:
        return False  # NaN (no catalogue density) compares False and passes
    if mes < VIABLE_INVERTED_MARGIN * inverted_best_mes:
        return False  # NaN (no inverted pass) compares False and passes
    return not _is_eb_like(row)


def _reference_outcome(reference: pd.Series, already_recovered: bool) -> tuple[str, str]:
    """Label and fp_source for a row that is this reference signal again."""
    label, fp_source = _reference_label(reference)
    if label == CANDIDATE_LABEL_CONFIRMED and already_recovered:
        return CANDIDATE_LABEL_UNKNOWN, ""  # a second row of a planet already found
    return label, fp_source


def relabel_unmatched_rows(
    rows: pd.DataFrame,
    reference: pd.DataFrame | None,
    inverted_best_mes: float = math.nan,
) -> pd.DataFrame:
    """Return ``rows`` with unmatched alias and viable rows relabelled.

    Adds ``candidate_label_original`` and ``relabel_reason`` (empty when the
    row is unchanged). Rows with any other label or ``fp_source`` pass through.
    """
    out = rows.copy()
    out["candidate_label_original"] = out.get("candidate_label", "")
    out["relabel_reason"] = ""
    if out.empty or "fp_source" not in out.columns:
        return out
    for column in ("matched_target", "fp_source"):
        out[column] = out[column].astype(object)

    matched = set(out["matched_target"].dropna().astype(str))
    reference = reference if reference is not None else pd.DataFrame()
    for index, row in out.iterrows():
        if row.get("fp_source") != FP_SOURCE_UNMATCHED:
            continue
        period = _num(row.get("period_days"))
        depth = _num(row.get("depth_mean_per_transit"))
        hit = None
        for _, ref in reference.iterrows():
            ref_period = _num(ref.get("period_days"))
            alias = _alias_ratio(period, ref_period)
            if alias is not None:
                hit = (REASON_ALIAS, ref, alias[1])
                break
            ref_depth = _num(ref.get("depth_mean_per_transit"))
            if (
                str(ref.get("target")) not in matched
                and ref_period > 0
                and abs(period / ref_period - 1.0) <= NEARBY_PERIOD_TOLERANCE
                and depth > 0
                and ref_depth > 0
                and max(depth / ref_depth, ref_depth / depth) <= NEARBY_DEPTH_FACTOR
            ):
                hit = (REASON_NEARBY, ref, period / ref_period)
                break
        if hit is not None:
            reason, ref, ratio = hit
            target = str(ref.get("target"))
            label, fp_source = _reference_outcome(ref, already_recovered=target in matched)
            out.at[index, "candidate_label"] = label
            out.at[index, "matched_target"] = target
            out.at[index, "matched_period_ratio"] = ratio
            out.at[index, "fp_source"] = fp_source or None
            out.at[index, "relabel_reason"] = reason
            if label == CANDIDATE_LABEL_CONFIRMED:
                matched.add(target)
        elif is_planet_viable(row.to_dict(), inverted_best_mes):
            out.at[index, "candidate_label"] = CANDIDATE_LABEL_UNKNOWN
            out.at[index, "fp_source"] = None
            out.at[index, "relabel_reason"] = REASON_VIABLE
    return out


def _read_rows(path: Path) -> pd.DataFrame:
    """An extraction CSV; a star with no detections is a near-empty file."""
    if path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest(directory: Path, stem: str) -> Path | None:
    """The newest dated ``<stem>_<YYYYMMDD>.csv`` in ``directory``."""
    paths = [p for p in directory.glob(f"{stem}_*.csv") if p.stem[len(stem) + 1 :].isdigit()]
    return max(paths, default=None)


def relabel_directory(directory: Path, targets: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Relabel every finished normal-pass CSV of ``targets``; return the changes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    changes = []
    for target in targets.to_dict("records"):
        stem = str(target["target"]).replace(" ", "")
        path = _latest(directory, stem)
        if path is None:
            continue
        rows = _read_rows(path)
        if rows.empty:
            (out_dir / path.name).write_bytes(path.read_bytes())
            continue
        reference_path = Path(str(target.get("reference") or ""))
        if not reference_path.is_absolute():
            reference_path = REPO_ROOT / reference_path
        reference = pd.read_csv(reference_path) if reference_path.is_file() else None
        inverted_path = _latest(directory, f"{stem}_inverted")
        inverted_best = math.nan
        if inverted_path is not None:
            inverted = _read_rows(inverted_path)
            inverted_best = 0.0 if inverted.empty else float(inverted["max_mes"].max())
        relabelled = relabel_unmatched_rows(rows, reference, inverted_best)
        relabelled.to_csv(out_dir / path.name, index=False)
        changed = relabelled[relabelled["relabel_reason"] != ""]
        for row in changed.to_dict("records"):
            changes.append(
                {
                    "file": path.name,
                    "period_days": row["period_days"],
                    "max_mes": row["max_mes"],
                    "relabel_reason": row["relabel_reason"],
                    "matched_target": row.get("matched_target"),
                    "matched_period_ratio": row.get("matched_period_ratio"),
                    "candidate_label_original": row["candidate_label_original"],
                    "candidate_label": row["candidate_label"],
                }
            )
    return pd.DataFrame(changes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="Directory of extraction CSVs")
    parser.add_argument(
        "--targets",
        type=Path,
        nargs="+",
        required=True,
        help="Target list CSV(s) with target and reference columns",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Where the relabelled copies go (default: <directory>-relabelled)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or args.directory.with_name(args.directory.name + "-relabelled")
    targets = pd.concat([pd.read_csv(path, dtype=str).fillna("") for path in args.targets])
    changes = relabel_directory(args.directory, targets, out_dir)
    changes.to_csv(out_dir / "relabel_changes.csv", index=False)
    print(f"Relabelled copies in {portable_path(out_dir)}; {len(changes)} rows changed.")
    if not changes.empty:
        counts = changes.groupby(["relabel_reason", "candidate_label"]).size()
        print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
