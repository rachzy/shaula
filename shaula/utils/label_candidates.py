"""Label extracted candidate rows against the confirmed catalog.

The decision is delegated to :func:`match_candidate_rows`, the same pairing the
comparison report uses, so a row is labelled exactly as that report would
classify it. Matching is on orbital period alone - deliberately, since the point
of the label is to say whether a period belongs to a known planet, not whether
the recovered features are any good.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compare_extracted_confirmed import (
    PERIOD_MATCH_TOLERANCE,
    match_candidate_rows,
)

CANDIDATE_LABEL_CONFIRMED = "CONFIRMED"
CANDIDATE_LABEL_FALSE_POSITIVE = "FALSE-POSITIVE"
CANDIDATE_LABEL_UNKNOWN = "UNKNOWN"

LABEL_COLUMNS = ("candidate_label", "matched_target", "matched_period_ratio")


def _as_confirmed_frame(confirmed_rows) -> pd.DataFrame | None:
    """Accept a DataFrame, a CSV path, or nothing."""
    if confirmed_rows is None:
        return None
    if isinstance(confirmed_rows, pd.DataFrame):
        frame = confirmed_rows
    elif isinstance(confirmed_rows, (str, Path)):
        path = Path(confirmed_rows)
        if not path.is_file():
            return None
        frame = pd.read_csv(path)
    else:
        frame = pd.DataFrame(list(confirmed_rows))
    if frame.empty or "period_days" not in frame.columns:
        return None
    return frame


def label_candidate_rows(
    rows,
    confirmed_rows=None,
    *,
    tolerance: float = PERIOD_MATCH_TOLERANCE,
) -> list[dict]:
    """Stamp ``candidate_label`` on every row, in the caller's original order.

    ``CONFIRMED`` when the row's period pairs with a confirmed planet,
    ``FALSE-POSITIVE`` when it pairs with none, and ``UNKNOWN`` for every row
    when no confirmed table is available for the star.

    A pairing at a small integer ratio of the catalog period counts as
    ``CONFIRMED`` - the comparison report treats those as recoveries of the
    planet rather than as unrelated rows - and ``matched_period_ratio`` records
    which ratio it was, so callers that want strict period agreement can filter
    on ``direct``. Each confirmed planet is claimed at most once, so a second
    row landing on an already-claimed planet is labelled ``FALSE-POSITIVE``.
    """
    labelled = [dict(row) for row in rows]
    if not labelled:
        return labelled

    confirmed = _as_confirmed_frame(confirmed_rows)
    for row in labelled:
        row["candidate_label"] = (
            CANDIDATE_LABEL_UNKNOWN if confirmed is None else CANDIDATE_LABEL_FALSE_POSITIVE
        )
        row["matched_target"] = ""
        row["matched_period_ratio"] = ""
    if confirmed is None:
        return labelled

    # Built without reordering, so match_candidate_rows' positional indices
    # address the same rows the caller handed in.
    extracted = pd.DataFrame(labelled)
    for match in match_candidate_rows(extracted, confirmed, tolerance=tolerance):
        if match["kind"] not in ("direct", "alias"):
            continue
        row = labelled[match["extracted_index"]]
        row["candidate_label"] = CANDIDATE_LABEL_CONFIRMED
        row["matched_target"] = match["target"] or ""
        row["matched_period_ratio"] = match["ratio_label"]
    return labelled


def summarize_labels(rows) -> str:
    """One-line tally of the labels applied, for run logs."""
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("candidate_label", CANDIDATE_LABEL_UNKNOWN))
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "no candidate rows to label"
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
