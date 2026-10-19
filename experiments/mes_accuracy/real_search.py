"""Full iterative search on downloaded Kepler hosts, graded against the catalogue.

    python experiments/mes_accuracy/real_search.py <repo-root> <out.csv> STAR [STAR ...]

``shaula`` is imported from ``<repo-root>`` (a worktree of main, or this
branch), so both versions run on identical light curves (see
``real_stars.py download``). A detection matches a planet when its period is
within 0.5% of the catalogue period or of 2x, 3x, 1/2 or 1/3 of it.
"""

import contextlib
import io
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parents[1]))

import os  # noqa: E402

from shaula import detrend_and_period  # noqa: E402
from shaula.extract_feats import extract_features_from_arrays  # noqa: E402

if os.environ.get("SHAULA_EXP_VARWIN"):
    detrend_and_period.BLS_VARIABILITY_WINDOW_DAYS = float(os.environ["SHAULA_EXP_VARWIN"])

if os.environ.get("SHAULA_EXP_GLOBAL_DUR"):
    detrend_and_period.GLOBAL_BLS_DURATIONS = int(os.environ["SHAULA_EXP_GLOBAL_DUR"])
if os.environ.get("SHAULA_EXP_PER_BAND"):
    detrend_and_period.BLS_PEAKS_PER_BAND = int(os.environ["SHAULA_EXP_PER_BAND"])

from experiments.mes_accuracy.real_stars import CONFIRMED, DATA  # noqa: E402

RATIOS = (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0)


def grade(star, rows, seconds):
    planets = pd.read_csv(CONFIRMED / f"{star}-confirmed.csv")
    out = []
    claimed = set()
    for _, planet in planets.iterrows():
        best = None
        for i, row in enumerate(rows):
            for ratio in RATIOS:
                close = abs(row["period_days"] / (planet["period_days"] * ratio) - 1) < 5e-3
                if close and (best is None or ratio == 1.0):
                    best = (i, ratio)
        rec = {"star": star, "target": planet["target"], "cat_max_mes": planet["max_mes"]}
        if best is not None:
            i, ratio = best
            claimed.add(i)
            rec.update(found=True, ratio=ratio, x_max_mes=rows[i]["max_mes"],
                       status=rows[i]["detection_status"])
        else:
            rec.update(found=False)
        out.append(rec)
    for i, row in enumerate(rows):
        if i not in claimed:
            out.append({"star": star, "target": "UNMATCHED", "found": False,
                        "x_period": row["period_days"], "x_max_mes": row["max_mes"],
                        "status": row["detection_status"]})
    for rec in out:
        rec["seconds"] = seconds
    return out


def main():
    out_path = Path(sys.argv[2])
    records = []
    for star in sys.argv[3:]:
        data = np.load(DATA / f"{star}.npz")
        start = _time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = extract_features_from_arrays(data["time"], data["flux"])
        recs = grade(star, rows, round(_time.time() - start))
        full = out_path.with_name(out_path.stem + "_rows.csv")
        frame = pd.DataFrame(rows).assign(star=star)
        frame.to_csv(full, mode="a", header=not full.exists(), index=False)
        records.extend(recs)
        pd.DataFrame(records).to_csv(out_path, index=False)
        found = sum(r.get("found", False) for r in recs if r["target"] != "UNMATCHED")
        extra = sum(r["target"] == "UNMATCHED" for r in recs)
        print(star, f"found {found}/{len(recs) - extra}, extra {extra}", flush=True)


if __name__ == "__main__":
    main()
