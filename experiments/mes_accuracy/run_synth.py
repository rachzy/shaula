"""Run the full extraction on synthetic light curves and compare MES to the ideal.

Usage: python -m experiments.mes_accuracy.run_synth <grid-name> [--out results.csv]
"""

import argparse
import contextlib
import io
import json
import sys
import time as _time

import numpy as np
import pandas as pd
from shaula.extract_feats import (
    _extract_single_candidate_from_arrays,
    extract_features_from_arrays,
)

from experiments.mes_accuracy import synth

TPS_DURATIONS_H = np.array(
    [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 12.5, 15.0]
)

GRIDS = {
    # White noise only: isolates filter shape, duration and fold losses.
    "white": [
        dict(period=p, depth=d, duration_days=dur, sigma_ppm=s, seed=seed)
        for (p, dur) in [(3.7, 0.12), (11.3, 0.18), (37.1, 0.26)]
        for d, s in [(300e-6, 400), (1000e-6, 400), (5000e-6, 400)]
        for seed in (1,)
    ],
    # Same signals plus spot modulation and granulation.
    "variable": [
        dict(period=p, depth=d, duration_days=dur, sigma_ppm=s, amp_rot=a,
             prot=pr, amp_gran=g, seed=seed)
        for (p, dur) in [(3.7, 0.12), (11.3, 0.18)]
        for d, s in [(1000e-6, 100), (300e-6, 100)]
        for a, pr, g in [(2000e-6, 12.0, 0.0), (500e-6, 3.0, 0.0), (0.0, 1.0, 80e-6)]
        for seed in (1,)
    ],
}


def run_case(case, n_quarters=8, quiet=True, fast=True):
    """``fast`` measures only the injected ephemeris (hint path, no global BLS)."""
    lc = synth.make_lightcurve(n_quarters=n_quarters, **case)
    model = lc["model"]
    oracle = synth.oracle_box_mes(lc["time"], model, case["sigma_ppm"], TPS_DURATIONS_H / 24)
    optimal = synth.optimal_snr(model, case["sigma_ppm"])
    start = _time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        if fast:
            hint = {
                "period": case["period"],
                "t0": lc["t0"],
                "duration": case["duration_days"],
            }
            row, _, _ = _extract_single_candidate_from_arrays(
                lc["time"], lc["flux"], candidate_hint=hint, dealias=False
            )
            rows = [row]
        else:
            rows = extract_features_from_arrays(lc["time"], lc["flux"])
    elapsed = _time.time() - start
    match = None
    for row in rows:
        ratio = row["period_days"] / case["period"]
        if abs(ratio - 1) < 1e-3:
            match = row
            break
    out = dict(case)
    out.update(
        oracle_mes=oracle,
        optimal_snr=optimal,
        found=match is not None,
        n_rows=len(rows),
        seconds=round(elapsed, 1),
    )
    if match is not None:
        for key in ("max_mes", "MES", "duration_days", "depth_mean_per_transit", "cdpp_6h"):
            out[f"x_{key}"] = match.get(key, np.nan)
        out["mes_ratio"] = out["x_max_mes"] / oracle
        out["dur_ratio"] = out["x_duration_days"] / case["duration_days"]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("grid", choices=sorted(GRIDS))
    parser.add_argument("--quarters", type=int, default=8)
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--full", action="store_true", help="run the whole search")
    parser.add_argument(
        "--trend",
        help="method:window_durations[:min_window_days], e.g. biweight:3",
    )
    parser.add_argument(
        "--noise",
        default="auto",
        help="flank_durations for the noise model: auto (library default), "
        "current (series-median baseline, the pre-2026-10-19 model) or a number",
    )
    args = parser.parse_args(argv)
    if args.noise != "auto":
        import functools

        from shaula import cdpp, sesmes

        value = None if args.noise == "current" else float(args.noise)
        patched = functools.partial(
            cdpp.duration_matched_statistics, flank_durations=value
        )
        cdpp.duration_matched_statistics = patched
        sesmes.duration_matched_statistics = patched
    if args.trend:
        from experiments.mes_accuracy import local_trend

        parts = args.trend.split(":")
        local_trend.patch(
            parts[0],
            float(parts[1]),
            float(parts[2]) if len(parts) > 2 else 0.0,
        )
    cases = GRIDS[args.grid][: args.limit]
    results = []
    for case in cases:
        res = run_case(case, n_quarters=args.quarters, fast=not args.full)
        results.append(res)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in res.items()}), flush=True)
    frame = pd.DataFrame(results)
    if args.out:
        frame.to_csv(args.out, index=False)
    if "mes_ratio" in frame:
        print("median mes_ratio:", frame["mes_ratio"].median(), file=sys.stderr)


if __name__ == "__main__":
    main()
