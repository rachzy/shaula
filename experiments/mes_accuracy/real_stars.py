"""Measure MES at the catalogue ephemeris on real Kepler light curves.

Two steps, both resumable:

    python -m experiments.mes_accuracy.real_stars download [STAR ...]
    python -m experiments.mes_accuracy.real_stars measure --mode main|branch [STAR ...]

``download`` fetches every long-cadence quarter of each host in
``data/confirmed/`` (stitched and cleaned exactly as ``extract()`` with
``download_all=True``) and stores time/flux as ``.npz`` under
``$SHAULA_EXPERIMENT_DATA`` (default: ``~/.cache/shaula-experiments``), then
deletes the FITS files. Nothing downloaded goes into the repository.

``measure`` runs ``_extract_single_candidate_from_arrays`` on the catalogue
ephemeris of each confirmed planet (the hint path: no global search), with
the other planets' transits gapped as the iterative search would gap them.
``--mode main`` reproduces main's noise model (series-median baseline, no
duration audit); ``branch`` uses the current defaults. One CSV row per
planet goes to ``results/real_<mode>.csv``.
"""

import argparse
import contextlib
import functools
import io
import os
import shutil
import sys
import tempfile
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CONFIRMED = REPO / "data" / "confirmed"
RESULTS = Path(__file__).resolve().parent / "results"
DATA = Path(
    os.environ.get(
        "SHAULA_EXPERIMENT_DATA", Path.home() / ".cache" / "shaula-experiments"
    )
)
BKJD_OFFSET = 2454833.0


def hosts(selected):
    names = sorted(p.name[: -len("-confirmed.csv")] for p in CONFIRMED.glob("*-confirmed.csv"))
    return [n for n in names if not selected or n in selected]


def kic_designation(star):
    """``KIC n`` for a host: MAST's name resolver is unreachable from the cloud
    environment (plain HTTP), but a KIC designation is searched directly."""
    table = DATA / "kepids.csv"
    if not table.exists():
        url = (
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
            "select+distinct+kepid,kepler_name+from+cumulative+"
            "where+kepler_name+is+not+null&format=csv"
        )
        pd.read_csv(url).to_csv(table, index=False)
    names = pd.read_csv(table)
    hosts_ = names["kepler_name"].str.rsplit(" ", n=1).str[0]
    match = names.loc[hosts_ == star, "kepid"].unique()
    if match.size != 1:
        raise ValueError(f"{star}: {match.size} KIC matches")
    return f"KIC {int(match[0])}"


def download(star):
    out = DATA / f"{star}.npz"
    if out.exists():
        return "cached"
    from shaula.download_and_clean import download_and_clean_lightcurve

    tmp = Path(tempfile.mkdtemp(prefix="lk-", dir=DATA))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lc = download_and_clean_lightcurve(
                kic_designation(star), "Kepler", all=True, cache_dir=tmp
            )
        time = np.asarray(lc.time.value, dtype=float)
        flux = np.asarray(lc.flux.value, dtype=float)
        radius = lc.meta.get("RADIUS", np.nan)
        np.savez_compressed(out, time=time, flux=flux, radius=float(radius or np.nan))
        return f"{time.size} cadences"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def set_mode(mode):
    from shaula import cdpp, extract_feats, sesmes

    if mode == "main":
        patched = functools.partial(cdpp.duration_matched_statistics, flank_durations=None)
        cdpp.duration_matched_statistics = patched
        sesmes.duration_matched_statistics = patched
        extract_feats._mes_optimal_duration = lambda t, f, p, t0, d, g: d
    elif mode == "flank":
        extract_feats._mes_optimal_duration = lambda t, f, p, t0, d, g: d
    elif mode == "gross":
        extract_feats.DURATION_AUDIT_FACTORS = (0.5, 2.0, 2.83, 4.0)
    elif mode != "branch":
        raise ValueError(mode)


def measure_star(star):
    from shaula.extract_feats import _extract_single_candidate_from_arrays, _periodic_mask

    data = np.load(DATA / f"{star}.npz")
    time, flux = data["time"], data["flux"]
    planets = pd.read_csv(CONFIRMED / f"{star}-confirmed.csv")
    planets = planets[np.isfinite(planets["period_days"]) & np.isfinite(planets["t0"])]
    rows = []
    for _, planet in planets.iterrows():
        t0 = float(planet["t0"]) - BKJD_OFFSET
        others = np.zeros(time.size, dtype=bool)
        for _, other in planets.iterrows():
            if other["target"] == planet["target"]:
                continue
            others |= _periodic_mask(
                time,
                float(other["period_days"]),
                float(other["t0"]) - BKJD_OFFSET,
                float(other["duration_days"]),
                width_factor=1.5,
            )
        hint = {
            "period": float(planet["period_days"]),
            "t0": t0,
            "duration": float(planet["duration_days"]),
        }
        start = _time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                feats, _, _ = _extract_single_candidate_from_arrays(
                    time, flux, candidate_hint=hint, gap_mask=others, dealias=False
                )
        except Exception as exc:  # report and keep going
            rows.append({"star": star, "target": planet["target"], "error": repr(exc)})
            continue
        rows.append(
            {
                "star": star,
                "target": planet["target"],
                "period_days": hint["period"],
                "cat_duration_days": hint["duration"],
                "cat_depth": planet["depth_mean_per_transit"],
                "cat_max_mes": planet["max_mes"],
                "cat_max_ses": planet["max_ses"],
                "x_period_days": feats["period_days"],
                "x_duration_days": feats["duration_days"],
                "x_depth": feats["depth_mean_per_transit"],
                "x_max_mes": feats["max_mes"],
                "x_MES": feats["MES"],
                "x_max_ses": feats["max_ses"],
                "x_cdpp_6h": feats["cdpp_6h"],
                "seconds": round(_time.time() - start, 1),
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("download", "measure"))
    parser.add_argument("stars", nargs="*")
    parser.add_argument("--mode", default="branch")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    if args.step == "download":
        for star in hosts(args.stars):
            try:
                print(star, download(star), flush=True)
            except Exception as exc:
                print(star, "FAILED", repr(exc), flush=True)
        return
    set_mode(args.mode)
    rows = []
    for star in hosts(args.stars):
        if not (DATA / f"{star}.npz").exists():
            print(star, "not downloaded", file=sys.stderr)
            continue
        star_rows = measure_star(star)
        rows.extend(star_rows)
        for r in star_rows:
            print(
                r["target"],
                r.get("error")
                or f"cat {r['cat_max_mes']:.1f} got {r['x_max_mes']:.1f}",
                flush=True,
            )
    out = Path(args.out) if args.out else RESULTS / f"real_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
