"""Fixed-ephemeris MES for planets the full search missed.

    python experiments/mes_accuracy/miss_diagnostic.py STAR:PLANET [...]

No period or epoch refinement: the light curve is flattened with a masked
running median (all known transits withheld from the trend), then
``compute_SES_MES`` is evaluated at the catalogue P, t0 and T14 with the other
planets gapped. Prints MES, the per-event SES spread and the TTV-like drift
(best local epoch offset per event, in units of T14), which separates "the
signal is there at MES >= 7.1" from "the signal is too weak or smeared".
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from experiments.mes_accuracy.real_stars import BKJD_OFFSET, CONFIRMED, DATA  # noqa: E402
from shaula.extract_feats import _periodic_mask  # noqa: E402
from shaula.sesmes import compute_SES_MES  # noqa: E402


def masked_running_median(time, flux, keep, window, step=0.02):
    edges = np.arange(time.min(), time.max() + step, step)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(time, edges) - 1
    trend = np.full(centres.size, np.nan)
    half = int(round(0.5 * window / step))
    order = np.argsort(idx)
    sorted_idx = idx[order]
    for b in range(centres.size):
        lo = np.searchsorted(sorted_idx, b - half)
        hi = np.searchsorted(sorted_idx, b + half + 1)
        sel = order[lo:hi]
        sel = sel[keep[sel]]
        if sel.size >= 10:
            trend[b] = np.median(flux[sel])
    good = np.isfinite(trend)
    return np.interp(time, centres[good], trend[good])


def event_offsets(time, flat, period, t0, dur):
    """Best box-centre offset per event (in T14), from a +-1 T14 scan."""
    shifts = np.linspace(-1.0, 1.0, 41) * dur
    n0 = int(np.floor((time.min() - t0) / period))
    n1 = int(np.ceil((time.max() - t0) / period))
    out = []
    for n in range(n0, n1 + 1):
        centre = t0 + n * period
        near = np.abs(time - centre) < 3 * dur
        if near.sum() < 10:
            continue
        depths = [1.0 - np.mean(flat[near & (np.abs(time - centre - s) < 0.5 * dur)])
                  if np.any(near & (np.abs(time - centre - s) < 0.5 * dur)) else np.nan
                  for s in shifts]
        if np.all(np.isnan(depths)):
            continue
        out.append(shifts[int(np.nanargmax(depths))] / dur)
    return np.array(out)


def diagnose(star, target):
    data = np.load(DATA / f"{star}.npz")
    time, flux = data["time"], data["flux"]
    planets = pd.read_csv(CONFIRMED / f"{star}-confirmed.csv")
    all_tr = np.zeros(time.size, bool)
    others = np.zeros(time.size, bool)
    for _, p in planets.iterrows():
        m = _periodic_mask(time, p["period_days"], p["t0"] - BKJD_OFFSET, p["duration_days"], 1.5)
        all_tr |= m
        if p["target"] != target:
            others |= m
    p = planets[planets["target"] == target].iloc[0]
    period, t0, dur = float(p["period_days"]), float(p["t0"]) - BKJD_OFFSET, float(p["duration_days"])
    window = max(1.0, 3.0 * dur)
    flat = flux / masked_running_median(time, flux, ~all_tr, window)
    own = _periodic_mask(time, period, t0, dur, 1.5)
    with contextlib.redirect_stdout(io.StringIO()):
        stats = compute_SES_MES(time, flat, period, t0, dur, transit_mask=own, gap_mask=others)
    ses = np.asarray(stats.get("SES", []), dtype=float)
    offs = event_offsets(time, flat, period, t0, dur)
    return {
        "target": target, "period": period, "T14_h": dur * 24,
        "cat_depth_ppm": p["depth_mean_per_transit"] * 1e6, "cat_max_mes": p["max_mes"],
        "MES": stats.get("MES"), "max_mes": stats.get("max_mes"),
        "n_events": int(np.isfinite(ses).sum()) if ses.size else 0,
        "SES_med": float(np.nanmedian(ses)) if ses.size else np.nan,
        "offset_mad_T14": float(np.median(np.abs(offs))) if offs.size else np.nan,
    }


if __name__ == "__main__":
    rows = [diagnose(*arg.split(":")) for arg in sys.argv[1:]]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
