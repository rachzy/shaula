"""Candidate-aware local detrending prototypes, patched into the pipeline.

``patch(method, window)`` wraps ``extract_feats.detrend_with_bls_mask`` so the
flux it returns is additionally divided by a robust local trend estimated from
out-of-transit cadences only (this candidate's and accepted candidates'
transits withheld). Nothing in ``shaula`` changes until an approach is chosen.
"""

import numpy as np
from shaula import extract_feats

_ORIGINAL = extract_feats.detrend_with_bls_mask


def _biweight(values, c=5.0, n_iter=5):
    loc = np.median(values)
    for _ in range(n_iter):
        mad = np.median(np.abs(values - loc))
        if mad == 0:
            return loc
        u = (values - loc) / (c * mad)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if w.sum() == 0:
            return loc
        loc = np.sum(w * values) / w.sum()
    return loc


def local_trend(time, flux, use, window_days, method="biweight", gap_days=0.5,
                step_days=None):
    """Robust location of ``flux[use]`` in a centred ``window_days`` window.

    Windows never cross a gap longer than ``gap_days``. Evaluated on a grid of
    ``step_days`` (default window/10) and linearly interpolated per segment.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    trend = np.full(time.size, np.nan)
    finite = np.isfinite(time) & np.isfinite(flux)
    use = use & finite
    if step_days is None:
        step_days = window_days / 10.0
    seg = np.zeros(time.size, int)
    seg[1:] = np.cumsum(np.diff(time) > gap_days)
    stat = _biweight if method == "biweight" else (np.median if method == "median" else np.mean)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        t_seg = time[idx]
        ok = use[idx]
        if ok.sum() < 3:
            continue
        t_ok = t_seg[ok]
        f_ok = flux[idx][ok]
        grid = np.arange(t_seg[0], t_seg[-1] + step_days, step_days)
        lo = np.searchsorted(t_ok, grid - window_days / 2)
        hi = np.searchsorted(t_ok, grid + window_days / 2)
        vals = np.full(grid.size, np.nan)
        for i, (a, b) in enumerate(zip(lo, hi)):
            if b - a >= 3:
                vals[i] = stat(f_ok[a:b])
        good = np.isfinite(vals)
        if good.sum() == 0:
            continue
        trend[idx] = np.interp(t_seg, grid[good], vals[good])
    return trend


def patch(method="biweight", window_durations=3.0, min_window_days=0.0, mask_factor=1.5):
    def wrapped(tTime, flux, *args, **kwargs):
        flux_detr, trend, mask_transit, info = _ORIGINAL(tTime, flux, *args, **kwargs)
        exclude = kwargs.get("exclude_mask")
        exclude = np.zeros(np.shape(tTime), bool) if exclude is None else np.asarray(exclude, bool)
        t = np.asarray(tTime, float)
        valid = np.isfinite(t) & np.isfinite(flux_detr)
        # mask_transit covers valid cadences only (same ordering).
        intransit = np.zeros(t.size, bool)
        intransit[valid] = np.asarray(mask_transit, bool)
        intransit |= extract_feats._periodic_mask(
            t, info["best_period"], info["t0"], info["best_duration"], mask_factor
        )
        use = valid & ~intransit & ~exclude
        window = max(window_durations * float(info["best_duration"]), min_window_days)
        lt = local_trend(t, flux_detr, use, window, method=method)
        ok = np.isfinite(lt) & (lt > 0)
        flux_detr = flux_detr.copy()
        flux_detr[ok] = flux_detr[ok] / lt[ok]
        trend = trend.copy()
        trend[ok] = trend[ok] * lt[ok]
        return flux_detr, trend, mask_transit, info

    extract_feats.detrend_with_bls_mask = wrapped


def unpatch():
    extract_feats.detrend_with_bls_mask = _ORIGINAL
