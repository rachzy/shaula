import numpy as np

# Trial total durations, as multiples of the candidate's T14. The extracted T14
# runs long by ~9% on average, and a box fitted with too long a duration reads
# its own edges as ingress, so the fit is allowed to choose its own width.
VSHAPE_DURATION_SCALES = np.linspace(0.6, 1.4, 17)
# Ingress fraction r = T12 / (T14 / 2): 0 is a flat-bottomed box, 1 a pure V.
VSHAPE_INGRESS_FRACTIONS = np.linspace(0.0, 1.0, 41)
# Out-of-transit flanks each side of the widest trial, in units of T14, so the
# baseline is fitted from the same events as the dip.
VSHAPE_WINDOW_DURATIONS = 1.5
VSHAPE_CADENCE_SUBSAMPLES = 7
MIN_VSHAPE_IN_TRANSIT_POINTS = 10
# Above this many points the fit runs on means in fine |dt| bins, weighted by
# their counts. The least-squares optimum is unchanged up to the bin width, and
# a short-period candidate stays as cheap as a long-period one.
VSHAPE_MAX_UNBINNED_POINTS = 3000
VSHAPE_BINS = 600


def _trapezoid(abs_dt, total, ingress):
    """Unit-depth symmetric trapezoid evaluated at ``|t - t_mid|``."""
    half = 0.5 * total
    if ingress <= 0:
        return (abs_dt < half).astype(float)
    return np.clip((half - abs_dt) / ingress, 0.0, 1.0)


def compute_vshape(time, flux_detr, period, t0, dur_days, cadence_days=None):
    """Fitted ingress fraction of the folded transit: 0 = box/U, 1 = V.

    A symmetric trapezoid ``baseline - depth * shape`` is fitted by least
    squares to the unbinned points around every transit, on a grid of total
    durations and ingress fractions ``r = T12 / (T14 / 2)``; for each grid
    point ``baseline`` and ``depth`` are solved linearly. The returned value is
    ``r`` of the best fit with a positive depth. A planet much smaller than its
    star gives a short ingress and a small ``r``; a grazing eclipse or a
    blended eclipsing binary gives a V and ``r`` near 1 (the trapezoid test
    behind Kepler's ``DEEP_V_SHAPED`` flag).

    The model is averaged over the cadence when ``cadence_days`` is given, so
    long-cadence smearing of a short transit is not mistaken for a V. Working
    on unbinned points in time-from-mid-transit keeps the measurement valid at
    long periods, where a transit is narrower than a phase bin.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux_detr, dtype=float)
    if not (
        np.isfinite(period)
        and period > 0
        and np.isfinite(t0)
        and np.isfinite(dur_days)
        and dur_days > 0
    ):
        return np.nan

    finite = np.isfinite(time) & np.isfinite(flux)
    dt = np.mod(time[finite] - t0 + 0.5 * period, period) - 0.5 * period
    half_window = min(
        (0.5 * VSHAPE_DURATION_SCALES[-1] + VSHAPE_WINDOW_DURATIONS) * dur_days,
        0.45 * period,
    )
    near = np.abs(dt) < half_window
    abs_dt = np.abs(dt[near])
    y = flux[finite][near]
    if np.sum(abs_dt < 0.5 * dur_days) < MIN_VSHAPE_IN_TRANSIT_POINTS:
        return np.nan
    if np.sum(abs_dt > 0.5 * VSHAPE_DURATION_SCALES[-1] * dur_days) < 3:
        return np.nan

    weights = np.ones_like(y)
    if abs_dt.size > VSHAPE_MAX_UNBINNED_POINTS:
        edges = np.linspace(0.0, half_window, VSHAPE_BINS + 1)
        which = np.clip(np.digitize(abs_dt, edges) - 1, 0, VSHAPE_BINS - 1)
        counts = np.bincount(which, minlength=VSHAPE_BINS).astype(float)
        sums = np.bincount(which, weights=y, minlength=VSHAPE_BINS)
        filled = counts > 0
        abs_dt = (0.5 * (edges[:-1] + edges[1:]))[filled]
        y = sums[filled] / counts[filled]
        weights = counts[filled]

    if cadence_days is not None and np.isfinite(cadence_days) and cadence_days > 0:
        offsets = np.linspace(-0.5, 0.5, VSHAPE_CADENCE_SUBSAMPLES) * cadence_days
    else:
        offsets = np.zeros(1)
    # Subsampled |dt| per point; folding the sign is safe because the model is
    # symmetric about mid-transit.
    sub_abs_dt = np.abs(abs_dt[:, None] + offsets[None, :])

    weight_sum = float(np.sum(weights))
    y_centered = y - float(np.dot(weights, y)) / weight_sum
    best_sse = np.inf
    best_ratio = np.nan
    for scale in VSHAPE_DURATION_SCALES:
        total = scale * dur_days
        for ratio in VSHAPE_INGRESS_FRACTIONS:
            shape = _trapezoid(sub_abs_dt, total, ratio * 0.5 * total).mean(axis=1)
            shape_centered = shape - float(np.dot(weights, shape)) / weight_sum
            weighted_shape = weights * shape_centered
            variance = float(np.dot(weighted_shape, shape_centered))
            if variance <= 0:
                continue
            # y = baseline - depth * shape, solved in closed form.
            depth = -float(np.dot(weighted_shape, y_centered)) / variance
            if depth <= 0:
                continue
            residual = y_centered + depth * shape_centered
            sse = float(np.dot(weights * residual, residual))
            if sse < best_sse:
                best_sse = sse
                best_ratio = float(ratio)
    return best_ratio
