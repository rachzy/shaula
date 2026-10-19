"""Synthetic Kepler-like light curves with injected transits.

Used to separate the sources of Shaula's MES deficit offline: filter shape,
duration mismatch, fold losses (TTVs) and noise inflation from stellar
variability. Everything is deterministic given ``seed``.
"""

import numpy as np

KEPLER_LC_DAYS = 29.4244 / 1440.0


def kepler_time(n_quarters=4, quarter_days=90.0, gap_days=1.0, seed=0):
    rng = np.random.default_rng(seed)
    times = []
    start = 130.0
    for _ in range(n_quarters):
        t = np.arange(start, start + quarter_days, KEPLER_LC_DAYS)
        # Monthly downlink gaps of ~1 day, plus a few % random dropped cadences.
        for month in (1, 2):
            g0 = start + month * quarter_days / 3.0
            t = t[(t < g0) | (t > g0 + 0.8)]
        t = t[rng.random(t.size) > 0.02]
        times.append(t)
        start += quarter_days + gap_days
    return np.concatenate(times)


def _overlap_fraction(z, p):
    """Fraction of the planet disk (radius p, stellar radius 1) over the star."""
    z = np.abs(z)
    frac = np.zeros_like(z)
    inside = z <= 1.0 - p
    frac[inside] = 1.0
    part = (z > 1.0 - p) & (z < 1.0 + p)
    if np.any(part):
        zz = z[part]
        k0 = np.arccos(np.clip((p**2 + zz**2 - 1.0) / (2 * p * zz), -1, 1))
        k1 = np.arccos(np.clip((1.0 - p**2 + zz**2) / (2 * zz), -1, 1))
        area = (
            p**2 * k0
            + k1
            - 0.5 * np.sqrt(np.clip(4 * zz**2 - (1 + zz**2 - p**2) ** 2, 0, None))
        )
        frac[part] = area / (np.pi * p**2)
    return frac


def transit_model(
    time, period, t0, depth, duration_days, b=0.3, u1=0.4, u2=0.25, ttv=None,
    supersample=7,
):
    """Small-planet limb-darkened transit; ``depth`` is the central deficit.

    ``duration_days`` is T14. ``ttv`` maps an epoch number to a time offset
    (days), or is an array-like callable of epoch. Integrated over the
    long-cadence exposure with ``supersample`` sub-samples.
    """
    time = np.asarray(time, dtype=float)
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    t_sub = (time[:, None] + offsets[None, :] * KEPLER_LC_DAYS).ravel()

    p = np.sqrt(depth)  # approximate; refined by the LD normalisation below
    epoch = np.round((t_sub - t0) / period)
    centre = t0 + epoch * period
    if ttv is not None:
        centre = centre + ttv(epoch)
    dt = t_sub - centre
    # Chord half-length in stellar radii for impact b, so z(T14/2) = 1 + p.
    half_chord = np.sqrt(max((1 + p) ** 2 - b**2, 1e-6))
    x = dt / (duration_days / 2.0) * half_chord
    z = np.sqrt(x**2 + b**2)

    def intensity(r):
        mu = np.sqrt(np.clip(1 - r**2, 0, 1))
        return 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

    mean_intensity = 1 - u1 / 3.0 - u2 / 6.0
    deficit = (
        p**2 * _overlap_fraction(z, p) * intensity(np.clip(z, 0, 1)) / mean_intensity
    )
    centre_deficit = p**2 * intensity(b) / mean_intensity
    deficit *= depth / centre_deficit
    return deficit.reshape(time.size, supersample).mean(axis=1)


def stellar_variability(time, amp_rot=0.0, prot=12.0, amp_gran=0.0, tau_gran=0.2, seed=0):
    """Spot modulation (evolving two-harmonic sinusoid) plus AR(1) granulation."""
    rng = np.random.default_rng(seed + 1000)
    out = np.zeros_like(time)
    if amp_rot > 0:
        # Slowly evolving amplitude/phase over ~2 rotations.
        knots = np.arange(time.min(), time.max() + 2 * prot, 2 * prot)
        a1 = np.interp(time, knots, rng.uniform(0.5, 1.0, knots.size))
        ph = np.interp(time, knots, rng.uniform(0, 2 * np.pi, knots.size))
        out += amp_rot * a1 * np.sin(2 * np.pi * time / prot + ph)
        out += 0.4 * amp_rot * a1 * np.sin(4 * np.pi * time / prot + 1.3 * ph)
    if amp_gran > 0:
        alpha = np.exp(-KEPLER_LC_DAYS / tau_gran)
        e = rng.normal(0, amp_gran * np.sqrt(1 - alpha**2), time.size)
        g = np.zeros_like(time)
        for i in range(1, time.size):
            g[i] = alpha * g[i - 1] + e[i]
        out += g
    return out


def make_lightcurve(
    period, depth, duration_days, sigma_ppm, n_quarters=8, t0=None, b=0.3,
    amp_rot=0.0, prot=12.0, amp_gran=0.0, tau_gran=0.2, ttv=None, seed=0,
):
    time = kepler_time(n_quarters=n_quarters, seed=seed)
    if t0 is None:
        t0 = time.min() + 0.37 * period
    rng = np.random.default_rng(seed)
    model = (
        transit_model(time, period, t0, depth, duration_days, b=b, ttv=ttv)
        if depth > 0
        else np.zeros_like(time)
    )
    var = stellar_variability(time, amp_rot, prot, amp_gran, tau_gran, seed=seed)
    noise = rng.normal(0, sigma_ppm * 1e-6, time.size)
    flux = (1.0 + var) * (1.0 - model) + noise
    return {"time": time, "flux": flux, "model": model, "t0": t0}


def oracle_box_mes(time, model, sigma_ppm, duration_grid_days):
    """Best white-noise box-filter MES over a duration grid (TPS-like ideal).

    For each trial duration D, the box centred on each true transit centre
    averages the noiseless model; MES = sum(depth_i * n_i) / (sigma *
    sqrt(sum n_i)). The maximum over D is what TPS reports.
    """
    sigma = sigma_ppm * 1e-6
    best = 0.0
    # Centres from the model minima.
    in_tr = model > 0
    if not np.any(in_tr):
        return 0.0
    idx = np.flatnonzero(in_tr)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    centres = [time[g[np.argmax(model[g])]] for g in groups]
    for dur in duration_grid_days:
        num = 0.0
        npts = 0
        for c in centres:
            sel = np.abs(time - c) <= dur / 2
            num += model[sel].sum()
            npts += sel.sum()
        if npts:
            best = max(best, num / (sigma * np.sqrt(npts)))
    return best


def optimal_snr(model, sigma_ppm):
    return float(np.sqrt(np.sum(model**2)) / (sigma_ppm * 1e-6))
