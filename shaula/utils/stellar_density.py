"""Stellar density implied by a transit, compared with the star's catalogue.

For a planet on a circular orbit the period and transit duration fix the
host's mean density (Seager & Mallen-Ornelas 2003). An eclipsing binary, a
blend or a background eclipse does not have to fit the target star, so the
ratio of the two densities separates them from planets.

Shaula fits no impact parameter, so ``b = 0`` is assumed. That gives the
largest ``a/R*`` a duration allows, so the transit density is an upper bound:
grazing planets read too dense (about +0.2 dex at ``b > 0.6`` on the
confirmed Kepler planets). Eccentric orbits shift it either way.

Every value is NaN, never infinite, when an input is missing or impossible.
"""

from __future__ import annotations

import numpy as np

GRAVITATIONAL_CONSTANT_CGS = 6.674e-8
SOLAR_RADIUS_CM = 6.957e10
SECONDS_PER_DAY = 86400.0

# Per-candidate columns. The catalogue density is a property of the star
# alone, so it is not repeated on every candidate row.
STELLAR_DENSITY_COLUMNS = (
    "rho_star_transit",
    "log_rho_ratio",
)


def _finite_float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def transit_stellar_density(period_days, duration_days, depth) -> float:
    """Mean stellar density (g/cm^3) implied by a central transit.

    ``duration_days`` is the total duration T14 and ``depth`` the fractional
    flux deficit, so ``k = sqrt(depth)`` is Rp/R*. For ``b = 0`` the exact
    circular-orbit relation is ``sin(pi T / P) = (1 + k) / (a/R*)``, and
    ``rho* = 3 pi (a/R*)^3 / (G P^2)``.
    """
    period = _finite_float(period_days)
    duration = _finite_float(duration_days)
    depth = _finite_float(depth)
    # A transit needs a dip (0 < depth < 1) and cannot outlast half an orbit.
    if not (period > 0 and 0 < duration < 0.5 * period and 0 < depth < 1):
        return float("nan")
    a_over_rstar = (1.0 + np.sqrt(depth)) / np.sin(np.pi * duration / period)
    period_s = period * SECONDS_PER_DAY
    rho = 3.0 * np.pi * a_over_rstar**3 / (GRAVITATIONAL_CONSTANT_CGS * period_s**2)
    return float(rho) if np.isfinite(rho) else float("nan")


def catalogue_stellar_density(logg, radius_rsun) -> float:
    """Mean stellar density (g/cm^3) from catalogue log g (cgs) and radius.

    ``M / (4/3 pi R^3)`` with ``M = g R^2 / G`` is ``3 g / (4 pi G R)``, the
    same mass ``stellar_properties.stellar_parameters`` derives.
    """
    logg = _finite_float(logg)
    radius = _finite_float(radius_rsun)
    if not (logg > 0 and radius > 0):
        return float("nan")
    rho = (
        3.0
        * 10.0**logg
        / (4.0 * np.pi * GRAVITATIONAL_CONSTANT_CGS * radius * SOLAR_RADIUS_CM)
    )
    return float(rho) if np.isfinite(rho) else float("nan")


def stellar_density_features(
    period_days, duration_days, depth, logg, radius_rsun
) -> dict:
    """The ``STELLAR_DENSITY_COLUMNS`` for one candidate.

    ``log_rho_ratio`` is ``log10(rho_transit / rho_catalogue)``: near 0 for a
    planet of that star, positive when the transit is too short for it (a
    grazing or stellar eclipse), negative when it is too long.
    """
    rho_transit = transit_stellar_density(period_days, duration_days, depth)
    rho_catalogue = catalogue_stellar_density(logg, radius_rsun)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(rho_transit / rho_catalogue)
    return {
        "rho_star_transit": rho_transit,
        "log_rho_ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
    }
