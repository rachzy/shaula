"""Physical planet properties from the period and the star's catalogue values.

Kepler, K2 and TESS light-curve files carry the target's catalogue parameters
in their primary header (KIC, EPIC or TIC), and lightkurve copies them into
``LightCurve.meta``: ``TEFF`` (K), ``LOGG`` (log10 of surface gravity in cgs)
and ``RADIUS`` (solar radii). The header has no mass, so it is derived from
gravity and radius. Every value is NaN when an input is missing.
"""

from __future__ import annotations

import numpy as np

SOLAR_LOGG_CGS = 4.438
SOLAR_TEFF_K = 5772.0
SOLAR_RADIUS_AU = 0.00465047
AU_KM = 1.495978707e8
DAYS_PER_YEAR = 365.25
SECONDS_PER_DAY = 86400.0
# A common choice for planets of unknown albedo; Earth's Bond albedo is ~0.3.
EQUILIBRIUM_BOND_ALBEDO = 0.3

STELLAR_PROPERTY_COLUMNS = (
    "stellar_teff_k",
    "stellar_logg",
    "stellar_radius_rsun",
    "stellar_mass_msun",
    "semi_major_axis_au",
    "orbital_speed_kms",
    "insolation_earth",
    "equilibrium_temp_k",
)


def _positive_float(value) -> float:
    """``value`` as a float, or NaN when it is missing, non-finite or <= 0."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) and value > 0 else float("nan")


def stellar_parameters(meta) -> dict:
    """Read Teff, log g and radius from light-curve ``meta``; derive the mass.

    KIC stars without catalogue values carry ``None``, not NaN.
    """
    teff = _positive_float(meta.get("TEFF"))
    logg = _positive_float(meta.get("LOGG"))
    radius = _positive_float(meta.get("RADIUS"))
    # g = G M / R^2, so in solar units M = (g / g_sun) * R^2.
    mass = 10.0 ** (logg - SOLAR_LOGG_CGS) * radius**2
    return {
        "stellar_teff_k": teff,
        "stellar_logg": logg,
        "stellar_radius_rsun": radius,
        "stellar_mass_msun": float(mass),
    }


def orbital_properties(period_days, star: dict) -> dict:
    """Orbit and irradiation of a planet with ``period_days`` around ``star``.

    Assumes a circular orbit and a planet much lighter than its star.
    """
    period_days = _positive_float(period_days)
    teff = star["stellar_teff_k"]
    radius = star["stellar_radius_rsun"]
    # Kepler's third law in AU, years and solar masses: a^3 = M P^2.
    a_au = (star["stellar_mass_msun"] * (period_days / DAYS_PER_YEAR) ** 2) ** (1 / 3)
    speed = 2.0 * np.pi * a_au * AU_KM / (period_days * SECONDS_PER_DAY)
    insolation = radius**2 * (teff / SOLAR_TEFF_K) ** 4 / a_au**2
    t_eq = (
        teff
        * np.sqrt(radius * SOLAR_RADIUS_AU / (2.0 * a_au))
        * (1.0 - EQUILIBRIUM_BOND_ALBEDO) ** 0.25
    )
    return {
        "semi_major_axis_au": float(a_au),
        "orbital_speed_kms": float(speed),
        "insolation_earth": float(insolation),
        "equilibrium_temp_k": float(t_eq),
    }
