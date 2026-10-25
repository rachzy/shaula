import numpy as np
import pytest

from ..utils.stellar_properties import (
    ORBIT_COLUMNS,
    orbital_properties,
    stellar_parameters,
)

SUN = {"TEFF": 5772.0, "LOGG": 4.438, "RADIUS": 1.0}
HEADER_KEYS = {"TEFF": "teff_k", "LOGG": "logg", "RADIUS": "radius_rsun"}


def test_the_sun_and_earth_come_out_as_one_au():
    star = stellar_parameters(SUN)
    assert star["mass_msun"] == pytest.approx(1.0)
    orbit = orbital_properties(365.25, star)
    assert orbit["semi_major_axis_au"] == pytest.approx(1.0)
    assert orbit["orbital_speed_kms"] == pytest.approx(29.78, abs=0.01)
    assert orbit["insolation_earth"] == pytest.approx(1.0)
    # Earth's equilibrium temperature at Bond albedo 0.3.
    assert orbit["equilibrium_temp_k"] == pytest.approx(255.0, abs=1.0)


def test_mass_follows_gravity_and_radius():
    star = stellar_parameters({"TEFF": 5000.0, "LOGG": 4.438 - np.log10(2), "RADIUS": 2.0})
    assert star["mass_msun"] == pytest.approx(2.0)


@pytest.mark.parametrize("missing", ["TEFF", "LOGG", "RADIUS"])
@pytest.mark.parametrize("value", [None, "", np.nan, 0.0, -1.0])
def test_a_missing_input_gives_nan_not_an_error(missing, value):
    star = stellar_parameters({**SUN, missing: value})
    orbit = orbital_properties(10.0, star)
    assert np.isnan(star[HEADER_KEYS[missing]])
    assert np.isnan(orbit["insolation_earth"])
    assert np.isnan(orbit["equilibrium_temp_k"])
    # Teff is not needed for the orbit itself.
    assert np.isnan(orbit["semi_major_axis_au"]) == (missing != "TEFF")


def test_lightcurve_rows_carry_the_orbit_but_not_the_star(monkeypatch):
    import lightkurve as lk

    from .. import extract_feats

    monkeypatch.setattr(
        extract_feats,
        "extract_features_from_arrays",
        lambda *a, **k: [{"period_days": 365.25, "depth_mean_per_transit": 1e-4}],
    )
    time = np.arange(0.0, 10.0, 0.02)
    lc = lk.LightCurve(time=time, flux=np.ones_like(time))
    lc.meta.update(SUN)
    row = extract_feats.extract_features_from_lightcurve(lc)[0]
    assert set(ORBIT_COLUMNS) <= set(row)
    assert row["semi_major_axis_au"] == pytest.approx(1.0)
    assert row["planet_radius_rearth"] == pytest.approx(109.1 * 0.01)
    assert "teff_k" not in row
