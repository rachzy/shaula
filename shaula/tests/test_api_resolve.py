"""Resolving a user-supplied name to an exact catalog identifier."""

from __future__ import annotations

from typing import ClassVar

import pytest
from astropy.coordinates import SkyCoord

from .. import api
from ..api import (
    MAX_QUERY_LENGTH,
    POSITION_MATCH_RADIUS_ARCSEC,
    ResolvedTarget,
    TargetNotFound,
    _catalog_id_from_target_name,
    canonical_mission,
    resolve,
)
from ..download_and_clean import default_product_selection


def _row(**overrides):
    row = {"target_name": "kplr008120608", "s_ra": 298.65272, "s_dec": 43.95502}
    row.update(overrides)
    return row


def _stub(monkeypatch, rows):
    monkeypatch.setattr(api, "_search_target", lambda q, m: rows)


def test_kepler_strips_prefix_and_leading_zeros(monkeypatch):
    _stub(monkeypatch, [_row()])
    resolved = resolve("Kepler-186", "Kepler")

    assert isinstance(resolved, ResolvedTarget)
    assert resolved.catalog == "KIC"
    assert resolved.catalog_id == "8120608"


def test_k2_strips_prefix(monkeypatch):
    _stub(monkeypatch, [_row(target_name="ktwo201111557")])
    resolved = resolve("EPIC201111557", "K2")

    assert resolved.catalog == "EPIC"
    assert resolved.catalog_id == "201111557"


def test_tess_keeps_bare_tic_number(monkeypatch):
    _stub(monkeypatch, [_row(target_name="251848941")])
    resolved = resolve("TOI-178", "TESS")

    assert resolved.catalog == "TIC"
    assert resolved.catalog_id == "251848941"


def test_strips_planet_letter_before_searching(monkeypatch):
    asked: list[tuple[str, str]] = []

    def spy(query, mission):
        asked.append((query, mission))
        return [_row()]

    monkeypatch.setattr(api, "_search_target", spy)
    resolved = resolve("Kepler-186f", "Kepler")

    assert asked == [("Kepler-186", "Kepler")]
    assert resolved.display_name == "Kepler-186"


def test_coordinates_are_floats(monkeypatch):
    _stub(monkeypatch, [_row(s_ra="298.65272", s_dec="43.95502")])
    resolved = resolve("Kepler-186", "Kepler")

    assert isinstance(resolved.ra_deg, float)
    assert isinstance(resolved.dec_deg, float)
    assert resolved.ra_deg == pytest.approx(298.65272)
    assert resolved.dec_deg == pytest.approx(43.95502)


def test_empty_result_raises_naming_the_query(monkeypatch):
    _stub(monkeypatch, [])

    with pytest.raises(TargetNotFound, match="NotAStar-999"):
        resolve("NotAStar-999", "Kepler")


def test_unsupported_mission_raises(monkeypatch):
    _stub(monkeypatch, [_row()])

    with pytest.raises(ValueError, match="Hubble"):
        resolve("Kepler-186", "Hubble")


@pytest.mark.parametrize(
    ("target_name", "expected"),
    [
        ("kplr008120608", "8120608"),
        ("KPLR008120608", "8120608"),
        ("ktwo201111557", "201111557"),
        ("251848941", "251848941"),
        ("kplr000000001", "1"),
    ],
)
def test_catalog_id_normalisation(target_name, expected):
    assert _catalog_id_from_target_name(target_name) == expected


@pytest.mark.parametrize("target_name", ["kplr000000000", "000", "", "kplr"])
def test_target_name_with_no_identifier_raises(target_name):
    with pytest.raises(ValueError, match="no catalogue identifier"):
        _catalog_id_from_target_name(target_name)


def test_unusable_identifier_raises_target_not_found(monkeypatch):
    monkeypatch.setattr(
        api,
        "_search_target",
        lambda q, m: [{"target_name": "kplr000000000", "s_ra": 1.0, "s_dec": 2.0}],
    )

    with pytest.raises(TargetNotFound, match="unusable identifier"):
        resolve("Kepler-186", "Kepler")


@pytest.mark.parametrize(
    "coords",
    [
        {"s_ra": float("nan"), "s_dec": 2.0},
        {"s_ra": 1.0, "s_dec": float("nan")},
        {"s_ra": float("inf"), "s_dec": 2.0},
        {"s_ra": None, "s_dec": 2.0},
        {"s_ra": 1.0, "s_dec": None},
    ],
    ids=["nan-ra", "nan-dec", "inf", "none-ra", "none-dec"],
)
def test_bad_coordinates_raise_target_not_found(monkeypatch, coords):
    _stub(monkeypatch, [_row(**coords)])

    with pytest.raises(TargetNotFound, match="Kepler-186"):
        resolve("Kepler-186", "Kepler")


def test_package_version_matches_installed_metadata():
    """The version is embedded in the calling service's cache keys."""
    import importlib.metadata

    import shaula

    assert importlib.metadata.version("shaula") == shaula.__version__


@pytest.mark.parametrize(
    ("catalog", "catalog_id", "expected"),
    [
        ("KIC", "8120608", "KIC 8120608"),
        ("EPIC", "201111557", "EPIC 201111557"),
        ("TIC", "251848941", "TIC 251848941"),
    ],
)
def test_designation_is_catalog_and_id(catalog, catalog_id, expected):
    target = ResolvedTarget(catalog, catalog_id, 1.0, 2.0, "x")
    assert target.designation == expected


@pytest.mark.parametrize(
    ("given", "canonical"),
    [("kepler", "Kepler"), (" TESS ", "TESS"), ("k2", "K2"), ("TeSs", "TESS")],
)
def test_canonical_mission(given, canonical):
    assert canonical_mission(given) == canonical


def test_canonical_mission_names_the_offender():
    with pytest.raises(ValueError, match="Hubble"):
        canonical_mission("Hubble")


def test_resolve_accepts_loose_mission_spelling(monkeypatch):
    asked = []

    def spy(query, mission):
        asked.append(mission)
        return [_row(target_name="251848941")]

    monkeypatch.setattr(api, "_search_target", spy)
    assert resolve("TOI-178", " TESS ").catalog == "TIC"
    assert resolve("Kepler-186", "kepler").catalog == "KIC"
    assert asked == ["TESS", "Kepler"]


def test_overlong_query_is_rejected_without_searching(monkeypatch):
    def boom(q, m):
        raise AssertionError("search must not be attempted")

    monkeypatch.setattr(api, "_search_target", boom)
    with pytest.raises(ValueError, match=str(MAX_QUERY_LENGTH)):
        resolve("x" * (MAX_QUERY_LENGTH + 1), "Kepler")


def test_query_at_the_limit_is_searched(monkeypatch):
    _stub(monkeypatch, [_row()])
    assert resolve("x" * MAX_QUERY_LENGTH, "Kepler").catalog_id == "8120608"


def test_search_target_uses_the_shared_filtered_helper(monkeypatch):
    seen = {}

    class _Result:
        table: ClassVar[list[str]] = ["the-table"]

    def fake_helper(target, mission, **kwargs):
        seen["args"] = (target, mission)
        return api.download_and_clean.ProductSearch(_Result(), "A", "short")

    monkeypatch.setattr(api.download_and_clean, "search_default_product", fake_helper)
    assert api._search_target("Kepler-186", "Kepler") == ["the-table"]
    assert seen["args"] == ("Kepler-186", "Kepler")


class _FakeSearchFn:
    """Mimics a memoised lightkurve.search_lightcurve."""

    def __init__(self, with_wrapped=True):
        self.calls = []
        self.memoised_calls = []
        if with_wrapped:
            self.__wrapped__ = self._wrapped

    def _wrapped(self, target, **kwargs):
        self.calls.append((target, kwargs))
        return ["result"]

    def __call__(self, target, **kwargs):
        self.memoised_calls.append((target, kwargs))
        return ["memoised"]


def test_helper_passes_default_author_and_exptime_to_the_unmemoised_function(monkeypatch):
    import lightkurve as lk

    from .. import download_and_clean as dac

    fake = _FakeSearchFn()
    monkeypatch.setattr(lk, "search_lightcurve", fake)
    author, exptime = default_product_selection("TESS")

    found = dac.search_default_product("TOI-178", "TESS")

    assert fake.memoised_calls == []
    assert fake.calls == [("TOI-178", {"mission": "TESS", "author": author, "exptime": exptime})]
    assert found.result == ["result"]
    assert (found.author, found.exptime) == (author, exptime)


def test_helper_falls_back_when_search_function_is_not_wrapped(monkeypatch):
    import lightkurve as lk

    from .. import download_and_clean as dac

    fake = _FakeSearchFn(with_wrapped=False)
    monkeypatch.setattr(lk, "search_lightcurve", fake)

    found = dac.search_default_product("Kepler-186", "Kepler")

    assert found.result == ["memoised"]
    assert fake.memoised_calls[0][1]["mission"] == "Kepler"


def _prow(target_name, distance, **overrides):
    return _row(target_name=target_name, distance=distance, **overrides)


def _stub_position(monkeypatch, rows):
    monkeypatch.setattr(api, "_search_position", lambda ra, dec, m: rows)


def _forbid_search(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("no search may be attempted")

    monkeypatch.setattr(api, "_search_target", boom)
    monkeypatch.setattr(api, "_search_position", boom)


def test_position_resolves_to_the_catalog_id(monkeypatch):
    _stub_position(monkeypatch, [_prow("kplr008120608", 0.4)])
    resolved = resolve(ra_deg=298.65272, dec_deg=43.95502, mission="Kepler")

    assert isinstance(resolved, ResolvedTarget)
    assert (resolved.catalog, resolved.catalog_id) == ("KIC", "8120608")
    assert resolved.designation == "KIC 8120608"
    assert resolved.display_name == "KIC 8120608"
    assert resolved.ra_deg == pytest.approx(298.65272)


def test_nearest_star_wins_regardless_of_row_order(monkeypatch):
    rows = [
        _prow("kplr000000002", 3.2),
        _prow("kplr008120608", 0.4),
        _prow("kplr000000003", 3.9),
    ]
    _stub_position(monkeypatch, rows)

    assert resolve(ra_deg=1.0, dec_deg=2.0).catalog_id == "8120608"
    _stub_position(monkeypatch, list(reversed(rows)))
    assert resolve(ra_deg=1.0, dec_deg=2.0).catalog_id == "8120608"


def test_several_light_curves_of_one_star_resolve_to_it(monkeypatch):
    _stub_position(
        monkeypatch,
        [_prow("kplr008120608", 0.4), _prow("kplr008120608", 0.4), _prow("kplr000000002", 2.0)],
    )
    assert resolve(ra_deg=1.0, dec_deg=2.0).catalog_id == "8120608"


def test_position_passes_coordinates_and_the_canonical_mission(monkeypatch):
    asked = []

    def spy(ra, dec, mission):
        asked.append((ra, dec, mission))
        return [_prow("251848941", 1.0)]

    monkeypatch.setattr(api, "_search_position", spy)
    resolved = resolve(ra_deg=10.5, dec_deg=-20.25, mission=" tess ")

    assert asked == [(10.5, -20.25, "TESS")]
    assert resolved.catalog == "TIC"


def test_zero_coordinates_are_a_real_position(monkeypatch):
    asked = []

    def spy(ra, dec, mission):
        asked.append((ra, dec))
        return [_prow("kplr008120608", 0.0)]

    monkeypatch.setattr(api, "_search_position", spy)
    resolve(ra_deg=0.0, dec_deg=0.0)

    assert asked == [(0.0, 0.0)]


def test_position_with_no_star_in_range_is_not_found(monkeypatch):
    _stub_position(monkeypatch, [])

    with pytest.raises(TargetNotFound, match="arcseconds"):
        resolve(ra_deg=10.0, dec_deg=10.0, mission="Kepler")


@pytest.mark.parametrize(
    "rows",
    [
        [_row()],
        [_prow("kplr008120608", float("nan"))],
        [_prow("kplr008120608", None)],
    ],
    ids=["no-distance-column", "nan-distance", "none-distance"],
)
def test_rows_without_a_readable_distance_are_not_found(monkeypatch, rows):
    _stub_position(monkeypatch, rows)

    with pytest.raises(TargetNotFound, match="distance"):
        resolve(ra_deg=10.0, dec_deg=10.0)


def test_position_row_with_unusable_identifier_is_not_found(monkeypatch):
    _stub_position(monkeypatch, [_prow("kplr000000000", 0.1)])

    with pytest.raises(TargetNotFound, match="unusable identifier"):
        resolve(ra_deg=10.0, dec_deg=10.0)


@pytest.mark.parametrize(
    "coords",
    [{"s_ra": float("nan"), "s_dec": 2.0}, {"s_ra": 1.0, "s_dec": None}],
    ids=["nan-ra", "none-dec"],
)
def test_position_row_with_bad_coordinates_is_not_found(monkeypatch, coords):
    _stub_position(monkeypatch, [_prow("kplr008120608", 0.1, **coords)])

    with pytest.raises(TargetNotFound, match=r"ra=10\.0"):
        resolve(ra_deg=10.0, dec_deg=10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": "Kepler-186", "ra_deg": 1.0, "dec_deg": 2.0},
        {"query": "Kepler-186", "ra_deg": 1.0},
        {"query": "Kepler-186", "dec_deg": 2.0},
        {},
        {"query": ""},
        {"query": "   "},
        {"ra_deg": 1.0},
        {"dec_deg": 2.0},
    ],
    ids=[
        "query-and-position",
        "query-and-ra",
        "query-and-dec",
        "nothing",
        "empty-query",
        "blank-query",
        "only-ra",
        "only-dec",
    ],
)
def test_invalid_mode_never_searches(monkeypatch, kwargs):
    _forbid_search(monkeypatch)

    with pytest.raises(ValueError, match=r"query|ra_deg"):
        resolve(**kwargs)


@pytest.mark.parametrize(
    ("ra", "dec", "message"),
    [
        (360.0, 0.0, "ra_deg"),
        (-0.1, 0.0, "ra_deg"),
        (10.0, 90.1, "dec_deg"),
        (10.0, -90.1, "dec_deg"),
        (float("nan"), 0.0, "ra_deg"),
        (0.0, float("inf"), "dec_deg"),
        ("10", 0.0, "ra_deg"),
        (0.0, "10", "dec_deg"),
        (True, 0.0, "ra_deg"),
    ],
    ids=[
        "ra-high",
        "ra-negative",
        "dec-high",
        "dec-low",
        "ra-nan",
        "dec-inf",
        "ra-string",
        "dec-string",
        "ra-bool",
    ],
)
def test_bad_position_values_never_search(monkeypatch, ra, dec, message):
    _forbid_search(monkeypatch)

    with pytest.raises(ValueError, match=message):
        resolve(ra_deg=ra, dec_deg=dec)


@pytest.mark.parametrize(("ra", "dec"), [(0.0, -90.0), (0.0, 90.0), (359.999999, 0.0)])
def test_position_range_edges_are_accepted(monkeypatch, ra, dec):
    _stub_position(monkeypatch, [_prow("kplr008120608", 0.1)])
    assert resolve(ra_deg=ra, dec_deg=dec).catalog_id == "8120608"


def test_position_with_unsupported_mission_never_searches(monkeypatch):
    _forbid_search(monkeypatch)

    with pytest.raises(ValueError, match="Hubble"):
        resolve(ra_deg=1.0, dec_deg=2.0, mission="Hubble")


def test_name_resolution_never_uses_the_position_search(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("position search must not run for a name")

    monkeypatch.setattr(api, "_search_position", boom)
    _stub(monkeypatch, [_row()])
    assert resolve("Kepler-186", "Kepler").catalog_id == "8120608"


def test_search_position_uses_the_shared_helper_with_a_cone(monkeypatch):
    seen = {}

    class _Result:
        table: ClassVar[list[str]] = ["the-table"]

    def fake_helper(target, mission, **kwargs):
        seen.update(target=target, mission=mission, kwargs=kwargs)
        return api.download_and_clean.ProductSearch(_Result(), "A", "short")

    monkeypatch.setattr(api.download_and_clean, "search_default_product", fake_helper)

    assert api._search_position(298.65, 43.95, "Kepler") == ["the-table"]
    assert isinstance(seen["target"], SkyCoord)
    assert seen["target"].ra.deg == pytest.approx(298.65)
    assert seen["target"].dec.deg == pytest.approx(43.95)
    assert seen["mission"] == "Kepler"
    assert seen["kwargs"] == {"radius": POSITION_MATCH_RADIUS_ARCSEC}
    assert POSITION_MATCH_RADIUS_ARCSEC == 4.0


def test_helper_forwards_a_radius_only_when_given(monkeypatch):
    import lightkurve as lk

    from .. import download_and_clean as dac

    fake = _FakeSearchFn()
    monkeypatch.setattr(lk, "search_lightcurve", fake)

    dac.search_default_product("Kepler-186", "Kepler")
    dac.search_default_product("Kepler-186", "Kepler", radius=4.0)

    assert "radius" not in fake.calls[0][1]
    assert fake.calls[1][1]["radius"] == 4.0
