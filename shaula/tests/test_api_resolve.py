"""Resolving a user-supplied name to an exact catalog identifier."""

from __future__ import annotations

from typing import ClassVar

import pytest

from .. import api
from ..api import (
    MAX_QUERY_LENGTH,
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


def test_non_finite_coordinates_raise_target_not_found(monkeypatch):
    monkeypatch.setattr(
        api,
        "_search_target",
        lambda q, m: [{"target_name": "kplr008120608", "s_ra": float("nan"), "s_dec": 2.0}],
    )

    with pytest.raises(TargetNotFound, match="non-finite coordinates"):
        resolve("Kepler-186", "Kepler")


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
