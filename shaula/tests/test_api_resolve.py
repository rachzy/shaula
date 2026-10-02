"""Resolving a user-supplied name to an exact catalog identifier."""

from __future__ import annotations

import pytest

from .. import api
from ..api import ResolvedTarget, TargetNotFound, _catalog_id_from_target_name, resolve


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
