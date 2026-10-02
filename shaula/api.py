"""Shaula's public surface: the only module external consumers import."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .download_and_clean import download_and_clean_lightcurve
from .extract_feats import extract_features_from_lightcurve
from .utils.target_names import host_star_name

STAGES: tuple[str, ...] = (
    "downloading",
    "detrending",
    "period_search",
    "done",
)


@dataclass(frozen=True)
class ProgressEvent:
    """One step of progress through an extraction."""

    stage: str
    message: str
    fraction: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True)
class ExtractionResult:
    """Candidate feature rows for one target, with the version that made them."""

    target: str
    mission: str
    features: list[dict[str, Any]] = field(default_factory=list)
    shaula_version: str = __version__


def _emit(
    progress: ProgressCallback | None,
    stage: str,
    message: str,
    fraction: float | None = None,
) -> None:
    """Hand one event to the caller.

    Exceptions raised by the callback are deliberately not caught: a caller
    raises to cancel, and that must abort the extraction.
    """
    if progress is not None:
        progress(ProgressEvent(stage=stage, message=message, fraction=fraction))


def extract(
    target: str,
    mission: str = "Kepler",
    *,
    sigma_clip: float = 5.0,
    download_all: bool = False,
    author: str | None = None,
    exptime: str | float | None = None,
    use_tls: bool = False,
    mask_eclipses: bool = False,
    cache_dir: Path | str | None = None,
    progress: ProgressCallback | None = None,
) -> ExtractionResult:
    """Download a light curve for ``target`` and extract candidate features.

    ``progress`` receives a :class:`ProgressEvent` at each stage boundary and
    may raise to cancel the run.
    """
    _emit(progress, "downloading", f"Downloading {target} from {mission}")
    lc = download_and_clean_lightcurve(
        target,
        mission,
        sigma_clip,
        download_all,
        author=author,
        exptime=exptime,
        cache_dir=cache_dir,
    )

    _emit(progress, "detrending", "Detrending and searching for periods")
    rows = extract_features_from_lightcurve(
        lc,
        use_tls=use_tls,
        mask_eclipses=mask_eclipses,
        progress=progress,
    )

    features = [dict(row) for row in rows]
    _emit(progress, "done", f"Extracted {len(features)} candidate(s)")
    return ExtractionResult(
        target=target,
        mission=mission,
        features=features,
        shaula_version=__version__,
    )


MISSION_CATALOGS: dict[str, str] = {
    "Kepler": "KIC",
    "K2": "EPIC",
    "TESS": "TIC",
}

_TARGET_NAME_PREFIXES = ("kplr", "ktwo")


class TargetNotFound(Exception):
    """No catalog entry matched the requested target."""


@dataclass(frozen=True)
class ResolvedTarget:
    """A target pinned to an exact catalog identifier.

    ``catalog_id`` is the cache key Antares uses, so it must be the catalog's
    own identifier and never a rounded coordinate.
    """

    catalog: str
    catalog_id: str
    ra_deg: float
    dec_deg: float
    display_name: str


def _search_target(query: str, mission: str):
    """Ask lightkurve to resolve a name, returning its search-result table.

    Isolated behind one function so tests can replace it without a network.
    """
    import lightkurve as lk

    return lk.search_lightcurve(query, mission=mission).table


def _catalog_id_from_target_name(target_name: str) -> str:
    """Strip lightkurve's mission prefix and zero padding from an identifier.

    lightkurve reports Kepler targets as ``kplr008120608`` and K2 targets as
    ``ktwo201111557``, while TESS targets are already a bare TIC number.

    Raises:
        ValueError: when nothing identifying survives the stripping.
    """
    name = str(target_name).strip()
    for prefix in _TARGET_NAME_PREFIXES:
        if name.lower().startswith(prefix):
            name = name[len(prefix) :]
            break
    identifier = name.lstrip("0")
    if not identifier:
        raise ValueError(f"no catalogue identifier in target name {target_name!r}")
    return identifier


def resolve(query: str, mission: str = "Kepler") -> ResolvedTarget:
    """Resolve a name or designation to an exact catalogue identifier."""
    if mission not in MISSION_CATALOGS:
        raise ValueError(
            f"Unsupported mission {mission!r}; expected one of "
            f"{sorted(MISSION_CATALOGS)}."
        )

    catalog = MISSION_CATALOGS[mission]
    star = host_star_name(query)

    table = _search_target(star, mission)
    if table is None or len(table) == 0:
        raise TargetNotFound(
            f"No {mission} light curves found for {query!r}, so it has no "
            f"{catalog} identifier this pipeline can use."
        )

    row = table[0]
    try:
        catalog_id = _catalog_id_from_target_name(row["target_name"])
    except ValueError as error:
        raise TargetNotFound(
            f"{query!r} resolved to an unusable identifier: {error}"
        ) from error

    ra_deg = float(row["s_ra"])
    dec_deg = float(row["s_dec"])
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        raise TargetNotFound(
            f"{query!r} resolved to non-finite coordinates "
            f"(ra={ra_deg!r}, dec={dec_deg!r})."
        )

    return ResolvedTarget(
        catalog=catalog,
        catalog_id=catalog_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        display_name=star,
    )
