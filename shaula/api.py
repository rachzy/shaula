"""Shaula's public surface: the only module external consumers import."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .download_and_clean import download_and_clean_lightcurve
from .extract_feats import extract_features_from_lightcurve

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
