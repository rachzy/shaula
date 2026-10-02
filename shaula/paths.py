"""Filesystem locations Shaula reads and writes."""

from __future__ import annotations

import os
from pathlib import Path

LIGHTCURVE_CACHE_ENV = "SHAULA_LIGHTCURVE_CACHE"


def lightcurve_cache_dir(explicit: Path | str | None = None) -> Path:
    """Where downloaded light curves are cached, creating it if needed.

    Resolution order: explicit argument, then ``SHAULA_LIGHTCURVE_CACHE``,
    then a package-relative default for local development.
    """
    if explicit is not None:
        directory = Path(explicit)
    else:
        from_env = os.environ.get(LIGHTCURVE_CACHE_ENV)
        directory = (
            Path(from_env)
            if from_env
            else Path(__file__).resolve().parent / "data" / "lightkurve"
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory
