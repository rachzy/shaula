"""Filesystem locations Shaula reads and writes."""

from __future__ import annotations

import os
from pathlib import Path

LIGHTCURVE_CACHE_ENV = "SHAULA_LIGHTCURVE_CACHE"


def lightcurve_cache_dir(explicit: Path | str | None = None) -> Path:
    """Point lightkurve's download cache at a configurable directory.

    Resolution order: explicit argument, then ``SHAULA_LIGHTCURVE_CACHE``,
    then lightkurve's own default.

    Assigning ``lightkurve.conf.cache_dir`` is what actually redirects
    downloads. lightkurve 2.6 ignores a ``LIGHTKURVE_CACHE_DIR``
    environment variable, so a mounted cache volume does nothing unless
    this function runs before the first search.
    """
    import lightkurve as lk
    from lightkurve.config import get_cache_dir

    if explicit is not None:
        directory = Path(explicit)
    else:
        from_env = os.environ.get(LIGHTCURVE_CACHE_ENV)
        directory = Path(from_env) if from_env else Path(get_cache_dir())

    directory.mkdir(parents=True, exist_ok=True)
    lk.conf.cache_dir = str(directory)
    return directory
