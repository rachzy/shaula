"""Filesystem locations Shaula reads and writes."""

from __future__ import annotations

import os
from pathlib import Path

LIGHTCURVE_CACHE_ENV = "SHAULA_LIGHTCURVE_CACHE"


def lightcurve_cache_dir(explicit: Path | str | None = None) -> Path | None:
    """Resolve where lightkurve should download files for one call.

    Resolution order: the explicit argument, then ``SHAULA_LIGHTCURVE_CACHE``
    (an empty value counts as unset), then ``None``, which means "let
    lightkurve use its own default". A returned directory is created,
    parents included.

    The result must be passed to lightkurve per call, as ``download_dir``.
    It is never assigned to ``lightkurve.conf.cache_dir``: that setting is
    process-wide and sticky, so concurrent calls in a threaded service would
    overwrite each other's location and a later call with no argument would
    inherit the previous one. lightkurve 2.6 also ignores a
    ``LIGHTKURVE_CACHE_DIR`` environment variable, so this resolver and the
    per-call parameter are the only way to redirect downloads. This function
    deliberately does not import or touch lightkurve.
    """
    if explicit is not None:
        directory = Path(explicit)
    else:
        from_env = os.environ.get(LIGHTCURVE_CACHE_ENV)
        if not from_env:
            return None
        directory = Path(from_env)

    directory.mkdir(parents=True, exist_ok=True)
    return directory
