from pathlib import Path
from typing import Any, NamedTuple

import lightkurve as lk
import numpy as np
from numpy import inf

from .paths import lightcurve_cache_dir
from .save import save_lightkurve

# 1 - Dowload and clean light curve


def default_product_selection(mission: str) -> tuple[str | None, str | None]:
    """Return a consistent official product and exposure class per mission."""
    mission_key = str(mission).strip().casefold()
    if mission_key == "kepler":
        return "Kepler", "long"
    if mission_key == "k2":
        return "K2", "long"
    if mission_key == "tess":
        return "SPOC", "short"
    return None, None


# Used, in this order, when a TESS star has no 2-minute SPOC light curve, which
# is most of them: only preselected targets get 2-minute data, everything else
# is cut from the full-frame images. TESS-SPOC comes first because its PDCSAP
# flux is corrected like the 2-minute product; QLP covers fainter stars.
# Neither is pinned to one exposure, because the full-frame cadence changed
# with the mission (1800 s, then 600 s, then 200 s).
TESS_DEFAULT_AUTHOR = "SPOC"
TESS_FFI_FALLBACK_AUTHORS = ("TESS-SPOC", "QLP")
# lightkurve's "short" exposure class, in seconds.
SHORT_EXPTIME_RANGE = (60.0, 200.0)
# Light curves whose cadence is within this factor of the coarsest are left
# alone when a stitched set is brought to one cadence.
COMMON_CADENCE_TOLERANCE = 0.75


def _normalize_exptime(
    exptime: str | float | None,
) -> str | float | None:
    """Let CLI callers pass either Lightkurve classes or seconds."""
    if not isinstance(exptime, str):
        return exptime
    value = exptime.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return value


class ProductSearch(NamedTuple):
    """A search result plus the author and exposure class that selected it."""

    result: Any
    author: str | None
    exptime: str | float | None


def search_default_product(
    target: Any,
    mission: str,
    author: str | None = None,
    exptime: str | float | None = None,
    *,
    radius: float | None = None,
) -> ProductSearch:
    """Search for one consistent product class, without lightkurve's memo.

    Shared by ``resolve()`` and the downloader so both look up the same
    thing. ``lk.search_lightcurve`` is memoised without a size or time limit,
    which would keep every network-supplied string for the life of the
    process and cache an empty result from a transient archive failure
    forever; the unmemoised function is called instead.

    A TESS search left on its defaults falls back to full-frame-image light
    curves (``TESS_FFI_FALLBACK_AUTHORS``, any exposure) when the star has no
    2-minute SPOC product. All three authors go into one query and the class
    is picked from its rows: a second query for a star with no 2-minute data
    would send lightkurve to MAST's name resolver first. An explicit
    ``author`` or ``exptime`` is searched exactly as given.
    """
    default_author, default_exptime = default_product_selection(mission)
    selected_author = author if author is not None else default_author
    # The default exposure class belongs to the default author: a caller who
    # names another one (QLP has no 2-minute data) gets whatever it offers.
    if exptime is None and selected_author != default_author:
        default_exptime = None
    selected_exptime = _normalize_exptime(
        exptime if exptime is not None else default_exptime
    )
    search_kwargs: dict[str, Any] = {"mission": mission}
    if selected_author is not None:
        search_kwargs["author"] = selected_author
    if selected_exptime is not None:
        search_kwargs["exptime"] = selected_exptime
    if radius is not None:
        search_kwargs["radius"] = radius

    search = getattr(lk.search_lightcurve, "__wrapped__", lk.search_lightcurve)
    on_defaults = author is None and exptime is None
    if on_defaults and default_author == TESS_DEFAULT_AUTHOR:
        search_kwargs.pop("exptime", None)
        search_kwargs["author"] = [TESS_DEFAULT_AUTHOR, *TESS_FFI_FALLBACK_AUTHORS]
        return _select_tess_default(
            search(target, **search_kwargs), nearest_only=radius is not None
        )
    return ProductSearch(
        search(target, **search_kwargs), selected_author, selected_exptime
    )


def _select_tess_default(result, *, nearest_only: bool) -> ProductSearch:
    """Keep one product class from a search over every default TESS author.

    2-minute SPOC rows when the star has any, otherwise the first
    full-frame-image author that has rows. ``nearest_only`` first narrows a
    cone search to its nearest star, so a neighbour with 2-minute data cannot
    displace the star that was asked for.
    """
    if len(result) == 0:
        return ProductSearch(result, TESS_DEFAULT_AUTHOR, "short")
    table = result.table
    if nearest_only and "distance" in table.colnames:
        distance = np.asarray(table["distance"], dtype=float)
        if np.any(np.isfinite(distance)):
            nearest = table["target_name"][int(np.nanargmin(distance))]
            result = result[np.asarray(table["target_name"] == nearest)]
            table = result.table

    authors = np.char.upper(np.asarray(table["author"], dtype=str))
    seconds = np.asarray(table["exptime"], dtype=float)
    low, high = SHORT_EXPTIME_RANGE
    two_minute = (authors == TESS_DEFAULT_AUTHOR) & (seconds >= low) & (seconds < high)
    if np.any(two_minute):
        return ProductSearch(result[two_minute], TESS_DEFAULT_AUTHOR, "short")
    for fallback_author in TESS_FFI_FALLBACK_AUTHORS:
        rows = authors == fallback_author
        if np.any(rows):
            return ProductSearch(result[rows], fallback_author, None)
    return ProductSearch(result[:0], TESS_DEFAULT_AUTHOR, "short")


def _cadence_days(lc) -> float:
    steps = np.diff(np.asarray(lc.time.value, dtype=float))
    steps = steps[np.isfinite(steps) & (steps > 0)]
    return float(np.median(steps)) if steps.size else float("nan")


def bin_to_common_cadence(light_curves):
    """Bin every light curve to the coarsest cadence in the set.

    Full-frame-image products of one star mix 1800 s, 600 s and 200 s
    sectors. Stitched as they are, the noise per point and every
    cadence-dependent feature would change part-way through the series.
    A set already at one cadence is returned untouched.
    """
    cadences = [_cadence_days(lc) for lc in light_curves]
    finite = [cadence for cadence in cadences if np.isfinite(cadence)]
    if not finite:
        return light_curves
    coarsest = max(finite)
    if min(finite) >= COMMON_CADENCE_TOLERANCE * coarsest:
        return light_curves
    binned = [
        lc.bin(time_bin_size=coarsest)
        if np.isfinite(cadence) and cadence < COMMON_CADENCE_TOLERANCE * coarsest
        else lc
        for lc, cadence in zip(light_curves, cadences)
    ]
    return lk.LightCurveCollection(binned)


def download_and_clean_lightcurve(
    target: str,
    mission: str,
    sigma_upper: float = 5.0,
    all: bool = False,
    savePath: str | None = None,
    verbose: bool = False,
    author: str | None = None,
    exptime: str | float | None = None,
    *,
    cache_dir: Path | str | None = None,
):
    """Download one consistent light-curve product class and clean it.

    Unless explicitly overridden, Kepler/K2 use official long-cadence products
    and TESS uses SPOC short-cadence products, or one full-frame-image author
    when the star has none.  Filtering happens before ``download_all`` so
    stitching cannot mix pipeline authors; a set searched without an exposure
    class is binned to its coarsest cadence first.
    """
    download_dir = lightcurve_cache_dir(cache_dir)
    download_dir_arg = str(download_dir) if download_dir is not None else None

    if verbose:
        print(f"Downloading light curve for {target} from {mission}...")
    search_result, selected_author, selected_exptime = search_default_product(
        target, mission, author, exptime
    )
    if len(search_result) == 0:
        raise ValueError(
            f"No {mission} light curves found for {target!r} with "
            f"author={selected_author!r} and exptime={selected_exptime!r}. "
            "Override author/exptime to select another consistent product."
        )

    if verbose:
        print(
            f"Selected {len(search_result)} product(s): "
            f"author={selected_author or 'any'}, "
            f"exptime={selected_exptime or 'any'}"
        )

    if all:
        if verbose:
            print(f"Downloading all {len(search_result)} selected files...")
        lc = search_result.download_all(download_dir=download_dir_arg)
        if selected_exptime is None:
            lc = bin_to_common_cadence(lc)
        lc = lc.stitch()
    else:
        lc = search_result.download(download_dir=download_dir_arg)

    lc = lc.remove_nans().normalize().remove_outliers(sigma_upper=sigma_upper, sigma_lower=inf)
    print(f"Downloaded {len(lc.time)} data points.")
    if savePath:
        save_lightkurve(lc, target, savePath)

    return lc
