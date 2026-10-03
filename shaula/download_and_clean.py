from pathlib import Path
from typing import Any, NamedTuple

import lightkurve as lk
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
    target: str,
    mission: str,
    author: str | None = None,
    exptime: str | float | None = None,
) -> ProductSearch:
    """Search for one consistent product class, without lightkurve's memo.

    Shared by ``resolve()`` and the downloader so both look up the same
    thing. ``lk.search_lightcurve`` is memoised without a size or time limit,
    which would keep every network-supplied string for the life of the
    process and cache an empty result from a transient archive failure
    forever; the unmemoised function is called instead.
    """
    default_author, default_exptime = default_product_selection(mission)
    selected_author = author if author is not None else default_author
    selected_exptime = _normalize_exptime(
        exptime if exptime is not None else default_exptime
    )
    search_kwargs: dict[str, Any] = {"mission": mission}
    if selected_author is not None:
        search_kwargs["author"] = selected_author
    if selected_exptime is not None:
        search_kwargs["exptime"] = selected_exptime

    search = getattr(lk.search_lightcurve, "__wrapped__", lk.search_lightcurve)
    return ProductSearch(
        search(target, **search_kwargs), selected_author, selected_exptime
    )


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
    and TESS uses SPOC short-cadence products.  Filtering happens before
    ``download_all`` so stitching cannot mix exposure times or pipeline
    authors.
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
        lc = lc.stitch()
    else:
        lc = search_result.download(download_dir=download_dir_arg)

    lc = lc.remove_nans().normalize().remove_outliers(sigma_upper=sigma_upper, sigma_lower=inf)
    print(f"Downloaded {len(lc.time)} data points.")
    if savePath:
        save_lightkurve(lc, target, savePath)

    return lc
