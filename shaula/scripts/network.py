"""Bounded MAST requests for the batch scripts.

astroquery and lightkurve send their MAST queries and file downloads through
``requests`` with no timeout, so a stalled connection blocks a worker until the
operating system drops it, which took hours during the first large training
run. The scripts install a default timeout and retry a failed download instead.
Library callers (``shaula.api``) are left alone: a process-wide patch is the
caller's decision.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TypeVar

import requests
from astroquery.exceptions import RemoteServiceError

# (connect, read) seconds. The read timeout bounds the wait for each chunk, not
# the whole transfer, so a slow but live download still completes.
DEFAULT_REQUEST_TIMEOUT = (30.0, 120.0)
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 30.0

T = TypeVar("T")


def install_default_request_timeout(timeout=DEFAULT_REQUEST_TIMEOUT) -> None:
    """Give every ``requests`` call in this process a timeout unless it sets one."""
    request = requests.Session.request
    if getattr(request, "_shaula_default_timeout", None) is not None:
        return

    @functools.wraps(request)
    def request_with_timeout(self, method, url, *args, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return request(self, method, url, *args, **kwargs)

    request_with_timeout._shaula_default_timeout = timeout
    requests.Session.request = request_with_timeout


def is_network_error(exc: BaseException) -> bool:
    # RemoteServiceError is MAST's own "Request failed" during an outage.
    return isinstance(
        exc,
        (
            requests.exceptions.RequestException,
            RemoteServiceError,
            TimeoutError,
            ConnectionError,
        ),
    )


def with_retries(
    call: Callable[[], T],
    *,
    attempts: int = DOWNLOAD_ATTEMPTS,
    backoff: float = RETRY_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``call``, retrying network errors with a linear backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts or not is_network_error(exc):
                raise
            wait = backoff * attempt
            print(f"Network error ({exc!r}); retry {attempt}/{attempts - 1} in {wait:.0f} s.")
            sleep(wait)
    raise AssertionError("unreachable")
