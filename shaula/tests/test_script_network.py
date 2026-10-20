"""Timeouts and retries around the batch scripts' MAST requests."""

from __future__ import annotations

import pytest
import requests

from ..scripts import network


def test_a_request_without_a_timeout_gets_the_default(monkeypatch):
    seen = {}

    def fake_request(self, method, url, *args, **kwargs):
        seen.update(kwargs)
        return "response"

    monkeypatch.setattr(requests.Session, "request", fake_request)
    network.install_default_request_timeout((1.0, 2.0))
    session = requests.Session()
    assert session.request("GET", "https://example.invalid") == "response"
    assert seen["timeout"] == (1.0, 2.0)
    session.request("GET", "https://example.invalid", timeout=5)
    assert seen["timeout"] == 5
    # Installing twice does not wrap twice.
    wrapped = requests.Session.request
    network.install_default_request_timeout()
    assert requests.Session.request is wrapped


def test_network_errors_are_retried_then_raised():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ReadTimeout("stalled")
        return "lc"

    assert network.with_retries(flaky, sleep=lambda s: None) == "lc"
    assert len(calls) == 3

    def always_down():
        raise requests.exceptions.ConnectionError("reset")

    with pytest.raises(requests.exceptions.ConnectionError):
        network.with_retries(always_down, attempts=2, sleep=lambda s: None)


def test_other_errors_are_not_retried():
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("No Kepler light curves found")

    with pytest.raises(ValueError):
        network.with_retries(broken, sleep=lambda s: None)
    assert len(calls) == 1


def test_a_mast_service_failure_is_retried():
    from astroquery.exceptions import RemoteServiceError

    calls = []

    def outage():
        calls.append(1)
        if len(calls) == 1:
            raise RemoteServiceError("Request failed.")
        return "lc"

    assert network.with_retries(outage, sleep=lambda s: None) == "lc"
