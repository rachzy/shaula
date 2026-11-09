"""Per-mission settings for the transit search.

Kepler stares at one field for four years, so the search can size itself from
the first-to-last span and ask for three transits. TESS observes in 27-day
sectors that may sit years apart, and its own pipeline accepts two transits.
A profile carries those differences; the default reproduces the Kepler rules.
"""

from __future__ import annotations

from dataclasses import dataclass

PERIOD_BASELINE_SPAN = "span"
PERIOD_BASELINE_OBSERVED = "observed"
# Kepler long cadence: 270 exposures of 6.02 s plus readout, 29.4244 minutes.
KEPLER_LONG_CADENCE_DAYS = 29.4244 / 1440.0


@dataclass(frozen=True)
class SearchProfile:
    """How the period search bounds itself and what counts as a detection.

    ``min_transit_events`` is the number of observed transits a candidate
    needs, and also sets the longest period searched (baseline / events).
    ``period_baseline`` is ``"span"`` (last minus first cadence) or
    ``"observed"`` (time actually covered by data, gaps left out).
    ``feature_cadence_days`` bins the detrended series to that cadence before
    the features and the detection statistics are measured; the period search
    and the detrending still run at the native cadence. ``None`` measures at
    the native cadence.
    """

    min_transit_events: int = 3
    period_baseline: str = PERIOD_BASELINE_SPAN
    feature_cadence_days: float | None = None

    def __post_init__(self) -> None:
        if int(self.min_transit_events) < 2:
            raise ValueError("min_transit_events must be at least 2")
        if self.period_baseline not in (
            PERIOD_BASELINE_SPAN,
            PERIOD_BASELINE_OBSERVED,
        ):
            raise ValueError(
                f"period_baseline must be 'span' or 'observed', "
                f"got {self.period_baseline!r}"
            )
        cadence = self.feature_cadence_days
        if cadence is not None and not cadence > 0:
            raise ValueError("feature_cadence_days must be positive or None")


DEFAULT_PROFILE = SearchProfile()
# Two transits, as in the TESS SPOC search; the baseline is the observed time
# because sectors of one star can be years apart (TOI-270: ~190 d of data over
# a 2,660 d span, which the span rule searched out to 887 d).
# Features are measured at Kepler's long cadence so a TESS row means what a
# Kepler row means: the autocorrelation lags, points per transit, flux moments
# and outlier fractions all depend on the cadence, and 2-minute data would
# otherwise put every TESS row outside what a Kepler-trained model has seen.
TESS_PROFILE = SearchProfile(
    min_transit_events=2,
    period_baseline=PERIOD_BASELINE_OBSERVED,
    feature_cadence_days=KEPLER_LONG_CADENCE_DAYS,
)

_MISSION_PROFILES = {"tess": TESS_PROFILE}


def profile_for_mission(mission: str | None) -> SearchProfile:
    """The search profile for ``mission``; Kepler's rules for anything else."""
    key = str(mission).strip().casefold() if mission is not None else ""
    return _MISSION_PROFILES.get(key, DEFAULT_PROFILE)
