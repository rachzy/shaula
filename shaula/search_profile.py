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


@dataclass(frozen=True)
class SearchProfile:
    """How the period search bounds itself and what counts as a detection.

    ``min_transit_events`` is the number of observed transits a candidate
    needs, and also sets the longest period searched (baseline / events).
    ``period_baseline`` is ``"span"`` (last minus first cadence) or
    ``"observed"`` (time actually covered by data, gaps left out).
    """

    min_transit_events: int = 3
    period_baseline: str = PERIOD_BASELINE_SPAN

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


DEFAULT_PROFILE = SearchProfile()
# Two transits, as in the TESS SPOC search; the baseline is the observed time
# because sectors of one star can be years apart (TOI-270: ~190 d of data over
# a 2,660 d span, which the span rule searched out to 887 d).
TESS_PROFILE = SearchProfile(
    min_transit_events=2, period_baseline=PERIOD_BASELINE_OBSERVED
)

_MISSION_PROFILES = {"tess": TESS_PROFILE}


def profile_for_mission(mission: str | None) -> SearchProfile:
    """The search profile for ``mission``; Kepler's rules for anything else."""
    key = str(mission).strip().casefold() if mission is not None else ""
    return _MISSION_PROFILES.get(key, DEFAULT_PROFILE)
