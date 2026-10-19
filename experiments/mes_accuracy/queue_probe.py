"""Print one search iteration's BLS queue with quick MES.

    python experiments/mes_accuracy/queue_probe.py STAR P1:T0:DUR [P2:T0:DUR ...]

The given ephemerides (BKJD, as accepted by an earlier search) are gapped at
1.5 x T14, as the iterative search gaps them, and the queue that the next
iteration would try is listed with each peak's quick MES. Used to tell
whether a missed planet's period is in the queue at all.
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import os  # noqa: E402

from shaula import detrend_and_period  # noqa: E402
from shaula.extract_feats import (  # noqa: E402
    MAX_PEAKS_TO_VALIDATE,
    _extract_single_candidate_from_arrays,
    _periodic_mask,
    _quick_candidate_diagnostics,
)

from experiments.mes_accuracy.real_stars import DATA  # noqa: E402

if os.environ.get("SHAULA_EXP_PER_BAND"):
    detrend_and_period.BLS_PEAKS_PER_BAND = int(os.environ["SHAULA_EXP_PER_BAND"])
if os.environ.get("SHAULA_EXP_GLOBAL_DUR"):
    detrend_and_period.GLOBAL_BLS_DURATIONS = int(os.environ["SHAULA_EXP_GLOBAL_DUR"])
LIMIT = int(os.environ.get("SHAULA_EXP_SHOW", MAX_PEAKS_TO_VALIDATE))

data = np.load(DATA / f"{sys.argv[1]}.npz")
time, flux = data["time"], data["flux"]
gap = np.zeros(time.size, bool)
for spec in sys.argv[2:]:
    p, t0, d = map(float, spec.split(":"))
    gap |= _periodic_mask(time, p, t0, d, width_factor=1.5)
with contextlib.redirect_stdout(io.StringIO()):
    feats, info, _ = _extract_single_candidate_from_arrays(time, flux, gap_mask=gap)
print(
    f"first: P={feats['period_days']:.4f} T14={feats['duration_days']:.3f} "
    f"max_mes={feats['max_mes']:.2f}"
)
for i, peak in enumerate(info.get("search_candidates", [])[:LIMIT]):
    with contextlib.redirect_stdout(io.StringIO()):
        q = _quick_candidate_diagnostics(time, flux, peak, gap_mask=gap)
    print(
        f"{i:2d} P={peak['period']:10.4f} T14={peak['duration']:.3f} "
        f"power={peak.get('power', np.nan):.4g} "
        f"quick_mes={q['max_mes']:.2f} events={q['n_mes_events']}"
    )
