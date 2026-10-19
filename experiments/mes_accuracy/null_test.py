"""Null test: MES at random ephemerides on transit-free light curves.

A calibrated statistic folds pure noise to MES ~ N(0, 1). Reports the mean,
std and 99th percentile of the exact-ephemeris MES for each noise model.
"""

import contextlib
import functools
import io

import numpy as np
from shaula import cdpp
from shaula.sesmes import fold_statistics

from experiments.mes_accuracy import synth

ORIGINAL = cdpp.duration_matched_statistics


MODELS = {
    "current": functools.partial(ORIGINAL, flank_durations=None),
    "flank1": functools.partial(ORIGINAL, flank_durations=1.0),
    "adaptive": functools.partial(ORIGINAL, flank_durations="auto"),
}

STARS = {
    "white": dict(sigma_ppm=100),
    "spot12": dict(sigma_ppm=100, amp_rot=2000e-6, prot=12.0),
    "spot3": dict(sigma_ppm=100, amp_rot=500e-6, prot=3.0),
    "gran": dict(sigma_ppm=100, amp_gran=80e-6),
}


def main(n_trials=300, seed=7):
    rng = np.random.default_rng(seed)
    for star, kw in STARS.items():
        lc = synth.make_lightcurve(period=10.0, depth=0.0, duration_days=0.15,
                                   n_quarters=8, seed=3, **kw)
        # Mimic the pipeline's (ineffective) final spline: none, i.e. raw flux.
        t, f = lc["time"], lc["flux"]
        for dur in (0.12, 0.25):
            for name, model in MODELS.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    stats = model(t, f, duration_hours=dur * 24)
                mes = []
                for _ in range(n_trials):
                    period = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
                    epoch = t[0] + rng.uniform(0, period)
                    mes.append(fold_statistics(stats, period, epoch, dur)["MES"])
                mes = np.asarray(mes)
                mes = mes[np.isfinite(mes)]
                print(
                    f"{star:7s} T14={dur:.2f} {name:9s} mean={mes.mean():+.2f} "
                    f"std={mes.std():.2f} p99={np.percentile(mes, 99):.2f} "
                    f"max={mes.max():.2f}"
                )


if __name__ == "__main__":
    main()
