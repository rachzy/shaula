"""Full search on a synthetic 3-planet system; see LOG.md E10.

Usage: python multi_planet.py <repo-root-to-import-shaula-from> '<cfg json>'
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])  # repo root whose shaula/ is imported
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))  # synth
from shaula.extract_feats import extract_features_from_arrays

from experiments.mes_accuracy import synth

PLANETS = [(4.2, 400e-6, 0.11), (13.7, 250e-6, 0.17), (37.9, 600e-6, 0.24)]
cfg = json.loads(sys.argv[2])
seed = cfg.get("seed", 5)
time = synth.kepler_time(n_quarters=8, seed=seed)
rng = np.random.default_rng(seed)
model = np.zeros_like(time)
for i, (p, d, dur) in enumerate(PLANETS):
    model += synth.transit_model(time, p, time.min() + (0.21 + 0.3 * i) * p, d, dur)
var = synth.stellar_variability(time, cfg.get("amp_rot", 0), cfg.get("prot", 8.0), 0, 0.2, seed)
flux = (1 + var) * (1 - model) + rng.normal(0, 150e-6, time.size)
with contextlib.redirect_stdout(io.StringIO()):
    rows = extract_features_from_arrays(time, flux)
out = []
for r in rows:
    match = [p for p, _, _ in PLANETS if abs(r["period_days"] / p - 1) < 2e-3]
    out.append(dict(P=round(r["period_days"], 4), mes=round(r["max_mes"], 1),
                    dur=round(r["duration_days"], 3), status=r["detection_status"],
                    planet=match[0] if match else None))
print(json.dumps(dict(cfg=cfg, rows=out)))
