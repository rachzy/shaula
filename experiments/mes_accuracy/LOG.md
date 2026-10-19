# MES accuracy experiments

Goal: bring Shaula's `max_mes` (and the features that depend on the same noise
model) closer to the Kepler KOI values (`koi_max_mult_ev`), then calibrate the
detection gate. Running log; newest entries at the bottom. A restarted session
resumes from "Next".

## Setup

- `synth.py`: Kepler-like long-cadence light curves (quarters, monthly gaps,
  2% dropped cadences), limb-darkened small-planet transit integrated over the
  exposure, optional spot modulation and AR(1) granulation, optional TTVs.
- Reference ("oracle") MES: white-noise box filter centred on each true
  transit, maximised over the TPS duration grid (1.5-15 h). This is what TPS
  would report on white noise with perfect detrending.
- `run_synth.py <grid>`: by default measures only the injected ephemeris via
  the `candidate_hint` path of `_extract_single_candidate_from_arrays`
  (~5 s per 8-quarter star). `--full` runs the whole iterative search
  (~7 min per 8-quarter star, so too slow for grids).
- `local_trend.py`: prototype detrenders patched in from outside; `shaula/`
  is untouched until one is chosen.

## 2026-10-19

### Network
MAST, archive.stsci.edu, exoplanetarchive and SIMBAD all return a proxy 403 in
this container. Real-data comparison is blocked; all work below is synthetic.

### E1. White noise: filter and fold are fine
8 quarters, sigma = 400 ppm/cadence, P = 3.7 d, depths 300-5000 ppm.
`max_mes / oracle` = 1.007, 1.04, 1.007 (fast path) and 1.03 with the full
search. **The matched filter, duration refit and fold lose nothing on white
noise.** So the ~25% deficit vs KOI is not the box shape or T14.

### E2. Stellar variability: the final detrend does almost nothing
Same signals with sigma = 100 ppm plus either spot modulation (2000 ppm,
Prot 12 d; 500 ppm, Prot 3 d) or granulation (80 ppm, tau 0.2 d).
`results/variable_baseline.csv`: median `max_mes / oracle` = **0.14**; spots
give 0.0-0.2 and `cdpp_6h` of 290-700 ppm against an ideal ~29 ppm.

Cause (from the code): the final trend is `UnivariateSpline(..., s=0.001)`
on 0.5 d bins of normalised flux. `s` is an absolute sum of squares, so it
lets the spline miss each bin by ~sqrt(0.001 / n_bins), about 600 ppm rms
for 8 quarters and more for fewer. Any variability below that is left in the
"detrended" flux, and the noise model's local median (a +-15 T14 window)
only partly absorbs it. The pre-BLS running median removes variability for
the search only. Bright, active or phase-curve stars (Kepler-13, HAT-P-7 =
Kepler-2, Kepler-7) are where KOI MES deficits are largest, which fits
(inferred; needs real data).

### E3. Local detrend after the spline: rejected (biased)
`local_trend.py` divides the detrended flux by a biweight/median trend of
out-of-transit cadences in a k x T14 window. Variable stars recover to ~1.0
of ideal, but white-noise MES rises to **1.1-1.2 of ideal**: the trend
absorbs part of every out-of-transit box (so the noise looks smaller) while
the transit itself is withheld (so the signal stays whole). That inflates
MES on noise too, i.e. more false alarms. Kept only as a record.

| config (`results/`) | white | variable |
|---|---|---|
| baseline | 1.01 | 0.14 |
| biweight 3 T14 | 1.22 | 1.00 |
| median 3 T14 | 1.15 | 1.01 |
| biweight 5 T14 | 1.09 | 0.77 |

### E4. Flank baseline in the noise model: adopted, with adaptive width
`cdpp.duration_matched_statistics(flank_durations=w)`: each box's depth is
measured against the mean of the flux in `w` box widths on either side
(left/right means averaged, so a linear trend cancels; known transits
withheld from the flanks). Every box, in or out of transit, gets the same
filter, so there is no signal/noise asymmetry.

| flanks (box widths) | white | variable |
|---|---|---|
| 1 | 0.78 | 0.65 |
| 2 | 0.88 | 0.59 |
| 3 | 0.91 | 0.47 |
| 5 | 0.94 | 0.39 |
| `auto` (lowest noise of 1,2,3,5,none) | **1.01** | **0.65** |

White-noise losses match the theory sqrt(1 + 1/(2w)). `auto` keeps the
option with the lowest median noise among noise-eligible boxes; transits are
never looked at, and every option preserves the full depth, so lowest noise
= highest expected MES with no look-elsewhere bias. What is left on the
variable grid is mostly granulation (red noise at the transit timescale,
which no high-pass removes; the white-only oracle overstates what is
reachable there) and the 3 d spot curvature.

Null test (`null_test.py`, `results/null_test.txt`): MES at 300 random
ephemerides on transit-free curves. `auto` gives mean -0.4..+0.1 and std
0.96-1.14 on white, spotted and granulated stars. The current model gives
**mean -6.1, std 4.1** on a 500 ppm / 3 d spotted star: it is not a
calibrated statistic there at all.

Cost: ~2x per-candidate time (the noise model is built 5 times).

`"auto"` is now the default in `shaula/cdpp.py`, so CDPP and SES/MES both
use it. Tests: 4 new ones in `FlankBaselineTests`; full suite passes.

### Not pursued: TTV-tolerant fold
`koi_max_mult_ev` is TPS's MES, and TPS folds on a strict linear ephemeris
too. Searching per-event offsets would raise Shaula's MES above the
quantity it is compared with (and raise noise MES by look-elsewhere), so it
would not improve agreement with the literature. The large deficits for
Kepler-89/51/138 are more likely the same variability problem or a period
mismatch; check with real data.

## Next
- Optimise `auto` (choose the width once per light curve or per candidate
  instead of rebuilding five times on every call).
- Full-search run on a few synthetic stars to see whether `auto` changes
  what the iterative search finds (false alarms from leftover variability
  should drop).
- When the network opens: rerun the 49-star comparison with `auto` and
  compare `max_mes` to `koi_max_mult_ev`; then calibrate the 7.1 gate.
- When the network opens: rerun the 49-star comparison with the chosen
  detrend and compare `max_mes` to `koi_max_mult_ev`.
