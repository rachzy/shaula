# MES accuracy experiments

Goal: bring Shaula's `max_mes` (and the features that depend on the same noise
model) closer to the Kepler KOI values (`koi_max_mult_ev`), then calibrate the
detection gate. Running log; newest entries at the bottom. A restarted session
resumes from "Next".

> `results/` paths below are local outputs of the scripts and are not committed (gitignored); the tables here hold the numbers.

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

### E5. Full iterative search on weak planets around spotted stars
`run_synth.py search --full`, 8 quarters, depth 300 ppm, sigma 100 ppm,
one planet each (`results/search_*.csv`).

| P (d) | spots | current: found / max_mes | auto: found / max_mes | oracle |
|---|---|---|---|---|
| 3.7 | 2000 ppm, 12 d | yes / 9.1 | yes / 59.6 | 84.3 |
| 3.7 | 500 ppm, 3 d | **missed** | yes / 56.8 | 84.3 |
| 11.3 | 2000 ppm, 12 d | yes / 15.9 | yes / 24.7 | 59.3 |
| 11.3 | 500 ppm, 3 d | **missed** | yes / 27.2 | 59.3 |

No spurious rows in either mode. The 11.3 d planets come out lower in the
full search (0.42-0.46 of ideal) than on the hint path (0.6-0.8); not yet
explained (duration or epoch from the search, or the spline).

### E6. Cheap width selection
`auto` first ranks the options by the pooled MAD of their noise anchors
(no local model), then builds the full model once. Same MES on every grid
case as building all five, at baseline runtime (7.2 s vs 8.3 s white,
6.5 s vs 5.7 s variable per star). Null test unchanged.

### E7. Duration audit by MES
Tracing the weak 11.3 d case: BLS found T14 = 0.169 d (true 0.18), then
`_folded_profile_duration` read a one-bin-wide (0.047 d, 240 phase bins)
dip off the un-flattened, spotted flux and capped T14 to 0.064 d; the
trapezoid refit, on the same flux, then gave 0.041 d. Every downstream
feature used that.

Fix (`extract_feats._mes_optimal_duration`): after detrending, score T14 x
{0.5, 0.71, 1.41, 2, 2.83, 4} by `max_mes` with the calibrated noise model
and adopt the best if it beats the refit T14 by 10%. TPS likewise maximises
MES over its duration grid.

| | before | after |
|---|---|---|
| search case 2 (11.3 d, 12 d spots) | 0.42 of ideal, T14 x0.23 | **0.80**, T14 x0.91 |
| search case 3 (11.3 d, 3 d spots) | 0.46, T14 x1.35 | **0.66**, T14 x0.96 |
| variable grid (hint path) median | 0.647 | **0.711** |
| white grid median | 1.009 | 1.017 |

One white case (300 ppm, P 3.7 d) moved to T14 x0.72 and MES 1.15 of the
noiseless oracle: picking the best of 7 durations adds some upward noise
bias at low MES, as it does in TPS. Worth remembering when calibrating the
gate.

### E8. Vectorised local noise windows
`_local_anchor_noise` computed a median and MAD per anchor in a Python loop
(~75% of noise-model time). Now one NaN-padded 2-D `nanmedian` per segment:
bit-identical SES on synthetic data, 5x faster (0.43 s -> 0.08 s per call on
8 quarters). With it, the audit costs nothing net: hint-path runtimes are
7.6 s white / 6.9 s variable vs 8.3 / 5.7 s originally.

### E9. Refit folds on the variability-removed flux: not adopted
Folding the pre-BLS-flattened flux (1 d binned median) in the profile and
trapezoid refits: white 1.017 -> 1.042, variable 0.711 -> 0.700, search
cases 0.80/0.66 -> 0.84/0.65, and T14 ratios scatter more (0.75-1.13).
No clear gain on top of the audit, so reverted (`results/*_flatfold*`).

### E10. Regression check: 3-planet system, main vs branch
`multi_planet.py <repo-root> '<cfg-json>'` (run with the branch venv;
`shaula` is imported from `<repo-root>`, e.g. a worktree of main). Planets
at 4.2 / 13.7 / 37.9 d, 400 / 250 / 600 ppm, sigma 150 ppm, 8 quarters.
Oracle MES 67.4 / 29.1 / 47.6. `results/multi_planet.txt`.

| star | main | branch |
|---|---|---|
| quiet | 3/3 found, MES 62.4 / 25.8 / 53.3, plus a spurious provisional at 2.85 d | 3/3, MES **69.3 / 28.8 / 48.7**, plus a provisional at 89.8 d (MES 6.1) |
| 1000 ppm spots, 8 d | **1/3** (MES 22.9 for the 4.2 d planet) | **3/3**, MES 60.5 / 21.9 / 38.0 |

No regression on the quiet star; the spotted star goes from one planet to
all three.

## 2026-10-19 (afternoon): real stars

Network opened at 12:47 UTC. MAST's name resolver (`mastresolver.stsci.edu`,
plain HTTP) is still unreachable, so `real_stars.py` resolves hosts to
`KIC n` through the Exoplanet Archive TAP first.

### E11. MES at the catalogue ephemeris, 56 hosts / 127 planets
`real_stars.py download` (all long-cadence quarters, cleaned as `extract()`
does; ~35 s per star) then `measure --mode M`: hint path on the KOI
ephemeris, other planets gapped at 1.5 x T14. Error = extracted / catalogue
- 1 (`koi_max_mult_ev`, `koi_duration`, `koi_depth`); 123 planets with a
finite value in every mode. `results/real_<mode>.csv`.

| mode | MES median | MES median abs | MES within 20% | MES 90th pct abs | T14 median / abs | depth median / abs |
|---|---|---|---|---|---|---|
| main | **-27.1%** | 31.4% | 35% | 74% | +7.7% / 10.2% | -7.4% / 8.2% |
| flank only (no audit) | +0.6% | 9.6% | 80% | 30% | +7.7% / 10.2% | -7.4% / 8.2% |
| flank + audit 0.5-4x (gross) | **+1.1%** | **9.1%** | 80% | 30% | +6.8% / 10.3% | -6.9% / 8.0% |
| flank + audit incl. 0.71/1.41 | +3.3% | 9.4% | 82% | 29% | +4.7% / 12.2% | -8.2% / 9.4% |

By catalogue MES (main -> full branch): <=10: -11% -> +13%; 10-15: -18% ->
+17%; 15-30: -19% -> +6%; 30-100: -21% -> +6%; >100: **-48% -> -8%**.
Kepler-7b: 1163 -> 2088 (catalogue 2252); Kepler-13b 2216 -> 4332 (6791);
Kepler-2b 2262 -> 3161 (3862).

The flank baseline does nearly all of it. The 0.71/1.41 audit steps pick a
box shorter than T14 on strong limb-darkened transits (that box has the
higher MES), which costs T14 and depth accuracy for no real MES gain, so
`DURATION_AUDIT_FACTORS` is now `(0.5, 2, 2.83, 4)`: it only fixes gross
misfits like the synthetic x4 case.

Still far off (branch, hint path): Kepler-411d, 51c, 102b/f (MES 4-8 vs
21-47), where T14 is also off by 1.4-20x, so likely contamination from
neighbouring planets or TTVs rather than the noise model. A few come out
far *above* the catalogue (Kepler-10b 181 vs 30, 42c 74 vs 10, 90f 44 vs
18); the catalogue values there look like TPS reporting a different event
set (to check).

### E12. Full search, branch v1 (flank baseline + audit) vs main
`real_search.py` on downloaded hosts, graded by period (0.5%, aliases 2x,
3x, 1/2, 1/3). `results/search_real_main.csv`, `search_real_branch_v1.csv`.
v1 on 6 hosts: Kepler-102 2/5 (main 1/5), 37 2/4 (1/4), 444 3/5 (1/5), 13,
25 unchanged, 411 0/3 (0/3). More planets, more unmatched rows (102: 9,
444: 7), and much slower (102: 3987 s vs 618 s; 444: 4764 s vs 898 s),
because the higher MES lets more candidates through the gate and each
accepted candidate costs another iteration.

### E13. Kepler-411: fine-window BLS pass and MES-ordered queue
Kepler-411's day-scale variability (~3700 ppm) is deeper than its transits,
so the 1 d+ pre-BLS running median leaves residual bumps that BLS ranks
above the planets. A global 0.75 d window recovered 411c (that test, not the
fine pass alone, is what commit 4b7f5ea's message describes; the message is
wrong). Adopted instead: a second BLS pass on 0.75 d flattened flux with
durations <= window/3, interleaved into the peak queue
(`_fine_window_candidates`), and the queue tried in order of quick MES, not
BLS power, so a strong true peak is accepted before junk that merely has
high BLS power.

### E14. Full search, branch v3 (E11 + E13) vs main
10 of 12 hosts finished before the 2 h job limit (Kepler-51 and 138 rerun
separately). `results/search_real_v3.csv`, `search_real_v3_rows.csv`.

| star | main found | v3 found | main extra | v3 extra | main s | v3 s |
|---|---|---|---|---|---|---|
| Kepler-102 | 1/5 | 2/5 | 0 | 4 | 618 | 1880 |
| Kepler-411 | 0/3 | **2/3** | 0 | 1 | 430 | 1005 |
| Kepler-37 | 1/4 | 2/4 | 1 | 10 | 1116 | 4509 |
| Kepler-7 | 1/1 | 1/1 | 0 | 2 | 398 | 1120 |
| Kepler-90 | 7/8 | **8/8** | 1 | 2 | 2005 | 3423 |
| Kepler-444 | 1/5 | 3/5 | 0 | 5 | 898 | 4073 |
| Kepler-25 | 2/2 | 2/2 | 0 | 2 | 800 | 1126 |
| Kepler-13 | 1/1 | 1/1 | 1 | 2 | 633 | 1681 |
| Kepler-89 | 4/4 | 4/4 | 1 | 2 | 1858 | 2822 |
| Kepler-62 | 3/5 | **5/5** | 0 | 3 | 984 | 2842 |
| **total** | **21/38** | **30/38** | 4 | 33 | 9740 | 24481 |

Recall 55% -> 79%; extra rows 4 -> 33 (acceptable per Henrique: recall over
precision); runtime 2.5x. Still missed: Kepler-102b/c/f (catalogue MES
14-21), 37b/e (no catalogue MES; 37b is ~Moon-sized), 444c/f, 411d.

Kepler-51 2/3 (main 2/3) and Kepler-138 2/3 (main 1/3) finished later;
over all 12 hosts v3 finds 34/44 planets (main 24/44), extra rows 6 -> 42.

### E15. Were the misses catchable? (`miss_diagnostic.py`, `queue_probe.py`)
Fixed-ephemeris MES (masked running-median flatten, catalogue P/t0/T14, no
refinement): 102b 15.9, 102c 19.8, 102f 21.2, 444c 34.3, 444f 36.6, 411d
25.6, 51c 24.9, 37b 9.7, 37e 3.8. So every miss except 37e (out of reach)
and 37b (12 ppm, marginal) had its signal well above the gate in our own
statistic: the search lost them. Causes found by printing one iteration's
queue with the found planets gapped:
- **Period grid.** The global BLS used 200 linear durations, and the
  compute budget (periods x points) left ~3,000 periods where a 1.7 h
  transit over 1,400 d needs ~100,000. 444c/f formed no peak in the queue at
  all (not even with 3 peaks per log band). BLS cost is ~ periods x
  durations; 24 log-spaced durations (~6% max box mismatch) buy an ~8x
  denser grid at equal cost, and both enter the queue with quick MES 19/36.
- **Refinement on active stars.** Kepler-411d's 2P (116 d) is in the queue;
  with the refinement budget raised to 20 it is refined, but the BLS-power
  zoom and de-aliaser move 58.03 d to 57.3 d and then to P/11 = 5.2 d (the
  star's rotation harmonic), MES 2.6. Needs MES-based refinement: not a
  minor change, left open.
- Kepler-51c (85 d, grazing, young active star): not in the queue either;
  likely the same grid problem at long period, not fixed by E16.

### E16. Full search, v4 = v3 + 24 log-spaced global durations
`GLOBAL_BLS_DURATIONS = 24` (adopted). `results/search_real_v4.csv`. Six
hosts ran on a separate 8-core machine (times not comparable).

| star | main | v3 | v4 | extra main/v3/v4 |
|---|---|---|---|---|
| Kepler-102 | 1/5 | 2/5 | **3/5** | 0/4/2 |
| Kepler-138 | 1/3 | 2/3 | **3/3** (b at 3P, provisional) | 2/8/5 |
| Kepler-444 | 1/5 | 3/5 | 3/5 (c instead of f) | 0/5/4 |
| Kepler-37 | 1/4 | 2/4 | 2/4 | 1/10/4 |
| Kepler-62 | 3/5 | 5/5 | 5/5 | 0/3/0 |
| Kepler-90 | 7/8 | 8/8 | 8/8 | 1/2/3 |
| Kepler-411 | 0/3 | 2/3 | 2/3 | 0/1/0 |
| Kepler-51 | 2/3 | 2/3 | 2/3 | 0/1/1 |
| others (7, 13, 25, 89) | 8/8 | 8/8 | 8/8 | 2/8/6 |
| **total** | **24/44** | **34/44** | **36/44** | **6/42/25** |

Still missed: 102b/f, 444b/f, 411d, 51c (catchable; see E15), 37b/e (not).
On 444, junk long-period candidates (168-330 d, T14 12-20 h) are accepted
between real planets and mask data; b and f were lost that way in v4.

## Next
- Junk long-duration, long-period acceptances (444, 7, 138): they cost
  masked data and iterations. Candidate fix without a recall veto: try them
  after short-duration peaks of similar MES, or mask them less widely.
- MES-based queue refinement on active stars (411d).
- Calibrate the 7.1 gate against `koi_max_mult_ev` now that MES is unbiased.
