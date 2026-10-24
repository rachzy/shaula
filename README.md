# LTP - Features

## Description

**Lightcurve Transit Photometry (LTP) - Features** is a python project for extracting features from light curve data useful for detecting exoplanets through the [Transit Photometry](https://en.wikipedia.org/wiki/Methods_of_detecting_exoplanets#Transit_photometry) method.

At first, it was intended to be pre-processing pipeline for digesting features to be consumed by ML model, but keep in mind that ,this is **NOT** a ML model, although it can be used as a pipeline for features digestion.

### What is Transit Photometry

When a planet passes in front of its host star along our line of sight, it blocks a small fraction of the star’s light. Photometers record a periodic dip in brightness whose depth scales roughly with the squared ratio of planet radius to stellar radius, and whose spacing in time reveals the orbital period. Surveys such as Kepler and TESS monitor many stars for these repeating dimmings; candidates from such data still need careful vetting (stellar activity, eclipsing binaries, instrumental effects) before they can be treated as planet detections.

<img width="487" height="271" alt="image" src="https://github.com/user-attachments/assets/0554c00d-14d5-44e3-9967-f115ecfa5ebc" />
<figcaption>
   Transit Photometry Example - <a href="https://www.apus.edu/academic-community/space-studies/exoplanet-transit-photometry/" target="_blank">Apus - Exoplanet Transit Photometry</a>
</figcaption>

### What are light curves

A **light curve** is the brightness of a star (or other source) measured over time—typically flux in detector units or normalized flux versus time in days. Space-based transit missions produce long, evenly or nearly evenly sampled series per target, often with millions of points. This repository treats those time and flux arrays as the raw input: cleaning, detrending, period search (e.g. BLS/TLS), folding at the trial period, and numerical summaries that describe shape, noise, and consistency of the signal.

<img width="288" height="180" alt="image" src="https://github.com/user-attachments/assets/e67310c7-315c-41ae-8c86-334a47d6caac" />
<figcaption>
   Lightcurve Example - <a href="https://imagine.gsfc.nasa.gov/features/yba/M31_velocity/lightcurve/lightcurve_more.html" target="_blank">Nasa - Imagine the Universe!</a>
</figcaption>

## Pipeline and Extraction

As explained above, light curves are basically (but not just) gigantic CSVs containing flux data of brightness variation of a star.

In order to know what to look for and what to calculate in this data, we use the following resources:

- [NASA Exoplanet Archive for the Kepler Mission](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [Kepler Science Data Processing Pipeline](https://github.com/nasa/kepler-pipeline)

#### Column meanings

| Column                             | Meaning                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `period_days`                      | Orbital period from BLS/TLS vs catalog period                                                                    |
| `t0`                               | Transit epoch phase error after mission zero-point conversion and modulo-period alignment, reported as % of transit duration |
| `duration_days` / `duration_hours` | Transit duration in days and hours                                                                               |
| `depth_mean_per_transit`           | Mean per-transit depth vs catalog depth                                                                          |
| `planet_radius_rjup`               | Inferred planet radius (Jupiter radii); `n/a` when stellar radius was unavailable                                |
| `mean_match_no_t0`                 | Mean of `max(0, 100 − \|%diff\|)` over all columns except `t0` (100% = perfect match, 0% = ≥100% off on average) |
| `max_ses`                          | Strongest single-transit detection statistic (SES) of any one event; compared with the KOI `koi_max_sngle_ev`     |
| `max_mes`                          | Multiple-event statistic (MES) at the best nearby phase; the detection gate requires ≥ 7.1. Compared with `koi_max_mult_ev` |
| `MES`                              | Multiple-event statistic at the exact fitted ephemeris (no phase scan)                                            |
| `SES_mean` / `SES_std`             | Mean and spread of the per-transit SES across all observed events                                                 |
| `depth_std_per_transit`            | Scatter of the per-transit depths; large values suggest variable depth or a misfit period                        |
| `depth_stability`                  | Std/mean of per-epoch depths from the folded light curve (0 = every transit equally deep)                        |
| `npts_transit_median`              | Median number of cadences inside each transit                                                                    |
| `cadence_hours`                    | Median time between cadences (~0.49 h for Kepler long cadence)                                                   |
| `cdpp_3h` / `cdpp_6h` / `cdpp_12h` | Noise (ppm) over 3, 6 and 12 hour windows, transits excluded; a time-domain stand-in for Kepler's CDPP           |
| `snr_global`                       | Mean transit depth divided by the noise (CDPP) at the transit duration                                           |
| `local_noise`                      | Robust (MAD-based) scatter of the out-of-transit flux                                                            |
| `resid_rms_global`                 | Standard deviation of the whole detrended light curve                                                            |
| `acf_lag_1h` … `acf_lag_24h`       | Autocorrelation of the flux at 1, 3, 6, 12 and 24 hour lags; high values mean correlated (red) noise             |
| `scale_mean` / `scale_std`         | Mean and spread of the robust-scaled flux (median 0, MAD-based unit)                                             |
| `scale_skewness` / `scale_kurtosis`| Shape of the scaled flux distribution; transits and eclipses push skewness negative                              |
| `scale_outlier_resistance`         | % of cadences more than 5σ from the median                                                                       |
| `vshape_metric`                    | Transit shape from a trapezoid fit: 0 = flat-bottomed box, 1 = pure V (grazing transit or eclipsing binary)      |
| `secondary_depth`                  | Signed depth at phase 0.5 (halfway between transits); ~0 when there is no secondary eclipse                      |
| `secondary_depth_snr`              | Signed significance of that secondary, on the same scale as `MES`; high values suggest an eclipsing binary        |
| `odd_even_depth_ratio`             | Mean odd-transit depth / mean even-transit depth; far from 1 suggests an eclipsing binary at twice the period     |
| `ingress_egress_asymmetry`         | Difference between ingress and egress slopes of the folded transit; ~0 for a symmetric transit                   |
| `planet_radius_rearth`             | Inferred planet radius (Earth radii); `n/a` when stellar radius was unavailable                                  |
| `detection_status`                 | `accepted` (passed the MES gate) or `provisional` (sub-threshold, MES ≥ 6); scripts can also write `rejected` or `harmonic_duplicate` |
| `candidate_label`                  | Training label: `CONFIRMED`, `FALSE-POSITIVE` or `UNKNOWN`, from a period match against reference KOIs (scripts only) |
| `matched_target` / `matched_period_ratio` | Reference planet the row was matched to, and how its period relates to the reference (`direct`, or an alias such as `2P`, `P/2`) |
| `fp_source`                        | Why a row is `FALSE-POSITIVE`: the matched KOI's flags (e.g. `koi:nt`, `koi:ss+co`), `unmatched`, or `inverted`  |

**What this suggests:** for quiet, deep hot-Jupiter hosts (Kepler-5b, HAT-P-7, Kepler-7b) the pipeline recovers period and duration to within a few percent and scores above ~96%. Their epochs also agree closely after converting BJD to BKJD and aligning transit numbers. Shallower or harder targets (aliases, grazing geometry, long-period / multi-planet systems such as Kepler-12b, Kepler-186f, Kepler-447) diverge more on depth, radius, and sometimes period—useful stress cases rather than failures of the comparison script itself.

## Using the pipeline to extract lightcurve data

The full pipeline flow and how to use it is explained below.

### 1. Choose a target

According to lightkurve's official documentation, a target can be any one of the following

> - The name of the object as a string, e.g. “Kepler-10”.
> - The KIC or EPIC identifier as an integer, e.g. 11904151.
> - A coordinate string in decimal format, e.g. “285.67942179 +50.24130576”.
> - A coordinate string in sexagesimal format, e.g. “19:02:43.1 +50:14:28.7”.
> - An astropy.coordinates.SkyCoord object.

A specific mission can also be specified. We use "Kepler" by default.

### 2. Download and clean light curve data

After choosing the target, we first have to download its curve data.
Let's use its host star, **Kepler-5**, as the example in this case:

```python
lc = lk.search_lightcurve(
    "Kepler-5", mission="Kepler", author="Kepler", exptime="long"
)
lc = lc.download_all() # Download all available data for this target (recommended for more precise data)
lc = lc.stitch() # Stitch all downloaded curves into a single one
```

The repository's downloader applies that product filtering automatically before
stitching. Kepler and K2 default to their official long-cadence products; TESS
defaults to SPOC short-cadence products. Use the CLI's `--author` and
`--exptime` options when a different, but still internally consistent, product
set is needed.

Now that we have the light curve, we can use some lightkurve native functions to do some data cleaning:

```python
lc = lc.remove_nans().normalize().remove_outliers(sigma=5.0)
```

With that, we're ready to throw it into our pipeline!

### 3. Pass the light curve into feature extraction

`extract_features_from_lightcurve` in `shaula/extract_feats.py` reads time and flux from the Lightkurve object and delegates to the same path as CSV input. Every extraction entry point returns a list with one feature dictionary per MES-qualified transit candidate, sorted by period.

```python
time = lc.time.value
flux = lc.flux.value
candidate_rows = extract_features_from_arrays(time, flux, ...)
```

If `RADIUS` is present in `lc.meta`, stellar radius is used to fill planet-radius features; otherwise those fields stay empty.

Candidate discovery is currently an experimental iterative masking search. On
each iteration it computes a BLS periodogram, clusters nearby local maxima,
keeps representatives across the searched period range, and evaluates them in
descending BLS power. Rejecting a candidate rejects only that period, epoch,
and duration neighborhood, so the next independent peak can still be tested.
An accepted candidate must have `max_mes >= 7.1` and at least three observed
events. Its predicted transit windows are then masked at 1.5 times the fitted
duration before the next periodogram is calculated.

Masking the windows a candidate predicts still leaves any deep dip it does not
explain in the data, and BLS will thread a box train through a handful of those
and report a high MES for a period no planet has. After each acceptance a
deep-event sweep removes dips at or beyond 8σ that either sit on an accepted
candidate's predicted transit (its own transits escaping an epoch-drifted mask)
or form a group of similar-depth events too small to ever clear the
three-event rule. Groups large enough to be detectable are left in place: those
are real signals a later iteration can still find.

Before an accepted candidate is masked, its harmonics are reconciled. A search
that locks onto `P/2` reports a period wrong by a whole factor and — far worse
here — masks the wrong windows: a `P/2` alias predicts a window over *every*
real transit, so the planet can never be recovered afterwards. MES is already
the right discriminator, because folding a true-period signal at `2P` drops
half its events while folding at `P/2` pads it with empty windows; either costs
a factor √2 in `ΣN/√ΣD`, so MES peaks at the true period. Each accepted
candidate is therefore re-folded at `2P`, `3P`, `P/2` and `P/3` — reusing the
same duration-matched statistics object, which carries no period or epoch of
its own, so no noise model is rebuilt — and the best harmonic is adopted if it
clears the incumbent by 15%. Only integer ratios are tried: Kepler-90d and e
sit 2.6% from a 3:2 ratio, so half-integer trials would pit real planets
against each other. Period de-duplication treats 2× and 3× ratios as the same
signal for the same reason, at a tolerance tight enough to leave near-resonant
pairs alone (Kepler-90b and i are only 3.1% from 2:1).

When the search stops because no remaining peak clears 7.1 — as opposed to
running out of iterations or masking budget — the peaks it already measured and
rejected are reconsidered at a lower bar of 6.0. Up to two are reported and
marked with `is_provisional_detection = 1.0` and the `mes_threshold_used` they
were judged against. Nothing is masked afterwards, which is what makes the
looser bar safe: an accepted candidate consumes an iteration slot and removes
cadences from everything found after it, while a recovered one costs a row.
This is how signals near the noise floor surface — Kepler-90i (87 ppm, ~101
transits of which only 68 are observable on a light curve that lost four
quarters to the Module 3 failure) measures around 6.2 and is genuinely below
the detection gate rather than missed by the search.

The 7.1 threshold is a first experimental stopping rule, not a calibrated
Kepler TCE decision boundary: this pipeline's time-domain MES is not
numerically equivalent to Kepler TPS. The search masks cadences rather than
subtracting a fitted transit model, stops after eight accepted candidates or
50% cumulative masking, and does not yet perform a final joint fit or common
re-detrending pass.

### 4. Detrending and period search

`detrend_with_bls_mask` in `shaula/detrend_and_period.py` runs first: Box Least Squares (BLS), optional TLS refinement, iterative detrending, and transit masking. After detrending, it repeats a fixed-period/fixed-duration BLS phase fit and canonicalizes `t0` to the first predicted transit in the data interval. That refit is bounded to half a transit duration: BLS reports the best phase anywhere in the fold, so a shallow candidate measured while a deeper signal is still unmasked would otherwise be relocated onto it — measured on Kepler-90i, the unconstrained refit landed 55 durations from the truth, reading Kepler-90g and h instead. When the fit lands outside the window the phase is re-picked inside it. It returns detrended flux plus `best_period`, `t0`, and transit duration used everywhere below.

The period search runs from 0.5 days up to a third of the observing baseline, which is the longest period that can still show the three transits a detection requires. There is no additional flat ceiling: over Kepler's ~1460 day baseline a 200 day one made every longer-period planet unsearchable by construction (Kepler-90g at 210.6 d and h at 331.6 d, for instance), and their unmodeled transits then fed the spurious candidates the deep-event sweep above now removes.

```python
flux_detr, trend, mask_transit, bls_info = detrend_with_bls_mask(
    time_arr, flux_arr, refine_duration=True, use_tls=True
)
```

### 5. Scaling metrics

`scaling_and_metrics` in `shaula/utils/scaling_and_metrics.py` standardizes the detrended flux and records summary statistics (mean, standard deviation, skewness, kurtosis, outlier resistance) into the feature dict.

### 6. Folded and binned metrics

`folded_binned_metrics` in `shaula/folded_binned_metrics.py` folds the series in phase at the BLS period and `t0`, builds a median phase profile, estimates a transit width in phase, then computes:

- **Cadence** from median short time steps (fed into CDPP later).
- **`local_noise`**: robust scatter (MAD) using out-of-transit points.
- **`depth_stability`**: how much per-epoch transit depths vary relative to the global folded depth.
- **`acf_lags`**: flux autocorrelation at configured hour lags (e.g. 1–24 h).

```python
binned = folded_binned_metrics(
    time_arr, flux_detr_full, period, t0, lags_hours=(1, 3, 6, 12, 24)
)
```

### 7. Per-transit statistics

`per_transit_stats_simple` in `shaula/per_trans_stat.py` walks each transit epoch, estimates a baseline outside the transit window, and collects per-transit depths and the number of samples actually inside each transit. Those median-based measurements remain inputs to the depth and shape features; SES/MES now use the duration-matched time series described below.

**Execution order vs. in-file labels:** Inside `extract_features_from_arrays`, this block runs _before_ CDPP even though `per_trans_stat.py` is tagged `# 6` and `cdpp.py` is `# 4`. Treat the `# N` lines in source files as module tags, not strict pipeline ordering.

### 8. CDPP

`calculate_cdpp` in `shaula/cdpp.py` builds box-template depth series lasting 3 h, 6 h, and 12 h. It estimates a robust, time-dependent uncertainty in a local window spanning 30 template durations, retains correlated noise measured at the transit timescale, deweights partial coverage, and prevents the BLS transit windows from contaminating the noise model. Each CDPP feature is the median valid local uncertainty in ppm. These values are a time-domain duration-matched approximation, not an exact reproduction of Kepler's adaptive wavelet whitening.

```python
cdpp = calculate_cdpp(
    flux_detr_full,
    cadence_hours=feats["cadence_hours"],
    time=time_arr,
    exclude_mask=mask_transit,
)
```

### 9. SES, MES, and remaining shape / vetting features

`compute_SES_MES` in `shaula/sesmes.py` evaluates the exact BLS duration at every eligible cadence. For each measurement it constructs a local depth uncertainty and the coherent matched-filter components `N = depth / uncertainty²`, `D = 1 / uncertainty²`, and `SES = N / sqrt(D)`. `max_ses` is the largest positive cadence-level SES. `MES` folds `N` and `D` at the final BLS ephemeris, while `max_mes` searches only a nearby phase window at that fixed period and duration. `SES_mean` and related per-transit features summarize the exact-ephemeris event SES values. These are scientifically coherent time-domain approximations, not numerical reproductions of Kepler TPS wavelet statistics or a full MES period search.

This change replaces the previous root-sum-square `MES` and global-CDPP SES semantics without changing the saved column names. Existing extracted CSVs and trained artifacts should be regenerated before they are compared or combined with new output.

The same extraction pass then adds folded **v-shape** metrics, **secondary eclipse** depth (and CDPP-based SNR variants), **odd/even depth ratio**, **ingress/egress asymmetry**, global residual RMS, and **skewness/kurtosis** on scaled flux, all still anchored to the BLS period, epoch, and duration from step 4.

## How to run the project

1. **Python environment** — Install the project and its dependencies with [uv](https://docs.astral.sh/uv/) (this creates `.venv` and installs `shaula` in editable mode):

   ```bash
   uv sync
   ```

### CLI

Run the installed console script:

```bash
uv run shaula-extract-lk --target HAT-P-7 --mission Kepler --out-features out/hatp7_features.csv
```

Use `--input-lightkurve path/to.csv` instead of `--target` if you already have a saved light curve file. Optional flags include `--mission` (e.g. `TESS`), `--author`, `--exptime`, `--sigma-clip`, `--download-all`, `--out-lightkurve` to write the downloaded/cleaned curve, and `--quiet`.

### Notebooks

Exploratory workflows live under `shaula/` (for example `lightcurve_analysis.ipynb`). The script `shaula/cli/extract_csv.py` is marked deprecated but may still reflect the batch CSV feature layout.

You need network access when downloading data through Lightkurve; first-time use may also pull mission-specific calibration dependencies.

## Credits

The prototype code used in this repository was originally meant for the Gatonautas team project for [Nasa Space Apps Challenge 2025](https://www.nasa.gov/nasa-space-apps-challenge-2025/)
