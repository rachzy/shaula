# Shaula

**Shaula** is a Python library that takes the light curve of a star (its brightness measured over time) and searches it for the small, repeating dips that a planet causes when it passes in front of the star. For every dip pattern it finds, it returns one row of numbers ("features") that describe the signal: its period, depth and duration, how strong it is compared with the noise, and a set of checks that help tell real planets apart from look-alikes such as eclipsing binary stars.

Shaula is the pipeline used to build the training data for [Fang](https://github.com/rachzy/fang), and it currently runs inside the [Antares](https://github.com/rachzy/antares) project, which calls it through a gRPC service. Shaula itself is **not** a machine-learning model: it measures signals, and models such as Fang learn from those measurements.

The project is named after [Shaula](https://en.wikipedia.org/wiki/Lambda_Scorpii) (λ Scorpii), one of the stars of the constellation Scorpius, where it marks the tip of the scorpion's stinger. It was previously called `ltp-features` (Lightcurve Transit Photometry Features).

## Background

### What is transit photometry

When a planet passes in front of its host star along our line of sight, it blocks a small fraction of the star's light. A telescope watching the star records a dip in brightness that repeats every orbit. The depth of the dip tells us roughly how big the planet is compared with its star (depth ≈ (planet radius / star radius)²), and the time between dips is the orbital period. Missions such as Kepler and TESS watched hundreds of thousands of stars looking for these repeating dips.

A repeating dip is only a *candidate*. Spots on the star, two stars eclipsing each other, or instrument glitches can all produce dips, so candidates have to be vetted before they are called planets. Many of Shaula's features exist to help with that vetting.

<img width="487" height="271" alt="Diagram of a planet transiting its star and the resulting dip in the light curve" src="https://github.com/user-attachments/assets/0554c00d-14d5-44e3-9967-f115ecfa5ebc" />
<figcaption>
   Transit Photometry Example - <a href="https://www.apus.edu/academic-community/space-studies/exoplanet-transit-photometry/" target="_blank">Apus - Exoplanet Transit Photometry</a>
</figcaption>

### What are light curves

A **light curve** is a table of brightness (flux) against time. Kepler measured each of its stars about every 29.4 minutes for four years, which gives roughly 65,000 measurements per star; TESS measures every 2 minutes in about 27-day sectors. Each measurement is called a *cadence*. Shaula treats the time and flux columns as its raw input.

<img width="288" height="180" alt="Example light curve" src="https://github.com/user-attachments/assets/e67310c7-315c-41ae-8c86-334a47d6caac" />
<figcaption>
   Lightcurve Example - <a href="https://imagine.gsfc.nasa.gov/features/yba/M31_velocity/lightcurve/lightcurve_more.html" target="_blank">Nasa - Imagine the Universe!</a>
</figcaption>

### Reference material

The detection statistics are modelled on the official Kepler pipeline, and results are checked against the Kepler catalogue:

- [NASA Exoplanet Archive, Kepler cumulative KOI table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [Kepler Science Data Processing Pipeline](https://github.com/nasa/kepler-pipeline) and the [Kepler Data Processing Handbook](https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/kepler/_documents/KSCI-19081-003-KDPH.pdf)

## Getting started

### Install

Shaula needs Python 3.11 or newer. From a clone of this repository, [uv](https://docs.astral.sh/uv/) creates a virtual environment and installs the package in editable mode:

```bash
uv sync
```

To use it from another project, install it straight from GitHub:

```bash
pip install "shaula @ git+https://github.com/rachzy/shaula"
```

Two optional extras add features: `tls` (`transitleastsquares`, for an extra period refinement step) and `parallel` (`threadpoolctl`, used by the batch script). Install them with `uv sync --extra tls` or `pip install "shaula[tls]"`.

Downloading light curves needs network access to MAST (the archive that hosts Kepler and TESS data).

### Python API

`shaula.api` is the public interface. `resolve()` turns a name, catalogue number or sky position into one exact star, and `extract()` downloads that star's light curve and returns its candidates:

```python
from shaula.api import extract, resolve

star = resolve("Kepler-186")              # or resolve(ra_deg=..., dec_deg=...)
print(star.designation)                   # "KIC 8120608"

result = extract(star.designation, mission="Kepler", download_all=True)
for row in result.features:               # one dict per candidate, sorted by period
    print(row["period_days"], row["max_mes"], row["detection_status"])

print(result.star.teff_k, result.star.radius_rsun, result.star.source)  # once per target
```

- `mission` is `Kepler`, `K2` or `TESS` (case-insensitive). Planet names such as `Kepler-186f` are accepted and searched as their host star.
- `download_all=True` downloads every quarter or sector and joins them; without it only the first file is used, which is much faster but finds far fewer planets.
- `progress=` takes a callback that receives a `ProgressEvent` at each stage (`downloading`, `detrending`, `period_search`, `done`). Raising inside the callback cancels the run.
- `result.star` holds the host star once: `teff_k`, `logg`, `radius_rsun`, `mass_msun` (derived from log g and radius) and `source` (`KIC`, `EPIC` or `TIC`), read from the light-curve file header. Kepler's KIC values are rough (radius off by 25-40%). These describe the star, so they are not repeated on candidate rows; each candidate's orbit (distance, speed, insolation, temperature) is on its row.
- Feature values can be `NaN` when a measurement is impossible (for example the planet radius when the star's radius is unknown).
- Set `SHAULA_LIGHTCURVE_CACHE` (or pass `cache_dir=`) to choose where downloaded light curves are stored.

### Command line

```bash
uv run shaula-extract-lk --target Kepler-186 --mission Kepler --download-all \
    --out-features out/kepler186_features.csv
```

Use `--input-lightkurve path/to.csv` instead of `--target` to start from a light curve saved earlier with `--out-lightkurve`. Other options: `--author` and `--exptime` to pick a different data product, `--sigma-clip` for outlier removal, and `--quiet`.

`shaula-extract-csv` is a deprecated batch tool for "exoTrain"-style CSVs (one star per row, columns `FLUX1` … `FLUXN`).

### Batch runs and validation scripts

These run as modules from the repository root:

- `python -m shaula.scripts.extract_multiple_stars --targets data/targets/training_targets.csv` extracts many stars in a row, skipping stars that already have an output file. `--label-output-candidates` labels each row `CONFIRMED`, `FALSE-POSITIVE` or `UNKNOWN`, and `--invert` runs the search on the flipped light curve to produce negative training examples.
- `python -m shaula.scripts.get_literature_data Kepler-90` downloads catalogue values for a star into `data/confirmed/`.
- `python -m shaula.scripts.compare_confirmed_and_extracted` compares extracted features with those catalogue values (see the sanity check below).

### Tests

```bash
uv run pytest
```

The notebooks (`shaula/lightcurve_analysis.ipynb`, `shaula/tests/unit_graph_test.ipynb`) are exploratory and not part of the test suite.

## How the pipeline works

This is what `extract()` does, step by step. Every module path below is under `shaula/`.

### 1. Find the star

A target can be a star or planet name (`Kepler-186`, `Kepler-186f`), a catalogue designation (`KIC 8120608`, `EPIC 201111557`, `TIC 307210830`) or, through `resolve()`, a sky position. Planet suffixes are stripped so the search is always for the host star.

### 2. Download and clean the light curve

`download_and_clean.py` searches MAST with [lightkurve](https://docs.lightkurve.org/) and keeps a single, consistent data product per mission: the official long-cadence (~29.4 min) light curves for Kepler and K2, and SPOC 2-minute light curves for TESS. Mixing products would put different noise levels side by side. With `download_all`, every quarter or sector is downloaded and stitched into one series. Then:

```python
lc = lc.remove_nans().normalize().remove_outliers(sigma_upper=5, sigma_lower=float("inf"))
```

Only *upward* outliers are removed, because a transit is a *downward* dip and must not be clipped away.

### 3. Remove the star's own variability

Stars flicker and spot, and the spacecraft drifts, so the raw brightness wanders far more than a planet's dip. `detrend_and_period.py` fits that slow trend and divides it out, leaving a flat series where the dips stand out. The light curve is first normalised piece by piece across data gaps, and the trend is fitted with the candidate transits held out, so the fit cannot "eat" the dips it is meant to reveal.

### 4. Search for a periodic dip

The search uses the Box Least Squares (BLS) algorithm (`astropy.timeseries.BoxLeastSquares`): it slides a box-shaped dip across many trial periods and durations and reports which ones line up with real dips. Periods from 0.5 days up to a third of the observing span are searched, so that at least three transits always fit. A second pass on a more aggressively flattened series is mixed in, which keeps short transits on very active stars from being hidden. The best peaks are then refined: the period is zoomed in on, checked for being a multiple or fraction of the true period, and the duration and transit time are re-fitted.

### 5. Decide whether a signal is real

Each refined peak is scored with the **multiple event statistic (MES)**: the combined signal-to-noise of all its transits, the same quantity the Kepler mission used (see `sesmes.py`). A candidate is **accepted** when `max_mes >= 7.1` (Kepler's own threshold) and at least three transits are observed. Shaula's MES is a time-domain approximation of Kepler's, not an exact reproduction.

### 6. Look for more planets

After a candidate is accepted, its transits are masked out and the search runs again on what is left, so the next-strongest planet in the same system can be found. Before masking, the candidate is re-checked at 2×, 3×, ½ and ⅓ of its period, because locking onto half the true period would mask every real transit and hide the planet for good. Isolated deep dips left over after masking are also removed so they cannot combine into a fake period. The loop stops after 12 candidates, when nothing passes the threshold, or when half the light curve is masked.

### 7. Report near-misses

If the search ends because nothing else clears 7.1, up to two of the best rejected signals with `max_mes >= 6.0` are returned with `detection_status = "provisional"`. They never mask anything, so they cannot affect the other candidates. This is how real planets close to the noise floor still show up.

### 8. Measure the features

For each candidate, Shaula measures the features listed in the [table below](#extracted-features):

- **Shape of the flux distribution** (`utils/scaling_and_metrics.py`): robust scaling, skewness, kurtosis and outlier fraction.
- **Folded light curve** (`folded_binned_metrics.py`): all transits stacked at the candidate's period, giving out-of-transit noise, depth consistency and the autocorrelation of the noise.
- **Per-transit measurements** (`per_trans_stat.py`): depth of each individual transit and how much it varies.
- **Noise and signal strength** (`cdpp.py`, `sesmes.py`): noise over 3, 6 and 12 hour windows (CDPP), and the single-event (SES) and multiple-event (MES) statistics.
- **Vetting checks**: transit shape (box-like or V-shaped), a secondary eclipse halfway between transits, odd versus even transit depths, and ingress/egress symmetry. These are the usual signs of an eclipsing binary.
- **Planet radius**, when the light curve carries the star's radius.

The result is one row per candidate, sorted by period.

## Extracted features

Depths are fractional (0.001 = 1,000 ppm = a 0.1% dip); positive means dimming.

| Column                             | Meaning                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `period_days`                      | Orbital period in days |
| `t0`                               | Time of the first transit in the data (BKJD for Kepler/K2, BTJD for TESS). In the sanity check its % difference is the timing offset as a % of the transit duration |
| `duration_days` / `duration_hours` | Transit duration in days and hours                                                                               |
| `depth_mean_per_transit`           | Mean depth of the individual transits |
| `planet_radius_rjup`               | Inferred planet radius (Jupiter radii); NaN when the star's radius is unknown |
| `max_ses`                          | Strongest single-transit detection statistic (SES) of any one event; compared with the KOI `koi_max_sngle_ev`     |
| `max_mes`                          | Multiple-event statistic (MES) at the best nearby phase; the detection gate requires ≥ 7.1. Compared with `koi_max_mult_ev` |
| `MES`                              | Multiple-event statistic at the exact fitted ephemeris (no phase scan)                                            |
| `SES_mean` / `SES_std`             | Mean and spread of the per-transit SES across all observed events                                                 |
| `snr_per_transit_mean` / `snr_per_transit_std` | Mean and spread of the per-transit signal-to-noise |
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
| `skewness_flux` / `kurtosis_flux` / `outlier_resistance` | The same shape measures recomputed on the scaled flux during extraction |
| `vshape_metric`                    | Transit shape from a trapezoid fit: 0 = flat-bottomed box, 1 = pure V (grazing transit or eclipsing binary)      |
| `secondary_depth`                  | Signed depth at phase 0.5 (halfway between transits); ~0 when there is no secondary eclipse                      |
| `secondary_depth_snr`              | Signed significance of that secondary, on the same scale as `MES`; high values suggest an eclipsing binary        |
| `secondary_depth_snr_log` / `secondary_depth_snr_capped` | The same significance compressed as sign × log(1 + \|SNR\|), or clipped to ±100, for models that dislike large values |
| `odd_even_depth_ratio`             | Mean odd-transit depth / mean even-transit depth; far from 1 suggests an eclipsing binary at twice the period     |
| `even_odd_depth_diff_sigma`        | How many standard errors apart the odd and even transit depths are; large values point to an eclipsing binary found at half its period |
| `secondary_max_snr_any_phase` / `secondary_phase` / `secondary_depth_ratio` | Strongest dip at any phase away from the transit, where it sits, and its depth over the transit's; catches eccentric binaries whose secondary is not at phase 0.5 |
| `secondary_scan_trials`            | How many phases that search tried; noise alone gives a higher maximum when it tries more |
| `ingress_egress_asymmetry`         | Difference between ingress and egress slopes of the folded transit; ~0 for a symmetric transit                   |
| `planet_radius_rearth`             | Inferred planet radius (Earth radii); NaN when the star's radius is unknown |
| `semi_major_axis_au`               | Orbital distance (AU), from the period and the star's mass via Kepler's third law (circular orbit); NaN without the star's catalogue values |
| `orbital_speed_kms`                | Orbital speed (km/s) on that orbit |
| `insolation_earth`                 | Starlight the planet receives, relative to Earth |
| `equilibrium_temp_k`               | Equilibrium temperature (K), assuming a Bond albedo of 0.3 and full heat redistribution |
| `rho_star_transit`                 | Star's mean density (g/cm³) implied by the transit's period, duration and depth (circular orbit, central transit) |
| `log_rho_ratio`                    | log10 of that over the density from the header's log g and radius: ~0 for a planet of that star; far from 0 when the event does not fit the star (eclipsing binary, blend); grazing planets read somewhat high |
| `detection_status`                 | `accepted` (passed the MES gate) or `provisional` (sub-threshold, MES ≥ 6); scripts can also write `rejected` or `harmonic_duplicate` |
| `mes_threshold_used` / `is_provisional_detection` | The MES bar the row was judged against, and `1.0` for provisional rows (`0.0` otherwise) |
| `candidate_label`                  | Training label: `CONFIRMED`, `FALSE-POSITIVE` or `UNKNOWN`, from a period match against reference KOIs (scripts only) |
| `matched_target` / `matched_period_ratio` | Reference planet the row was matched to, and how its period relates to the reference (`direct`, or an alias such as `2P`, `P/2`) |
| `fp_source`                        | Why a row is `FALSE-POSITIVE`: the matched KOI's flags (e.g. `koi:nt`, `koi:ss+co`), `unmatched`, or `inverted`  |

## Sanity check against the Kepler catalogue

To check the measurements, we compare features written to `data/extracted/` with catalogue values in `data/confirmed/` for a set of well-known confirmed Kepler planets, using `shaula/scripts/compare_confirmed_and_extracted.py`. Extracted files are named by catalogue number (`KIC<kepid>_<date>.csv`, using the latest normal, non-inverted run per star) and hold one row per candidate. Rows are paired by period, including small integer aliases (a candidate found at a third of the true period is labelled `P/3` rather than counted as a separate detection). Each cell is the percent difference `((extracted − confirmed) / |confirmed|) × 100`, and rows are sorted by `mean_match_no_t0`, the average agreement over every column except `t0`.

All 18 planets of the 9 hosts below (Kepler-2, 4, 5, 7, 8, 11, 12, 15 and 186; KIC 10666592, 11853905, 8191672, 5780885, 6922244, 6541920, 11804465, 11359879 and 8120608) are recovered at their catalogue period, none through an alias.

| Candidate   | period_days | depth_mean_per_transit | duration_hours |  max_ses | max_mes | duration_days | planet_radius_rjup |      t0 | mean_match_no_t0 |
| ----------- | ----------: | ---------------------: | -------------: | -------: | ------: | ------------: | -----------------: | ------: | ---------------: |
| Kepler-5b   |      +0.00% |                 -3.96% |         +2.02% |  +13.81% |  -5.02% |        +2.02% |             +0.09% |  -0.51% |           96.15% |
| Kepler-11c  |      +0.00% |                 -6.29% |         +7.40% |   +0.10% |  -4.53% |        +7.40% |             +6.99% |  -0.08% |           95.33% |
| Kepler-11e  |      +0.00% |                 -5.07% |         +2.67% |  +22.46% |  -2.18% |        +2.67% |             +0.09% |  +1.27% |           94.98% |
| Kepler-186b |      -0.00% |                -18.78% |         +3.50% |   -3.41% |  +1.59% |        +3.50% |             +4.50% |  -3.43% |           94.96% |
| Kepler-8b   |      +0.00% |                 -3.58% |         +7.27% |  +11.72% |  -7.49% |        +7.27% |             -6.36% |  -0.36% |           93.76% |
| Kepler-2b   |      +0.00% |                 -2.11% |         +6.63% |   +9.31% | -17.41% |        +6.63% |             +2.82% |  +0.12% |           93.58% |
| Kepler-15b  |      +0.00% |                 -5.33% |         -1.64% |  +28.46% |  -6.74% |        -1.64% |             +2.37% |  -0.02% |           93.40% |
| Kepler-12b  |      -0.00% |                 -4.34% |         +3.29% |  +40.51% |  -5.58% |        +3.29% |             -0.71% |  +0.71% |           91.76% |
| Kepler-4b   |      +0.01% |                -19.01% |        +13.12% |   +8.94% |  -2.81% |       +13.12% |            +10.92% |  +0.52% |           90.30% |
| Kepler-11g  |      -0.00% |                 +0.79% |         +9.82% |  -20.08% | -19.96% |        +9.82% |            +10.88% |  +4.62% |           89.81% |
| Kepler-11d  |      +0.00% |                 -5.31% |         +7.01% |  +41.10% |  -5.68% |        +7.01% |             +7.50% |  -1.15% |           89.48% |
| Kepler-186e |      -0.00% |                 -6.73% |         +7.19% |  +35.80% | +12.25% |        +7.19% |            +11.89% |  +7.45% |           88.42% |
| Kepler-7b   |      -0.00% |                 -3.95% |         +6.35% |  +63.44% |  -7.02% |        +6.35% |             +0.36% |  -0.43% |           87.50% |
| Kepler-11b  |      +0.00% |                 -9.69% |         +9.21% |  +62.40% | +11.81% |        +9.21% |             +4.77% |  +1.02% |           84.70% |
| Kepler-186c |      +0.00% |                -11.28% |        +21.06% |  +49.35% |  +1.32% |       +21.06% |             +9.90% |  +0.28% |           83.72% |
| Kepler-186d |      +0.00% |                 -7.00% |        +26.78% |  +53.06% |  +2.41% |       +26.78% |             +9.53% |  -9.09% |           82.06% |
| Kepler-186f |      +0.00% |                -11.66% |         -9.94% |  +70.80% | +26.68% |        -9.94% |             -7.71% | -13.87% |           80.47% |
| Kepler-11f  |      +0.00% |                 -7.37% |        +18.90% | +109.88% | +16.32% |       +18.90% |             +5.73% |  +0.66% |           76.11% |

<figcaption>
   Percent difference of pipeline-extracted features vs confirmed KOI catalog values (sorted by <code>mean_match_no_t0</code>)
</figcaption>

**What this shows:**

- **Periods** match the catalogue to within 0.01% for every planet, including Kepler-186f, a small planet with only about a dozen transits in four years.
- **Depths** are mostly within about 10%. Shaula tends to measure them slightly shallower than the catalogue, by up to ~19% for Kepler-4b and Kepler-186b. Planet radii follow, mostly within ~12%.
- **Durations** agree within ~13% for most planets; the small, shallow planets (Kepler-186c, Kepler-186d, Kepler-11f) come out 19–27% longer, because their transit edges are hard to see in the noise.
- **`max_mes`**, the detection statistic, is within ~20% of Kepler's for every planet except Kepler-186f (+27%). **`max_ses`**, the strongest single transit, runs higher than the catalogue (up to +110% for Kepler-11f), as expected for a time-domain approximation of Kepler's wavelet-based statistic.
- Every host except Kepler-2 also has 1 to 6 extra candidates that match no confirmed planet, mostly at long periods with `max_mes` between 6 and 19. They are not shown in the table. The search deliberately favours finding every real planet over avoiding false alarms; telling the two apart is the job of the vetting features and the downstream classifier.

## Credits

The prototype code in this repository was first written for the Gatonautas team's project at the [NASA Space Apps Challenge 2025](https://www.nasa.gov/nasa-space-apps-challenge-2025/).
