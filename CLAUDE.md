# CLAUDE.md

`shaula` is an installable library for extracting transit-photometry features from light curves. The package and the distribution are both named `shaula`; this repository was formerly `ltp-features`.

## Commands

```bash
uv sync                              # install
uv run pytest                        # tests: 248 passed, 19 subtests
uv run ruff check <paths>            # lint only the files you touch
```

- `uv run ruff check shaula` is configured but **exits non-zero: it currently reports 116 errors**. This is style debt in code that predates packaging (line length, nested `with`, import order, ambiguous Unicode), concentrated in `shaula/detrend_and_period.py`. New code must be clean. Never run `--fix` across the package.
- Tests live in `shaula/tests/`. Older ones are `unittest.TestCase` classes run under pytest; the `api` tests are plain pytest functions.
- One third-party `UserWarning` from lightkurve/oktopus fires on any import of the package. It is not a regression.
- CLIs: `shaula-extract-lk` (download by target name, then extract) and `shaula-extract-csv` (extract from a saved light-curve CSV with `time`/`flux` columns). Validation scripts run as modules, e.g. `python -m shaula.scripts.get_literature_data Kepler-90` and `python -m shaula.scripts.compare_confirmed_and_extracted`.

## Public surface: `shaula/api.py`

The only module an external consumer imports (the `apps/shaula` gRPC service in Antares): `extract()`, `resolve()`, `ProgressEvent`, `ExtractionResult`, `ResolvedTarget`, `TargetNotFound`, `STAGES`. Changing their signatures is a breaking change for that service.

- `STAGES` has exactly four entries: `downloading`, `detrending`, `period_search`, `done`. Every declared stage must actually be emitted; a stage that never fires is a broken contract, and a test pins it. `period_search` is emitted from inside `extract_features_from_arrays`, once per search iteration.
- The `progress` callback may raise to cancel. `extract()` deliberately does not catch it. Never wrap a callback invocation in `try`/`except`.
- `resolve()` returns an exact catalogue identifier (`KIC`/`EPIC`/`TIC`). The caller uses `catalog_id` as a cache key, so `resolve()` raises `TargetNotFound` for an unusable identifier or non-finite coordinates instead of returning a plausible-looking value.
- `resolve()` takes either a `query` or both `ra_deg` and `dec_deg` (never both, never one coordinate); a blank query counts as none. A position resolves to the nearest star with light curves within `POSITION_MATCH_RADIUS_ARCSEC` (4 arcseconds) and sets `display_name` to the designation. `0.0` is a valid coordinate, so test with `is None`, never truthiness.
- `extract()` returns accepted detections plus up to two `provisional` sub-threshold rows; it never asks for false candidates or labels (those are CLI/script options).

## Pipeline workflow

What `extract(target, mission)` does, end to end. Module paths are under `shaula/`.

1. **Validate and normalise the target** (`api.py`). Reject queries over `MAX_QUERY_LENGTH`, canonicalise the mission (`Kepler`, `K2`, `TESS`), and strip a planet or candidate suffix with `utils/target_names.host_star_name` (`Kepler-186f` searches `Kepler-186`, `EPIC201111557.01` searches `EPIC201111557`).
2. **Search MAST** (`download_and_clean.search_default_product`). One product class per mission: Kepler and K2 use the official pipeline's long cadence (~29.4 min), TESS uses SPOC short cadence. Filtering happens before download so stitching never mixes authors or exposure times. The unmemoised `lk.search_lightcurve.__wrapped__` is called on purpose (see Gotchas).
3. **Download and clean** (`download_and_clean_lightcurve`). Download one quarter/sector, or all of them and `stitch()` with `download_all=True`; then `remove_nans().normalize().remove_outliers(sigma_upper=5, sigma_lower=inf)`. Only upward outliers are clipped, because transits are downward.
4. **Iterative multi-planet search** (`extract_feats.extract_features_from_arrays`). Up to `MAX_TRANSIT_CANDIDATES` (12) iterations. Each one searches the cadences still active, measures the strongest independent BLS peaks until one passes the MES gate, settles harmonics, then masks that candidate's transits (`TRANSIT_MASK_WIDTH` 1.5 x T14) plus any unexplained deep single events before the next iteration. This mirrors Kepler TPS "gapping" of detected transits, not a joint multi-planet fit. Stops when nothing passes, fewer than 50 cadences remain, or 50% of cadences are masked.
5. **Detrend and period search, per candidate** (`detrend_and_period.detrend_with_bls_mask`):
   - per-segment robust normalisation and 0.3% winsorisation across data gaps;
   - optional deep-eclipse masking (`mask_eclipses`);
   - BLS-only variability removal (running median, window = 3 x widest searched duration);
   - stage-1 global `astropy.timeseries.BoxLeastSquares` on a binned series with a compute-budgeted, duration-aware log-period grid from 0.5 d to span/3 (so at least 3 transits fit), over `GLOBAL_BLS_DURATIONS` (24) log-spaced durations; the budget is periods x durations, so fewer durations buy a denser period grid, which short transits over a 4-year baseline need;
   - `rank_independent_bls_candidates` clusters peaks (`scipy.signal.find_peaks`) into a ranked queue;
   - a second BLS pass on flux flattened with a 0.75 d window (`BLS_FINE_VARIABILITY_WINDOW_DAYS`, durations up to a third of it) is interleaved into that queue, so short transits on strongly variable stars (Kepler-411) are not buried under residual variability;
   - local BLS zoom, de-aliasing up to `DEALIAS_MAX_MULTIPLE` (10), folded-profile and trapezoid duration refits, optional TLS refinement within ±1.5% of the period;
   - final trend: `scipy.interpolate.UnivariateSpline` on binned out-of-transit medians, with this candidate's and earlier candidates' transits withheld; flux is divided by the trend;
   - final epoch refit, limited to ±0.5 T14 so a shallow candidate cannot slide onto a deeper unmasked signal.
6. **Noise model, CDPP and SES/MES** (`cdpp.py`, `sesmes.py`). `duration_matched_statistics` builds a local box matched filter at T14, with noise from a robust median/MAD of half-duration-spaced anchors, excluding transit windows. Each box's depth is measured against its flanks (1, 2, 3 or 5 box widths per side) or the series median, whichever gives the lowest noise (`flank_durations="auto"`, `ADAPTIVE_FLANK_DURATIONS`); this removes stellar variability the final spline leaves in the flux, which otherwise cuts MES several-fold on active stars. Before features are measured, `_mes_optimal_duration` scores T14 x 0.5 to 4 by `max_mes` and adopts a trial that beats the refit T14 by 10%. `compute_SES_MES` folds its numerator/denominator at the ephemeris (MES = sum N / sqrt(sum D)), and `max_mes` scans a small phase offset. This is a time-domain approximation of Kepler TPS; it deliberately does not reproduce TPS's adaptive wavelet whitening. Synthetic validation of both lives in `experiments/mes_accuracy/` (`LOG.md`).
7. **Detection gate** (`_candidate_passes_mes`): `max_mes >= MES_DETECTION_THRESHOLD` (7.1, the Kepler TPS threshold) with at least 3 observed events. Queued peaks are tried in order of quick MES, not BLS power, and pre-screened at MES 4.0 before full refinement. When de-aliasing moves a queued peak by more than 1% (`DEALIAS_RETRY_TOLERANCE`) and the result fails the gate, the peak is refined again at its queued period with `dealias=False`. The `_mes_optimal_duration` audit is capped by `detrend_and_period.max_physical_duration`, the same limit the BLS refits use. `_reconcile_accepted_harmonic` refolds the same statistics object at 2P, 3P, P/2 and P/3 and adopts one only if its MES beats the original by `HARMONIC_ADOPT_MARGIN` (1.15).
8. **Feature measurement** (`_extract_single_candidate_from_arrays`). See the table below.
9. **Recovery, sort and radius** (`extract_features_from_arrays`, `extract_features_from_lightcurve`). When the search runs out of signal, up to 2 rejected peaks with `max_mes >= 6.0` come back as `provisional` rows (they never masked anything). Rows are sorted by `period_days`. When `lc.meta["RADIUS"]` is finite, planet radius = R* x sqrt(depth) x 109.1 (Earth) or 9.731 (Jupiter); otherwise NaN.
10. **Return** (`api.py`). Each row becomes a plain `dict` in `ExtractionResult.features`, with `shaula_version`. The CLIs instead write CSV or JSON via `save.save_features`, one row per candidate.

## Extracted features

One row per candidate. Depths are fractional flux deficits (positive = dimming); CDPP is in ppm.

| Column(s) | How it is computed |
|---|---|
| `period_days`, `t0`, `duration_days`, `duration_hours` | Final BLS/refit ephemeris. `t0` is canonicalised to the first epoch after the data start (`utils/ephemeris.canonical_epoch`), in BKJD for Kepler/K2 and BTJD for TESS. |
| `scale_mean`, `scale_std`, `scale_skewness`, `scale_kurtosis`, `scale_outlier_resistance` | `utils/scaling_and_metrics`: detrended flux robust-scaled by 1.4826 x MAD; moments from `scipy.stats`; outlier resistance is the % of points beyond 5 sigma. |
| `skewness_flux`, `kurtosis_flux`, `outlier_resistance` | The same moments recomputed on the scaled flux in `extract_feats`. |
| `local_noise`, `depth_stability`, `cadence_hours` | `folded_binned_metrics`: 1.4826 x MAD of out-of-transit flux; std/mean of per-epoch depths; median cadence. |
| `acf_lag_1h` … `acf_lag_24h` | `statsmodels.tsa.stattools.acf` (FFT) at 1, 3, 6, 12 and 24 hour lags; measures correlated (red) noise. |
| `depth_mean_per_transit`, `depth_std_per_transit`, `npts_transit_median` | `per_trans_stat.per_transit_stats_simple`: for each epoch, local out-of-transit median minus the median of the central half of the transit. |
| `cdpp_3h`, `cdpp_6h`, `cdpp_12h` | `cdpp.calculate_cdpp`: median duration-matched noise (ppm) from the same model as SES/MES, transit windows excluded. Not Kepler's wavelet CDPP. |
| `SES_mean`, `SES_std`, `MES`, `max_ses`, `max_mes` | `sesmes.compute_SES_MES`. `MES` is at the exact ephemeris; `max_mes` is the best nearby phase and is what the gate uses. |
| `snr_global` | `depth_mean_per_transit x 1e6 / CDPP` interpolated at T14 (`utils/interp_cdpp`). |
| `snr_per_transit_mean`, `snr_per_transit_std` | Mean and std of the per-event SES. |
| `resid_rms_global` | Std of the detrended flux about its median. |
| `vshape_metric` | `utils/compute_vshape`: least-squares trapezoid fit to the unbinned points around each transit, on a grid of total durations (0.6–1.4 x T14) and ingress fractions, with the model averaged over the cadence. The value is the fitted ingress fraction T12 / (T14/2): 0 for a flat-bottomed box, 1 for a pure V (grazing transit or EB). Noisy at low MES. |
| `secondary_depth`, `secondary_depth_snr`, `secondary_depth_snr_log`, `secondary_depth_snr_capped` | `sesmes.secondary_eclipse`: the candidate's own SES/MES statistics folded at `t0 + P/2` with no phase scan. Depth is the inverse-variance weighted box depth (sum N / sum D), SNR is that fold's MES, so it is directly comparable to `MES`. Both are **signed** and scatter around 0 when there is no secondary. `_log` is sign(SNR) x log1p(abs(SNR)); `_capped` clips SNR to ±100. A hot Jupiter's occultation (e.g. Kepler-2b) is a real secondary too. |
| `odd_even_depth_ratio` | Mean odd-epoch depth over mean even-epoch depth; far from 1 suggests an EB at twice the period. |
| `ingress_egress_asymmetry` | Difference of absolute linear slopes either side of the folded minimum. |
| `planet_radius_rearth`, `planet_radius_rjup` | From stellar radius and depth (step 9), else NaN. |
| `mes_threshold_used`, `is_provisional_detection`, `detection_status` | Gate the row was judged against; `1.0` and `"provisional"` for recovered sub-threshold rows, `0.0` and `"accepted"` otherwise. Scripts can also emit `"rejected"` and `"harmonic_duplicate"`. |
| `candidate_label`, `matched_target`, `matched_period_ratio` | Only with `label_output_candidates` (CLI/scripts): `CONFIRMED`, `FALSE-POSITIVE` or `UNKNOWN`, by period match against `data/confirmed/`. |

## Libraries and how they are used

- **lightkurve**: MAST search (`search_lightcurve`), download, `stitch`, `remove_nans`, `normalize`, `remove_outliers`. Its `LightCurve.meta["RADIUS"]` feeds planet radius. It uses **astroquery** internally for MAST queries; Shaula does not import astroquery directly.
- **astropy**: `timeseries.BoxLeastSquares` is the period search engine; `coordinates.SkyCoord` builds the cone-search position in `resolve()`.
- **numpy**: all array work. Missing values are NaN throughout, and helpers return NaN instead of raising when a measurement is impossible.
- **scipy**: `stats.binned_statistic` for phase-folded median profiles and BLS input binning; `signal.find_peaks` for periodogram peaks; `interpolate.UnivariateSpline` for the detrending trend; `stats.skew` / `kurtosis` for flux moments.
- **statsmodels**: only the autocorrelation function in `folded_binned_metrics.py`.
- **pandas**: CSV/JSON I/O (`save.py`, `cli/extract_csv.py`), the confirmed catalogue and the comparison tooling under `utils/` and `scripts/`. The core extraction works on numpy arrays.
- **transitleastsquares** (optional `tls` extra): imported lazily inside `detrend_with_bls_mask` for a narrow TLS refinement when `use_tls=True`.
- **threadpoolctl** (optional `parallel` extra): caps BLAS threads in `scripts/extract_multiple_stars.py`.

## Validation data

- `data/confirmed/<Star>-confirmed.csv`: literature parameters per host, written by `scripts/get_literature_data.py` from the NASA Exoplanet Archive KOI cumulative table (TAP).
- `data/extracted/`, `data/testing/`: feature CSVs written by the pipeline, compared against confirmed values with `scripts/compare_confirmed_and_extracted.py` (pairs rows by period, including small integer aliases).
- Notebooks (`shaula/lightcurve_analysis.ipynb`, `shaula/tests/unit_graph_test.ipynb`) are exploratory and not run by pytest.

## Gotchas

- `SHAULA_LIGHTCURVE_CACHE` sets the download cache. `shaula/paths.py` only resolves the directory (explicit argument, then the variable, then `None` for lightkurve's own default); `download_and_clean_lightcurve` passes it to lightkurve per call as `download_dir`. It never assigns `lightkurve.conf.cache_dir`: lightkurve ignores a `LIGHTKURVE_CACHE_DIR` environment variable, and a process-wide setting is wrong for a threaded service (concurrent calls overwrite each other, and the setting sticks to later calls).
- `lk.search_lightcurve` is memoised without limit and would cache a transient empty result forever, so `search_default_product` calls `__wrapped__`.
- Pass `ResolvedTarget.designation` (for example `"KIC 8120608"`) to `extract()` so the star extracted is exactly the star resolved. `resolve()` and `extract()` share one filtered, unmemoised search (`search_default_product`).
- `use_tls=True` raises `RuntimeError` before any download when the optional `tls` extra (`transitleastsquares`) is not installed.
- A query longer than 200 characters (`MAX_QUERY_LENGTH`) raises `ValueError` in both `resolve()` and `extract()`; a mission is matched case-insensitively (`canonical_mission`).
- Feature values in `ExtractionResult.features` are `float`, `str` or `int`. A float **may be `NaN`** (for example `planet_radius_rearth` when the light curve carries no stellar radius). Callers that serialise to JSON or protobuf must map non-finite floats themselves; Shaula keeps `NaN` deliberately because downstream pandas code relies on it.
- Accepted candidates' cadences are passed to the noise model as `gap_mask` on the full series, never sliced out: deleting cadences fragments the noise model's segments and biases its MAD low. Keep that pattern in new code.
- Many constants in `extract_feats.py` and `detrend_and_period.py` carry comments naming the Kepler system that set them (Kepler-90, Kepler-186e, Kepler-447). Read them before tuning; they record regressions.
- The pipeline `print`s progress to stdout heavily. Structured progress goes through the `progress` callback, not stdout.
- Extraction imports `ProgressEvent` from `.api` inside the loop because `api` imports `extract_feats`; a top-level import would be circular.
- `shaula/__init__.py` defines `__version__` before importing from `.api`. That ordering is load-bearing against a circular import, and the `# noqa: E402` on that import is required. Ruff's `RUF100` calls it unused, but removing it produces 13 errors. Do not "clean it up".

## References (Kepler mission pipeline)

The detection statistics here are approximations of the official Kepler Science Operations Center (SOC) pipeline. Use these when changing CDPP, SES/MES, the 7.1 threshold or vetting features.

- Original pipeline source code: [github.com/nasa/kepler-pipeline](https://github.com/nasa/kepler-pipeline). Start with its [Source Code Road Map](https://github.com/nasa/kepler-pipeline/blob/master/kscrm.pdf).
- [Kepler Data Processing Handbook (KSCI-19081-003)](https://archive.stsci.edu/files/live/sites/mast/files/home/missions-and-data/kepler/_documents/KSCI-19081-003-KDPH.pdf), ed. J. Jenkins, 2020; describes every pipeline module including TPS and DV. Other MAST handbooks (Archive Manual, Data Characteristics, Instrument) are listed at [archive.stsci.edu/missions-and-data/kepler/documents](https://archive.stsci.edu/missions-and-data/kepler/documents).
- Jenkins et al. 2010, [Overview of the Kepler Science Processing Pipeline](https://arxiv.org/abs/1001.0258), ApJL 713, L87.
- Jenkins et al. 2010, Transiting planet search in the Kepler pipeline, Proc. SPIE 7740, [doi:10.1117/12.856764](https://doi.org/10.1117/12.856764). The TPS algorithm behind SES, MES and the 7.1 threshold.
- Christiansen et al. 2012, [The Derivation, Properties and Value of Kepler's Combined Differential Photometric Precision](https://arxiv.org/abs/1208.0595), PASP. Defines CDPP.
- Twicken et al. 2018, [Kepler Data Validation I: Architecture, Diagnostic Tests, and Data Products for Vetting Transiting Planet Candidates](https://arxiv.org/abs/1803.04526), PASP. Odd/even, secondary eclipse and other vetting tests.
- NASA Exoplanet Archive [KOI table column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html), including `koi_max_sngle_ev` / `koi_max_mult_ev`, which the confirmed CSVs' `max_ses` / `max_mes` come from.

## Commits

`type: brief message`, one commit per logical change. **No attribution trailers: never add a `Co-authored-by:` line** (or any other trailer) to a commit message in this repo. Never push. Never commit `plans/` or downloaded data; use `git add <explicit paths>`.
