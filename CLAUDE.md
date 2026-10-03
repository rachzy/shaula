# CLAUDE.md

`shaula` is an installable library for extracting transit-photometry features from light curves. The package and the distribution are both named `shaula`; this repository was formerly `ltp-features`.

## Commands

```bash
uv sync                              # install
uv run pytest                        # tests: 194 passed, 19 subtests
uv run ruff check <paths>            # lint only the files you touch
```

- `uv run ruff check shaula` is configured but **exits non-zero: it currently reports 118 errors**. This is style debt in code that predates packaging (line length, nested `with`, import order, ambiguous Unicode), concentrated in `shaula/detrend_and_period.py`. New code must be clean. Never run `--fix` across the package.
- Tests live in `shaula/tests/`. Older ones are `unittest.TestCase` classes run under pytest; the `api` tests are plain pytest functions.
- One third-party `UserWarning` from lightkurve/oktopus fires on any import of the package. It is not a regression.

## Public surface: `shaula/api.py`

The only module an external consumer imports (the `apps/shaula` gRPC service in Antares): `extract()`, `resolve()`, `ProgressEvent`, `ExtractionResult`, `ResolvedTarget`, `TargetNotFound`, `STAGES`. Changing their signatures is a breaking change for that service.

- `STAGES` has exactly four entries: `downloading`, `detrending`, `period_search`, `done`. Every declared stage must actually be emitted; a stage that never fires is a broken contract, and a test pins it.
- The `progress` callback may raise to cancel. `extract()` deliberately does not catch it. Never wrap a callback invocation in `try`/`except`.
- `resolve()` returns an exact catalogue identifier (`KIC`/`EPIC`/`TIC`). The caller uses `catalog_id` as a cache key, so `resolve()` raises `TargetNotFound` for an unusable identifier or non-finite coordinates instead of returning a plausible-looking value.

## Gotchas

- `SHAULA_LIGHTCURVE_CACHE` sets the download cache. `shaula/paths.py` only resolves the directory (explicit argument, then the variable, then `None` for lightkurve's own default); `download_and_clean_lightcurve` passes it to lightkurve per call as `download_dir`. It never assigns `lightkurve.conf.cache_dir`: lightkurve ignores a `LIGHTKURVE_CACHE_DIR` environment variable, and a process-wide setting is wrong for a threaded service (concurrent calls overwrite each other, and the setting sticks to later calls).
- Pass `ResolvedTarget.designation` (for example `"KIC 8120608"`) to `extract()` so the star extracted is exactly the star resolved. `resolve()` and `extract()` share one filtered, unmemoised search (`search_default_product`).
- `use_tls=True` raises `RuntimeError` before any download when the optional `tls` extra (`transitleastsquares`) is not installed.
- A query longer than 200 characters (`MAX_QUERY_LENGTH`) raises `ValueError` in both `resolve()` and `extract()`; a mission is matched case-insensitively (`canonical_mission`).
- Feature values in `ExtractionResult.features` are `float`, `str` or `int`. A float **may be `NaN`** (for example `planet_radius_rearth` when the light curve carries no stellar radius). Callers that serialise to JSON or protobuf must map non-finite floats themselves; Shaula keeps `NaN` deliberately because downstream pandas code relies on it.
- `shaula/__init__.py` defines `__version__` before importing from `.api`. That ordering is load-bearing against a circular import, and the `# noqa: E402` on that import is required. Ruff's `RUF100` calls it unused, but removing it produces 13 errors. Do not "clean it up".

## Commits

`type: brief message`, one commit per logical change, no attribution trailers. Never push. Never commit `plans/` or downloaded data; use `git add <explicit paths>`.
