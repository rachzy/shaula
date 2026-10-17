---
name: star-finder
description: Find host stars that match a requested set of characteristics (planet count, discovery method, stellar multiplicity, spectral type, period or depth range, mission) and download their light curves with Shaula's batch extraction script. Use this whenever someone asks for light-curve data, training data or "stars like X" described by properties rather than by name, for example "binary systems with at least 3 transiting planets from Kepler", "M dwarfs with TESS planets", or "more hot Jupiters for the dataset", even if they never say "find" or name the script.
---

# Star finder

Turn a description of stars ("binary systems with at least 3 confirmed transiting exoplanets, Kepler only") into a short list of real, verified host stars that are not yet in this repo's dataset, tell the user which stars you picked, then download and extract them with `shaula/scripts/extract_multiple_stars.py` and report what came back.

Work through the steps in order. Each one exists because skipping it has a cost: a star from the wrong mission wastes a long download, a star already in `data/confirmed/` adds a duplicate, and a star with no light-curve products fails after minutes of work.

## 1. Parse the request and narrow the search

Pull these out of the prompt before searching:

- **How many stars.** If the user gives no number, pick 5 and say so.
- **Mission.** If the user names one dataset (Kepler, K2 or TESS), search only that one, and every star you return must have light curves from it. If they name none, any mission is fair game, but prefer Kepler: it is what the pipeline was tuned on and the only mission whose confirmed-planet labels are fetched automatically (see step 5).
- **Hard criteria.** Each property the user stated (planet count, discovery method, stellar multiplicity, spectral type, magnitude, period range, and so on). Treat the user's thresholds literally: "at least 3" means 3 or more confirmed planets, not candidates.

Keep the list of criteria; you will check every chosen star against each one and cite the result in step 4.

## 2. Research candidates

Use your own knowledge to brainstorm, then confirm against a catalogue, because recalled planet counts and multiplicity flags are often out of date.

The most reliable source is the NASA Exoplanet Archive's Planetary Systems Composite table (`pscomppars`, one row per confirmed planet), queried over the same TAP endpoint `shaula/scripts/get_literature_data.py` uses. Useful columns: `hostname`, `sy_snum` (stars in the system, so `>= 2` means binary or multiple), `sy_pnum` (confirmed planets, any method), `discoverymethod`, `disc_facility`, `st_teff`, `st_rad`, `sy_kepmag`, `sy_tmag`, `pl_orbper`. Example for the prompt above:

```bash
curl -sS "https://exoplanetarchive.ipac.caltech.edu/TAP/sync" \
  --data-urlencode "query=select hostname, sy_snum, count(*) as n_transit
    from pscomppars
    where discoverymethod = 'Transit' and disc_facility = 'Kepler' and sy_snum >= 2
    group by hostname, sy_snum
    having count(*) >= 3
    order by n_transit desc" \
  --data-urlencode "format=csv"
```

Count transiting planets with `count(*)` and a `discoverymethod` filter rather than reading `sy_pnum`, which also counts radial-velocity and imaging planets. For K2 use `disc_facility = 'K2'`; for TESS use `'Transiting Exoplanet Survey Satellite (TESS)'`. If the archive is unreachable, fall back to web searches (the archive's overview pages, discovery papers, stellarcatalog.com) and say in step 4 that the values come from those sources.

Gather a few more candidates than requested; some will drop out in step 3.

## 3. Filter out stars the dataset already has, and confirm each one resolves

**Already in the dataset.** A star counts as present when any of these exist (compare on the host-star name, so `Kepler-90h` and `Kepler-90` are the same star; `shaula.utils.target_names.host_star_name` does this normalisation):

- `data/confirmed/<Star>-confirmed.csv` (the confirmed-planet catalogue the user pointed to)
- `data/extracted/<Star>_*.csv` (the batch script would skip it anyway)
- an entry, commented or not, in the `STARS` list in `shaula/scripts/extract_multiple_stars.py`

```bash
ls data/confirmed/ data/extracted/ data/testing/
grep -n 'Star("' shaula/scripts/extract_multiple_stars.py
```

Also watch for the same star under another name (a Kepler-N host is also a KOI and a KIC number, a TOI is also a TIC). The catalogue's `hostname` is the name to use, because it is what the files are keyed on.

**Has light curves for the mission.** Resolve every survivor with Shaula's own search, which applies the same mission filter the extraction will use. A `TargetNotFound` means the download would fail, so drop that star:

```bash
uv run python -c "
from shaula.api import resolve, TargetNotFound
for name in ['Kepler-47', 'Kepler-132']:
    try:
        print(name, '->', resolve(name, 'Kepler'))
    except TargetNotFound as exc:
        print(name, '-> NOT FOUND:', exc)
"
```

If too few stars survive, go back to step 2 rather than relaxing a criterion. If the criteria genuinely cannot be met (there are only so many circumbinary multi-planet systems), return the ones that do qualify and say how many exist.

## 4. Tell the user which stars you chose, then start

Right before running the download, send the user one message listing the chosen stars. For each star give its mission, how it meets every criterion (with the values you checked, for example "2 stars, 3 transiting planets"), and a link to its page on Stellar Catalog so they can inspect the star themselves.

Stellar Catalog star pages are `https://www.stellarcatalog.com/stars/<slug>`, where the slug is the star's name in lower case with spaces replaced by hyphens: `Kepler-90` is `https://www.stellarcatalog.com/stars/kepler-90` and `HD 191939` is `https://www.stellarcatalog.com/stars/hd-191939`. Not every star has a page; if you could not confirm one exists, say the link is unverified rather than leaving it out.

Also state the options you are running with (step 5) and any star you dropped and why. This message informs; it does not wait for approval, so move straight on to step 5 unless the user asked to review the list first.

## 5. Run the batch extraction

`extract_multiple_stars.py` runs `extract_and_compare` once per star: download the light curve from MAST, extract candidate features, and compare against `data/confirmed/`. It skips any star that already has an output CSV, and a star that fails is reported without stopping the batch.

Things to know about the script:

- It reads its targets from the module-level `STARS` list; there is no command-line option for star names. Call its `run_batch()` function with your own list instead of editing the file, so the curated `STARS` list stays as it is. Only add your stars to `STARS` if the user wants them kept in the curated batch.
- `--include-false-candidates` and `--label-output-candidates` (each with a `--no-` form) are required on the command line, and are keyword arguments to `run_batch()`. Default to both `True` (labelled training rows including rejected peaks, the script's documented use) unless the user asked only for the confirmed planets' features, and say which you chose in step 4.
- Labels come from `data/confirmed/`. A missing Kepler confirmed CSV is fetched from the NASA Exoplanet Archive automatically and written there. That fetch is Kepler-only, so K2 and TESS rows are labelled `UNKNOWN`; mention this when the stars are not from Kepler.
- Each star is a `Star(name, group, mission)`. `group` is a free-text tag printed in the log; use a short slug of the request, such as `"binary-3-transiting"`. `mission` is `"Kepler"`, `"K2"` or `"TESS"`.
- `workers` runs several stars in parallel processes. Each one uses about 350 MB and one core, so keep it at or below the core count (`nproc`).
- Light curves download into lightkurve's cache, or the directory in `SHAULA_LIGHTCURVE_CACHE` if set.

Run it from the repo root. A full multi-quarter Kepler download and search takes minutes per star, so run it in the background and capture the log:

```bash
uv run python -c "
from shaula.scripts.extract_multiple_stars import Star, run_batch
stars = [
    Star('Kepler-47', 'binary-3-transiting', mission='Kepler'),  # example names; use your step-3 list
    Star('Kepler-132', 'binary-3-transiting', mission='Kepler'),
]
succeeded, skipped, failed = run_batch(
    stars,
    include_false_candidates=True,
    label_output_candidates=True,
    workers=2,
)
print('Succeeded:', sorted(succeeded))
print('Skipped:  ', sorted(skipped))
print('Failed:   ', sorted(failed))
" 2>&1 | tee star-finder-run.log
```

If you did add the stars to `STARS`, the equivalent command-line run is:

```bash
uv run python -m shaula.scripts.extract_multiple_stars \
  --include-false-candidates --label-output-candidates --workers 2
```

That runs the whole `STARS` list, but stars with existing output are skipped, so only new entries do any work.

## 6. Wait for it to finish, then report

Wait until the batch prints its succeeded / skipped / failed summary. Don't report early from a partial log.

Then tell the user:

- which stars succeeded, were skipped (already had output) or failed, with the error line for each failure (search the log for `Extraction failed`);
- the CSVs written, at `data/extracted/<Star>_<YYYYMMDD>.csv`, and how many candidate rows each holds;
- with labelling on, how many rows per star matched a confirmed planet (`candidate_label == "CONFIRMED"`) compared with the planets in that star's confirmed CSV, since that is the quickest sign of whether the extraction recovered the system;
- any confirmed CSVs newly fetched into `data/confirmed/`.

```bash
uv run python -c "
import pandas as pd, glob
for path in sorted(glob.glob('data/extracted/Kepler-47_*.csv')):
    df = pd.read_csv(path)
    print(path, len(df), 'rows')
    if 'candidate_label' in df:
        print(df['candidate_label'].value_counts().to_string())
"
```

Leave the outputs uncommitted unless the user asks; the repo's rule is to never commit downloaded data, and a log file like `star-finder-run.log` should be deleted once you have reported.
