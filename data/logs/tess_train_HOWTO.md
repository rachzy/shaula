# Running one fold of the TESS training targets

Nothing has been downloaded yet. Each fold is independent, so folds can run on different machines at the same time. N is the fold number (1-5).

## Setup (once per machine)

Use a checkout of `feature/tess-lightcurves` (on the PC: the worktree `~/shaula-tess`), with `uv sync` done. Unpack the bundle at the repo root:

    unzip -o tess-train-bundle-20261110.zip        # adds data/targets/tess_train_fold*.csv, data/toi/TIC*-toi.csv, data/logs/run_tess_fold.py

In the cloud the zip is at `/mnt/project-files/analysis/new-targets/tess-20261110/tess-train-bundle-20261110.zip`.

## Run

    nohup uv run python data/logs/run_tess_fold.py N --workers 4 > data/logs/tess_foldN.log 2>&1 &

- Normal pass, then inverted pass, each retried once for failures. `--skip-inverted` runs the normal pass only.
- All sectors are stitched (`--download-all`), rows are labelled against `data/toi/`, rejected peaks are not included.
- Output: `data/extracted/tess-train-20261110/TIC<id>_<date>.csv` and `TIC<id>_inverted_<date>.csv`.
- Progress: `data/logs/tess_foldN.status` (one line per pass). Failures: search the log for `Extraction failed`.
- Resumable: stars with an output CSV are skipped, so after a cut just run the same command again.
- Use `--workers` up to the core count; each worker takes about one core and 350 MB (more for stars with many 2-minute sectors).

## Size of the job

About 80 stars and 285 sectors per fold. Rough estimate: 34 hours of single-core work per fold for both passes (4 min per 2-minute sector and pass, from the TOI-270 run), so 8 to 9 hours with 4 workers. Each target list is sorted longest star first.

## Afterwards

Zip `data/extracted/tess-train-20261110/` and the `.status` and `.log` files. Don't commit the outputs unless asked; commit format is `type: brief message`, no trailers.
