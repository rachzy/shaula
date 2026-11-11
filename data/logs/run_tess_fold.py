"""Normal and inverted passes over one fold of the TESS training targets.

Usage, from the repo root on feature/tess-lightcurves:
    uv run python data/logs/run_tess_fold.py 1 --workers 4

Resumable: a star with a CSV in the output folder is skipped, so rerun the same
command after a cut. Each pass runs twice, the second time only for failures.
"""
import argparse
from datetime import datetime
from pathlib import Path

from shaula.scripts import extract_multiple_stars as m

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "extracted" / "tess-train-20261110"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fold", type=int, choices=range(1, 6))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-inverted", action="store_true")
    args = p.parse_args()
    stars = m.load_targets(ROOT / "data" / "targets" / f"tess_train_fold{args.fold}.csv")
    status = ROOT / "data" / "logs" / f"tess_fold{args.fold}.status"

    def note(text: str) -> None:
        with status.open("a") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")
        print(text, flush=True)

    note(f"fold {args.fold} start, {len(stars)} stars, {args.workers} workers")
    for invert in ([False] if args.skip_inverted else [False, True]):
        name = "inverted" if invert else "normal"
        for attempt in (1, 2):
            ok, skipped, failed = m.run_batch(
                stars,
                include_false_candidates=False,
                label_output_candidates=True,
                out_dir=OUT,
                confirmed_dir=m.CONFIRMED_DIR,
                workers=args.workers,
                invert=invert,
                extra_args=["--download-all"],
            )
            note(f"{name} attempt {attempt}: ok {len(ok)}, skipped {len(skipped)}, failed {len(failed)} {sorted(failed)}")
            if not failed:
                break
    note(f"fold {args.fold} done")


if __name__ == "__main__":
    main()
