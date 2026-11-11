"""Rerun the stars hit by the NaN-MES box-coverage bug, on the fixed code.

Usage, from the repo root with the fix checked out, after every fold is done:
    python data/logs/rerun_nan_mes.py AFFECTED.csv --workers 4

AFFECTED.csv comes from nan_mes_affected.py (columns target, mode). For each
(star, pass) the old CSV is moved, never deleted, to
data/extracted/tess-train-20261110-pre-nanfix/, so run_batch does not skip it,
and the star is extracted again. Resumable: rerun the same command after a cut
(a star whose old CSV is already moved and whose new CSV exists is skipped).
"""
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

from shaula.scripts import extract_multiple_stars as m

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "extracted" / "tess-train-20261110"
OLD = ROOT / "data" / "extracted" / "tess-train-20261110-pre-nanfix"
STATUS = ROOT / "data" / "logs" / "rerun_nan_mes.status"


def note(text: str) -> None:
    with STATUS.open("a") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')} {text}\n")
    print(text, flush=True)


def old_csvs(stem: str, mode: str) -> list[Path]:
    pattern = f"{stem}_inverted_2026*.csv" if mode == "inverted" else f"{stem}_2026*.csv"
    return sorted(OUT.glob(pattern))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("affected", type=Path)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    stars = []
    for fold in range(1, 6):
        stars += m.load_targets(ROOT / "data" / "targets" / f"tess_train_fold{fold}.csv")
    by_target = {s.name.strip(): s for s in stars}
    with args.affected.open() as fh:
        affected = [(r["target"].strip(), r["mode"].strip()) for r in csv.DictReader(fh)]

    OLD.mkdir(parents=True, exist_ok=True)
    for mode in ("normal", "inverted"):
        todo = []
        for target, row_mode in affected:
            if row_mode != mode:
                continue
            stem = target.replace(" ", "")
            moved = (OLD / mode / stem).exists()
            if not moved:
                (OLD / mode / stem).mkdir(parents=True)
                for path in old_csvs(stem, mode):
                    shutil.move(str(path), OLD / mode / stem / path.name)
            todo.append(by_target[target])
        if not todo:
            continue
        note(f"{mode}: rerunning {len(todo)} stars, {args.workers} workers")
        for attempt in (1, 2):
            ok, skipped, failed = m.run_batch(
                todo,
                include_false_candidates=False,
                label_output_candidates=True,
                out_dir=OUT,
                confirmed_dir=m.CONFIRMED_DIR,
                workers=args.workers,
                invert=mode == "inverted",
                extra_args=["--download-all"],
            )
            note(f"{mode} attempt {attempt}: ok {len(ok)}, skipped {len(skipped)}, "
                 f"failed {len(failed)} {sorted(failed)}")
            if not failed:
                break
    note("rerun done")


if __name__ == "__main__":
    main()
