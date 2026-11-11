"""List stars whose run hit the NaN-MES box-coverage bug (read-only).

python -I nan_mes_affected.py OUT.csv LOG [LOG ...]

A refined candidate is affected when its log block prints
"SES/MES computed: MES=nan" after a "Feature cadence" line (statistics on the
29.4 min bins) and its final T14 is 3.75 to 4.0 bins. Writes one row per
(star, pass) with the affected periods.
"""
import csv
import re
import sys

BIN_HOURS = 29.4244 / 60.0
LOW, HIGH = 3.74, 4.01

out_path, logs = sys.argv[1], sys.argv[2:]
hits = {}
for log in logs:
    star = mode = last = None
    binned = False
    for line in open(log, errors="ignore"):
        m = re.match(r"=== \[\d+/\d+\] (TIC \d+)", line)
        if m:
            star, mode, last = m.group(1), "normal", None
            continue
        if star is None:
            continue
        if line.startswith("Light curve inverted"):
            mode = "inverted"
        m = re.match(r"Final result: P=([\d.]+) days, T14=([\d.]+) days", line)
        if m:
            last = (float(m.group(1)), float(m.group(2)) * 24.0)
            binned = False
        if line.startswith("Feature cadence"):
            binned = True
        if "SES/MES computed: MES=nan" in line and last and binned:
            bins = last[1] / BIN_HOURS
            if LOW <= bins <= HIGH:
                hits.setdefault((star, mode), set()).add(round(last[0], 4))

with open(out_path, "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["target", "mode", "nan_periods"])
    for (star, mode), periods in sorted(hits.items()):
        writer.writerow([star, mode, " ".join(str(p) for p in sorted(periods))])
print(f"{len(hits)} (star, pass) pairs, "
      f"{len({s for s, _ in hits})} stars -> {out_path}")
