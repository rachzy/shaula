#!/usr/bin/env bash
S=/home/henrique/Documents/GitHub/shaula; cd $S
echo "time $(date +%H:%M) | $(tail -1 data/logs/status)"
python3 - <<'PY'
import csv,glob,re
S="/home/henrique/Documents/GitHub/shaula"
local=[r["target"].replace("KIC ","") for r in csv.DictReader(open(f"{S}/data/targets/targets_local.csv"))]
files=glob.glob(f"{S}/data/extracted/KIC*.csv")
norm={re.match(r".*/KIC(\d+)_",f).group(1) for f in files if "_inverted_" not in f}
inv={re.match(r".*/KIC(\d+)_",f).group(1) for f in files if "_inverted_" in f}
print(f"normal {sum(k in norm for k in local)}/{len(local)} | inverted {sum(k in inv for k in local)}/{len(local)}")
PY
R=$(grep -n "=== restart" data/logs/normal.log | tail -1 | cut -d: -f1)
echo "failed since restart: $(tail -n +$R data/logs/normal.log | grep -c -- '-> FAILED') | workers busy: $(ps -eo pcpu,args | awk '/[f]orkserver.*extract_multiple/ && $1>20' | wc -l)/4 | inhibitor: $(systemd-inhibit --list | grep -c shaula-fp) | AC: $(cat /sys/class/power_supply/ACAD/online) | $(sensors | awk '/Package/{print $4}')"
