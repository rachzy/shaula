#!/usr/bin/env bash
# Normal and inverted passes over the 2026-10-29 pool targets (unary/binary x few/many KOIs).
# Usage: data/logs/run_pools_20261029.sh
set -u
F=pools
cd "$(dirname "$0")/../.."
ARGS="--targets data/targets/targets_pools_20261029.csv --no-include-false-candidates --label-output-candidates --workers 4"
run() { uv run python -m shaula.scripts.extract_multiple_stars $ARGS "$@"; }
LOG=data/logs/pools_20261029
echo "pools normal start $(date -Is) on $(git rev-parse --short HEAD)" >> $LOG.status
run >> $LOG.normal.log 2>&1; echo "normal exit $? $(date -Is)" >> $LOG.status
run >> $LOG.normal.log 2>&1; echo "normal retry-failed exit $? $(date -Is)" >> $LOG.status
run --invert >> $LOG.inverted.log 2>&1; echo "inverted exit $? $(date -Is)" >> $LOG.status
run --invert >> $LOG.inverted.log 2>&1; echo "inverted retry-failed exit $? $(date -Is)" >> $LOG.status
echo "pools done $(date -Is)" >> $LOG.status
