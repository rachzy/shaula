#!/usr/bin/env bash
# Normal and inverted passes over one fold of the new training targets (new_targets_v2).
# Usage: data/logs/run_new_fold.sh 1
set -u
F=$1
cd "$(dirname "$0")/../.."
ARGS="--targets data/targets/targets_new_fold$F.csv --no-include-false-candidates --label-output-candidates --workers 4"
run() { uv run python -m shaula.scripts.extract_multiple_stars $ARGS "$@"; }
LOG=data/logs/new_fold$F
echo "fold $F normal start $(date -Is) on $(git rev-parse --short HEAD)" >> $LOG.status
run >> $LOG.normal.log 2>&1; echo "normal exit $? $(date -Is)" >> $LOG.status
run >> $LOG.normal.log 2>&1; echo "normal retry-failed exit $? $(date -Is)" >> $LOG.status
run --invert >> $LOG.inverted.log 2>&1; echo "inverted exit $? $(date -Is)" >> $LOG.status
run --invert >> $LOG.inverted.log 2>&1; echo "inverted retry-failed exit $? $(date -Is)" >> $LOG.status
echo "fold $F done $(date -Is)" >> $LOG.status
