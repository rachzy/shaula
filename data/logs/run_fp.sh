#!/usr/bin/env bash
# Local share (rows 1-204 and 285) on claude/project-thread-sglr7m @ 14c4111.
cd /home/henrique/Documents/GitHub/shaula
ARGS="--targets data/targets/targets_local.csv --no-include-false-candidates --label-output-candidates --workers 4"
run() { uv run python -m shaula.scripts.extract_multiple_stars $ARGS "$@"; }
echo "normal start (rows 1-204+285, 4 workers) $(date -Is)" >> data/logs/status
echo "=== restart $(date -Is) on $(git rev-parse --short HEAD) ===" >> data/logs/normal.log
run >> data/logs/normal.log 2>&1; echo "normal exit $? $(date -Is)" >> data/logs/status
run >> data/logs/normal.log 2>&1; echo "normal retry-failed exit $? $(date -Is)" >> data/logs/status
run --invert > data/logs/inverted.log 2>&1; echo "inverted exit $? $(date -Is)" >> data/logs/status
run --invert >> data/logs/inverted.log 2>&1; echo "inverted retry-failed exit $? $(date -Is)" >> data/logs/status
