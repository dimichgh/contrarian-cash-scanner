#!/usr/bin/env bash
# Entry point for scheduled runs: run the monitor, then commit and push its state so the next run
# (a fresh container) picks up where this one left off.
#   bash scripts/cloud_run.sh            # the run after the close; extra args go to `python -m monitor daily`
#   bash scripts/cloud_run.sh --preview  # before the close: commits only reports/preview.*
#   MONITOR_BRANCH=main bash scripts/cloud_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BRANCH="${MONITOR_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

if [ ! -x .venv/bin/python ]; then
  { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; } > /dev/null 2>&1 \
    || { echo "MONITOR FAILED: could not install dependencies"; exit 1; }
fi
mkdir -p reports
MODE=daily
if [ "${1:-}" = "--preview" ]; then MODE=preview; shift; fi
if ! .venv/bin/python -m monitor "$MODE" "$@" > reports/run.log 2>&1; then
  echo "MONITOR FAILED:"
  tail -n 25 reports/run.log
  exit 1
fi

if [ "$MODE" = preview ]; then
  git add reports/preview.md reports/preview.html
  msg="monitor: pre-close preview $(TZ=America/New_York date '+%Y-%m-%d %H:%M') ET"
else
  git add state reports/summary.md reports/*.html
  msg="monitor: close of $(.venv/bin/python -c 'import json; print(json.load(open("state/last_run.json"))["as_of"])')"
fi
if ! git diff --cached --quiet; then
  git commit -q -m "$msg"
  pushed=""
  for delay in 2 4 8 16; do
    if git pull -q --rebase origin "$BRANCH" && git push -q origin "HEAD:${BRANCH}"; then
      pushed=1
      break
    fi
    git rebase --abort > /dev/null 2>&1
    sleep "$delay"
  done
  [ -n "$pushed" ] || echo "PUSH FAILED: state was not saved to ${BRANCH}; the next run rebuilds signals from prices"
fi
if [ "$MODE" = preview ]; then cat reports/preview.md; else cat reports/summary.md; fi
