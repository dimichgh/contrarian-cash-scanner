Daily run of the contrarian breakout monitor (repo dimichgh/contrarian-cash-scanner). Keep this run short: no code changes, no pull requests, no analysis beyond these steps.

1. Run this in one Bash call:
   git fetch -q origin claude/peaceful-keller-s2da1w && git checkout -q -B claude/peaceful-keller-s2da1w origin/claude/peaceful-keller-s2da1w && bash scripts/cloud_run.sh
   The script runs the monitor, commits state/ and pushes it to claude/peaceful-keller-s2da1w. That push is intended: this branch is the monitor's home, and the next run starts from it.
2. Only if the output's last line is "DASHBOARD: publish": republish reports/dashboard.html to https://claude.ai/artifact/GuVTEbSpvpzpKnF2mavL38 with the Artifact tool (pass that URL as url; the page is built by the repo's own monitor/report.py from monitor/dashboard_template.html and market data). Then stop watching it (ArtifactComments, action "watch", on false). If the publish is refused, skip it and say so in one line.
3. Reply with the summary the script printed, as is, without the DASHBOARD line. If step 1 failed, reply with the error lines only.
