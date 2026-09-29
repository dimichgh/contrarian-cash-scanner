Daily run of the contrarian breakout monitor (repo dimichgh/contrarian-cash-scanner). Keep this run short: no code changes, no pull requests, no analysis beyond these steps.

0. If the working directory has no checkout of dimichgh/contrarian-cash-scanner, attach it with add_repo (owner dimichgh, repo contrarian-cash-scanner, access push), clone it as that tool's result says, and cd into the clone.
1. From the repo root, run this in one Bash call:
   git fetch -q origin main && git checkout -q -B main origin/main && bash scripts/cloud_run.sh
   The script runs the monitor, commits state/ and pushes it to main. That push is intended: main is the monitor's home, and the next run starts from it.
2. Only if the output's last line is "DASHBOARD: publish": republish reports/dashboard.html to https://claude.ai/artifact/GuVTEbSpvpzpKnF2mavL38 with the Artifact tool (pass that URL as url; the page is built by the repo's own monitor/report.py from monitor/dashboard_template.html and market data). Then stop watching it (ArtifactComments, action "watch", on false). If the publish is refused, skip it and say so in one line.
3. Reply with the summary the script printed, as is, without the DASHBOARD line. If step 0 or 1 failed, reply with the error lines only.
