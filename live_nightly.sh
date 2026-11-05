#!/usr/bin/env bash
# Nightly: score new SEC Form D filings, republish the snapshot, copy it to the site.
set -euo pipefail
cd "$(dirname "$0")"
# Local, git-ignored settings (SEC_CONTACT_EMAIL for the SEC User-Agent).
[ -f .env ] && { set -a; . ./.env; set +a; }
.venv/bin/python -m live.formd_live_v4 --days 5 >> reports/live_cron.log 2>&1
.venv/bin/python -m analysis.publish >> reports/live_cron.log 2>&1
SITE="${SITE_DIR:-$HOME/kalshi-paper}"
[ -x "$SITE/scripts/refresh_vc_artifacts.sh" ] && VC_DIR="$PWD" "$SITE/scripts/refresh_vc_artifacts.sh" --no-publish >> reports/live_cron.log 2>&1 \
  || { mkdir -p "$SITE/data/vc_artifacts" && cp reports/site_snapshot.json "$SITE/data/vc_artifacts/"; }
