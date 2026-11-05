#!/usr/bin/env bash
# EDGAR full-index form.gz, 2009Q1-2026Q3, filtered to IPO/public-company forms. <=3 req/s.
set -u
cd "$(dirname "$0")/.."
: "${SEC_CONTACT_EMAIL:?SEC_CONTACT_EMAIL is not set. The SEC requires a contact email in the User-Agent. Set it first: export SEC_CONTACT_EMAIL=you@example.com}"
UA="QuantVCLab research $SEC_CONTACT_EMAIL"
for y in $(seq 2009 2026); do for q in 1 2 3 4; do
  [ "$y" = 2026 ] && [ "$q" = 4 ] && continue
  out="data/edgar_idx/${y}Q${q}.txt"; [ -s "$out" ] && continue
  curl -s -A "$UA" "https://www.sec.gov/Archives/edgar/full-index/$y/QTR$q/form.gz" | gunzip -c 2>/dev/null \
    | grep -E '^(S-1 |F-1 |424B4 |424B1 |10-K |20-F |S-4 |25-NSE |15-12G |DEFM14A |SC TO-T |D  )' > "$out"
  sleep 0.4
done; done
echo DONE
