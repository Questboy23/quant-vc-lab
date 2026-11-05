#!/usr/bin/env bash
# Fetch the raw inputs. Crunchbase 2015 export is non-commercial; SEC data is public.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SEC_CONTACT_EMAIL:?SEC_CONTACT_EMAIL is not set. The SEC requires a contact email in the User-Agent. Set it first: export SEC_CONTACT_EMAIL=you@example.com}"
UA="QuantVCLab research $SEC_CONTACT_EMAIL"
mkdir -p data/raw data/formd
[ -d data/raw/crunchbase-data ] || git clone --depth 1 https://github.com/notpeter/crunchbase-data.git data/raw/crunchbase-data
curl -s -A "$UA" "https://www.sec.gov/data-research/sec-markets-data/form-d-data-sets" \
  | grep -o 'href="[^"]*20\(08\|09\|10\|11\|12\)q[1-4][^"]*\.zip"' | sed 's/href="//;s/"$//' | sort -u > data/formd/urls.txt
while read -r u; do
  f="data/formd/$(basename "$u")"; [ -s "$f" ] || { curl -s -A "$UA" -o "$f" "https://www.sec.gov$u"; sleep 0.5; }
done < data/formd/urls.txt
echo "fetched $(ls data/formd/*.zip | wc -l) Form D quarters"
