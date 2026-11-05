"""Nightly live scorer: pull the last N business days of SEC Form D filings, keep plausible
venture-style operating-company raises, and score them with the REDUCED model (only the
surviving features that a Form D actually reveals). Writes reports/live.json.

    python -m live.formd_live [--days 5]

SEC fair access: descriptive User-Agent with a contact address, <= 5 requests/second here
(limit is 10). Every fetch failure is counted and surfaced, never silently skipped.

What the score is NOT: an investment recommendation. Form D reveals none of the investor-side
features (syndicate, accelerator, investor experience) that carried most of the historical
signal, so the live score is a weak, partial version of the backtested one and the page says so."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from defusedxml import ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _ua() -> str:
    """SEC EDGAR asks for a contact email in the User-Agent. Set SEC_CONTACT_EMAIL before running."""
    email = os.environ.get("SEC_CONTACT_EMAIL", "").strip()
    if not email:
        raise RuntimeError("SEC_CONTACT_EMAIL is not set. The SEC requires a contact email in the "
                           "User-Agent of every request. Set it first: export SEC_CONTACT_EMAIL=you@example.com")
    return f"QuantVCLab research (reedsmoneymaker23.com/vc) {email}"


OUT = os.path.join(ROOT, "reports", "live.json")
MODEL = os.path.join(ROOT, "reports", "live_model.json")
SPACING = 0.2  # seconds between requests -> 5 req/s

# Industry groups that are not venture-style operating companies.
EXCLUDE_GROUPS = {"Pooled Investment Fund", "REITS and Finance", "Commercial", "Residential",
                  "Other Real Estate", "Oil and Gas", "Coal Mining", "Mining", "Commercial Banking",
                  "Insurance", "Investing", "Investment Banking", "Other Banking and Financial Services",
                  "Lodging and Conventions", "Restaurants", "Agriculture", "Construction",
                  "Airlines and Airports", "Tourism and Travel Services", "Other Travel",
                  "Electric Utilities", "Other Energy", "Environmental Services", "Retailing",
                  "Business Services"}
FUNDISH = re.compile(r"\b(fund|lp|l\.p\.|partners|capital|holdings|trust|reit|properties|realty|"
                     r"spv|series|investors|drilling|program|album|production|productions)\b", re.I)


class Fetcher:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": _ua(), "Accept-Encoding": "gzip, deflate"})
        self.last = 0.0
        self.calls = self.fails = self.throttled = 0

    def get(self, url):
        wait = SPACING - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        self.calls += 1
        for attempt in range(3):
            try:
                r = self.s.get(url, timeout=20)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if r.status_code == 429:
                self.throttled += 1; time.sleep(5 * (attempt + 1)); continue
            if r.status_code == 200:
                return r.text
            break
        self.fails += 1
        return None


def business_days(n):
    d = date.today() - timedelta(days=1)
    out = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def index_entries(f: Fetcher, d: date):
    q = (d.month - 1) // 3 + 1
    txt = f.get(f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{q}/form.{d:%Y%m%d}.idx")
    if not txt:
        return []
    rows = []
    for line in txt.splitlines():
        if not line.startswith("D "):
            continue
        m = re.match(r"^D\s+(.*?)\s{2,}(\d+)\s+(\d{8})\s+(\S+)", line)
        if m:
            rows.append({"name": m.group(1).strip(), "cik": m.group(2), "filed": m.group(3), "path": m.group(4)})
    return rows


def _t(node, path):
    x = node.find(path)
    return x.text.strip() if x is not None and x.text else None


def parse(xml: str) -> dict | None:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    iss = root.find("primaryIssuer")
    off = root.find("offeringData")
    if iss is None or off is None:
        return None
    sale = _t(off, "typeOfFiling/dateOfFirstSale/value")
    return {
        "entity": _t(iss, "entityName"), "city": _t(iss, "issuerAddress/city"),
        "state": _t(iss, "issuerAddress/stateOrCountry"), "zip": _t(iss, "issuerAddress/zipCode"),
        "entity_type": _t(iss, "entityType"), "year_inc": _t(iss, "yearOfInc/value"),
        "industry": _t(off, "industryGroup/industryGroupType"),
        "amendment": (_t(off, "typeOfFiling/newOrAmendment/isAmendment") or "false") == "true",
        "first_sale": sale,
        "equity": (_t(off, "typesOfSecuritiesOffered/isEquityType") or "false") == "true",
        "offering": _num(_t(off, "offeringSalesAmounts/totalOfferingAmount")),
        "sold": _num(_t(off, "offeringSalesAmounts/totalAmountSold")),
        "n_investors": _num(_t(off, "investors/totalNumberAlreadyInvested")),
        "exemptions": [x.text for x in off.findall("federalExemptionsExclusions/item")],
    }


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def bay_area(rec):
    z = (rec.get("zip") or "")[:3]
    return rec.get("state") == "CA" and z.isdigit() and 940 <= int(z) <= 951


def score(rec, model, words):
    name = rec["entity"] or ""
    clean = re.sub(r",?\s+(inc\.?|llc|corp\.?|corporation|co\.?|ltd\.?)$", "", name, flags=re.I).strip()
    norm = re.sub(r"[^a-z0-9]", "", clean.lower())
    raised = max(rec["sold"] or rec["offering"] or 0, 1e4)
    x = {"sf": float(bay_area(rec)), "name_real_word": float(clean.lower() in words),
         "name_len": float(len(norm)), "log_raised": math.log10(raised)}
    z = model["intercept"] + sum(model["coef"][k] * (x[k] - model["mean"][k]) / model["sd"][k] for k in model["coef"])
    prob = 1 / (1 + math.exp(-z))
    dil = 0.20 if raised < 2.5e6 else 0.30
    post = max(raised / dil, model["floor"])
    return {**x, "p_bigwin": prob, "post_money_est": post, "score": prob / post * 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    a = ap.parse_args()
    model = json.load(open(MODEL))
    words = {w.strip().lower() for w in open("/usr/share/dict/words")} if os.path.exists("/usr/share/dict/words") else set()
    f = Fetcher()
    seen, kept, dropped = 0, [], {"amendment": 0, "industry": 0, "fund_like": 0, "not_equity": 0,
                                  "size_out_of_range": 0, "non_us": 0, "parse_failed": 0,
                                  "not_corporation": 0}
    for d in business_days(a.days):
        for e in index_entries(f, d):
            seen += 1
            if FUNDISH.search(e["name"]):
                dropped["fund_like"] += 1; continue
            folder = e["path"].rsplit("/", 1)[0] + "/" + e["path"].rsplit("/", 1)[1].replace("-", "").replace(".txt", "")
            xml = f.get(f"https://www.sec.gov/Archives/{folder}/primary_doc.xml")
            rec = parse(xml) if xml else None
            if rec is None:
                dropped["parse_failed"] += 1; continue
            if rec["amendment"]:
                dropped["amendment"] += 1; continue
            if rec["entity_type"] != "Corporation" or FUNDISH.search(rec["entity"] or ""):
                dropped["not_corporation"] += 1; continue
            if rec["industry"] in EXCLUDE_GROUPS:
                dropped["industry"] += 1; continue
            if not rec["equity"]:
                dropped["not_equity"] += 1; continue
            amt = rec["sold"] or rec["offering"] or 0
            if not (1e5 <= amt <= 3e7):
                dropped["size_out_of_range"] += 1; continue
            if not rec["state"] or not re.fullmatch(r"[A-Z]{2}", rec["state"]) or rec["state"] in {"X1"}:
                dropped["non_us"] += 1; continue
            s = score(rec, model, words)
            kept.append({"filed": e["filed"], "cik": e["cik"], "name": rec["entity"], "city": rec["city"],
                         "state": rec["state"], "industry": rec["industry"], "amount_sold": rec["sold"],
                         "offering": rec["offering"], "first_sale": rec["first_sale"],
                         "year_inc": rec["year_inc"], "n_investors": rec["n_investors"],
                         "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={e['cik']}&type=D",
                         **{k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()}})
    kept.sort(key=lambda r: -r["score"])
    for i, r in enumerate(kept):
        r["rank_pct"] = round(1 - i / max(len(kept), 1), 4)
    out = {"generated": datetime.now(timezone.utc).isoformat(), "days": [str(d) for d in business_days(a.days)],
           "filings_seen": seen, "scored": len(kept), "dropped": dropped,
           "fetch": {"calls": f.calls, "failed": f.fails, "throttled_429": f.throttled},
           "model": {k: model[k] for k in ("features", "note", "train_auc")},
           "rows": kept[:200]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # Forward test: append every scored filing, once, with its score as of today. In ~2031+ these
    # rows get matched to real outcomes - the only truly out-of-sample test left.
    led = os.path.join(ROOT, "reports", "forward_ledger.csv")
    seen_keys = set()
    if os.path.exists(led):
        seen_keys = {tuple(l.split(",")[:2]) for l in open(led).read().splitlines()[1:]}
    with open(led, "a") as fh:
        if fh.tell() == 0:
            fh.write("filed,cik,name,state,industry,amount,p_bigwin,score,rank_pct,logged_at\n")
        for r in kept:
            if (r["filed"], r["cik"]) in seen_keys:
                continue
            nm = str(r["name"]).replace(",", " ")
            fh.write(f'{r["filed"]},{r["cik"]},{nm},{r["state"]},{str(r["industry"]).replace(",", " ")},'
                     f'{r["amount_sold"] or r["offering"]},{r["p_bigwin"]},{r["score"]},{r["rank_pct"]},{datetime.now(timezone.utc).date()}\n')
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"seen {seen}, scored {len(kept)}, fetch {f.calls} calls / {f.fails} failed / {f.throttled} 429s")


if __name__ == "__main__":
    main()
