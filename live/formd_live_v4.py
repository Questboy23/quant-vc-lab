"""Live scorer v4: new SEC Form D filings scored with the round-4 model (built on Form D fields
alone, so the live score uses the same information the backtest did).

Rule shown on the site = "quality in the cheaper half": the company's first Form D, raise below
the recent median ($2M), combined quality score in the top half. Holdout 2018-21: +37% paper
markup vs owning every first raise ($5M floor, CI -1%..+85%). Not investment advice.

    python -m live.formd_live_v4 [--days 5]     -> reports/live.json (+ forward_ledger_v4.csv)
"""
from __future__ import annotations

import argparse, json, math, os, re
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from live.formd_live import Fetcher, business_days, index_entries, ET, _t, _num
from analysis.fd_panel import EXCL, FUNDISH, person_key
from analysis.v4 import FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "live.json")
LEDGER = os.path.join(ROOT, "reports", "forward_ledger_v4.csv")
PIDX = os.path.join(ROOT, "data", "fd", "person_index.parquet")


def person_index():
    if os.path.exists(PIDX):
        return pd.read_parquet(PIDX)
    f = pd.read_parquet(os.path.join(ROOT, "data", "fd", "filings.parquet"),
                        columns=["ACCESSIONNUMBER", "CIK", "filing_date", "INDUSTRYGROUPTYPE", "TOTALAMOUNTSOLD"])
    p = pd.read_parquet(os.path.join(ROOT, "data", "fd", "persons.parquet"))
    p["key"] = person_key(p)
    p = p.merge(f[["ACCESSIONNUMBER", "CIK", "filing_date", "INDUSTRYGROUPTYPE"]], on="ACCESSIONNUMBER")
    p["fund"] = p.INDUSTRYGROUPTYPE == "Pooled Investment Fund"
    pc = p.groupby(["key", "CIK"]).agg(d=("filing_date", "min"), fund=("fund", "max")).reset_index()
    su = f[f.TOTALAMOUNTSOLD >= 2e7].groupby("CIK").filing_date.min()
    pc["su"] = pc.CIK.map(su)
    pc.to_parquet(PIDX)
    return pc


def parse_full(xml):
    try:
        root = ET.fromstring(xml)
    except Exception:
        return None
    iss, off = root.find("primaryIssuer"), root.find("offeringData")
    if iss is None or off is None:
        return None
    people = []
    for rp in root.findall("relatedPersonsList/relatedPersonInfo"):
        rels = [r.text or "" for r in rp.findall("relatedPersonRelationshipList/relationship")]
        people.append({"FIRSTNAME": _t(rp, "relatedPersonName/firstName"), "MIDDLENAME": _t(rp, "relatedPersonName/middleName"),
                       "LASTNAME": _t(rp, "relatedPersonName/lastName"), "is_dir": "Director" in rels,
                       "is_exec": any("Executive" in r for r in rels)})
    sc = _num(_t(off, "salesCommissionsFindersFees/salesCommissions/dollarAmount")) or 0
    ff = _num(_t(off, "salesCommissionsFindersFees/findersFees/dollarAmount")) or 0
    return {
        "entity": _t(iss, "entityName"), "state": _t(iss, "issuerAddress/stateOrCountry"), "zip": _t(iss, "issuerAddress/zipCode") or "",
        "city": _t(iss, "issuerAddress/city"), "entity_type": _t(iss, "entityType"), "jur": _t(iss, "jurisdictionOfInc") or "",
        "within5": (_t(iss, "yearOfInc/withinFiveYears") or "") == "true", "year_inc": _num(_t(iss, "yearOfInc/value")),
        "industry": _t(off, "industryGroup/industryGroupType"), "rev": _t(off, "issuerSize/revenueRange") or "",
        "amendment": (_t(off, "typeOfFiling/newOrAmendment/isAmendment") or "false") == "true",
        "more_than_year": (_t(off, "durationOfOffering/moreThanOneYear") or "false") == "true",
        "equity": (_t(off, "typesOfSecuritiesOffered/isEquityType") or "") == "true",
        "option": (_t(off, "typesOfSecuritiesOffered/isOptionToAcquireType") or "") == "true",
        "sec_acq": (_t(off, "typesOfSecuritiesOffered/isSecurityToBeAcquiredType") or "") == "true",
        "bizcomb": (_t(off, "businessCombinationTransaction/isBusinessCombinationTransaction") or "") == "true",
        "min_inv": _num(_t(off, "minimumInvestmentAccepted")) or 0,
        "offered": _num(_t(off, "offeringSalesAmounts/totalOfferingAmount")), "sold": _num(_t(off, "offeringSalesAmounts/totalAmountSold")),
        "n_inv": _num(_t(off, "investors/totalNumberAlreadyInvested")), "n_nonacc": _num(_t(off, "investors/numberNonAccreditedInvestors")) or 0,
        "exemptions": " ".join(x.text or "" for x in off.findall("federalExemptionsExclusions/item")),
        "broker": int(sc > 0 or ff > 0), "people": people, "first_sale": _t(off, "typeOfFiling/dateOfFirstSale/value"),
    }


def features(r, filed, pidx_by, peers_median, heat):
    amt = r["sold"] if (r["sold"] or 0) > 0 else r["offered"]
    z3 = float(r["zip"][:3]) if r["zip"][:3].isdigit() else -1
    x = {"log_amount": math.log10(max(amt, 1e4)), "sold_share": min(max((r["sold"] or 0) / r["offered"], 0), 1) if r["offered"] else np.nan,
         "log_investors": math.log1p(r["n_inv"]) if r["n_inv"] is not None else np.nan, "nonaccredited": int(r["n_nonacc"] > 0),
         "broker": r["broker"], "rule506c": int("06c" in r["exemptions"]), "min_invest": math.log10(r["min_inv"] + 1),
         "more_than_year": int(r["more_than_year"]), "convertible": int(not r["equity"] and (r["option"] or r["sec_acq"])),
         "rev_none": int(r["rev"] == "No Revenues"), "rev_disclosed": int(r["rev"].startswith("$") or r["rev"].startswith("Over")),
         "de_out_of_state": int("DELAWARE" in r["jur"].upper() and r["state"] != "DE"),
         "age_years": min(max(filed.year - r["year_inc"], 0), 5) if r["year_inc"] else np.nan,
         "bay_area": int(r["state"] == "CA" and 940 <= z3 <= 951), "nyc": int(r["state"] == "NY" and 100 <= z3 <= 104),
         "boston": int(r["state"] == "MA" and 21 <= z3 <= 24), "la": int(r["state"] == "CA" and 900 <= z3 <= 918),
         "tech": int(r["industry"] in ("Other Technology", "Computers", "Telecommunications")),
         "bio": int(r["industry"] in ("Biotechnology", "Pharmaceuticals")), "heat": heat,
         "size_vs_peers": math.log(max(amt, 1) / peers_median.get(r["industry"], 2e6)),
         "name_len": len(re.sub(r"(?i),?\s+(inc\.?|corp\.?|corporation)$", "", r["entity"] or "")),
         "has_ai": int(bool(re.search(r"(?i)\bai\b|\.ai\b|artificial|intelligen|neural|machine", r["entity"] or "")))}
    x["hub"] = max(x["bay_area"], x["nyc"], x["boston"])
    pf = dict(n_people=0, n_dirs=0, n_execs=0, serial_exec=0, exec_prior_scaleup=0, vc_director=0, board_pro=0,
              board_track=np.nan, board_prior_cos=0, name_ambiguous=0)
    for person in r["people"]:
        pf["n_people"] += 1; pf["n_dirs"] += int(person["is_dir"]); pf["n_execs"] += int(person["is_exec"])
        key = person_key(pd.DataFrame([person])).iloc[0]
        h = pidx_by.get(key)
        if h is None:
            continue
        if len(h) > 300:
            pf["name_ambiguous"] = 1; continue
        prior = h[h.d < filed]
        ops = prior[~prior.fund]
        if person["is_exec"] and len(ops):
            pf["serial_exec"] = 1
            if (ops.su < filed).any():
                pf["exec_prior_scaleup"] = 1
        if person["is_dir"]:
            pf["vc_director"] |= int(prior.fund.any())
            pf["board_pro"] |= int(len(ops) >= 3)
            pf["board_prior_cos"] = max(pf["board_prior_cos"], len(ops))
            old = ops[ops.d < filed - pd.Timedelta(days=730)]
            if len(old) >= 3:
                tr = float((old.su < filed).mean())
                pf["board_track"] = tr if np.isnan(pf["board_track"]) else max(pf["board_track"], tr)
    x.update(pf)
    return x, amt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", type=int, default=5); a = ap.parse_args()
    bundle = joblib.load(os.path.join(ROOT, "reports", "live_model_v4.joblib"))
    models, ref = bundle["models"], bundle["ref"]
    panel = pd.read_parquet(os.path.join(ROOT, "data", "fd", "panel.parquet"), columns=["industry", "year", "amount"])
    peers_median = panel[panel.year >= panel.year.max() - 1].groupby("industry").amount.median().to_dict()
    known_ciks = set(pd.read_parquet(os.path.join(ROOT, "data", "fd", "filings.parquet"), columns=["CIK"]).CIK)
    pidx = person_index()
    pidx_by = {k: v for k, v in pidx.groupby("key")}
    qref = np.array(ref["quality_ref"])
    pref = {y: np.array(v) for y, v in ref["prob_ref"].items()}
    f = Fetcher()
    seen, rows = 0, []
    dropped = {k: 0 for k in ("fund_like", "parse_failed", "amendment", "not_startup", "size_out_of_range", "not_first_raise")}
    for d in business_days(a.days):
        for e in index_entries(f, d):
            seen += 1
            if FUNDISH.search(e["name"]):
                dropped["fund_like"] += 1; continue
            folder = e["path"].rsplit("/", 1)[0] + "/" + e["path"].rsplit("/", 1)[1].replace("-", "").replace(".txt", "")
            xml = f.get(f"https://www.sec.gov/Archives/{folder}/primary_doc.xml")
            r = parse_full(xml) if xml else None
            if r is None:
                dropped["parse_failed"] += 1; continue
            if r["amendment"]:
                dropped["amendment"] += 1; continue
            if (r["entity_type"] != "Corporation" or r["industry"] in EXCL or not r["within5"] or r["bizcomb"]
                    or not (r["equity"] or r["option"] or r["sec_acq"]) or not re.fullmatch(r"[A-Z]{2}", r["state"] or "")
                    or FUNDISH.search(r["entity"] or "")):
                dropped["not_startup"] += 1; continue
            amt = r["sold"] if (r["sold"] or 0) > 0 else r["offered"]
            if not amt or not (1e5 <= amt <= 2e7):
                dropped["size_out_of_range"] += 1; continue
            first = e["cik"].lstrip("0") not in known_ciks
            filed = pd.Timestamp(datetime.strptime(e["filed"], "%Y%m%d"))
            x, amt = features(r, filed, pidx_by, peers_median, ref["heat_recent"])
            X = pd.DataFrame([{c: 0.0 for c in ref["columns"]}])
            for fcol in FEATS:
                v = x.get(fcol, np.nan)
                if fcol + "_na" in X:
                    X[fcol + "_na"] = float(pd.isna(v))
                X[fcol] = ref["medians"][fcol] if pd.isna(v) else float(v)
            ind = "ind_" + (r["industry"] or "")
            if ind in X:
                X[ind] = 1.0
            probs = {y: float(m.predict_proba(X[ref["columns"]])[:, 1][0]) for y, m in models.items()}
            pct = {y: float(np.searchsorted(pref[y], p) / len(pref[y])) for y, p in probs.items()}
            quality = float(np.searchsorted(qref, np.mean(list(pct.values()))) / len(qref))
            cheap = amt < ref["cheap_cutoff_amount"]
            rows.append({"filed": e["filed"], "cik": e["cik"], "name": r["entity"], "city": r["city"], "state": r["state"],
                         "industry": r["industry"], "amount": amt, "first_raise": first, "quality_pct": round(quality, 3),
                         "p_growth20": round(probs["growth20"], 4), "p_exit": round(probs["exit"], 4),
                         "cheap_half": bool(cheap), "rule_pick": bool(first and cheap and quality >= 0.5),
                         "signals": {k: x[k] for k in ("vc_director", "serial_exec", "exec_prior_scaleup", "board_pro", "hub", "de_out_of_state", "broker", "rule506c")},
                         "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={e['cik']}&type=D"})
    rows.sort(key=lambda r: (-int(r["rule_pick"]), -r["quality_pct"]))
    for r in rows:
        if not r["first_raise"]:
            dropped["not_first_raise"] += 1
    out = {"generated": datetime.now(timezone.utc).isoformat(), "model": "round-4 Form D model", "days": [str(d) for d in business_days(a.days)],
           "filings_seen": seen, "scored": len(rows), "rule_picks": sum(r["rule_pick"] for r in rows), "dropped": dropped,
           "fetch": {"calls": f.calls, "failed": f.fails, "throttled_429": f.throttled},
           "rule": "first Form D, raise below $%.1fM, quality score in the top half" % (ref["cheap_cutoff_amount"] / 1e6), "rows": rows[:200]}
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    new = not os.path.exists(LEDGER)
    have = set() if new else {tuple(l.split(",")[:2]) for l in open(LEDGER).read().splitlines()[1:]}
    with open(LEDGER, "a") as fh:
        if new:
            fh.write("filed,cik,name,state,industry,amount,first_raise,quality_pct,p_growth20,p_exit,rule_pick,logged_at\n")
        for r in rows:
            if (r["filed"], r["cik"]) in have:
                continue
            nm = str(r["name"]).replace(",", " ")
            fh.write(f'{r["filed"]},{r["cik"]},{nm},{r["state"]},{str(r["industry"]).replace(",", " ")},{r["amount"]},'
                     f'{r["first_raise"]},{r["quality_pct"]},{r["p_growth20"]},{r["p_exit"]},{r["rule_pick"]},{datetime.now(timezone.utc).date()}\n')
    print(f"seen {seen}, scored {len(rows)}, rule picks {out['rule_picks']}, fetch {f.calls}/{f.fails} failed/{f.throttled} 429s")


if __name__ == "__main__":
    main()
