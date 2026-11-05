"""Round 4 universe: every young US startup's FIRST Form D, 2010-2025, from SEC filings alone.

Entry   = the company's first original Form D (CIK not seen before), filed 2010-2025, by a
          Corporation incorporated within 5 years, in an operating industry, raising equity or
          SAFE/convertible-style securities, $100K-$20M sold (or offered if nothing sold yet).
Outcome = what happened in the 4.5 years after entry (cohorts through 2021 have full windows;
          data ends 2026Q2):
            next_raise   any later Form D offering
            growth20     a later offering that sold >= $20M   (Series B+ scale)
            big75        a later offering that sold >= $75M   (late-stage / unicorn path)
            ipo          the same CIK filed an IPO prospectus (424B1/424B4) or S-1/F-1 then 10-K/20-F
            markup       PAPER multiple proxy: (largest later raise / 15%) / entry post-money,
                         diluted 15% per offering in between; IPO -> at least $500M value.
                         Not realised cash. Outlier = markup >= 20x.
Features are point-in-time: people (directors, officers) are matched by name across all earlier
Form D filings, including the funds they manage."""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDD = os.path.join(ROOT, "data", "fd")
OUT = os.path.join(FDD, "panel.parquet")
WINDOW = pd.DateOffset(months=54)
EXCL = {"Pooled Investment Fund", "Commercial", "Other Real Estate", "Residential", "Investing", "REITS and Finance",
        "Oil and Gas", "Other Banking and Financial Services", "Insurance", "Other Energy", "Retailing", "Restaurants",
        "Commercial Banking", "Agriculture", "Construction", "Lodging and Conventions", "Investment Banking",
        "Other Travel", "Tourism and Travel Services", "Electric Utilities", "Coal Mining", "Airlines and Airports",
        "Health Insurance", "Mining", "Business Services"}
FUNDISH = re.compile(r"\b(fund|lp|l\.p\.|partners|capital|holdings|trust|reit|properties|realty|spv|series|"
                     r"investors|drilling|program|bank|bancorp|acquisition corp)\b", re.I)
BAY = lambda z: z.str[:3].str.isdigit() & z.str[:3].astype(float, errors="ignore").between(940, 951) if False else None


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def offerings(f: pd.DataFrame) -> pd.DataFrame:
    """Collapse each original D + its amendments into one offering: amount = max sold seen."""
    f = f.sort_values(["CIK", "filing_date"])
    f["is_orig"] = f.SUBMISSIONTYPE == "D"
    f["off_id"] = f.groupby("CIK").is_orig.cumsum()
    g = f[f.off_id > 0].groupby(["CIK", "off_id"])
    o = g.agg(start=("filing_date", "first"), sold=("TOTALAMOUNTSOLD", "max"), offered=("TOTALOFFERINGAMOUNT", "max"),
              acc=("ACCESSIONNUMBER", "first")).reset_index()
    o["amount"] = o.sold.where(o.sold > 0, o.offered)
    return o


def ipo_dates() -> pd.Series:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "edgar_idx", "*.txt"))):
        for line in open(path, encoding="latin-1"):
            form = line[:12].strip()
            m = re.search(r"\s(\d{1,10})\s+(\d{4}-\d{2}-\d{2})\s", line)
            if m:
                rows.append((form, m.group(1), m.group(2)))
    e = pd.DataFrame(rows, columns=["form", "cik", "date"])
    e["date"] = pd.to_datetime(e.date, errors="coerce")
    pro = e[e.form.isin(["424B4", "424B1"])].groupby("cik").date.min()
    s1 = e[e.form.isin(["S-1", "F-1"])].groupby("cik").date.min()
    rep = e[e.form.isin(["10-K", "20-F"])].groupby("cik").date.min()
    both = pd.concat([s1, rep], axis=1, keys=["s1", "rep"]).dropna()
    s1rep = both[both.rep > both.s1].s1
    # Underwritten IPOs only: S-1 + 10-K alone is mostly tiny self-registrations (shells), not VC exits.
    return pro


def person_key(p: pd.DataFrame) -> pd.Series:
    fn = p.FIRSTNAME.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    ln = p.LASTNAME.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)
    mi = p.MIDDLENAME.fillna("").str.lower().str[:1]
    return fn + "_" + mi + "_" + ln


def build() -> pd.DataFrame:
    f = pd.read_parquet(os.path.join(FDD, "filings.parquet"))
    f = f.dropna(subset=["filing_date", "CIK"])
    o = offerings(f)
    first = f.sort_values("filing_date").drop_duplicates("CIK")          # first filing of any kind per CIK
    e = first[first.SUBMISSIONTYPE == "D"].copy()
    e["year"] = e.filing_date.dt.year
    e = e[(e.year >= 2010) & (e.year <= 2025)]
    e = e[~e.INDUSTRYGROUPTYPE.isin(EXCL) & (e.ENTITYTYPE == "Corporation")]
    e = e[~e.ENTITYNAME.fillna("").str.contains(FUNDISH)]
    e = e[e.STATEORCOUNTRY.fillna("").str.fullmatch(r"[A-Z]{2}") & ~e.STATEORCOUNTRY.isin(["X1"])]
    yi = e.YEAROFINC_VALUE_ENTERED
    e = e[(e.YEAROFINC_TIMESPAN_CHOICE == "withinFiveYears") & ((yi.isna()) | (yi >= e.year - 5))]
    sec_ok = (e.ISEQUITYTYPE == "true") | (e.ISOPTIONTOACQUIRETYPE == "true") | (e.ISSECURITYTOBEACQUIREDTYPE == "true")
    e = e[sec_ok & (e.ISBUSINESSCOMBINATIONTRANS != "true")]
    e["amount"] = e.TOTALAMOUNTSOLD.where(e.TOTALAMOUNTSOLD > 0, e.TOTALOFFERINGAMOUNT)
    e = e[e.amount.between(1e5, 2e7)].set_index("CIK")
    print("universe", len(e), flush=True)
    e = e[~e.index.duplicated()]

    # ---------------- outcomes ----------------
    o = o[o.CIK.isin(e.index)]
    o = o.merge(e[["filing_date"]].rename(columns={"filing_date": "entry"}), left_on="CIK", right_index=True)
    later = o[(o.start > o.entry + pd.Timedelta(days=90)) & (o.start <= o.entry + WINDOW)].copy()
    later["amount"] = later.amount.fillna(0)
    g = later.groupby("CIK")
    e["n_later"] = g.size().reindex(e.index).fillna(0)
    e["max_later"] = g.amount.max().reindex(e.index)
    e["next_raise"] = (e.n_later > 0).astype(int)
    e["growth20"] = (e.max_later >= 2e7).astype(int)
    e["big75"] = (e.max_later >= 7.5e7).astype(int)
    nxt = later.sort_values("start").groupby("CIK").head(1).set_index("CIK")
    e["next_date"] = nxt.start.reindex(e.index)
    e["next_amount"] = nxt.amount.reindex(e.index)
    e["years_to_next"] = (e.next_date - e.filing_date).dt.days / 365.25
    # offerings between entry and the largest later raise (dilution)
    idx_max = later.loc[later.groupby("CIK").amount.idxmax()].set_index("CIK")
    k = later.merge(idx_max[["start"]].rename(columns={"start": "tmax"}), left_on="CIK", right_index=True)
    e["k_before_max"] = k[k.start <= k.tmax].groupby("CIK").size().reindex(e.index).fillna(0)
    ipo = ipo_dates()
    e["ipo_date"] = ipo.reindex(e.index)
    e["ipo"] = (e.ipo_date.notna() & (e.ipo_date > e.filing_date) & (e.ipo_date <= e.filing_date + WINDOW)).astype(int)
    e["ipo_ever"] = (e.ipo_date.notna() & (e.ipo_date > e.filing_date)).astype(int)
    post_entry = np.maximum(e.amount / 0.20, 1e6)
    later_post = e.max_later / 0.15
    mk = (later_post * 0.85 ** np.minimum(e.k_before_max, 6) / post_entry).fillna(0)
    mk = np.where(e.ipo == 1, np.maximum(mk, 5e8 * 0.85 ** np.minimum(e.n_later, 6) / post_entry), mk)
    e["markup"] = mk
    # name-matched exits/valuations from Crunchbase-2023 (M&A / IPO status), YC (Acquired/Public),
    # unicorn lists (2022, 2025). Matching is by normalised name; it is a lookup, timing unknown.
    from analysis.fd_enrich import enrich
    e = e.join(enrich(e))
    uni_mk = (e.unicorn_val * 0.85 ** np.minimum(e.n_later, 6) / post_entry).fillna(0)
    e["markup"] = np.maximum(e.markup, uni_mk)
    e["outlier"] = (e.markup >= 20).astype(int)
    e["window_complete"] = (e.filing_date + WINDOW <= pd.Timestamp("2026-06-30")).astype(int)

    # ---------------- entry features ----------------
    e["log_amount"] = np.log10(e.amount)
    e["sold_share"] = (e.TOTALAMOUNTSOLD / e.TOTALOFFERINGAMOUNT).clip(0, 1)
    e["n_investors"] = e.TOTALNUMBERALREADYINVESTED
    e["log_investors"] = np.log1p(e.n_investors)
    e["nonaccredited"] = (_num(e.NUMBERNONACCREDITEDINVESTORS).fillna(0) > 0).astype(int)
    e["broker"] = ((e.SALESCOMM_DOLLARAMOUNT.fillna(0) > 0) | (e.FINDERSFEE_DOLLARAMOUNT.fillna(0) > 0)).astype(int)
    ex = e.FEDERALEXEMPTIONS_ITEMS_LIST.fillna("")
    e["rule506c"] = ex.str.contains("06c").astype(int)
    e["min_invest"] = np.log10(e.MINIMUMINVESTMENTACCEPTED.fillna(0) + 1)
    e["more_than_year"] = (e.MORETHANONEYEAR == "true").astype(int)
    e["convertible"] = ((e.ISEQUITYTYPE != "true") & ((e.ISOPTIONTOACQUIRETYPE == "true") | (e.ISSECURITYTOBEACQUIREDTYPE == "true"))).astype(int)
    rr = e.REVENUERANGE.fillna("")
    e["rev_none"] = (rr == "No Revenues").astype(int)
    e["rev_disclosed"] = rr.str.startswith("$").astype(int) | rr.str.startswith("Over").astype(int)
    e["delaware"] = e.JURISDICTIONOFINC.fillna("").str.upper().str.contains("DELAWARE").astype(int)
    e["de_out_of_state"] = ((e.delaware == 1) & (e.STATEORCOUNTRY != "DE")).astype(int)
    e["age_years"] = (e.year - e.YEAROFINC_VALUE_ENTERED).clip(0, 5)
    z3 = pd.to_numeric(e.ZIPCODE.fillna("").str[:3], errors="coerce")
    e["bay_area"] = ((e.STATEORCOUNTRY == "CA") & z3.between(940, 951)).astype(int)
    e["nyc"] = ((e.STATEORCOUNTRY == "NY") & z3.between(100, 104)).astype(int)
    e["boston"] = ((e.STATEORCOUNTRY == "MA") & z3.between(21, 24)).astype(int)
    e["la"] = ((e.STATEORCOUNTRY == "CA") & z3.between(900, 918)).astype(int)
    e["hub"] = e[["bay_area", "nyc", "boston"]].max(axis=1)
    e["industry"] = e.INDUSTRYGROUPTYPE
    e["tech"] = e.industry.isin(["Other Technology", "Computers", "Telecommunications"]).astype(int)
    e["bio"] = e.industry.isin(["Biotechnology", "Pharmaceuticals"]).astype(int)
    e["quarter"] = e.filing_date.dt.to_period("Q").astype(str)
    # market heat: first-time startup filings in the quarter vs trailing 8-quarter mean
    qc = e.groupby("quarter").size()
    qs = qc.rolling(8, min_periods=4).mean().shift(1)
    e["heat"] = np.log(e.quarter.map(qc) / e.quarter.map(qs))
    ind_med = e.groupby(["industry", "year"]).amount.transform("median")
    e["size_vs_peers"] = np.log(e.amount / ind_med)
    e["name_len"] = e.ENTITYNAME.fillna("").str.replace(r"(?i),?\s+(inc\.?|corp\.?|corporation)$", "", regex=True).str.len()
    e["has_ai"] = e.ENTITYNAME.fillna("").str.contains(r"(?i)\bai\b|\.ai\b|artificial|intelligen|neural|machine").astype(int)
    return e


def person_features(e: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time people features from RELATEDPERSONS across ALL filings (funds included)."""
    f = pd.read_parquet(os.path.join(FDD, "filings.parquet"), columns=["ACCESSIONNUMBER", "CIK", "filing_date", "INDUSTRYGROUPTYPE"])
    p = pd.read_parquet(os.path.join(FDD, "persons.parquet"))
    p["key"] = person_key(p)
    p = p[p.key.str.len() > 4]
    rel = (p.RELATIONSHIP_1.fillna("") + "|" + p.RELATIONSHIP_2.fillna("") + "|" + p.RELATIONSHIP_3.fillna(""))
    p["is_dir"] = rel.str.contains("Director")
    p["is_exec"] = rel.str.contains("Executive")
    p = p.merge(f, on="ACCESSIONNUMBER")
    p["CIK"] = p.CIK.astype(str)
    p["fund"] = p.INDUSTRYGROUPTYPE == "Pooled Investment Fund"
    # each (person, CIK): first date seen, fund flag
    pc = p.groupby(["key", "CIK"]).agg(d=("filing_date", "min"), fund=("fund", "max")).reset_index()
    # scale-up date of every CIK (first offering >= $20M), known point-in-time
    fl = pd.read_parquet(os.path.join(FDD, "filings.parquet"), columns=["CIK", "filing_date", "TOTALAMOUNTSOLD"])
    su = fl[fl.TOTALAMOUNTSOLD >= 2e7].groupby("CIK").filing_date.min()
    pc["su"] = pc.CIK.map(su)
    by = {k: v.sort_values("d") for k, v in pc.groupby("key")}
    # people on each entry filing
    ent = p[p.ACCESSIONNUMBER.isin(e.ACCESSIONNUMBER)][["ACCESSIONNUMBER", "key", "is_dir", "is_exec"]]
    acc2cik = pd.Series(e.index, index=e.ACCESSIONNUMBER)
    ent["CIK"] = ent.ACCESSIONNUMBER.map(acc2cik)
    ent = ent.merge(e[["filing_date"]], left_on="CIK", right_index=True)
    rows = {}
    for r in ent.itertuples(index=False):
        h = by.get(r.key)
        out = rows.setdefault(r.CIK, dict(n_people=0, n_dirs=0, n_execs=0, serial_exec=0, exec_prior_scaleup=0,
                                          vc_director=0, board_pro=0, board_track=np.nan, board_prior_cos=0,
                                          name_ambiguous=0))
        out["n_people"] += 1; out["n_dirs"] += int(r.is_dir); out["n_execs"] += int(r.is_exec)
        if h is None:
            continue
        prior = h[(h.d < r.filing_date) & (h.CIK != r.CIK)]
        if len(h) > 300:                         # a very common name: unusable
            out["name_ambiguous"] = 1
            continue
        ops = prior[~prior.fund]
        if r.is_exec and len(ops):
            out["serial_exec"] = 1
            if (ops.su < r.filing_date).any():
                out["exec_prior_scaleup"] = 1
        if r.is_dir:
            if prior.fund.any():
                out["vc_director"] = 1
            if len(ops) >= 3:
                out["board_pro"] = 1
            old = ops[ops.d < r.filing_date - pd.Timedelta(days=730)]
            out["board_prior_cos"] = max(out["board_prior_cos"], len(ops))
            if len(old) >= 3:
                tr = float((old.su < r.filing_date).mean())
                out["board_track"] = tr if np.isnan(out["board_track"]) else max(out["board_track"], tr)
    pf = pd.DataFrame.from_dict(rows, orient="index")
    pf.index.name = "CIK"
    return pf


if __name__ == "__main__":
    e = build()
    pf = person_features(e)
    e = e.join(pf)
    for c in ("n_people", "n_dirs", "n_execs", "serial_exec", "exec_prior_scaleup", "vc_director", "board_pro", "board_prior_cos"):
        e[c] = e[c].fillna(0)
    keep = [c for c in e.columns if not c.isupper() or c in ("ACCESSIONNUMBER", "ENTITYNAME", "STATEORCOUNTRY")]
    e[keep].to_parquet(OUT)
    c = e[e.window_complete == 1]
    print(e.year.value_counts().sort_index().to_dict())
    print("complete-window", len(c), c[["next_raise", "growth20", "big75", "ipo", "outlier"]].mean().round(4).to_dict())
    print(c[["vc_director", "board_pro", "serial_exec", "exec_prior_scaleup", "broker", "rule506c", "nonaccredited", "bay_area"]].mean().round(3).to_dict())
