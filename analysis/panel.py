"""Build the company-level panel: one row per US company whose FIRST equity round is seed,
angel or Series A and happened 2003-2011. Every feature is defined as of the first-round date
unless its docstring says otherwise.

Cohort splits (fixed, never tuned):  train 2003-2007 | test 2008-2010 | holdout 2011 (sealed).

Things Crunchbase 2015 does NOT have, and how we cope (see METHODOLOGY in reports/):
  * no valuations      -> entry post-money is IMPUTED from round size (returns.py)
  * no IPO date/value  -> IPO exit value is an ASSUMPTION (returns.py)
  * no lead-investor flag -> "follow-on" uses participation, not leadership
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "crunchbase-data")
OUT = os.path.join(ROOT, "data", "panel.parquet")

TRAIN = range(2003, 2008)
TEST = range(2008, 2011)
HOLDOUT = (2011,)

EQUITY = {"seed", "angel", "venture"}
BIG_TECH = {
    "Google", "Apple", "Facebook", "Microsoft", "Amazon", "Yahoo!", "eBay", "Cisco",
    "Oracle Corporation", "IBM (International Business Machines)", "Intel", "Salesforce",
    "Twitter", "LinkedIn", "Adobe Systems", "AOL", "Hewlett-Packard", "EMC", "VMware",
    "Dell", "Qualcomm", "Nokia", "Groupon", "Zynga", "IAC",
}
# Seed programmes whose participation is a documented selection filter.
ACCELERATORS = re.compile(r"y combinator|techstars|500 startups|dreamit|angelpad|"
                          r"seedcamp|capital factory|launchpad la|amplify|kicklabs|"
                          r"betaspring|excelerate|brandery|tech wildcatters|jumpstart",
                          re.I)
FIN_WORDS = re.compile(r"capital|partners|equity|holdings|investment|ventures|group lp|fund",
                       re.I)


def _load():
    c = pd.read_csv(os.path.join(RAW, "companies.csv"))
    r = pd.read_csv(os.path.join(RAW, "rounds.csv"))
    i = pd.read_csv(os.path.join(RAW, "investments.csv"))
    a = pd.read_csv(os.path.join(RAW, "acquisitions.csv"))
    for df, col in ((r, "funded_at"), (i, "funded_at"), (a, "acquired_at"),
                    (c, "founded_at"), (c, "first_funding_at")):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return c, r, i, a


def _first_rounds(r: pd.DataFrame) -> pd.DataFrame:
    eq = r[r.funding_round_type.isin(EQUITY)].dropna(subset=["funded_at"])
    eq = eq[(eq.funded_at.dt.year >= 1990) & (eq.funded_at.dt.year <= 2015)]
    eq = eq.sort_values(["company_permalink", "funded_at", "raised_amount_usd"])
    first = eq.groupby("company_permalink").head(1).copy()
    code = first.funding_round_code.fillna("")
    t = first.funding_round_type
    first["stage"] = np.where(t == "seed", "seed", np.where(t == "angel", "angel",
                              np.where(code == "A", "A", np.where(code == "", "venture_unlabeled", "later"))))
    return first, eq


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def build(max_cohort: int = 2011) -> pd.DataFrame:
    c, r, i, a = _load()
    first, eq = _first_rounds(r)
    us = c[c.country_code == "USA"].set_index("permalink")

    p = first[first.company_permalink.isin(us.index)].copy()
    p = p[p.stage.isin(["seed", "angel", "A"])]
    p["cohort"] = p.funded_at.dt.year
    p = p[(p.cohort >= 2003) & (p.cohort <= max_cohort)]
    p = p.rename(columns={"company_permalink": "permalink", "funded_at": "first_date",
                          "raised_amount_usd": "first_raised",
                          "funding_round_permalink": "first_round"})
    p = p.set_index("permalink")
    info = us.loc[p.index]
    for col in ("name", "homepage_url", "category_list", "status", "region", "city",
                "state_code", "founded_at", "funding_total_usd"):
        p[col] = info[col]
    p["funding_total_usd"] = pd.to_numeric(p.funding_total_usd, errors="coerce")
    p["split"] = np.select([p.cohort.isin(TRAIN), p.cohort.isin(TEST), p.cohort.isin(HOLDOUT)],
                           ["train", "test", "holdout"], "lockbox")   # 2012 = round-3 lockbox

    # ---------------- outcomes ----------------
    acq = a.sort_values("acquired_at").drop_duplicates("company_permalink", keep="last").set_index("company_permalink")
    acq_usd = acq.price_amount.where(acq.price_currency_code == "USD")
    p["acq_price"] = acq_usd.reindex(p.index)
    p["acq_date"] = acq.acquired_at.reindex(p.index)
    p["acquirer"] = acq.acquirer_name.reindex(p.index)
    p["success"] = p.status.isin(["acquired", "ipo"]).astype(int)
    p["ipo"] = (p.status == "ipo").astype(int)
    p["bigwin"] = ((p.status == "ipo") | ((p.status == "acquired") & (p.acq_price >= 100e6))).astype(int)
    p["closed"] = (p.status == "closed").astype(int)

    # later equity rounds: count (for dilution) and next round (for follow-on)
    eqc = eq[eq.company_permalink.isin(p.index)]
    eqc = eqc.merge(p[["first_round"]], left_on="company_permalink", right_index=True)
    later = eqc[eqc.funding_round_permalink != eqc.first_round]
    p["n_later_rounds"] = later.groupby("company_permalink").size().reindex(p.index).fillna(0).astype(int)
    nxt = later.sort_values("funded_at").groupby("company_permalink").head(1).set_index("company_permalink")
    p["next_round"] = nxt.funding_round_permalink.reindex(p.index)
    p["next_date"] = nxt.funded_at.reindex(p.index)
    p["next_raised"] = nxt.raised_amount_usd.reindex(p.index)

    # ---------------- first-round investors ----------------
    inv = i[["funding_round_permalink", "investor_permalink", "investor_name", "investor_region",
             "investor_country_code", "company_permalink", "funded_at"]]
    fri = inv[inv.funding_round_permalink.isin(p.first_round)]
    g = fri.groupby("funding_round_permalink")
    p["n_investors"] = g.investor_permalink.nunique().reindex(p.first_round).fillna(0).values
    p["has_investor_record"] = (p.n_investors > 0).astype(int)
    p["n_inv_regions"] = g.investor_region.nunique().reindex(p.first_round).fillna(0).values
    p["share_person"] = g.investor_permalink.apply(lambda s: s.str.startswith("/person/").mean()).reindex(p.first_round).values
    p["accelerator"] = g.investor_name.apply(lambda s: s.fillna("").str.contains(ACCELERATORS).any()).reindex(p.first_round).fillna(False).astype(int).values
    loc = fri.merge(p[["first_round", "region"]], left_on="funding_round_permalink", right_on="first_round")
    loc["local"] = (loc.investor_region == loc.region).astype(float)
    p["share_local"] = loc.groupby("funding_round_permalink").local.mean().reindex(p.first_round).values

    # follow-on: any first-round investor also in the next equity round
    nri = inv[inv.funding_round_permalink.isin(p.next_round.dropna())]
    fset = fri.groupby("funding_round_permalink").investor_permalink.apply(set)
    nset = nri.groupby("funding_round_permalink").investor_permalink.apply(set)
    fo = []
    for fr, nr in zip(p.first_round, p.next_round):
        if not isinstance(nr, str) or nr not in nset or fr not in fset:
            fo.append(np.nan)
        else:
            fo.append(float(bool(fset[fr] & nset[nr])))
    p["insider_follow_on"] = fo

    # point-in-time investor experience: prior deals before the first-round date
    inv_all = inv.dropna(subset=["funded_at"]).sort_values("funded_at")
    exp = []
    by_inv = {k: v.funded_at.values for k, v in inv_all.groupby("investor_permalink")}
    fr_inv = fri.groupby("funding_round_permalink").investor_permalink.apply(list)
    for fr, d in zip(p.first_round, p.first_date):
        lst = fr_inv.get(fr, [])
        if not lst:
            exp.append(np.nan); continue
        dd = np.datetime64(d)
        exp.append(np.mean([np.searchsorted(by_inv[x], dd) for x in lst]))
    p["inv_prior_deals"] = exp

    # ---------------- category features ----------------
    p["categories"] = p.category_list.fillna("").str.split("|")
    p["primary_cat"] = p.categories.str[0].replace("", np.nan)
    p["n_categories"] = p.categories.apply(lambda x: len([y for y in x if y]))

    # crowding: all companies (any country) whose first equity round in same primary cat, years t-1..t
    allf = first.merge(c[["permalink", "category_list"]], left_on="company_permalink", right_on="permalink")
    allf["pc"] = allf.category_list.fillna("").str.split("|").str[0]
    allf["yr"] = allf.funded_at.dt.year
    cy = allf.groupby(["pc", "yr"]).size()
    p["crowding"] = [cy.get((pc, y), 0) + cy.get((pc, y - 1), 0) for pc, y in zip(p.primary_cat, p.cohort)]

    # Uzzi atypicality: min over category pairs of log((obs+1)/(exp+1)), counts from companies
    # whose first round was STRICTLY before this cohort year (point-in-time).
    allf["cats"] = allf.category_list.fillna("").str.split("|")
    atyp = {}
    for y in sorted(p.cohort.unique()):
        prior = allf[allf.yr < y].cats
        n = len(prior)
        single = prior.explode().value_counts()
        pairs = {}
        for cats in prior:
            cs = sorted(set(x for x in cats if x))
            for ai in range(len(cs)):
                for bi in range(ai + 1, len(cs)):
                    pairs[(cs[ai], cs[bi])] = pairs.get((cs[ai], cs[bi]), 0) + 1
        atyp[y] = (n, single, pairs)
    vals = []
    for cats, y in zip(p.categories, p.cohort):
        cs = sorted(set(x for x in cats if x))
        if len(cs) < 2:
            vals.append(np.nan); continue
        n, single, pairs = atyp[y]
        sc = []
        for ai in range(len(cs)):
            for bi in range(ai + 1, len(cs)):
                e = single.get(cs[ai], 0) * single.get(cs[bi], 0) / max(n, 1)
                sc.append(np.log((pairs.get((cs[ai], cs[bi]), 0) + 1) / (e + 1)))
        vals.append(min(sc))
    p["typicality"] = vals  # low = atypical

    # acquirer appetite & kill zone: acquisitions of companies in same primary cat, prior 2/3 yrs
    a2 = a.dropna(subset=["acquired_at"]).copy()
    a2["pc"] = a2.company_category_list.fillna("").str.split("|").str[0]
    fin = a2.acquirer_category_list.fillna("").str.contains("Private Equity|Venture|Finance") | \
        a2.acquirer_name.fillna("").str.contains(FIN_WORDS)
    a2["atype"] = np.where(a2.acquirer_name.isin(BIG_TECH), "bigtech", np.where(fin, "pe", "strategic"))
    a_by = {k: v for k, v in a2.groupby("pc")}
    kz, app_b, app_p, app_s = [], [], [], []
    for pc, d in zip(p.primary_cat, p.first_date):
        v = a_by.get(pc)
        if v is None:
            kz.append(0); app_b.append(0); app_p.append(0); app_s.append(0); continue
        w3 = v[(v.acquired_at < d) & (v.acquired_at >= d - pd.DateOffset(years=3))]
        w2 = v[(v.acquired_at < d) & (v.acquired_at >= d - pd.DateOffset(years=2))]
        kz.append(int((w3.atype == "bigtech").any()))
        app_b.append((w2.atype == "bigtech").sum()); app_p.append((w2.atype == "pe").sum())
        app_s.append((w2.atype == "strategic").sum())
    p["kill_zone"] = kz
    p["appetite_bigtech"] = app_b; p["appetite_pe"] = app_p; p["appetite_strategic"] = app_s

    # ---------------- geography ----------------
    p["sf"] = (p.region == "SF Bay Area").astype(int)
    p["hub_ny_bos"] = p.region.isin(["New York City", "Boston"]).astype(int)

    # ---------------- names & domains ----------------
    words = set()
    for path in ("/usr/share/dict/words",):
        if os.path.exists(path):
            words = {w.strip().lower() for w in open(path)}
    nm = p.name.fillna("").map(_norm)
    p["name_len"] = nm.str.len()
    p["name_real_word"] = p.name.fillna("").str.lower().str.strip().isin(words).astype(int)
    p["name_has_digit"] = nm.str.contains(r"\d").astype(int)
    host = p.homepage_url.fillna("").str.lower().str.replace(r"^https?://", "", regex=True)\
        .str.replace(r"^www\.", "", regex=True).str.split("/").str[0]
    p["has_homepage"] = (host != "").astype(int)
    p["dot_com"] = np.where(host == "", np.nan, host.str.endswith(".com").astype(float))
    stem = host.str.rsplit(".", n=1).str[0].map(_norm)
    p["exact_domain"] = np.where(host == "", np.nan, (stem == nm).astype(float))

    # ---------------- age & size ----------------
    age = (p.first_date - p.founded_at).dt.days / 365.25
    p["age_at_first"] = age.where((age >= -1) & (age <= 30))
    p["has_founded"] = p.age_at_first.notna().astype(int)
    p["raised_missing"] = p.first_raised.isna().astype(int)
    med = p.groupby(["stage", "cohort"]).first_raised.transform("median")
    p["raised_filled"] = p.first_raised.fillna(med)
    p["log_raised"] = np.log10(p.raised_filled.clip(lower=1e4))
    cat_med = p.groupby(["primary_cat", "cohort"]).first_raised.transform("median")
    p["size_surprise"] = np.log(p.first_raised.where(p.first_raised > 0) / cat_med.where(cat_med > 0))
    p["complete_record"] = ((p.has_investor_record == 1) & (p.has_homepage == 1) &
                            (p.raised_missing == 0)).astype(int)
    p = p.drop(columns=["categories"])
    return p


if __name__ == "__main__":
    df = build()
    df.to_parquet(OUT)
    print(df.shape)
    print(df.groupby("split")[["success", "bigwin", "closed", "has_investor_record"]].mean().round(3))
    print(df.split.value_counts())
