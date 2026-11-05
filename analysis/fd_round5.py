"""Round 5 features + 30-month outcomes on the Form D panel.

New point-in-time features:
  vc_fund_size      log10 of the largest VC fund (Pooled Investment Fund offering) any director
                    appeared on before the first raise  (top-tier VC proxy)
  n_vc_directors    distinct directors who appear on VC fund filings
  vc_dir_track      share of a VC director's earlier startups that reached a $20M raise before this date
  zip_density       log(1 + first raises in the same 3-digit zip in the prior 2 years)
  sector_tailwind   share of same-industry first raises from 3-5 years earlier that had reached a
                    $20M raise by this date
  exec_vc_tie       an executive previously appeared on a filing alongside a VC person
  serial_count      log(1 + number of earlier startups the executives appeared on)
  exec_prior_vc     an executive's earlier startup had a VC director
30-month outcomes (lockbox 2022-2023 first raises have complete 30-month windows by 2026Q2):
  next30, growth20_30, markup30 (paper, $5M floor), outlier30 (markup30 >= 10x or unicorn)
Round-2 follow-on: second offering within 18 months of the first; step-up = second/first raise;
  markup from second-raise price to the largest raise in the following 30 months."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.fd_panel import offerings, person_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDD = os.path.join(ROOT, "data", "fd")
W30 = pd.DateOffset(months=30)
FLOOR = 5e6


def build():
    e = pd.read_parquet(os.path.join(FDD, "panel.parquet"))
    f = pd.read_parquet(os.path.join(FDD, "filings.parquet"))
    f = f.dropna(subset=["filing_date", "CIK"])
    o = offerings(f)
    o = o[o.CIK.isin(e.index)].merge(e[["filing_date", "amount"]].rename(columns={"filing_date": "entry", "amount": "entry_amt"}),
                                     left_on="CIK", right_index=True)
    o["amount"] = o.amount.fillna(0)
    lat = o[(o.start > o.entry + pd.Timedelta(days=90))]
    w = lat[lat.start <= lat.entry + W30]
    g = w.groupby("CIK")
    e["next30"] = g.size().reindex(e.index).fillna(0).gt(0).astype(int)
    mx = g.amount.max().reindex(e.index)
    e["growth20_30"] = (mx >= 2e7).astype(int)
    kk = w.merge(w.loc[w.groupby("CIK").amount.idxmax(), ["CIK", "start"]].rename(columns={"start": "tmax"}), on="CIK")
    kb = kk[kk.start <= kk.tmax].groupby("CIK").size().reindex(e.index).fillna(0)
    pe = np.maximum(e.amount / 0.2, FLOOR)
    m30 = (mx / 0.15 * 0.85 ** np.minimum(kb, 6) / pe).fillna(0)
    e["markup30"] = m30
    e["outlier30"] = ((m30 >= 10) | (e.unicorn == 1)).astype(int)
    e["w30_complete"] = (e.filing_date + W30 <= pd.Timestamp("2026-06-30")).astype(int)
    # round-2 follow-on
    sec = lat.sort_values("start").groupby("CIK").head(1).set_index("CIK")
    e["r2_date"] = sec.start.reindex(e.index)
    e["r2_amt"] = sec.amount.reindex(e.index)
    e["r2_months"] = (e.r2_date - e.filing_date).dt.days / 30.44
    e["r2_stepup"] = e.r2_amt / e.amount
    after = lat.merge(e[["r2_date"]], left_on="CIK", right_index=True)
    after = after[(after.start > after.r2_date + pd.Timedelta(days=60)) & (after.start <= after.r2_date + W30)]
    mx2 = after.groupby("CIK").amount.max().reindex(e.index)
    pe2 = np.maximum(e.r2_amt / 0.2, FLOOR * 2)
    e["r2_markup30"] = (mx2 / 0.15 * 0.85 / pe2).fillna(0)
    e["r2_w30_complete"] = (e.r2_date + W30 <= pd.Timestamp("2026-06-30")).fillna(False).astype(int)

    # ---------------- people / VC features ----------------
    p = pd.read_parquet(os.path.join(FDD, "persons.parquet"))
    p["key"] = person_key(p)
    p = p[p.key.str.len() > 4]
    rel = p.RELATIONSHIP_1.fillna("") + "|" + p.RELATIONSHIP_2.fillna("") + "|" + p.RELATIONSHIP_3.fillna("")
    p["is_dir"] = rel.str.contains("Director"); p["is_exec"] = rel.str.contains("Executive")
    p = p.merge(f[["ACCESSIONNUMBER", "CIK", "filing_date", "INDUSTRYGROUPTYPE", "TOTALOFFERINGAMOUNT"]], on="ACCESSIONNUMBER")
    p["fund"] = p.INDUSTRYGROUPTYPE == "Pooled Investment Fund"
    su = f[f.TOTALAMOUNTSOLD >= 2e7].groupby("CIK").filing_date.min()
    pc = p.groupby(["key", "CIK"]).agg(d=("filing_date", "min"), fund=("fund", "max"),
                                       fsize=("TOTALOFFERINGAMOUNT", "max")).reset_index()
    pc["su"] = pc.CIK.map(su)
    # which startups had a VC director (any director who had a fund filing before that startup's first filing)
    first_fund = pc[pc.fund].groupby("key").d.min()
    dirs = p[p.is_dir & ~p.fund][["key", "CIK", "filing_date"]]
    dirs = dirs.assign(ff=dirs.key.map(first_fund))
    vc_backed = set(dirs[dirs.ff.notna() & (dirs.ff < dirs.filing_date)].CIK)
    pc["vcb"] = pc.CIK.isin(vc_backed)
    # co-appearance with VC people: accessions that include a VC person
    vc_people = set(first_fund.index)
    acc_has_vc = set(p[p.key.isin(vc_people)].ACCESSIONNUMBER)
    p["acc_vc"] = p.ACCESSIONNUMBER.isin(acc_has_vc)
    tie_first = p[p.acc_vc].groupby("key").filing_date.min()
    by = {k: v.sort_values("d") for k, v in pc.groupby("key")}
    ent = p[p.ACCESSIONNUMBER.isin(e.ACCESSIONNUMBER)][["ACCESSIONNUMBER", "key", "is_dir", "is_exec"]]
    acc2cik = pd.Series(e.index, index=e.ACCESSIONNUMBER)
    ent["CIK"] = ent.ACCESSIONNUMBER.map(acc2cik)
    ent = ent.merge(e[["filing_date"]], left_on="CIK", right_index=True)
    rows = {}
    for r in ent.itertuples(index=False):
        o_ = rows.setdefault(r.CIK, dict(vc_fund_size=0.0, n_vc_directors=0, vc_dir_track=np.nan, exec_vc_tie=0,
                                         serial_count=0, exec_prior_vc=0))
        h = by.get(r.key)
        if h is None or len(h) > 300:
            continue
        prior = h[(h.d < r.filing_date) & (h.CIK != r.CIK)]
        funds = prior[prior.fund]
        ops = prior[~prior.fund]
        if r.is_dir and len(funds):
            o_["n_vc_directors"] += 1
            fs = funds.fsize.max()
            if pd.notna(fs) and fs > 0:
                o_["vc_fund_size"] = max(o_["vc_fund_size"], float(np.log10(fs)))
            old = ops[ops.d < r.filing_date - pd.Timedelta(days=730)]
            if len(old) >= 2:
                tr = float((old.su < r.filing_date).mean())
                o_["vc_dir_track"] = tr if np.isnan(o_["vc_dir_track"]) else max(o_["vc_dir_track"], tr)
        if r.is_exec:
            o_["serial_count"] = max(o_["serial_count"], float(np.log1p(len(ops))))
            t = tie_first.get(r.key)
            if t is not None and t < r.filing_date:
                o_["exec_vc_tie"] = 1
            if ops.vcb.any():
                o_["exec_prior_vc"] = 1
    pf = pd.DataFrame.from_dict(rows, orient="index")
    e = e.drop(columns=[c for c in pf.columns if c in e.columns]).join(pf)
    for c in ("vc_fund_size", "n_vc_directors", "exec_vc_tie", "serial_count", "exec_prior_vc"):
        e[c] = e[c].fillna(0)

    # ---------------- place and sector ----------------
    e["zip3"] = e.ZIPCODE.fillna("").str[:3] if "ZIPCODE" in e.columns else ""
    allfirst = e[["filing_date", "industry"]].copy()
    allfirst["zip3"] = pd.read_parquet(os.path.join(FDD, "filings.parquet"), columns=["CIK", "ZIPCODE"]).drop_duplicates("CIK").set_index("CIK").ZIPCODE.reindex(e.index).fillna("").str[:3]
    e["zip3"] = allfirst.zip3
    zd = []
    by_zip = {k: np.sort(v.filing_date.values) for k, v in allfirst.groupby("zip3")}
    for z, d in zip(allfirst.zip3, e.filing_date):
        arr = by_zip.get(z)
        lo = np.searchsorted(arr, np.datetime64(d - pd.DateOffset(years=2)))
        hi = np.searchsorted(arr, np.datetime64(d))
        zd.append(np.log1p(hi - lo))
    e["zip_density"] = zd
    sul = e[["industry", "filing_date"]].assign(su=e.index.map(su))
    tw = []
    by_ind = {k: v for k, v in sul.groupby("industry")}
    for ind, d in zip(e.industry, e.filing_date):
        v = by_ind[ind]
        coh = v[(v.filing_date >= d - pd.DateOffset(years=5)) & (v.filing_date < d - pd.DateOffset(years=3))]
        tw.append(float((coh.su < d).mean()) if len(coh) >= 30 else np.nan)
    e["sector_tailwind"] = tw
    return e


if __name__ == "__main__":
    e = build()
    e.to_parquet(os.path.join(FDD, "panel_r5.parquet"))
    c = e[(e.w30_complete == 1) & (e.year <= 2021)]
    print(c[["next30", "growth20_30", "outlier30"]].mean().round(4).to_dict())
    print(e[["vc_fund_size", "n_vc_directors", "exec_vc_tie", "exec_prior_vc", "zip_density", "sector_tailwind"]].describe().T[["count", "mean"]].round(3))
    print("r2 rows", e.r2_date.notna().sum())
