"""Features that need either a train-window fit (investor rankings, standardisation cut-offs)
or extra point-in-time bookkeeping (investor exit momentum). Everything fitted here is fitted
on TRAIN cohorts (2003-2007) only and then applied unchanged to test and holdout."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analysis.panel import RAW, ROOT

MIN_TRAIN_COMPANIES = 10


def _investments():
    i = pd.read_csv(os.path.join(RAW, "investments.csv"),
                    usecols=["company_permalink", "investor_permalink", "funding_round_permalink", "funded_at"])
    i["funded_at"] = pd.to_datetime(i.funded_at, errors="coerce")
    return i


def investor_rankings(p: pd.DataFrame, i: pd.DataFrame) -> pd.Series:
    """Success rate of each investor over TRAIN-cohort panel companies they backed in any round.
    NOTE: outcomes are as of Dec 2015, so this is a cohort split, not a point-in-time one; an
    investor in 2008 could not have known the 2007 cohort's fate. investor_momentum() below is
    the point-in-time alternative."""
    tr = p[p.split == "train"]
    x = i[i.company_permalink.isin(tr.index)].drop_duplicates(["investor_permalink", "company_permalink"])
    x = x.merge(tr[["success"]], left_on="company_permalink", right_index=True)
    g = x.groupby("investor_permalink").success.agg(["mean", "size"])
    g = g[g["size"] >= MIN_TRAIN_COMPANIES].sort_values(["mean", "size"], ascending=False)
    return g


def exit_dates() -> pd.Series:
    """Date each company's exit became public: acquisition date, or for IPOs the first
    post-IPO round (Crunchbase 2015 has no IPO date; IPOs without one are dropped here, which
    UNDER-counts momentum, never over-counts it)."""
    a = pd.read_csv(os.path.join(RAW, "acquisitions.csv"), usecols=["company_permalink", "acquired_at"])
    a["d"] = pd.to_datetime(a.acquired_at, errors="coerce")
    r = pd.read_csv(os.path.join(RAW, "rounds.csv"), usecols=["company_permalink", "funding_round_type", "funded_at"])
    r = r[r.funding_round_type.isin(["post_ipo_equity", "post_ipo_debt"])]
    r["d"] = pd.to_datetime(r.funded_at, errors="coerce")
    s = pd.concat([a[["company_permalink", "d"]], r[["company_permalink", "d"]]]).dropna()
    return s.groupby("company_permalink").d.min()


def investor_momentum(p: pd.DataFrame, i: pd.DataFrame):
    """Point-in-time, per first-round investor at the company's first-round date d:
        momentum3 = exits realised in [d-3y, d) / portfolio companies first backed before d
        alltime   = exits realised before d  / portfolio companies first backed before d
    denominators < 5 -> NaN. Company value = max over its first-round investors."""
    ex = exit_dates()
    first_touch = i.dropna(subset=["funded_at"]).groupby(["investor_permalink", "company_permalink"]).funded_at.min().reset_index()
    first_touch["exit"] = first_touch.company_permalink.map(ex)
    by = {k: (v.funded_at.values, v.exit.values) for k, v in first_touch.groupby("investor_permalink")}
    fr = i[i.funding_round_permalink.isin(p.first_round)].groupby("funding_round_permalink").investor_permalink.apply(list)
    mom, allt = [], []
    for rnd, d in zip(p.first_round, p.first_date):
        d = np.datetime64(d)
        d3 = d - np.timedelta64(3 * 365, "D")
        bm, ba = np.nan, np.nan
        for inv in fr.get(rnd, []):
            touched, exits = by[inv]
            before = touched < d
            n = before.sum()
            if n < 5:
                continue
            e = exits[before]
            ok = ~pd.isna(e)
            e_all = (e[ok] < d).sum()
            e_3 = ((e[ok] < d) & (e[ok] >= d3)).sum()
            bm = np.nanmax([bm, e_3 / n]); ba = np.nanmax([ba, e_all / n])
        mom.append(bm); allt.append(ba)
    return np.array(mom), np.array(allt)


def name_semantic_distance(p: pd.DataFrame) -> np.ndarray:
    """Cosine distance of the company NAME (char 2-4 grams, TF-IDF) from its primary-category
    centroid, centroids computed on TRAIN names only. Crunchbase 2015 carries no tagline or
    description, so this is name-only; a sentence-embedding model on names alone adds little."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    names = p.name.fillna("").str.lower()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3)
    vec.fit(names[p.split == "train"])
    X = normalize(vec.transform(names))
    tr = (p.split == "train").values
    cats = p.primary_cat.fillna("?").values
    out = np.full(len(p), np.nan)
    for c in np.unique(cats[tr]):
        rows = tr & (cats == c)
        if rows.sum() < 10:
            continue
        cen = normalize(np.asarray(X[rows].mean(axis=0)))
        tgt = cats == c
        out[tgt] = 1 - (X[tgt] @ cen.T).ravel()
    return out


def add_derived(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    i = _investments()
    rank = investor_rankings(p, i)
    top40 = set(rank.index[:40])
    tier2 = set(rank.index[40:120])
    fr = i[i.funding_round_permalink.isin(p.first_round)].groupby("funding_round_permalink").investor_permalink.apply(set)
    sets = p.first_round.map(fr)
    p["smart_money"] = sets.apply(lambda s: int(bool(s & top40)) if isinstance(s, set) else 0)
    p["second_tier_money"] = sets.apply(lambda s: int(bool(s & tier2) and not (s & top40)) if isinstance(s, set) else 0)
    p["inv_momentum3"], p["inv_alltime_pit"] = investor_momentum(p, i)
    p["name_sem_dist"] = name_semantic_distance(p)

    p["log_syndicate"] = np.log1p(p.n_investors)
    p["solo_investor"] = np.where(p.n_investors == 0, np.nan, (p.n_investors == 1).astype(float))
    p["log_crowding"] = np.log1p(p.crowding)
    p["atypicality"] = -p.typicality
    for k in ("appetite_bigtech", "appetite_pe", "appetite_strategic"):
        p["log_" + k] = np.log1p(p[k])
    p["log_inv_prior_deals"] = np.log1p(p.inv_prior_deals)
    # category momentum: primary-category first rounds in year t vs t-2 (log ratio, +1 smoothed)
    cnt = p.groupby(["primary_cat", "cohort"]).size()
    allcnt = p.assign(n=1)
    p["cat_momentum"] = [np.log((cnt.get((c, y), 0) + 1) / (cnt.get((c, y - 2), 0) + 1))
                         for c, y in zip(p.primary_cat, p.cohort)]
    ttn = (p.next_date - p.first_date).dt.days / 365.25
    p["years_to_next"] = ttn.where(ttn > 0)

    fd_path = os.path.join(ROOT, "data", "formd_hist.parquet")
    if os.path.exists(fd_path):
        fd = pd.read_parquet(fd_path)
        p["formd_delay_days"] = fd.formd_delay_days.reindex(p.index)
        p["formd_late"] = np.where(p.formd_delay_days.isna(), np.nan, (p.formd_delay_days > 15).astype(float))
    p.attrs["top40"] = sorted(top40)
    return p
