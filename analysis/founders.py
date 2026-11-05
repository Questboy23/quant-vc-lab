"""Round 3 features: who the founders are, and how well connected the investors are.

Founder data comes from the Crunchbase Dec-2013 dump (people, relationships, degrees). That
dump post-dates every first round, and better-documented companies list more people, which is a
known coverage bias. Every founder test therefore runs only on companies with >=1 founder
recorded AND controls for log(people listed).

Point-in-time where possible: a "prior exit" counts only if the founder's earlier company was
acquired/IPO'd BEFORE this company's first round."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from analysis.panel import ROOT, RAW

CB13 = os.path.join(ROOT, "data", "raw", "cb2013")
FOUNDER = re.compile(r"found", re.I)
EXEC = re.compile(r"found|ceo|cto|chief|president|vp|vice president|head of|director", re.I)
BIGTECH = {"Google", "Facebook", "Microsoft", "Apple", "Amazon", "Yahoo!", "PayPal", "eBay", "Oracle",
           "Cisco", "LinkedIn", "Twitter", "Salesforce", "Intel", "IBM", "Adobe Systems", "VMware"}
ELITE = re.compile(r"stanford|massachusetts institute|\bmit\b|harvard|berkeley|carnegie mellon|princeton|"
                   r"yale|caltech|california institute of technology|wharton|university of pennsylvania|columbia|"
                   r"cornell|university of washington|georgia tech|university of illinois|university of michigan", re.I)


def founder_features(p: pd.DataFrame) -> pd.DataFrame:
    obj = pd.read_csv(os.path.join(CB13, "objects.csv"),
                      usecols=["id", "entity_type", "name", "permalink", "founded_at", "first_funding_at", "status"],
                      low_memory=False)
    comp = obj[obj.entity_type == "Company"].copy()
    comp["slug"] = comp.permalink.str.replace("/company/", "", regex=False)
    comp = comp.drop_duplicates("slug")
    comp["founded_at"] = pd.to_datetime(comp.founded_at, errors="coerce")
    comp["first_funding_at"] = pd.to_datetime(comp.first_funding_at, errors="coerce")
    rel = pd.read_csv(os.path.join(CB13, "relationships.csv"),
                      usecols=["person_object_id", "relationship_object_id", "title", "is_past"])
    rel = rel[rel.relationship_object_id.str.startswith("c:", na=False)]
    deg = pd.read_csv(os.path.join(CB13, "degrees.csv"), usecols=["object_id", "degree_type", "institution"])
    ms = pd.read_csv(os.path.join(CB13, "milestones.csv"), usecols=["object_id", "milestone_at"])
    ms["milestone_at"] = pd.to_datetime(ms.milestone_at, errors="coerce")

    # exit dates for every company (2013 dump acquisitions + IPOs, plus 2015 export acquisitions)
    a13 = pd.read_csv(os.path.join(CB13, "acquisitions.csv"), usecols=["acquired_object_id", "acquired_at"])
    a13["d"] = pd.to_datetime(a13.acquired_at, errors="coerce")
    ipo = pd.read_csv(os.path.join(CB13, "ipos.csv"), usecols=["object_id", "public_at"])
    ipo["d"] = pd.to_datetime(ipo.public_at, errors="coerce")
    exit_d = pd.concat([a13.rename(columns={"acquired_object_id": "oid"})[["oid", "d"]],
                        ipo.rename(columns={"object_id": "oid"})[["oid", "d"]]]).dropna().groupby("oid").d.min()

    slug = pd.Series(p.index.str.replace("/organization/", "", regex=False), index=p.index)
    oid = slug.map(comp.set_index("slug").id)
    out = pd.DataFrame(index=p.index)
    out["cb13_matched"] = oid.notna().astype(int)
    name_of = comp.set_index("id").name

    rel["is_founder"] = rel.title.fillna("").str.contains(FOUNDER)
    rel["is_exec"] = rel.title.fillna("").str.contains(EXEC)
    by_company = {k: v for k, v in rel.groupby("relationship_object_id")}
    by_person = {k: v for k, v in rel.groupby("person_object_id")}
    deg_by = {k: v for k, v in deg.groupby("object_id")}
    ms_by = {k: v.milestone_at.values for k, v in ms.groupby("object_id")}
    comp_start = comp.set_index("id")[["founded_at", "first_funding_at"]].min(axis=1)

    # investor history for founder-investor repeat ties (2015 export)
    inv = pd.read_csv(os.path.join(RAW, "investments.csv"),
                      usecols=["company_permalink", "investor_permalink", "funding_round_permalink", "funded_at"])
    inv["funded_at"] = pd.to_datetime(inv.funded_at, errors="coerce")
    inv["slug"] = inv.company_permalink.str.replace("/organization/", "", regex=False)
    inv_by_slug = {k: set(v.investor_permalink) for k, v in inv.groupby("slug")}
    fr_inv = inv[inv.funding_round_permalink.isin(p.first_round)].groupby("funding_round_permalink").investor_permalink.apply(set)
    id_to_slug = comp.set_index("id").slug

    rows = []
    for perm, o, d, fr in zip(p.index, oid, p.first_date, p.first_round):
        r = dict(n_people=0, n_founders=0, serial=0, prior_exit=0, bigtech_alum=0, elite_school=0, phd=0,
                 mba=0, ms_pre=0, investor_repeat=0)
        if isinstance(o, str):
            staff = by_company.get(o)
            if staff is not None:
                r["n_people"] = staff.person_object_id.nunique()
                f = staff[staff.is_founder].person_object_id.unique()
                if len(f) == 0:      # fall back to CEO/CTO-level execs listed as current
                    f = staff[staff.title.fillna("").str.contains(r"ceo|cto", case=False)].person_object_id.unique()
                r["n_founders"] = len(f)
                first_inv = fr_inv.get(fr, set())
                for person in f:
                    hist = by_person.get(person)
                    if hist is not None:
                        others = hist[hist.relationship_object_id != o]
                        for _, h in others.iterrows():
                            c2 = h.relationship_object_id
                            started = comp_start.get(c2)
                            if name_of.get(c2) in BIGTECH:
                                r["bigtech_alum"] = 1
                                continue
                            if not h.is_exec or pd.isna(started) or started >= d:
                                continue
                            r["serial"] = 1
                            ed = exit_d.get(c2)
                            if ed is not None and ed < d:
                                r["prior_exit"] = 1
                            s2 = id_to_slug.get(c2)
                            if s2 and first_inv and (inv_by_slug.get(s2, set()) & first_inv):
                                r["investor_repeat"] = 1
                    dg = deg_by.get(person)
                    if dg is not None:
                        if dg.institution.fillna("").str.contains(ELITE).any():
                            r["elite_school"] = 1
                        t = dg.degree_type.fillna("").str.lower()
                        r["phd"] |= int(t.str.contains(r"ph\.?d").any())
                        r["mba"] |= int(t.str.contains("mba").any())
            m = ms_by.get(o)
            if m is not None:
                r["ms_pre"] = int((m < np.datetime64(d)).sum())
        rows.append(r)
    f = pd.DataFrame(rows, index=p.index)
    out = out.join(f)
    out["has_founder_record"] = (out.n_founders > 0).astype(int)
    out["log_people"] = np.log1p(out.n_people)
    out["multi_founder"] = np.where(out.n_founders == 0, np.nan, (out.n_founders >= 2).astype(float))
    out["log_ms_pre"] = np.log1p(out.ms_pre)
    return out


def network_centrality(p: pd.DataFrame) -> pd.Series:
    """Point-in-time co-investment degree: for each first-round investor, the number of distinct
    co-investors it had shared a round with BEFORE the first-round date; company = log1p(max)."""
    inv = pd.read_csv(os.path.join(RAW, "investments.csv"),
                      usecols=["investor_permalink", "funding_round_permalink", "funded_at"])
    inv["funded_at"] = pd.to_datetime(inv.funded_at, errors="coerce")
    inv = inv.dropna(subset=["funded_at", "investor_permalink"]).sort_values("funded_at")
    rounds = inv.groupby("funding_round_permalink").agg(d=("funded_at", "first"), inv=("investor_permalink", list))
    rounds = rounds[rounds.inv.str.len() >= 2].sort_values("d")
    # build per-investor list of (date, co-investor) edges
    edges = []
    for d, lst in zip(rounds.d.values, rounds.inv.values):
        u = sorted(set(lst))
        for a in u:
            for b in u:
                if a != b:
                    edges.append((a, d, b))
    e = pd.DataFrame(edges, columns=["a", "d", "b"]).sort_values("d")
    first_tie = e.groupby(["a", "b"]).d.min().reset_index().sort_values("d")
    tie_dates = {k: v.d.values for k, v in first_tie.groupby("a")}
    fr = pd.read_csv(os.path.join(RAW, "investments.csv"), usecols=["investor_permalink", "funding_round_permalink"])
    fr = fr[fr.funding_round_permalink.isin(p.first_round)].groupby("funding_round_permalink").investor_permalink.apply(list)
    vals = []
    for rnd, d in zip(p.first_round, p.first_date):
        best = np.nan
        for i in fr.get(rnd, []):
            td = tie_dates.get(i)
            k = 0 if td is None else int(np.searchsorted(td, np.datetime64(d)))
            best = k if np.isnan(best) else max(best, k)
        vals.append(best)
    return np.log1p(pd.Series(vals, index=p.index))


if __name__ == "__main__":
    p = pd.read_parquet(os.path.join(ROOT, "data", "panel_v3.parquet"))
    f = founder_features(p)
    f["net_centrality"] = network_centrality(p)
    f.to_parquet(os.path.join(ROOT, "data", "founders_v3.parquet"))
    tr = f[p.split != "lockbox"]
    print(tr.describe().T[["count", "mean"]].round(3))
