"""Round 5: registry G1-G12 (30-month outcomes) and strategies L1-L5, lockbox = 2022-2023.

    python -m analysis.v5                  # registry + validation (2018-21, second look)
    python -m analysis.v5 --open-lockbox   # + 2022-23 lockbox ONCE (data/lockbox_2223/.consumed.json)
"""
from __future__ import annotations

import argparse, json, os, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from analysis import v4

warnings.filterwarnings("ignore")
ROOT = v4.ROOT
REG = os.path.join(ROOT, "vc", "hypotheses_v5.yaml")
OUT = os.path.join(ROOT, "reports", "v5.json")
LOCK = os.path.join(ROOT, "data", "lockbox_2223", ".consumed.json")
NEW = ["vc_fund_size", "n_vc_directors", "vc_dir_track", "zip_density", "sector_tailwind", "exec_vc_tie",
       "exec_prior_vc", "serial_count"]


def load():
    e = pd.read_parquet(os.path.join(ROOT, "data", "fd", "panel_r5.parquet"))
    e["split"] = np.select([e.year <= 2017, e.year.between(2018, 2021), e.year.between(2022, 2023)],
                           ["train", "valid", "lockbox"], "x")
    e = e[(e.split != "x") & (e.w30_complete == 1)].copy()
    tr = e[e.split == "train"]
    e["zip_sparse"] = (e.zip_density <= tr.zip_density.quantile(0.25)).astype(int)
    e["sector_cold"] = np.where(e.sector_tailwind.isna(), np.nan,
                                (e.sector_tailwind <= tr.sector_tailwind.quantile(0.25)).astype(float))
    e["unpopular_ind"] = e.industry.isin(["Manufacturing", "Other", "Energy Conservation", "Environmental Services"]).astype(int)
    e["bio_cheap"] = ((e.bio == 1) & (e.amount < tr.amount.median())).astype(int)
    e["growth20"] = e.growth20_30; e["outlier"] = e.outlier30; e["markup"] = e.markup30
    e["exit"] = ((e.ipo_ever == 1) | (e.acquired_any == 1) | (e.unicorn == 1)).astype(int)
    return e


def registry(e):
    hyps = yaml.safe_load(open(REG))
    va = e[e.split == "valid"].copy()
    m = va.markup30.values
    res = []
    for h in hyps:
        f = h["feature"]
        va["_f"] = v4.zs(e.assign(split=e.split.replace({"valid": "test"})), f).loc[va.index]
        prim = v4.logit(va, "_f", h["primary"])
        side = {o: v4.logit(va, "_f", o) for o in ("growth20_30", "outlier30", "next30")}
        side["returns"] = v4.returns_contrast(va.assign(outlier=va.outlier30), f, m)
        res.append((h, prim, side))
    holm = multipletests([r["p"] if r else 1.0 for _, r, _ in res], method="holm")[1]
    out = []
    for (h, r, side), ph in zip(res, holm):
        want = {"+": 1, "-": -1}.get(h["direction"], 0)
        v = ("untestable" if r is None else "no evidence" if ph >= 0.05 else
             "rejected (opposite sign)" if want and np.sign(r["coef"]) != want else "passes")
        out.append({**h, "result": {"verdict": v, "effect": r and r["coef"], "p_raw": r and r["p"], "p_holm": float(ph),
                                    "n": r and r["n"], "side": side}})
        rr = side["returns"]
        print(f'{h["id"]:4s} {h["name"][:44]:44s} {v:24s} eff={r and round(r["coef"],3)} holm={ph:.3g} '
              f'markup30 {rr["mult_hi_excl_top"]:.2f}/{rr["mult_lo_excl_top"]:.2f}', flush=True)
    return out


def quality(e, feats, fit_split="train"):
    """Round-4 style ensemble (ridge on growth20 / outlier / exit), fit on TRAIN only."""
    old = v4.FEATS
    v4.FEATS = feats
    try:
        tr = e[e.split == fit_split]
        Xtr = v4.design(tr, tr); X = v4.design(e, tr)[Xtr.columns]
        sc = []
        for y in ("growth20", "outlier", "exit"):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)).fit(Xtr, tr[y])
            sc.append(pd.Series(mdl.predict_proba(X)[:, 1], index=e.index).groupby(e.year).rank(pct=True))
    finally:
        v4.FEATS = old
    return (sum(sc) / len(sc)).groupby(e.year).rank(pct=True)


def evaluate(d, w, m, label):
    sel = w > 0
    keep = m < m.max()
    mm = (w * m).sum() / w.sum()
    rng = np.random.default_rng(41); bs = []
    for _ in range(2000):
        b = rng.choice(len(d), len(d))
        if sel[b].any():
            bs.append((w[b] * m[b]).sum() / w[b].sum() / m[b].mean())
    return {"rule": label, "deals": int(sel.sum()), "markup": float(mm), "all_markup": float(m.mean()),
            "markup_excl_top": float((w * m)[keep].sum() / w[keep].sum()), "all_excl_top": float(m[keep].mean()),
            "vs_all_pct": float(100 * (mm / m.mean() - 1)),
            "vs_all_ci_pct": [float(100 * (np.percentile(bs, 2.5) - 1)), float(100 * (np.percentile(bs, 97.5) - 1))],
            "growth20_rate": float(d.growth20_30.values[sel].mean()), "growth20_all": float(d.growth20_30.mean()),
            "outliers": int(d.outlier30.values[sel].sum()), "outliers_total": int(d.outlier30.sum())}


def strategies(e, split):
    d = e[e.split == split]
    m = d.markup30.values
    cheap = (d.amount < d.groupby("year").amount.transform("median")).values
    q1 = quality(e, v4.FEATS).loc[d.index].values
    q2 = quality(e, v4.FEATS + NEW).loc[d.index].values
    res = {"L0_own_all": evaluate(d, np.ones(len(d)), m, "own every first raise"),
           "L1_quality_cheap_r4": evaluate(d, (cheap & (q1 > .5)).astype(float), m, "quality in cheap half (round-4 features)"),
           "L2_quality_cheap_r5": evaluate(d, (cheap & (q2 > .5)).astype(float), m, "quality in cheap half (+ round-5 features)")}
    w3 = (cheap & (q2 > .5)).astype(float) * np.where(q2 > .75, 2.0, 1.0)
    res["L3_tilted"] = evaluate(d, w3, m, "L2 with double weight on top-quarter quality")
    # second-raise rules: decision unit = companies whose second raise falls in the split's years
    allr2 = e[e.r2_date.notna() & (e.r2_w30_complete == 1)].copy()
    allr2["r2_year"] = allr2.r2_date.dt.year
    yrs = {"train": range(2010, 2018), "valid": range(2018, 2022), "lockbox": range(2022, 2024)}[split]
    r2 = allr2[allr2.r2_year.isin(yrs) & (allr2.r2_months <= 18)]
    m2 = r2.r2_markup30.values
    base2 = np.ones(len(r2))
    mom = (r2.r2_stepup >= 2).values.astype(float); con = (r2.r2_stepup <= 1).values.astype(float)
    def ev2(w, lab):
        x = evaluate(r2.assign(growth20_30=0, outlier30=(m2 >= 10).astype(int)), w, m2, lab)
        return x
    res["L4_r2_momentum"] = ev2(mom, "second raise >= 2x the first, within 18 months")
    res["L5_r2_contrarian"] = ev2(con, "second raise <= 1x the first, within 18 months")
    res["r2_all"] = ev2(base2, "every second raise within 18 months")
    qr2 = quality(e, v4.FEATS).loc[r2.index].values
    res["L7_r2_contrarian_quality"] = ev2(con * (qr2 > .5), "flat second raise, first-raise quality top half")
    # L6: two equal sleeves, each measured vs its own all-deals baseline, then averaged
    a, b = res["L1_quality_cheap_r4"], res["L5_r2_contrarian"]
    res["L6_two_sleeves"] = {"rule": "half L1 + half L5", "deals": a["deals"] + b["deals"],
                             "markup": (a["markup"] + b["markup"]) / 2, "all_markup": (a["all_markup"] + b["all_markup"]) / 2,
                             "vs_all_pct": 100 * ((a["markup"] + b["markup"]) / (a["all_markup"] + b["all_markup"]) - 1),
                             "vs_all_ci_pct": [(a["vs_all_ci_pct"][0] + b["vs_all_ci_pct"][0]) / 2, (a["vs_all_ci_pct"][1] + b["vs_all_ci_pct"][1]) / 2],
                             "note": "CI is the average of the two sleeves' CIs (approximate)"}
    try:
        res["auc_growth20_q2"] = float(roc_auc_score(d.growth20_30, q2))
    except ValueError:
        pass
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--open-lockbox", action="store_true"); a = ap.parse_args()
    e = load()
    out = {"generated": datetime.now(timezone.utc).isoformat(), "n": e.split.value_counts().to_dict()}
    out["registry"] = registry(e)
    out["valid"] = strategies(e, "valid")
    for k, v in out["valid"].items():
        if isinstance(v, dict):
            print(f'VALID {k:22s} {v["markup"]:.2f}x vs {v["all_markup"]:.2f}x ({v["vs_all_pct"]:+.0f}%, CI {v["vs_all_ci_pct"][0]:+.0f}..{v["vs_all_ci_pct"][1]:+.0f}) n={v["deals"]}')
    if a.open_lockbox or os.path.exists(LOCK):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        if os.path.exists(LOCK):
            rec = json.load(open(LOCK)); rec["note"] = "LOCKBOX 2022-23 ALREADY SPENT - cached"
        else:
            rec = {"consumed_at": datetime.now(timezone.utc).isoformat(), "result": strategies(e, "lockbox")}
            json.dump(rec, open(LOCK, "w"), indent=1, default=float)
        out["lockbox"] = rec
        for k, v in rec["result"].items():
            if isinstance(v, dict):
                print(f'LOCKBOX {k:22s} {v["markup"]:.2f}x vs {v["all_markup"]:.2f}x ({v["vs_all_pct"]:+.0f}%, CI {v["vs_all_ci_pct"][0]:+.0f}..{v["vs_all_ci_pct"][1]:+.0f}) n={v["deals"]}')
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()
