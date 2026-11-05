"""Composite models, the price-adjusted strategy, fund simulation and the sealed holdout.

Pre-declared, never tuned: portfolio = top 10% of each cohort year by score; fund draws are
equal-weighted deals; 8-year average hold for IRR; 50% haircut on excess return."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.panel import ROOT
from analysis.returns import BASE, boot_diff, grid, multiples, post_money

TOP_FRAC = 0.10
HOLD_YEARS = 8
HAIRCUT = 0.5
LOCK = os.path.join(ROOT, "data", "holdout", ".consumed.json")

PRIOR_FEATURES = ["smart_money", "log_syndicate", "has_investor_record", "log_crowding",
                  "atypicality", "kill_zone", "sf", "name_real_word", "dot_com", "exact_domain",
                  "has_homepage"]
CONTROLS = ["log_raised", "stage_seed", "stage_angel"]


def matrix(p: pd.DataFrame, feats, train_ref: pd.DataFrame):
    X = pd.DataFrame(index=p.index)
    for f in feats:
        x = p[f].astype(float)
        if x.isna().any() or train_ref[f].isna().any():
            X[f + "_missing"] = x.isna().astype(float)
            x = x.fillna(train_ref[f].median())
        X[f] = x
    X["log_raised"] = p.log_raised
    X["stage_seed"] = (p.stage == "seed").astype(float)
    X["stage_angel"] = (p.stage == "angel").astype(float)
    return X


def choose_family(Xtr, ytr, cohorts):
    """Leave-one-cohort-out AUC on TRAIN only; pick the family, never look at test."""
    fams = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      min_samples_leaf=40, random_state=0),
    }
    scores = {}
    for k, mk in fams.items():
        preds = np.zeros(len(ytr))
        for c in np.unique(cohorts):
            tr, te = cohorts != c, cohorts == c
            m = mk().fit(Xtr[tr], ytr[tr])
            preds[te] = m.predict_proba(Xtr[te])[:, 1]
        scores[k] = float(roc_auc_score(ytr, preds))
    best = max(scores, key=scores.get)
    return best, fams[best](), scores


def fit_score(p: pd.DataFrame, feats, outcome="bigwin"):
    tr = p[p.split == "train"]
    Xtr = matrix(tr, feats, tr)
    fam, model, cv = choose_family(Xtr.values, tr[outcome].values, tr.cohort.values)
    model.fit(Xtr.values, tr[outcome].values)
    X = matrix(p, feats, tr)[Xtr.columns]
    prob = pd.Series(model.predict_proba(X.values)[:, 1], index=p.index)
    return prob, {"family": fam, "train_cv_auc": cv, "features": list(Xtr.columns)}


def top_mask(p: pd.DataFrame, score: pd.Series, frac=TOP_FRAC) -> pd.Series:
    """Top `frac` of each cohort year. Ties (e.g. many rounds sitting on the valuation floor)
    are broken by a fixed-seed random order, never by anything outcome-related."""
    jitter = pd.Series(np.random.default_rng(7).random(len(score)), index=score.index)
    order = pd.DataFrame({"s": score, "j": jitter, "c": p.cohort.reindex(score.index)})
    rk = order.sort_values(["s", "j"], ascending=False).groupby("c").cumcount() + 1
    size = order.groupby("c").s.transform("size")
    return (rk / size).reindex(score.index) <= frac


def evaluate(p, prob, sel_score, mult, split="test"):
    d = p[p.split == split]
    sel = top_mask(d, sel_score.loc[d.index]).values
    allm = np.ones(len(d), bool)
    m = mult.loc[d.index].values
    diff, lo, hi, pv = boot_diff(sel, allm, m)
    return {
        "n": int(len(d)), "n_selected": int(sel.sum()),
        "auc_bigwin": float(roc_auc_score(d.bigwin, prob.loc[d.index])),
        "auc_success": float(roc_auc_score(d.success, prob.loc[d.index])),
        "bigwin_rate_selected": float(d.bigwin.values[sel].mean()),
        "bigwin_rate_all": float(d.bigwin.mean()),
        "success_rate_selected": float(d.success.values[sel].mean()),
        "success_rate_all": float(d.success.mean()),
        "multiple_selected": float(m[sel].mean()), "multiple_all": float(m.mean()),
        "excess": diff, "excess_ci": [lo, hi], "excess_p": pv,
        "median_post_selected": float(post_money(d).values[sel].mean()),
        "median_post_all": float(post_money(d).mean()),
    }


def sensitivity(p, prob, split="test"):
    """Re-run the price-adjusted strategy across the full assumption grid."""
    d = p[p.split == split]
    rows = []
    for a in grid():
        post = post_money(d, a)
        score = prob.loc[d.index] / post
        sel = top_mask(d, score).values
        m = multiples(d, a).values
        diff, lo, hi, _ = boot_diff(sel, np.ones(len(d), bool), m, n=500)
        rows.append({"floor": a.floor, "ipo": a.ipo_value, "undisclosed_x": a.undisclosed_x,
                     "operating_x": a.operating_x, "strategy": float(m[sel].mean()),
                     "random": float(m.mean()), "excess": diff, "ci": [lo, hi]})
    return rows


def fund_sim(sel_mult: np.ndarray, all_mult: np.ndarray, sizes=(25, 50, 100), n=10000, seed=1):
    rng = np.random.default_rng(seed)
    excess = sel_mult.mean() - all_mult.mean()
    out = {}
    for N in sizes:
        s = rng.choice(sel_mult, (n, N)).mean(1)
        r = rng.choice(all_mult, (n, N)).mean(1)
        # haircut: scale each simulated fund so its EXPECTED multiple keeps only half the excess
        scale = (all_mult.mean() + (1 - HAIRCUT) * excess) / sel_mult.mean()
        h = s * scale
        def summ(x):
            q = np.percentile(x, [5, 25, 50, 75, 95])
            return {"p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
                    "p_below_1x": float((x < 1).mean()), "p_at_least_3x": float((x >= 3).mean()),
                    "irr_median": float(np.sign(q[2]) * (max(q[2], 0) ** (1 / HOLD_YEARS)) - 1),
                    "mean": float(x.mean())}
        out[N] = {"strategy": summ(s), "strategy_haircut": summ(h), "random": summ(r)}
    return out


def histogram(sel_mult, all_mult, N=50, n=10000, seed=2):
    rng = np.random.default_rng(seed)
    s = rng.choice(sel_mult, (n, N)).mean(1)
    r = rng.choice(all_mult, (n, N)).mean(1)
    excess = sel_mult.mean() - all_mult.mean()
    hc = s * (all_mult.mean() + (1 - HAIRCUT) * excess) / sel_mult.mean()
    edges = np.linspace(0, max(np.percentile(s, 99.5), np.percentile(r, 99.5)), 41)
    return {"edges": edges.round(3).tolist(),
            "strategy": np.histogram(s, edges)[0].tolist(), "random": np.histogram(r, edges)[0].tolist(),
            "strategy_haircut": np.histogram(hc, edges)[0].tolist(),
            "fund_size": N}


def paired_diff(p, score_a, score_b, mult, split="test", n=2000, seed=3):
    """Mean multiple of top-decile-by-A minus top-decile-by-B on the same bootstrap draws."""
    d = p[p.split == split]
    a = top_mask(d, score_a.loc[d.index]).values
    b = top_mask(d, score_b.loc[d.index]).values
    m = mult.loc[d.index].values
    rng = np.random.default_rng(seed)
    ds = [m[k][a[k]].mean() - m[k][b[k]].mean() for k in (rng.choice(len(m), len(m)) for _ in range(n))]
    return {"diff": float(m[a].mean() - m[b].mean()),
            "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
            "overlap_share": float((a & b).sum() / max(a.sum(), 1))}


def unseal_holdout(fn):
    """Run fn() against the 2011 holdout exactly once; persist the result; refuse a second run."""
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    if os.path.exists(LOCK):
        with open(LOCK) as fh:
            rec = json.load(fh)
        rec["note"] = "HOLDOUT ALREADY SPENT - this is the cached single-touch result, not a re-run."
        return rec
    res = fn()
    rec = {"consumed_at": datetime.now(timezone.utc).isoformat(), "result": res}
    with open(LOCK, "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    return rec
