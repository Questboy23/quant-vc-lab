"""Run every registered hypothesis on the TEST cohorts (2008-2010), with Holm correction and
leakage checks. Never touches the holdout."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from analysis.returns import boot_diff

warnings.filterwarnings("ignore")

OUTCOMES = ("success", "bigwin", "closed")
COVERAGE = ["has_investor_record", "has_homepage", "has_founded", "raised_missing"]


def _design(d: pd.DataFrame, cols, size_control=True, extra=()):
    X = d[list(cols) + list(extra)].astype(float).copy()
    if size_control:
        X["log_raised"] = d.log_raised
    for s in ("seed", "angel"):
        X["stage_" + s] = (d.stage == s).astype(float)
    for y in sorted(d.cohort.unique())[1:]:
        X[f"y{y}"] = (d.cohort == y).astype(float)
    X = X.loc[:, X.std() > 0]
    return sm.add_constant(X, has_constant="add")


def logit(d, feat, outcome, controls=(), size_control=True, extra=()):
    d = d.dropna(subset=[feat, *controls])
    if d[outcome].sum() < 5 or d[feat].nunique() < 2:
        return None
    X = _design(d, [feat, *controls], size_control, extra)
    if feat not in X:
        return None
    try:
        m = sm.Logit(d[outcome].astype(float), X).fit(disp=0, cov_type="HC1", maxiter=200)
    except Exception:
        try:
            m = sm.Logit(d[outcome].astype(float), X).fit_regularized(disp=0, alpha=1e-4)
            return {"coef": float(m.params[feat]), "p": np.nan, "n": int(len(d)), "events": int(d[outcome].sum())}
        except Exception:
            return None
    return {"coef": float(m.params[feat]), "se": float(m.bse[feat]), "p": float(m.pvalues[feat]),
            "n": int(len(d)), "events": int(d[outcome].sum())}


def spline_lr(d, feat, outcome):
    from patsy import dmatrix
    d = d.dropna(subset=[feat])
    base = _design(d, [], True)
    S = dmatrix(f"cr(x, df=3) - 1", {"x": d[feat].clip(d[feat].quantile(.01), d[feat].quantile(.99))},
                return_type="dataframe")
    S.index = d.index
    full = pd.concat([base, S.add_prefix("s_")], axis=1)
    y = d[outcome].astype(float)
    try:
        m0 = sm.Logit(y, base).fit(disp=0, maxiter=200)
        m1 = sm.Logit(y, full).fit(disp=0, maxiter=200)
    except Exception:
        return None
    lr = 2 * (m1.llf - m0.llf)
    dfree = full.shape[1] - base.shape[1]
    # report outcome rate by age band for readability
    bands = pd.cut(d[feat], [-1, 0.5, 1.5, 3, 6, 40], labels=["<0.5y", "0.5-1.5y", "1.5-3y", "3-6y", "6y+"])
    rates = d.groupby(bands, observed=True)[outcome].agg(["mean", "size"])
    return {"coef": float(lr), "p": float(stats.chi2.sf(lr, dfree)), "n": int(len(d)),
            "events": int(y.sum()), "lr_df": int(dfree),
            "bands": {str(k): {"rate": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in rates.iterrows()}}


def _std(p, feat):
    tr = p.loc[p.split == "train", feat].dropna()
    ref = tr if len(tr) > 30 and tr.std() > 0 else p[feat].dropna()
    if set(ref.unique()) <= {0, 1}:
        return p[feat]
    return (p[feat] - ref.mean()) / ref.std()


def _groups(p, feat, test):
    """Returns-side contrast: binary -> 1 vs 0; continuous -> top vs bottom tercile (train cut-offs)."""
    x = test[feat]
    if set(x.dropna().unique()) <= {0, 1}:
        return (x == 1).values, (x == 0).values, "1 vs 0"
    tr = p.loc[p.split == "train", feat].dropna()
    ref = tr if len(tr) > 30 else x.dropna()
    lo, hi = ref.quantile([1 / 3, 2 / 3])
    return (x > hi).values, (x <= lo).values, "top vs bottom tercile"


def run_logit_hyp(p, h, mult):
    test = p[p.split == "test"].copy()
    if h.get("subset"):
        test = test.query(h["subset"])
    feat = h["feature"]
    test[feat] = _std(p, feat).loc[test.index]
    ctrl = h.get("controls", []) or []
    for c in ctrl:
        test[c] = _std(p, c).loc[test.index]
    size = not h.get("no_size_control", False)
    res = {"outcomes": {}}
    for o in OUTCOMES:
        if h["kind"] == "spline":
            res["outcomes"][o] = spline_lr(test, feat, o)
        else:
            res["outcomes"][o] = logit(test, feat, o, ctrl, size)
    prim = h["primary"]
    # leakage 1: complete-record subsample
    cr = test[test.complete_record == 1]
    res["complete_record"] = spline_lr(cr, feat, prim) if h["kind"] == "spline" else logit(cr, feat, prim, ctrl, size)
    # leakage 2: add coverage indicators as controls
    res["coverage_controlled"] = None if h["kind"] == "spline" else logit(test, feat, prim, ctrl, size, extra=COVERAGE)
    # leakage 3: does MISSINGNESS of the feature predict the outcome?
    miss = test[feat].isna()
    res["missing_share"] = float(miss.mean())
    if 0.05 < miss.mean() < 0.95:
        t2 = test.assign(_miss=miss.astype(float))
        res["missingness_predicts_outcome"] = logit(t2, "_miss", prim)
    # returns contrast
    tm = mult.loc[test.index].values
    ok = test[feat].notna().values
    g1, g0, lab = _groups(p, feat, p.loc[test.index])
    g1, g0 = g1 & ok, g0 & ok
    if g1.sum() >= 10 and g0.sum() >= 10:
        diff, lo, hi, pv = boot_diff(g1, g0, tm)
        res["returns"] = {"contrast": lab, "mult_hi": float(tm[g1].mean()), "mult_lo": float(tm[g0].mean()),
                          "n_hi": int(g1.sum()), "n_lo": int(g0.sum()), "diff": diff, "ci": [lo, hi], "p": pv}
    return res


def verdict(h, r, p_holm):
    prim = r["outcomes"].get(h["primary"]) if "outcomes" in r else r.get("primary_result")
    if prim is None or p_holm is None or np.isnan(p_holm):
        return "untestable"
    sign = np.sign(prim["coef"])
    want = {"+": 1, "-": -1}.get(h["direction"], 0)
    if h["kind"] == "spline" or h["kind"] == "returns" and want == 0:
        sign = want = 0
    if p_holm >= 0.05:
        return "no evidence"
    if want and sign != want:
        return "rejected (opposite sign)"
    checks = []
    for k in ("complete_record", "coverage_controlled"):
        c = r.get(k)
        if c is None:
            continue
        checks.append(c["p"] < 0.05 and (want == 0 or np.sign(c["coef"]) == sign))
    mp = r.get("missingness_predicts_outcome")
    leak = mp is not None and mp["p"] < 0.05
    if checks and not all(checks):
        return "coverage artifact"
    return "passes (leakage flag)" if leak else "passes"
