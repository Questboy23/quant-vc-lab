"""Round 4: SEC Form D universe 2010-2021 (complete 4.5y windows), registry F1-F24, combined
score, four pre-declared portfolio rules, sealed 2018-2021 holdout (opened once).

    python -m analysis.v4                  # test window only
    python -m analysis.v4 --open-holdout   # + holdout, ONCE (data/holdout_fd/.consumed.json)
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "vc", "hypotheses_v4.yaml")
OUT = os.path.join(ROOT, "reports", "v4.json")
LOCK = os.path.join(ROOT, "data", "holdout_fd", ".consumed.json")
TRAIN, TEST, HOLD = range(2010, 2015), range(2015, 2018), range(2018, 2022)
FEATS = ["vc_director", "board_track", "board_pro", "serial_exec", "exec_prior_scaleup", "broker", "rule506c",
         "nonaccredited", "de_out_of_state", "bay_area", "hub", "nyc", "boston", "la", "size_vs_peers", "heat",
         "convertible", "rev_none", "rev_disclosed", "log_investors", "min_invest", "sold_share", "age_years",
         "has_ai", "n_dirs", "n_execs", "n_people", "more_than_year", "tech", "bio", "log_amount", "name_len",
         "board_prior_cos", "name_ambiguous"]


def load():
    e = pd.read_parquet(os.path.join(ROOT, "data", "fd", "panel.parquet"))
    e = e[(e.window_complete == 1) & (e.year <= 2021)].copy()
    e["split"] = np.select([e.year.isin(TRAIN), e.year.isin(TEST), e.year.isin(HOLD)], ["train", "test", "holdout"], "x")
    e["exit"] = ((e.ipo == 1) | (e.acquired_any == 1) | (e.unicorn == 1)).astype(int)
    e["outlier"] = ((e.outlier == 1) | (e.unicorn == 1)).astype(int)
    return e


def zs(e, f):
    tr = e.loc[e.split == "train", f].astype(float)
    x = e[f].astype(float)
    if set(tr.dropna().unique()) <= {0, 1}:
        return x
    return (x - tr.mean()) / (tr.std() or 1)


def logit(d, f, y, size=True):
    d = d.dropna(subset=[f])
    if d[y].sum() < 5 or d[f].nunique() < 2:
        return None
    X = pd.DataFrame({f: d[f].astype(float)})
    if size:
        X["log_amount"] = d.log_amount
    X["tech"] = d.tech; X["bio"] = d.bio
    for yr in sorted(d.year.unique())[1:]:
        X[f"y{yr}"] = (d.year == yr).astype(float)
    X = sm.add_constant(X.loc[:, X.std() > 0], has_constant="add")
    try:
        m = sm.Logit(d[y].astype(float), X).fit(disp=0, cov_type="HC1", maxiter=300)
    except Exception:
        return None
    return {"coef": float(m.params[f]), "p": float(m.pvalues[f]), "n": int(len(d)), "events": int(d[y].sum())}


def returns_contrast(d, f, m):
    x = d[f].astype(float)
    ok = x.notna().values
    xv = x.values
    hi = (xv == 1) if set(np.unique(xv[ok])) <= {0, 1} else (xv > np.nanmedian(xv))
    lo = ok & ~hi
    hi = ok & hi
    keep = m < m.max()
    return {"mult_hi": float(m[hi].mean()), "mult_lo": float(m[lo].mean()),
            "mult_hi_excl_top": float(m[hi & keep].mean()), "mult_lo_excl_top": float(m[lo & keep].mean()),
            "outlier_rate_hi": float(d.outlier.values[hi].mean()), "outlier_rate_lo": float(d.outlier.values[lo].mean()),
            "exit_rate_hi": float(d.exit.values[hi].mean()), "exit_rate_lo": float(d.exit.values[lo].mean()),
            "n_hi": int(hi.sum()), "n_lo": int(lo.sum())}


def registry(e):
    hyps = yaml.safe_load(open(REG))
    te = e[e.split == "test"].copy()
    m = te.markup.values
    res = []
    for h in hyps:
        f = h["feature"]
        te[f + "_z"] = zs(e, f).loc[te.index]
        prim = logit(te.rename(columns={f + "_z": "_f"}), "_f", h["primary"], not h.get("no_size_control"))
        side = {o: logit(te.rename(columns={f + "_z": "_f"}), "_f", o, not h.get("no_size_control"))
                for o in ("growth20", "outlier", "exit", "ipo")}
        side["returns"] = returns_contrast(te, f, m)
        res.append((h, prim, side))
    ps = [r["p"] if r else 1.0 for _, r, _ in res]
    holm = multipletests(ps, method="holm")[1]
    out = []
    for (h, r, side), ph in zip(res, holm):
        want = {"+": 1, "-": -1}.get(h["direction"], 0)
        v = ("untestable" if r is None else "no evidence" if ph >= 0.05 else
             "rejected (opposite sign)" if want and np.sign(r["coef"]) != want else "passes")
        out.append({**h, "result": {"verdict": v, "effect": r and r["coef"], "p_raw": r and r["p"], "p_holm": float(ph),
                                    "n": r and r["n"], "events": r and r["events"], "side": side}})
        rr = side["returns"]
        print(f'{h["id"]:4s} {h["name"][:42]:42s} {v:24s} eff={r and round(r["coef"],3)} holm={ph:.3g}  '
              f'markup(no top) {rr["mult_hi_excl_top"]:.2f}/{rr["mult_lo_excl_top"]:.2f}  outl {rr["outlier_rate_hi"]:.3f}/{rr["outlier_rate_lo"]:.3f}', flush=True)
    return out


def design(e, ref):
    X = pd.DataFrame(index=e.index)
    for f in FEATS:
        x = e[f].astype(float)
        if ref[f].isna().any() or x.isna().any():
            X[f + "_na"] = x.isna().astype(float)
        X[f] = x.fillna(ref[f].median())
    for ind in ("Other Technology", "Other", "Other Health Care", "Biotechnology", "Manufacturing", "Computers", "Pharmaceuticals"):
        X["ind_" + ind] = (e.industry == ind).astype(float)
    return X


def combined(e):
    tr = e[e.split == "train"]
    Xtr = design(tr, tr); X = design(e, tr)[Xtr.columns]
    scores, info = {}, {}
    for y in ("outlier", "growth20", "exit"):
        best = None
        for C in (0.003, 0.01, 0.03, 0.1, 0.3):
            pr = np.zeros(len(tr))
            for yr in TRAIN:
                msk = (tr.year == yr).values
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr[~msk], tr[y][~msk])
                pr[msk] = mdl.predict_proba(Xtr[msk])[:, 1]
            auc = roc_auc_score(tr[y], pr)
            best = (C, auc) if best is None or auc > best[1] else best
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=3000)).fit(Xtr, tr[y])
        scores["ridge_" + y] = pd.Series(mdl.predict_proba(X)[:, 1], index=e.index)
        info["ridge_" + y] = {"C": best[0], "train_cv_auc": round(best[1], 4),
                              "top_coefs": dict(sorted(zip(Xtr.columns, mdl[-1].coef_[0].round(3)), key=lambda t: -abs(t[1]))[:12])}
        # equal-weight vote signed by train correlation
        v = pd.Series(0.0, index=e.index)
        for f in FEATS:
            x = tr[f].astype(float); ok = x.notna()
            if ok.sum() < 100 or x[ok].std() == 0:
                continue
            r = np.corrcoef(x[ok], tr.loc[ok, y])[0, 1]
            if np.isnan(r):
                continue
            v += np.sign(r) * ((e[f].astype(float) - x[ok].mean()) / x[ok].std()).fillna(0).clip(-3, 3)
        scores["vote_" + y] = v
    rk = lambda s: s.groupby(e.year).rank(pct=True)
    ens = rk(sum(rk(s) for s in scores.values()) / len(scores))      # re-rank so top-k% is exactly k%
    return ens, scores, info


def remark(e, floor):
    """Paper markup recomputed with a valuation floor on the entry post-money."""
    pe = np.maximum(e.amount / 0.2, floor)
    m = (e.max_later / 0.15 * 0.85 ** np.minimum(e.k_before_max, 6) / pe).fillna(0)
    m = np.where(e.ipo == 1, np.maximum(m, 5e8 * 0.85 ** np.minimum(e.n_later, 6) / pe), m)
    u = (e.unicorn_val * 0.85 ** np.minimum(e.n_later, 6) / pe).fillna(0)
    out = e.copy()
    out["markup"] = np.maximum(m, u)
    out["outlier"] = ((out.markup >= 20) | (out.unicorn == 1)).astype(int)
    out.attrs["floor"] = floor
    return out


def rules(e, score, split, scores=None):
    d = e[e.split == split]
    m = d.markup.values; s = score.loc[d.index].values
    extra = []
    if scores is not None:
        post = np.maximum(e.amount / 0.20, e.attrs.get("floor", 1e6))
        pa = (scores["ridge_exit"] / post).groupby(e.year).rank(pct=True).loc[d.index].values
        cheap = (d.amount < d.groupby("year").amount.transform("median")).values
        extra = [("price_adj_top20", (pa > 0.8).astype(float)), ("quality_in_cheap_half", (cheap & (s > 0.5)).astype(float))]
    keep = m < m.max()
    res = {}
    for name, w in (("own_all", np.ones(len(d))), ("drop_bottom_third", (s > 1 / 3).astype(float)),
                    ("top_20pct", (s > 0.8).astype(float)), ("top_10pct", (s > 0.9).astype(float)), *extra):
        sel = w > 0
        rng = np.random.default_rng(31); bs = []
        for _ in range(2000):
            b = rng.choice(len(d), len(d))
            if sel[b].any():
                bs.append(m[b][sel[b]].mean() / m[b].mean())
        res[name] = {"deals": int(sel.sum()), "markup": float(m[sel].mean()), "markup_excl_top": float(m[sel & keep].mean()),
                     "vs_all_pct": float(100 * (m[sel].mean() / m.mean() - 1)),
                     "vs_all_ci_pct": [float(100 * (np.percentile(bs, 2.5) - 1)), float(100 * (np.percentile(bs, 97.5) - 1))],
                     "outliers": int(d.outlier.values[sel].sum()), "outliers_total": int(d.outlier.sum()),
                     "unicorns": int(d.unicorn.values[sel].sum()), "unicorns_total": int(d.unicorn.sum()),
                     "ipos": int(d.ipo.values[sel].sum()), "ipos_total": int(d.ipo.sum()),
                     "exits": int(d.exit.values[sel].sum()), "exits_total": int(d.exit.sum()),
                     "outlier_rate": float(d.outlier.values[sel].mean()), "growth20_rate": float(d.growth20.values[sel].mean())}
    try:
        res["auc_outlier"] = float(roc_auc_score(d.outlier, s)); res["auc_exit"] = float(roc_auc_score(d.exit, s))
        res["auc_growth20"] = float(roc_auc_score(d.growth20, s))
    except ValueError:
        pass
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--open-holdout", action="store_true"); a = ap.parse_args()
    e = load()
    out = {"generated": datetime.now(timezone.utc).isoformat(),
           "n": {s: int((e.split == s).sum()) for s in ("train", "test", "holdout")},
           "base_rates": e.groupby("split")[["next_raise", "growth20", "big75", "ipo", "unicorn", "acquired_any", "exit", "outlier"]].mean().round(4).to_dict()}
    tail = {}
    for s in ("train", "test", "holdout"):
        ms = np.sort(e.loc[e.split == s, "markup"].values)[::-1]
        tail[s] = {"share_value_top1pct": float(ms[: len(ms) // 100].sum() / ms.sum()), "share_zero": float((ms == 0).mean()),
                   "share_below_1x": float((ms < 1).mean()), "mean": float(ms.mean())}
    out["tail"] = tail
    out["registry"] = registry(e)
    ens, scores, info = combined(e)
    out["model"] = info
    out["test_rules"] = rules(e, ens, "test", scores)
    out["test_rules_by_floor"] = {f"{fl/1e6:g}M": rules(remark(e, fl), ens, "test", scores) for fl in (3e6, 5e6, 8e6)}
    print("TEST", json.dumps(out["test_rules"])[:1600], flush=True)
    if a.open_holdout or os.path.exists(LOCK):
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        if os.path.exists(LOCK):
            rec = json.load(open(LOCK)); rec["note"] = "HOLDOUT 2018-21 ALREADY SPENT - cached"
        else:
            rec = {"consumed_at": datetime.now(timezone.utc).isoformat(), "primary": "price_adj_top20 @ $5M floor",
                   "result": rules(e, ens, "holdout", scores),
                   "by_floor": {f"{fl/1e6:g}M": rules(remark(e, fl), ens, "holdout", scores) for fl in (3e6, 5e6, 8e6)}}
            json.dump(rec, open(LOCK, "w"), indent=1, default=float)
        out["holdout"] = rec
        print("HOLDOUT", json.dumps(rec)[:1800])
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()


def export_live_model():
    """Fit the round-4 ridge models on ALL complete-window cohorts (2010-2021) for forward use
    (no evaluation happens on this fit) and save them with the reference distributions the live
    rule needs: quality percentile and the 'cheaper half' amount cut-off."""
    import joblib
    e = load()
    X = design(e, e)
    models = {}
    for y in ("growth20", "exit", "outlier"):
        C = json.load(open(OUT))["model"]["ridge_" + y]["C"]
        models[y] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X, e[y])
    q = sum(pd.Series(m.predict_proba(X)[:, 1], index=e.index).rank(pct=True) for m in models.values()) / 3
    recent = e[e.year >= 2019]
    ref = {"columns": list(X.columns), "medians": {f: float(e[f].median()) for f in FEATS},
           "quality_ref": np.sort(q.loc[recent.index].values).tolist()[:: max(1, len(recent) // 2000)],
           "prob_ref": {y: np.sort(models[y].predict_proba(X.loc[recent.index])[:, 1]).tolist()[:: max(1, len(recent) // 2000)] for y in models},
           "cheap_cutoff_amount": float(recent.amount.median()),
           "heat_recent": float(e.heat.dropna().iloc[-500:].mean())}
    joblib.dump({"models": models, "ref": ref}, os.path.join(ROOT, "reports", "live_model_v4.joblib"))
    print("exported live model; cheap cut-off", ref["cheap_cutoff_amount"])


if __name__ == "__main__" and os.environ.get("EXPORT_LIVE"):
    export_live_model()
