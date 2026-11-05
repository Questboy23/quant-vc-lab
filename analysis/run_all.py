"""One command reproduces everything:   python -m analysis.run_all  [--rebuild] [--unseal-holdout]

Stages: panel -> Form D match -> derived features -> prior-findings reproduction ->
hypothesis registry (Holm + leakage) -> final score -> fund simulation -> (once) holdout ->
reports/results.json + vc/hypotheses.yaml results."""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "4")

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from analysis import formd, panel
from analysis.features import add_derived
from analysis.hypotheses import run_logit_hyp, verdict
from analysis.returns import BASE, as_dict, boot_diff, multiples, post_money
from analysis.strategy import (PRIOR_FEATURES, evaluate, fit_score, fund_sim, histogram, paired_diff,
                               sensitivity, top_mask, unseal_holdout)

ROOT = panel.ROOT
REG = os.path.join(ROOT, "vc", "hypotheses.yaml")
REPORTS = os.path.join(ROOT, "reports")
FULL = os.path.join(ROOT, "data", "panel_full.parquet")
# Features the final first-round score may use: known AT the first round, and with a train window.
FIRST_ROUND_USABLE = {"smart_money", "log_syndicate", "solo_investor", "log_crowding", "atypicality",
                      "kill_zone", "sf", "name_real_word", "dot_com", "exact_domain", "size_surprise",
                      "inv_momentum3", "log_appetite_bigtech", "log_appetite_pe", "log_appetite_strategic",
                      "name_sem_dist", "age_at_first", "n_inv_regions", "accelerator", "share_person",
                      "log_inv_prior_deals", "share_local", "n_categories", "hub_ny_bos", "name_has_digit",
                      "name_len", "second_tier_money", "cat_momentum"}


def load(rebuild: bool) -> pd.DataFrame:
    if rebuild or not os.path.exists(FULL):
        p = panel.build(); p.to_parquet(panel.OUT)
        formd.match(p)
        p = add_derived(p); p.to_parquet(FULL)
    return pd.read_parquet(FULL)


def data_facts(p):
    us_no_inv = p.has_investor_record.eq(0).mean()
    hp = p.groupby("status").has_homepage.mean().round(3).to_dict()
    return {
        "n_companies": int(len(p)), "by_split": p.split.value_counts().to_dict(),
        "share_no_investor_record": round(float(us_no_inv), 3),
        "homepage_rate_by_status": hp,
        "founded_date_rate_by_status": p.groupby("status").has_founded.mean().round(3).to_dict(),
        "base_rates": p.groupby("split")[["success", "bigwin", "closed"]].mean().round(4).to_dict(),
        "acq_price_disclosed_share": round(float(p.loc[p.status == "acquired", "acq_price"].notna().mean()), 3),
    }


def returns_hyps(p, mult, prior_prob, h):
    t = p[p.split == "test"]
    m = mult.loc[t.index].values
    tr = p[p.split == "train"]
    if h["id"] == "N5":
        top = (prior_prob.loc[t.index].groupby(t.cohort).rank(ascending=False, pct=True) <= 0.2).values
        g1, g0 = top & (t.sf.values == 0), top & (t.sf.values == 1)
        extra = {"post_nonsf": float(post_money(t).values[g1].mean()), "post_sf": float(post_money(t).values[g0].mean()),
                 "bigwin_nonsf": float(t.bigwin.values[g1].mean()), "bigwin_sf": float(t.bigwin.values[g0].mean())}
    elif h["id"] == "N10":
        cheap_cut = post_money(tr).quantile(1 / 3)
        thiel = ((t.log_crowding < tr.log_crowding.median()) & (t.atypicality > tr.atypicality.median())
                 & (t.solo_investor == 1)).values
        g1 = thiel & (post_money(t).values <= cheap_cut)
        g0 = np.ones(len(t), bool)
        extra = {"thiel_all_multiple": float(m[thiel].mean()) if thiel.any() else None,
                 "thiel_all_n": int(thiel.sum()), "cheap_cut_post": float(cheap_cut),
                 "bigwin_thiel_cheap": float(t.bigwin.values[g1].mean()) if g1.any() else None}
    elif h["id"] == "N19":
        g1, g0 = t.stage.isin(["seed", "angel"]).values, (t.stage == "A").values
        extra = {}
    if g1.sum() < 5 or g0.sum() < 5:
        return {"primary_result": None, "note": "too few deals", "n_hi": int(g1.sum())}
    diff, lo, hi, pv = boot_diff(g1, g0, m)
    r = {"coef": diff, "p": pv, "n": int(g1.sum() + g0.sum())}
    return {"primary_result": r, "returns": {"contrast": "see definition", "mult_hi": float(m[g1].mean()),
            "mult_lo": float(m[g0].mean()), "n_hi": int(g1.sum()), "n_lo": int(g0.sum()), "diff": diff,
            "ci": [lo, hi], "p": pv}, **extra}


def reproduce_prior(p, mult):
    """Section 2 of the brief: composite model + price-adjusted score, on TEST cohorts 2008-2010.
    (The prior analysis used 2008-2011; 2011 is our sealed holdout, so it is excluded here.)"""
    prob, info = fit_score(p, PRIOR_FEATURES)
    comp = evaluate(p, prob, prob, mult)
    pa = evaluate(p, prob, prob / post_money(p), mult)
    floors = {}
    for f in (0.5e6, 1e6, 2e6, 4e6):
        a = BASE.__class__(floor=f)
        m = multiples(p, a)
        floors[f"{f/1e6:g}M"] = evaluate(p, prob, prob / post_money(p, a), m)
    return prob, {"model": info, "composite_top_decile": comp, "price_adjusted_top_decile": pa,
                  "price_adjusted_by_floor": floors}


def write_registry(hyps):
    header = "".join(l for l in open(REG) if l.startswith("#"))
    clean = json.loads(json.dumps(hyps, default=float))
    with open(REG, "w") as fh:
        fh.write(header + "\n")
        yaml.safe_dump(clean, fh, sort_keys=False, width=110, allow_unicode=True)


def export_live_model(p):
    """Reduced model for the live Form D scorer: only surviving features a Form D reveals
    (Bay Area address, name features) plus round size. Fit on TRAIN cohorts only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    feats = ["sf", "name_real_word", "name_len", "log_raised"]
    tr = p[p.split == "train"]
    mu, sd = tr[feats].mean(), tr[feats].std()
    Z = (tr[feats] - mu) / sd
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, tr.bigwin)
    te = p[p.split == "test"]
    auc = roc_auc_score(te.bigwin, m.predict_proba((te[feats] - mu) / sd)[:, 1])
    out = {"features": feats, "coef": dict(zip(feats, m.coef_[0].round(6).tolist())),
           "intercept": float(m.intercept_[0]), "mean": mu.round(6).to_dict(), "sd": sd.round(6).to_dict(),
           "floor": BASE.floor, "train_auc": None, "test_auc_bigwin": round(float(auc), 4),
           "note": ("Reduced model: Form D shows no investors, so syndicate, accelerator and investor-"
                    "experience features cannot be scored. Mostly a round-size-and-location score.")}
    with open(os.path.join(REPORTS, "live_model.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def _p(x):
    """p-values keep 3 significant figures so tiny ones do not print as 0."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(f"{float(x):.3g}")


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--unseal-holdout", action="store_true")
    a = ap.parse_args()
    os.makedirs(REPORTS, exist_ok=True)

    p = load(a.rebuild)
    mult = multiples(p, BASE)
    facts = data_facts(p)
    print("facts", json.dumps(facts, default=str)[:600])

    prior_prob, prior = reproduce_prior(p, mult)
    print("prior composite", json.dumps(prior["composite_top_decile"], default=float)[:500])
    print("prior price-adj", json.dumps(prior["price_adjusted_top_decile"], default=float)[:500])

    hyps = yaml.safe_load(open(REG))
    for h in hyps:
        h.pop("result", None)
        if h["kind"] == "returns":
            h["_r"] = returns_hyps(p, mult, prior_prob, h)
        else:
            h["_r"] = run_logit_hyp(p, h, mult)

    def prim(h):
        r = h["_r"]
        return r.get("primary_result") if h["kind"] == "returns" else r["outcomes"].get(h["primary"])
    ps = [prim(h)["p"] if prim(h) and prim(h).get("p") is not None and not np.isnan(prim(h)["p"]) else 1.0 for h in hyps]
    holm = multipletests(ps, method="holm")[1]
    for h, ph in zip(hyps, holm):
        r = h.pop("_r")
        v = verdict(h, r, ph)
        pr = prim(dict(h, _r=r)) if False else (r.get("primary_result") if h["kind"] == "returns" else r["outcomes"].get(h["primary"]))
        h["result"] = {
            "verdict": v, "primary_outcome": h["primary"],
            "effect": _r(pr["coef"]) if pr else None,
            "effect_units": ("multiple difference" if h["kind"] == "returns" else
                             "LR chi2" if h["kind"] == "spline" else "log-odds per SD (or per unit if binary)"),
            "p_raw": _p(pr["p"]) if pr else None, "p_holm": _p(ph),
            "n": pr["n"] if pr else None, "events": pr.get("events") if pr else None,
            "all_outcomes": {o: ({"coef": _r(x["coef"]), "p": _p(x["p"]), "n": x["n"]} if x else None)
                             for o, x in r.get("outcomes", {}).items()},
            "complete_record": ({"coef": _r(r["complete_record"]["coef"]), "p": _p(r["complete_record"]["p"]),
                                 "n": r["complete_record"]["n"]} if r.get("complete_record") else None),
            "coverage_controlled": ({"coef": _r(r["coverage_controlled"]["coef"]), "p": _p(r["coverage_controlled"]["p"])}
                                    if r.get("coverage_controlled") else None),
            "missing_share": _r(r.get("missing_share"), 3),
            "missingness_predicts_outcome_p": _p((r.get("missingness_predicts_outcome") or {}).get("p")),
            "returns": ({k: (_r(v2) if not isinstance(v2, list) else [_r(x) for x in v2]) for k, v2 in r["returns"].items()
                         if k != "contrast"} | {"contrast": r["returns"]["contrast"]}) if r.get("returns") else None,
        }
        if "bands" in (r.get("outcomes", {}).get(h["primary"]) or {}):
            h["result"]["bands"] = r["outcomes"][h["primary"]]["bands"]
        for k in ("post_nonsf", "post_sf", "bigwin_nonsf", "bigwin_sf", "thiel_all_multiple", "thiel_all_n",
                  "cheap_cut_post", "bigwin_thiel_cheap"):
            if k in r:
                h["result"][k] = _r(r[k]) if isinstance(r[k], float) else r[k]
        print(f'{h["id"]:4s} {h["name"][:38]:38s} {v:26s} eff={h["result"]["effect"]} p={h["result"]["p_raw"]} holm={h["result"]["p_holm"]}')
    write_registry(hyps)

    # ---------- final score: survivors usable at the first round ----------
    survivors = [h["feature"] for h in hyps if h["result"]["verdict"].startswith("passes")
                 and h["feature"] in FIRST_ROUND_USABLE]
    feats = sorted(set(survivors))
    if not feats:
        feats = []  # size + stage only
    prob, info = fit_score(p, feats) if feats else fit_score(p, [])
    score = prob / post_money(p)
    final_test = evaluate(p, prob, score, mult)
    # Baselines that answer "do the features add anything beyond price?"
    size_prob, _ = fit_score(p, [])
    baselines = {
        "cheapest_only": evaluate(p, size_prob, 1 / post_money(p), mult),
        "size_stage_model_over_price": evaluate(p, size_prob, size_prob / post_money(p), mult),
    }
    baselines["final_minus_cheapest"] = paired_diff(p, score, 1 / post_money(p), mult)
    baselines["final_minus_size_only"] = paired_diff(p, score, size_prob / post_money(p), mult)
    print("baselines", json.dumps(baselines, default=float)[:900])
    sens = sensitivity(p, prob)
    t = p[p.split == "test"]
    sel = top_mask(t, score.loc[t.index]).values
    sm_, am_ = mult.loc[t.index].values[sel], mult.loc[t.index].values
    funds = fund_sim(sm_, am_)
    hist = histogram(sm_, am_)
    print("final", feats, json.dumps(final_test, default=float)[:400])

    export_live_model(p)
    holdout = None
    if a.unseal_holdout or os.path.exists(os.path.join(ROOT, "data", "holdout", ".consumed.json")):
        def _h():
            h = p[p.split == "holdout"]
            hs = top_mask(h, score.loc[h.index]).values
            hm = mult.loc[h.index].values
            return {"final_strategy": evaluate(p, prob, score, mult, split="holdout"),
                    "prior_price_adjusted": evaluate(p, prior_prob, prior_prob / post_money(p), mult, split="holdout"),
                    "prior_composite": evaluate(p, prior_prob, prior_prob, mult, split="holdout"),
                    "size_only_over_price": evaluate(p, size_prob, size_prob / post_money(p), mult, split="holdout"),
                    "final_minus_size_only": paired_diff(p, score, size_prob / post_money(p), mult, split="holdout"),
                    "sensitivity": sensitivity(p, prob, split="holdout"),
                    "funds": fund_sim(hm[hs], hm), "features": feats}
        holdout = unseal_holdout(_h)
        print("HOLDOUT", json.dumps(holdout, default=float)[:800])

    out = {"generated": pd.Timestamp.now(tz="UTC").isoformat(), "assumptions": as_dict(BASE),
           "assumption_label": BASE.label(), "facts": facts, "prior": prior,
           "hypotheses": [{k: h[k] for k in ("id", "source", "name", "intuition", "definition", "kind",
                                              "primary", "direction", "result")} for h in hyps],
           "final": {"features": feats, "model": info, "test": final_test, "baselines": baselines, "sensitivity": sens,
                     "funds": funds, "histogram": hist},
           "holdout": holdout}
    with open(os.path.join(REPORTS, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("wrote reports/results.json")


if __name__ == "__main__":
    main()
