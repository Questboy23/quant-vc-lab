"""Round 3: founder and network features, the R1-R12 registry, rules R-A/R-B, and the 2012
lockbox (opened once). Uncapped calibrated multiples throughout (analysis/v2.py).

    python -m analysis.v3                 # registry + test window only
    python -m analysis.v3 --open-lockbox  # also opens 2012, ONCE (data/lockbox2012/.consumed.json)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

from analysis import v2
from analysis.hypotheses import logit, _std
from analysis.panel import ROOT
from analysis.strategy import top_mask

REG = os.path.join(ROOT, "vc", "hypotheses_v3.yaml")
OUT = os.path.join(ROOT, "reports", "v3.json")
LOCK = os.path.join(ROOT, "data", "lockbox2012", ".consumed.json")
NEW_FEATS = ["prior_exit", "serial", "bigtech_alum", "elite_school", "phd", "mba", "multi_founder",
             "log_ms_pre", "investor_repeat", "net_centrality", "founder_index", "log_people", "has_founder_record"]


def load():
    p = pd.read_parquet(os.path.join(ROOT, "data", "panel_v3.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data", "founders_v3.parquet"))
    p = p.join(f)
    p = p[~v2.already_public(p)]
    p["ipo_known"] = v2.known_ipo_values(p)
    p["smart_money"] = v2.oof_smart_money(p)
    p["founder_index"] = p.prior_exit + p.bigtech_alum + p.elite_school
    for c in ("prior_exit", "serial", "bigtech_alum", "elite_school", "phd", "mba", "log_ms_pre",
              "investor_repeat", "founder_index"):
        p[c] = p[c].where(p.has_founder_record == 1)          # unknown, not zero, without founders
    return p


def run_registry(p, mult):
    hyps = yaml.safe_load(open(REG))
    te = p[p.split == "test"].copy()
    mt = mult.loc[te.index]
    res = []
    for h in hyps:
        f = h["feature"]
        if h["kind"] == "returns":
            g1 = (te[f] == 1).values
            m = mt.values
            lo, hi, ple = v2.boot_ci(g1, np.ones(len(te), bool), m)
            keep = m < m.max()
            r = {"coef": float(m[g1].mean() - m.mean()), "p": float(min(1, 2 * min(ple, 1 - ple))), "n": int(g1.sum()),
                 "mult_group": float(m[g1].mean()), "mult_all": float(m.mean()), "ci": [lo, hi],
                 "mult_group_excl_top": float(m[g1 & keep].mean()), "mult_all_excl_top": float(m[keep].mean()),
                 "outliers_in_group": int((g1 & (m >= v2.OUTLIER_X)).sum()), "outliers_total": int((m >= v2.OUTLIER_X).sum())}
            res.append((h, r, {}))
            continue
        d = te[te.has_founder_record == 1].copy() if h.get("founder") else te[te.has_investor_record == 1].copy()
        d[f] = _std(p, f).loc[d.index]
        ctrl = ["log_people"] if h.get("founder") else []
        for c in ctrl:
            d[c] = _std(p, c).loc[d.index]
        prim = logit(d, f, h["primary"], ctrl)
        side = {o: logit(d, f, o, ctrl) for o in ("success", "bigwin", "closed")}
        # returns: feature-high group vs rest (uncapped, and without the single best deal)
        m = mt.loc[d.index].values
        x = d[f].values
        hi_g = x > np.nanmedian(x) if len(np.unique(x[~np.isnan(x)])) > 2 else x == np.nanmax(x)
        keep = m < m.max()
        side["returns"] = {"mult_hi": float(np.nanmean(m[hi_g])), "mult_lo": float(np.nanmean(m[~hi_g])),
                           "mult_hi_excl_top": float(m[hi_g & keep].mean()), "mult_lo_excl_top": float(m[~hi_g & keep].mean()),
                           "outlier_rate_hi": float((m[hi_g] >= v2.OUTLIER_X).mean()),
                           "outlier_rate_lo": float((m[~hi_g] >= v2.OUTLIER_X).mean()),
                           "n_hi": int(hi_g.sum()), "n_lo": int((~hi_g).sum())}
        res.append((h, prim, side))
    ps = [r["p"] if r and r.get("p") is not None and not np.isnan(r["p"]) else 1.0 for _, r, _ in res]
    holm = multipletests(ps, method="holm")[1]
    out = []
    for (h, r, side), ph in zip(res, holm):
        want = {"+": 1, "-": -1}.get(h["direction"], 0)
        if r is None:
            v = "untestable"
        elif ph >= 0.05:
            v = "no evidence"
        elif want and np.sign(r["coef"]) != want:
            v = "rejected (opposite sign)"
        else:
            v = "passes"
        out.append({**{k: h[k] for k in ("id", "name", "feature", "kind", "primary", "direction", "intuition")},
                    "result": {"verdict": v, "effect": r and r["coef"], "p_raw": r and r["p"], "p_holm": float(ph),
                               "n": r and r["n"], **({k: v2_ for k, v2_ in r.items() if k not in ("coef", "p", "n", "se")} if r else {}),
                               "side": {k: v_ for k, v_ in side.items()}}})
        print(f'{h["id"]:4s} {h["name"][:40]:40s} {v:24s} eff={r and round(r["coef"], 3)} p={r and r["p"]:.3g} holm={ph:.3g}')
    return out


def combined_score(p, extra_feats):
    """Round-2 ensemble (ridge + vote, big-win + outlier targets) over 28 old + 12 new features."""
    old = v2.ALL_FEATS
    v2.ALL_FEATS = old + extra_feats
    try:
        pb, ib = v2.ridge_all(p, "bigwin"); po, io = v2.ridge_all(p, "outlier")
        vb, _ = v2.dawes_vote(p, "bigwin"); vo, _ = v2.dawes_vote(p, "outlier")
    finally:
        v2.ALL_FEATS = old
    rk = lambda s: s.groupby(p.cohort).rank(pct=True)
    return (rk(pb) + rk(po) + rk(vb) + rk(vo)) / 4, {"ridge_bigwin_cv_auc": ib["train_cv_auc"], "ridge_outlier_cv_auc": io["train_cv_auc"]}


def rules(p, score, mult, split):
    d = p[p.split == split]
    m = mult.loc[d.index].values
    s = score.loc[d.index].values
    keep_top = m < m.max()
    out = {}
    w_all = np.ones(len(d))
    w_a = (s > 1 / 3).astype(float)
    w_b = w_a * np.where(s > 0.9, 2.0, 1.0)
    for name, w in (("own_everything", w_all), ("R-A_drop_bottom_third", w_a), ("R-B_plus_double_top10", w_b)):
        mm = (w * m).sum() / w.sum()
        mm_ex = (w * m)[keep_top].sum() / w[keep_top].sum()
        rng = np.random.default_rng(21); bs = []
        for _ in range(2000):
            b = rng.choice(len(d), len(d))
            bs.append((w[b] * m[b]).sum() / w[b].sum() / m[b].mean())
        out[name] = {"multiple": float(mm), "multiple_excl_top_deal": float(mm_ex),
                     "vs_everything_pct": float(100 * (mm / m.mean() - 1)),
                     "vs_everything_ci_pct": [float(100 * (np.percentile(bs, 2.5) - 1)), float(100 * (np.percentile(bs, 97.5) - 1))],
                     "outliers_held": int(((w > 0) & (m >= v2.OUTLIER_X)).sum()), "outliers_total": int((m >= v2.OUTLIER_X).sum()),
                     "deals": int((w > 0).sum())}
    return out


def open_lockbox(fn):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    if os.path.exists(LOCK):
        rec = json.load(open(LOCK)); rec["note"] = "LOCKBOX 2012 ALREADY SPENT - cached single-touch result"
        return rec
    rec = {"consumed_at": datetime.now(timezone.utc).isoformat(), "result": fn()}
    json.dump(rec, open(LOCK, "w"), indent=1, default=float)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--open-lockbox", action="store_true"); a = ap.parse_args()
    p = load()
    mult = v2.multiples(p, v2.ASSUMPTIONS["calibrated"])
    p["outlier"] = (mult >= v2.OUTLIER_X).astype(int)
    p.loc[p.split == "lockbox", "outlier"] = 0      # never used for fitting; models fit on train only
    out = {"generated": datetime.now(timezone.utc).isoformat()}
    out["registry"] = run_registry(p, mult)
    extra = ["prior_exit", "serial", "bigtech_alum", "elite_school", "phd", "mba", "multi_founder",
             "log_ms_pre", "investor_repeat", "net_centrality", "log_people", "has_founder_record"]
    score, info = combined_score(p, extra)
    score_old, _ = combined_score(p, [])
    out["model"] = info
    out["test_rules_v3"] = rules(p, score, mult, "test")
    out["test_rules_v2_features_only"] = rules(p, score_old, mult, "test")
    print("test v3", json.dumps(out["test_rules_v3"])[:700])
    print("test v2", json.dumps(out["test_rules_v2_features_only"])[:700])
    if a.open_lockbox or os.path.exists(LOCK):
        lk = p[p.split == "lockbox"]
        def _f():
            return {"rules_v3": rules(p, score, mult, "lockbox"), "rules_v2_features": rules(p, score_old, mult, "lockbox"),
                    "n": int(len(lk)), "years_of_follow_up": "3.0-3.9 (first rounds in 2012, outcomes frozen Dec 2015)",
                    "base": {"success": float(lk.success.mean()), "bigwin": float(lk.bigwin.mean()),
                             "mean_multiple": float(mult.loc[lk.index].mean())}}
        out["lockbox2012"] = open_lockbox(_f)
        print("LOCKBOX", json.dumps(out["lockbox2012"], default=float)[:1200])
    json.dump(out, open(OUT, "w"), indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


if __name__ == "__main__":
    main()
