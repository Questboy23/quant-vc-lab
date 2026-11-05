"""Round 2 (2026-10-01): outliers first, a combined "every weak signal" method, how to invest
(diversification, stage, follow-on, vintage), emerging-manager / persistence tests, and two
critique passes (risk/portfolio, then pricing/adverse selection).

Ground rules unchanged: every model is fit on TRAIN (2003-07); TEST (2008-10) is where results
are read. The 2011 holdout is SPENT; anything shown for 2011 here is a labelled second look.

Metrics are UNCAPPED multiples (VC is about the tail) with bootstrap intervals, alongside the
old 100x-capped mean for continuity. Percent returns only.

    python -m analysis.v2      -> reports/v2.json
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.features import _investments, investor_rankings
from analysis.panel import ROOT
from analysis.strategy import top_mask

warnings.filterwarnings("ignore")
OUT = os.path.join(ROOT, "reports", "v2.json")
CB13 = os.path.join(ROOT, "data", "raw", "cb2013")
OUTLIER_X = 20.0           # a deal returning >= 20x its entry is an outlier
rng0 = np.random.default_rng(11)

# All 28 pre-registered features that exist at the first round (regardless of test verdict:
# including everything avoids choosing features on the test window).
ALL_FEATS = ["smart_money", "log_syndicate", "solo_investor", "log_crowding", "atypicality", "kill_zone",
             "sf", "name_real_word", "dot_com", "exact_domain", "size_surprise", "inv_momentum3",
             "log_appetite_bigtech", "log_appetite_pe", "log_appetite_strategic", "name_sem_dist",
             "age_at_first", "n_inv_regions", "accelerator", "share_person", "log_inv_prior_deals",
             "share_local", "n_categories", "hub_ny_bos", "name_has_digit", "name_len",
             "second_tier_money", "cat_momentum"]

ASSUMPTIONS = {
    # what round 1 used
    "base": dict(dil_seed=0.20, dil_a=0.30, floor=2e6, ipo_default=500e6, use_known_ipo=False),
    # calibrated on the 2013 Crunchbase dump's real post-money rows (angel median 12.9% sold,
    # $1.35M post; Series A 25%, $8.0M) and real IPO valuations (2007+ median $314M); known IPO
    # valuations used where they exist
    "calibrated": dict(dil_seed=0.13, dil_a=0.25, floor=1e6, ipo_default=314e6, use_known_ipo=True),
}


# ----------------------------------------------------------------------------- data
def known_ipo_values(p: pd.DataFrame) -> pd.Series:
    obj = pd.read_csv(os.path.join(CB13, "objects.csv"), usecols=["id", "entity_type", "permalink"], low_memory=False)
    obj = obj[obj.entity_type == "Company"]
    obj["slug"] = obj.permalink.str.replace("/company/", "", regex=False)
    ipo = pd.read_csv(os.path.join(CB13, "ipos.csv"))
    ipo = ipo[(ipo.valuation_amount > 0) & (ipo.valuation_currency_code == "USD")]
    ipo = ipo.merge(obj[["id", "slug"]], left_on="object_id", right_on="id")
    v = ipo.groupby("slug").valuation_amount.max()
    slug = pd.Series(p.index.str.replace("/organization/", "", regex=False), index=p.index)
    return slug.map(v)


def already_public(p: pd.DataFrame) -> pd.Series:
    """Audit fix (2026-10-01): companies whose IPO (2013 dump public_at) predates their 'first
    round' were already public - e.g. Emcore (IPO 1997) - and are not venture deals."""
    obj = pd.read_csv(os.path.join(CB13, "objects.csv"), usecols=["id", "entity_type", "permalink"], low_memory=False)
    obj = obj[obj.entity_type == "Company"]
    obj["slug"] = obj.permalink.str.replace("/company/", "", regex=False)
    ipo = pd.read_csv(os.path.join(CB13, "ipos.csv")).merge(obj[["id", "slug"]], left_on="object_id", right_on="id")
    pub = pd.to_datetime(ipo.public_at, errors="coerce").groupby(ipo.slug).min()
    d = pd.Series(p.index.str.replace("/organization/", "", regex=False), index=p.index).map(pub)
    return (d.notna() & (d <= p.first_date)).astype(bool)


def oof_smart_money(p: pd.DataFrame) -> pd.Series:
    """Out-of-fold top-40 flag for TRAIN rows (5 folds), so the train fit never sees a feature
    built from its own label. Test/holdout rows keep the full-train ranking."""
    i = _investments()
    fr = i[i.funding_round_permalink.isin(p.first_round)].groupby("funding_round_permalink").investor_permalink.apply(set)
    out = p.smart_money.astype(float).copy()
    tr_idx = p.index[p.split == "train"].to_numpy()
    folds = np.array_split(rng0.permutation(tr_idx), 5)
    for f in folds:
        q = p.copy()
        q.loc[f, "split"] = "fold_out"
        top = set(investor_rankings(q, i).index[:40])
        out.loc[f] = [float(bool(s & top)) if isinstance(s, set) else 0.0 for s in p.loc[f, "first_round"].map(fr)]
    return out


def multiples(p: pd.DataFrame, a: dict, entry="first") -> pd.Series:
    """UNCAPPED price-adjusted multiple. entry='first' (round 1) or 'next' (follow-on at round 2)."""
    if entry == "first":
        raised = p.raised_filled
        dil = np.where(p.stage == "A", a["dil_a"], a["dil_seed"])
        later = p.n_later_rounds
    else:
        raised = p.next_raised.fillna(p.next_raised.median())
        dil = a["dil_a"]
        later = (p.n_later_rounds - 1).clip(lower=0)
    post = np.maximum(raised / dil, a["floor"] if entry == "first" else a["floor"] * 3)
    ipo_v = p.ipo_known.fillna(a["ipo_default"]) if a["use_known_ipo"] else a["ipo_default"]
    funding = p.funding_total_usd.fillna(p.raised_filled)
    ev = np.select([p.status == "ipo", (p.status == "acquired") & p.acq_price.notna(), p.status == "acquired"],
                   [ipo_v, p.acq_price, funding], 0.0)
    return pd.Series(ev * 0.8 ** np.minimum(later, 6) / post, index=p.index)


# ----------------------------------------------------------------------------- models
def design(p, feats, ref):
    X = pd.DataFrame(index=p.index)
    for f in feats:
        x = p[f].astype(float)
        if ref[f].isna().any() or x.isna().any():
            X[f + "_na"] = x.isna().astype(float)
        X[f] = x.fillna(ref[f].median())
    X["log_raised"] = p.log_raised
    X["seed"] = (p.stage == "seed").astype(float)
    X["angel"] = (p.stage == "angel").astype(float)
    return X


def ridge_all(p, target):
    """L2 logistic on all 28 features; C chosen by leave-one-cohort-out AUC on train only."""
    tr = p[p.split == "train"]
    Xtr = design(tr, ALL_FEATS, tr)
    y = tr[target].values
    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        pr = np.zeros(len(tr))
        for c in tr.cohort.unique():
            m = tr.cohort.values == c
            mod = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr[~m], y[~m])
            pr[m] = mod.predict_proba(Xtr[m])[:, 1]
        auc = roc_auc_score(y, pr)
        if best is None or auc > best[1]:
            best = (C, auc)
    mod = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=3000)).fit(Xtr, y)
    X = design(p, ALL_FEATS, tr)[Xtr.columns]
    coefs = dict(zip(Xtr.columns, mod[-1].coef_[0].round(4)))
    return pd.Series(mod.predict_proba(X)[:, 1], index=p.index), {"C": best[0], "train_cv_auc": round(best[1], 4), "coef": coefs}


def dawes_vote(p, target):
    """Improper linear model: z-score each feature on train, sign it by its TRAIN univariate
    correlation with the target, add them up with equal weight. Robust to overfitting."""
    tr = p[p.split == "train"]
    s = pd.Series(0.0, index=p.index)
    signs = {}
    for f in ALL_FEATS:
        x = tr[f].astype(float)
        ok = x.notna()
        if x[ok].std() == 0 or ok.sum() < 50:
            continue
        r = np.corrcoef(x[ok], tr.loc[ok, target])[0, 1]
        if np.isnan(r):
            continue
        sg = np.sign(r)
        signs[f] = int(sg)
        z = ((p[f].astype(float) - x[ok].mean()) / x[ok].std()).fillna(0).clip(-3, 3)
        s += sg * z
    return s, {"signs": signs}


# ----------------------------------------------------------------------------- metrics
def boot_ci(sel, base, m, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(m)); ds = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        s, bb = sel[b], base[b]
        if s.any() and bb.any():
            ds.append(m[b][s].mean() - m[b][bb].mean())
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(np.array(ds) <= 0))]


def portfolio_stats(d, sel, m, outl):
    base = np.ones(len(d), bool)
    lo, hi, p_le0 = boot_ci(sel, base, m)
    cap = np.minimum(m, 100)
    share_sel = sel.mean()
    cap_rate = (outl & sel).sum() / max(outl.sum(), 1)
    keep = m < m.max()                       # drop the single biggest deal in the window
    return {"mean_excl_top_deal": float(m[sel & keep].mean()), "mean_all_excl_top_deal": float(m[keep].mean()),
            "top_deal_selected": bool(sel[np.argmax(m)]),
            "n_selected": int(sel.sum()), "n": int(len(d)),
            "mean_uncapped": float(m[sel].mean()), "mean_all_uncapped": float(m.mean()),
            "excess_uncapped_pct": float(100 * (m[sel].mean() / m.mean() - 1)),
            "excess_ci": [lo, hi], "p_excess_le_0": p_le0,
            "mean_capped100": float(cap[sel].mean()), "mean_all_capped100": float(cap.mean()),
            "outliers_captured": int((outl & sel).sum()), "outliers_total": int(outl.sum()),
            "outlier_capture_lift": float(cap_rate / share_sel) if share_sel else None,
            "outlier_rate_selected": float(outl[sel].mean()), "outlier_rate_all": float(outl.mean())}


def fund_curve(pool, sizes=(5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500), n=20000, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for N in sizes:
        draws = rng.choice(pool, (n, N))
        f = draws.mean(1)
        out[N] = {"p_loss": float((f < 1).mean()), "median": float(np.median(f)), "mean": float(f.mean()),
                  "p5": float(np.percentile(f, 5)), "p95": float(np.percentile(f, 95)),
                  "p_2x": float((f >= 2).mean()), "p_3x": float((f >= 3).mean()),
                  "p_has_outlier": float((draws >= OUTLIER_X).any(1).mean()),
                  "exp_log_growth": float(np.mean(np.log(np.maximum(f, 1e-3)))),
                  "cvar5": float(np.sort(f)[: n // 20].mean())}
    return out


# ----------------------------------------------------------------------------- investors
def investor_analysis(p, mult):
    """Emerging managers, persistence, and 'top 10 firms' vs 'whole market', deal-level.
    A firm's portfolio = panel companies where it was in the FIRST round (the only entry we can
    price). Firms = organisation investors only."""
    i = _investments()
    org = i[i.investor_permalink.str.startswith("/organization/")].dropna(subset=["funded_at"])
    first_deal = org.groupby("investor_permalink").funded_at.min()
    fr = org[org.funding_round_permalink.isin(p.first_round)][["funding_round_permalink", "investor_permalink", "funded_at"]]
    fr = fr.merge(p.reset_index()[["permalink", "first_round", "split", "cohort"]],
                  left_on="funding_round_permalink", right_on="first_round")
    fr["firm_age"] = (fr.funded_at - fr.investor_permalink.map(first_deal)).dt.days / 365.25
    fr["m"] = fr.permalink.map(mult)
    fr["outlier"] = fr.m >= OUTLIER_X
    res = {}
    te = fr[fr.split == "test"]
    # deal level: emerging (firm <= 3y old at the deal) vs established (>= 8y)
    for lab, mask in (("emerging_le3y", te.firm_age <= 3), ("mid_3to8y", (te.firm_age > 3) & (te.firm_age < 8)),
                      ("established_ge8y", te.firm_age >= 8)):
        x = te[mask].drop_duplicates(["investor_permalink", "permalink"])
        res[lab] = {"deals": int(len(x)), "firms": int(x.investor_permalink.nunique()),
                    "mean_multiple": float(x.m.mean()), "outlier_rate": float(x.outlier.mean()),
                    "median_multiple": float(x.m.median())}
    # firm-level portfolios in test
    tf = te.drop_duplicates(["investor_permalink", "permalink"]).groupby("investor_permalink").agg(
        n=("m", "size"), mult=("m", "mean"), outl=("outlier", "sum"))
    tf = tf[tf.n >= 5]
    # persistence: rank firms on TRAIN-cohort deals (>=5), look at their TEST deals
    trf = fr[fr.split == "train"].drop_duplicates(["investor_permalink", "permalink"]).groupby("investor_permalink").agg(
        n=("m", "size"), mult=("m", "mean"))
    trf = trf[trf.n >= 5]
    both = trf.join(tf, lsuffix="_tr", rsuffix="_te", how="inner")
    top10 = trf.sort_values("mult", ascending=False).head(10).index
    t10 = te[te.investor_permalink.isin(top10)].drop_duplicates("permalink")
    from scipy.stats import spearmanr
    rho = spearmanr(both.mult_tr, both.mult_te) if len(both) > 10 else (np.nan, np.nan)
    q = both.mult_tr.rank(pct=True)
    res["persistence"] = {"firms_in_both": int(len(both)), "spearman_rho": float(rho[0]), "p": float(rho[1]),
                          "top_quartile_train_then_test_mult": float(both[q >= .75].mult_te.mean()),
                          "bottom_quartile_train_then_test_mult": float(both[q <= .25].mult_te.mean())}
    res["top10_by_train"] = {"test_deals": int(len(t10)), "test_mean_multiple": float(t10.m.mean()),
                             "test_outlier_rate": float(t10.outlier.mean())}
    res["top10_by_test_hindsight"] = {"mean_multiple": float(tf.sort_values("mult", ascending=False).head(10).mult.mean()),
                                      "note": "hindsight - not investable, the 'best 10 firms' benchmark"}
    res["all_firms_market"] = {"test_deals_equal_weight": float(te.drop_duplicates("permalink").m.mean()),
                               "firm_equal_weight": float(tf.mult.mean()), "firms": int(len(tf)),
                               "firm_dispersion": {k: float(v) for k, v in tf.mult.quantile([.1, .25, .5, .75, .9]).items()},
                               "share_of_firms_below_1x": float((tf.mult < 1).mean())}
    # fund of emerging managers: firms whose first deal was 2005-2008, their test-cohort deals
    new_firms = first_deal[(first_deal.dt.year >= 2005) & (first_deal.dt.year <= 2008)].index
    nf = tf[tf.index.isin(new_firms)]
    res["fund_of_emerging_firms"] = {"firms": int(len(nf)), "firm_equal_weight_multiple": float(nf.mult.mean()) if len(nf) else None,
                                     "deal_level": float(te[te.investor_permalink.isin(new_firms)].drop_duplicates("permalink").m.mean())}
    # bootstrap a fund-of-funds of K firms (equal weight across firm portfolios)
    rng = np.random.default_rng(9)
    fof = {}
    for lab, pool in (("emerging_firms", nf.mult.values), ("all_firms", tf.mult.values)):
        if len(pool) < 5:
            continue
        for K in (5, 10, 20):
            sims = rng.choice(pool, (20000, K)).mean(1)
            fof[f"{lab}_{K}"] = {"median": float(np.median(sims)), "p_loss": float((sims < 1).mean()),
                                 "p5": float(np.percentile(sims, 5)), "p95": float(np.percentile(sims, 95))}
    res["fund_of_funds_sim"] = fof
    return res


# ----------------------------------------------------------------------------- main
def main():
    p = pd.read_parquet(os.path.join(ROOT, "data", "panel_full.parquet"))
    pub = already_public(p)
    p = p[~pub]                                    # audit fix: not venture deals
    p["ipo_known"] = known_ipo_values(p)
    p["smart_money_insample"] = p.smart_money
    p["smart_money"] = oof_smart_money(p)
    out = {"generated": pd.Timestamp.now(tz="UTC").isoformat(), "outlier_threshold_x": OUTLIER_X,
           "excluded_already_public": int(pub.sum()),
           "ipo_known_n": int(p.ipo_known.notna().sum()), "ipo_status_n": int((p.status == "ipo").sum())}

    res_by_assump = {}
    for an, a in ASSUMPTIONS.items():
        m = multiples(p, a)
        p["outlier"] = (m >= OUTLIER_X).astype(int)
        te = p[p.split == "test"]
        mt, ot = m.loc[te.index].values, p.outlier.loc[te.index].values.astype(bool)
        r = {"tail": {}}
        for s in ("train", "test", "holdout"):
            ms = np.sort(m[p.split == s].values)[::-1]
            r["tail"][s] = {"n": int(len(ms)), "outliers": int((ms >= OUTLIER_X).sum()),
                            "share_value_top1pct": float(ms[: max(1, len(ms) // 100)].sum() / ms.sum()),
                            "share_value_top5pct": float(ms[: max(1, len(ms) // 20)].sum() / ms.sum()),
                            "share_deals_lose_money": float((ms < 1).mean()),
                            "mean_multiple": float(ms.mean())}
        scores = {}
        prob_b, info_b = ridge_all(p, "bigwin")
        prob_o, info_o = ridge_all(p, "outlier")
        vote_b, vinfo = dawes_vote(p, "bigwin")
        vote_o, _ = dawes_vote(p, "outlier")
        rank = lambda s: s.groupby(p.cohort).rank(pct=True)
        ens = (rank(prob_b) + rank(prob_o) + rank(vote_b) + rank(vote_o)) / 4
        dil = np.where(p.stage == "A", a["dil_a"], a["dil_seed"])
        post = np.maximum(p.raised_filled / dil, a["floor"])
        scores.update({"combined_ridge_bigwin": prob_b, "combined_ridge_outlier": prob_o,
                       "combined_vote_bigwin": vote_b, "combined_vote_outlier": vote_o,
                       "combined_ensemble": ens, "combined_ensemble_price_adj": ens / np.log10(post),
                       "ridge_outlier_price_adj": prob_o / post})
        r["models"] = {"ridge_bigwin": info_b, "ridge_outlier": info_o, "vote_signs": vinfo["signs"]}
        r["strategies"] = {}
        for k, sc in scores.items():
            sel = top_mask(te, sc.loc[te.index]).values
            st = portfolio_stats(te, sel, mt, ot)
            try:
                st["auc_outlier"] = float(roc_auc_score(ot, sc.loc[te.index]))
            except ValueError:
                st["auc_outlier"] = None
            r["strategies"][k] = st
        # each single signal alone ("works a little bit")
        singles = {}
        for f in ALL_FEATS:
            x = p[f].astype(float)
            trx = x[p.split == "train"]
            ok = trx.notna()
            if ok.sum() < 50 or trx[ok].std() == 0:
                continue
            sg = np.sign(np.corrcoef(trx[ok], p.loc[trx[ok].index, "bigwin"])[0, 1]) or 1
            sc = (sg * x).fillna(-1e9 if sg > 0 else 1e9) if False else (sg * x).fillna((sg * x).min())
            sel = top_mask(te, sc.loc[te.index] + 1e-9 * rng0.random(len(te))).values
            singles[f] = {"mean_uncapped": float(mt[sel].mean()), "excess_pct": float(100 * (mt[sel].mean() / mt.mean() - 1)),
                          "outlier_lift": float((ot & sel).sum() / max(ot.sum(), 1) / sel.mean())}
        r["single_signals"] = singles
        r["n_single_positive"] = int(sum(v["excess_pct"] > 0 for v in singles.values()))

        # ---------------- how to invest ----------------
        best_key = "combined_ensemble"
        sel_best = top_mask(te, scores[best_key].loc[te.index]).values
        r["diversification"] = {"random": fund_curve(mt), "combined_ensemble": fund_curve(mt[sel_best])}
        # stage
        r["stage"] = {st: {"n": int((te.stage == st).sum()), "mean_uncapped": float(mt[(te.stage == st).values].mean()),
                           "outlier_rate": float(ot[(te.stage == st).values].mean())} for st in ("seed", "angel", "A")}
        # follow-on: reserve at round 2 for companies that raised it
        m2 = multiples(p, a, entry="next").loc[te.index].values
        has2 = te.next_round.notna().values
        fast = has2 & (te.years_to_next.values <= 1.5)
        r["follow_on"] = {"n_with_round2": int(has2.sum()), "round2_mean_multiple": float(m2[has2].mean()),
                          "round2_fast_le18mo_mean": float(m2[fast].mean()), "n_fast": int(fast.sum()),
                          "initial_mean_multiple_all": float(mt.mean())}
        rng = np.random.default_rng(4)
        sims = {}
        for res_share in (0.0, 0.3, 0.5):
            f = []
            for _ in range(10000):
                d = rng.choice(len(te), 50)
                init = mt[d].mean()
                fo_pool = d[fast[d]]
                fo = m2[fo_pool].mean() if len(fo_pool) else 1.0
                f.append((1 - res_share) * init + res_share * fo)
            f = np.array(f)
            sims[f"reserve_{int(res_share*100)}pct"] = {"median": float(np.median(f)), "mean": float(f.mean()),
                                                        "p_loss": float((f < 1).mean())}
        r["follow_on"]["fund_sims_50co"] = sims
        # vintage dispersion (random deal portfolio by cohort)
        all_m = m
        r["vintage"] = {int(y): {"mean": float(all_m[p.cohort == y].mean()), "n": int((p.cohort == y).sum())}
                        for y in sorted(p.cohort.unique()) if y <= 2010}
        res_by_assump[an] = r
        print(an, "tail", r["tail"]["test"], flush=True)
        for k, v in r["strategies"].items():
            print(f"  {k:30s} {v['mean_uncapped']:.2f}x vs {v['mean_all_uncapped']:.2f}x  ({v['excess_uncapped_pct']:+.0f}%)  "
                  f"CI {v['excess_ci'][0]:+.2f}..{v['excess_ci'][1]:+.2f}  outliers {v['outliers_captured']}/{v['outliers_total']} lift {v['outlier_capture_lift']:.2f}", flush=True)
    out["by_assumption"] = res_by_assump

    # investor analyses use the calibrated multiples
    mcal = multiples(p, ASSUMPTIONS["calibrated"])
    out["investors"] = investor_analysis(p, mcal)
    print(json.dumps(out["investors"], default=float)[:1500])

    # ---------------- critique passes (calibrated assumptions) ----------------
    a = ASSUMPTIONS["calibrated"]
    te = p[p.split == "test"]
    mt = mcal.loc[te.index].values
    ot = mt >= OUTLIER_X
    prob_b, _ = ridge_all(p.assign(outlier=(mcal >= OUTLIER_X).astype(int)), "bigwin")
    p["outlier"] = (mcal >= OUTLIER_X).astype(int)
    prob_o, _ = ridge_all(p, "outlier")
    vote_b, _ = dawes_vote(p, "bigwin"); vote_o, _ = dawes_vote(p, "outlier")
    rank = lambda s: s.groupby(p.cohort).rank(pct=True)
    ens = ((rank(prob_b) + rank(prob_o) + rank(vote_b) + rank(vote_o)) / 4).loc[te.index]
    sel = top_mask(te, ens).values
    crit = {}
    # Pass 1 (risk / portfolio construction): log-growth-optimal size, concentration
    crit["sizing"] = {}
    for N in (10, 25, 50, 100, 200):
        draws = rng0.choice(mt[sel], (20000, N)).mean(1)
        crit["sizing"][N] = {"exp_log_growth": float(np.mean(np.log(np.maximum(draws, 1e-3)))),
                             "p_loss": float((draws < 1).mean()), "median": float(np.median(draws))}
    # top-5% vs top-10% vs top-20% selection breadth (fixed menu, reported, not tuned)
    crit["breadth"] = {}
    for frac in (0.05, 0.10, 0.20, 0.33, 1.0):
        s2 = top_mask(te, ens, frac=frac).values
        crit["breadth"][frac] = {"n": int(s2.sum()), "mean": float(mt[s2].mean()), "outlier_rate": float(ot[s2].mean())}
    # Pass 2 (pricing / adverse selection): you rarely get into the hot deals
    acc = {}
    sm = te.smart_money_insample.values.astype(bool)
    acc["exclude_top40_investor_deals"] = {"mean": float(mt[sel & ~sm].mean()), "n": int((sel & ~sm).sum()),
                                           "random_excl": float(mt[~sm].mean())}
    rng = np.random.default_rng(12)
    for p_out in (0.5, 0.25, 0.1):
        vals = []
        for _ in range(2000):
            access = rng.random(len(te)) < np.where(ot, p_out, 0.6)
            s3 = sel & access
            vals.append(mt[s3].mean() / mt[access].mean() if s3.any() else np.nan)
        acc[f"outlier_access_{p_out}"] = {"strategy_over_random": float(np.nanmean(vals))}
    crit["adverse_selection"] = acc
    # Bayesian shrinkage of the measured excess toward zero, using round 1's holdout failure as
    # evidence that test-window excess overstates truth: posterior mean with a N(0, tau) prior.
    ex = mt[sel].mean() / mt.mean() - 1
    lo, hi, _ = boot_ci(sel, np.ones(len(te), bool), mt)
    se = (hi - lo) / 3.92 / mt.mean()
    for tau in (0.1, 0.25, 0.5):
        crit.setdefault("shrunk_excess", {})[tau] = float(ex * tau ** 2 / (tau ** 2 + se ** 2))
    crit["raw_excess"] = float(ex); crit["raw_excess_se"] = float(se)
    out["critique"] = crit

    # Pass 2 improvement: use the model only as a NEGATIVE screen on a broad index.
    ens_all0 = (rank(prob_b) + rank(prob_o) + rank(vote_b) + rank(vote_o)) / 4
    scr = {}
    for split in ("test", "holdout"):
        d = p[p.split == split]; mm = mcal.loc[d.index].values; keep = mm < mm.max()
        for drop in (0.0, 0.2, 0.33, 0.5):
            s4 = (ens_all0.loc[d.index] > drop).values if drop else np.ones(len(d), bool)
            scr[f"{split}_drop{int(drop*100)}"] = {"n": int(s4.sum()), "mean": float(mm[s4].mean()),
                                                  "mean_excl_top_deal": float(mm[s4 & keep].mean()),
                                                  "outliers": int((s4 & (mm >= OUTLIER_X)).sum()),
                                                  "outliers_total": int((mm >= OUTLIER_X).sum())}
    crit["negative_screen"] = scr
    crit["negative_screen_note"] = "holdout rows are a SECOND look at a spent holdout"

    # Holdout: SPENT in round 1. Second look, labelled, for the combined ensemble only.
    ho = p[p.split == "holdout"]
    mh = mcal.loc[ho.index].values
    ens_all = (rank(prob_b) + rank(prob_o) + rank(vote_b) + rank(vote_o)) / 4
    sh = top_mask(ho, ens_all.loc[ho.index]).values
    lo, hi, pl = boot_ci(sh, np.ones(len(ho), bool), mh)
    out["holdout_second_look"] = {"label": "NOT CLEAN: 2011 was already consumed in round 1",
                                  "strategy": float(mh[sh].mean()), "random": float(mh.mean()), "ci": [lo, hi],
                                  "n": int(sh.sum()), "outliers_captured": int((sh & (mh >= OUTLIER_X)).sum()),
                                  "outliers_total": int((mh >= OUTLIER_X).sum())}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("crit", json.dumps(crit, default=float)[:1500])
    print("holdout2", out["holdout_second_look"])


if __name__ == "__main__":
    main()
