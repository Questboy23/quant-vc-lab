"""Pool the quality-in-cheap-half edge across its three out-of-sample windows, shrink it, and turn
it into fund-level net IRRs under three ways of holding it (standard fund 2/20, low-fee 1/10, direct).

Windows (paper markup vs owning every first raise, $5M floor):
  test 2015-17 (54m)    +70%  CI [0, +153]
  holdout 2018-21 (54m) +37%  CI [-1, +85]
  lockbox 2022-23 (30m) +52%  CI [+20, +87]   <- fresh, opened once
Pooled by inverse variance on log(1+edge); then HALVED for paper-to-cash conversion and winner's curse."""
import json, os
import numpy as np
from scipy.optimize import brentq
from analysis import jcurve

ROOT = jcurve.ROOT
W = [(0.70, 0.00, 1.53), (0.37, -0.01, 0.85), (0.52, 0.20, 0.87)]


def pooled():
    lg = [np.log1p(e) for e, _, _ in W]
    se = [(np.log1p(h) - np.log1p(l)) / 3.92 for _, l, h in W]
    w = [1 / s ** 2 for s in se]
    m = sum(a * b for a, b in zip(w, lg)) / sum(w)
    s = (1 / sum(w)) ** 0.5
    return float(np.expm1(m)), float(np.expm1(m - 1.96 * s)), float(np.expm1(m + 1.96 * s))


def flows(G, fee_early, fee_late, carry, years=20):
    paid = nav = dist_cum = 0.0
    out = []
    for age in range(1, years + 1):
        c = jcurve.CALL[age - 1] * (1 - paid) if age <= 8 else 0.0
        fee = fee_early if age <= 5 else fee_late if age <= 10 else 0.0
        paid += c
        grow = nav * (1 + G)
        d_gross = grow * min((age / jcurve.LIFE) ** jcurve.BOW, 1.0)
        nav = grow - d_gross + c - min(fee, c)
        if c < fee:
            nav -= fee - c
        cr = carry * max(0.0, dist_cum + d_gross - max(paid, dist_cum)) if dist_cum + d_gross > paid else 0.0
        d = d_gross - cr
        dist_cum += d
        out.append((c, d, max(nav, 0)))
    return out


def irr(fl):
    return jcurve.irr(fl)


def gross_multiple(G):
    fl = flows(G, 0, 0, 0)
    return sum(d for _, d, _ in fl) + fl[-1][2]


def main():
    edge, lo, hi = pooled()
    used = edge / 2
    Gm = brentq(lambda G: irr(flows(G, .02, .015, .2)) - 0.10, 0.0, 1.0)
    gm = gross_multiple(Gm)
    Gs = brentq(lambda G: gross_multiple(G) - gm * (1 + used), 0.0, 2)
    Gs_full = brentq(lambda G: gross_multiple(G) - gm * (1 + edge), 0.0, 2)
    vehicles = {"standard fund (2% fee, 20% carry)": (.02, .015, .2), "low-fee vehicle (1% fee, 10% carry)": (.01, .01, .1),
                "direct / own index (no fees)": (0, 0, 0)}
    res = {"pooled_edge": edge, "pooled_ci": [lo, hi], "edge_used_halved": used,
           "market_gross_multiple": gm, "G_market": Gm, "G_strategy": Gs}
    res["net_irr"] = {}
    for name, (a, b, c) in vehicles.items():
        res["net_irr"][name] = {"market": irr(flows(Gm, a, b, c)), "strategy_halved_edge": irr(flows(Gs, a, b, c)),
                                "strategy_full_edge": irr(flows(Gs_full, a, b, c))}
    json.dump(res, open(os.path.join(ROOT, "reports", "edge.json"), "w"), indent=1)
    print(f"pooled edge {edge:+.0%} (CI {lo:+.0%}..{hi:+.0%}); used {used:+.0%}")
    for k, v in res["net_irr"].items():
        print(f"{k:40s} market {v['market']:.1%}  strategy {v['strategy_halved_edge']:.1%}  (full edge {v['strategy_full_edge']:.1%})")


if __name__ == "__main__":
    main()
