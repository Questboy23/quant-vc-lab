"""Horizon returns that respect HOW venture money moves: Takahashi-Alexander (Yale) fund cash-flow
model with fees, carry and the J-curve, run as a program that starts with $1 today.

Mechanics (yearly):
  * You set aside 100. Uncalled cash earns CASH_RATE (3.5%).
  * Each fund calls capital: 25% / 33% / 50%-of-remaining in years 1/2/3+ (TA defaults).
  * Fees 2%/yr of commitment years 1-5, 1.5% years 6-10, paid out of called capital.
  * Invested NAV grows at gross rate G. Distributions = NAV*(1+G)*(age/LIFE)^BOW, LIFE=12, BOW=2.5.
  * 20% carry on distributions once a fund has returned its paid-in capital.
  * Every distribution is recommitted to a new fund the next year (a perpetual VC programme).
  G is solved so that a single fund's NET IRR equals the target:
      market 10%  (CA US VC index, 20y 12.9% / 25y 8.1%)
      strategy  round 5: quality-in-cheap-half edge pooled over 3 out-of-sample windows (+50%),
                halved to +25% on the fund multiple, held through a low-fee vehicle (1%/10%)
                -> net IRR from analysis/edge.py (~16.7%); via a standard 2/20 fund it is ~14.2%
      best-10 hindsight: 31% for fund 1 (top-decile vintages 2007-14), 20% for later funds.
Value at horizon T = cash + NAV of all funds; percent = value - 100.
"""
import json, os
import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASH_RATE, LIFE, BOW = 0.035, 12, 2.5
CALL = [0.25, 0.333] + [0.5] * 20
H = [1, 3, 5, 10, 20, 30, 40]


def fund_flows(G, years=20):
    """Single fund, commitment 1: yearly (contributions, distributions_net, nav)."""
    paid = nav = dist_cum = 0.0
    out = []
    for age in range(1, years + 1):
        c = CALL[age - 1] * (1 - paid) if age <= 8 else 0.0
        fee = (0.02 if age <= 5 else 0.015 if age <= 10 else 0.0)
        fee = min(fee, c + nav * 0)  # fees come out of calls; if no call, from NAV
        paid += c
        grow = nav * (1 + G)
        rd = min((age / LIFE) ** BOW, 1.0)
        d_gross = grow * rd
        nav = grow - d_gross + (c - min(fee, c))
        if c < fee:
            nav -= (fee - c)
        carry = 0.0
        if dist_cum + d_gross > paid:
            carry = 0.2 * (dist_cum + d_gross - max(paid, dist_cum))
        d = d_gross - carry
        dist_cum += d
        out.append((c, d, max(nav, 0)))
    return out


def irr(flows):
    cf = [-c + d for c, d, _ in flows]
    cf[-1] += flows[-1][2]
    f = lambda r: sum(x / (1 + r) ** (i + 1) for i, x in enumerate(cf))
    try:
        return brentq(f, -0.9, 3.0)
    except ValueError:
        return np.nan


def solve_G(target):
    return brentq(lambda G: irr(fund_flows(G)) - target, -0.5, 2.0)


def program(G_first, G_later, T_max=40):
    """Start with 100. Commit everything to fund 0 at t=0; recommit distributions each year."""
    cash = 100.0
    funds = [(0, 100.0, G_first)]          # (start_year, commitment, G)
    flows = {}
    values = []
    for t in range(1, T_max + 1):
        cash *= 1 + CASH_RATE
        nav_total, new_commit = 0.0, 0.0
        for (s, k, G) in funds:
            age = t - s
            if age < 1:
                continue
            if (s, k, G) not in flows:
                flows[(s, k, G)] = fund_flows(G, years=40)
            c, d, nav = flows[(s, k, G)][age - 1]
            cash += k * (d - c)
            nav_total += k * nav
            new_commit += k * d
        if new_commit > 0.5:
            funds.append((t, new_commit, G_later))
        values.append(cash + nav_total)
    return values


def main():
    edge = json.load(open(os.path.join(ROOT, "reports", "edge.json")))
    strat = edge["net_irr"]["low-fee vehicle (1% fee, 10% carry)"]["strategy_halved_edge"]   # recommended way to hold it
    targets = {"strategy": (strat, strat), "market": (0.10, 0.10), "best10_hindsight": (0.31, 0.20)}
    G = {k: (solve_G(a), solve_G(b)) for k, (a, b) in targets.items()}
    rows = []
    paths = {k: program(*G[k]) for k in targets}
    for T in H:
        rows.append({"years": T, **{k + "_pct": paths[k][T - 1] - 100 for k in targets}})
    # single-fund diagnostics (TVPI path for the market fund)
    ff = fund_flows(G["market"][0])
    tvpi = []
    paid = dist = 0
    for c, d, nav in ff[:15]:
        paid += c; dist += d
        tvpi.append(round((dist + nav) / paid, 3) if paid else None)
    out = {"assumptions": __doc__, "gross_growth": {k: [round(a, 4), round(b, 4)] for k, (a, b) in G.items()},
           "net_irr_targets": targets, "rows": rows, "market_fund_tvpi_by_year": tvpi}
    json.dump(out, open(os.path.join(ROOT, "reports", "jcurve.json"), "w"), indent=1)
    print("gross G", out["gross_growth"])
    print("market single-fund TVPI by year", tvpi)
    for r in rows:
        print(f"{r['years']:>2}y  strategy {r['strategy_pct']:>9.0f}%  market {r['market_pct']:>9.0f}%  best-10 {r['best10_hindsight_pct']:>11.0f}%")


if __name__ == "__main__":
    main()
