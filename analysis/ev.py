"""Expected percent return by horizon (1-40y): our strategy vs total VC market vs the best 10 VCs.

Annual rates (net to investor) and where each comes from:
  MARKET   10.0%/yr (range 7-13). Cambridge Associates US VC index, pooled net IRR to Dec 2025:
           20y 12.9%, 25y 8.1%, 10y 14.9%; post-2000 VC PME ~0.94-0.96 (Harris-Jenkinson-Kaplan),
           i.e. roughly public-equity returns going forward.
  STRATEGY (updated round 4, SEC Form D 2010-2021, 25k startups) market + 1.5 pt (range -1..+4):
           "best quality in the cheaper half of first raises" beat owning every first raise by +37%
           on the sealed 2018-21 holdout ($5M floor; +45% at $3M, +27% at $8M) and +40..+90% on
           2015-17. Paper markups over 4.5 years, not cash: halved, spread over a 10-year fund.
  (round 3) market + 0.5 pt (range -2..+2): the drop-bottom-third rule scored
           +26% on 2008-10 but only +6.5% (CI -34..+28) on the sealed 2012 lockbox; the lockbox wins.
           (Round-2 text follows.) market + 1.0 pt (range -1..+3). Broad index of every accessible seed/A deal (or 20+ funds)
           across >=5 vintages, minus the model's bottom third. Screen measured +21% (2008-10) and
           +19% (2011, second look) over a ~10y life ~= +2 pt/yr, halved for shrinkage/frictions.
           Emerging/prior-top-quartile manager tilt (HJKS PME 1.24/1.20 vs ~0.96) adds ~+0.5, an
           extra fee layer costs ~-0.5: net +1.0.
  TOP 10   HINDSIGHT, NOT INVESTABLE. Years 1-10: 31%/yr = mean top-decile net IRR of 2007-2014
           vintages (PitchBook 2Q18). Beyond 10y no fund stays top decile; the best long records in
           public LP disclosures are ~17-22%/yr (Sequoia legacy 17.4%, Khosla IV 22%), so 20%/yr after year 10.
EV = mean outcome ((1+r)^T - 1). Median uses 25%/yr volatility for a diversified VC book.
"""
import json, math, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = [1, 3, 5, 10, 20, 30, 40]
MKT, MKT_LO, MKT_HI = 0.10, 0.07, 0.13
EDGE, EDGE_LO, EDGE_HI = 0.067, 0.02, 0.10    # round 5: pooled edge +50% halved, low-fee vehicle -> 16.7% (analysis/edge.py)
SIG = 0.25
CA_HIST = {1: 21.1, 3: 8.7, 5: 10.2, 10: 14.9, 20: 12.9}   # CA US VC pooled IRR to 2025-12-31


def cum(r, T):
    return 100 * ((1 + r) ** T - 1)


def med(r, T):
    return 100 * (math.exp(T * (math.log(1 + r) - SIG ** 2 / 2)) - 1)


def top10(T):
    return 100 * (1.31 ** min(T, 10) * 1.20 ** max(T - 10, 0) - 1)


GP = json.load(open(os.path.join(ROOT, "reports", "gp.json")))["rows"]
GP_FUND_MULT = GP["our rule (+25%, round 5 halved)"]["gp_multiple_incl_fee_profit"]     # per 10-year fund


def gp_cum(T):
    """Manager running ONE strategy fund with 2% of their own money in it: fee profit + 20% carry +
    their own stake (analysis/gp.py), spread geometrically over the 10-year life (carry is really
    paid at exits, so early years are marks, not cash). After year 10 the proceeds compound at the
    market rate. Later funds are NOT counted, so long horizons are a floor for an ongoing firm."""
    if T <= 10:
        return 100 * (GP_FUND_MULT ** (T / 10) - 1)
    return 100 * (GP_FUND_MULT * (1 + MKT) ** (T - 10) - 1)


rows = []
for T in H:
    s = MKT + EDGE
    rows.append({"years": T,
                 "strategy_ev_pct": cum(s, T), "strategy_range": [cum(MKT_LO + EDGE_LO, T), cum(MKT_HI + EDGE_HI, T)],
                 "strategy_median_pct": med(s, T), "strategy_annual_pct": 100 * s,
                 "market_ev_pct": cum(MKT, T), "market_range": [cum(MKT_LO, T), cum(MKT_HI, T)],
                 "market_median_pct": med(MKT, T), "market_annual_pct": 100 * MKT,
                 "market_realized_pct": (100 * ((1 + CA_HIST[T] / 100) ** T - 1)) if T in CA_HIST else None,
                 "market_realized_annual_pct": CA_HIST.get(T),
                 "top10_hindsight_pct": top10(T), "top10_annual_pct": 100 * ((1 + top10(T) / 100) ** (1 / T) - 1),
                 "strategy_vs_market_pts": cum(s, T) - cum(MKT, T),
                 "gp_manager_pct": gp_cum(T),
                 "strategy_as_share_of_top10": cum(s, T) / top10(T)})
for r in rows:
    print(f"{r['years']:>2}y  strategy {r['strategy_ev_pct']:>9.0f}%  market {r['market_ev_pct']:>9.0f}%  top10 {r['top10_hindsight_pct']:>11.0f}%  "
          f"strat median {r['strategy_median_pct']:>8.0f}%  realized mkt {r['market_realized_pct'] or float('nan'):>7.0f}%")
json.dump({"rows": rows, "notes": __doc__}, open(os.path.join(ROOT, "reports", "ev.json"), "w"), indent=1)
