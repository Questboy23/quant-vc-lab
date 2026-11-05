"""Fund economics: % return on the manager's (GP's) own money vs an outside investor (LP) in
the same fund. Percent only.

Fund: 10-year life, management fee 2%/yr of commitments years 1-5 and 1.5% years 6-10 (17.5% of
commitments in total; that's money not invested), carry 20% of profits above returned capital, no
hurdle (standard in VC). GP commits 2% of the fund alongside LPs and pays no fee/carry on it.
Fee income minus operating costs is shown separately (costs = 70% of fees for a small fund).
Gross multiple = what the invested capital returns; LP net follows from it.

Gross scenarios (multiple on invested capital over ~10y), sourced:
  market: LP net 10%/yr over a ~7y average hold ~ net TVPI 1.95x -> gross ~2.65x
  our rule: market +6.5% (2012 lockbox point estimate for the drop-bottom-third rule)
  top quartile fund: net TVPI ~2.5x (CA/PitchBook 2007-2014 vintages) -> gross ~3.2x
  top decile: net ~3.0x -> gross ~3.9x
  bad: gross 1.0x (money back before fees)
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEES = 0.02 * 5 + 0.015 * 5
INVESTED = 1 - FEES
GP_COMMIT = 0.02
COST_SHARE = 0.70
HOLD = 7.0     # average years capital is out


def fund(gross):
    proceeds = INVESTED * gross
    profit = max(proceeds - 1.0, 0)
    carry = 0.20 * profit * (1 - GP_COMMIT)                   # carry only on LP money
    lp_net = (proceeds - carry) / 1.0                          # LPs paid fees: multiple on commitment
    gp_capital_mult = proceeds                                 # GP's own 2% earns gross, no fees/carry
    gp_total = (GP_COMMIT * gp_capital_mult + carry) / GP_COMMIT
    fee_profit = FEES * (1 - COST_SHARE)
    gp_total_w_fees = (GP_COMMIT * gp_capital_mult + carry + fee_profit) / GP_COMMIT
    irr = lambda m: 100 * (max(m, 1e-9) ** (1 / HOLD) - 1)
    return {"gross": gross, "lp_net_multiple": lp_net, "lp_net_irr_pct": irr(lp_net),
            "gp_multiple_on_own_money": gp_total, "gp_irr_pct": irr(gp_total),
            "gp_multiple_incl_fee_profit": gp_total_w_fees, "gp_irr_incl_fees_pct": irr(gp_total_w_fees)}


SCEN = {"bad (money back)": 1.0, "VC market": 2.65, "our rule (+25%, round 5 halved)": 2.65 * 1.25,
        "top-quartile fund": 3.2, "top-decile fund": 3.9}
rows = {k: fund(g) for k, g in SCEN.items()}
for k, r in rows.items():
    print(f"{k:22s} gross {r['gross']:.2f}x | LP {100*(r['lp_net_multiple']-1):5.0f}% ({r['lp_net_irr_pct']:.1f}%/yr) | "
          f"GP own money {100*(r['gp_multiple_on_own_money']-1):6.0f}% ({r['gp_irr_pct']:.0f}%/yr) | with fee profit {100*(r['gp_multiple_incl_fee_profit']-1):6.0f}%")
json.dump({"assumptions": __doc__, "rows": rows}, open(os.path.join(ROOT, "reports", "gp.json"), "w"), indent=1)
