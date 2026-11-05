"""Assemble reports/site_snapshot.json (what reedsmoneymaker23.com/vc renders) and
reports/SCOREBOARD.md from results.json + live.json + the spent-holdout record.

Crunchbase 2015 is licensed non-commercially: only AGGREGATES leave this repo. No company-level
Crunchbase row is ever written to the snapshot. Live rows come from public SEC Form D filings."""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "reports")


def _get(h, hid):
    return next(x for x in h if x["id"] == hid)["result"]


def prior_table(res):
    h = res["hypotheses"]
    pr = res["prior"]
    ao = lambda hid, o: _get(h, hid)["all_outcomes"].get(o) or {}
    c, pa = pr["composite_top_decile"], pr["price_adjusted_top_decile"]
    floors = pr["price_adjusted_by_floor"]
    fl_ok = [k for k, v in floors.items() if v["excess_ci"][0] > 0]
    fmt = lambda d: f'{d.get("coef", 0):+.2f} (p={d.get("p", 1):.3g})' if d else "n/a"
    return [
        {"hypothesis": "Smart money (top-40 investors)", "prior": "Replicates; strongest for success and big wins",
         "ours": f"success {fmt(ao('P1','success'))}; big win {fmt(ao('P1','bigwin'))}, gone in complete records",
         "match": "partial"},
        {"hypothesis": "Big syndicate / consensus", "prior": "Replicates for success; solo champion WORSE",
         "ours": f"syndicate success {fmt(ao('P2','success'))}; solo success {fmt(ao('P2b','success'))}",
         "match": "yes"},
        {"hypothesis": "Crowded category (Girard)", "prior": "Rejected: crowded had MORE big wins",
         "ours": f"big win {fmt(ao('P3','bigwin'))}: same sign, not significant", "match": "direction only"},
        {"hypothesis": "Atypical combinations (Uzzi)", "prior": "Rejected: typical did better",
         "ours": f"big win {fmt(ao('P4','bigwin'))}: no effect either way", "match": "null, not reversed"},
        {"hypothesis": "Kill zone", "prior": "Barbell: more acquisitions, fewer big wins, more closures",
         "ours": f"success {fmt(ao('P5','success'))}, closed {fmt(ao('P5','closed'))}, big win {fmt(ao('P5','bigwin'))}",
         "match": "partial (no big-win penalty)"},
        {"hypothesis": "SF Bay Area", "prior": "Replicates", "ours": f"success {fmt(ao('P6','success'))}", "match": "yes"},
        {"hypothesis": "Names / .com / exact domain", "prior": "Mostly coverage artifacts",
         "ours": (f"real word {fmt(ao('P7','success'))}, exact domain {fmt(ao('P9','success'))} survive the coverage "
                  f"checks here; .com {fmt(ao('P8','success'))}"), "match": "no (they survive here)"},
        {"hypothesis": "Composite model", "prior": "AUC 0.74; big-win 15.6% vs 4.9%; multiple 1.82x vs 1.91x",
         "ours": (f"AUC {c['auc_bigwin']:.2f}; big-win {c['bigwin_rate_selected']:.1%} vs {c['bigwin_rate_all']:.1%}; "
                  f"multiple {c['multiple_selected']:.2f}x vs {c['multiple_all']:.2f}x"),
         "match": "hit rate yes; multiple no (beats market here)"},
        {"hypothesis": "Price-adjusted P(big win)/post-money", "prior": "3.0-3.4x vs 1.5-1.9x, CI excludes 0 at every floor",
         "ours": (f"{pa['multiple_selected']:.2f}x vs {pa['multiple_all']:.2f}x; CI excludes 0 only at floor(s) "
                  f"{', '.join(fl_ok) or 'none'} of {', '.join(floors)}"), "match": "no"},
    ]


def main():
    res = json.load(open(os.path.join(R, "results.json")))
    live_path = os.path.join(R, "live.json")
    live = json.load(open(live_path)) if os.path.exists(live_path) else {"available": False}
    hold = res.get("holdout")
    v2p, evp = os.path.join(R, "v2.json"), os.path.join(R, "ev.json")
    v2 = json.load(open(v2p)) if os.path.exists(v2p) else None
    ev = json.load(open(evp)) if os.path.exists(evp) else None
    v3p, gpp = os.path.join(R, "v3.json"), os.path.join(R, "gp.json")
    v3 = json.load(open(v3p)) if os.path.exists(v3p) else None
    v4p = os.path.join(R, "v4.json")
    v4 = json.load(open(v4p)) if os.path.exists(v4p) else None
    jcp = os.path.join(R, "jcurve.json")
    jc = json.load(open(jcp)) if os.path.exists(jcp) else None
    v5 = json.load(open(os.path.join(R, "v5.json"))) if os.path.exists(os.path.join(R, "v5.json")) else None
    edge = json.load(open(os.path.join(R, "edge.json"))) if os.path.exists(os.path.join(R, "edge.json")) else None
    gp = json.load(open(gpp)) if os.path.exists(gpp) else None
    if v2:   # model coefficients are not needed on the page
        for a in v2["by_assumption"].values():
            a.pop("models", None)
    snap = {
        "generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "assumptions": res["assumptions"], "assumption_label": res["assumption_label"],
        "facts": res["facts"], "prior_table": prior_table(res), "prior": res["prior"],
        "hypotheses": res["hypotheses"], "final": res["final"], "holdout": hold, "live": live,
        "v2": v2, "ev": ev, "v3": v3, "gp": gp, "v4": v4, "jcurve": jc, "v5": v5, "edge": edge,
        "license": ("Historical data: Crunchbase 2015 export (github.com/notpeter/crunchbase-data), "
                    "non-commercial use only; this page is non-commercial and shows aggregates only. "
                    "Live data: SEC EDGAR Form D, public."),
    }
    with open(os.path.join(R, "site_snapshot.json"), "w") as fh:
        json.dump(snap, fh, indent=1)

    lines = ["# Hypothesis scoreboard (test cohorts 2008-2010, Holm across all primaries)\n",
             "| id | hypothesis | primary | effect | p | p (Holm) | n | verdict |", "|---|---|---|---|---|---|---|---|"]
    for h in res["hypotheses"]:
        r = h["result"]
        lines.append(f'| {h["id"]} | {h["name"]} | {h["primary"]} | {r["effect"]} | {r["p_raw"]} | {r["p_holm"]} | '
                     f'{r["n"]} | {r["verdict"]} |')
    with open(os.path.join(R, "SCOREBOARD.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote site_snapshot.json and SCOREBOARD.md")


if __name__ == "__main__":
    main()
