# Quant VC Lab — final report (2026-09-30)

**Verdict: no demonstrated edge.** 11 of 34 hypotheses predict *outcomes* (Holm-corrected
and leakage-checked). None of them turned into *returns* that survived the sealed 2011 holdout.

## Hypotheses (test cohorts 2008–2010, n=3,224; full table in SCOREBOARD.md)

Passes (11): P2 big syndicate, P6 SF Bay Area, P7 real-word name, P9 exact-match domain,
N6a Big Tech acquirer appetite, N6c strategic acquirer appetite, N11 accelerator, N13 investor
experience, N18 short name, N21 category momentum (negative: hot categories do *worse*),
N22 speed to next round (leakage-flagged, and only tradable at round 2).
Everything else, 23 tests, showed no evidence after Holm, including all of these:
smart money *for big wins*, the champion rule, Girard, Uzzi, kill zone *for big wins*, Form D
timing, round-size surprise, investor momentum, follow-on, geographic arbitrage, name
semantics, company age, syndicate diversity, and contrarian-but-cheap.
No test was significant in the opposite direction after Holm. Geographic arbitrage (non-SF
top-signal deals returned *less*, raw p=0.04) and seed-vs-A entry came closest.

## Reproducing the earlier findings

| Claim | Here | Match |
|---|---|---|
| Smart money strongest | success +0.94 log-odds (p<1e-6); big win +0.27 (p=0.33), gone in complete records | partial |
| Syndicate yes, champion worse | yes / yes (solo −0.65 on success) | yes |
| Girard rejected | crowded +0.24 on big win, p=0.052 | direction only |
| Uzzi rejected | zero effect | null, not reversed |
| Kill zone barbell | more acquisitions and more closures, but no big-win penalty | partial |
| SF | +0.57 | yes |
| Names are artifacts | real-word and exact-domain survive the coverage checks here | no |
| Composite AUC 0.74, 15.6% vs 4.9%, multiple ≈ market | AUC 0.72, 12.8% vs 3.6%, **2.10x vs 1.21x** | hit rate yes; multiple no |
| Price-adjusted 3.0–3.4x vs 1.5–1.9x, CI>0 at all floors | 2.25x vs 1.21x; CI>0 only at the $4M floor | no |

The numbers can't match "within rounding": no prior code exists, Crunchbase has no valuations
or IPO values (both have to be modelled), and the earlier test window included 2011, which is
the sealed holdout here.

## Strategy

Score = P(big win) from a train-fit logistic model (10 surviving first-round features) ÷
imputed post-money; top 10% of each cohort.

- Test window: 2.41x vs 1.21x, excess +1.20 (CI +0.00..+2.51). But a model with only round
  size and stage gets 2.24x; final minus size-only = +0.18 (CI −1.56..+1.80). **The edge is
  round size plus the valuation floor, not the hypotheses.**
- **Holdout 2011 (touched once, 2026-09-30T21:34Z): 0.54x vs 0.47x random, excess +0.07
  (CI −0.38..+0.76), n=203 of 2,031.** The prior price-adjusted score: 0.34x (below random).
  Size-only ÷ price: 0.18x, significantly *worse* than random. 10 of 54 assumption configs
  have CI>0 on the holdout.

## Expected fund multiple (50 companies, median [5th–95th])

| | Test 2008–10 | Holdout 2011 |
|---|---|---|
| Strategy | 2.18x [0.13–5.82], IRR 10.2% | 0.24x [0.04–1.70] |
| Strategy, 50% haircut | 1.64x [0.10–4.37], IRR 6.4% | 0.22x |
| Random | 0.80x [0.11–3.47] | 0.21x [0.02–2.02] |

Holdout multiples are low for *everything* because the 2011 cohort had about 4 years to exit
before the data ends and still-operating companies are valued at 0. Only the difference from
random is meaningful, and on the holdout it is indistinguishable from zero. Honest answer:
**expect the random-portfolio multiple**. The test-window 2.2x is a selection-contaminated
upper bound.

## Three biggest ways the backtest could be lying

1. **Prices are modelled, not observed.** No post-money, no IPO values, and 73% of
   acquisitions have no disclosed price. The floor makes every round under ~$400K equally
   "cheap", and that is where the price-adjusted score looks for picks.
2. **Cohort split, not point-in-time.** The top-40 investor list and the models use 2003–07
   outcomes as observed in 2015. The one point-in-time investor test (momentum) was null.
3. **Censoring and coverage.** Outcomes are frozen in Dec 2015. 27% of companies have no
   recorded investors, and Crunchbase backfills winners. The earlier analysis had already
   looked at 2011, so the holdout was sealed from this run's code, not from the idea.

## Round 2 (2026-10-01): outliers, combined method, how to invest

- Uncapped: 93% of deals lose money; the top 1% carries 89% of value; one deal (2009 cohort) moves the
  2008-10 average from 1.35x to 3.31x. Audit: 6 already-public "first rounds" removed; entry prices
  calibrated on real 2013-dump post-money rows; 21 real IPO values used.
- All 28 signals combined (ridge + equal-weight vote, big-win + 20x targets): 1.1-2.5x the random
  outlier hit-rate, but lower mean than owning everything; 2011 second look 0.22x vs 0.31x, 0/6 outliers.
- Only consistent rule: index everything, drop the model's bottom third: +21% (2008-10), +19% (2011, not clean).
- How: 200+ companies or 20+ funds (P(loss) 79% at 10 deals -> 47% at 200 -> 30% at 500), 5+ vintages,
  seed entry, equal checks, no big follow-on reserve (round-2 entry 1.33x vs 3.31x).
- Managers: persistence real but partial (rho 0.27, p=0.02); emerging firms weaker per deal, fine as a
  basket; GP stakes (Petershill) 16% total 2021-25.
- Expected return (reports/ev.json): strategy ~11%/yr vs market ~10%/yr; 10y 184% vs 159%;
  40y 6,400% vs 4,426%; best-10 hindsight 31%/yr (not investable).
