"""Deal-level price-adjusted multiples.

Crunchbase has neither valuations nor IPO values, so both ends of the multiple are modelled.
Every assumption is a named parameter and every headline number is re-run over a grid of them
(sensitivity() below). The base case is deliberately unexciting:

  entry post-money = max(first_raised / stage_dilution, floor)
                     stage_dilution: seed/angel 20%, Series A 30%
  ownership at exit = entry share * (1 - 20%) ** (later equity rounds, capped at 6)
  exit value        = acquisition price if disclosed (USD)
                      IPO                       -> IPO_VALUE (base $500M)
                      acquired, price unknown   -> UNDISCLOSED_X * total funding raised
                      closed / still operating  -> 0 (base) or OPERATING_X * total funding
  multiple          = exit value * dilution / post-money, winsorised at 100x
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

CAP = 100.0


@dataclass(frozen=True)
class Assumptions:
    floor: float = 2e6
    ipo_value: float = 500e6
    undisclosed_x: float = 1.0
    operating_x: float = 0.0
    later_dilution: float = 0.20

    def label(self) -> str:
        return (f"floor ${self.floor/1e6:g}M, IPO ${self.ipo_value/1e6:g}M, undisclosed "
                f"{self.undisclosed_x:g}x funding, operating {self.operating_x:g}x funding")


BASE = Assumptions()


def post_money(p: pd.DataFrame, a: Assumptions = BASE) -> pd.Series:
    dil = np.where(p.stage == "A", 0.30, 0.20)
    return np.maximum(p.raised_filled / dil, a.floor)


def multiples(p: pd.DataFrame, a: Assumptions = BASE) -> pd.Series:
    post = post_money(p, a)
    funding = p.funding_total_usd.fillna(p.raised_filled)
    exit_v = np.select(
        [p.status == "ipo",
         (p.status == "acquired") & p.acq_price.notna(),
         p.status == "acquired",
         p.status == "operating"],
        [a.ipo_value, p.acq_price, a.undisclosed_x * funding, a.operating_x * funding],
        0.0)
    keep = (1 - a.later_dilution) ** np.minimum(p.n_later_rounds, 6)
    return pd.Series(np.minimum(exit_v * keep / post, CAP), index=p.index)


def grid():
    """Assumption grid for sensitivity: 3 floors x 3 IPO values x 3 undisclosed x 2 operating."""
    for f, ipo, u, o in itertools.product((1e6, 2e6, 4e6), (250e6, 500e6, 1e9), (0.5, 1.0, 2.0), (0.0, 1.0)):
        yield Assumptions(floor=f, ipo_value=ipo, undisclosed_x=u, operating_x=o)


def boot_diff(sel: np.ndarray, base_mask: np.ndarray, m: np.ndarray, n: int = 2000, seed: int = 0):
    """Bootstrap the company set; return (diff, lo, hi, p_two_sided) for mean(m[sel]) - mean(m[base])."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(m))
    obs = m[sel].mean() - m[base_mask].mean()
    ds = np.empty(n)
    for k in range(n):
        b = rng.choice(idx, len(idx))
        s, bb = sel[b], base_mask[b]
        ds[k] = (m[b][s].mean() if s.any() else np.nan) - (m[b][bb].mean() if bb.any() else np.nan)
    ds = ds[~np.isnan(ds)]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    pval = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return float(obs), float(lo), float(hi), float(max(pval, 1 / n))


def as_dict(a: Assumptions) -> dict:
    return asdict(a)
