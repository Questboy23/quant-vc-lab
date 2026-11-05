"""Guards on the properties the whole design rests on."""
import json
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_holdout_single_touch(tmp_path, monkeypatch):
    from analysis import strategy
    monkeypatch.setattr(strategy, "LOCK", str(tmp_path / ".consumed.json"))
    calls = []
    strategy.unseal_holdout(lambda: calls.append(1) or {"x": 1})
    rec = strategy.unseal_holdout(lambda: calls.append(1) or {"x": 2})
    assert calls == [1] and rec["result"] == {"x": 1} and "SPENT" in rec["note"]


def test_multiple_winsorised_and_floor():
    from analysis.returns import Assumptions, multiples
    p = pd.DataFrame({"stage": ["seed", "A", "seed"], "raised_filled": [1e4, 5e6, 5e5],
                      "funding_total_usd": [1e4, 5e6, 5e5], "status": ["ipo", "acquired", "closed"],
                      "acq_price": [np.nan, 3e7, np.nan], "n_later_rounds": [0, 1, 0]})
    m = multiples(p, Assumptions(floor=2e6, ipo_value=5e8))
    assert m.iloc[0] == 100.0                      # 500M/2M = 250 -> capped
    assert abs(m.iloc[1] - 3e7 * 0.8 / (5e6 / 0.3)) < 1e-9
    assert m.iloc[2] == 0.0


def test_top_mask_is_per_cohort_and_breaks_ties_without_outcomes():
    from analysis.strategy import top_mask
    p = pd.DataFrame({"cohort": [2008] * 10 + [2009] * 10}, index=[f"c{i}" for i in range(20)])
    s = pd.Series(1.0, index=p.index)             # all tied
    m = top_mask(p, s)
    assert m[p.cohort == 2008].sum() == 1 and m[p.cohort == 2009].sum() == 1


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "panel_full.parquet")), reason="no panel")
def test_features_are_point_in_time_and_split_fixed():
    p = pd.read_parquet(os.path.join(ROOT, "data", "panel_full.parquet"))
    assert set(p.loc[p.split == "train", "cohort"]) <= set(range(2003, 2008))
    assert set(p.loc[p.split == "test", "cohort"]) <= set(range(2008, 2011))
    assert set(p.loc[p.split == "holdout", "cohort"]) == {2011}
    assert p.stage.isin(["seed", "angel", "A"]).all()


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "reports", "site_snapshot.json")), reason="no snapshot")
def test_snapshot_has_no_company_level_crunchbase_rows():
    s = json.load(open(os.path.join(ROOT, "reports", "site_snapshot.json")))
    text = json.dumps({k: v for k, v in s.items() if k != "live"})
    assert "/organization/" not in text and "permalink" not in text
