"""Name-match Form D startups to outcome lists: Crunchbase 2023 snapshot (Funding Status M&A /
IPO), YC directory 2026 (Acquired / Public / top company), unicorn lists 2022 and ~2025."""
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = os.path.join(ROOT, "data", "raw", "k")
SUF = re.compile(r"\b(incorporated|inc|llc|corp|corporation|co|company|ltd|limited|technologies|technology|labs|holdings|group|the)\b")


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower().replace("&", " and "))
    return re.sub(r"\s+", "", SUF.sub(" ", s))


def _val(x):
    x = str(x).replace("$", "").strip()
    m = re.match(r"([\d.]+)\s*([BM]?)", x)
    if not m:
        return np.nan
    v = float(m.group(1)); return v * (1e9 if m.group(2) == "B" or m.group(2) == "" else 1e6)


def enrich(e: pd.DataFrame) -> pd.DataFrame:
    key = e.ENTITYNAME.map(norm)
    out = pd.DataFrame(index=e.index)
    cb = pd.read_csv(os.path.join(K, "sezginvural_crunchbase-2015-2023-total-funding-amount-750k", "crunchbase_last.csv"))
    cb = cb[cb["Headquarter Country"] == "United States"]
    cb["k"] = cb["Organization Name"].map(norm)
    cb = cb.drop_duplicates("k").set_index("k")
    st = key.map(cb["Funding Status"])
    out["cb23_matched"] = st.notna().astype(int)
    out["acq_cb23"] = (st == "M&A").astype(int)
    yc = pd.read_csv(os.path.join(K, "nitishraj_ycombinatorcompanies", "yc_companies.csv"))
    yc["k"] = yc.name.map(norm); yc = yc.drop_duplicates("k").set_index("k")
    ys = key.map(yc.status)
    out["yc"] = ys.notna().astype(int)
    out["acq_yc"] = (ys == "Acquired").astype(int)
    out["yc_top"] = key.map(yc.top_company).fillna(False).astype(int)
    u1 = pd.read_csv(os.path.join(K, "deepcontractor_unicorn-companies-dataset", "Unicorn_Companies.csv"))
    u1["k"] = u1.Company.map(norm); u1["v"] = u1["Valuation ($B)"].map(_val)
    u2 = pd.read_csv(os.path.join(K, "adilshamim8_startup-growth-and-investment-data", "Unicorn_Companies.csv"))
    u2["k"] = u2.Company.map(norm); u2["v"] = u2.Valuation.map(_val)
    u = pd.concat([u1[["k", "v"]], u2[["k", "v"]]]).groupby("k").v.max()
    out["unicorn_val"] = key.map(u)
    out["unicorn"] = out.unicorn_val.notna().astype(int)
    out["acquired_any"] = out[["acq_cb23", "acq_yc"]].max(axis=1)
    return out
