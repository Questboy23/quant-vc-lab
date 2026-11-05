"""Historical SEC Form D filings (DERA structured data sets, 2008Q1-2012Q4) matched to the
Crunchbase panel by normalised issuer name + state.

Electronic Form D filing only became mandatory in March 2009, so 2008 coverage is thin by
construction. That is a coverage pattern, not a finding; the leakage check handles it.

Feature: filing delay = FILING_DATE - SALE_DATE (date of first sale) on the ORIGINAL (non
amendment) notice nearest the Crunchbase first-round date. Rule 503 requires filing within
15 calendar days of first sale."""
from __future__ import annotations

import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, "data", "formd")
OUT = os.path.join(ROOT, "data", "formd_hist.parquet")

SUFFIX = re.compile(r"\b(incorporated|inc|llc|l l c|corp|corporation|co|company|ltd|limited|lp|l p)\b")


def norm_name(s: str) -> str:
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", "", s)


def _read(z: zipfile.ZipFile, stem: str, cols) -> pd.DataFrame:
    name = [n for n in z.namelist() if n.upper().endswith(stem + ".TSV")][0]
    with z.open(name) as fh:
        return pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, on_bad_lines="skip",
                           encoding="latin-1", quoting=3)


def load_all() -> pd.DataFrame:
    parts = []
    for path in sorted(glob.glob(os.path.join(FD, "20*q*_d*.zip"))):
        z = zipfile.ZipFile(path)
        sub = _read(z, "FORMDSUBMISSION", ["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE"])
        iss = _read(z, "ISSUERS", ["ACCESSIONNUMBER", "IS_PRIMARYISSUER_FLAG", "ENTITYNAME",
                                   "STATEORCOUNTRY", "CITY"])
        off = _read(z, "OFFERING", ["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "ISAMENDMENT",
                                    "SALE_DATE", "YETTOOCCUR", "TOTALOFFERINGAMOUNT",
                                    "TOTALAMOUNTSOLD"])
        iss = iss[iss.IS_PRIMARYISSUER_FLAG == "YES"]
        parts.append(sub.merge(iss, on="ACCESSIONNUMBER").merge(off, on="ACCESSIONNUMBER"))
    fd = pd.concat(parts, ignore_index=True).drop_duplicates("ACCESSIONNUMBER")
    fd["filing_date"] = pd.to_datetime(fd.FILING_DATE.str[:10], errors="coerce")
    fd["sale_date"] = pd.to_datetime(fd.SALE_DATE, errors="coerce")
    fd["key"] = fd.ENTITYNAME.map(norm_name)
    fd["original"] = (fd.SUBMISSIONTYPE == "D") & (fd.ISAMENDMENT.str.lower() != "true")
    return fd


def match(panel: pd.DataFrame) -> pd.DataFrame:
    fd = load_all()
    fd = fd[fd.original & fd.sale_date.notna()]
    p = panel[panel.cohort >= 2008][["name", "state_code", "first_date"]].copy()
    p["key"] = p.name.map(norm_name)
    m = p.reset_index().merge(fd, on="key", how="inner")
    m = m[(m.STATEORCOUNTRY == m.state_code) | m.state_code.isna()]
    m["gap"] = (m.sale_date - m.first_date).dt.days.abs()
    m = m[m.gap <= 180].sort_values("gap").drop_duplicates("permalink")
    m["formd_delay_days"] = (m.filing_date - m.sale_date).dt.days
    out = m.set_index("permalink")[["formd_delay_days", "filing_date", "sale_date",
                                    "INDUSTRYGROUPTYPE", "TOTALAMOUNTSOLD"]]
    out.to_parquet(OUT)
    return out


if __name__ == "__main__":
    panel = pd.read_parquet(os.path.join(ROOT, "data", "panel.parquet"))
    out = match(panel)
    print(len(out), "matched of", (panel.cohort >= 2008).sum())
    print(out.formd_delay_days.describe())
