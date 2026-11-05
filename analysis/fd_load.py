"""Load every SEC DERA Form D quarterly data set (2008Q1-2026Q2) into four parquet tables:
filings (one row per accession: CIK, issuer, offering terms), persons (related persons)."""
from __future__ import annotations

import glob
import os
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, "data", "formd")
OUT = os.path.join(ROOT, "data", "fd")

OFF_COLS = ["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "REVENUERANGE", "FEDERALEXEMPTIONS_ITEMS_LIST", "ISAMENDMENT",
            "SALE_DATE", "YETTOOCCUR", "MORETHANONEYEAR", "ISEQUITYTYPE", "ISDEBTTYPE", "ISOPTIONTOACQUIRETYPE",
            "ISSECURITYTOBEACQUIREDTYPE", "ISBUSINESSCOMBINATIONTRANS", "MINIMUMINVESTMENTACCEPTED",
            "TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "TOTALREMAINING", "HASNONACCREDITEDINVESTORS",
            "NUMBERNONACCREDITEDINVESTORS", "TOTALNUMBERALREADYINVESTED", "SALESCOMM_DOLLARAMOUNT",
            "FINDERSFEE_DOLLARAMOUNT", "GROSSPROCEEDSUSED_DOLLARAMOUNT"]
ISS_COLS = ["ACCESSIONNUMBER", "IS_PRIMARYISSUER_FLAG", "CIK", "ENTITYNAME", "CITY", "STATEORCOUNTRY", "ZIPCODE",
            "JURISDICTIONOFINC", "ENTITYTYPE", "YEAROFINC_TIMESPAN_CHOICE", "YEAROFINC_VALUE_ENTERED"]
SUB_COLS = ["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE"]
PER_COLS = ["ACCESSIONNUMBER", "FIRSTNAME", "MIDDLENAME", "LASTNAME", "CITY", "STATEORCOUNTRY",
            "RELATIONSHIP_1", "RELATIONSHIP_2", "RELATIONSHIP_3"]


def _read(z, stem, cols):
    name = [n for n in z.namelist() if n.upper().endswith(stem + ".TSV")][0]
    with z.open(name) as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, on_bad_lines="skip", encoding="latin-1", quoting=3)
    return df[[c for c in cols if c in df.columns]]


def main():
    os.makedirs(OUT, exist_ok=True)
    F, P = [], []
    for path in sorted(glob.glob(os.path.join(FD, "20*q*_d*.zip"))):
        z = zipfile.ZipFile(path)
        sub = _read(z, "FORMDSUBMISSION", SUB_COLS)
        iss = _read(z, "ISSUERS", ISS_COLS)
        iss = iss[iss.IS_PRIMARYISSUER_FLAG == "YES"].drop(columns="IS_PRIMARYISSUER_FLAG")
        off = _read(z, "OFFERING", OFF_COLS)
        per = _read(z, "RELATEDPERSONS", PER_COLS)
        F.append(sub.merge(iss, on="ACCESSIONNUMBER").merge(off, on="ACCESSIONNUMBER"))
        P.append(per)
        print(os.path.basename(path), len(F[-1]), flush=True)
    f = pd.concat(F, ignore_index=True).drop_duplicates("ACCESSIONNUMBER")
    # pre-2020Q3 files use "2020-03-31 17:28:41"; later files use "30-SEP-2020"
    d1 = pd.to_datetime(f.FILING_DATE.str[:10], format="%Y-%m-%d", errors="coerce")
    d2 = pd.to_datetime(f.FILING_DATE.str[:11], format="%d-%b-%Y", errors="coerce")
    f["filing_date"] = d1.fillna(d2)
    s1 = pd.to_datetime(f.SALE_DATE, format="%Y-%m-%d", errors="coerce")
    s2 = pd.to_datetime(f.SALE_DATE, format="%d-%b-%Y", errors="coerce")
    f["sale_date"] = s1.fillna(s2)
    assert f.filing_date.notna().mean() > 0.999, "unparsed filing dates"
    for c in ("TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "TOTALREMAINING", "MINIMUMINVESTMENTACCEPTED",
              "NUMBERNONACCREDITEDINVESTORS", "TOTALNUMBERALREADYINVESTED", "SALESCOMM_DOLLARAMOUNT",
              "FINDERSFEE_DOLLARAMOUNT", "YEAROFINC_VALUE_ENTERED"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["CIK"] = f.CIK.str.lstrip("0")
    f.to_parquet(os.path.join(OUT, "filings.parquet"))
    p = pd.concat(P, ignore_index=True)
    p.to_parquet(os.path.join(OUT, "persons.parquet"))
    print("filings", len(f), "persons", len(p), "ciks", f.CIK.nunique())


if __name__ == "__main__":
    main()
