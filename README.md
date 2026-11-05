# Quant VC Lab

Tests hypotheses about which startups make venture investors money, on the Crunchbase 2015
export (non-commercial licence) plus SEC Form D. Published read-only at reedsmoneymaker23.com/vc
(branch `feature/quant-vc-lab` of ~/kalshi-paper).

    make data      # clone Crunchbase export, fetch SEC Form D data sets 2008-2012
    make vc        # everything: panel -> hypotheses (Holm + leakage) -> strategy -> snapshot
    make rebuild   # same, rebuilding the panel from raw first
    make live      # score new Form D filings (cron: live_nightly.sh, 06:30 Tue-Sat)
    make test

The SEC requires a contact email in the User-Agent of every EDGAR request. Downloads stop with an error until you set yours (there is no default):

    export SEC_CONTACT_EMAIL="you@example.com"

The cron script reads it from a local `.env` file (`SEC_CONTACT_EMAIL=you@example.com`), which git ignores.

**The 2011 holdout is SPENT** (`data/holdout/.consumed.json`). `make vc` reads the cached
single-touch result and never recomputes it. Results: `reports/FINAL.md`, `reports/SCOREBOARD.md`,
registry `vc/hypotheses.yaml` (`vc/.hypotheses.registered.yaml` is the pre-result copy).
