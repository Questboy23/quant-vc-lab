PY := .venv/bin/python

vc:            ## full pipeline (holdout result is read from the spent lock, never recomputed)
	$(PY) -m analysis.run_all
	$(PY) -m analysis.publish

rebuild:       ## rebuild the panel + Form D match from raw data first
	$(PY) -m analysis.run_all --rebuild
	$(PY) -m analysis.publish

live:          ## nightly Form D scorer, then republish the snapshot
	$(PY) -m live.formd_live --days 5
	$(PY) -m analysis.publish

data:          ## fetch raw data (Crunchbase 2015 export + SEC Form D data sets 2008-2012)
	./scripts/fetch_data.sh

test:
	$(PY) -m pytest -q tests
