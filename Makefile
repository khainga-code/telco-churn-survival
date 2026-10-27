.PHONY: setup data test lint train score notebooks all clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/download_data.py

test:
	$(PYTHON) -m pytest

lint:
	ruff check .

train: data
	$(PYTHON) scripts/train_baselines.py

score:
	$(PYTHON) scripts/score.py --capacity 500

notebooks:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/01_framing_and_eda.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/02_models_calibration_value.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/03_survival_analysis.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/05_experiment_design.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/06_monitoring.ipynb

# Notebook 04 needs PyMC and takes a few minutes; run it separately.
notebooks-bayes:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace notebooks/04_bayesian_price_effect.ipynb

all: data train score notebooks

clean:
	rm -rf .pytest_cache .ruff_cache
