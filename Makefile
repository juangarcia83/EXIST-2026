# Developer entry points. `make help` lists them.
.DEFAULT_GOAL := help
PYTHON ?= python3

.PHONY: help setup setup-all lint format test test-cov check clean nb-clean fewshot-memes fewshot-videos fusion-memes fusion-videos

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and the dev tools (no GPU stack)
	$(PYTHON) -m pip install -e ".[dev]"

setup-all:  ## Install everything, including torch, transformers and PyEvALL
	$(PYTHON) -m pip install -e ".[all]"

lint:  ## Check style and imports
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

format:  ## Apply formatting and the safe lint fixes
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m black src tests scripts

test:  ## Run the test suite
	$(PYTHON) -m pytest

test-cov:  ## Run the tests with a coverage report
	$(PYTHON) -m pytest --cov=exist2026 --cov-report=term-missing

check: lint test  ## Everything CI runs

nb-clean:  ## Strip outputs from every notebook
	$(PYTHON) -m nbstripout notebooks/*.ipynb

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

# --- pipelines -------------------------------------------------------------
# Set EXIST2026_ROOT to the directory holding the corpus, or edit configs/.

fewshot-memes:  ## Few-shot cascade over memes (STAGE=sanity|submission)
	$(PYTHON) scripts/run_fewshot.py --config fewshot_memes --stage $(or $(STAGE),sanity)

fewshot-videos:  ## Few-shot cascade over videos (STAGE=sanity|submission)
	$(PYTHON) scripts/run_fewshot.py --config fewshot_videos --stage $(or $(STAGE),sanity)

fusion-memes:  ## Cross-attention fusion over memes (STEPS=hpo,train,submit)
	$(PYTHON) scripts/run_fusion.py --config fusion_memes --steps $(or $(STEPS),train)

fusion-videos:  ## Cross-attention fusion over videos (STEPS=hpo,train,submit)
	$(PYTHON) scripts/run_fusion.py --config fusion_videos --steps $(or $(STEPS),train)
