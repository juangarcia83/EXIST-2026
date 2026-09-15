# Notebooks

Three notebooks, each a narrative over the `exist2026` package rather than a
copy of it. They are short on purpose: the logic lives in `src/exist2026/`,
where it is tested, reused across modalities and runnable headless.

| Notebook | What it is for |
|---|---|
| `00_dataset_overview.ipynb` | What the corpus looks like: decided vs. undecided instances, class balance per language, category co-occurrence, annotator agreement. |
| `01_fewshot_cascade.ipynb` | System B — inspect the pools, read the assembled prompt, check the token budget, run the cascade, score it, package a submission. |
| `02_fusion_crossattention.ipynb` | System A — the text and physiology streams, the cross-attention model, training with ICM-Soft-Norm selection, refit and submission. |

Each notebook switches modality with a single variable in its configuration
cell (`MODALITY = "memes"` or `"videos"`); nothing else changes.

## Running them

```bash
pip install -e ".[all]"
export EXIST2026_ROOT=/path/to/the/EXIST-2026/corpus
jupyter lab notebooks/
```

`EXIST2026_ROOT` is the only path you should ever need to set. The notebooks
read it through the files in `configs/`, so no cell contains a machine-specific
path — that is enforced by a test.

## Long runs belong on the command line

A full inference pass over the official test split takes hours, and a browser
tab is a poor place to keep it. Use the scripts, which resume from their
checkpoints after an interruption:

```bash
python scripts/run_fewshot.py --config fewshot_videos --stage submission
python scripts/run_fusion.py  --config fusion_memes   --steps hpo,train,submit
```

## House rules

- **Notebooks are committed without outputs.** `nbstripout` runs as a
  pre-commit hook and CI verifies it. Outputs make diffs unreadable, bloat the
  repository and can leak instance-level data.
- **No logic in a notebook that a script would need.** If a cell grows into
  something another run would want, it belongs in the package with a test.
- **No hard-coded paths, no ad-hoc recovery cells.** Resumability is a property
  of the pipeline, not something to patch in by hand when a run dies.
