# Reproducing the runs

Everything here assumes the EXIST 2026 corpus is on disk. It is distributed by
the organizers and is not redistributed in this repository.

## 1 · Install

```bash
git clone https://github.com/juangarcia83/EXIST-2026.git
cd EXIST-2026
python -m pip install -e ".[all]"      # or ".[dev]" for the light core only
export EXIST2026_ROOT=/path/to/the/corpus
```

The extras are separate on purpose:

| Extra | Pulls in | Needed for |
|---|---|---|
| *(core)* | numpy, pandas, PyYAML, scikit-learn, tqdm | configs, labels, pools, prompts, parsing, submission format |
| `fewshot` | torch, transformers, accelerate, pillow, qwen-vl-utils | System B |
| `fusion` | torch, transformers, optuna, matplotlib | System A |
| `eval` | PyEvALL | official scoring |
| `dev` | pytest, ruff, black, nbstripout, pre-commit | development |

The core is light enough that the whole test suite runs without a GPU stack,
which is what keeps CI to a couple of minutes.

## 2 · Point the configs at your corpus

`configs/*.yaml` declare each dataset file as a **list of candidates**; the
first one that exists wins. If your layout differs, add your path to the list
rather than editing code. `EXIST2026_ROOT` overrides `project_root` in every
config, so one environment variable is usually enough.

Check the wiring without loading a model:

```bash
python scripts/run_fewshot.py --config fewshot_memes --stage sanity --dry-run
```

This builds the few-shot pools and the prompts and stops before the model load.

## 3 · System B — the few-shot cascade

Always run a sanity stage first. It scores a stratified TRAIN subset with
PyEvALL in the hard-hard setting, which is what tells you whether the prompts
work before you spend GPU hours on the test split.

```bash
python scripts/run_fewshot.py --config fewshot_memes  --stage sanity
python scripts/run_fewshot.py --config fewshot_videos --stage sanity --backend qwen35
```

Read three things from the output:

1. **The fallback report.** Above ~5% the prompt or the token budget is at
   fault, and the metrics below it mean little.
2. **The majority baseline.** A system that does not beat it is not working.
3. **The per-subtask error CSVs** under `runs/.../metrics/errors/`.

Then the real run:

```bash
python scripts/run_fewshot.py --config fewshot_memes --stage submission \
    --team-name ELiRF_UPV --run-id 1
```

It writes `runs/<name>/submission/exist2026_<team>.zip`, already validated
against the official format. Interrupted runs resume: re-run the same command.

## 4 · System A — cross-attention fusion

```bash
python scripts/run_fusion.py --config fusion_videos --steps hpo,train,submit
```

The steps are separable, and `hpo` is by far the longest:

- `hpo` — sequential Optuna search (x.1, then x.2 / x.3 through x.1's gate).
  Writes `best_hyperparameters.json`; later steps pick it up automatically.
- `train` — final training of the three subtasks, keeping the best epoch by
  ICM-Soft-Norm, plus `figures/train_curves.png`.
- `submit` — refit on train+validation and write the soft submission.

Useful overrides while iterating: `--trials 5 --epochs 1 --device cpu`.

## 5 · Where the outputs go

Everything a run writes lands under `$EXIST2026_ROOT/runs/<run_dir>/`, never in
the repository:

```
runs/fewshot_memes_gemma4/
├── fewshot_pools/     exemplar pools (cached, so prompts stay identical)
├── checkpoints/       partial predictions — this is what makes runs resumable
├── predictions/       PyEvALL prediction files
├── metrics/           metrics_summary.csv, errors/*.csv
├── logs/              parse_fallbacks.log
├── pyevall_work/      scratch files for the scorer
└── submission/        exist2026_<team>/ and its .zip
```

Run directories are namespaced by modality and backend, so comparing two
backends never means overwriting the first one's results.

## 6 · Reproducibility notes

- Few-shot decoding is greedy (`do_sample=False`). Given the same corpus,
  backend and pools, a run is deterministic.
- Pools are a pure function of the corpus: ties break on id, not at random.
  Pass a seed only to measure pool variance on purpose.
- Training seeds `random`, `numpy` and `torch`, but exact GPU reproducibility
  across different hardware is not guaranteed and is not claimed.
- Physiological preprocessing statistics are fitted on the training split only.
  The refit before submission uses train+validation, never the test split.
