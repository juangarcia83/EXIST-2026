# Contributing

This repository accompanies a shared-task paper, so the bar is *reproducibility*
rather than feature velocity: a reader should be able to run what the paper
describes and get what the paper reports.

## Setup

```bash
python -m pip install -e ".[dev]"   # light core + tooling, no GPU stack
pre-commit install
make check                          # ruff + black + pytest
```

Install `".[all]"` only when you actually need to run a model.

## Conventions

**Prompt text lives in `prompts/`, never in code.** The pipeline reads those
files at run time. If a prompt string appears in a `.py` file, the published
prompt and the executed one can drift, and the paper's claims stop being
checkable.

**Paths live in `configs/`, never in code.** Dataset locations are candidate
lists resolved against `EXIST2026_ROOT`. A hard-coded path is a bug even when
it works on your machine.

**Notebooks carry no outputs and no logic.** `nbstripout` runs as a pre-commit
hook and CI verifies it. A notebook cell that a second run would want belongs
in `src/exist2026/` with a test; notebooks are for narrative and inspection.

**Modality is a parameter.** Memes and videos differ in thresholds, pool design,
physiological representation and encoder — all of it expressed as configuration.
Resist adding a `if modality == ...` branch to a new place; if the difference is
real, give it a field.

**Keep the core importable without torch.** Everything that decides correctness
— labels, pools, prompts, parsing, submission, the cascade's control flow — must
import with numpy and pandas alone. Heavy imports go inside the function that
needs them, and their packages go in an optional extra. This is what keeps the
test suite runnable anywhere.

## Tests

Add a test with any behaviour change. The existing suite shows the style:

- `tests/conftest.py` builds a small synthetic corpus in the organizers' record
  shape, including the awkward cases (UNKNOWN votes, `"-"` for NO, ties below
  threshold, multi-label annotations, both languages).
- `tests/test_pipeline.py` drives the cascade with a scripted stand-in backend,
  so hierarchy propagation, checkpoint resume and leakage exclusion are tested
  as properties of this code rather than of whichever model is loaded.

Name tests for the behaviour, not the function: `test_the_gate_overrides_a_cached_downstream_value`
says what broke when it fails.

## Changing the systems

Before touching selection, prompting or scoring, be clear about which of these
you are changing — they are separate on purpose:

| Concern | Where |
|---|---|
| What counts as a hard label | `labels/hard.py` (official thresholds) |
| What the model is shown | `fewshot/pools.py` + `prompts/` |
| How an answer becomes a label | `fewshot/parsing.py` |
| What runs when | `fewshot/pipeline.py` (the cascade) |
| What gets scored | `evaluation/` |

A change that touches a published number should say so in the pull request,
with the before/after metric.

## Pull requests

- One concern per pull request.
- `make check` passes.
- If behaviour changed, say what the metric did.
- If a prompt changed, say which decision rule changed and why — prompt edits
  are experiment changes, not copy edits.
