"""Console entry points, thin wrappers around the scripts in ``scripts/``.

They exist so an installed package offers ``exist2026-fewshot`` and
``exist2026-fusion`` without the ``python scripts/...`` incantation. The
implementations live in the scripts, which stay runnable from a clone.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run(name: str) -> int:
    path = SCRIPTS / name
    if not path.exists():
        raise SystemExit(
            f"{name} not found at {path}. The console scripts run from a repository "
            "checkout; use `python scripts/{name}` or clone the repository."
        )
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name="__main__")
    return 0


def fewshot_main() -> int:
    return _run("run_fewshot.py")


def fusion_main() -> int:
    return _run("run_fusion.py")
