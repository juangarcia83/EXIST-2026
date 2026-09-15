"""Notebooks are code too: they must parse, compile and carry no outputs.

Notebooks rot quietly — a rename in the package leaves a stale call that nobody
notices until someone runs the notebook. These checks cannot execute the
notebooks (they need the corpus and a GPU), but they do catch syntax errors,
committed outputs and references to package symbols that no longer exist.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_there_are_notebooks():
    assert NOTEBOOKS, "expected at least one notebook in notebooks/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebook:
    def test_is_valid_nbformat(self, path):
        notebook = load(path)
        assert notebook["nbformat"] == 4
        assert notebook["cells"]

    def test_every_code_cell_compiles(self, path):
        for index, source in enumerate(code_cells(load(path))):
            try:
                ast.parse(source)
            except SyntaxError as error:  # pragma: no cover - failure message only
                pytest.fail(f"{path.name} code cell {index} does not parse: {error}")

    def test_carries_no_outputs(self, path):
        """Outputs make diffs unreadable, bloat the repository and can leak data."""
        for cell in load(path)["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert not cell.get("outputs"), f"{path.name} has committed outputs"
            assert cell.get("execution_count") is None

    def test_opens_with_a_markdown_title(self, path):
        first = load(path)["cells"][0]
        assert first["cell_type"] == "markdown"
        assert "".join(first["source"]).lstrip().startswith("# ")

    def test_contains_no_hard_coded_local_path(self, path):
        """Paths come from the configs, which read EXIST2026_ROOT."""
        text = path.read_text(encoding="utf-8")
        for fragment in ("C:/Users", "c:/Users", "/content/drive/MyDrive/EXIST"):
            assert fragment not in text, f"{path.name} hard-codes {fragment}"


def test_imported_package_symbols_exist():
    """Every `from exist2026... import X` in a notebook resolves.

    Modules behind an optional extra (torch, transformers, optuna) are skipped
    rather than failed: CI installs only the light core on purpose.
    """
    import importlib

    checked, skipped = 0, set()
    for path in NOTEBOOKS:
        for source in code_cells(load(path)):
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if not node.module.startswith("exist2026"):
                    continue
                try:
                    module = importlib.import_module(node.module)
                except ImportError:
                    skipped.add(node.module)
                    continue
                for alias in node.names:
                    if hasattr(module, alias.name):
                        checked += 1
                        continue
                    # `from package import submodule` is also legal.
                    try:
                        importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        pytest.fail(
                            f"{path.name} imports {alias.name} from {node.module}, " "which does not exist"
                        )
                    checked += 1
    assert checked, "expected the notebooks to import from the package"
