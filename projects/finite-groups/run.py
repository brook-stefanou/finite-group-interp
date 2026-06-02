"""Launcher for grokking experiments.

    uv run python projects/finite-groups/run.py data.group=A4 optim.epochs=20000

Thin wrapper: it puts the repo root (for `core`) and the package src (for
`finite_groups`) on the path, then hands the CLI overrides to the cli module.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root -> `core`
sys.path.insert(0, str(_HERE.parent / "src"))  # finite-groups src -> `finite_groups`

from finite_groups.experiments.cli import main  # noqa: E402  (after sys.path setup)

if __name__ == "__main__":
    main(sys.argv[1:])
