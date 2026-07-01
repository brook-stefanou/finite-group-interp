"""Launcher for grokking experiments.

    uv run python scripts/run.py data.group=A4 optim.epochs=20000

Thin wrapper: `same_character_table_interp` is installed as a package (uv sync), so
this just hands the CLI overrides to the cli module.
"""

import sys

from same_character_table_interp.training.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])
