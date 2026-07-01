"""Entry point for batched-ensemble training. Mirrors scripts/run.py but trains
all `ensemble.seeds` in one vmapped batch. Requires `ensemble.enabled=true`."""

import sys
from pathlib import Path

from same_character_table_interp.training.cli import build_config
from same_character_table_interp.training.ensemble import run_ensemble


def main(overrides: list[str]) -> list[Path]:
    config = build_config(overrides)
    if not config.ensemble.enabled:
        raise SystemExit("run_ensemble requires ensemble.enabled=true")
    if not config.ensemble.seeds:
        raise SystemExit("run_ensemble requires a non-empty ensemble.seeds list")
    dirs = run_ensemble(config)
    print(f"ensemble done: {len(dirs)} run dirs")
    for d in dirs:
        print(" ", d)
    return dirs


if __name__ == "__main__":
    main(sys.argv[1:])
