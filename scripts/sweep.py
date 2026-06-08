"""Sequential hyperparameter sweep: the Cartesian product of the lists below,
one isolated `scripts/run.py` subprocess per combination.

Each run is its own process so torch/global RNG/determinism state can't leak
between runs. NOT Hydra multirun -- just a transparent product loop you can edit.
Runs write to runs/<date>/<run_id>/ as usual (gitignored).

Overnight (macOS, prevents sleep, detached, logged):

    caffeinate -i nohup uv run python scripts/sweep.py > sweep.log 2>&1 &
    tail -f sweep.log        # watch progress
    # afterwards: grep -E "RUN |final" sweep.log     # see results per run

Trim or extend the grid by editing the constants at the top.
"""

import itertools
import os
import subprocess
import sys
import time

# --- the grid (edit me) ---------------------------------------------------
# D52 = Dih(104), Dic26 = Dic(104): the primary same-character-table pair.
# Both MUST be swept together so their learned structure is comparable.
GROUPS = ["D52", "Dic26"]
SEEDS = [0, 1, 2]
WEIGHT_DECAYS = [0.5, 1.0]
TRAIN_FRACS = [0.3, 0.4]
EPOCHS = 50_000  # cap; stop_on_grok ends grokking runs earlier
STOP_ON_GROK = True  # stop ~5 evals after test_acc crosses 0.99
# --------------------------------------------------------------------------


def main() -> None:
    combos = list(itertools.product(GROUPS, SEEDS, WEIGHT_DECAYS, TRAIN_FRACS))
    env = {**os.environ, "PYTHONPATH": "src", "WANDB_MODE": "disabled"}
    print(f"SWEEP: {len(combos)} runs | epochs<= {EPOCHS} | stop_on_grok={STOP_ON_GROK}")
    sweep_start = time.time()

    for i, (group, seed, wd, frac) in enumerate(combos, 1):
        name = f"pair-{group}-s{seed}-wd{wd}-f{frac}"
        overrides = [
            f"data.group={group}",
            f"data.train_frac={frac}",
            f"experiment.seed={seed}",
            f"experiment.name={name}",
            "experiment.use_wandb=false",
            f"optim.weight_decay={wd}",
            f"optim.epochs={EPOCHS}",
            f"optim.stop_on_grok={str(STOP_ON_GROK).lower()}",
        ]
        elapsed = (time.time() - sweep_start) / 60.0
        print(f"\nRUN [{i}/{len(combos)}] {name}  (+{elapsed:.0f} min into sweep)", flush=True)
        result = subprocess.run([sys.executable, "scripts/run.py", *overrides], env=env)
        if result.returncode != 0:
            print(f"  !! {name} exited {result.returncode}; continuing", flush=True)

    print(f"\nSWEEP DONE in {(time.time() - sweep_start) / 60.0:.0f} min", flush=True)


if __name__ == "__main__":
    main()
