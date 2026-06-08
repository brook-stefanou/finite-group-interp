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
# The order-104 triple, all swept at identical hyperparameters so they're
# directly comparable:
#   D52     = Dih(104) ┐ primary same-character-table pair (the adjudication)
#   Dic26   = Dic(104) ┘
#   C13sdC8 = C13 ⋊ C8  -- same order, DIFFERENT character table (the contrast)
# 3 groups x 3 seeds x 2 weight_decays x 1 train_frac = 18 runs.
# At ~25 min/run (80k epochs, order 104) that's ~7.4 h worst case, less if they
# grok early (stop_on_grok). Sized to fit an ~8 h window.
GROUPS = ["D52", "Dic26", "C13sdC8"]
SEEDS = [0, 1, 2]
WEIGHT_DECAYS = [0.5, 1.0]
TRAIN_FRACS = [0.4]  # more training data -> better grokking odds when uncertain
EPOCHS = 80_000  # ~2.6x C113's grokking budget; stop_on_grok ends grokkers early
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
