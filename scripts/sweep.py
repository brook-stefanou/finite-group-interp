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
# NOTE (2026-06-09 sweep): the matched Dih-vs-Dic comparison uses wd 1.0 ONLY.
# wd 0.5 is grok-fragile for Dic26 (2/3 seeds memorise within 80k); Dih groks at
# both. C13sdC8 never groks at any setting here -- its irreps are dim-4 (vs dim-2
# for the pair), a genuinely harder target, not a hyperparameter miss.
GROUPS = ["D52", "Dic26", "C13sdC8"]
SEEDS = [0, 1, 2]
WEIGHT_DECAYS = [0.5, 1.0]
TRAIN_FRACS = [0.4]  # more training data -> better grokking odds when uncertain
EPOCHS = 80_000  # ~2.6x C113's grokking budget; stop_on_grok ends grokkers early
STOP_ON_GROK = True  # stop ~5 evals after test_acc crosses 0.99
# W&B logging mode:
#   "off"     -- no logging at all (what the first overnight sweep used)
#   "offline" -- full W&B runs to a local wandb/ dir, NO login/network needed;
#                push later with `wandb sync wandb/offline-run-*`. Safe for
#                unattended runs and you still get dashboards. (recommended)
#   "online"  -- live to wandb.ai; requires `wandb login` first, and will STALL
#                an unattended run if you're not logged in.
WANDB = "online"  # preference: see runs live on wandb.ai (logged in, so no stall)
# --------------------------------------------------------------------------


def main() -> None:
    combos = list(itertools.product(GROUPS, SEEDS, WEIGHT_DECAYS, TRAIN_FRACS))
    if WANDB == "off":
        env = {**os.environ, "PYTHONPATH": "src", "WANDB_MODE": "disabled"}
        wandb_override = "experiment.use_wandb=false"
    else:
        env = {**os.environ, "PYTHONPATH": "src", "WANDB_MODE": WANDB}  # offline | online
        wandb_override = "experiment.use_wandb=true"
    print(
        f"SWEEP: {len(combos)} runs | epochs<= {EPOCHS} | "
        f"stop_on_grok={STOP_ON_GROK} | wandb={WANDB}"
    )
    sweep_start = time.time()

    for i, (group, seed, wd, frac) in enumerate(combos, 1):
        name = f"pair-{group}-s{seed}-wd{wd}-f{frac}"
        overrides = [
            f"data.group={group}",
            f"data.train_frac={frac}",
            f"experiment.seed={seed}",
            f"experiment.name={name}",
            wandb_override,
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
