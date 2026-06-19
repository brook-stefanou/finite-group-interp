import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Every knob below has an env-var override so the SAME committed file runs both
# locally (its defaults) and on a fat cloud VM without editing -- e.g.
#   SEEDS=8-47 MAX_WORKERS=20 WANDB=off uv run python scripts/sweep_parallel.py
# SEEDS accepts a range "8-47" (inclusive) or a list "8,9,10"; GROUPS is comma-sep.


def _seeds(default: list[int]) -> list[int]:
    raw = os.environ.get("SEEDS")
    if not raw:
        return default
    if "-" in raw and "," not in raw:
        lo, hi = (int(x) for x in raw.split("-", 1))
        return list(range(lo, hi + 1))  # inclusive upper bound
    return [int(x) for x in raw.split(",")]


GROUPS = os.environ.get("GROUPS", "Dic26,D52").split(",")  # Dic first: slow one starts earliest
SEEDS = _seeds(list(range(8, 18)))  # local default: 10 new seeds -> 20 runs
WEIGHT_DECAYS = [1.0]  # matched comparison uses wd1.0 only (Dic is grok-fragile below)
TRAIN_FRACS = [0.4]
EPOCHS = int(os.environ.get("EPOCHS", 80_000))
STOP_ON_GROK = True
# ARCH=fc runs the fully-connected baseline (architecture confound). FC runs get
# an "fc-" name prefix (not "pair-") so the transformer pair figures/compare,
# which glob "pair-<group>-s", never pick them up.
ARCH = os.environ.get("ARCH", "transformer")

# Snapshot/log cadence -- defaults preserve prior behaviour; raise both for very
# long (e.g. 1M-epoch) runs so they don't write million-line metrics or thousands
# of checkpoints. Event-based snapshotting around grokking stays on regardless.
SNAPSHOT_INTERVAL = int(os.environ.get("SNAPSHOT_INTERVAL", 1000))
LOG_EVERY = int(os.environ.get("LOG_EVERY", 1))

# --- parallelism ----------------------------------------------------------
# Local default tuned for this 10-core Mac (leave a couple for the OS). On a
# 32-vCPU VM, MAX_WORKERS=20 runs the whole sweep at once. NB a GCP vCPU is one
# hyperthread, so keep THREADS_PER_RUN=1 and size MAX_WORKERS to ~vCPU count.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 8))
THREADS_PER_RUN = int(os.environ.get("THREADS_PER_RUN", 1))  # pin each run; the whole point

# W&B logging mode -- see sweep.py for the full explanation.
#   "off" | "offline" | "online".  On a throwaway VM use "off" (no login needed;
#   the analysis only consumes runs/). Locally "online" is the user's preference.
WANDB = os.environ.get("WANDB", "online")
# --------------------------------------------------------------------------

LOG_DIR = Path("sweep_logs")


def run_one(
    combo: tuple[str, int, float, float], env: dict[str, str], wandb_override: str
) -> tuple[str, int, float]:
    """Launch one run.py subprocess; return (name, returncode, minutes)."""
    group, seed, wd, frac = combo
    prefix = "fc" if ARCH == "fc" else "pair"
    name = f"{prefix}-{group}-s{seed}-wd{wd}-f{frac}"
    overrides = [
        f"data.group={group}",
        f"data.train_frac={frac}",
        f"experiment.seed={seed}",
        f"experiment.name={name}",
        f"model.arch={ARCH}",
        wandb_override,
        f"optim.weight_decay={wd}",
        f"optim.epochs={EPOCHS}",
        f"optim.stop_on_grok={str(STOP_ON_GROK).lower()}",
        f"optim.log_every={LOG_EVERY}",
        f"snapshot.interval={SNAPSHOT_INTERVAL}",
    ]
    started = time.time()
    log_path = LOG_DIR / f"{name}.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "scripts/run.py", *overrides],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return name, result.returncode, (time.time() - started) / 60.0


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    combos = list(itertools.product(GROUPS, SEEDS, WEIGHT_DECAYS, TRAIN_FRACS))

    # Pin threads PER PROCESS so MAX_WORKERS runs don't oversubscribe the cores.
    thread_env = {
        "OMP_NUM_THREADS": str(THREADS_PER_RUN),
        "MKL_NUM_THREADS": str(THREADS_PER_RUN),
        "OPENBLAS_NUM_THREADS": str(THREADS_PER_RUN),
        "VECLIB_MAXIMUM_THREADS": str(THREADS_PER_RUN),
        "NUMEXPR_NUM_THREADS": str(THREADS_PER_RUN),
        "PYTHONPATH": "src",
    }
    if WANDB == "off":
        env = {**os.environ, **thread_env, "WANDB_MODE": "disabled"}
        wandb_override = "experiment.use_wandb=false"
    else:
        env = {**os.environ, **thread_env, "WANDB_MODE": WANDB}
        wandb_override = "experiment.use_wandb=true"

    print(
        f"PARALLEL SWEEP: {len(combos)} runs | {MAX_WORKERS} workers x "
        f"{THREADS_PER_RUN} thread | epochs<= {EPOCHS} | wandb={WANDB}",
        flush=True,
    )
    print(f"per-run logs in {LOG_DIR}/<name>.log", flush=True)
    sweep_start = time.time()

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, c, env, wandb_override): c for c in combos}
        for fut in as_completed(futures):
            name, rc, minutes = fut.result()
            done += 1
            elapsed = (time.time() - sweep_start) / 60.0
            status = "ok" if rc == 0 else f"FAILED(rc={rc})"
            print(
                f"[{done}/{len(combos)}] {name}: {status} in {minutes:.0f} min "
                f"(+{elapsed:.0f} min wall)",
                flush=True,
            )

    print(f"\nPARALLEL SWEEP DONE in {(time.time() - sweep_start) / 60.0:.0f} min wall", flush=True)


if __name__ == "__main__":
    main()
