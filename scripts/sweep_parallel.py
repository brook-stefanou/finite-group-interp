import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GROUPS = ["Dic26", "D52"]  # Dic first: it's the slow one, so it starts earliest
SEEDS = list(range(8, 18))  # 10 NEW seeds -> 20 runs total
WEIGHT_DECAYS = [1.0]  # matched comparison uses wd1.0 only (Dic is grok-fragile below)
TRAIN_FRACS = [0.4]
EPOCHS = 80_000
STOP_ON_GROK = True

# --- parallelism ----------------------------------------------------------
# Physical cores on this Mac is 10. Leave a couple for the OS / W&B uploads / you.
MAX_WORKERS = 8
THREADS_PER_RUN = 1  # pin each run to 1 core; the whole point (see module docstring)

# W&B logging mode -- see sweep.py for the full explanation.
#   "off" | "offline" | "online".  Many simultaneous online runs are fine (distinct
#   run ids), just noisier on the dashboard. "offline" then `wandb sync` is calmest.
WANDB = "online"
# --------------------------------------------------------------------------

LOG_DIR = Path("sweep_logs")


def run_one(
    combo: tuple[str, int, float, float], env: dict[str, str], wandb_override: str
) -> tuple[str, int, float]:
    """Launch one run.py subprocess; return (name, returncode, minutes)."""
    group, seed, wd, frac = combo
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
