"""Overnight hyperparameter sweep to locate the grokking regime.

Runs a small grid of (group, train_frac, weight_decay) and writes per-run
metrics + snapshots under runs/. Read the test_acc curves afterwards to see
which settings grok. Epoch count is overridable:

    uv run python projects/finite-groups/sweep.py          # 50k epochs/run
    uv run python projects/finite-groups/sweep.py 2        # smoke test
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root -> core
sys.path.insert(0, str(_HERE.parent / "src"))  # finite-groups src

from core.config_schema import ExperimentConfig  # noqa: E402
from finite_groups.grokking.config import (  # noqa: E402
    DataConfig,
    GrokkingConfig,
    OptimConfig,
    SnapshotConfig,
)
from finite_groups.grokking.trainer import GroupGrokkingTrainer  # noqa: E402

# A few structurally varied groups: two non-abelian families, the quaternionic
# case, and an abelian baseline.
GROUPS = ["S3", "Q8", "A4", "C8"]
TRAIN_FRACS = [0.3, 0.4, 0.5, 0.6]  # the biggest grokking lever
WEIGHT_DECAYS = [0.1, 1.0, 3.0]


def main(epochs: int) -> None:
    # Group is the inner loop so early runs cover all groups (breadth first).
    combos = [(group, tf, wd) for tf in TRAIN_FRACS for wd in WEIGHT_DECAYS for group in GROUPS]
    print(f"sweep: {len(combos)} runs x {epochs} epochs", flush=True)
    for i, (group, tf, wd) in enumerate(combos, 1):
        name = f"grok-{group}-tf{tf}-wd{wd}"
        config = GrokkingConfig(
            experiment=ExperimentConfig(name=name, seed=0),
            data=DataConfig(group=group, train_frac=tf),
            optim=OptimConfig(epochs=epochs, weight_decay=wd, log_every=100),
            # Sweep reads curves only; skip snapshots (re-run winners with them later).
            snapshot=SnapshotConfig(enabled=False, event_based=False),
        )
        print(f"[{i}/{len(combos)}] {name}", flush=True)
        try:
            final = GroupGrokkingTrainer.from_config(config).fit()
            print(
                f"    test_acc={final['test_acc']:.3f} train_acc={final['train_acc']:.3f}",
                flush=True,
            )
        except Exception as exc:  # keep the sweep going if one run fails
            print(f"    FAILED: {exc}", flush=True)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    main(epochs)
