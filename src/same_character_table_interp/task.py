"""The group-multiplication learning task: predict a * b from the pair (a, b).

Inputs are pairs of element indices into the group; the target is the index of
their product (read straight off the Cayley table). The train/test split
fraction is the knob that drives grokking -- a small train fraction forces the
model to generalise rather than memorise.
"""

from dataclasses import dataclass

import numpy as np

from finite_group_interp.groups.group import FiniteGroup


@dataclass(frozen=True)
class GroupTask:
    """All |G|^2 multiplication examples for a group."""

    group_order: int
    inputs: np.ndarray  # shape (|G|^2, 2): element-index pairs (a, b)
    targets: np.ndarray  # shape (|G|^2,): index of a * b


@dataclass(frozen=True)
class TrainTestSplit:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray


def build_group_task(group: FiniteGroup) -> GroupTask:
    """Enumerate every ordered pair (a, b) and its product a * b."""
    n = group.order
    grid = np.arange(n)
    # Row k = (k // n, k % n), so it lines up with cayley_table.reshape(-1).
    inputs = np.stack([np.repeat(grid, n), np.tile(grid, n)], axis=1)
    targets = group.cayley_table.reshape(-1).copy()
    return GroupTask(group_order=n, inputs=inputs, targets=targets)


def train_test_split(task: GroupTask, train_frac: float, seed: int) -> TrainTestSplit:
    """Randomly split the task into train/test, seeded for reproducibility.

    Inputs and targets are indexed by the same permutation, so every example
    keeps its correct label.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in the open interval (0, 1), got {train_frac}")

    n = task.inputs.shape[0]
    permutation = np.random.default_rng(seed).permutation(n)
    n_train = round(train_frac * n)
    train_idx, test_idx = permutation[:n_train], permutation[n_train:]

    return TrainTestSplit(
        train_inputs=task.inputs[train_idx],
        train_targets=task.targets[train_idx],
        test_inputs=task.inputs[test_idx],
        test_targets=task.targets[test_idx],
    )
