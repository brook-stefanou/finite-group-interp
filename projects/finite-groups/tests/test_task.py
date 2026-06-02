import numpy as np
import pytest

from finite_groups.generators import GroupGenerators
from finite_groups.task import build_group_task, train_test_split


def test_task_has_every_ordered_pair_once():
    g = GroupGenerators.cyclic_group(4)
    task = build_group_task(g)

    assert task.inputs.shape == (16, 2)
    assert task.targets.shape == (16,)
    pairs = [tuple(row) for row in task.inputs]
    assert sorted(pairs) == sorted((i, j) for i in range(4) for j in range(4))


def test_targets_are_the_group_product_cyclic():
    g = GroupGenerators.cyclic_group(4)  # product index = (i + j) % 4
    task = build_group_task(g)
    for (i, j), t in zip(task.inputs, task.targets):
        assert t == (i + j) % 4


def test_targets_match_cayley_table_nonabelian():
    g = GroupGenerators.symmetric_group(3)
    task = build_group_task(g)
    for (i, j), t in zip(task.inputs, task.targets):
        assert t == g.cayley_table[i, j]


def test_split_covers_all_pairs_without_overlap():
    g = GroupGenerators.cyclic_group(5)  # 25 pairs
    task = build_group_task(g)
    split = train_test_split(task, train_frac=0.6, seed=0)

    assert split.train_inputs.shape[0] == round(0.6 * 25)
    assert split.train_inputs.shape[0] + split.test_inputs.shape[0] == 25

    combined = np.concatenate([split.train_inputs, split.test_inputs])
    pairs = {tuple(row) for row in combined}
    assert pairs == {(i, j) for i in range(5) for j in range(5)}  # complete, no overlap


def test_split_is_deterministic_for_a_seed():
    task = build_group_task(GroupGenerators.cyclic_group(5))
    a = train_test_split(task, 0.6, seed=42)
    b = train_test_split(task, 0.6, seed=42)
    assert np.array_equal(a.train_inputs, b.train_inputs)
    assert np.array_equal(a.train_targets, b.train_targets)


def test_split_differs_across_seeds():
    task = build_group_task(GroupGenerators.cyclic_group(6))
    a = train_test_split(task, 0.5, seed=1)
    b = train_test_split(task, 0.5, seed=2)
    assert not np.array_equal(a.train_inputs, b.train_inputs)


def test_split_keeps_targets_aligned_with_inputs():
    # The dangerous bug: shuffling inputs and targets out of step. Every split
    # row must still satisfy target == cayley_table[i, j].
    g = GroupGenerators.symmetric_group(3)
    task = build_group_task(g)
    split = train_test_split(task, 0.7, seed=3)

    for (i, j), t in zip(split.train_inputs, split.train_targets):
        assert t == g.cayley_table[i, j]
    for (i, j), t in zip(split.test_inputs, split.test_targets):
        assert t == g.cayley_table[i, j]


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_invalid_train_frac_raises(bad_frac):
    task = build_group_task(GroupGenerators.cyclic_group(3))
    with pytest.raises(ValueError):
        train_test_split(task, bad_frac, seed=0)
