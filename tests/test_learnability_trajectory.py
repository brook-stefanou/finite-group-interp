import numpy as np

from same_character_table_interp.analysis.irrep_metrics import EnergyTrajectory
from same_character_table_interp.analysis.learnability_trajectory import (
    block_rep_types,
    class_excess_trajectory,
    concentration_index,
    is_grokked,
    onset_epoch,
)
from same_character_table_interp.groups.catalog import build_group
from same_character_table_interp.representations.projectors import (
    IsotypicBlock,
    real_isotypic_blocks,
)


def test_onset_epoch_returns_first_epoch_reaching_half_of_final():
    epochs = [0, 10, 20, 30]
    values = np.array([0.0, 0.2, 0.6, 1.0])
    assert onset_epoch(epochs, values, frac=0.5) == 20


def test_onset_epoch_none_when_class_never_concentrates():
    epochs = [0, 10, 20]
    values = np.array([0.0, -0.01, 0.0])  # final excess ~0: no concentration
    assert onset_epoch(epochs, values, frac=0.5) is None


def test_block_rep_types_dihedral_only_real():
    labels = block_rep_types(build_group("D52"))
    assert labels.count("1d") == 4
    assert labels.count("2d-real") == 25
    assert "2d-quaternionic" not in labels


def test_block_rep_types_dicyclic_has_quaternionic():
    labels = block_rep_types(build_group("Dic26"))
    assert labels.count("1d") == 4
    assert labels.count("2d-real") == 12
    assert labels.count("2d-quaternionic") == 13


def test_block_rep_types_aligns_with_real_isotypic_blocks():
    group = build_group("Dic26")
    assert len(block_rep_types(group)) == len(real_isotypic_blocks(group))


def test_concentration_index_is_total_energy_above_uniform():
    # Two trace-2 blocks in R^4 -> baseline 0.5 each. At [0.8, 0.2] the only
    # positive excess is 0.3 in block 0; uniform [0.5, 0.5] gives 0.
    p = np.eye(4)
    blocks = [
        IsotypicBlock(projector=p[:2].T @ p[:2], dimension=1, irrep_indices=(0,)),
        IsotypicBlock(projector=p[2:].T @ p[2:], dimension=1, irrep_indices=(1,)),
    ]
    traj = EnergyTrajectory(epochs=[0, 100], fractions=np.array([[0.5, 0.5], [0.8, 0.2]]))
    assert np.allclose(concentration_index(traj, blocks), [0.0, 0.3])


def test_is_grokked_requires_all_three_conditions():
    # The strict rule: completed status AND a grokked checkpoint AND final
    # test_acc >= threshold.
    assert is_grokked(status="completed", has_grokked_checkpoint=True, final_test_acc=0.991)
    assert is_grokked(status="completed", has_grokked_checkpoint=True, final_test_acc=0.99)


def test_is_grokked_rejects_non_completed_status():
    # "running" = crashed/interrupted; "failed" = errored -- neither grokked.
    assert not is_grokked(status="running", has_grokked_checkpoint=True, final_test_acc=1.0)
    assert not is_grokked(status="failed", has_grokked_checkpoint=True, final_test_acc=1.0)


def test_is_grokked_rejects_missing_checkpoint_or_low_acc():
    assert not is_grokked(status="completed", has_grokked_checkpoint=False, final_test_acc=1.0)
    assert not is_grokked(status="completed", has_grokked_checkpoint=True, final_test_acc=0.98)
    assert not is_grokked(status="completed", has_grokked_checkpoint=True, final_test_acc=None)


def test_class_excess_trajectory_sums_above_baseline_per_label():
    # Two blocks of trace 2 in R^4 -> baseline 0.5 each.
    p = np.eye(4)
    blocks = [
        IsotypicBlock(projector=p[:2].T @ p[:2], dimension=1, irrep_indices=(0,)),
        IsotypicBlock(projector=p[2:].T @ p[2:], dimension=1, irrep_indices=(1,)),
    ]
    traj = EnergyTrajectory(epochs=[0, 100], fractions=np.array([[0.5, 0.5], [0.8, 0.2]]))
    out = class_excess_trajectory(traj, blocks, ["x", "y"])
    assert np.allclose(out["x"], [0.0, 0.3])
    assert np.allclose(out["y"], [0.0, -0.3])


def test_class_excess_trajectory_include_restricts_to_circuit_blocks():
    # Same blocks, but two share a label; `include` keeps only the circuit block.
    p = np.eye(4)
    blocks = [
        IsotypicBlock(projector=p[:2].T @ p[:2], dimension=1, irrep_indices=(0,)),
        IsotypicBlock(projector=p[2:].T @ p[2:], dimension=1, irrep_indices=(1,)),
    ]
    traj = EnergyTrajectory(epochs=[0, 100], fractions=np.array([[0.5, 0.5], [0.8, 0.2]]))
    out = class_excess_trajectory(traj, blocks, ["x", "x"], include=[0])
    assert set(out) == {"x"}
    assert np.allclose(out["x"], [0.0, 0.3])  # block 1's -0.3 deficit excluded
