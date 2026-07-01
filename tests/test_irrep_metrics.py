import numpy as np
import pytest
import torch

from same_character_table_interp.analysis.irrep_metrics import (
    block_ablation,
    energy_trajectory,
    evaluate,
    isotypic_energy,
    restricted_loss,
    weight_as_functions,
)
from same_character_table_interp.groups.catalog import resolve_group
from same_character_table_interp.representations.projectors import real_isotypic_blocks
from same_character_table_interp.task import build_group_task
from same_character_table_interp.training.config import ExperimentConfig, GrokkingConfig
from same_character_table_interp.training.trainer import build_model


def _c8_blocks():
    return real_isotypic_blocks(resolve_group("C8"))


def _analytic_fourier_projectors(n):
    """Real Fourier projectors for C_n: k=0 (constant), k=n/2 if n even
    (alternating, 1-dim), and 2-dim cos/sin planes for 0 < k < n/2."""
    x = np.arange(n)
    projs = []
    for k in range(n // 2 + 1):
        u = np.cos(2 * np.pi * k * x / n)
        v = np.sin(2 * np.pi * k * x / n)
        P = np.outer(u, u) / np.dot(u, u)
        if np.dot(v, v) > 1e-9:
            P = P + np.outer(v, v) / np.dot(v, v)
        projs.append(P)
    return projs


def test_library_blocks_equal_analytic_fourier_projectors():
    # THE calibration test: for a cyclic group the general character-table
    # machinery must reproduce the Fourier basis exactly.
    blocks = _c8_blocks()
    analytic = _analytic_fourier_projectors(8)
    assert len(blocks) == len(analytic) == 5  # k = 0..4 (4 = alternating block)
    unmatched = list(analytic)
    for b in blocks:
        hit = next(
            (i for i, P in enumerate(unmatched) if np.allclose(b.projector, P, atol=1e-8)),
            None,
        )
        assert hit is not None, "library block has no analytic Fourier partner"
        unmatched.pop(hit)
    assert not unmatched


def test_planted_energy_is_recovered():
    blocks = _c8_blocks()
    rng = np.random.default_rng(0)
    for i in range(len(blocks)):
        W = blocks[i].projector @ rng.normal(size=(8, 16))
        spec = isotypic_energy(W, blocks)
        assert spec.fractions[i] > 0.999
        assert spec.fractions.sum() == pytest.approx(1.0, abs=1e-4)


def test_planted_mixture_ratios_recovered():
    blocks = _c8_blocks()
    rng = np.random.default_rng(1)
    a = blocks[1].projector @ rng.normal(size=(8, 16))
    b = blocks[2].projector @ rng.normal(size=(8, 16))
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    spec = isotypic_energy(3.0 * a + 1.0 * b, blocks)  # orthogonal blocks: 9:1 energy
    # 1e-4 not 1e-6: projector idempotency noise is platform-dependent (BLAS),
    # and CI's leakage (~4e-6) exceeded 1e-6 where local runs passed. Same
    # rationale as the sum-to-1 tolerance above; the 9:1 distinction under
    # test is 1e-1-scale.
    assert spec.fractions[1] == pytest.approx(0.9, abs=1e-4)
    assert spec.fractions[2] == pytest.approx(0.1, abs=1e-4)


def test_baseline_is_block_dims_over_group_order():
    blocks = _c8_blocks()
    spec = isotypic_energy(np.random.default_rng(2).normal(size=(8, 4)), blocks)
    assert spec.baseline.sum() == pytest.approx(1.0)
    assert spec.block_dims.sum() == 8
    assert np.allclose(spec.baseline, spec.block_dims / 8)


def test_random_matrix_energy_approaches_baseline():
    blocks = _c8_blocks()
    W = np.random.default_rng(3).normal(size=(8, 20000))
    spec = isotypic_energy(W, blocks)
    assert np.allclose(spec.fractions, spec.baseline, atol=0.02)


def test_zero_matrix_raises():
    with pytest.raises(ValueError, match="zero"):
        isotypic_energy(np.zeros((8, 4)), _c8_blocks())


def test_wrong_row_count_raises():
    with pytest.raises(ValueError, match="rows"):
        isotypic_energy(np.ones((7, 4)), _c8_blocks())


def test_non_2d_raises():
    with pytest.raises(ValueError, match="2-d"):
        isotypic_energy(np.ones(8), _c8_blocks())


def test_incomplete_block_list_raises():
    blocks = _c8_blocks()
    W = np.random.default_rng(4).normal(size=(8, 4))
    with pytest.raises(ValueError, match="partition"):
        isotypic_energy(W, blocks[:3])


def test_empty_block_list_raises():
    with pytest.raises(ValueError, match="empty"):
        isotypic_energy(np.ones((8, 4)), [])


def _all_pairs_eval_set(group):
    """Every (a, b, =) sequence and its product -- the full task as eval data."""
    task = build_group_task(group)
    eq = np.full((len(task.inputs), 1), group.order)
    tokens = torch.tensor(np.concatenate([task.inputs, eq], axis=1), dtype=torch.long)
    targets = torch.tensor(task.targets, dtype=torch.long)
    return tokens, targets


def _planted_model(group, blocks, block_index, seed=0):
    """A model whose group-element embeddings live entirely in one block."""
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=seed, use_wandb=False))
    model = build_model(config, group)
    rng = np.random.default_rng(seed)
    planted = blocks[block_index].projector @ rng.normal(size=(group.order, model.d_model))
    with torch.no_grad():
        model.W_E[: group.order] = torch.tensor(planted, dtype=model.W_E.dtype)
    model.eval()
    return model


def test_ablating_planted_block_changes_loss_others_dont():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=2)
    tokens, targets = _all_pairs_eval_set(group)
    results = block_ablation(model, blocks, tokens, targets, matrix="W_E")
    assert results[2].delta_loss != 0.0  # the planted block mattered
    for i, r in enumerate(results):
        if i != 2:
            # other blocks hold no W_E component: ablating them is a no-op
            # float32 cross-entropy rounding can produce ~3e-6 noise; acc is exactly 0
            assert abs(r.delta_loss) < 1e-5 and r.delta_acc == 0.0


def test_ablation_is_non_destructive():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tokens, targets = _all_pairs_eval_set(group)
    block_ablation(model, blocks, tokens, targets, matrix="W_E")
    block_ablation(model, blocks, tokens, targets, matrix="W_U")
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key]), key


def test_restricted_to_all_blocks_equals_plain_loss():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=1)
    tokens, targets = _all_pairs_eval_set(group)
    base_loss, base_acc = evaluate(model, tokens, targets)
    loss, acc = restricted_loss(model, blocks, list(range(len(blocks))), tokens, targets)
    assert loss == pytest.approx(base_loss, abs=1e-5)
    assert acc == base_acc


def test_restricted_to_planted_block_equals_plain_loss():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=3)
    tokens, targets = _all_pairs_eval_set(group)
    base_loss, _ = evaluate(model, tokens, targets)
    loss, _ = restricted_loss(model, blocks, [3], tokens, targets)
    assert loss == pytest.approx(base_loss, abs=1e-5)  # everything lived there anyway


def test_restricted_keep_out_of_range_raises():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=0)
    tokens, targets = _all_pairs_eval_set(group)
    with pytest.raises(ValueError, match="block"):
        restricted_loss(model, blocks, [99], tokens, targets)


def test_restricted_to_no_blocks_zeroes_the_matrix():
    # keep=[] is the degenerate control: restrict W_E to nothing.
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_model(group, blocks, block_index=1)
    tokens, targets = _all_pairs_eval_set(group)
    base_loss, _ = evaluate(model, tokens, targets)
    loss, acc = restricted_loss(model, blocks, [], tokens, targets)
    assert np.isfinite(loss) and 0.0 <= acc <= 1.0
    assert loss != pytest.approx(base_loss)  # zeroing embeddings is not a no-op


def test_weight_as_functions_shapes_and_orientation():
    group = resolve_group("C8")
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=0, use_wandb=False))
    model = build_model(config, group)
    W_E = weight_as_functions(model, "W_E", group.order)
    W_U = weight_as_functions(model, "W_U", group.order)
    assert W_E.shape == (8, model.d_model)  # '=' row dropped
    assert W_U.shape == (8, model.d_model)  # transposed: columns are functions on G
    assert np.allclose(W_E, model.W_E.detach().numpy()[:8])
    assert np.allclose(W_U, model.W_U.detach().numpy().T)


def _planted_wu_model(group, blocks, block_index, seed=0):
    """A model whose unembedding columns (functions on G) live in one block."""
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=seed, use_wandb=False))
    model = build_model(config, group)
    rng = np.random.default_rng(seed)
    planted = blocks[block_index].projector @ rng.normal(size=(group.order, model.d_model))
    with torch.no_grad():
        model.W_U[:] = torch.tensor(planted.T, dtype=model.W_U.dtype)
    model.eval()
    return model


def test_ablating_planted_wu_block_changes_loss_others_dont():
    group = resolve_group("C8")
    blocks = _c8_blocks()
    model = _planted_wu_model(group, blocks, block_index=2)
    tokens, targets = _all_pairs_eval_set(group)
    results = block_ablation(model, blocks, tokens, targets, matrix="W_U")
    assert results[2].delta_loss != 0.0
    for i, r in enumerate(results):
        if i != 2:
            assert abs(r.delta_loss) < 1e-5 and r.delta_acc == 0.0


def _save_planted_checkpoint(path, group, blocks, block_index, epoch):
    config = GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=0, use_wandb=False),
        data={"group": "C8"},
    )
    model = _planted_model(group, blocks, block_index)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config.model_dump(),
    }
    torch.save(payload, path)


def test_energy_trajectory_tracks_planted_switch(tmp_path):
    group = resolve_group("C8")
    blocks = _c8_blocks()
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _save_planted_checkpoint(ckpt_dir / "step_1.pt", group, blocks, block_index=1, epoch=1)
    _save_planted_checkpoint(ckpt_dir / "step_2.pt", group, blocks, block_index=3, epoch=2)

    traj = energy_trajectory(tmp_path, blocks, matrix="W_E")

    assert traj.epochs == [1, 2]
    assert traj.fractions.shape == (2, len(blocks))
    assert traj.fractions[0, 1] > 0.999  # first snapshot planted in block 1
    assert traj.fractions[1, 3] > 0.999  # second snapshot planted in block 3
