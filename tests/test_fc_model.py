"""The FC baseline (architecture confound) -- drop-in shape contract + learnability.

The FC must satisfy the same contract the transformer does so the trainer, eval,
and every analysis instrument work unchanged: logits [batch, n_ctx, d_vocab_out]
read at position -1, a shared W_E, and a cache whose resid_final[:, -1] is the
representation the coset probe reads.
"""

import numpy as np
import torch
import torch.nn.functional as F

from same_character_table_interp.groups.catalog import resolve_group
from same_character_table_interp.groups.group import FiniteGroup
from same_character_table_interp.model import FCModel, OneLayerTransformer
from same_character_table_interp.task import build_group_task
from same_character_table_interp.training.config import (
    DataConfig,
    ExperimentConfig,
    GrokkingConfig,
    ModelConfig,
    OptimConfig,
)
from same_character_table_interp.training.trainer import build_model


def _config(arch: str, group: str = "C4") -> GrokkingConfig:
    return GrokkingConfig(
        experiment=ExperimentConfig(name="test", seed=0),
        data=DataConfig(group=group),
        model=ModelConfig(arch=arch, d_model=32, d_mlp=64),
        optim=OptimConfig(epochs=10),
    )


def _tokens(group: FiniteGroup) -> tuple[torch.Tensor, torch.Tensor]:
    """All-pairs (a, b, '=') tokens and their a*b targets, via the task builder."""
    task = build_group_task(group)
    eq = np.full((task.inputs.shape[0], 1), group.order)
    tokens = torch.tensor(np.hstack([task.inputs, eq]), dtype=torch.long)
    return tokens, torch.tensor(task.targets, dtype=torch.long)


def test_build_model_selects_arch() -> None:
    g = resolve_group("C4")
    assert isinstance(build_model(_config("fc"), g), FCModel)
    assert isinstance(build_model(_config("transformer"), g), OneLayerTransformer)


def test_fc_forward_shape_and_broadcast_read_position() -> None:
    g = resolve_group("S3")
    n = g.order
    m = FCModel(d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=32, activation="relu")
    tokens, _ = _tokens(g)
    logits = m(tokens)
    assert logits.shape == (n * n, 3, n)
    # The prediction is broadcast across positions, so every position equals the
    # read position -- the contract the trainer/eval/functional-form rely on.
    assert torch.allclose(logits[:, 0, :], logits[:, -1, :])


def test_fc_cache_exposes_read_representation() -> None:
    g = resolve_group("S3")
    n, d_mlp = g.order, 32
    m = FCModel(
        d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=d_mlp, activation="relu"
    )
    tokens, _ = _tokens(g)
    cache = m(tokens, return_cache=True)
    assert cache["resid_final"].shape == (n * n, 3, d_mlp)
    assert cache["logits"].shape == (n * n, 3, n)
    assert cache["embed"].shape == (n * n, 3, 16)
    # The coset probe reads resid_final[:, -1]: the post-ReLU hidden, so non-negative.
    assert (cache["resid_final"][:, -1, :] >= 0).all()


def test_fc_learns_a_batch() -> None:
    g = resolve_group("C4")
    n = g.order
    torch.manual_seed(0)
    m = FCModel(d_vocab_in=n + 1, d_vocab_out=n, n_ctx=3, d_model=16, d_mlp=64, activation="relu")
    tokens, targets = _tokens(g)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = last = 0.0
    for i in range(60):
        opt.zero_grad()
        loss = F.cross_entropy(m(tokens)[:, -1, :], targets)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first  # the FC actually optimises the task
