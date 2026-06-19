from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.func import stack_module_state

from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.task import build_group_task, train_test_split
from finite_group_interp.training.config import GrokkingConfig
from finite_group_interp.training.trainer import build_model, set_seed


@dataclass(frozen=True)
class SeedBatches:
    train_tokens: Tensor  # [N, n_train, 3] int64
    train_targets: Tensor  # [N, n_train] int64
    test_tokens: Tensor  # [N, n_test, 3] int64
    test_targets: Tensor  # [N, n_test] int64


def _to_tokens(pairs: np.ndarray, eq_id: int) -> np.ndarray:
    eq_col = np.full((len(pairs), 1), eq_id)
    return np.concatenate([pairs, eq_col], axis=1)


def build_seed_batches(
    group: FiniteGroup, train_frac: float, seeds: list[int], device: str
) -> SeedBatches:
    task = build_group_task(group)
    eq = group.order
    tr_tok, tr_tgt, te_tok, te_tgt = [], [], [], []
    for seed in seeds:
        split = train_test_split(task, train_frac, seed)
        tr_tok.append(_to_tokens(split.train_inputs, eq))
        tr_tgt.append(split.train_targets)
        te_tok.append(_to_tokens(split.test_inputs, eq))
        te_tgt.append(split.test_targets)
    long = torch.long
    return SeedBatches(
        train_tokens=torch.tensor(np.stack(tr_tok), dtype=long, device=device),
        train_targets=torch.tensor(np.stack(tr_tgt), dtype=long, device=device),
        test_tokens=torch.tensor(np.stack(te_tok), dtype=long, device=device),
        test_targets=torch.tensor(np.stack(te_tgt), dtype=long, device=device),
    )


def stack_seeded_models(
    config: GrokkingConfig, group: FiniteGroup, seeds: list[int], device: str
) -> tuple[torch.nn.Module, dict[str, Tensor], dict[str, Tensor]]:
    models = []
    for seed in seeds:
        set_seed(seed)
        models.append(build_model(config, group).to(device))
    params, buffers = stack_module_state(models)
    base = models[0]  # template; its params are ignored by functional_call
    return base, params, buffers
