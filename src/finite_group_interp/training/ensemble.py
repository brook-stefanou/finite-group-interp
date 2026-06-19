from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.task import build_group_task, train_test_split
from finite_group_interp.training.config import GrokkingConfig
from finite_group_interp.training.logging_jsonl import JSONLLogger
from finite_group_interp.training.manifest import (
    create_manifest,
    create_run_dir,
    create_run_id,
    save_resolved_config,
    update_manifest,
)
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


def make_grad_fn(
    base_model: torch.nn.Module,
) -> Callable[
    [dict[str, Tensor], dict[str, Tensor], Tensor, Tensor], tuple[dict[str, Tensor], Tensor]
]:
    def loss_of(
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        tokens: Tensor,
        targets: Tensor,
    ) -> Tensor:
        logits = functional_call(base_model, (params, buffers), (tokens,))
        return F.cross_entropy(logits[:, -1, :], targets)

    return vmap(grad_and_value(loss_of), in_dims=(0, 0, 0, 0))


def make_eval_fn(
    base_model: torch.nn.Module,
) -> Callable[[dict[str, Tensor], dict[str, Tensor], Tensor, Tensor], tuple[Tensor, Tensor]]:
    def eval_of(
        params: dict[str, Tensor],
        buffers: dict[str, Tensor],
        tokens: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = functional_call(base_model, (params, buffers), (tokens,))
        readout = logits[:, -1, :]
        loss = F.cross_entropy(readout, targets)
        acc = (readout.argmax(dim=-1) == targets).float().mean()
        return loss, acc

    return vmap(eval_of, in_dims=(0, 0, 0, 0))


def weight_norms(params: dict[str, Tensor]) -> Tensor:
    # per-member L2 over all params: sqrt(sum_k sum_dims!=0 p^2)
    per = [p.detach().pow(2).flatten(start_dim=1).sum(dim=1) for p in params.values()]
    return torch.stack(per, dim=0).sum(dim=0).sqrt()


def member_config(config: GrokkingConfig, seed: int) -> GrokkingConfig:
    data = config.model_dump()
    arch = config.model.arch
    prefix = "fc" if arch == "fc" else "pair"
    name = (
        f"{prefix}-{config.data.group}-s{seed}"
        f"-wd{config.optim.weight_decay}-f{config.data.train_frac}"
    )
    data["experiment"]["seed"] = seed
    data["experiment"]["name"] = name
    data["experiment"]["use_wandb"] = False
    return GrokkingConfig(**data)


def slice_state_dict(
    params: dict[str, Tensor], buffers: dict[str, Tensor], i: int
) -> dict[str, Tensor]:
    sd = {k: v[i].detach().clone() for k, v in params.items()}
    sd.update({k: v[i].detach().clone() for k, v in buffers.items()})
    return sd


class MemberWriter:
    """Writes one ensemble member as a standard run dir (manifest + metrics + ckpts)."""

    def __init__(self, config: GrokkingConfig, seed: int) -> None:
        self.config = member_config(config, seed)
        run_id = create_run_id(self.config.experiment.name)
        self.run_dir = create_run_dir(run_id)
        create_manifest(self.config, self.run_dir)  # status="running"
        save_resolved_config(self.config, self.run_dir)

    def save_checkpoint(self, name: str, state_dict: dict[str, Tensor], epoch: int) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": state_dict, "epoch": epoch, "config": self.config.model_dump()},
            ckpt_dir / f"{name}.pt",
        )

    def write_metrics(self, rows: list[dict[str, Any]]) -> None:
        logger = JSONLLogger(self.run_dir / "metrics.jsonl")
        for row in rows:
            logger.log(dict(row))
        logger.close()

    def finalize(self, final_metrics: dict[str, float]) -> None:
        update_manifest(self.run_dir, status="completed", final_metrics=final_metrics)
