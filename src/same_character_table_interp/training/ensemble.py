import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

from same_character_table_interp.groups.catalog import resolve_group
from same_character_table_interp.groups.group import FiniteGroup
from same_character_table_interp.task import build_group_task, train_test_split
from same_character_table_interp.training.batched_adamw import BatchedAdamW
from same_character_table_interp.training.config import GrokkingConfig
from same_character_table_interp.training.logging_jsonl import JSONLLogger
from same_character_table_interp.training.manifest import (
    create_manifest,
    create_run_dir,
    create_run_id,
    save_resolved_config,
    update_manifest,
)
from same_character_table_interp.training.trainer import build_model, set_seed, should_snapshot


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
    group: FiniteGroup,
    train_frac: float,
    seeds: list[int],
    device: str,
    split_seed: int | None = None,
) -> SeedBatches:
    task = build_group_task(group)
    eq = group.order
    tr_tok, tr_tgt, te_tok, te_tgt = [], [], [], []
    for seed in seeds:
        # Mirror GroupGrokkingTrainer: use split_seed if provided, else the member seed.
        effective_split_seed = split_seed if split_seed is not None else seed
        split = train_test_split(task, train_frac, effective_split_seed)
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
        # Open the logger once so rows are streamed per logged epoch (durability parity
        # with single-run trainer whose JSONLLogger.log flushes on every call).
        self._logger = JSONLLogger(self.run_dir / "metrics.jsonl")

    def save_checkpoint(self, name: str, state_dict: dict[str, Tensor], epoch: int) -> None:
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": state_dict, "epoch": epoch, "config": self.config.model_dump()},
            ckpt_dir / f"{name}.pt",
        )

    def log_metrics(self, row: dict[str, Any]) -> None:
        """Write and flush one metrics row immediately (streaming durability)."""
        self._logger.log(dict(row))

    def write_metrics(self, rows: list[dict[str, Any]]) -> None:
        """Batch-write metrics rows. Kept for back-compatibility; loops log_metrics."""
        for row in rows:
            self.log_metrics(row)

    def finalize(self, final_metrics: dict[str, float]) -> None:
        self._logger.close()
        update_manifest(self.run_dir, status="completed", final_metrics=final_metrics)


def _run_one_batch(config: GrokkingConfig, group: FiniteGroup, seeds: list[int]) -> list[Path]:
    # Fix 4: mirror GroupGrokkingTrainer deterministic flag.
    if config.experiment.deterministic:
        torch.use_deterministic_algorithms(True)

    device = config.experiment.device
    base, params, buffers = stack_seeded_models(config, group, seeds, device)
    # Fix 3: pass config.data.split_seed so each member uses the same split-seed
    # logic as GroupGrokkingTrainer (split_seed if not None else member seed).
    batches = build_seed_batches(
        group, config.data.train_frac, seeds, device, split_seed=config.data.split_seed
    )
    grad_fn = make_grad_fn(base)
    eval_fn = make_eval_fn(base)

    opt = BatchedAdamW(
        params,
        lr=config.optim.lr,
        betas=config.optim.betas,
        eps=1e-8,
        weight_decay=config.optim.weight_decay,
    )
    writers = [MemberWriter(config, s) for s in seeds]
    last_metrics: list[dict[str, Any]] = [{} for _ in seeds]

    snap = config.snapshot
    n = len(seeds)
    last_epoch = config.optim.epochs - 1
    prev_test_loss = torch.full((n,), float("inf"), device=device)
    grok_streak = torch.zeros(n, dtype=torch.long, device=device)

    try:
        for epoch in range(config.optim.epochs):
            # Evaluate train metrics BEFORE the gradient step so that logged train_loss and
            # train_acc reflect the pre-update weights — exactly matching GroupGrokkingTrainer,
            # which computes loss/logits in the forward pass and logs them before optimizer.step().
            should_log = epoch % config.optim.log_every == 0 or epoch == last_epoch
            if should_log:
                tr_loss, tr_acc = eval_fn(
                    params, buffers, batches.train_tokens, batches.train_targets
                )

            grads, _ = grad_fn(params, buffers, batches.train_tokens, batches.train_targets)
            opt.step(params, grads)

            event = torch.zeros(n, dtype=torch.bool, device=device)
            if should_log:
                te_loss, te_acc = eval_fn(
                    params, buffers, batches.test_tokens, batches.test_targets
                )
                wn = weight_norms(params)
                for i in range(n):
                    row = {
                        "step": epoch,
                        "train_loss": tr_loss[i].item(),
                        "train_acc": tr_acc[i].item(),
                        "test_loss": te_loss[i].item(),
                        "test_acc": te_acc[i].item(),
                        "weight_norm": wn[i].item(),
                    }
                    # Fix 2: stream each row immediately (flush on every logged epoch).
                    writers[i].log_metrics(row)
                    last_metrics[i] = row
                if snap.event_based:
                    finite = torch.isfinite(prev_test_loss)
                    rel_drop = (prev_test_loss - te_loss) / prev_test_loss
                    event = finite & (rel_drop > snap.event_rel_drop)
                prev_test_loss = te_loss
                grok_streak = torch.where(
                    te_acc >= config.optim.grok_test_acc,
                    grok_streak + 1,
                    torch.zeros_like(grok_streak),
                )

            if should_snapshot(epoch, snap) or bool(event.any()):
                for i in range(n):
                    writers[i].save_checkpoint(
                        f"step_{epoch}", slice_state_dict(params, buffers, i), epoch
                    )

            if config.optim.stop_on_grok and bool(
                (grok_streak >= config.optim.grok_patience).all()
            ):
                for i in range(n):
                    writers[i].save_checkpoint(
                        f"grokked_step_{epoch}", slice_state_dict(params, buffers, i), epoch
                    )
                break

    except Exception:
        # Fix 1: mirror BaseTrainer.fit — mark all in-flight members failed and re-raise.
        tb_str = traceback.format_exc()
        for writer in writers:
            update_manifest(writer.run_dir, status="failed", error=tb_str)
        raise

    for i in range(n):
        writers[i].finalize(last_metrics[i])
    return [w.run_dir for w in writers]


def run_ensemble(config: GrokkingConfig) -> list[Path]:
    group = resolve_group(config.data.group)
    seeds = list(config.ensemble.seeds)
    chunk = config.ensemble.chunk_size or len(seeds)
    out: list[Path] = []
    for start in range(0, len(seeds), chunk):
        out.extend(_run_one_batch(config, group, seeds[start : start + chunk]))
    return out
