import torch
import torch.nn.functional as F
import numpy as np
from core.trainer import BaseTrainer, set_seed
from core.models.one_layer_transformer import OneLayerTransformer
from finite_groups.group import FiniteGroup
from finite_groups.catalog import resolve_group
from finite_groups.experiments.config import GrokkingConfig, SnapshotConfig
from finite_groups.task import build_group_task, train_test_split


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def should_snapshot(step: int, config: SnapshotConfig) -> bool:
    """When to checkpoint weights: step 0, powers of two early, then periodic."""
    if not config.enabled or step < 0:
        return False
    if step == 0:
        return True
    if step <= config.log_dense_until and _is_power_of_two(step):
        return True
    return step % config.interval == 0


class GroupGrokkingTrainer(BaseTrainer):
    config: GrokkingConfig

    def __init__(self, config: GrokkingConfig, model: torch.nn.Module, group: FiniteGroup):
        super().__init__(config, model)
        if config.experiment.deterministic:
            torch.use_deterministic_algorithms(True)
        self.group = group
        self.device = torch.device(config.experiment.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )
        task = build_group_task(self.group)
        split = train_test_split(
            task,
            self.config.data.train_frac,
            self.config.data.split_seed or self.config.experiment.seed,
        )

        eq = self.group.order

        def _to_tokens(
            pairs: np.ndarray,
        ) -> torch.Tensor:  # [train_frac * group.order **2 ,2] -> [train_frac * group.order **2, 3]
            eq_col = np.full((len(pairs), 1), eq)  # [train_frac * group.order **2, 1]
            seqs = np.concatenate([pairs, eq_col], axis=1)
            return torch.tensor(seqs, dtype=torch.long, device=self.device)

        self.train_tokens = _to_tokens(split.train_inputs)
        self.test_tokens = _to_tokens(split.test_inputs)
        self.train_targets = torch.tensor(split.train_targets, dtype=torch.long, device=self.device)
        self.test_targets = torch.tensor(split.test_targets, dtype=torch.long, device=self.device)

    @classmethod
    def from_config(cls, config: GrokkingConfig) -> "GroupGrokkingTrainer":
        set_seed(config.experiment.seed)  # seed before model init so weights are reproducible
        group = resolve_group(config.data.group)
        model = OneLayerTransformer(
            d_vocab_in=group.order + 1,  # group elements + '='
            d_vocab_out=group.order,
            n_ctx=3,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            use_mlp=config.model.use_mlp,
            d_mlp=config.model.d_mlp,
            activation=config.model.activation,
            init_std=config.model.init_std,
        )
        return cls(config, model, group)

    def train_loop(self):
        snap = self.config.snapshot
        last_epoch = self.config.optim.epochs - 1
        prev_test_loss = float("inf")
        metrics = {}

        for epoch in range(self.config.optim.epochs):
            self.current_epoch = epoch
            self.optimizer.zero_grad()
            logits = self.model(self.train_tokens)
            readout = logits[:, -1, :]  # the '=' position
            loss = F.cross_entropy(readout, self.train_targets)
            loss.backward()
            self.optimizer.step()

            event = False
            if epoch % self.config.optim.log_every == 0 or epoch == last_epoch:
                train_acc = (readout.argmax(dim=-1) == self.train_targets).float().mean().item()
                test_m = self._evaluate(self.test_tokens, self.test_targets)
                metrics = {
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "test_loss": test_m["loss"],
                    "test_acc": test_m["accuracy"],
                }
                self.log(metrics, step=epoch)

                # Snapshot densely when the test loss drops sharply
                if snap.event_based and prev_test_loss < float("inf"):
                    rel_drop = (prev_test_loss - test_m["loss"]) / prev_test_loss
                    event = rel_drop > snap.event_rel_drop
                prev_test_loss = test_m["loss"]

            if should_snapshot(epoch, snap) or event:
                self.save_checkpoint(f"step_{epoch}")
        return metrics

    def _evaluate(self, tokens: torch.Tensor, targets: torch.Tensor) -> dict:
        with torch.no_grad():
            logits = self.model(tokens)
            readout = logits[:, -1, :]
            loss = F.cross_entropy(readout, targets)
            accuracy = (readout.argmax(dim=1) == targets).float().mean()
        return {"loss": loss.item(), "accuracy": accuracy.item()}
