import random
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.model import OneLayerTransformer
from finite_group_interp.task import build_group_task, train_test_split

from .config import BaseConfig, GrokkingConfig, SnapshotConfig
from .logging_jsonl import JSONLLogger
from .manifest import (
    create_manifest,
    create_run_dir,
    create_run_id,
    save_resolved_config,
    update_manifest,
)

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseTrainer:
    """A highly extensible, robust base trainer.

    Handles:
        - Absolute reproducibility (seed management)
        - Structured metadata & manifest tracking (run lifecycle, status, Git hashes)
        - Dual logging: Local structured JSONL + Weights & Biases (W&B)
        - Built-in exception safety (crashes are caught, tracebacks stored, loggers closed)
        - Clean lifecycle callback hooks for subclasses to inject custom logic
          (e.g., mechanistic interpretability metrics, attention SVDs, Cayley tables)
    """

    def __init__(self, config: BaseConfig, model: torch.nn.Module):
        self.config = config
        self.model = model
        self.current_epoch = 0

        # Absolute reproducibility
        set_seed(self.config.experiment.seed)

        # Lifecycle directory and run ID
        self.run_id = create_run_id(self.config.experiment.name)
        self.run_dir = create_run_dir(self.run_id)

        # Local logger initialization
        self.jsonl_logger = JSONLLogger(self.run_dir / "metrics.jsonl")

        # Optional W&B logger initialization
        self.wandb_run: Run | None = None
        if self.config.experiment.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.experiment.wandb_project,
                entity=self.config.experiment.wandb_entity,
                name=self.run_id,
                config=self.config.model_dump(),
                id=self.run_id,
            )
        except ImportError:
            print(
                "Warning: use_wandb=True but 'wandb' package is not installed. Logging locally only."
            )

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to both JSONL and W&B."""
        # Log locally
        log_entry: dict[str, Any] = {"step": step} if step is not None else {}
        log_entry.update(metrics)
        self.jsonl_logger.log(log_entry)

        # Log to W&B
        if self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def save_checkpoint(self, name: str, metadata: dict[str, Any] | None = None) -> Path:
        """Saves a PyTorch state dict checkpoint to the run directory."""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{name}.pt"

        checkpoint_payload = {
            "model_state_dict": self.model.state_dict(),
            "epoch": self.current_epoch,
            "config": self.config.model_dump(),
        }
        if metadata is not None:
            checkpoint_payload.update(metadata)

        torch.save(checkpoint_payload, checkpoint_path)
        return checkpoint_path

    def fit(self) -> dict[str, float]:
        """Main training lifecycle wrapper with safety context."""
        create_manifest(self.config, self.run_dir)
        save_resolved_config(self.config, self.run_dir)

        try:
            self.on_train_start()
            final_metrics = self.train_loop()
            self.on_train_end(final_metrics)

            # Record successful completion
            update_manifest(self.run_dir, status="completed", final_metrics=final_metrics)
            return final_metrics

        except Exception as e:
            # Catch any crash, record traceback, update manifest, and propagate
            tb_str = traceback.format_exc()
            print(f"Error occurred during training:\n{tb_str}", file=sys.stderr)
            update_manifest(self.run_dir, status="failed", error=tb_str)

            # Log to W&B
            if self.wandb_run is not None:
                self.wandb_run.alert(
                    title="Run Failed", text=f"Run {self.run_id} failed with error: {str(e)}"
                )

            raise e
        finally:
            self.close()

    def train_loop(self) -> dict[str, float]:
        """Abstract training loop. Override this in subclasses."""
        raise NotImplementedError("Subclasses must implement train_loop()")

    # === Callbacks/Hooks for Downstream Customization ===

    def on_train_start(self) -> None:
        """Called before training starts."""
        pass

    def on_train_end(self, final_metrics: dict[str, float]) -> None:
        """Called after training finishes successfully."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, epoch_metrics: dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at the beginning of each batch/step."""
        pass

    def on_step_end(self, step: int, step_metrics: dict[str, float]) -> None:
        """Called at the end of each batch/step."""
        pass

    def close(self) -> None:
        """Cleanup file handles and close W&B session."""
        self.jsonl_logger.close()
        if self.wandb_run is not None:
            import wandb

            wandb.finish()


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


def build_model(config: GrokkingConfig, group: FiniteGroup) -> OneLayerTransformer:
    """The one model-construction recipe, shared by trainer and analysis loader.

    Vocab sizes derive from the group (one token per element, plus '=' on the
    input side); architecture hyperparameters come from ``config.model``.
    """
    return OneLayerTransformer(
        d_vocab_in=group.order + 1,  # group elements + '='
        d_vocab_out=group.order,
        n_ctx=3,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        use_mlp=config.model.use_mlp,
        d_mlp=config.model.d_mlp,
        activation=config.model.activation,
    )


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
            betas=self.config.optim.betas,
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
        model = build_model(config, group)
        return cls(config, model, group)

    def train_loop(self) -> dict[str, float]:
        snap = self.config.snapshot
        last_epoch = self.config.optim.epochs - 1
        prev_test_loss = float("inf")
        metrics: dict[str, float] = {}
        grok_streak = 0  # consecutive evals with test_acc above the grok threshold

        for epoch in range(self.config.optim.epochs):
            self.current_epoch = epoch
            self.optimizer.zero_grad()
            logits = self.model(self.train_tokens)
            readout = logits[:, -1, :]  # the '=' position
            loss = F.cross_entropy(readout, self.train_targets)
            torch.autograd.backward(loss)
            self.optimizer.step()

            event = False
            if epoch % self.config.optim.log_every == 0 or epoch == last_epoch:
                train_acc = (readout.argmax(dim=-1) == self.train_targets).float().mean().item()
                test_m = self._evaluate(self.test_tokens, self.test_targets)
                # Total L2 norm of all weights -- the grokking "progress measure":
                # weight decay drives this down, and generalization tracks it.
                weight_norm = (
                    torch.stack([p.detach().pow(2).sum() for p in self.model.parameters()])
                    .sum()
                    .sqrt()
                    .item()
                )
                metrics = {
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "test_loss": test_m["loss"],
                    "test_acc": test_m["accuracy"],
                    "weight_norm": weight_norm,
                }
                self.log(metrics, step=epoch)

                if epoch % self.config.optim.print_every == 0 or epoch == last_epoch:
                    print(
                        f"[{epoch:>6}/{self.config.optim.epochs}] "
                        f"train loss={metrics['train_loss']:.4f} acc={metrics['train_acc']:.3f} | "
                        f"test loss={metrics['test_loss']:.4f} acc={metrics['test_acc']:.3f}",
                        flush=True,
                    )

                # Snapshot densely when the test loss drops sharply
                if snap.event_based and prev_test_loss < float("inf"):
                    rel_drop = (prev_test_loss - test_m["loss"]) / prev_test_loss
                    event = rel_drop > snap.event_rel_drop
                prev_test_loss = test_m["loss"]

                if test_m["accuracy"] >= self.config.optim.grok_test_acc:
                    grok_streak += 1
                else:
                    grok_streak = 0

            if should_snapshot(epoch, snap) or event:
                self.save_checkpoint(f"step_{epoch}")

            if self.config.optim.stop_on_grok and grok_streak >= self.config.optim.grok_patience:
                self.save_checkpoint(f"grokked_step_{epoch}")
                print(
                    f"grokked: test_acc >= {self.config.optim.grok_test_acc} for "
                    f"{self.config.optim.grok_patience} evals -- stopping at epoch {epoch}",
                    flush=True,
                )
                break
        return metrics

    def _evaluate(self, tokens: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            logits = self.model(tokens)
            readout = logits[:, -1, :]
            loss = F.cross_entropy(readout, targets)
            accuracy = (readout.argmax(dim=1) == targets).float().mean()
        return {"loss": loss.item(), "accuracy": accuracy.item()}
