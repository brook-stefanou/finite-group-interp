import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from .config_schema import BaseConfig
from .logging_jsonl import JSONLLogger
from .manifest import create_manifest, create_run_dir, update_manifest, save_resolved_config


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
    """A highly extensible, robust base trainer for the AI Safety Portfolio.
    
    Handles:
        - Absolute reproducibility (seed management)
        - Structured metadata & manifest tracking (run lifecycle, status, Git hashes)
        - Dual logging: Local structured JSONL + Weights & Biases (W&B)
        - Built-in exception safety (crashes are caught, tracebacks stored, loggers closed)
        - Clean lifecycle callback hooks for downstream projects to inject custom logic 
          (e.g., mechanistic interpretability metrics, attention SVDs, Cayley tables)
    """

    def __init__(self, config: BaseConfig, model: torch.nn.Module):
        self.config = config
        self.model = model
        self.current_epoch = 0
        
        # Absolute reproducibility
        set_seed(self.config.experiment.seed)
        
        # Lifecycle directory and run ID
        from .manifest import create_run_id
        self.run_id = create_run_id(self.config.experiment.name)
        self.run_dir = create_run_dir(self.run_id)
        
        # Local logger initialization
        self.jsonl_logger = JSONLLogger(self.run_dir / "metrics.jsonl")
        
        # Optional W&B logger initialization
        self.wandb_run = None
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
            print("Warning: use_wandb=True but 'wandb' package is not installed. Logging locally only.")

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Log metrics to both JSONL and W&B."""
        # Log locally
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)
        self.jsonl_logger.log(log_entry)
        
        # Log to W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def save_checkpoint(self, name: str, metadata: dict | None = None) -> Path:
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

    def fit(self) -> dict:
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
                import wandb
                wandb.alert(title="Run Failed", text=f"Run {self.run_id} failed with error: {str(e)}")
            
            raise e
        finally:
            self.close()

    def train_loop(self) -> dict:
        """Abstract training loop. Override this in child projects."""
        raise NotImplementedError("Subclasses must implement train_loop()")

    # === Callbacks/Hooks for Downstream Customization ===

    def on_train_start(self) -> None:
        """Called before training starts."""
        pass

    def on_train_end(self, final_metrics: dict) -> None:
        """Called after training finishes successfully."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, epoch_metrics: dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at the beginning of each batch/step."""
        pass

    def on_step_end(self, step: int, step_metrics: dict) -> None:
        """Called at the end of each batch/step."""
        pass

    def close(self) -> None:
        """Cleanup file handles and close W&B session."""
        self.jsonl_logger.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
