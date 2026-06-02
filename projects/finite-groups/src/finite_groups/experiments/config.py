"""Configuration schema for group-grokking experiments.

Extends the core ``BaseConfig`` (so ``BaseTrainer`` still accepts it) with the
data, model, optimiser, and snapshot sub-configs this project needs. The model
hyperparameters live here as plain data -- the model reads them, this schema
does not depend on the model.
"""

from typing import Literal

from pydantic import BaseModel, Field

from core.config_schema import BaseConfig


class DataConfig(BaseModel):
    group: str = "C8"  # a catalog / build_group spec, e.g. "C8", "S3", "16,3"
    train_frac: float = Field(0.5, gt=0.0, lt=1.0)  # the grokking knob
    split_seed: int | None = None  # falls back to experiment.seed when None


class ModelConfig(BaseModel):
    d_model: int = Field(64, gt=0)
    n_heads: int = Field(4, gt=0)
    d_mlp: int = Field(256, gt=0)
    use_mlp: bool = True  # MLP carries the group-multiply circuit; usually keep on
    activation: Literal["relu", "gelu", "silu"] = "relu"
    init_std: float = Field(0.02, gt=0.0)


class OptimConfig(BaseModel):
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(1.0, ge=0.0)  # high weight decay drives grokking
    epochs: int = Field(10_000, gt=0)
    full_batch: bool = True
    log_every: int = Field(1, gt=0)  # evaluate + log metrics every N epochs


class SnapshotConfig(BaseModel):
    enabled: bool = True
    log_dense_until: int = Field(1024, ge=0)  # powers of 2 snapshotted up to here
    interval: int = Field(1000, gt=0)  # then snapshot every `interval` steps
    # Event-based densification: also snapshot when the test loss drops sharply between evals
    event_based: bool = True
    event_rel_drop: float = Field(0.1, gt=0.0)  # relative test-loss drop that triggers a snapshot


class GrokkingConfig(BaseConfig):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
