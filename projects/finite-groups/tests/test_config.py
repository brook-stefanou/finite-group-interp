import pytest
from pydantic import ValidationError

from core.config_schema import BaseConfig, ExperimentConfig
from finite_groups.experiments.config import (
    DataConfig,
    GrokkingConfig,
    ModelConfig,
    OptimConfig,
    SnapshotConfig,
)


def _experiment() -> ExperimentConfig:
    return ExperimentConfig(name="test", seed=0)


def test_defaults_are_sensible():
    cfg = GrokkingConfig(experiment=_experiment())
    assert 0.0 < cfg.data.train_frac < 1.0
    assert cfg.model.use_mlp is True
    assert cfg.optim.epochs > 0
    assert cfg.snapshot.enabled is True


def test_train_frac_default_comes_from_config():
    # The config is the single source of the train_frac default.
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.data.train_frac == DataConfig().train_frac


def test_train_frac_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DataConfig(train_frac=1.5)
    with pytest.raises(ValidationError):
        DataConfig(train_frac=0.0)


def test_overrides_apply():
    cfg = GrokkingConfig(
        experiment=_experiment(),
        data=DataConfig(group="S3", train_frac=0.3),
        model=ModelConfig(d_model=64, use_mlp=False),
    )
    assert cfg.data.group == "S3"
    assert cfg.data.train_frac == 0.3
    assert cfg.model.d_model == 64
    assert cfg.model.use_mlp is False


def test_remains_a_base_config():
    # BaseTrainer accepts a BaseConfig, so the subclass must still be one.
    cfg = GrokkingConfig(experiment=_experiment())
    assert isinstance(cfg, BaseConfig)


def test_positive_constraints_enforced():
    with pytest.raises(ValidationError):
        OptimConfig(lr=0.0)
    with pytest.raises(ValidationError):
        OptimConfig(epochs=0)


def test_activation_default_and_validation():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.model.activation == "relu"
    with pytest.raises(ValidationError):
        ModelConfig(activation="tanh")  # not in the allowed set


def test_init_std_is_positive():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.model.init_std > 0
    with pytest.raises(ValidationError):
        ModelConfig(init_std=0.0)


def test_log_every_default_and_positive():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.optim.log_every == 1
    with pytest.raises(ValidationError):
        OptimConfig(log_every=0)


def test_print_every_default_and_positive():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.optim.print_every == 1000
    with pytest.raises(ValidationError):
        OptimConfig(print_every=0)


def test_snapshot_event_fields():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.snapshot.event_based is True
    assert cfg.snapshot.event_rel_drop > 0
    with pytest.raises(ValidationError):
        SnapshotConfig(event_rel_drop=0.0)


def test_experiment_reproducibility_defaults():
    exp = _experiment()
    assert exp.device == "cpu"  # reproducible + fast for small models
    assert exp.deterministic is True
    assert exp.use_wandb is True  # tracking on by default; tests/CI disable via WANDB_MODE
