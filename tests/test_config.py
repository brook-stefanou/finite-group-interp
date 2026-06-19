import pytest
from pydantic import ValidationError

from finite_group_interp.training.config import BaseConfig, ExperimentConfig
from finite_group_interp.training.config import (
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


def test_grok_early_stop_defaults_and_validation():
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.optim.stop_on_grok is False  # opt-in
    assert cfg.optim.grok_test_acc == 0.99
    assert cfg.optim.grok_patience == 5
    with pytest.raises(ValidationError):
        OptimConfig(grok_test_acc=1.5)  # must be <= 1
    with pytest.raises(ValidationError):
        OptimConfig(grok_patience=0)


def test_betas_default_and_validation():
    # beta2=0.98 is the modular-addition default that damps slingshot loss spikes.
    cfg = GrokkingConfig(experiment=_experiment())
    assert cfg.optim.betas == (0.9, 0.98)
    with pytest.raises(ValidationError):
        OptimConfig(betas=(0.9, 1.0))  # beta must be < 1
    with pytest.raises(ValidationError):
        OptimConfig(betas=(-0.1, 0.98))  # beta must be >= 0


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


def test_ensemble_config_defaults_off():
    from finite_group_interp.training.config import GrokkingConfig

    cfg = GrokkingConfig(experiment={"name": "x", "seed": 0})
    assert cfg.ensemble.enabled is False
    assert cfg.ensemble.seeds == []
    assert cfg.ensemble.chunk_size is None


def test_ensemble_config_overrides():
    from finite_group_interp.training.config import GrokkingConfig

    cfg = GrokkingConfig(
        experiment={"name": "x", "seed": 0},
        ensemble={"enabled": True, "seeds": [1, 2, 3], "chunk_size": 2},
    )
    assert cfg.ensemble.enabled is True
    assert cfg.ensemble.seeds == [1, 2, 3]
    assert cfg.ensemble.chunk_size == 2
