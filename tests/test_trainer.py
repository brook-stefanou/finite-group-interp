import json

import torch

from finite_group_interp.training.config import ExperimentConfig
from finite_group_interp.training.config import (
    DataConfig,
    GrokkingConfig,
    ModelConfig,
    OptimConfig,
)
from finite_group_interp.training.trainer import GroupGrokkingTrainer


def _config(group="C4", train_frac=0.8, epochs=300, weight_decay=1.0, log_every=1):
    return GrokkingConfig(
        experiment=ExperimentConfig(name="test", seed=0),
        data=DataConfig(group=group, train_frac=train_frac),
        model=ModelConfig(d_model=32, n_heads=4, d_mlp=64),
        optim=OptimConfig(lr=1e-3, weight_decay=weight_decay, epochs=epochs, log_every=log_every),
    )


def test_from_config_derives_vocab_from_group(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4"))

    assert trainer.group.order == 4
    assert trainer.model.d_vocab_in == 5  # |G| + 1 for the "=" token
    assert trainer.model.d_vocab_out == 4  # predict a group element
    assert trainer.model.n_ctx == 3  # [a, b, =]


def test_inputs_have_equals_token_appended(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4"))
    eq = trainer.group.order

    for tokens in (trainer.train_tokens, trainer.test_tokens):
        assert tokens.shape[1] == 3  # [a, b, =]
        assert tokens.dtype == torch.long
        assert (tokens[:, -1] == eq).all()  # last column is the "=" id
        assert (tokens[:, :2] < eq).all()  # operands are real group elements


def test_split_sizes_cover_all_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", train_frac=0.75))

    n_total = trainer.group.order**2  # every ordered pair
    n_train = trainer.train_tokens.shape[0]
    n_test = trainer.test_tokens.shape[0]
    assert n_train + n_test == n_total
    assert n_train == round(0.75 * n_total)


def test_targets_match_the_group_product(tmp_path, monkeypatch):
    # Every [a, b, =] row must be paired with the true product a*b.
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="S3"))

    rows = trainer.train_tokens.cpu().tolist()
    targets = trainer.train_targets.cpu().tolist()
    for (a, b, _eq), t in zip(rows, targets):
        assert trainer.group.cayley_table[a, b] == t


def test_fit_reduces_train_loss_and_memorises(tmp_path, monkeypatch):
    # No weight decay + enough epochs: the model should easily fit the tiny
    # train set, so train loss drops and train accuracy approaches 1.
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", epochs=500, weight_decay=0.0))

    initial = trainer._evaluate(trainer.train_tokens, trainer.train_targets)["loss"]
    final = trainer.fit()

    assert final["train_loss"] < initial
    assert final["train_acc"] > 0.9


def test_fit_writes_weight_snapshots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", epochs=20))
    trainer.fit()

    checkpoints = list((trainer.run_dir / "checkpoints").glob("*.pt"))
    assert len(checkpoints) > 0  # snapshot schedule fired (step 0 + powers of two)


def test_log_every_controls_logging_cadence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", epochs=20, log_every=5))
    trainer.fit()

    lines = (trainer.run_dir / "metrics.jsonl").read_text().splitlines()
    steps = [json.loads(line)["step"] for line in lines if line.strip()]
    # logged at multiples of 5, plus the final epoch (19)
    assert steps == [0, 5, 10, 15, 19]


def test_from_config_initialises_reproducibly(tmp_path, monkeypatch):
    # Same config (same seed) => identical initial weights, because from_config
    # seeds before building the model.
    monkeypatch.chdir(tmp_path)
    a = GroupGrokkingTrainer.from_config(_config(group="S3"))
    b = GroupGrokkingTrainer.from_config(_config(group="S3"))
    assert torch.equal(a.model.W_E, b.model.W_E)


def test_determinism_enabled_and_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", epochs=5))
    assert torch.are_deterministic_algorithms_enabled()
    assert str(trainer.device) == "cpu"

    trainer.fit()
    manifest = json.loads((trainer.run_dir / "manifest.json").read_text())
    assert manifest["deterministic"] is True
    assert manifest["device"] == "cpu"


def test_optimizer_uses_configured_betas(tmp_path, monkeypatch):
    # The configured betas must reach the AdamW instance, not PyTorch's defaults.
    monkeypatch.chdir(tmp_path)
    config = _config(group="C4")
    config.optim.betas = (0.9, 0.95)
    trainer = GroupGrokkingTrainer.from_config(config)
    assert trainer.optimizer.param_groups[0]["betas"] == (0.9, 0.95)


def test_logs_weight_norm(tmp_path, monkeypatch):
    # weight_norm is logged as the grokking progress measure.
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", epochs=5))
    final = trainer.fit()
    assert "weight_norm" in final
    assert final["weight_norm"] > 0


def test_stop_on_grok_halts_before_max_epochs(tmp_path, monkeypatch):
    # With early-stop on and a trivially-reached threshold, training halts well
    # before the epoch cap and saves a grokked checkpoint.
    monkeypatch.chdir(tmp_path)
    config = _config(group="C4", train_frac=0.75, weight_decay=0.0, epochs=400, log_every=1)
    config.optim.stop_on_grok = True
    config.optim.grok_test_acc = 0.01  # exercises the early-stop path deterministically
    config.optim.grok_patience = 2
    trainer = GroupGrokkingTrainer.from_config(config)
    trainer.fit()
    assert trainer.current_epoch < 399  # stopped early
    assert list((trainer.run_dir / "checkpoints").glob("grokked_step_*.pt"))


def test_no_early_stop_runs_full_budget(tmp_path, monkeypatch):
    # Default (stop_on_grok=False) must run every epoch even if test_acc is high.
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(
        _config(group="C4", train_frac=0.75, weight_decay=0.0, epochs=20, log_every=1)
    )
    trainer.fit()
    assert trainer.current_epoch == 19
