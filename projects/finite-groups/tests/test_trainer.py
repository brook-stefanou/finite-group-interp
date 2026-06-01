import json

import torch

from core.config_schema import ExperimentConfig
from finite_groups.grokking.config import (
    DataConfig,
    GrokkingConfig,
    ModelConfig,
    OptimConfig,
)
from finite_groups.grokking.trainer import GroupGrokkingTrainer


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
    assert trainer.model.d_vocab_in == 5   # |G| + 1 for the "=" token
    assert trainer.model.d_vocab_out == 4  # predict a group element
    assert trainer.model.n_ctx == 3        # [a, b, =]


def test_inputs_have_equals_token_appended(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4"))
    eq = trainer.group.order

    for tokens in (trainer.train_tokens, trainer.test_tokens):
        assert tokens.shape[1] == 3                  # [a, b, =]
        assert tokens.dtype == torch.long
        assert (tokens[:, -1] == eq).all()           # last column is the "=" id
        assert (tokens[:, :2] < eq).all()            # operands are real group elements


def test_split_sizes_cover_all_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = GroupGrokkingTrainer.from_config(_config(group="C4", train_frac=0.75))

    n_total = trainer.group.order ** 2  # every ordered pair
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
