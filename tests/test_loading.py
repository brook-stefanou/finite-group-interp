import json
from pathlib import Path

import numpy as np
import pytest
import torch

from finite_group_interp.analysis.loading import list_checkpoints, load_checkpoint, load_run
from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.task import build_group_task, train_test_split
from finite_group_interp.training.config import ExperimentConfig, GrokkingConfig
from finite_group_interp.training.trainer import build_model


def _config(group: str = "C8") -> GrokkingConfig:
    return GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=0, use_wandb=False),
        data={"group": group},
    )


def _save_checkpoint(path: Path, config: GrokkingConfig, epoch: int = 7) -> torch.nn.Module:
    """Save a payload in the trainer's exact format (mirrors BaseTrainer.save_checkpoint)."""
    group = resolve_group(config.data.group)
    model = build_model(config, group)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "config": config.model_dump(),
    }
    torch.save(payload, path)
    return model


def test_load_checkpoint_round_trips_weights_exactly(tmp_path):
    config = _config()
    original = _save_checkpoint(tmp_path / "step_7.pt", config, epoch=7)
    loaded = load_checkpoint(tmp_path / "step_7.pt")
    assert set(loaded.model.state_dict()) == set(original.state_dict())
    for key, tensor in original.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], tensor), key


def test_load_checkpoint_metadata(tmp_path):
    config = _config()
    _save_checkpoint(tmp_path / "step_7.pt", config, epoch=7)
    loaded = load_checkpoint(tmp_path / "step_7.pt")
    assert loaded.config == config
    assert loaded.group.order == 8
    assert loaded.epoch == 7
    assert loaded.path == tmp_path / "step_7.pt"
    assert not loaded.model.training  # eval mode


def test_load_checkpoint_tolerates_missing_causal_mask(tmp_path):
    # Pre-June-4 checkpoints predate the causal_mask buffer; it is rebuilt in
    # __init__, so a payload without it must still load.
    config = _config()
    original = _save_checkpoint(tmp_path / "step_3.pt", config, epoch=3)
    payload = torch.load(tmp_path / "step_3.pt", weights_only=True)
    del payload["model_state_dict"]["causal_mask"]
    torch.save(payload, tmp_path / "old_style.pt")
    loaded = load_checkpoint(tmp_path / "old_style.pt")
    assert torch.equal(loaded.model.W_E, original.state_dict()["W_E"])


def test_load_checkpoint_rejects_other_state_dict_mismatches(tmp_path):
    config = _config()
    _save_checkpoint(tmp_path / "step_3.pt", config, epoch=3)
    payload = torch.load(tmp_path / "step_3.pt", weights_only=True)
    del payload["model_state_dict"]["W_E"]
    torch.save(payload, tmp_path / "broken.pt")
    with pytest.raises(RuntimeError, match="W_E"):
        load_checkpoint(tmp_path / "broken.pt")


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        load_checkpoint(tmp_path / "nope.pt")


def test_load_checkpoint_rejects_config_weight_dim_disagreement(tmp_path):
    # A config whose dims disagree with the saved weights must fail loudly,
    # never load garbage: strict=False still raises on size mismatches.
    config = _config()
    _save_checkpoint(tmp_path / "step_3.pt", config, epoch=3)
    payload = torch.load(tmp_path / "step_3.pt", weights_only=True)
    payload["config"]["model"]["d_model"] = 32  # saved weights are d_model=64
    torch.save(payload, tmp_path / "mismatched.pt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_checkpoint(tmp_path / "mismatched.pt")


def test_load_checkpoint_not_a_trainer_payload_raises(tmp_path):
    torch.save({"weights": torch.zeros(3)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a trainer checkpoint"):
        load_checkpoint(tmp_path / "other.pt")


def test_load_checkpoint_round_trips_no_mlp_model(tmp_path):
    from finite_group_interp.training.config import ModelConfig

    config = GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=0, use_wandb=False),
        data={"group": "C8"},
        model=ModelConfig(use_mlp=False),
    )
    original = _save_checkpoint(tmp_path / "step_5.pt", config, epoch=5)
    loaded = load_checkpoint(tmp_path / "step_5.pt")
    assert "W_in" not in loaded.model.state_dict()
    for key, tensor in original.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], tensor), key


def _make_run_dir(tmp_path: Path, config: GrokkingConfig) -> Path:
    """A minimal run dir matching the trainer's on-disk layout."""
    run_dir = tmp_path / "2026-06-07_000000_t"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    _save_checkpoint(ckpt_dir / "step_3.pt", config, epoch=3)
    _save_checkpoint(ckpt_dir / "step_12.pt", config, epoch=12)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 3, "test_acc": 0.1})
        + "\n"
        + json.dumps({"step": 12, "test_acc": 0.9})
        + "\n"
    )
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": "t", "status": "completed"}))
    return run_dir


def test_list_checkpoints_sorts_numerically(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    # Ordering only reads filenames, so empty files suffice.
    for name in ["step_10.pt", "step_2.pt", "grokked_step_5.pt", "step_0.pt"]:
        (ckpt_dir / name).touch()
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["step_0.pt", "step_2.pt", "grokked_step_5.pt", "step_10.pt"]


def test_list_checkpoints_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoints"):
        list_checkpoints(tmp_path)


def test_list_checkpoints_empty_dir_raises(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match="no .pt checkpoints"):
        list_checkpoints(tmp_path)


def test_load_run_defaults_to_latest_epoch(tmp_path):
    run_dir = _make_run_dir(tmp_path, _config())
    run = load_run(run_dir)
    assert run.checkpoint.epoch == 12
    assert run.run_dir == run_dir


def test_load_run_checkpoint_override_by_stem_and_path(tmp_path):
    run_dir = _make_run_dir(tmp_path, _config())
    assert load_run(run_dir, checkpoint="step_3").checkpoint.epoch == 3
    explicit = run_dir / "checkpoints" / "step_3.pt"
    assert load_run(run_dir, checkpoint=explicit).checkpoint.epoch == 3


def test_load_run_parses_metrics_and_manifest(tmp_path):
    run = load_run(_make_run_dir(tmp_path, _config()))
    assert [m["step"] for m in run.metrics] == [3, 12]
    assert run.metrics[-1]["test_acc"] == 0.9
    assert run.manifest["status"] == "completed"


def test_load_run_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run dir"):
        load_run(tmp_path / "no-such-run")


def test_load_run_unknown_checkpoint_name_raises(tmp_path):
    run_dir = _make_run_dir(tmp_path, _config())
    with pytest.raises(FileNotFoundError, match="step_999"):
        load_run(run_dir, checkpoint="step_999")


def test_list_checkpoints_grokked_wins_step_tie(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ["step_7.pt", "grokked_step_7.pt", "step_3.pt"]:
        (ckpt_dir / name).touch()
    names = [p.name for p in list_checkpoints(tmp_path)]
    assert names == ["step_3.pt", "step_7.pt", "grokked_step_7.pt"]


def test_list_checkpoints_ignores_non_checkpoint_pt_files(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ["step_1.pt", "model_final.pt", "best.pt"]:
        (ckpt_dir / name).touch()
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_1.pt"]


def test_load_run_relative_path_resolves_against_run_dir(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, _config())
    monkeypatch.chdir(tmp_path)  # ensure CWD-relative resolution would fail
    run = load_run(run_dir, checkpoint=Path("step_3.pt"))
    assert run.checkpoint.epoch == 3


_REAL_RUN = Path(__file__).resolve().parents[1] / "runs/2026-06-04/2026-06-04_050749_grok-C113"


@pytest.mark.skipif(not _REAL_RUN.is_dir(), reason="local-only: runs/ is gitignored")
def test_real_grokked_c113_run_loads_and_generalises():
    run = load_run(_REAL_RUN)
    ckpt = run.checkpoint
    assert ckpt.group.order == 113
    assert ckpt.epoch == 29158  # the run's latest (event) snapshot — proves default selection

    # Rebuild the test split exactly as the trainer did (same seed fallback),
    # run the loaded model on it, and demand grokked-level accuracy: proof the
    # loaded weights are the trained weights, not just the right shapes.
    cfg = ckpt.config
    task = build_group_task(ckpt.group)
    split = train_test_split(task, cfg.data.train_frac, cfg.data.split_seed or cfg.experiment.seed)
    eq_col = np.full((len(split.test_inputs), 1), ckpt.group.order)
    tokens = torch.tensor(np.concatenate([split.test_inputs, eq_col], axis=1), dtype=torch.long)
    targets = torch.tensor(split.test_targets, dtype=torch.long)
    with torch.no_grad():
        readout = ckpt.model(tokens)[:, -1, :]
    acc = (readout.argmax(dim=-1) == targets).float().mean().item()
    assert acc >= 0.99  # final logged test_acc was 0.9998
