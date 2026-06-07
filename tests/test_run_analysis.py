import json
from pathlib import Path

import numpy as np
import pytest
import torch

from finite_group_interp.analysis.run_analysis import HEADLINE_FIGURES, analyze
from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.training.config import ExperimentConfig, GrokkingConfig
from finite_group_interp.training.trainer import build_model


def _make_run_dir(tmp_path, metrics_rows=None):
    """Minimal C8 run dir in the trainer's layout, two checkpoints.

    metrics_rows: list of dicts (step, train_loss, train_acc, test_loss, test_acc).
    Defaults to a memorization-phase scenario: step 1 memorised, step 5 grokked.
    """
    config = GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=0, use_wandb=False),
        data={"group": "C8"},
    )
    group = resolve_group("C8")
    run_dir = tmp_path / "2026-06-07_000000_t"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for epoch in (1, 5):
        model = build_model(config, group)
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": config.model_dump(),
        }
        torch.save(payload, ckpt_dir / f"step_{epoch}.pt")

    if metrics_rows is None:
        # Default: step 1 = memorised (train 1.0, test 0.1); step 5 = grokked
        metrics_rows = [
            {"step": 1, "train_loss": 0.1, "train_acc": 1.0, "test_loss": 5.0, "test_acc": 0.1},
            {"step": 5, "train_loss": 0.05, "train_acc": 1.0, "test_loss": 0.1, "test_acc": 0.99},
        ]

    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics_rows) + "\n")
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "t", "status": "completed", "git_commit": "abc1234"})
    )
    return run_dir


def test_analyze_writes_metrics_and_figures(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    publish = tmp_path / "pub"
    result = analyze(run_dir, publish=publish)

    metrics_path = run_dir / "analysis" / "metrics.json"
    assert metrics_path.is_file()
    saved = json.loads(metrics_path.read_text())
    assert saved == result

    # provenance + structure
    assert saved["provenance"]["git_commit"] == "abc1234"
    assert saved["group"]["order"] == 8
    assert saved["provenance"]["checkpoint_epoch"] == 5  # latest by default

    # energy fractions sum to ~1 for both matrices
    for m in ("W_E", "W_U"):
        assert sum(saved["energy"][m]["fractions"]) == pytest.approx(1.0, abs=1e-4)
        assert len(saved["ablation"][m]) == len(saved["energy"][m]["fractions"])

    # provenance includes analysis-time commit
    assert saved["provenance"]["analysis_commit"] not in (None, "")

    # restricted keeps blocks above 2x baseline
    assert saved["restricted"]["rule"] == "blocks with W_E energy > 2x baseline"
    assert isinstance(saved["restricted"]["keep"], list)
    assert saved["restricted"]["n_kept"] == len(saved["restricted"]["keep"])

    # trajectory covers both checkpoints
    assert saved["trajectory"]["epochs"] == [1, 5]

    # memorization phase: step 1 has train_acc 1.0, test_acc 0.1 → memorisation detected
    assert saved["provenance"]["memorization_epoch"] == 1

    # figures written next to metrics
    fig_dir = run_dir / "analysis" / "figures"
    # pre-grok figure must exist (memorization detected)
    assert (fig_dir / "spectrum-pre-grok-W_E.png").is_file()

    # All HEADLINE_FIGURES must be present (pre-grok substitution NOT needed here)
    for name in HEADLINE_FIGURES:
        src = fig_dir / name
        assert src.is_file() and src.stat().st_size > 0, f"missing figure: {name}"
        assert (publish / f"c8-{name}").is_file(), f"not published: c8-{name}"


def test_analyze_memorization_epoch_in_provenance(tmp_path):
    """Explicit assertion on memorization_epoch and pre-grok figure."""
    run_dir = _make_run_dir(tmp_path)
    analyze(run_dir, publish=None)
    saved = json.loads((run_dir / "analysis" / "metrics.json").read_text())

    assert saved["provenance"]["memorization_epoch"] == 1
    assert saved["provenance"]["memorization_checkpoint"] is not None
    fig_dir = run_dir / "analysis" / "figures"
    assert (fig_dir / "spectrum-pre-grok-W_E.png").is_file()


def test_analyze_no_memorization_path(tmp_path):
    """When no memorization phase, memorization_epoch is None and spectrum substituted."""
    no_mem_rows = [
        {"step": 1, "train_loss": 1.0, "train_acc": 0.5, "test_loss": 1.0, "test_acc": 0.5},
        {"step": 5, "train_loss": 1.0, "train_acc": 0.5, "test_loss": 1.0, "test_acc": 0.5},
    ]
    run_dir = _make_run_dir(tmp_path, metrics_rows=no_mem_rows)
    publish = tmp_path / "pub"
    analyze(run_dir, publish=publish)
    saved = json.loads((run_dir / "analysis" / "metrics.json").read_text())

    assert saved["provenance"]["memorization_epoch"] is None
    assert saved["provenance"]["memorization_checkpoint"] is None

    # pre-grok figure should NOT exist
    fig_dir = run_dir / "analysis" / "figures"
    assert not (fig_dir / "spectrum-pre-grok-W_E.png").is_file()

    # The substituted spectrum figure should be published instead
    prefix = "c8"
    # spectrum-pre-grok substituted by energy-spectrum-W_E
    assert (publish / f"{prefix}-energy-spectrum-W_E.png").is_file()
    # Other headline figures still published
    for name in ("accuracy.png", "loss.png", "energy-vs-ablation-W_E.png", "energy-trajectory.png"):
        assert (publish / f"{prefix}-{name}").is_file(), f"not published: {prefix}-{name}"


def test_analyze_without_publish(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    analyze(run_dir, publish=None)
    assert (run_dir / "analysis" / "metrics.json").is_file()


_REAL_RUN = Path(__file__).resolve().parents[1] / "runs/2026-06-04/2026-06-04_050749_grok-C113"


@pytest.mark.skipif(not _REAL_RUN.is_dir(), reason="local-only: runs/ is gitignored")
def test_real_c113_run_shows_concentration(tmp_path):
    result = analyze(_REAL_RUN, publish=None)
    fractions = np.array(result["energy"]["W_E"]["fractions"])
    baseline = np.array(result["energy"]["W_E"]["baseline"])
    # preregistered: sparse concentration far above the random baseline
    assert fractions.max() > 5 * baseline[fractions.argmax()]
    # positive control: the model restricted to its top blocks still works
    assert result["restricted"]["acc"] >= 0.9
