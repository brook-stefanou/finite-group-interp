import numpy as np

from same_character_table_interp.analysis.figures import (
    plot_accuracy_curve,
    plot_energy_spectrum,
    plot_energy_trajectory,
    plot_energy_vs_ablation,
    plot_loss_curve,
    plot_metric_by_group,
)
from same_character_table_interp.analysis.irrep_metrics import (
    AblationResult,
    EnergySpectrum,
    EnergyTrajectory,
)


def _spectrum(n_blocks: int = 5) -> EnergySpectrum:
    rng = np.random.default_rng(0)
    f = rng.random(n_blocks)
    dims = np.array([1, 2, 2, 2, 1] if n_blocks == 5 else [2] * n_blocks)
    return EnergySpectrum(fractions=f / f.sum(), baseline=dims / (dims.sum()), block_dims=dims)


def _written(path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _metrics(n: int = 20) -> list[dict]:
    """Generate n rows of synthetic metrics."""
    return [
        {
            "step": s,
            "train_loss": max(0.01, 2.0 / (s + 1)),
            "train_acc": min(1.0, s / (n * 0.5)),
            "test_loss": max(0.01, 3.0 / (s + 1)),
            "test_acc": min(1.0, s / n),
        }
        for s in range(0, n * 10, 10)
    ]


# ---------------------------------------------------------------------------
# Smoke tests for figure functions
# ---------------------------------------------------------------------------


def test_plot_accuracy_curve(tmp_path):
    out = tmp_path / "acc.png"
    plot_accuracy_curve(_metrics(), out)
    assert _written(out)


def test_plot_loss_curve(tmp_path):
    out = tmp_path / "loss.png"
    plot_loss_curve(_metrics(50), out)
    assert _written(out)


def test_plot_energy_spectrum_no_highlight(tmp_path):
    out = tmp_path / "spec.png"
    plot_energy_spectrum(_spectrum(), out, title="W_E energy")
    assert _written(out)


def test_plot_energy_spectrum_with_highlight(tmp_path):
    out = tmp_path / "spec_hl.png"
    plot_energy_spectrum(_spectrum(), out, title="W_E energy", highlight=[0, 2])
    assert _written(out)


def test_plot_energy_vs_ablation(tmp_path):
    spectrum = _spectrum()
    ablation = [AblationResult(i, float(i) * 0.1, -0.01 * i) for i in range(5)]
    out = tmp_path / "ea.png"
    plot_energy_vs_ablation(spectrum, ablation, out, title="W_E ablation", highlight=[1, 3])
    assert _written(out)


def test_plot_energy_vs_ablation_57_blocks(tmp_path):
    rng = np.random.default_rng(2)
    f = rng.random(57)
    dims = np.full(57, 2)
    spectrum = EnergySpectrum(fractions=f / f.sum(), baseline=dims / 113, block_dims=dims)
    ablation = [AblationResult(i, rng.normal() * 0.1, 0.0) for i in range(57)]
    out = tmp_path / "ea57.png"
    plot_energy_vs_ablation(
        spectrum, ablation, out, title="57-block ablation", highlight=[0, 5, 10]
    )
    assert _written(out)


def test_plot_energy_trajectory_with_keep(tmp_path):
    rng = np.random.default_rng(1)
    f = rng.random((10, 5))
    traj = EnergyTrajectory(
        epochs=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        fractions=f / f.sum(axis=1, keepdims=True),
    )
    out = tmp_path / "traj.png"
    plot_energy_trajectory(traj, out, keep=[1, 3])
    assert _written(out)


def test_plot_energy_trajectory_empty_keep(tmp_path):
    rng = np.random.default_rng(3)
    f = rng.random((5, 4))
    traj = EnergyTrajectory(
        epochs=[0, 10, 20, 30, 40],
        fractions=f / f.sum(axis=1, keepdims=True),
    )
    out = tmp_path / "traj_empty.png"
    plot_energy_trajectory(traj, out, keep=[])
    assert _written(out)


def test_plot_metric_by_group(tmp_path):
    out = tmp_path / "by_group.png"
    # Unequal group sizes (a non-grokked seed dropped from one group) + an hline.
    plot_metric_by_group(
        {"Dih(104)": [0.12, 0.19, 0.01, 0.07], "Dic(104)": [0.05, 0.0, 0.03]},
        out,
        title="gap per seed",
        ylabel="FVE gap",
        hline=0.0,
        hline_label="no signal",
    )
    assert _written(out)


def test_plot_metric_by_group_single_value(tmp_path):
    # n=1 group must not crash on the jitter/std path.
    out = tmp_path / "by_group_single.png"
    plot_metric_by_group({"A": [0.5], "B": [0.1, 0.2]}, out, title="t", ylabel="y")
    assert _written(out)
