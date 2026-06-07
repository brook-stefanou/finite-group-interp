"""Publication figures for run analysis. Data in -> PNG out; no metric math.

Uses the Agg backend (file rendering only, no display) so figures generate
identically headless, in tests, and in CI.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from finite_group_interp.analysis.irrep_metrics import (  # noqa: E402
    AblationResult,
    EnergySpectrum,
    EnergyTrajectory,
)

_DPI = 300

# Okabe-Ito colorblind-safe palette for significant blocks; grey for the rest.
_HIGHLIGHT_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
_NEUTRAL = "#B8B8B8"
_BASELINE_COLOR = "#333333"
_ACCENT = "#0072B2"


def _block_color(block: int, highlight: list[int]) -> str:
    """Bar-chart color: significant blocks get a single accent blue; others neutral grey.

    For trajectories, use _HIGHLIGHT_COLORS directly by position instead.
    """
    if block in highlight:
        return _ACCENT
    return _NEUTRAL


def _style(ax: matplotlib.axes.Axes) -> None:
    """Apply shared style to an Axes: remove top/right spines, subtle y-grid,
    uniform tick/label sizes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.title.set_fontsize(12)


def plot_accuracy_curve(metrics: list[dict[str, Any]], out: Path) -> None:
    """Train/test accuracy vs epoch (symlog x), single panel."""
    rows = [m for m in metrics if "test_acc" in m]
    steps = [m["step"] for m in rows]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(steps, [m["train_acc"] for m in rows], color="#777777", label="train")
    ax.plot(steps, [m["test_acc"] for m in rows], color="#0072B2", label="test")
    ax.set_xscale("symlog")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.legend(frameon=False, fontsize=11)
    _style(ax)
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_loss_curve(metrics: list[dict[str, Any]], out: Path) -> None:
    """Train/test loss vs epoch (linear x, log y), raw lines."""
    rows = [m for m in metrics if "test_loss" in m]
    steps = [m["step"] for m in rows]
    train_raw = [m["train_loss"] for m in rows]
    test_raw = [m["test_loss"] for m in rows]

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(steps, train_raw, color="#777777", linewidth=1.0, label="train")
    ax.plot(steps, test_raw, color="#0072B2", linewidth=1.0, label="test")
    # Linear x-axis: slingshot cycles resolve into well-separated periodic events
    # instead of stacking in the compressed log tail; accuracy figure keeps log-x
    # for early dynamics.
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(frameon=False, fontsize=11)
    _style(ax)
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def _draw_spectrum_panel(
    ax: matplotlib.axes.Axes,
    spectrum: EnergySpectrum,
    highlight: list[int],
    annotate: bool = True,
) -> None:
    """Shared drawing logic for a single energy spectrum panel."""
    x = np.arange(len(spectrum.fractions))
    colors = [_block_color(i, highlight) for i in range(len(spectrum.fractions))]
    ax.bar(x, spectrum.fractions, color=colors)
    ax.step(
        x,
        spectrum.baseline,
        where="mid",
        color=_BASELINE_COLOR,
        linestyle="--",
        label="random baseline",
    )
    ax.set_xlabel("isotypic block")
    ax.set_ylabel("fraction of energy")
    if annotate:
        for i in highlight:
            if 0 <= i < len(spectrum.fractions):
                ax.annotate(
                    str(i),
                    (i, spectrum.fractions[i]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
    _style(ax)


def plot_energy_spectrum(
    spectrum: EnergySpectrum,
    out: Path,
    title: str,
    highlight: list[int] | None = None,
) -> None:
    """Per-block energy fractions as bars, random-matrix baseline as a dashed line."""
    hl = highlight if highlight else []
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    _draw_spectrum_panel(ax, spectrum, hl, annotate=True)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=11)
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_energy_vs_ablation(
    spectrum: EnergySpectrum,
    ablation: list[AblationResult],
    out: Path,
    title: str,
    highlight: list[int] | None = None,
) -> None:
    """Two stacked panels: top = energy fractions, bottom = delta_loss, block-index order."""
    hl = highlight if highlight else []
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")

    # Top: energy spectrum
    _draw_spectrum_panel(ax_top, spectrum, hl, annotate=True)
    ax_top.set_xlabel("")  # shared x — label only on bottom
    ax_top.set_title(title)
    ax_top.legend(frameon=False, fontsize=11)

    # Bottom: delta loss
    x = np.arange(len(ablation))
    colors = [_block_color(r.block_index, hl) for r in ablation]
    ax_bot.bar(x, [r.delta_loss for r in ablation], color=colors)
    ax_bot.axhline(0, color=_BASELINE_COLOR, linewidth=0.8)
    ax_bot.set_xlabel("isotypic block")
    ax_bot.set_ylabel("Δ test loss")
    _style(ax_bot)

    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_energy_trajectory(traj: EnergyTrajectory, out: Path, keep: list[int]) -> None:
    """Energy of the kept blocks across training; the rest pooled as dashed grey.

    Lines are annotated at their right end for readability.
    """
    all_blocks = np.arange(traj.fractions.shape[1])
    rest = [i for i in all_blocks if i not in keep]
    epochs = traj.epochs

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

    for pos, i in enumerate(keep):
        color = _HIGHLIGHT_COLORS[pos % len(_HIGHLIGHT_COLORS)]
        fractions_i = traj.fractions[:, i]
        ax.plot(epochs, fractions_i, color=color, label=f"block {i}")
        if epochs:
            ax.annotate(
                f"block {i}",
                (epochs[-1], fractions_i[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=9,
                va="center",
            )

    if rest:
        pooled = traj.fractions[:, rest].sum(axis=1)
        ax.plot(
            epochs,
            pooled,
            color=_NEUTRAL,
            linestyle="--",
            label=f"other {len(rest)} blocks",
        )

    ax.set_xscale("symlog")
    ax.set_xlabel("epoch")
    ax.set_ylabel("energy fraction")
    ax.set_title("Isotypic energy across training (W_E)")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    _style(ax)
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
