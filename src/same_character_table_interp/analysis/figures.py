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
import seaborn as sns  # noqa: E402

from same_character_table_interp.analysis.irrep_metrics import (  # noqa: E402
    AblationResult,
    EnergySpectrum,
    EnergyTrajectory,
)
from same_character_table_interp.analysis.functional_form import FunctionalFormResult  # noqa: E402

_DPI = 300

# Shared authored width (inches). LessWrong rescales every image to one column
# width, so apparent text/line size is set by font_pt / width_in, not pixels.
# Holding the width constant across figures keeps that ratio constant, so labels
# render at the same size on the page. Heights vary with content.
_FIG_W = 7.0

# Okabe-Ito colorblind-safe palette for significant blocks; grey for the rest.
_HIGHLIGHT_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
_NEUTRAL = "#B8B8B8"
_BASELINE_COLOR = "#333333"
_ACCENT = "#0072B2"

# Refined-minimal ink/grid colours and a robust font stack (falls back to
# DejaVu on machines/CI without Helvetica).
_INK = "#2B2B2B"
_MUTE = "#6A6A6A"
_GRID = "#EAEAEA"
_SPINE = "#9A9A9A"
_FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _darken(hex_color: str, f: float = 0.72) -> str:
    """Scale an #rrggbb colour toward black by factor f (for deep mean markers)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def _apply_pub_theme() -> None:
    """seaborn publication theme: clean typography, despined axes, transparent
    background so figures sit on any page colour.

    Colour is still assigned per-artist from the Okabe-Ito palette above (accent
    + grey carries the meaning), so we deliberately do NOT let seaborn pick a
    categorical palette. This only upgrades fonts, spacing, spines, and saving.
    """
    sns.set_theme(
        context="notebook",
        style="ticks",
        font_scale=1.0,
        rc={
            "font.family": _FONT,
            "savefig.dpi": _DPI,
            "savefig.transparent": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": _SPINE,
            "axes.linewidth": 0.9,
            "axes.titlelocation": "left",
            "axes.titleweight": "regular",
            "axes.titlecolor": _INK,
            "axes.titlesize": 13,
            "axes.titlepad": 12,
            "axes.labelcolor": _INK,
            "text.color": _INK,
            "xtick.color": _INK,
            "ytick.color": _INK,
            "axes.grid": False,
            "svg.fonttype": "none",
        },
    )


_apply_pub_theme()


def _block_color(block: int, highlight: list[int]) -> str:
    """Bar-chart color: significant blocks get a single accent blue; others neutral grey.

    For trajectories, use _HIGHLIGHT_COLORS directly by position instead.
    """
    if block in highlight:
        return _ACCENT
    return _NEUTRAL


def _style(ax: matplotlib.axes.Axes) -> None:
    """Apply shared style to an Axes: remove top/right spines, light y-grid,
    tickless axes, uniform label sizes. Title size/weight/colour come from the
    theme rc so every figure left-aligns consistently."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=11)
    ax.xaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_fontsize(11.5)


def plot_accuracy_curve(metrics: list[dict[str, Any]], out: Path) -> None:
    """Train/test accuracy vs epoch (symlog x), single panel."""
    rows = [m for m in metrics if "test_acc" in m]
    steps = [m["step"] for m in rows]
    fig, ax = plt.subplots(figsize=(_FIG_W, 4.0), layout="constrained")
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
    # Symlog x to match the accuracy figure and the standard grokking
    # presentation: it compresses the long pre-grok plateau and makes the
    # memorisation -> generalisation transition legible. (Trade-off: the late
    # slingshot cycles bunch up in the compressed tail; the transition is the
    # point, so that's acceptable.)
    ax.set_xscale("symlog")
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
    fig, ax = plt.subplots(figsize=(_FIG_W, 3.4), layout="constrained")
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
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(_FIG_W, 5.0), sharex=True, layout="constrained"
    )

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


def plot_metric_by_group(
    groups: dict[str, list[float]],
    out: Path,
    *,
    title: str,
    ylabel: str,
    yscale: str = "linear",
    hline: float | None = None,
    hline_label: str | None = None,
) -> None:
    """Per-seed values for each group as a jittered strip + mean±std marker.

    ``groups`` maps a group label -> its per-seed values (varying length is
    fine; non-grokked seeds are simply omitted by the caller). One faint marker
    per seed, a heavy marker at the mean with a std error bar. This is the
    matched-pair summary view: the SPREAD across seeds is the point, so the
    figure must show every seed, not just an aggregate. ``hline`` draws a
    reference line (e.g. 0 for "no signal").
    """
    labels = list(groups)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.8 * len(labels) + 1.4), 4.7), layout="constrained")
    if hline is not None:
        ax.axhline(hline, color="#C2C2C2", linestyle=(0, (4, 3)), linewidth=1.0, label=hline_label)
    for pos, label in enumerate(labels):
        vals = np.asarray(groups[label], dtype=float)
        light = _HIGHLIGHT_COLORS[pos % len(_HIGHLIGHT_COLORS)]
        deep = _darken(light)
        n = len(vals)
        # Jittered strip (white-edged) on the left, deep mean marker offset to the
        # right so the summary never sits on the points. Deterministic jitter =>
        # the figure reproduces exactly (no RNG).
        jitter = np.linspace(-0.085, 0.085, n) if n > 1 else np.zeros(1)
        ax.scatter(
            pos - 0.13 + jitter,
            vals,
            color=light,
            alpha=0.55,
            s=46,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        if n:
            mean = float(vals.mean())
            # Sample std (ddof=1) to match the report's Welch summary; 0 for n==1.
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            ax.errorbar(
                pos + 0.17,
                mean,
                yerr=std,
                fmt="o",
                color=deep,
                markersize=7.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=4,
                elinewidth=1.4,
                zorder=3,
            )
            label_txt = f"{mean:.3f} ± {std:.3f}" if n > 1 else f"{mean:.3f}"
            ax.annotate(
                label_txt,
                (pos + 0.17, mean),
                xytext=(13, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=_MUTE,
            )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{lbl}\nn = {len(groups[lbl])}" for lbl in labels])
    ax.set_xlim(-0.6, len(labels) - 1 + 0.95)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    if hline_label is not None:
        ax.legend(frameon=False, fontsize=10)
    _style(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)


def plot_paired_difference(
    diffs: list[float],
    out: Path,
    *,
    title: str,
    ylabel: str,
    xlabel: str,
    n_above: int | None = None,
) -> None:
    """Per-matched-seed paired difference (group A minus group B) as a jittered
    strip around a zero reference line, with a mean±std marker.

    The two-strip ``plot_metric_by_group`` view shows each group's marginal
    spread but hides the WITHIN-seed pairing. This view makes the pairing the
    point: how many matched seeds favour A over B, and by how much. ``n_above``
    annotates the count above zero (the sign-test support)."""
    vals = np.asarray(diffs, dtype=float)
    n = len(vals)
    fig, ax = plt.subplots(figsize=(5.2, 4.7), layout="constrained")
    ax.axhline(0.0, color="#C2C2C2", linestyle=(0, (4, 3)), linewidth=1.0, label="no difference")
    light = _HIGHLIGHT_COLORS[0]
    deep = _darken(light)
    # Deterministic jitter (sorted) => reproducible figure, no RNG.
    jitter = np.linspace(-0.085, 0.085, n) if n > 1 else np.zeros(1)
    ax.scatter(
        -0.13 + jitter,
        vals,
        color=light,
        alpha=0.55,
        s=46,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    if n:
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        ax.errorbar(
            0.17,
            mean,
            yerr=std,
            fmt="o",
            color=deep,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=4,
            elinewidth=1.4,
            zorder=3,
        )
        ax.annotate(
            f"{mean:+.3f} ± {std:.3f}",
            (0.17, mean),
            xytext=(13, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=_MUTE,
        )
    if n_above is not None:
        ax.annotate(
            f"above zero: {n_above}/{n}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=10,
            color=_MUTE,
        )
    ax.set_xticks([0])
    ax.set_xticklabels([f"{xlabel}\nn = {n}"])
    ax.set_xlim(-0.6, 0.95)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=10)
    _style(ax)
    ax.grid(axis="x", visible=False)
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)


def plot_functional_form_fve(
    result: FunctionalFormResult, out: Path, title: str = "Functional-form R²"
) -> None:
    """Per-kept-irrep full-matrix vs trace-only R², with cumulative annotated.

    Equal-height bar pairs (full == trace) are the 1-dim signature: no
    sub-character structure, so the group cannot adjudicate the debate.
    """
    keep = sorted(result.per_irrep_full)
    full = [result.per_irrep_full[j] for j in keep]
    trace = [result.per_irrep_trace[j] for j in keep]
    x = np.arange(len(keep))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(keep) + 2), 4.0))
    ax.bar(x - width / 2, full, width, label="full matrix", color=_ACCENT)
    ax.bar(x + width / 2, trace, width, label="trace only", color=_NEUTRAL)
    ax.set_xticks(x)
    ax.set_xticklabels([f"irrep {j}" for j in keep])
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"{title}\ncumulative full {result.cumulative_full:.3f}  |  "
        f"trace {result.cumulative_trace:.3f}  |  gap {result.gap:.3f}"
    )
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=_DPI)
    plt.close(fig)
