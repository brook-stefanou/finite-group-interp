"""Orchestrate the full irrep analysis of one training run.

Reads a run directory, computes energy spectra / ablations / restricted loss
on the latest checkpoint plus the energy trajectory over every snapshot, and
writes runs/<id>/analysis/metrics.json + figures. The CLI shim is
scripts/analyze_run.py; this module holds the logic so tests can import it
(same split as training/cli.py and scripts/run.py).
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from same_character_table_interp.analysis.figures import (
    plot_accuracy_curve,
    plot_energy_spectrum,
    plot_energy_trajectory,
    plot_energy_vs_ablation,
    plot_functional_form_fve,
    plot_loss_curve,
)
from same_character_table_interp.analysis.functional_form import functional_form_fit
from same_character_table_interp.analysis.irrep_metrics import (
    EnergySpectrum,
    block_ablation,
    energy_trajectory,
    evaluate,
    isotypic_energy,
    restricted_loss,
    weight_as_functions,
)
from same_character_table_interp.analysis.loading import (
    LoadedCheckpoint,
    list_checkpoints,
    load_checkpoint,
    load_run,
    step_of,
)
from same_character_table_interp.model import GroupModel
from same_character_table_interp.training.manifest import get_git_commit
from same_character_table_interp.groups.group import FiniteGroup
from same_character_table_interp.representations.irreps import extract_irreps
from same_character_table_interp.representations.projectors import (
    IsotypicBlock,
    real_isotypic_blocks,
)
from same_character_table_interp.task import build_group_task, train_test_split

HEADLINE_FIGURES = (
    "accuracy.png",
    "loss.png",
    "spectrum-pre-grok-W_E.png",
    "energy-vs-ablation-W_E.png",
    "energy-trajectory.png",
)

_TOP_BLOCK_RULE = "blocks with W_E energy > 2x baseline"

_MATRICES: tuple[Literal["W_E"], Literal["W_U"]] = ("W_E", "W_U")


def _keep_blocks(spectrum: EnergySpectrum) -> list[int]:
    """Indices of isotypic blocks whose energy fraction exceeds 2x the random baseline."""
    return [i for i, f in enumerate(spectrum.fractions) if f > 2 * spectrum.baseline[i]]


def _keep_rows(keep: list[int], blocks: list[IsotypicBlock]) -> list[int]:
    """Character-table irrep rows spanned by the kept blocks."""
    return sorted({idx for i in keep for idx in blocks[i].irrep_indices})


def _test_split_tensors(ckpt: LoadedCheckpoint) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the run's exact test split (same seed fallback as the trainer)."""
    cfg = ckpt.config
    task = build_group_task(ckpt.group)
    seed = cfg.data.split_seed if cfg.data.split_seed is not None else cfg.experiment.seed
    split = train_test_split(task, cfg.data.train_frac, seed)
    eq_col = np.full((len(split.test_inputs), 1), ckpt.group.order)
    tokens = torch.tensor(np.concatenate([split.test_inputs, eq_col], axis=1), dtype=torch.long)
    targets = torch.tensor(split.test_targets, dtype=torch.long)
    return tokens, targets


def _memorization_epoch(metrics: list[dict[str, Any]]) -> int | None:
    """First logged epoch where the model has memorised (train acc >= 0.99)
    but not generalised (test acc < 0.5) -- the pre-grokking snapshot used to
    show energy is smeared before the transition. None if no such phase."""
    for m in metrics:
        if "train_acc" in m and m.get("train_acc", 0.0) >= 0.99 and m.get("test_acc", 1.0) < 0.5:
            return int(m["step"])
    return None


def _embedding_rank(block: IsotypicBlock, w_e: np.ndarray) -> int:
    """Rank of the embedding inside one block: singular values of
    ``block.projector @ W_E`` thresholded at 1e-2 * the largest (matches
    scripts/compare_pairs.py's matrix_report)."""
    sv = np.linalg.svd(block.projector @ w_e, compute_uv=False)
    return int(np.sum(sv > 1e-2 * sv[0])) if sv[0] > 0 else 0


def irrep_metrics(
    model: GroupModel,
    group: FiniteGroup,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Model-level irrep tier, computed without touching disk (no trajectory).

    Returns a compact summary of where the model's W_E energy concentrates, the
    causal cost of ablating those blocks, the positive-control restricted loss,
    and the functional-form FVE gap -- the same quantities ``analyze`` derives,
    minus the run-dir-dependent trajectory.
    """
    n = group.order
    blocks = real_isotypic_blocks(group)
    w_e = weight_as_functions(model, "W_E", n)
    spec = isotypic_energy(w_e, blocks)
    keep = _keep_blocks(spec)
    ablations = block_ablation(model, blocks, tokens, targets, matrix="W_E")
    restricted_l, restricted_a = restricted_loss(model, blocks, keep, tokens, targets)

    keep_rows = _keep_rows(keep, blocks)
    irreps = extract_irreps(group)
    ff = functional_form_fit(model, group, irreps, keep_rows)

    return {
        "energy_concentration": float(sum(spec.fractions[i] for i in keep)),
        "kept_blocks": [
            {
                "block": i,
                "irrep_dim": blocks[i].dimension,
                "block_dim": int(round(float(blocks[i].projector.trace().real))),
                "energy": float(spec.fractions[i]),
                "w_e_rank": _embedding_rank(blocks[i], w_e),
            }
            for i in keep
        ],
        "ablation_deltas": [
            {"block": r.block_index, "delta_loss": r.delta_loss, "delta_acc": r.delta_acc}
            for r in ablations
            if r.block_index in keep
        ],
        "restricted_loss": float(restricted_l),
        "restricted_acc": float(restricted_a),
        "functional_form": {
            "cumulative_full": ff.cumulative_full,
            "cumulative_trace": ff.cumulative_trace,
            "gap": ff.gap,
        },
    }


def analyze(run_dir: Path | str, publish: Path | None = None) -> dict[str, Any]:
    """Full analysis of one run. Returns the metrics dict it also writes."""
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    ckpt = run.checkpoint
    n = ckpt.group.order
    blocks = real_isotypic_blocks(ckpt.group)
    tokens, targets = _test_split_tensors(ckpt)

    base_loss, base_acc = evaluate(ckpt.model, tokens, targets)

    spectra = {m: isotypic_energy(weight_as_functions(ckpt.model, m, n), blocks) for m in _MATRICES}
    ablations = {
        m: block_ablation(ckpt.model, blocks, tokens, targets, matrix=m) for m in _MATRICES
    }
    keep = _keep_blocks(spectra["W_E"])
    keep_wu = _keep_blocks(spectra["W_U"])
    restricted_l, restricted_a = restricted_loss(ckpt.model, blocks, keep, tokens, targets)

    # Matrix-level functional-form fit on the energy-kept blocks. `keep` indexes
    # blocks; map to the character-table rows the irreps are keyed by.
    keep_rows = _keep_rows(keep, blocks)
    irreps = extract_irreps(ckpt.group)
    ff = functional_form_fit(ckpt.model, ckpt.group, irreps, keep_rows)

    trajectory = energy_trajectory(run_dir, blocks, matrix="W_E")

    # Early-checkpoint selection: memorization phase (pre-grokking snapshot)
    mem_epoch = _memorization_epoch(run.metrics)
    early_ckpt: LoadedCheckpoint | None = None
    early_checkpoint_name: str | None = None
    if mem_epoch is not None:
        ckpt_paths = list_checkpoints(run_dir)
        closest = min(ckpt_paths, key=lambda p: abs(step_of(p) - mem_epoch))
        early_ckpt = load_checkpoint(closest)
        early_checkpoint_name = closest.name

    metrics: dict[str, Any] = {
        "provenance": {
            "run_id": run_dir.name,
            "checkpoint": ckpt.path.name,
            "checkpoint_epoch": ckpt.epoch,
            "git_commit": run.manifest.get("git_commit"),  # training-time code
            "analysis_commit": get_git_commit(),  # analysis-time code
            "generated": datetime.now(timezone.utc).isoformat(),
            "memorization_epoch": mem_epoch,
            "memorization_checkpoint": early_checkpoint_name,
        },
        "group": {"spec": ckpt.config.data.group, "order": n, "n_blocks": len(blocks)},
        "base": {"test_loss": base_loss, "test_acc": base_acc},
        "energy": {
            m: {
                "fractions": s.fractions.tolist(),
                "baseline": s.baseline.tolist(),
                "block_dims": s.block_dims.tolist(),
            }
            for m, s in spectra.items()
        },
        "ablation": {
            m: [
                {"block": r.block_index, "delta_loss": r.delta_loss, "delta_acc": r.delta_acc}
                for r in rs
            ]
            for m, rs in ablations.items()
        },
        "restricted": {
            "rule": _TOP_BLOCK_RULE,
            "n_kept": len(keep),
            "keep": keep,
            "loss": restricted_l,
            "acc": restricted_a,
        },
        "functional_form": {
            "keep_blocks": keep,
            "keep_irrep_rows": keep_rows,
            "per_irrep_full": {str(k): v for k, v in ff.per_irrep_full.items()},
            "per_irrep_trace": {str(k): v for k, v in ff.per_irrep_trace.items()},
            "cumulative_full": ff.cumulative_full,
            "cumulative_trace": ff.cumulative_trace,
            "gap": ff.gap,
            "n_features_full": ff.n_features_full,
            "n_features_trace": ff.n_features_trace,
        },
        "trajectory": {
            "epochs": trajectory.epochs,
            "fractions": np.round(trajectory.fractions, 8).tolist(),
        },
    }

    out_dir = run_dir / "analysis"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Accuracy and loss curves
    plot_accuracy_curve(run.metrics, fig_dir / "accuracy.png")
    plot_loss_curve(run.metrics, fig_dir / "loss.png")

    # Energy spectra (single-panel) and combined energy-vs-ablation panels
    for m, s in spectra.items():
        hl = keep if m == "W_E" else keep_wu
        plot_energy_spectrum(
            s, fig_dir / f"energy-spectrum-{m}.png", title=f"{m} isotypic energy", highlight=hl
        )
        plot_energy_vs_ablation(
            s,
            ablations[m],
            fig_dir / f"energy-vs-ablation-{m}.png",
            title=f"{m} block ablation",
            highlight=hl,
        )

    # Energy trajectory
    plot_energy_trajectory(trajectory, fig_dir / "energy-trajectory.png", keep=keep)

    plot_functional_form_fve(
        ff,
        fig_dir / "functional-form-fve.png",
        title=f"{ckpt.config.data.group} functional-form FVE (full vs trace)",
    )

    # Pre-grok spectrum (memorization phase)
    has_pre_grok = False
    if early_ckpt is not None:
        early_W_E = weight_as_functions(early_ckpt.model, "W_E", n)
        early_spectrum = isotypic_energy(early_W_E, blocks)
        early_epoch = early_ckpt.epoch
        plot_energy_spectrum(
            early_spectrum,
            fig_dir / "spectrum-pre-grok-W_E.png",
            title=f"W_E isotypic energy at epoch {early_epoch} (memorised, pre-grok)",
            highlight=keep,
        )
        has_pre_grok = True

    if publish is not None:
        publish = Path(publish)
        publish.mkdir(parents=True, exist_ok=True)
        prefix = ckpt.config.data.group.lower()
        # Build publish list: substitute spectrum if no pre-grok phase
        publish_names = list(HEADLINE_FIGURES)
        if not has_pre_grok:
            publish_names = [
                "energy-spectrum-W_E.png" if n == "spectrum-pre-grok-W_E.png" else n
                for n in publish_names
            ]
        for name in publish_names:
            src = fig_dir / name
            if src.is_file():
                shutil.copyfile(src, publish / f"{prefix}-{name}")

    return metrics


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Irrep analysis of a training run.")
    parser.add_argument("run_dir", type=Path, help="runs/<date>/<run_id> directory")
    parser.add_argument(
        "--publish",
        type=Path,
        default=None,
        help="also copy the headline figures here (e.g. docs/figures)",
    )
    args = parser.parse_args(argv)
    metrics = analyze(args.run_dir, publish=args.publish)
    fractions = metrics["energy"]["W_E"]["fractions"]
    top = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)[:8]
    print(
        f"run: {metrics['provenance']['run_id']} (epoch {metrics['provenance']['checkpoint_epoch']})"
    )
    print(f"base test acc: {metrics['base']['test_acc']:.4f}")
    print(f"top W_E blocks by energy: {top}")
    print(f"restricted to {metrics['restricted']['keep']}: acc {metrics['restricted']['acc']:.4f}")
    print(f"memorization epoch: {metrics['provenance']['memorization_epoch']}")
    print(f"artifacts: {args.run_dir / 'analysis'}")
