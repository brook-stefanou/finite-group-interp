"""Cross-seed summary figures for the matched character-table pair.

The per-run figures (energy spectrum, ablation, trajectory, functional-form)
come from scripts/analyze_run.py. This script makes the figures that only exist
ACROSS seeds -- the matched-pair story from the Jun 11 log:

  1. grok-epoch comparison   -- learnability asymmetry (the robust signal)
  2. FVE-gap distribution    -- the irrep gap overlaps across seeds (honest)
  3. coset excess-over-irrep -- clusters at/below 0 (no coset signal beyond irreps)

It recomputes the gap and coset excess from the grokked checkpoints (so the
figure can't drift from stale numbers), keyed by matched (seed, wd). Only
wd 1.0 is the clean matched setting (Dic is grok-fragile at wd 0.5).

Usage:
    PYTHONPATH=src uv run python scripts/pair_figures.py            # runs/, wd1.0
    PYTHONPATH=src uv run python scripts/pair_figures.py runs/2026-06-10 --wd 1.0
    PYTHONPATH=src uv run python scripts/pair_figures.py --out docs/figures
"""

import argparse
import statistics
from pathlib import Path

from finite_group_interp.analysis.coset_metrics import (
    _all_pairs_resid,
    coset_probe_suite,
)
from finite_group_interp.analysis.figures import plot_metric_by_group
from finite_group_interp.analysis.functional_form import functional_form_fit
from finite_group_interp.analysis.fve_gap_stats import welch_ttest
from finite_group_interp.analysis.irrep_metrics import isotypic_energy, weight_as_functions
from finite_group_interp.analysis.loading import load_run
from finite_group_interp.representations.irreps import extract_irreps
from finite_group_interp.representations.projectors import real_isotypic_blocks

# compare_pairs is a sibling script (scripts/ is sys.path[0] when run directly).
from compare_pairs import _parse, find_run_dirs, grok_summary  # noqa: E402

# Display labels for the two pair tokens (sweep uses the catalog tokens).
_LABELS = {"D52": "Dih(104)", "Dic26": "Dic(104)"}


def _gap_and_keep_rows(run_dir: Path) -> tuple[float, list[int]]:
    """Functional-form FVE gap (full - trace) on the energy-kept irreps, plus
    the kept irrep rows (reused for the coset irrep-feature control)."""
    run = load_run(run_dir)
    model, group = run.checkpoint.model, run.checkpoint.group
    n = group.order
    blocks = real_isotypic_blocks(group)
    spec = isotypic_energy(weight_as_functions(model, "W_E", n), blocks)
    keep = [i for i in range(len(blocks)) if spec.fractions[i] > 2 * spec.baseline[i]]
    keep_rows = sorted({idx for i in keep for idx in blocks[i].irrep_indices})
    irreps = extract_irreps(group)
    ff = functional_form_fit(model, group, irreps, keep_rows)
    return ff.gap, keep_rows


def _coset_excess(run_dir: Path, keep_rows: list[int]) -> list[float]:
    """excess_over_irrep for target=ab over every proper normal subgroup."""
    run = load_run(run_dir)
    model, group = run.checkpoint.model, run.checkpoint.group
    n = group.order
    resid, _ = _all_pairs_resid(model, group)
    normals = [h for h in group.subgroups() if 1 < len(h) < n and group.is_normal(h)]
    out: list[float] = []
    for h in normals:
        pa, _nm, _ns, ir, _k = coset_probe_suite(
            resid, group, h, "ab", keep_rows, seed=0, null_draws=3
        )
        out.append(pa - ir)  # excess over the irrep-feature reference
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-seed matched-pair summary figures.")
    parser.add_argument("roots", nargs="*", default=["runs"], help="run dirs / date dirs")
    parser.add_argument("--wd", default="wd1.0", help="weight-decay tag to keep (default wd1.0)")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"), help="figure output dir")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(args.roots or ["runs"])
    # Keep only grokked pair runs at the matched weight decay.
    # grok_epoch is a per-group property (every grokked seed counts), so it is
    # pooled by group. The gap and coset contrasts are only honest at MATCHED
    # seeds -- the seeds BOTH groups grokked -- so they are keyed by seed and
    # intersected below, matching the report's Welch test (it would be unfair to
    # compare a Dih seed against a Dic seed that never grokked).
    grok_epochs: dict[str, list[float]] = {}
    gap_by_seed: dict[str, dict[str, float]] = {}
    excess_by_seed: dict[str, dict[str, list[float]]] = {}
    n_failed: dict[str, int] = {}
    for d in sorted(run_dirs):
        group, seed, wd = _parse(d.name)
        if group not in _LABELS or wd != args.wd:
            continue
        label = _LABELS[group]
        grok, _acc, _last = grok_summary(d / "metrics.jsonl")
        if grok is None:
            n_failed[label] = n_failed.get(label, 0) + 1
            continue
        grok_epochs.setdefault(label, []).append(float(grok))
        gap, keep_rows = _gap_and_keep_rows(d)
        gap_by_seed.setdefault(label, {})[seed] = gap
        excess_by_seed.setdefault(label, {})[seed] = _coset_excess(d, keep_rows)
        print(f"  {d.name}: grok@{grok} gap={gap:+.3f}")

    if not gap_by_seed:
        print(f"no grokked pair runs at {args.wd} under {args.roots}")
        return

    order = [lbl for lbl in _LABELS.values() if lbl in gap_by_seed]
    if n_failed:
        print("non-grok (excluded): " + ", ".join(f"{k} {v}" for k, v in n_failed.items()))

    # Matched seeds: grokked by EVERY group present (intersection of seed sets).
    matched = sorted(set.intersection(*(set(gap_by_seed[lbl]) for lbl in order)))
    print(
        f"matched seeds (both groups grokked, n={len(matched)}): {', '.join(matched) or '(none)'}"
    )
    gaps = {lbl: [gap_by_seed[lbl][s] for s in matched] for lbl in order}
    excess = {lbl: [v for s in matched for v in excess_by_seed[lbl][s]] for lbl in order}

    # Stats block for report 02 / README (so one heavy run yields figures + numbers).
    print("\n--- stats (for report 02 / README) ---")
    for lbl in order:
        total = len(grok_epochs[lbl]) + n_failed.get(lbl, 0)
        print(f"{lbl}: grokked {len(grok_epochs[lbl])}/{total} at {args.wd}")
    for lbl in order:
        g = gaps[lbl]
        s = statistics.stdev(g) if len(g) > 1 else 0.0
        print(
            f"  {lbl} R2-gap matched n={len(g)}: "
            f"mean={statistics.mean(g):.3f} std={s:.3f} range=[{min(g):.3f}, {max(g):.3f}]"
        )
    if len(order) == 2:
        w = welch_ttest(gaps[order[0]], gaps[order[1]])
        print(
            f"  Welch {order[0]} vs {order[1]}: "
            f"t={w['t']:.3f} dof={w['dof']:.1f} p={w['p_two_sided']:.4f}"
        )
    for lbl in order:
        e = excess[lbl]
        s = statistics.stdev(e) if len(e) > 1 else 0.0
        print(
            f"  {lbl} coset excess_over_irrep n={len(e)}: "
            f"mean={statistics.mean(e):+.3f} std={s:.3f} max={max(e):+.3f}"
        )

    plot_metric_by_group(
        {lbl: grok_epochs[lbl] for lbl in order},
        args.out / "pair-grok-epochs.png",
        title=f"Grok epoch, per grokked seed ({args.wd})",
        ylabel="grok epoch (test acc ≥ 0.99)",
    )
    plot_metric_by_group(
        gaps,
        args.out / "pair-fve-gap.png",
        title=f"Matrix-vs-trace R² gap, matched seeds ({args.wd})",
        ylabel="R² gap (full − trace)",
    )
    plot_metric_by_group(
        excess,
        args.out / "pair-coset-excess.png",
        title=f"Coset excess over irrep reference, matched seeds ({args.wd})",
        ylabel="probe acc − irrep-ref acc",
        hline=0.0,
        hline_label="no signal beyond irreps",
    )
    print(f"\nwrote 3 figures to {args.out}/  (pair-grok-epochs, pair-fve-gap, pair-coset-excess)")


if __name__ == "__main__":
    main()
