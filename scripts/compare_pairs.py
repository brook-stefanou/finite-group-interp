"""Post-sweep comparison for the same-character-table pairs.

Two reports:

  1. LEARNABILITY -- for every run under the given dirs: grok epoch (first
     test_acc >= 0.99), final test_acc, epochs reached. The coarsest evidence
     tier: does it grok, and how fast.

  2. MATRIX-LEVEL -- for the best grokked run of each group, the structure the
     character table can't see: per high-energy isotypic block, the W_E energy,
     the block dimension, and the *rank* of the embedding inside that block;
     plus the functional-form FVE gap (full-matrix minus trace-only) from the 2b
     instrument -- which is identically 0 only for 1-dim irreps and is the
     sub-character quantity the whole debate turns on.

     Read these by COMPARING the groups side by side (Dih vs Dic share a
     character table, so any divergence here is sub-character signal). Do NOT
     assume a specific predicted rank: empirically Dih(104) already uses the full
     4-dim block (rank 4), so the naive "real=2 vs quaternionic=4" heuristic does
     not hold -- the real evidence is how block_dim / rank / energy / FVE-gap
     differ between the two grokked models.

Usage:
    uv run python scripts/compare_pairs.py                 # everything under runs/
    uv run python scripts/compare_pairs.py runs/2026-06-09 # one date dir
    uv run python scripts/compare_pairs.py runs/.../<run>  # explicit run dirs
    uv run python scripts/compare_pairs.py --coset         # also run the coset tier (slower)
"""

import json
import sys
from pathlib import Path

import numpy as np

from finite_group_interp.analysis.coset_metrics import (
    _all_pairs_resid,
    ablate_coset_direction,
    coset_probe_suite,
)
from finite_group_interp.analysis.functional_form import functional_form_fit
from finite_group_interp.analysis.irrep_metrics import isotypic_energy, weight_as_functions
from finite_group_interp.analysis.loading import load_run
from finite_group_interp.representations.irreps import extract_irreps
from finite_group_interp.representations.projectors import real_isotypic_blocks

GROK = 0.99


def find_run_dirs(roots: list[str]) -> list[Path]:
    """Run dirs are those containing a metrics.jsonl (handles dirs OR run dirs)."""
    dirs: set[Path] = set()
    for root in roots:
        p = Path(root)
        if (p / "metrics.jsonl").exists():
            dirs.add(p)
        dirs.update(m.parent for m in p.glob("**/metrics.jsonl"))
    return sorted(dirs)


def grok_summary(metrics_path: Path) -> tuple[int | None, float | None, int | None]:
    """(first step with test_acc >= GROK or None, final test_acc, last step)."""
    grok: int | None = None
    final_acc: float | None = None
    last: int | None = None
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if "test_acc" not in rec:
            continue
        final_acc = float(rec["test_acc"])
        last = int(rec.get("step", -1))
        if grok is None and final_acc >= GROK:
            grok = last
    return grok, final_acc, last


def print_learnability(run_dirs: list[Path]) -> dict[str, list[tuple[Path, float]]]:
    """Print the table; return {group_token: [(run_dir, final_acc), ...]} for grokked runs."""
    print(f"\n{'run':54}{'grok@':>9}{'final_acc':>11}{'epochs':>9}")
    print("-" * 83)
    grokked: dict[str, list[tuple[Path, float]]] = {}
    for d in run_dirs:
        grok, acc, last = grok_summary(d / "metrics.jsonl")
        acc_s = "-" if acc is None else f"{acc:.4f}"
        print(f"{d.name:54}{str(grok):>9}{acc_s:>11}{str(last):>9}")
        if grok is not None and acc is not None:
            group, _, _ = _parse(d.name)
            grokked.setdefault(group or d.name, []).append((d, acc))
    return grokked


def matrix_report(run_dir: Path, with_coset: bool = False) -> None:
    """Energy + embedding rank per high-energy block, the FVE gap, and (with
    --coset) coset probes on the proper normal subgroups."""
    run = load_run(run_dir)
    model, group = run.checkpoint.model, run.checkpoint.group
    n = group.order
    w_e = weight_as_functions(model, "W_E", n)
    blocks = real_isotypic_blocks(group)
    spec = isotypic_energy(w_e, blocks)
    keep = [i for i in range(len(blocks)) if spec.fractions[i] > 2 * spec.baseline[i]]

    print(f"\n=== {run_dir.name}  (order {n}, checkpoint epoch {run.checkpoint.epoch}) ===")
    print(f"  kept blocks (W_E energy > 2x baseline): {keep}")
    for i in keep:
        b = blocks[i]
        pw = b.projector @ w_e
        sv = np.linalg.svd(pw, compute_uv=False)
        rank = int(np.sum(sv > 1e-2 * sv[0])) if sv[0] > 0 else 0
        block_dim = int(round(float(np.trace(b.projector).real)))
        print(
            f"    block {i:2d}: irrep_dim={b.dimension} block_dim={block_dim} "
            f"energy={spec.fractions[i]:.3f} W_E_rank={rank} "
            f"svals={np.round(sv[:6], 3).tolist()}"
        )

    keep_rows = sorted({idx for i in keep for idx in blocks[i].irrep_indices})
    irreps = extract_irreps(group)
    ff = functional_form_fit(model, group, irreps, keep_rows)
    print(
        f"  functional-form FVE: full={ff.cumulative_full:.3f} "
        f"trace={ff.cumulative_trace:.3f}  GAP={ff.gap:.3f}"
    )
    if not with_coset:
        return
    # Coset side (opt-in: heavier). Probe coset membership of the proper NORMAL
    # subgroups (the quotient set -- few, unlike all subgroups), target ab.
    # excess_over_irrep is the decisive number (coset signal beyond the model's
    # own irreps); abl_cross_coset_delta is the causal check.
    resid, targets = _all_pairs_resid(model, group)
    normals = [h for h in group.subgroups() if 1 < len(h) < n and group.is_normal(h)]
    print(f"  coset probes (target=ab) on {len(normals)} proper normal subgroups:")
    for h in normals:
        pa, nm, _ns, ir, k = coset_probe_suite(
            resid, group, h, "ab", keep_rows, seed=0, null_draws=3
        )
        abl = ablate_coset_direction(model, group, h, "ab", resid, targets, seed=0)
        cross = abl["ablation_cross_coset_delta"]
        ctrl = abl["random_cross_coset_delta"]
        print(
            f"    |H|={len(h):3d} k={k:3d}: probe={pa:.3f} "
            f"excess_null={pa - nm:+.3f} excess_irrep={pa - ir:+.3f} | "
            f"abl_cross={cross:+.3f} ctrl={ctrl:+.3f} excess={cross - ctrl:+.3f}"
        )


def _parse(name: str) -> tuple[str, str, str]:
    """(group, seed, wd) from a run-dir name; ('', '', '') if not a pair run.

    Names carry a date prefix, e.g. '2026-06-08_190917_pair-D52-s0-wd1.0-f0.4',
    so parse from the 'pair-' marker, not a blind split (the date has hyphens).
    """
    if "pair-" not in name:
        return ("", "", "")
    p = name.split("pair-", 1)[1].split("-")  # ['D52', 's0', 'wd1.0', 'f0.4']
    return (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")


def main(argv: list[str]) -> None:
    with_coset = "--coset" in argv  # opt-in: adds the (heavier) coset-probe tier
    roots = [a for a in argv if a != "--coset"] or ["runs"]
    run_dirs = find_run_dirs(roots)
    if not run_dirs:
        print(f"no runs found under {roots}")
        return

    grokked = print_learnability(run_dirs)
    if not grokked:
        print("\n(no grokked runs yet -- matrix/coset report skipped)")
        return

    # Group grokked runs by (seed, wd) so groups are compared at MATCHED settings
    # -- the only honest way to attribute a difference to the group, not the hp.
    matched: dict[tuple[str, str], dict[str, Path]] = {}
    for token, runs in grokked.items():
        for run_dir, _acc in runs:
            _g, seed, wd = _parse(run_dir.name)
            matched.setdefault((seed, wd), {})[token] = run_dir

    comparable = {
        sw: groups
        for sw, groups in matched.items()
        if len(groups) >= 2 and sw != ("", "")  # skip non-pair runs (no seed/wd tag)
    }
    print("\n" + "=" * 83)
    if comparable:
        print("MATRIX-LEVEL CONTRAST (matched seed x weight-decay; >=2 groups grokked)")
        print("=" * 83)
        for (seed, wd), groups in sorted(comparable.items()):
            print(f"\n########## matched setting: {seed} {wd} ##########")
            for token in sorted(groups):
                matrix_report(groups[token], with_coset)
    else:
        print("NO MATCHED (seed, wd) HAS >=2 GROUPS GROKKED -- falling back to best per group")
        print("(comparison is hyperparameter-confounded; interpret with care)")
        print("=" * 83)
        for token in sorted(grokked):
            best_dir, _ = max(grokked[token], key=lambda t: t[1])
            matrix_report(best_dir, with_coset)


if __name__ == "__main__":
    main(sys.argv[1:])
