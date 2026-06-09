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
"""

import json
import sys
from pathlib import Path

import numpy as np

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
            # group token from the run name, e.g. "pair-D52-s0-wd1.0-f0.4" -> "D52"
            parts = d.name.split("-")
            token = parts[1] if len(parts) > 1 and parts[0] == "pair" else d.name
            grokked.setdefault(token, []).append((d, acc))
    return grokked


def matrix_report(run_dir: Path) -> None:
    """Energy + embedding rank per high-energy block, and the FVE gap."""
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


def main(argv: list[str]) -> None:
    roots = argv if argv else ["runs"]
    run_dirs = find_run_dirs(roots)
    if not run_dirs:
        print(f"no runs found under {roots}")
        return

    grokked = print_learnability(run_dirs)
    if not grokked:
        print("\n(no grokked runs yet -- matrix-level report skipped)")
        return

    print("\n" + "=" * 83)
    print("MATRIX-LEVEL CONTRAST (best grokked run per group)")
    print("=" * 83)
    for token in sorted(grokked):
        best_dir, _ = max(grokked[token], key=lambda t: t[1])
        matrix_report(best_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
