"""Driver: launch N independent shard processes (scripts/_pairfig_shard.py),
wait, then aggregate the per-run results into the cross-seed stats + figures
(identical output to scripts/pair_figures.py, just parallel).

This is the macOS-safe way to parallelise the per-run extraction. The obvious
approach -- an mp.Pool of workers -- deadlocks on macOS: the default 'spawn'
start method re-imports the module in each worker, which loads torch + Accelerate
BLAS, and the workers never come up (0% CPU, no children, parent blocked forever
in imap_unordered). Here each shard is a plain independent subprocess, so there
is no shared multiprocessing to hang on.

Usage:
    PYTHONPATH=src uv run python scripts/pairfig_run.py [--shards 9] [--wd wd1.0] [--out docs/figures] [roots...]
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from finite_group_interp.analysis.figures import (  # noqa: E402
    plot_metric_by_group,
    plot_paired_difference,
)
from finite_group_interp.analysis.fve_gap_stats import welch_ttest  # noqa: E402

_LABELS = {"D52": "Dih(104)", "Dic26": "Dic(104)"}


def _aggregate_and_plot(arch: str, results: list[tuple[Any, ...]], out: Path, wd: str) -> None:
    """Cross-seed aggregation, stats, and figures for one architecture, fed the
    precomputed per-run (label, seed, grok, gap, excess) tuples from the shards.
    Byte-for-byte the same stats/figures as scripts/pair_figures.py."""
    prefix = "fc-" if arch == "fc" else "pair-"
    arch_note = " [FC baseline]" if arch == "fc" else ""

    grok_epochs: dict[str, list[float]] = {}
    gap_by_seed: dict[str, dict[str, float]] = {}
    excess_by_seed: dict[str, dict[str, list[float]]] = {}
    n_failed: dict[str, int] = {}
    for label, seed, grok, gap, excess in results:
        if grok is None:
            n_failed[label] = n_failed.get(label, 0) + 1
            continue
        grok_epochs.setdefault(label, []).append(grok)
        gap_by_seed.setdefault(label, {})[seed] = gap
        excess_by_seed.setdefault(label, {})[seed] = excess

    if not gap_by_seed:
        print(f"[{arch}] no grokked pair runs at {wd}")
        return

    order = [lbl for lbl in _LABELS.values() if lbl in gap_by_seed]
    if n_failed:
        print(
            f"[{arch}] non-grok (excluded): " + ", ".join(f"{k} {v}" for k, v in n_failed.items())
        )

    matched = sorted(set.intersection(*(set(gap_by_seed[lbl]) for lbl in order)))
    print(f"[{arch}] matched seeds (both groups grokked, n={len(matched)})")
    gaps = {lbl: [gap_by_seed[lbl][s] for s in matched] for lbl in order}
    excess = {lbl: [v for s in matched for v in excess_by_seed[lbl][s]] for lbl in order}

    print(f"\n--- stats [{arch}] (for report 02 / README / write-up) ---")
    for lbl in order:
        total = len(grok_epochs[lbl]) + n_failed.get(lbl, 0)
        ge = grok_epochs[lbl]
        print(
            f"{lbl}: grokked {len(ge)}/{total} at {wd}  "
            f"grok-epoch mean={statistics.mean(ge):.0f} median={statistics.median(ge):.0f} "
            f"range=[{min(ge):.0f}, {max(ge):.0f}]"
        )
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
            f"mean={statistics.mean(e):+.3f} std={s:.3f} max={max(e):+.3f} min={min(e):+.3f}"
        )

    plot_metric_by_group(
        {lbl: grok_epochs[lbl] for lbl in order},
        out / f"{prefix}grok-epochs.png",
        title=f"Grok epoch, per grokked seed ({wd}){arch_note}",
        ylabel="grok epoch (test acc ≥ 0.99)",
    )
    plot_metric_by_group(
        gaps,
        out / f"{prefix}fve-gap.png",
        title=f"Matrix-vs-trace R² gap, matched seeds ({wd}){arch_note}",
        ylabel="R² gap (full − trace)",
    )
    plot_metric_by_group(
        excess,
        out / f"{prefix}coset-excess.png",
        title=f"Coset excess over irrep reference, matched seeds ({wd}){arch_note}",
        ylabel="probe acc − irrep-ref acc",
        hline=0.0,
        hline_label="no signal beyond irreps",
    )
    # Paired view of the gap: gaps[order[0]] and gaps[order[1]] are aligned by
    # matched seed, so their elementwise difference is the per-seed pairing the
    # two-strip figure hides (how many seeds favour the first group, by how much).
    n_fig = 3
    if len(order) == 2:
        a, b = order
        diffs = [ga - gb for ga, gb in zip(gaps[a], gaps[b])]
        n_above = sum(1 for d in diffs if d > 0)
        plot_paired_difference(
            diffs,
            out / f"{prefix}fve-gap-paired.png",
            title=f"Matrix-vs-trace R² gap, per matched seed ({wd}){arch_note}",
            ylabel=f"R² gap difference ({a} − {b})",
            xlabel=f"{a} − {b}",
            n_above=n_above,
        )
        n_fig = 4
    print(
        f"[{arch}] wrote {n_fig} figures to {out}/ ({prefix}grok-epochs, {prefix}fve-gap, "
        f"{prefix}coset-excess" + (f", {prefix}fve-gap-paired)" if n_fig == 4 else ")")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["runs"])
    ap.add_argument("--shards", type=int, default=9)
    ap.add_argument("--wd", default="wd1.0")
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    ap.add_argument("--tmp", type=Path, default=Path("scratch/pairfig_shards"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)
    roots = args.roots or ["runs"]

    procs = []
    for sid in range(args.shards):
        out_json = args.tmp / f"shard_{sid}.json"
        log = open(args.tmp / f"shard_{sid}.log", "w")
        p = subprocess.Popen(
            [
                sys.executable,
                "scripts/_pairfig_shard.py",
                str(sid),
                str(args.shards),
                args.wd,
                str(out_json),
                *roots,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((p, out_json, log))
    print(f"launched {len(procs)} shards; waiting...", flush=True)

    for p, _oj, log in procs:
        p.wait()
        log.close()
    failed = [i for i, (p, _oj, _l) in enumerate(procs) if p.returncode != 0]
    if failed:
        print(f"WARNING: shards {failed} exited nonzero; see {args.tmp}/shard_*.log", flush=True)

    results = []
    for _p, oj, _l in procs:
        if oj.exists():
            results.extend([tuple(r) for r in json.loads(oj.read_text())])
    print(f"collected {len(results)} run results from {len(procs)} shards\n", flush=True)

    for arch in ("transformer", "fc"):
        rows = [r[1:] for r in results if r[0] == arch]  # drop arch tag
        print(f"\n========== {arch}: {len(rows)} runs ==========")
        _aggregate_and_plot(arch, rows, args.out, args.wd)


if __name__ == "__main__":
    main()
