"""Learnability trajectory for the matched Dih(104)/Dic(104) pair at d_model=64
-- the regime where the learnability gap is real and the post reports it.

This redoes the d=256 probe (scripts/analyze_learnability_trajectory.py) at d=64
and full sample size, reusing the same library machinery
(analysis.learnability_trajectory + analysis.irrep_metrics); only the run
selection and the grok rule change.

Scope: the canonical pair sweep, runs named pair-{D52,Dic26}-s*-wd1.0-f0.4 across
several date dirs. d_model=64 is confirmed from each manifest.

Grok rule (strict, not transient-0.99): a run grokked iff manifest status ==
"completed" AND a grokked_step_*.pt checkpoint exists (written only after test_acc
>= 0.99 held for the early-stop patience window) AND manifest
final_metrics.test_acc >= 0.99. The grokked_step_*.pt epoch is the clean grok-time
reference for question (b). A seed is matched-grokked iff BOTH its groups grokked.

  (a) Global concentration onset = epoch reaching 50% of final energy-above-uniform.
      Matched seed-for-seed, is the dicyclic onset later than the dihedral onset?
  (b) Within each dicyclic run (same optimiser/seed = the clean control), does the
      quaternionic-2d sector concentrate later than the real-2d sector?

Usage:
    PYTHONPATH=src .venv/bin/python scripts/analyze_learnability_trajectory_d64.py
"""

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from finite_group_interp.analysis import figures as figs  # applies pub theme on import
from finite_group_interp.analysis.irrep_metrics import energy_trajectory
from finite_group_interp.analysis.learnability_trajectory import (
    block_rep_types,
    class_excess_trajectory,
    concentration_index,
    is_grokked,
    onset_epoch,
)
from finite_group_interp.groups.presentations import build_group
from finite_group_interp.representations.projectors import IsotypicBlock, real_isotypic_blocks

GROK_THRESHOLD = 0.99
MIN_CHECKPOINTS = 10  # below this the 50%-of-final onset crossing is unreliable
D_MODEL = 64
GROUPS = ("D52", "Dic26")
_GROUP_LABEL = {"D52": "Dih(104)", "Dic26": "Dic(104)"}


@dataclass
class RunRecord:
    group: str
    seed: int
    run_dir: Path
    d_model: int
    grokked: bool
    grok_epoch: int | None
    n_checkpoints: int


@dataclass
class RunResult:
    group: str
    seed: int
    grok_epoch: int
    global_onset: int | None  # epoch reaching 50% of final concentration index
    onsets: dict[str, int | None]  # rep-type label -> onset epoch (kept blocks)


def _grokked_epoch(run_dir: Path) -> int | None:
    """Step of the run's grokked_step_*.pt checkpoint, or None if absent."""
    hits = sorted((run_dir / "checkpoints").glob("grokked_step_*.pt"))
    if not hits:
        return None
    match = re.search(r"grokked_step_(\d+)", hits[0].name)
    return int(match.group(1)) if match else None


def scan_run(run_dir: Path) -> RunRecord:
    """Read a run's manifest + checkpoint dir and apply the strict grok rule."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    final_metrics = manifest.get("final_metrics") or {}
    grok_epoch = _grokked_epoch(run_dir)
    n_ckpts = len(list((run_dir / "checkpoints").glob("*.pt")))
    grokked = is_grokked(
        status=manifest.get("status", ""),
        has_grokked_checkpoint=grok_epoch is not None,
        final_test_acc=final_metrics.get("test_acc"),
        threshold=GROK_THRESHOLD,
    )
    return RunRecord(
        group=cfg["data"]["group"],
        seed=int(cfg["experiment"]["seed"]),
        run_dir=run_dir,
        d_model=int(cfg["model"]["d_model"]),
        grokked=grokked,
        grok_epoch=grok_epoch,
        n_checkpoints=n_ckpts,
    )


def discover(runs_root: Path) -> list[RunRecord]:
    records = []
    for group in GROUPS:
        for run_dir in runs_root.glob(f"*/*pair-{group}-s*-wd1.0-f0.4"):
            records.append(scan_run(run_dir))
    return records


def _baseline(blocks: list[IsotypicBlock]) -> np.ndarray:
    n = blocks[0].projector.shape[0]
    traces = np.array([float(np.trace(b.projector).real) for b in blocks])
    return np.asarray(traces / n)


def analyze(rec: RunRecord, blocks: list[IsotypicBlock], labels: list[str]) -> RunResult:
    """Per-run trajectory analysis: global onset + per-rep-type onset (kept blocks).

    The "kept" circuit blocks are those with W_E energy > 2x the random baseline at
    the final (grokked) checkpoint -- the same rule the rest of the analysis uses --
    read off the last trajectory row, so no checkpoint is loaded twice.
    """
    assert rec.grok_epoch is not None  # only called for grokked runs
    traj = energy_trajectory(rec.run_dir, blocks, matrix="W_E")
    baseline = _baseline(blocks)
    final = traj.fractions[-1]
    kept = [i for i in range(len(blocks)) if final[i] > 2 * baseline[i]]
    class_excess = class_excess_trajectory(traj, blocks, labels, include=kept)
    onsets = {lab: onset_epoch(traj.epochs, vals) for lab, vals in class_excess.items()}
    return RunResult(
        group=rec.group,
        seed=rec.seed,
        grok_epoch=rec.grok_epoch,
        global_onset=onset_epoch(traj.epochs, concentration_index(traj, blocks)),
        onsets=onsets,
    )


# --- figure -----------------------------------------------------------------

_DIH = figs._NEUTRAL  # dihedral = grey (baseline rep type: all real)
_DIC = figs._ACCENT  # dicyclic = accent
_QUAT = "#D55E00"  # quaternionic sector highlighted in panel B (Okabe-Ito vermillion)
# Lower than the 300-dpi house default: this panel rasterises 210 points + 210
# connector lines on a transparent ground, which balloons the PNG; 200 dpi stays
# crisp at this physical size and keeps the file under the repo's large-file gate.
_FIG_DPI = 200


def make_figure(results: dict[tuple[str, int], RunResult], out: Path) -> None:
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.0, 8.4), layout="constrained")
    seeds = sorted({s for (_, s) in results})

    # Panel A: matched-seed global concentration onset (Dih vs Dic), one line/seed.
    n_pairs = 0
    n_dic_later = 0
    for seed in seeds:
        dih, dic = results.get(("D52", seed)), results.get(("Dic26", seed))
        if dih is None or dic is None or dih.global_onset is None or dic.global_onset is None:
            continue
        n_pairs += 1
        n_dic_later += dic.global_onset > dih.global_onset
        axA.plot(
            [0, 1],
            [dih.global_onset, dic.global_onset],
            color=figs._GRID,
            lw=0.8,
            alpha=0.6,
            zorder=1,
        )
    for x, group, color in [(0, "D52", _DIH), (1, "Dic26", _DIC)]:
        ys = [
            r.global_onset
            for (g, _), r in results.items()
            if g == group and r.global_onset is not None
        ]
        axA.scatter(
            [x] * len(ys),
            ys,
            color=color,
            s=28,
            zorder=3,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.4,
        )
        if ys:
            axA.scatter(
                [x],
                [statistics.median(ys)],
                marker="_",
                s=1400,
                color=figs._darken(color),
                zorder=4,
                linewidth=2.6,
            )
    axA.set_xticks([0, 1])
    axA.set_xticklabels([_GROUP_LABEL["D52"], _GROUP_LABEL["Dic26"]])
    axA.set_xlim(-0.5, 1.5)
    axA.set_ylabel("concentration onset (epoch)")
    axA.set_title(f"(a) Dicyclic concentrates later ({n_dic_later}/{n_pairs} matched seeds)")
    figs._style(axA)

    # Panel B: within-Dic real-2d vs quaternionic-2d onset, relative to grok.
    cats = [("2d-real", "Dic\nreal-2d"), ("2d-quaternionic", "Dic\nquat-2d")]

    def rel(r: RunResult, lab: str) -> int | None:
        o = r.onsets.get(lab)
        return None if o is None else o - r.grok_epoch

    n_comp = 0
    n_quat_last = 0
    for seed in seeds:
        dic = results.get(("Dic26", seed))
        if dic is None:
            continue
        yr, yq = rel(dic, "2d-real"), rel(dic, "2d-quaternionic")
        if yr is None or yq is None:
            continue
        n_comp += 1
        n_quat_last += yq >= yr
        axB.plot([0, 1], [yr, yq], color=figs._GRID, lw=0.8, alpha=0.6, zorder=1)
    for x, (lab, _) in enumerate(cats):
        ys = [v for (g, _), r in results.items() if g == "Dic26" and (v := rel(r, lab)) is not None]
        color = _QUAT if lab == "2d-quaternionic" else _DIC
        axB.scatter(
            [x] * len(ys),
            ys,
            color=color,
            s=28,
            zorder=3,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.4,
        )
        if ys:
            axB.scatter(
                [x],
                [statistics.median(ys)],
                marker="_",
                s=1400,
                color=figs._darken(color),
                zorder=4,
                linewidth=2.6,
            )
    axB.axhline(0, color=figs._SPINE, lw=0.9, ls=(0, (4, 3)))
    axB.set_xticks([0, 1])
    axB.set_xticklabels([c[1] for c in cats])
    axB.set_xlim(-0.5, 1.5)
    axB.set_ylabel("concentration onset\n(epochs relative to grok)")
    axB.set_title(f"(b) Quaternionic sector last ({n_quat_last}/{n_comp} Dic runs)")
    figs._style(axB)

    fig.savefig(out, dpi=_FIG_DPI)
    plt.close(fig)


# --- summary ----------------------------------------------------------------


def summarize(
    results: dict[tuple[str, int], RunResult],
) -> dict[str, Any]:
    seeds = sorted({s for (_, s) in results})

    # (a) matched-seed global onset
    a_rows = []
    for seed in seeds:
        dih, dic = results.get(("D52", seed)), results.get(("Dic26", seed))
        if dih and dic and dih.global_onset is not None and dic.global_onset is not None:
            a_rows.append((seed, dih.global_onset, dic.global_onset))
    a_gaps = [dic - dih for _, dih, dic in a_rows]
    a_later = sum(g > 0 for g in a_gaps)

    # (b) within-Dic real-2d vs quaternionic-2d onset
    b_rows = []
    for seed in seeds:
        dic = results.get(("Dic26", seed))
        if dic is None:
            continue
        ro, qo = dic.onsets.get("2d-real"), dic.onsets.get("2d-quaternionic")
        if ro is not None and qo is not None:
            b_rows.append((seed, ro, qo, dic.grok_epoch))
    b_gaps = [qo - ro for _, ro, qo, _ in b_rows]
    b_quat_later = sum(g > 0 for g in b_gaps)
    b_tie = sum(g == 0 for g in b_gaps)

    return {
        "a_global_onset": {
            "n": len(a_rows),
            "dic_later_count": a_later,
            "median_gap_dic_minus_dih": statistics.median(a_gaps) if a_gaps else None,
            "median_dih_onset": statistics.median([d for _, d, _ in a_rows]) if a_rows else None,
            "median_dic_onset": statistics.median([c for _, _, c in a_rows]) if a_rows else None,
            "rows": [{"seed": s, "dih": d, "dic": c} for s, d, c in a_rows],
        },
        "b_within_dic": {
            "n_comparable": len(b_rows),
            "quaternionic_later_count": b_quat_later,
            "quaternionic_later_or_tie_count": b_quat_later + b_tie,
            "tie_count": b_tie,
            "median_gap_quat_minus_real": statistics.median(b_gaps) if b_gaps else None,
            "median_real_onset_rel_grok": (
                statistics.median([ro - g for _, ro, _, g in b_rows]) if b_rows else None
            ),
            "median_quat_onset_rel_grok": (
                statistics.median([qo - g for _, _, qo, g in b_rows]) if b_rows else None
            ),
            "rows": [
                {"seed": s, "real_2d": ro, "quat_2d": qo, "grok": g} for s, ro, qo, g in b_rows
            ],
        },
    }


def _structural_check() -> dict[str, dict[str, int]]:
    out = {}
    for group in GROUPS:
        labels = block_rep_types(build_group(group))
        out[group] = {lab: labels.count(lab) for lab in sorted(set(labels))}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("scratch/learnability-trajectory-d64.json"),
        help="summary-stats sidecar (gitignored scratch by default; not a committed artifact)",
    )
    args = parser.parse_args(argv)

    records = discover(args.runs_root)

    # --- exclusions, logged ---
    wrong_dmodel = [r for r in records if r.d_model != D_MODEL]
    sized = [r for r in records if r.d_model == D_MODEL]
    grokked = [r for r in sized if r.grokked]
    nongrok = [r for r in sized if not r.grokked]
    sparse = [r for r in grokked if r.n_checkpoints < MIN_CHECKPOINTS]
    usable = {(r.group, r.seed): r for r in grokked if r.n_checkpoints >= MIN_CHECKPOINTS}

    seeds = sorted({s for (_, s) in usable})
    matched = [s for s in seeds if ("D52", s) in usable and ("Dic26", s) in usable]

    print("=" * 72)
    print("LEARNABILITY TRAJECTORY @ d_model=64  (canonical pair, wd1.0, f0.4)")
    print("=" * 72)
    print(
        f"runs scanned: {len(records)}  (D52 {sum(r.group == 'D52' for r in records)}, "
        f"Dic26 {sum(r.group == 'Dic26' for r in records)})"
    )
    print(f"excluded, d_model != {D_MODEL}: {len(wrong_dmodel)}")
    print(
        f"strict-grokked: {len(grokked)}  "
        f"(D52 {sum(r.group == 'D52' for r in grokked)}, "
        f"Dic26 {sum(r.group == 'Dic26' for r in grokked)})"
    )
    print(
        f"excluded, not grokked (status/ckpt/final-acc): {len(nongrok)}  "
        f"(D52 {sum(r.group == 'D52' for r in nongrok)}, "
        f"Dic26 {sum(r.group == 'Dic26' for r in nongrok)})"
    )
    print(
        f"excluded, sparse (< {MIN_CHECKPOINTS} checkpoints): {len(sparse)}"
        + (
            ""
            if not sparse
            else "  " + ", ".join(f"{r.group}-s{r.seed}({r.n_checkpoints})" for r in sparse)
        )
    )
    print(f"MATCHED-GROKKED SEEDS (final n): {len(matched)}")

    print("\nstructural check (block rep types):")
    for group, counts in _structural_check().items():
        print(f"  {group}: {counts}")

    # --- run the trajectory analysis on matched-grokked seeds ---
    blocks_by_group = {g: real_isotypic_blocks(build_group(g)) for g in GROUPS}
    labels_by_group = {g: block_rep_types(build_group(g)) for g in GROUPS}
    results: dict[tuple[str, int], RunResult] = {}
    for seed in matched:
        for group in GROUPS:
            rec = usable[(group, seed)]
            results[(group, seed)] = analyze(rec, blocks_by_group[group], labels_by_group[group])

    summary = summarize(results)
    a, b = summary["a_global_onset"], summary["b_within_dic"]

    print("\n" + "-" * 72)
    print(f"(a) GLOBAL CONCENTRATION ONSET (matched seed-for-seed, n={a['n']})")
    print(f"    Dicyclic onset LATER than dihedral in {a['dic_later_count']}/{a['n']} seeds")
    print(
        f"    median onset: Dih {a['median_dih_onset']:.0f}  Dic {a['median_dic_onset']:.0f}  "
        f"-> median gap (Dic - Dih) = {a['median_gap_dic_minus_dih']:.0f} epochs"
    )
    print(f"\n(b) WITHIN-DIC REP-TYPE ONSET (comparable Dic runs, n={b['n_comparable']})")
    print(
        f"    quaternionic-2d concentrates at-or-after real-2d in "
        f"{b['quaternionic_later_or_tie_count']}/{b['n_comparable']} "
        f"(strictly later {b['quaternionic_later_count']}, tie {b['tie_count']})"
    )
    print(
        f"    median onset rel. grok: real-2d {b['median_real_onset_rel_grok']:.0f}  "
        f"quat-2d {b['median_quat_onset_rel_grok']:.0f}  "
        f"-> median gap (quat - real) = {b['median_gap_quat_minus_real']:.0f} epochs"
    )
    print("\n    caveat: correlational; n is the matched-grokked count above.")

    args.out.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    fig_path = args.out / "learnability-trajectory-d64.png"
    json_path = args.json_out
    make_figure(results, fig_path)
    json_path.write_text(
        json.dumps(
            {
                "regime": "d_model=64",
                "final_n_matched_grokked": len(matched),
                "exclusions": {
                    "wrong_d_model": len(wrong_dmodel),
                    "not_grokked": len(nongrok),
                    "sparse": [f"{r.group}-s{r.seed}({r.n_checkpoints})" for r in sparse],
                },
                "structural_check": _structural_check(),
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nfigure: {fig_path}\njson:   {json_path}")


if __name__ == "__main__":
    main()
