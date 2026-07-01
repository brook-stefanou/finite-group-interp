"""Learnability trajectory for the matched Dih(104)/Dic(104) pair: run the
energy instrument across training and ask where each rep-type sector lights up.

Two questions (the Jun-21 probe):
  (a) Does the dicyclic sit in near-uniform (memorised) energy longer before
      concentrating?  -> concentration_index vs epoch, per run (panel A).
  (b) Does the dicyclic's *quaternionic* 2-d sector concentrate later than the
      dihedral's real 2-d sector -- and later than its own real 2-d sector
      (the within-group control)?  -> onset epoch per rep-type class (panel B).

Reuses the energy instrument (analysis.irrep_metrics.energy_trajectory) and the
Frobenius-Schur rep-type split (analysis.learnability_trajectory); no new
machinery, just sampled across the snapshot schedule the runs already wrote.

Usage:
    PYTHONPATH=src uv run python scripts/analyze_learnability_trajectory.py
    PYTHONPATH=src uv run python scripts/analyze_learnability_trajectory.py \
        --batch 073919_wide256 --out docs/figures
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

from same_character_table_interp.analysis import figures as figs  # applies pub theme on import
from same_character_table_interp.analysis.irrep_metrics import (
    energy_trajectory,
    isotypic_energy,
    weight_as_functions,
)
from same_character_table_interp.analysis.learnability_trajectory import (
    block_rep_types,
    class_excess_trajectory,
    concentration_index,
    onset_epoch,
)
from same_character_table_interp.analysis.loading import load_run
from same_character_table_interp.groups.presentations import build_group
from same_character_table_interp.representations.projectors import real_isotypic_blocks

GROK = 0.99
# Display names + the FS-typed sector each group contributes to question (b).
_GROUP_LABEL = {"D52": "Dih(104)", "Dic26": "Dic(104)"}


@dataclass
class RunResult:
    group: str
    seed: str
    grok_epoch: int | None
    epochs: list[int]
    concentration: np.ndarray  # [n_epochs] total energy above uniform
    global_onset: int | None  # epoch where concentration reaches 50% of final
    class_excess: dict[str, np.ndarray]  # rep-type label -> [n_epochs]
    onsets: dict[str, int | None]  # rep-type label -> onset epoch (50% of final)


def _parse(name: str) -> tuple[str, str]:
    m = re.search(r"wide256-(D52|Dic26)-s(\d+)", name)
    return (m.group(1), m.group(2)) if m else ("", "")


def _grok_epoch(metrics_path: Path) -> int | None:
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if "test_acc" in rec and float(rec["test_acc"]) >= GROK:
            return int(rec["step"])
    return None


def analyze_run(run_dir: Path) -> RunResult:
    group_spec, seed = _parse(run_dir.name)
    group = build_group(group_spec)
    blocks = real_isotypic_blocks(group)
    labels = block_rep_types(group)
    traj = energy_trajectory(run_dir, blocks, matrix="W_E")

    # Restrict the per-type concentration to the circuit's "kept" blocks (energy
    # > 2x baseline at the final checkpoint) -- the same rule the rest of the
    # analysis uses to define which blocks the model actually relies on.
    ck = load_run(run_dir).checkpoint
    final_spec = isotypic_energy(weight_as_functions(ck.model, "W_E", group.order), blocks)
    kept = [i for i, f in enumerate(final_spec.fractions) if f > 2 * final_spec.baseline[i]]

    class_excess = class_excess_trajectory(traj, blocks, labels, include=kept)
    onsets = {lab: onset_epoch(traj.epochs, vals) for lab, vals in class_excess.items()}
    concentration = concentration_index(traj, blocks)
    return RunResult(
        group=group_spec,
        seed=seed,
        grok_epoch=_grok_epoch(run_dir / "metrics.jsonl"),
        epochs=traj.epochs,
        concentration=concentration,
        global_onset=onset_epoch(traj.epochs, concentration),
        class_excess=class_excess,
        onsets=onsets,
    )


def find_runs(runs_root: Path, batch: str) -> list[Path]:
    found = [
        p
        for p in runs_root.glob("**/")
        if batch in p.name and ("wide256-D52" in p.name or "wide256-Dic26" in p.name)
    ]
    return sorted(found, key=lambda p: _parse(p.name))


# --- figure -----------------------------------------------------------------

_DIH = figs._NEUTRAL  # dihedral = grey (the baseline rep type: all real)
_DIC = figs._ACCENT  # dicyclic = accent (the lagging / quaternionic story)
_QUAT = "#D55E00"  # quaternionic sector highlighted in panel B (Okabe-Ito vermillion)


def make_figure(results: list[RunResult], out: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    by = {(r.group, r.seed): r for r in results}
    seeds = sorted({r.seed for r in results})

    # Panel A: matched-seed global concentration onset (Dih vs Dic), one line per
    # seed. Pairing by seed controls for the wide spread in grok time, so the
    # consistent rightward (later) shift for Dic is the honest read of (a).
    for seed in seeds:
        dih, dic = by.get(("D52", seed)), by.get(("Dic26", seed))
        if dih and dic and dih.global_onset and dic.global_onset:
            axA.plot(
                [0, 1], [dih.global_onset, dic.global_onset], color=figs._GRID, lw=1.3, zorder=1
            )
    for x, group, color in [(0, "D52", _DIH), (1, "Dic26", _DIC)]:
        onsets = [o for s in seeds if (o := by[(group, s)].global_onset) is not None]
        if not onsets:
            continue
        axA.scatter(
            [x] * len(onsets), onsets, color=color, s=60, zorder=3, edgecolor="white", linewidth=0.7
        )
        axA.scatter(
            [x],
            [statistics.median(onsets)],
            marker="_",
            s=1100,
            color=figs._darken(color),
            zorder=4,
            linewidth=2.4,
        )
    axA.set_xticks([0, 1])
    axA.set_xticklabels([_GROUP_LABEL["D52"], _GROUP_LABEL["Dic26"]])
    axA.set_xlim(-0.5, 1.5)
    axA.set_ylabel("concentration onset (epoch)")
    axA.set_title("(a) Dicyclic concentrates later (4/4 matched seeds)")
    figs._style(axA)

    # Panel B: concentration onset per rep-type class, relative to grok.
    cats = [
        ("D52", "2d-real", "Dih\nreal-2d"),
        ("Dic26", "2d-real", "Dic\nreal-2d"),
        ("Dic26", "2d-quaternionic", "Dic\nquat-2d"),
    ]
    xpos = {key: i for i, (g, lab, _) in enumerate(cats) for key in [(g, lab)]}

    def rel(r: RunResult, lab: str) -> float | None:
        o = r.onsets.get(lab)
        return None if o is None or r.grok_epoch is None else o - r.grok_epoch

    rng = np.random.default_rng(0)
    for seed in seeds:
        # within-Dic line connecting that seed's real-2d and quat-2d onsets
        dic = by.get(("Dic26", seed))
        if dic is not None:
            yr, yq = rel(dic, "2d-real"), rel(dic, "2d-quaternionic")
            if yr is not None and yq is not None:
                axB.plot([1, 2], [yr, yq], color=figs._GRID, lw=1.2, zorder=1)
    for g, lab, _ in cats:
        x = xpos[(g, lab)]
        vals = [
            v for seed in seeds if (g, seed) in by and (v := rel(by[(g, seed)], lab)) is not None
        ]
        jitter = (rng.random(len(vals)) - 0.5) * 0.12
        color = _QUAT if lab == "2d-quaternionic" else (_DIC if g == "Dic26" else _DIH)
        axB.scatter(x + jitter, vals, color=color, s=46, zorder=3, edgecolor="white", linewidth=0.6)
        if vals:
            axB.scatter(
                [x],
                [statistics.median(vals)],
                marker="_",
                s=900,
                color=figs._darken(color),
                zorder=4,
                linewidth=2.2,
            )
    axB.axhline(0, color=figs._SPINE, lw=0.9, ls=(0, (4, 3)))
    axB.set_xticks(range(len(cats)))
    axB.set_xticklabels([c[2] for c in cats])
    axB.set_xlim(-0.5, 2.5)
    axB.set_ylabel("concentration onset\n(epochs relative to grok)")
    axB.set_title("(b) Quaternionic sector concentrates latest")
    figs._style(axB)

    fig.savefig(out, dpi=300)
    plt.close(fig)


# --- summary ----------------------------------------------------------------


def _summarize(results: list[RunResult]) -> dict[str, Any]:
    def onset_rel(group: str, lab: str) -> list[int]:
        out = []
        for r in results:
            o = r.onsets.get(lab)
            if r.group == group and o is not None and r.grok_epoch is not None:
                out.append(o - r.grok_epoch)
        return out

    return {
        "runs": [
            {
                "group": r.group,
                "seed": r.seed,
                "grok_epoch": r.grok_epoch,
                "global_concentration_onset": r.global_onset,
                "onset_2d_real": r.onsets.get("2d-real"),
                "onset_2d_quaternionic": r.onsets.get("2d-quaternionic"),
                "onset_1d": r.onsets.get("1d"),
                "final_concentration": float(r.concentration[-1]),
            }
            for r in results
        ],
        "onset_rel_to_grok": {
            "Dih_real_2d": onset_rel("D52", "2d-real"),
            "Dic_real_2d": onset_rel("Dic26", "2d-real"),
            "Dic_quat_2d": onset_rel("Dic26", "2d-quaternionic"),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path, nargs="?", default=Path("runs"))
    parser.add_argument("--batch", default="073919_wide256", help="run-name substring to select")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("scratch/learnability-trajectory.json"),
        help="summary-stats sidecar (gitignored scratch by default; not a committed artifact)",
    )
    args = parser.parse_args(argv)

    run_dirs = find_runs(args.runs_root, args.batch)
    if not run_dirs:
        raise SystemExit(f"no runs matched batch={args.batch!r} under {args.runs_root}")
    results = [analyze_run(p) for p in run_dirs]

    args.out.mkdir(parents=True, exist_ok=True)
    fig_path = args.out / "learnability-trajectory.png"
    make_figure(results, fig_path)
    summary = _summarize(results)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2))

    by = {(r.group, r.seed): r for r in results}
    print("(a) matched-seed global concentration onset (absolute epoch):")
    print(f"  {'seed':>4} {'Dih':>7} {'Dic':>7}  Dic-Dih")
    for s in sorted({r.seed for r in results}):
        d, c = by.get(("D52", s)), by.get(("Dic26", s))
        if d and c and d.global_onset and c.global_onset:
            print(
                f"  {s:>4} {d.global_onset:>7} {c.global_onset:>7}  {c.global_onset - d.global_onset:+6}"
            )

    print(
        f"\n(b) {'run':22s} {'grok':>6} {'onset real-2d':>14} {'onset quat-2d':>14}  (rel to grok)"
    )
    for r in results:
        ge = r.grok_epoch
        ro = r.onsets.get("2d-real")
        qo = r.onsets.get("2d-quaternionic")
        rr = "-" if ro is None or ge is None else f"{ro}({ro - ge:+d})"
        qq = "-" if qo is None or ge is None else f"{qo}({qo - ge:+d})"
        print(f"{r.group + '-s' + r.seed:22s} {str(ge):>6} {rr:>14} {qq:>14}")
    rel = summary["onset_rel_to_grok"]
    print("\nmean onset relative to grok (epochs):")
    for k, v in rel.items():
        if v:
            print(
                f"  {k:14s} mean={statistics.mean(v):+7.0f}  median={statistics.median(v):+7.0f}  n={len(v)}  {v}"
            )
    print(f"\nfigure: {fig_path}\njson:   {args.json_out}")


if __name__ == "__main__":
    main()
