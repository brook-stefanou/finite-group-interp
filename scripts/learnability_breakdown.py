"""Fast learnability breakdown from metrics.jsonl only (no checkpoint loading).

Complements scripts/pair_figures.py: that script regenerates the figures and the
matrix/coset stats (which need the checkpoints), this one streams just the
metrics to get the narrative learnability numbers the write-up cites -- grokked
N/total, mean+median grok epoch, and the miss breakdown (near-miss vs
stuck-in-memorisation) with final-acc ranges -- for both the transformer pair
(D52/Dic26 wd1.0) and the FC baseline.

Usage:
    PYTHONPATH=src uv run python scripts/learnability_breakdown.py
"""

import json
import statistics
from pathlib import Path

GROK = 0.99
MEM = 0.5  # final test_acc below this = "never left memorisation"


def grok_summary(p: Path) -> tuple[int | None, float | None, int | None, float]:
    grok: int | None = None
    final: float | None = None
    last: int | None = None
    best = 0.0
    for line in p.read_text().splitlines():
        rec = json.loads(line)
        if "test_acc" not in rec:
            continue
        final = float(rec["test_acc"])
        best = max(best, final)
        last = int(rec.get("step", -1))
        if grok is None and final >= GROK:
            grok = last
    return grok, final, last, best


def parse(name: str) -> tuple[str, str, str]:
    marker = "fc-" if "fc-" in name else "pair-" if "pair-" in name else None
    if marker is None:
        return ("", "", "")
    p = name.split(marker, 1)[1].split("-")
    return (p[0], p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")


def collect(
    group_tok: str, arch: str, wd: str = "wd1.0"
) -> tuple[list[int], list[tuple[str, float, float]], int]:
    marker = "fc-" if arch == "fc" else "pair-"
    grok_epochs: list[int] = []
    misses: list[tuple[str, float, float]] = []  # (seed, final, best)
    seeds = set()
    for mp in Path("runs").glob("**/metrics.jsonl"):
        name = mp.parent.name
        g, seed, w = parse(name)
        if g != group_tok or w != wd:
            continue
        if (marker == "fc-") != ("fc-" in name):
            continue
        seeds.add(seed)
        grok, final, last, best = grok_summary(mp)
        if grok is not None:
            grok_epochs.append(grok)
        else:
            misses.append((seed, final if final is not None else 0.0, best))
    return grok_epochs, misses, len(seeds)


def report(label: str, group_tok: str, arch: str) -> None:
    ge, misses, n = collect(group_tok, arch)
    n_grok = len(ge)
    print(f"\n=== {label} [{arch}] ===  total seeds={n}")
    print(f"  grokked: {n_grok}/{n}")
    if ge:
        print(
            f"  grok epoch: mean={statistics.mean(ge):.0f} "
            f"median={statistics.median(ge):.0f} "
            f"range=[{min(ge)}, {max(ge)}]"
        )
    n_mem = sum(1 for _, f, _ in misses if f < MEM)
    n_near = len(misses) - n_mem
    print(
        f"  misses: {len(misses)}  (stuck-in-memorisation final<{MEM}: {n_mem}, near-miss: {n_near})"
    )
    if misses:
        fa = sorted(f for _, f, _ in misses)
        print(f"    miss final_acc sorted: {[round(x, 3) for x in fa]}")
        near = sorted(f for _, f, _ in misses if f >= MEM)
        if near:
            print(f"    near-miss final_acc range: [{min(near):.3f}, {max(near):.3f}]")
        print(f"    lowest miss final_acc: {min(fa):.3f}")


if __name__ == "__main__":
    print("THRESHOLDS: grok test_acc >=", GROK, " memorisation final <", MEM)
    report("Dih(104)", "D52", "transformer")
    report("Dic(104)", "Dic26", "transformer")
    report("Dih(104)", "D52", "fc")
    report("Dic(104)", "Dic26", "fc")
