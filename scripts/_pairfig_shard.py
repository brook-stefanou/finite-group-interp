"""One shard of the pair-figure per-run extraction (gap + coset excess + grok).

Each shard is an INDEPENDENT OS process launched by the driver
(scripts/pairfig_run.py): it imports torch fresh, computes its assigned runs, and
writes a JSON list of (arch, label, seed, grok, gap, excess). No shared
multiprocessing -- an mp.Pool deadlocks on macOS (the default 'spawn' method
re-imports torch + Accelerate BLAS in each worker and they never come up), so the
work is sharded across plain subprocesses instead.

Pin BLAS to 1 thread per shard *before* importing torch/numpy (the driver runs
~one shard per core).

Usage (driver-invoked):
    PYTHONPATH=src python scripts/_pairfig_shard.py <shard_id> <n_shards> <wd> <out_json> [roots...]
"""

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).parent))  # sibling imports

from compare_pairs import _arch, _parse, find_run_dirs, grok_summary  # noqa: E402
from pair_figures import _coset_excess, _gap_and_keep_rows  # noqa: E402

_LABELS = {"D52": "Dih(104)", "Dic26": "Dic(104)"}
ARCHES = ["transformer", "fc"]


def main() -> None:
    shard_id, n_shards = int(sys.argv[1]), int(sys.argv[2])
    wd, out_json = sys.argv[3], Path(sys.argv[4])
    roots = sys.argv[5:] or ["runs"]

    run_dirs = sorted(find_run_dirs(roots))
    todo: list[tuple[str, Path]] = []  # (arch, run_dir)
    for arch in ARCHES:
        for d in run_dirs:
            g, _s, w = _parse(d.name)
            if g in _LABELS and w == wd and _arch(d.name) == arch:
                todo.append((arch, d))
    mine = [t for i, t in enumerate(todo) if i % n_shards == shard_id]

    results: list[tuple[str, str, str, float | None, float | None, list[float] | None]] = []
    for arch, d in mine:
        group, seed, _wd = _parse(d.name)
        label = _LABELS[group]
        grok, _acc, _last = grok_summary(d / "metrics.jsonl")
        if grok is None:
            results.append((arch, label, seed, None, None, None))
            continue
        gap, keep_rows = _gap_and_keep_rows(d)
        excess = _coset_excess(d, keep_rows)
        results.append((arch, label, seed, float(grok), gap, excess))
        print(f"[shard {shard_id}] {d.name}: grok@{grok} gap={gap:+.3f}", flush=True)

    out_json.write_text(json.dumps(results))
    print(f"[shard {shard_id}] wrote {len(results)} results -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
