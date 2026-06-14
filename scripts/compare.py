"""Aggregate evidence.json across runs: scripts/compare.py <roots...> [--out results]."""

import argparse
from pathlib import Path

from finite_group_interp.analysis.compare import comparison_csv, comparison_markdown, load_evidences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["runs"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    evs = load_evidences(args.roots or ["runs"])
    if not evs:
        print("no evidence.json found -- run scripts/evaluate.py first")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.csv").write_text(comparison_csv(evs))
    (out / "comparison.md").write_text(comparison_markdown(evs))
    print(f"wrote {out}/comparison.csv and comparison.md  ({len(evs)} runs)")


if __name__ == "__main__":
    main()
