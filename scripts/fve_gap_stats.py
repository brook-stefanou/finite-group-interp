"""Summarise FVE-gap statistics from compare_pairs.py output.

Usage:
    # from a saved compare_pairs log:
    uv run python scripts/fve_gap_stats.py compare_pairs_crossseed.log

    # or pipe compare_pairs straight in:
    uv run python scripts/compare_pairs.py --coset runs/ | uv run python scripts/fve_gap_stats.py

Parses the matrix-level `GAP=` lines (one per matched run), reports each group's
gap mean/std/range, and runs a Welch t-test between the two groups. Pure text in,
no model loading -- safe to run alongside a training sweep. Drop the printed
mean+/-std and p-value into report 02's FVE-gap section.
"""

import sys
from pathlib import Path

from finite_group_interp.analysis.fve_gap_stats import gaps_from_csv, parse_fve_gaps, summarize


def main(argv: list[str]) -> None:
    if argv:
        path = Path(argv[0])
        text = path.read_text()
        parsed = gaps_from_csv(text) if path.suffix == ".csv" else parse_fve_gaps(text)
    else:
        text = sys.stdin.read()
        parsed = parse_fve_gaps(text)
    if not parsed:
        print("no gap records found -- pass a comparison.csv or compare_pairs.py log")
        return
    print(summarize(parsed))


if __name__ == "__main__":
    main(sys.argv[1:])
