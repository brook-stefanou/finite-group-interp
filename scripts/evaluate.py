"""Write an Evidence record for one run: scripts/evaluate.py <run_dir> [--checkpoint step_N]."""

import argparse
from pathlib import Path

from same_character_table_interp.analysis.evidence import run_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    ev = run_all(args.run_dir, args.checkpoint)
    out = Path(args.run_dir) / "analysis" / "evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ev.model_dump_json(indent=2))
    print(f"wrote {out}  ({ev.verdict.label})")


if __name__ == "__main__":
    main()
