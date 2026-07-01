"""Aggregate per-run evidence.json into rows, a CSV, and a markdown comparison."""

import json
import re
from pathlib import Path
from typing import Any, cast

_NAME = re.compile(r"pair-([A-Za-z0-9]+)-s(\d+)-wd([\d.]+)")
_CSV_COLS = [
    "run_id",
    "group",
    "seed",
    "wd",
    "grokked",
    "grok_epoch",
    "energy_concentration",
    "gap",
    "coset_max_excess_over_irrep",
    "label",
]


def evidence_rows(evidences: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ev in evidences:
        prov: dict[str, Any] = ev["provenance"]
        learn: dict[str, Any] = ev["learnability"]
        irrep: dict[str, Any] = ev["irrep"]
        ff: dict[str, Any] = irrep["functional_form"]
        verdict: dict[str, Any] = ev["verdict"]
        components: dict[str, Any] = verdict["components"]

        run_id: str = prov["run_id"]
        m: re.Match[str] | None = _NAME.search(run_id)
        rows.append(
            {
                "run_id": run_id,
                "group": prov["group_spec"],
                "seed": int(m.group(2)) if m is not None else None,
                "wd": m.group(3) if m is not None else None,
                "grokked": learn["grokked"],
                "grok_epoch": learn["grok_epoch"],
                "energy_concentration": irrep["energy_concentration"],
                "gap": ff["gap"],
                "coset_max_excess_over_irrep": components["coset_max_excess_over_irrep"],
                "label": verdict["label"],
            }
        )
    return rows


def comparison_csv(evidences: list[dict[str, Any]]) -> str:
    rows = evidence_rows(evidences)
    out = [",".join(_CSV_COLS)]
    for r in rows:
        out.append(",".join("" if r[c] is None else str(r[c]) for c in _CSV_COLS))
    return "\n".join(out) + "\n"


def load_evidences(roots: list[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for root in roots:
        for p in Path(root).glob("**/analysis/evidence.json"):
            found.append(json.loads(p.read_text()))
    return found


def comparison_markdown(evidences: list[dict[str, Any]]) -> str:
    """Learnability + matrix-level (gap) + label table, grouped by group then seed."""
    rows = sorted(
        evidence_rows(evidences),
        key=lambda r: (str(r["group"]), cast(int, r["seed"]) if r["seed"] is not None else -1),
    )
    lines = [
        "# Pair comparison",
        "",
        "| run | group | seed | wd | grok@ | gap | label |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gap_val = r["gap"]
        gap_str = f"{cast(float, gap_val):.3f}" if gap_val is not None else ""
        lines.append(
            f"| {r['run_id']} | {r['group']} | {r['seed']} | {r['wd']} | "
            f"{r['grok_epoch']} | {gap_str} | {r['label']} |"
        )
    return "\n".join(lines) + "\n"
