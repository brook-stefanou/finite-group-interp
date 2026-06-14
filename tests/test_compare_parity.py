import os
from pathlib import Path
from typing import cast

import pytest

from finite_group_interp.analysis.compare import evidence_rows
from finite_group_interp.analysis.evidence import run_all

LOCAL = Path("runs/2026-06-09").is_dir() and Path("runs/2026-06-10").is_dir()


@pytest.mark.skipif(
    not LOCAL or os.environ.get("RUN_PARITY") != "1",
    reason="CPU-heavy parity check; set RUN_PARITY=1 with the order-104 sweep runs present",
)
def test_compare_reproduces_report02_numbers() -> None:
    # Known values from compare_pairs_crossseed.log (matched wd1.0 seeds).
    expected_gap = {("D52", 1): 0.006, ("Dic26", 1): 0.028, ("D52", 4): 0.217}
    expected_grok = {("Dic26", 5): 71891, ("D52", 1): 8822}
    pair_dirs = list(Path("runs/2026-06-09").glob("*pair*")) + list(
        Path("runs/2026-06-10").glob("*pair*")
    )
    evs = [run_all(str(d)).model_dump() for d in pair_dirs]
    rows = {(r["group"], r["seed"]): r for r in evidence_rows(evs)}
    for key, gap in expected_gap.items():
        assert abs(cast(float, rows[key]["gap"]) - gap) < 1e-2, key
    for key, ep in expected_grok.items():
        assert rows[key]["grok_epoch"] == ep, key
