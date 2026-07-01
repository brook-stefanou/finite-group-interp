from same_character_table_interp.analysis.compare import evidence_rows, comparison_csv


def _ev(
    group: str, seed: int, wd: str, gap: float, grok: int | None
) -> dict[str, object]:  # minimal evidence-shaped dict
    return {
        "provenance": {
            "run_id": f"x_pair-{group}-s{seed}-wd{wd}-f0.4",
            "group_spec": group,
            "group_order": 104,
            "checkpoint": "c",
            "checkpoint_epoch": 1,
            "git_commit": "a",
            "analysis_commit": "b",
            "generated": "t",
        },
        "learnability": {
            "grokked": grok is not None,
            "grok_epoch": grok,
            "final_test_acc": 0.99,
            "final_test_loss": 0.05,
        },
        "irrep": {
            "energy_concentration": 0.9,
            "kept_blocks": [],
            "ablation_deltas": [],
            "restricted_loss": 0.1,
            "restricted_acc": 0.97,
            "functional_form": {"cumulative_full": 0.5, "cumulative_trace": 0.5 - gap, "gap": gap},
        },
        "coset": None,
        "verdict": {
            "components": {
                "irrep_energy_concentration": 0.9,
                "fve_gap": gap,
                "coset_max_excess_over_irrep": 0.0,
                "grokked": grok is not None,
                "grok_epoch": grok,
            },
            "label": "irrep-consistent; no-subgroups",
        },
    }


def test_rows_extract_group_seed_wd_gap() -> None:
    rows = evidence_rows([_ev("D52", 1, "1.0", 0.006, 8822), _ev("Dic26", 1, "1.0", 0.028, 35292)])
    by = {(r["group"], r["seed"]): r for r in rows}
    assert by[("D52", 1)]["gap"] == 0.006
    assert by[("Dic26", 1)]["grok_epoch"] == 35292


def test_csv_has_header_and_rows() -> None:
    csv = comparison_csv([_ev("D52", 1, "1.0", 0.006, 8822)])
    lines = csv.strip().splitlines()
    assert lines[0].split(",")[:4] == ["run_id", "group", "seed", "wd"]
    assert "D52" in lines[1]
