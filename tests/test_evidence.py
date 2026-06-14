from finite_group_interp.analysis.evidence import (
    Evidence,
    VerdictComponents,
    compute_label,
    learnability_from_metrics,
)


def _components(**kw):
    base = dict(
        irrep_energy_concentration=0.9,
        fve_gap=0.01,
        coset_max_excess_over_irrep=0.0,
        grokked=True,
        grok_epoch=8822,
    )
    base.update(kw)
    return VerdictComponents(**base)


def test_label_not_grokked():
    assert (
        compute_label(_components(grokked=False, grok_epoch=None), has_subgroups=True)
        == "not-grokked"
    )


def test_label_prime_group():
    assert compute_label(_components(), has_subgroups=False) == "irrep-consistent; no-subgroups"


def test_label_coset_adds_nothing():
    assert (
        compute_label(_components(coset_max_excess_over_irrep=0.0), has_subgroups=True)
        == "irrep-consistent; coset-adds-nothing"
    )


def test_label_coset_signal():
    out = compute_label(_components(coset_max_excess_over_irrep=0.20), has_subgroups=True)
    assert out == "irrep-consistent; coset-signal"


def test_label_inconclusive_when_irrep_weak():
    assert (
        compute_label(_components(irrep_energy_concentration=0.10), has_subgroups=True)
        == "inconclusive"
    )


def test_evidence_round_trips_through_json():
    ev = Evidence(
        provenance={
            "run_id": "r",
            "group_spec": "D52",
            "group_order": 104,
            "checkpoint": "step_1.pt",
            "checkpoint_epoch": 1,
            "git_commit": "abc",
            "analysis_commit": "def",
            "generated": "2026-06-14T00:00:00Z",
        },
        learnability={
            "grokked": True,
            "grok_epoch": 8822,
            "final_test_acc": 0.99,
            "final_test_loss": 0.05,
        },
        irrep={
            "energy_concentration": 0.9,
            "kept_blocks": [],
            "ablation_deltas": [],
            "restricted_loss": 0.1,
            "restricted_acc": 0.97,
            "functional_form": {"cumulative_full": 0.5, "cumulative_trace": 0.49, "gap": 0.01},
        },
        coset=None,
        verdict={
            "components": _components().model_dump(),
            "label": "irrep-consistent; no-subgroups",
        },
    )
    again = Evidence.model_validate_json(ev.model_dump_json())
    assert again.verdict.label == "irrep-consistent; no-subgroups"
    assert again.coset is None


def test_learnability_grokked():
    metrics = [
        {"step": 0, "test_acc": 0.01, "test_loss": 4.6},
        {"step": 100, "test_acc": 0.50, "test_loss": 2.0},
        {"step": 200, "test_acc": 0.995, "test_loss": 0.05},
        {"step": 300, "test_acc": 0.999, "test_loss": 0.01},
    ]
    lr = learnability_from_metrics(metrics)
    assert lr.grokked is True
    assert lr.grok_epoch == 200  # first step with test_acc >= 0.99
    assert lr.final_test_acc == 0.999  # last recorded
    assert lr.final_test_loss == 0.01


def test_learnability_never_grokked():
    metrics = [
        {"step": 0, "test_acc": 0.01, "test_loss": 4.6},
        {"step": 100, "test_acc": 0.10, "test_loss": 3.0},
    ]
    lr = learnability_from_metrics(metrics)
    assert lr.grokked is False
    assert lr.grok_epoch is None
    assert lr.final_test_acc == 0.10
