from pathlib import Path

import numpy as np
import torch

from finite_group_interp.analysis.evidence import (
    Evidence,
    VerdictComponents,
    compute_label,
    evidence_from_run,
    learnability_from_metrics,
)
from finite_group_interp.analysis.loading import LoadedCheckpoint, LoadedRun
from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.representations.projectors import real_isotypic_blocks
from finite_group_interp.training.config import (
    DataConfig,
    ExperimentConfig,
    GrokkingConfig,
)
from finite_group_interp.training.trainer import build_model


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


def _planted_run(group_spec, block_index, tmp_path, *, seed=0):
    """A LoadedRun whose W_E lives entirely in one isotypic block of the group.

    Reuses the planted-model recipe from test_irrep_metrics: project a random
    embedding onto a single block's projector so all W_E energy concentrates
    there. Metrics are hand-set to a grokked trajectory.
    """
    group = resolve_group(group_spec)
    blocks = real_isotypic_blocks(group)
    config = GrokkingConfig(
        experiment=ExperimentConfig(name="t", seed=seed, use_wandb=False),
        data=DataConfig(group=group_spec),
    )
    model = build_model(config, group)
    rng = np.random.default_rng(seed)
    planted = blocks[block_index].projector @ rng.normal(size=(group.order, model.d_model))
    with torch.no_grad():
        model.W_E[: group.order] = torch.tensor(planted, dtype=model.W_E.dtype)
    model.eval()
    ckpt = LoadedCheckpoint(
        model=model, config=config, group=group, epoch=1, path=Path("step_1.pt")
    )
    return LoadedRun(
        checkpoint=ckpt,
        run_dir=tmp_path,
        metrics=[{"step": 1, "test_acc": 0.999, "test_loss": 0.01}],
        manifest={"git_commit": "planted"},
    )


def test_planted_block_yields_irrep_consistent_label(tmp_path):
    # S3 is small, non-abelian, and HAS proper subgroups, so the coset tier runs.
    # Plant in the sign irrep's block (index 1): a non-trivial 1-dim block whose
    # random-baseline energy is low enough that the keep rule (energy > 2x
    # baseline) retains it. The 2-dim block's baseline is 2/3, so a block planted
    # there can never beat 2x baseline -- a known property of the keep rule, not
    # the right target for a concentration test.
    blocks = real_isotypic_blocks(resolve_group("S3"))
    sign = next(i for i, b in enumerate(blocks) if b.dimension == 1 and i > 0)
    run = _planted_run("S3", sign, tmp_path)
    ev = evidence_from_run(run)

    assert ev.irrep.energy_concentration > 0.8
    assert ev.verdict.label.startswith("irrep-consistent")
    assert ev.learnability.grokked is True
    assert ev.coset is not None  # S3 has proper subgroups -> coset tier runs
    assert ev.provenance.group_spec == "S3"
    assert ev.provenance.group_order == 6
    assert ev.provenance.checkpoint == "step_1.pt"


def test_planted_prime_group_has_no_coset_tier(tmp_path):
    # C7 is prime: no proper subgroups, so coset_analysis returns [] and the
    # label takes the no-subgroups branch.
    blocks = real_isotypic_blocks(resolve_group("C7"))
    nontrivial = next(i for i, b in enumerate(blocks) if i > 0)
    run = _planted_run("C7", nontrivial, tmp_path)
    ev = evidence_from_run(run)

    assert ev.coset is None
    assert ev.verdict.label == "irrep-consistent; no-subgroups"
    assert ev.learnability.grokked is True
