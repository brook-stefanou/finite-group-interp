"""Typed, provenanced per-run evidence model and the threshold-derived verdict label.

The label is a compound descriptor, never a single-winner verdict: the project's
controls (especially the irrep-feature control on the coset side) exist precisely so
we do not overclaim. Thresholds are heuristic and tunable; they are named here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from same_character_table_interp.analysis.coset_metrics import CosetResult, coset_analysis
from same_character_table_interp.analysis.loading import LoadedRun, load_run
from same_character_table_interp.analysis.run_analysis import (
    _test_split_tensors,
    irrep_metrics,
)
from same_character_table_interp.representations.projectors import real_isotypic_blocks
from same_character_table_interp.training.manifest import get_git_commit

# Tunable label thresholds (documented heuristics, not learned).
EPS_COSET = 0.05  # coset excess_over_irrep below this == "adds nothing"
IRREP_CONC_BAR = 0.50  # kept blocks must hold at least this fraction of W_E energy
GROK_ACC = 0.99  # test_acc threshold that marks grokking


class KeptBlock(BaseModel):
    block: int
    irrep_dim: int
    block_dim: int
    energy: float
    w_e_rank: int


class AblationDelta(BaseModel):
    block: int
    delta_loss: float
    delta_acc: float


class FunctionalForm(BaseModel):
    cumulative_full: float
    cumulative_trace: float
    gap: float


class IrrepTier(BaseModel):
    energy_concentration: float
    kept_blocks: list[KeptBlock]
    ablation_deltas: list[AblationDelta]
    restricted_loss: float
    restricted_acc: float
    functional_form: FunctionalForm


class CosetSubgroup(BaseModel):
    h_order: int
    k: int
    probe_acc: float  # held-out coset probe accuracy (fraction, [0, 1])
    excess_null: float  # probe_acc minus random-partition null mean (accuracy points)
    excess_irrep: float  # probe_acc minus irrep-feature control (acc pts); coset signal vs irreps
    abl_cross: float  # cross-coset ERROR RATE increase post-ablation (dimensionless, [-1, 1])
    abl_ctrl: float  # matched random-subspace control: CE-loss delta (nats)
    abl_excess: float  # coset ablation CE-loss delta minus ctrl (nats); != abl_cross units


class Provenance(BaseModel):
    run_id: str
    group_spec: str
    group_order: int
    checkpoint: str
    checkpoint_epoch: int
    git_commit: str | None
    analysis_commit: str | None
    generated: str


class Learnability(BaseModel):
    grokked: bool
    grok_epoch: int | None
    final_test_acc: float
    final_test_loss: float


class VerdictComponents(BaseModel):
    irrep_energy_concentration: float
    fve_gap: float  # reported component; intentionally NOT used by compute_label — FVE gap is too seed-noisy to threshold a label on
    coset_max_excess_over_irrep: float
    grokked: bool
    grok_epoch: int | None


class Verdict(BaseModel):
    """Threshold-derived compound label plus the components it was computed from.

    The ``label`` field must be produced by ``compute_label``; ``Evidence`` records
    should be assembled via the harness path so label and components stay consistent —
    do not hand-set ``label`` in production code.
    """

    components: VerdictComponents
    label: str


class Evidence(BaseModel):
    provenance: Provenance
    learnability: Learnability
    irrep: IrrepTier
    coset: list[CosetSubgroup] | None
    verdict: Verdict


def _to_float(v: object, default: float = float("nan")) -> float:
    """Coerce an ``object``-typed value to float, falling back to *default*."""
    if v is None:
        return default
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def learnability_from_metrics(metrics: list[dict[str, object]]) -> Learnability:
    """Grok epoch (first step with test_acc >= 0.99) and final test metrics."""
    grok_epoch: int | None = None
    final_acc = float("nan")
    final_loss = float("nan")
    for rec in metrics:
        if "test_acc" not in rec:
            continue
        final_acc = _to_float(rec["test_acc"])
        final_loss = _to_float(rec.get("test_loss"), final_loss)
        if grok_epoch is None and final_acc >= GROK_ACC:
            step_val = rec.get("step", -1)
            grok_epoch = int(step_val) if isinstance(step_val, (int, float)) else -1
    return Learnability(
        grokked=grok_epoch is not None,
        grok_epoch=grok_epoch,
        final_test_acc=final_acc,
        final_test_loss=final_loss,
    )


def compute_label(c: VerdictComponents, *, has_subgroups: bool) -> str:
    """Compound, conservative label derived from verdict components. Never a winner."""
    if not c.grokked:
        return "not-grokked"
    irrep_ok = c.irrep_energy_concentration >= IRREP_CONC_BAR
    if not irrep_ok:
        return "inconclusive"
    if not has_subgroups:
        return "irrep-consistent; no-subgroups"
    if c.coset_max_excess_over_irrep < EPS_COSET:
        return "irrep-consistent; coset-adds-nothing"
    return "irrep-consistent; coset-signal"


def _coset_subgroup(r: CosetResult) -> CosetSubgroup:
    """Map one ``CosetResult`` onto the provenanced ``CosetSubgroup`` record.

    ``abl_excess`` is the coset-subspace ablation damage beyond the matched
    random-partition control (``ablation_delta_loss - random_ablation_delta_loss``);
    ``abl_ctrl`` is that control's own CE increase, ``abl_cross`` the cross-coset
    error-rate increase from removing the coset directions.
    """
    return CosetSubgroup(
        h_order=r.subgroup_size,
        k=r.n_cosets,
        probe_acc=r.probe_acc,
        excess_null=r.excess_over_null,
        excess_irrep=r.excess_over_irrep,
        abl_cross=r.ablation_cross_coset_delta,
        abl_ctrl=r.random_ablation_delta_loss,
        abl_excess=r.ablation_delta_loss - r.random_ablation_delta_loss,
    )


def evidence_from_run(run: LoadedRun) -> Evidence:
    """Assemble a full, provenanced ``Evidence`` record from a loaded run."""
    ckpt = run.checkpoint
    group = ckpt.group
    tokens, targets = _test_split_tensors(ckpt)
    irrep_raw = irrep_metrics(ckpt.model, group, tokens, targets)
    blocks = real_isotypic_blocks(group)
    keep_rows = sorted(
        {idx for kb in irrep_raw["kept_blocks"] for idx in blocks[kb["block"]].irrep_indices}
    )
    coset_results = coset_analysis(ckpt.model, group, keep_rows)
    coset = [_coset_subgroup(r) for r in coset_results] if coset_results else None

    learn = learnability_from_metrics(run.metrics)
    irrep = IrrepTier.model_validate(irrep_raw)
    coset_max = max((c.excess_irrep for c in coset), default=0.0) if coset else 0.0
    components = VerdictComponents(
        irrep_energy_concentration=irrep.energy_concentration,
        fve_gap=irrep.functional_form.gap,
        coset_max_excess_over_irrep=coset_max,
        grokked=learn.grokked,
        grok_epoch=learn.grok_epoch,
    )
    label = compute_label(components, has_subgroups=coset is not None)
    return Evidence(
        provenance=Provenance(
            run_id=run.run_dir.name,
            group_spec=ckpt.config.data.group,
            group_order=group.order,
            checkpoint=ckpt.path.name,
            checkpoint_epoch=ckpt.epoch,
            git_commit=run.manifest.get("git_commit"),
            analysis_commit=get_git_commit(),
            generated=datetime.now(timezone.utc).isoformat(),
        ),
        learnability=learn,
        irrep=irrep,
        coset=coset,
        verdict=Verdict(components=components, label=label),
    )


def run_all(run_dir: str, checkpoint: str | None = None) -> Evidence:
    """Load a run directory (latest or named checkpoint) and build its Evidence."""
    return evidence_from_run(load_run(run_dir, checkpoint))
