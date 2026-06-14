"""Typed, provenanced per-run evidence model and the threshold-derived verdict label.

The label is a compound descriptor, never a single-winner verdict: the project's
controls (especially the irrep-feature control on the coset side) exist precisely so
we do not overclaim. Thresholds are heuristic and tunable; they are named here.
"""

from __future__ import annotations

from pydantic import BaseModel

# Tunable label thresholds (documented heuristics, not learned).
EPS_COSET = 0.05  # coset excess_over_irrep below this == "adds nothing"
IRREP_CONC_BAR = 0.50  # kept blocks must hold at least this fraction of W_E energy


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
    probe_acc: float
    excess_null: float
    excess_irrep: float
    abl_cross: float
    abl_ctrl: float
    abl_excess: float


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
