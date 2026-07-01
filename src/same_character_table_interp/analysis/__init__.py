"""Mechanistic analysis of trained checkpoints.

Implemented: loading (checkpoints -> models), cache (models -> activations),
irrep_metrics (energy/ablation/trajectory), figures, run_analysis (one-call
pipeline). Planned (TODO.md Tier 1): coset_metrics, evidence.
"""

from .cache import ActivationCache, forward_with_cache
from .coset_metrics import (
    CosetResult,
    ablate_coset_direction,
    coset_analysis,
    coset_labels,
    coset_probe_suite,
    coset_subspace,
    fit_linear_probe,
    random_partition_null,
)
from .functional_form import (
    FunctionalFormResult,
    fit_logit_tensor,
    functional_form_fit,
    logit_tensor,
    representation_product_features,
)
from .irrep_metrics import (
    AblationResult,
    EnergySpectrum,
    EnergyTrajectory,
    block_ablation,
    energy_trajectory,
    evaluate,
    isotypic_energy,
    restricted_loss,
    weight_as_functions,
)
from .loading import (
    LoadedCheckpoint,
    LoadedRun,
    list_checkpoints,
    load_checkpoint,
    load_run,
)
from .run_analysis import analyze

__all__ = [
    "AblationResult",
    "ActivationCache",
    "CosetResult",
    "EnergySpectrum",
    "EnergyTrajectory",
    "FunctionalFormResult",
    "LoadedCheckpoint",
    "LoadedRun",
    "ablate_coset_direction",
    "analyze",
    "block_ablation",
    "coset_analysis",
    "coset_labels",
    "coset_probe_suite",
    "coset_subspace",
    "energy_trajectory",
    "evaluate",
    "fit_linear_probe",
    "fit_logit_tensor",
    "forward_with_cache",
    "functional_form_fit",
    "isotypic_energy",
    "list_checkpoints",
    "load_checkpoint",
    "load_run",
    "logit_tensor",
    "random_partition_null",
    "representation_product_features",
    "restricted_loss",
    "weight_as_functions",
]
