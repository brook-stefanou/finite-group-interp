"""Mechanistic analysis of trained checkpoints.

Implemented: loading (checkpoints -> models), cache (models -> activations).
Planned (see TODO.md Tier 1): irrep_metrics, coset_metrics, evidence, figures.
"""

from .cache import ActivationCache, forward_with_cache
from .loading import (
    LoadedCheckpoint,
    LoadedRun,
    list_checkpoints,
    load_checkpoint,
    load_run,
)

__all__ = [
    "ActivationCache",
    "LoadedCheckpoint",
    "LoadedRun",
    "forward_with_cache",
    "list_checkpoints",
    "load_checkpoint",
    "load_run",
]
