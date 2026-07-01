"""Observation-mode access to a model's internal activations.

``forward_with_cache`` is the public analysis entry point: one forward pass
under ``no_grad`` in eval mode, returning every intermediate. Callers that
need gradients *through the model* (attribution-style methods) should bypass
this and call ``model(tokens, return_cache=True)`` directly.
"""

import torch

from same_character_table_interp.model import ActivationCache, GroupModel

__all__ = ["ActivationCache", "forward_with_cache"]


def forward_with_cache(model: GroupModel, tokens: torch.Tensor) -> ActivationCache:
    """Run ``tokens`` through ``model`` and return all intermediate activations.

    no_grad means the returned tensors are detached observations -- analyses
    (energy metrics, probes on cached features, ablations) cannot accidentally
    backprop into the model under study. The model's train/eval mode is
    restored on exit.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            cache: ActivationCache = model(tokens, return_cache=True)
    finally:
        if was_training:
            model.train()
    return cache
