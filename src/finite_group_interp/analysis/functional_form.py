"""Representation-product functional-form fit: how well does a model's logit
tensor L[a,b,c] reconstruct from irrep features rho_i(a)rho_i(b)rho_i(c)^H?

Reports full-matrix FVE (all d_i^2 matrix elements) vs trace-only FVE (just
the characters chi_i(abc^-1)); the gap = full - trace is the sub-character
structure that distinguishes the irreps hypothesis from a character-level
account. Pure functions over numpy; the model is touched only to read logits.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.model import OneLayerTransformer
from finite_group_interp.representations.irreps import Irrep
from finite_group_interp.task import build_group_task


def center_over_c(logits: np.ndarray) -> np.ndarray:
    """Subtract the per-(a,b) mean over the output axis c (softmax-inert)."""
    return cast(np.ndarray, logits - logits.mean(axis=2, keepdims=True))


def logit_tensor(model: OneLayerTransformer, group: FiniteGroup) -> np.ndarray:
    """RAW logits L[a,b,c]: the model's score for output c on every input (a,b).

    Centering is the fit's job, so this returns un-centered logits. Output is
    sliced to the |G| group-element columns (any '=' output column dropped).
    """
    n = group.order
    task = build_group_task(group)  # inputs row k = (k//n, k%n)
    eq_col = np.full((task.inputs.shape[0], 1), n)  # the '=' input token id
    tokens = torch.tensor(np.concatenate([task.inputs, eq_col], axis=1), dtype=torch.long)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            readout: np.ndarray = model(tokens)[:, -1, :].detach().numpy()  # [n^2, d_vocab_out]
    finally:
        if was_training:
            model.train()
    return cast(np.ndarray, readout[:, :n].reshape(n, n, n))  # [a, b, c]


def _irrep_stack(irrep: Irrep, group: FiniteGroup) -> np.ndarray:
    """[n, d, d] complex: rho_i(elements[k]) stacked by element index."""
    return np.stack([irrep.matrices[g] for g in group.elements], axis=0)


def representation_product_features(
    irreps: list[Irrep], group: FiniteGroup, keep: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrices over flattened (a, b, c), for the kept irreps.

    full:  real & imag parts of every entry of M = rho(a) rho(b) rho(c)^H
    trace: real & imag parts of tr M = chi(abc^-1)
    Both shaped [|G|^3, n_features], rows aligned with logit_tensor(...).reshape(-1).
    """
    bad = [j for j in keep if not 0 <= j < len(irreps)]
    if bad:
        raise ValueError(f"irrep indices out of range: {bad} (have {len(irreps)} irreps)")
    n = group.order
    if not keep:
        zero = np.empty((n**3, 0))
        return zero, zero
    full_cols: list[np.ndarray] = []
    trace_cols: list[np.ndarray] = []
    for j in keep:
        rho = _irrep_stack(irreps[j], group)  # [n, d, d]
        ab = np.einsum("asu,buv->absv", rho, rho)
        m = np.einsum("absv,ctv->abcst", ab, np.conj(rho))
        m_flat = m.reshape(n**3, -1)  # [n^3, d^2]
        full_cols.append(m_flat.real)
        full_cols.append(m_flat.imag)
        tr = np.trace(m, axis1=3, axis2=4).reshape(n**3, 1)  # [n^3, 1] = chi(abc^-1)
        trace_cols.append(tr.real)
        trace_cols.append(tr.imag)
    x_full = np.concatenate(full_cols, axis=1)
    x_trace = np.concatenate(trace_cols, axis=1)
    return x_full, x_trace


@dataclass(frozen=True)
class FunctionalFormResult:
    per_irrep_full: dict[int, float]  # FVE of each kept irrep alone, full-matrix
    per_irrep_trace: dict[int, float]  # FVE of each kept irrep alone, trace-only
    cumulative_full: float  # FVE of the whole kept set, full-matrix
    cumulative_trace: float  # FVE of the whole kept set, trace-only
    gap: float  # cumulative_full - cumulative_trace
    n_features_full: int
    n_features_trace: int


def _fve(y: np.ndarray, x: np.ndarray, total: float) -> float:
    """1 - ||y - X beta||^2 / ||y||^2 via least squares; 0.0 if no features.

    ``total`` (= y . y) is passed in so the caller validates it once. Conjugate-
    pair features make X rank-deficient (real+imag of an irrep and its conjugate
    span the same space); lstsq's SVD min-norm solution handles that without
    error, and FVE depends only on the residual, so it is invariant to which
    min-norm beta is chosen.
    """
    if x.shape[1] == 0:
        return 0.0
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return 1.0 - float(resid @ resid) / total


def fit_logit_tensor(
    logits: np.ndarray, group: FiniteGroup, irreps: list[Irrep], keep: list[int]
) -> FunctionalFormResult:
    """Fit (mean-centered) logits onto kept irreps' features; full vs trace FVE."""
    y = center_over_c(logits).reshape(-1)
    # Build the cumulative features first: this validates `keep` (out-of-range
    # raises here) before the zero-variance guard, so a bad keep-list is
    # reported regardless of the logits passed.
    xf_all, xt_all = representation_product_features(irreps, group, keep)
    total = float(y @ y)
    if total == 0.0:
        raise ValueError("logit tensor has zero variance after centering")
    per_full: dict[int, float] = {}
    per_trace: dict[int, float] = {}
    for j in keep:
        xf, xt = representation_product_features(irreps, group, [j])
        per_full[j] = _fve(y, xf, total)
        per_trace[j] = _fve(y, xt, total)
    cum_full = _fve(y, xf_all, total)
    cum_trace = _fve(y, xt_all, total)
    return FunctionalFormResult(
        per_irrep_full=per_full,
        per_irrep_trace=per_trace,
        cumulative_full=cum_full,
        cumulative_trace=cum_trace,
        gap=cum_full - cum_trace,
        n_features_full=xf_all.shape[1],
        n_features_trace=xt_all.shape[1],
    )


def functional_form_fit(
    model: OneLayerTransformer, group: FiniteGroup, irreps: list[Irrep], keep: list[int]
) -> FunctionalFormResult:
    """Convenience wrapper: read the model's logits, then fit."""
    return fit_logit_tensor(logit_tensor(model, group), group, irreps, keep)
