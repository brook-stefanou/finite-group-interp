"""Irrep-level metrics: where in the group's isotypic decomposition do a
model's weights live, and does the model causally depend on those blocks?

The blocks come from ``representations.projectors.real_isotypic_blocks`` --
orthogonal subspaces of R^|G| (functions on the group) determined by the
character table alone. For cyclic groups they are exactly the Fourier
cos/sin frequency planes (pinned by the calibration test).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F

from finite_group_interp.analysis.loading import list_checkpoints, load_checkpoint
from finite_group_interp.model import GroupModel
from finite_group_interp.representations.projectors import IsotypicBlock


@dataclass(frozen=True)
class EnergySpectrum:
    fractions: np.ndarray  # [n_blocks] energy fraction per block; sums to 1
    baseline: np.ndarray  # [n_blocks] expected fraction for a random matrix
    block_dims: np.ndarray  # [n_blocks] subspace dimension (= trace of projector)


def isotypic_energy(W: np.ndarray, blocks: list[IsotypicBlock]) -> EnergySpectrum:
    """Fraction of ``W``'s total variance lying in each isotypic block.

    ``W`` is [|G|, d]: each column is a function on the group (e.g. W_E with
    the '=' row dropped, or W_U.T). The baseline is what a random matrix
    shows: dim(block)/|G| -- concentration claims are ratios to it.
    """
    if not blocks:
        raise ValueError("blocks is empty -- pass real_isotypic_blocks(group)")
    if W.ndim != 2:
        raise ValueError(f"W must be 2-d [|G|, d], got shape {W.shape}")
    n = blocks[0].projector.shape[0]
    if W.shape[0] != n:
        raise ValueError(f"W has {W.shape[0]} rows but blocks act on R^{n}")
    total = float(np.sum(W * W))
    if total == 0.0:
        raise ValueError("W is identically zero -- energy fractions undefined")
    raw = np.array([float(np.sum((b.projector @ W) ** 2)) for b in blocks])
    coverage = raw.sum() / total
    if abs(coverage - 1.0) > 1e-3:
        raise ValueError(
            f"blocks do not partition R^{n}: sum ||P_i W||^2 / ||W||^2 = {coverage:.6f}; "
            "pass the complete block list from real_isotypic_blocks()"
        )
    fractions = raw / total
    block_dims = np.array([int(round(float(np.trace(b.projector)))) for b in blocks])
    return EnergySpectrum(fractions=fractions, baseline=block_dims / n, block_dims=block_dims)


def evaluate(model: GroupModel, tokens: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """(cross-entropy loss, accuracy) at the '=' position, no gradients."""
    with torch.no_grad():
        readout = model(tokens)[:, -1, :]
        loss = F.cross_entropy(readout, targets).item()
        acc = (readout.argmax(dim=-1) == targets).float().mean().item()
    return loss, acc


def weight_as_functions(model: GroupModel, matrix: Literal["W_E", "W_U"], n: int) -> np.ndarray:
    """The weight matrix as [|G|, d]: columns are functions on the group.

    W_E: drop the '=' row (not a group element). W_U: transpose, so each
    output-vocab direction becomes a function on G.
    """
    param: torch.nn.Parameter = getattr(model, matrix)
    W: np.ndarray = param.detach().numpy()
    result = W[:n].copy() if matrix == "W_E" else W.T.copy()
    return cast(np.ndarray, result)


@contextmanager
def _swapped(param: torch.nn.Parameter, new_value: torch.Tensor) -> Iterator[None]:
    original = param.data
    param.data = new_value
    try:
        yield
    finally:
        param.data = original


def _with_projected(
    W: torch.Tensor, P: np.ndarray, matrix: Literal["W_E", "W_U"], n: int, keep: bool
) -> torch.Tensor:
    """W with one block's component kept (keep=True) or removed (keep=False).

    W_E rows 0..n-1 are functions on G (the '=' row is always left untouched);
    W_U rows are functions on G, so it is projected from the right (projectors
    are symmetric).
    """
    P_t = torch.from_numpy(P).to(W.dtype)
    if matrix == "W_E":
        group_rows = W[:n]
        projected = P_t @ group_rows
        new_rows = projected if keep else group_rows - projected
        return torch.cat([new_rows, W[n:]], dim=0)
    projected = W @ P_t
    return projected if keep else W - projected


@dataclass(frozen=True)
class AblationResult:
    block_index: int
    delta_loss: float  # ablated loss - base loss (positive = block mattered)
    delta_acc: float


def block_ablation(
    model: GroupModel,
    blocks: list[IsotypicBlock],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    matrix: Literal["W_E", "W_U"] = "W_E",
) -> list[AblationResult]:
    """Causal check: zero each block's component of ``matrix``, measure damage.

    Non-destructive -- the model's weights are restored after every ablation.
    """
    n = blocks[0].projector.shape[0]
    param = getattr(model, matrix)
    base_loss, base_acc = evaluate(model, tokens, targets)
    results = []
    for i, block in enumerate(blocks):
        ablated = _with_projected(param.detach(), block.projector, matrix, n, keep=False)
        with _swapped(param, ablated):
            loss, acc = evaluate(model, tokens, targets)
        results.append(
            AblationResult(block_index=i, delta_loss=loss - base_loss, delta_acc=acc - base_acc)
        )
    return results


def restricted_loss(
    model: GroupModel,
    blocks: list[IsotypicBlock],
    keep: list[int],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    matrix: Literal["W_E", "W_U"] = "W_E",
) -> tuple[float, float]:
    """Positive control: keep ONLY the named blocks' components of ``matrix``.

    If the circuit lives in those blocks, performance must survive.
    """
    bad = [i for i in keep if not 0 <= i < len(blocks)]
    if bad:
        raise ValueError(f"block indices out of range: {bad} (have {len(blocks)} blocks)")
    n = blocks[0].projector.shape[0]
    if not keep:
        P_keep = np.zeros((n, n), dtype=float)
    else:
        P_keep = np.sum([blocks[i].projector for i in keep], axis=0)
    param = getattr(model, matrix)
    restricted = _with_projected(param.detach(), P_keep, matrix, n, keep=True)
    with _swapped(param, restricted):
        return evaluate(model, tokens, targets)


@dataclass(frozen=True)
class EnergyTrajectory:
    epochs: list[int]
    fractions: np.ndarray  # [n_checkpoints, n_blocks]


def energy_trajectory(
    run_dir: Path | str,
    blocks: list[IsotypicBlock],
    matrix: Literal["W_E", "W_U"] = "W_E",
) -> EnergyTrajectory:
    """Isotypic energy of ``matrix`` at every checkpoint of a run.

    The one function in this module that does I/O (it walks the run's
    checkpoint files); it lives here so tests can drive it on synthetic runs.
    """
    n = blocks[0].projector.shape[0]
    epochs: list[int] = []
    rows: list[np.ndarray] = []
    for path in list_checkpoints(run_dir):
        ckpt = load_checkpoint(path)
        W = weight_as_functions(ckpt.model, matrix, n)
        rows.append(isotypic_energy(W, blocks).fractions)
        epochs.append(ckpt.epoch)
    return EnergyTrajectory(epochs=epochs, fractions=np.stack(rows))
