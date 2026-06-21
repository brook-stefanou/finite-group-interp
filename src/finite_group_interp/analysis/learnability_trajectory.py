"""Learnability trajectory: where, along training, each rep-type sector of the
weights concentrates its energy -- the energy instrument (irrep_metrics) sampled
across epochs and grouped by Frobenius-Schur type.

Motivation: the matched Dih(104)/Dic(104) pair groks at different speeds (the
robust black-box result). The two groups share the same 1-d and have the same
count of 2-d irreps, but the dicyclic's 2-d sector splits into real and
*quaternionic* blocks while the dihedral's is entirely real
(:func:`block_rep_types`). Sampling isotypic energy across the trajectory turns
"dicyclic is slower" into a rep-type story: does the quaternionic sector
concentrate later than the real sector (within Dic, the cleanest control), and
later than the dihedral's real sector (across groups)?

Pure functions here (labelling, per-class excess energy, onset detection) are
unit-tested; the run-walking orchestration lives in
scripts/analyze_learnability_trajectory.py.
"""

from collections.abc import Sequence

import numpy as np

from finite_group_interp.analysis.irrep_metrics import EnergyTrajectory
from finite_group_interp.groups.group import FiniteGroup
from finite_group_interp.representations.characters import frobenius_schur_indicators
from finite_group_interp.representations.projectors import IsotypicBlock, real_isotypic_blocks

# Frobenius-Schur indicator -> rep-type tag for the >=2-d blocks.
_FS_TAG = {1: "real", -1: "quaternionic", 0: "complex"}


def block_rep_types(group: FiniteGroup) -> list[str]:
    """Rep-type label for each block of :func:`real_isotypic_blocks`, aligned.

    1-d blocks are tagged ``"1d"``; higher-dim blocks carry their Frobenius-
    Schur type, e.g. ``"2d-real"`` (dihedral) vs ``"2d-quaternionic"``
    (dicyclic). A complex block bundles an irrep with its conjugate; both share
    the indicator, so the first irrep index is representative.
    """
    nu = frobenius_schur_indicators(group)
    blocks = real_isotypic_blocks(group)
    labels: list[str] = []
    for b in blocks:
        if b.dimension == 1:
            labels.append("1d")
            continue
        tag = _FS_TAG[int(round(float(nu[b.irrep_indices[0]])))]
        labels.append(f"{b.dimension}d-{tag}")
    return labels


def class_excess_trajectory(
    trajectory: EnergyTrajectory,
    blocks: list[IsotypicBlock],
    labels: Sequence[str],
    include: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Summed energy-above-random-baseline per rep-type class, over epochs.

    For each block, excess = energy_fraction - baseline, where baseline =
    trace(projector)/|G| is the fraction a random matrix puts in that block.
    Summing the excess over a class measures how much that rep-type sector has
    concentrated above uniform (0 at random init); the result is a
    ``{label: array[n_epochs]}`` dict.

    ``include`` restricts the sum to those block indices (e.g. the circuit's
    "kept" blocks, energy > 2x baseline at the final checkpoint). Without it the
    whole sector is summed, where the many unused blocks' small deficits cancel
    the few circuit blocks and can drive the net negative -- pass the kept set to
    measure the energy the circuit actually uses, per rep type.
    """
    n = blocks[0].projector.shape[0]
    baseline = np.array([float(np.trace(b.projector).real) for b in blocks]) / n
    excess = trajectory.fractions - baseline  # [n_epochs, n_blocks]
    allowed = set(range(len(blocks)) if include is None else include)
    out: dict[str, np.ndarray] = {}
    for label in sorted({labels[i] for i in allowed}):
        idx = [i for i, lab in enumerate(labels) if lab == label and i in allowed]
        out[label] = excess[:, idx].sum(axis=1)
    return out


def concentration_index(trajectory: EnergyTrajectory, blocks: list[IsotypicBlock]) -> np.ndarray:
    """Total energy above the random-uniform baseline, per epoch (0..1).

    Sum of positive excess over all blocks: 0 when energy is spread exactly as a
    random matrix would (memorised / pre-grok), rising as the circuit piles
    energy into a few blocks. Signed excess sums to zero by construction, so the
    positive part is the natural one-number concentration measure. This is the
    global "how far from uniform" signal for question (a).
    """
    n = blocks[0].projector.shape[0]
    baseline = np.array([float(np.trace(b.projector).real) for b in blocks]) / n
    return np.asarray(np.maximum(trajectory.fractions - baseline, 0.0).sum(axis=1))


def onset_epoch(
    epochs: Sequence[int], values: np.ndarray, frac: float = 0.5, eps: float = 1e-6
) -> int | None:
    """First epoch where ``values`` reaches ``frac`` of its final value.

    ``values`` is a per-class excess-energy trajectory. The "onset" of
    concentration is the first epoch crossing ``frac * values[-1]``. Returns
    None when the class never meaningfully concentrates (final excess <= eps),
    so a sector that stays near uniform doesn't report a spurious onset.
    """
    final = float(values[-1])
    if final <= eps:
        return None
    threshold = frac * final
    for epoch, value in zip(epochs, values, strict=True):
        if value >= threshold:
            return int(epoch)
    return None
