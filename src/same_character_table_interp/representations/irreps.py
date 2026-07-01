"""Extract concrete irrep matrices rho_i(g) numerically from the regular
representation, for any finite group.

For each character-table row i, the isotypic projector P_i carves out a
d_i^2-dimensional component of the regular representation, on which G acts as
rho_i (x) I_{d_i} -- d_i identical copies of the irrep. An orthonormal basis S
of that component reduces the regular rep to A(g) = S^H L(g) S. Averaging a
random Hermitian seed over the group lands in the commutant, which by Schur
acts as I_{d_i} (x) M; the eigenvectors of its top eigenvalue cluster (size
d_i) span exactly one irreducible copy Q, and rho_i(g) = Q^H L(g) Q is
automatically unitary and a homomorphism. (A single generic vector does NOT
work for d_i >= 2: its orbit span is the whole d_i^2-dim component, not one
copy.) Verified exhaustively at construction: characters determine irreps up
to equivalence, so character-match + homomorphism + irreducibility is a
correctness proof.
"""

from dataclasses import dataclass

import numpy as np

from same_character_table_interp.groups.group import Element, FiniteGroup
from same_character_table_interp.representations.characters import compute_character_table
from same_character_table_interp.representations.core import Representation
from same_character_table_interp.representations.projectors import isotypic_projectors


@dataclass(frozen=True)
class Irrep:
    """One irreducible representation, extracted numerically.

    ``matrices`` maps each group element to its d_i x d_i unitary matrix;
    ``table_index`` is the row of ``compute_character_table`` this came from
    (irreps are returned aligned with that table, sorted by dimension).
    """

    matrices: dict[Element, np.ndarray]
    character: dict[Element, complex]
    dimension: int
    table_index: int


def _extract_one(
    group: FiniteGroup,
    projector: np.ndarray,
    regular: dict[Element, np.ndarray],
    d_i: int,
    table_index: int,
) -> Irrep:
    n = group.order
    elements = group.elements

    # Orthonormal basis S of the full isotypic component (dimension d_i^2). On
    # this component G acts as rho_i (x) I_{d_i}: d_i identical copies of the
    # irrep. A relative gap (not an absolute atol) separates the d_i^2 true
    # directions (projector singular values == 1) from numerical leakage
    # (~1e-6), which an absolute atol=1e-8 would miscount as signal. The gap
    # from 1 down to ~1e-6 is wide, so a 1e-3 relative cut is robust on both
    # sides. The same relative gap is reused below for the eigenvalue-cluster
    # width, where the commutant's distinct eigenvalues are likewise
    # well-separated.
    rtol = 1e-3
    up, sp, _ = np.linalg.svd(projector)
    m = int(np.sum(sp > rtol * max(sp[0], 1.0)))
    if m != d_i * d_i:
        raise ValueError(
            f"irrep row {table_index}: isotypic component has dim {m} != d_i^2={d_i * d_i}; "
            "extraction failed"
        )
    s_basis = up[:, :m]  # [n, m] orthonormal

    # Restricted rep A(g) = S^H L(g) S, isomorphic to rho_i (x) I_{d_i}.
    a_rep = [s_basis.conj().T @ regular[g] @ s_basis for g in elements]

    # A generic element of the commutant (operators commuting with every A(g))
    # acts as I_{d_i} (x) M on the multiplicity space; each of its eigenspaces is
    # exactly one irreducible copy (dimension d_i, group-invariant). We average a
    # random Hermitian seed over the group to land in the commutant, then take
    # the eigenvectors of the top eigenvalue cluster as that single copy. Retry
    # with fresh seeds if a draw is degenerate (top cluster size != d_i).
    for attempt in range(5):
        rng = np.random.default_rng(1000 * table_index + attempt)
        seed = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        seed = seed + seed.conj().T
        commutant = sum((a @ seed @ a.conj().T for a in a_rep), np.zeros((m, m), dtype=complex)) / n
        evals, evecs = np.linalg.eigh(commutant)  # ascending
        order = np.argsort(-evals)
        evals, evecs = evals[order], evecs[:, order]
        top = evals[0]
        cluster = int(np.sum(np.abs(evals - top) <= rtol * (abs(top) + 1.0)))
        if cluster == d_i:
            q = s_basis @ evecs[:, :d_i]  # [n, d_i] orthonormal copy basis
            break
    else:
        raise ValueError(
            f"irrep row {table_index}: commutant eigen-cluster != d_i={d_i} after retries "
            "(degenerate generic seed); extraction failed"
        )

    matrices = {g: q.conj().T @ regular[g] @ q for g in elements}
    character = {g: complex(np.trace(mat)) for g, mat in matrices.items()}
    return Irrep(matrices=matrices, character=character, dimension=d_i, table_index=table_index)


def extract_irreps(group: FiniteGroup) -> list[Irrep]:
    """All irreps of ``group``, one per character-table row (sorted by dim).

    Raises ValueError if any extracted matrix fails the homomorphism,
    unitarity, character, or irreducibility check.
    """
    table, classes = compute_character_table(group)
    el_to_class = {el: i for i, cls in enumerate(classes) for el in cls}
    projectors = isotypic_projectors(group)
    regular = Representation.regular(group).map

    irreps: list[Irrep] = []
    for i in range(table.shape[0]):
        d_i = int(round(table[i, 0].real))
        irr = _extract_one(group, projectors[i], regular, d_i, i)
        _verify(group, irr, table[i], el_to_class)
        irreps.append(irr)
    return irreps


def _verify(
    group: FiniteGroup,
    irr: Irrep,
    table_row: np.ndarray,
    el_to_class: dict[Element, int],
) -> None:
    d = irr.dimension
    eye = np.eye(d)
    for g in group.elements:
        if not np.allclose(irr.matrices[g].conj().T @ irr.matrices[g], eye, atol=1e-6):
            raise ValueError(f"irrep row {irr.table_index}: rho({g!r}) not unitary")
        chi = table_row[el_to_class[g]]
        if not np.isclose(irr.character[g], chi, atol=1e-5):
            raise ValueError(
                f"irrep row {irr.table_index}: tr rho({g!r})={irr.character[g]:.4f} != chi={chi:.4f}"
            )
        for h in group.elements:
            gh = group.multiply(g, h)
            if not np.allclose(irr.matrices[g] @ irr.matrices[h], irr.matrices[gh], atol=1e-6):
                raise ValueError(
                    f"irrep row {irr.table_index}: homomorphism fails at ({g!r}, {h!r})"
                )
    inner = sum(abs(irr.character[g]) ** 2 for g in group.elements) / group.order
    if not np.isclose(inner, 1.0, atol=1e-5):
        raise ValueError(f"irrep row {irr.table_index}: not irreducible (<chi,chi>={inner:.4f})")
