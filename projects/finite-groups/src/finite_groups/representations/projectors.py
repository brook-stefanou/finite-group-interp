from dataclasses import dataclass

import numpy as np

from finite_groups.group import FiniteGroup
from finite_groups.representations.characters import compute_character_table
from finite_groups.representations.core import Representation


@dataclass
class IsotypicBlock:
    """A real projector onto one isotypic component of the regular representation.

    A self-conjugate (real) irrep gives a block of its own; a complex irrep is
    bundled with its conjugate so the projector comes out real.
    """

    projector: np.ndarray
    dimension: int  # the (common) dimension d_i of the irrep(s) in this block
    irrep_indices: tuple  # row indices into the character table this block covers


def isotypic_projectors(group: FiniteGroup) -> list[np.ndarray]:
    """Projector onto the isotypic component of each irrep of ``group``.

    Returned list is aligned with the rows of ``compute_character_table`` (sorted
    by dimension). Projectors for complex irreps have complex entries; group a
    complex irrep with its conjugate (see :func:`real_isotypic_blocks`) for a real
    projector.

    P_i = (d_i / |G|) * sum_g conj(chi_i(g)) * L(g)
    """
    table, classes = compute_character_table(group)
    n = group.order
    regular = Representation.regular(group).map  # {element: L(g)}

    el_to_class = {el: idx for idx, cls in enumerate(classes) for el in cls}

    projectors = []
    for i in range(table.shape[0]):
        d_i = int(round(table[i, 0].real))
        P = np.zeros((n, n), dtype=complex)
        for g in group.elements:
            chi_i_g = table[i, el_to_class[g]]
            P += np.conj(chi_i_g) * regular[g]
        P *= d_i / n
        projectors.append(P)
    return projectors


def real_isotypic_blocks(group: FiniteGroup) -> list[IsotypicBlock]:
    """Isotypic projectors with complex-conjugate irreps grouped into real blocks.

    Each block's projector is real: self-conjugate irreps project to themselves,
    while a complex irrep is summed with its conjugate so the imaginary parts
    cancel.
    """
    table, _ = compute_character_table(group)
    projectors = isotypic_projectors(group)
    k = table.shape[0]

    used = set()
    blocks = []
    for i in range(k):
        if i in used:
            continue

        conjugate_row = np.conj(table[i])
        partner = next(
            (
                j
                for j in range(k)
                if j != i
                and j not in used
                and np.allclose(table[j], conjugate_row, atol=1e-6)
            ),
            None,
        )

        d_i = int(round(table[i, 0].real))
        if partner is None:
            block = IsotypicBlock(
                projector=projectors[i].real,
                dimension=d_i,
                irrep_indices=(i,),
            )
            used.add(i)
        else:
            block = IsotypicBlock(
                projector=(projectors[i] + projectors[partner]).real,
                dimension=d_i,
                irrep_indices=(i, partner),
            )
            used.update({i, partner})
        blocks.append(block)

    return blocks
