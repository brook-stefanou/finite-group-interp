import cmath

import numpy as np

from finite_groups.group import FiniteGroup


class Representation:
    def __init__(self, group: FiniteGroup, matrix_mapping: dict[object, np.ndarray]):
        # Group: an instance of FiniteGroup
        # matrix_mapping: a dictionary mapping group elements to matrices (Numpy arrays)
        self.group = group
        self.map = matrix_mapping
        self.degree = list(matrix_mapping.values())[0].shape[0]

    def character(self):
        # Returns the character of the representation as a class function
        return {g: m.trace() for g, m in self.map.items()}

    def is_irreducible(self) -> bool:
        # Uses Schur's lemma
        # A rep is irred iff <chi,chi> = 1

        chi = self.character()
        order = self.group.order
        inner_product = sum(abs(chi[g]) ** 2 for g in self.group.elements) / order

        return cmath.isclose(inner_product, 1.0)

    @classmethod
    def regular(cls, group: FiniteGroup) -> "Representation":
        # Method to create the regular representation of g
        matrix_mapping = {}
        n = group.order
        # The matrix represents a linear transformation on a vector space
        # with basis {e_g: g in G}, so each group element mapped to a basis vector index
        el_to_inx = {el: i for i, el in enumerate(group.elements)}

        for g in group.elements:
            matrix = np.zeros((n, n), dtype=complex)
            # Determine the permutation. The jth col represents basis vector e_h
            # The transformation g sends e_h to e_{g*h}
            for j, h in enumerate(group.elements):
                # Identify which basis the product sends it to and set to 1
                product = group.multiply(g, h)
                matrix[el_to_inx[product], j] = 1.0

            # Map the group element to its completed permutation matrix
            matrix_mapping[g] = matrix
        return cls(group, matrix_mapping)
