import cmath

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from finite_groups.group import Element, FiniteGroup


class Representation(BaseModel):
    # numpy matrices require arbitrary types.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: FiniteGroup
    # Maps group elements to their representing matrices (numpy arrays).
    map: dict[Element, np.ndarray]

    @property
    def degree(self) -> int:
        return int(next(iter(self.map.values())).shape[0])

    @model_validator(mode="after")
    def _validate(self) -> "Representation":
        # 1. Domain completeness: every group element must have a matrix.
        missing = [g for g in self.group.elements if g not in self.map]
        if missing:
            raise ValueError(f"Representation missing matrices for: {missing}")

        # 2. All matrices must be square and of one consistent degree.
        d = self.degree
        for g, m in self.map.items():
            if m.shape != (d, d):
                raise ValueError(f"Matrix for {g!r} is not {d}x{d}")

        # 3. Homomorphism law: rho(g) @ rho(h) == rho(g * h) for all g, h.
        for g in self.group.elements:
            for h in self.group.elements:
                gh = self.group.multiply(g, h)
                if not np.allclose(self.map[g] @ self.map[h], self.map[gh]):
                    raise ValueError(
                        f"Not a homomorphism: rho({g!r}) @ rho({h!r}) != rho({g!r} * {h!r})"
                    )
        return self

    def character(self) -> dict[Element, complex]:
        # Returns the character of the representation as a class function
        return {g: complex(m.trace()) for g, m in self.map.items()}

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
        matrix_mapping: dict[Element, np.ndarray] = {}
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
        return cls(group=group, map=matrix_mapping)
