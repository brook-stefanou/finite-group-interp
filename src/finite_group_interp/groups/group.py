from collections.abc import Hashable

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator, model_validator

# Group elements are opaque labels (strings in the catalog, ints/tuples in
# tests); all the structure lives in the Cayley table. Hashable is the one
# real requirement -- elements are used as dict keys for index lookups.
Element = Hashable


class FiniteGroup(BaseModel):
    # Pydantic has no native numpy support, so allow arbitrary types for the table.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elements: list[Element]
    cayley_table: np.ndarray

    # Cached identity index, populated during validation.
    _identity_index: int | None = PrivateAttr(default=None)

    @property
    def order(self) -> int:
        return len(self.elements)

    # ---- Layer 1: structural validation (field-level) ----
    @field_validator("cayley_table")
    @classmethod
    def _table_is_integer_matrix(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise ValueError("Cayley table must be a numpy array")
        if v.ndim != 2:
            raise ValueError("Cayley table must be 2-dimensional")
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError("Cayley table must contain integer indices")
        return v

    # ---- Layer 1 + 2: cross-field structure, then mathematical axioms ----
    @model_validator(mode="after")
    def _validate(self) -> "FiniteGroup":
        n = self.order

        if self.cayley_table.shape != (n, n):
            raise ValueError("Cayley table dimensions do not match number of elements")

        if n > 0 and (self.cayley_table.min() < 0 or self.cayley_table.max() >= n):
            raise ValueError("Cayley table entries must be valid indices in [0, n)")

        # Mathematical group axioms (identity, inverses, associativity).
        self._validate_group_axioms()
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiniteGroup):
            return False
        return self.elements == other.elements and np.array_equal(
            self.cayley_table, other.cayley_table
        )

    def el_to_inx(self, g: Element) -> int:
        return {el: inx for inx, el in enumerate(self.elements)}[g]

    def multiply(self, g: Element, h: Element) -> Element:
        element_inx = {el: inx for inx, el in enumerate(self.elements)}
        g_inx = element_inx[g]
        h_inx = element_inx[h]
        return self.elements[int(self.cayley_table[g_inx, h_inx])]

    def __repr__(self) -> str:
        return f"Group(elements = {self.elements}, order = {self.order})"

    def _validate_group_axioms(self) -> bool:
        # Validates identity, inverse and associativity
        self._identity_index = self._check_identity()
        self._check_inverses(self._identity_index)
        self._check_associativity()

        return True

    def _check_associativity(self) -> bool:
        n = self.order

        for a in range(n):
            for b in range(n):
                # pre-caculation (a * b)
                ab = self.cayley_table[a, b]

                # (a * b) * G
                left_side = self.cayley_table[ab, :]

                # a * (b * G)
                right_side = self.cayley_table[a, self.cayley_table[b, :]]

                if not np.array_equal(left_side, right_side):
                    # Extract the first failing index as an integer
                    c_idx = np.flatnonzero(left_side != right_side)[0]
                    raise ValueError(
                        f"Associativity fails for triplet: ({self.elements[a]}, {self.elements[b]}, {self.elements[c_idx]})"
                    )
        return True

    def _check_identity(self) -> int:
        # Returns index of the identity
        # Raises ValueError if no identity or multiple found
        n = self.order
        expected = np.arange(n)

        row_check = np.all(self.cayley_table == expected, axis=1)
        col_check = np.all(self.cayley_table == expected[:, None], axis=0)

        possible_identities = np.flatnonzero(row_check & col_check)

        if len(possible_identities) == 0:
            raise ValueError("Invalid Group: No identity found")

        return int(possible_identities[0])

    def _check_inverses(self, identity_index: int) -> bool:
        # Verifies that every element has an inverse
        # Condition: The identity must appear exactly once for every row

        # Mask
        is_identity = self.cayley_table == identity_index

        # Sum rows (right inverses)
        row_counts = np.sum(is_identity, axis=1)

        # Verify exactly one inverse per row
        if not np.all(row_counts == 1):
            raise ValueError("Right inverses are not unique")

        # Sum cols (left inverses)
        col_counts = np.sum(is_identity, axis=0)
        if not np.all(col_counts == 1):
            raise ValueError("Left inverses are not unique")

        return True

    def get_inverse(self, g: Element) -> Element:
        for h in self.elements:
            if self.multiply(g, h) == self.elements[self._check_identity()]:
                return h
        raise ValueError("Invalid Group: No inverse found")

    def conjugacy_classes(self) -> list[list[Element]]:
        classes: list[list[Element]] = []
        seen: set[Element] = set()

        for i, x in enumerate(self.elements):
            if x in seen:
                continue

            current_class = set()
            # calculate the orbit of x under conjugation g*x*g^-1
            for j, g in enumerate(self.elements):
                g_inv = self.get_inverse(g)
                g_inv_inx = self.el_to_inx(g_inv)

                step_1_inx = self.cayley_table[j][i]
                conj_inx = self.cayley_table[step_1_inx][g_inv_inx]
                conj_element = self.elements[conj_inx]

                current_class.add(conj_element)
                seen.add(conj_element)

            classes.append(list(current_class))

        identity_elment = self.elements[self._check_identity()]
        for i, cls in enumerate(classes):
            if identity_elment in cls:
                classes.insert(0, classes.pop(i))
                break

        return classes
