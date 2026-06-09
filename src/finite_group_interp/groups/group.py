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
    # Lazily-cached lookups so element ops aren't O(n) per call (rebuilding the
    # index dict on every multiply made subgroup enumeration unusable at order ~100).
    _el_index: dict[Element, int] | None = PrivateAttr(default=None)
    _inverse_index: dict[int, int] | None = PrivateAttr(default=None)

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

    def _index(self) -> dict[Element, int]:
        if self._el_index is None:
            self._el_index = {el: inx for inx, el in enumerate(self.elements)}
        return self._el_index

    def el_to_inx(self, g: Element) -> int:
        return self._index()[g]

    def multiply(self, g: Element, h: Element) -> Element:
        idx = self._index()
        return self.elements[int(self.cayley_table[idx[g], idx[h]])]

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
        if self._inverse_index is None:
            e = self._check_identity()
            # g^-1 is the column where row g equals the identity (left inverse;
            # equals the right inverse in a group). Cayley table is validated.
            self._inverse_index = {
                i: int(np.flatnonzero(self.cayley_table[i] == e)[0]) for i in range(self.order)
            }
        return self.elements[self._inverse_index[self._index()[g]]]

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

    def _closure_idx(self, gens: tuple[int, ...]) -> frozenset[int]:
        """Subgroup (as element indices) generated by ``gens`` (indices).

        Right-multiplication BFS from the identity over the generators: every
        word in the generators is reached, so the orbit is the generated
        subgroup (inverses come for free, g^-1 = g^(order-1), in a finite group).
        Works in index space off the Cayley table -- no per-op dict rebuilds.
        """
        table = self.cayley_table
        e = self._check_identity()
        seen = {e, *gens}
        frontier = list(seen)
        while frontier:
            a = frontier.pop()
            for g in gens:
                p = int(table[a, g])
                if p not in seen:
                    seen.add(p)
                    frontier.append(p)
        return frozenset(seen)

    def subgroups(self) -> list[list[Element]]:
        """All subgroups, each as an element list (identity first), sorted by
        size then element index.

        Cyclic extension: start from the trivial subgroup and repeatedly extend
        every found subgroup by one element. Every subgroup H = <g1,...,gk> is
        reached ({e} -> <g1> -> <g1,g2> -> ...), so this is complete; closures
        run on small generating sets in index space, so it scales to order ~350.
        """
        e = self._check_identity()
        trivial = frozenset({e})
        gens_of: dict[frozenset[int], tuple[int, ...]] = {trivial: ()}
        queue: list[frozenset[int]] = [trivial]
        while queue:
            h = queue.pop()
            base = gens_of[h]
            for g in range(self.order):
                if g in h:
                    continue
                k = self._closure_idx((*base, g))
                if k not in gens_of:
                    gens_of[k] = (*base, g)
                    queue.append(k)

        def as_list(indices: frozenset[int]) -> list[Element]:
            rest = sorted(i for i in indices if i != e)
            return [self.elements[e], *(self.elements[i] for i in rest)]

        return sorted(
            (as_list(s) for s in gens_of),
            key=lambda lst: (len(lst), [self.el_to_inx(x) for x in lst]),
        )

    def is_normal(self, H: list[Element]) -> bool:
        """True if gHg^-1 = H for all g (H given as an element list)."""
        h_set = set(H)
        for g in self.elements:
            g_inv = self.get_inverse(g)
            for h in H:
                if self.multiply(self.multiply(g, h), g_inv) not in h_set:
                    return False
        return True

    def left_cosets(self, H: list[Element]) -> list[list[Element]]:
        """The partition {gH}, deterministic order (by smallest element index)."""
        idx = {el: i for i, el in enumerate(self.elements)}
        seen: set[Element] = set()
        cosets: list[list[Element]] = []
        for g in self.elements:
            if g in seen:
                continue
            coset = sorted((self.multiply(g, h) for h in H), key=lambda e: idx[e])
            cosets.append(coset)
            seen.update(coset)
        return cosets

    def center(self) -> list[Element]:
        """{z : zg = gz for all g}, sorted by element index."""
        idx = {el: i for i, el in enumerate(self.elements)}
        z = [
            g
            for g in self.elements
            if all(self.multiply(g, x) == self.multiply(x, g) for x in self.elements)
        ]
        return sorted(z, key=lambda e: idx[e])
