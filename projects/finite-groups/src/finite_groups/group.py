import numpy as np


class FiniteGroup:
    def __init__(self, elements: list, cayley_table: np.ndarray):
        # Assigns data
        self.elements = elements
        self.cayley_table = cayley_table

        # Calculates Order
        self.order = len(self.elements)

        # Validation
        if self.cayley_table.shape != (self.order, self.order):
            raise ValueError("Cayley table dimensions do not match number of elements")

        # Defined placeholder identity index
        self._identity_index: int | None = None

    def __eq__(self, other):
        if not isinstance(other, FiniteGroup):
            return False
        return self.elements == other.elements and np.array_equal(
            self.cayley_table, other.cayley_table
        )

    def el_to_inx(self, g):
        return {el: inx for inx, el in enumerate(self.elements)}[g]

    def multiply(self, g, h):
        element_inx = {el: inx for inx, el in enumerate(self.elements)}
        g_inx = element_inx[g]
        h_inx = element_inx[h]
        return self.elements[self.cayley_table[g_inx][h_inx]]

    def __repr__(self):
        return f"Group(elements = {self.elements}, order = {self.order})"

    def _validate_group_axioms(self):
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

    def _check_inverses(self, identity_index: int):
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

    def get_inverse(self, g):
        for h in self.elements:
            if self.multiply(g, h) == self.elements[self._check_identity()]:
                return h
        return ValueError("Invalid Group: No inverse found")

    def conjugacy_classes(self) -> list:
        classes = []
        seen = set()

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
