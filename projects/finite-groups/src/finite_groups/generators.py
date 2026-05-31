import numpy as np

from .group import FiniteGroup


class GroupGenerators:
    @staticmethod
    def cyclic_group(n: int) -> FiniteGroup:
        # Generates the Cyclic group C_n
        elements = [f"z{i}" for i in range(n)]

        range_arr = np.arange(n)
        table = (range_arr + range_arr[:, None]) % n

        return FiniteGroup(elements=elements, cayley_table=table)

    @classmethod
    def symmetric_group(cls, n: int):
        """Generates S_n of order n! without itertools."""

        # Recursive helper to find all permutations
        def get_permutations(arr):
            if len(arr) == 0:
                return [[]]
            res = []
            for i in range(len(arr)):
                rest = arr[:i] + arr[i + 1 :]
                for p in get_permutations(rest):
                    res.append([arr[i]] + p)
            return res

        elements_list = get_permutations(list(range(n)))
        # Convert to tuples for dictionary hashing (mapping)
        elements_tuples = [tuple(p) for p in elements_list]

        perm_to_idx = {p: i for i, p in enumerate(elements_tuples)}
        order = len(elements_tuples)
        table = np.zeros((order, order), dtype=np.int64)

        for i in range(order):
            p1 = elements_tuples[i]
            for j in range(order):
                p2 = elements_tuples[j]
                # Composition: (p1 ∘ p2)(k) = p1[p2[k]]
                res = tuple(p1[p2[k]] for k in range(n))
                table[i, j] = perm_to_idx[res]

        str_elements = ["".join(map(str, p)) for p in elements_tuples]
        return FiniteGroup(elements=str_elements, cayley_table=table)

    @classmethod
    def direct_product(cls, group_a, group_b):
        """Combines two FiniteGroups without itertools."""
        n_a, n_b = group_a.order, group_b.order
        new_order = n_a * n_b

        # Build element names manually
        new_elements = []
        for a_name in group_a.elements:
            for b_name in group_b.elements:
                new_elements.append(f"({a_name},{b_name})")

        table = np.zeros((new_order, new_order), dtype=np.int64)

        for i in range(new_order):
            # i = idx_a * n_b + idx_b
            idx_a_i, idx_b_i = divmod(i, n_b)

            for j in range(new_order):
                idx_a_j, idx_b_j = divmod(j, n_b)

                # Operation is done component-wise
                res_a = group_a.cayley_table[idx_a_i, idx_a_j]
                res_b = group_b.cayley_table[idx_b_i, idx_b_j]

                # Resulting index uses the same mapping logic
                table[i, j] = res_a * n_b + res_b

        return FiniteGroup(elements=new_elements, cayley_table=table)

    @classmethod
    def dihedral_group(cls, n: int):
        """Generates D_n of order 2n."""
        # Elements: (reflection, rotation) where reflection is 0 or 1
        elements_data = []
        for s in range(2):
            for r in range(n):
                elements_data.append((s, r))

        order = 2 * n
        table = np.zeros((order, order), dtype=np.int64)

        for i in range(order):
            s1, r1 = elements_data[i]
            for j in range(order):
                s2, r2 = elements_data[j]

                # Applying (s1, r1) * (s2, r2):
                if s1 == 0:
                    # Identity or pure rotation on the left
                    res_s = s2
                    res_r = (r1 + r2) % n
                else:
                    # Reflection on the left: s * s = e, s * r = r^-1 * s
                    res_s = 1 - s2
                    res_r = (r1 - r2) % n

                # Calculate index: (res_s * n) + res_r
                table[i, j] = res_s * n + res_r

        # Friendly names: e, r, r2... s, sr, sr2...
        names = []
        for s, r in elements_data:
            prefix = "s" if s == 1 else ""
            suffix = f"r{r}" if r > 0 else ("e" if s == 0 else "")
            names.append(prefix + suffix)

        return FiniteGroup(elements=names, cayley_table=table)

    @classmethod
    def semidirect_product(cls, normal, acting, action):
        """Builds the semidirect product N (x) H of `normal` by `acting`.

        `action(h, n)` is the automorphism phi_h applied to n: it takes an
        element h of `acting` and an element n of `normal`, and returns the
        image phi_h(n) as an element of `normal`. A trivial action (phi_h = id)
        recovers the direct product.

        Multiplication: (n1, h1) * (n2, h2) = (n1 * phi_h1(n2), h1 * h2).
        """
        n_N, n_H = normal.order, acting.order
        normal_idx = {el: i for i, el in enumerate(normal.elements)}

        # Resolve and validate the action as an index table phi[h_idx, n_idx].
        phi = np.zeros((n_H, n_N), dtype=np.int64)
        for hi, h in enumerate(acting.elements):
            for ni, n in enumerate(normal.elements):
                image = action(h, n)
                if image not in normal_idx:
                    raise ValueError(
                        f"Action of {h!r} sent {n!r} outside the normal subgroup"
                    )
                phi[hi, ni] = normal_idx[image]

            # Each phi_h must be a bijection (necessary for an automorphism).
            if len(set(phi[hi].tolist())) != n_N:
                raise ValueError(f"Action of {h!r} is not a bijection on the normal subgroup")

            # ...and must preserve the operation: phi_h(a*b) == phi_h(a)*phi_h(b).
            for a in range(n_N):
                for b in range(n_N):
                    ab = normal.cayley_table[a, b]
                    if phi[hi, ab] != normal.cayley_table[phi[hi, a], phi[hi, b]]:
                        raise ValueError(
                            f"Action of {h!r} is not an automorphism of the normal subgroup"
                        )

        new_order = n_N * n_H
        new_elements = []
        for n_name in normal.elements:
            for h_name in acting.elements:
                new_elements.append(f"({n_name},{h_name})")

        table = np.zeros((new_order, new_order), dtype=np.int64)
        for i in range(new_order):
            ni, hi = divmod(i, n_H)
            for j in range(new_order):
                nj, hj = divmod(j, n_H)

                # (n_i, h_i) * (n_j, h_j) = (n_i * phi_{h_i}(n_j), h_i * h_j)
                res_n = normal.cayley_table[ni, phi[hi, nj]]
                res_h = acting.cayley_table[hi, hj]
                table[i, j] = res_n * n_H + res_h

        return FiniteGroup(elements=new_elements, cayley_table=table)
