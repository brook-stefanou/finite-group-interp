import numpy as np

from finite_group_interp.groups.catalog import build_group
from finite_group_interp.groups.generators import GroupGenerators
from finite_group_interp.representations.characters import (
    compute_character_table,
    decompose_character,
    frobenius_schur_indicators,
)


def test_s3_character_table():
    group = GroupGenerators.symmetric_group(3)
    order = group.order

    table, classes = compute_character_table(group)

    assert len(classes) == 3
    assert table.shape == (3, 3)

    dimensions = sorted([abs(row[0]) for row in table])
    assert np.allclose(dimensions, [1.0, 1.0, 2.0])

    # Check row orthogonality
    class_sizes = [len(c) for c in classes]
    for i in range(len(table)):
        for j in range(len(table)):
            inner_product = (
                sum(class_sizes[k] * table[i, k] * np.conj(table[j, k]) for k in range(3)) / order
            )

            expected = 1.0 if i == j else 0.0
            assert np.isclose(inner_product, expected, atol=1e-7)

    # Check column orthogonality
    for i in range(len(classes)):
        for j in range(len(classes)):
            col_inner_product = np.sum(table[:, i] * np.conj(table[:, j]))
            expected = (order / class_sizes[i]) if i == j else 0.0
            assert np.isclose(col_inner_product, expected, atol=1e-7)


def test_cyclic_group_table():
    n = 4
    group = GroupGenerators.cyclic_group(n)
    table, classes = compute_character_table(group)

    assert len(classes) == n
    for row in table:
        assert np.isclose(abs(row[0]), 1.0)


def test_fs_indicators_dihedral_all_real():
    # Every irrep of a dihedral group is realizable over R: indicator +1.
    group = build_group("D52")
    table, _ = compute_character_table(group)
    nu = frobenius_schur_indicators(group)
    assert nu.shape == (table.shape[0],)
    assert np.allclose(nu, 1.0, atol=1e-6)


def test_fs_indicators_quaternion_has_symplectic_irrep():
    # Q8: four 1-d real irreps (+1) and one 2-d quaternionic irrep (-1).
    group = build_group("Q8")
    table, _ = compute_character_table(group)
    nu = frobenius_schur_indicators(group)
    dims = np.array([int(round(row[0].real)) for row in table])
    assert np.allclose(nu[dims == 1], 1.0, atol=1e-6)
    two_d = nu[dims == 2]
    assert two_d.shape == (1,)
    assert np.isclose(two_d[0], -1.0, atol=1e-6)


def test_fs_indicators_cyclic_complex_irreps_are_zero():
    # C4: the two complex (non-real-character) irreps have indicator 0.
    group = GroupGenerators.cyclic_group(4)
    nu = frobenius_schur_indicators(group)
    # Two real characters (trivial + the order-2 sign rep) and two complex ones.
    assert np.isclose(np.sum(np.isclose(nu, 1.0, atol=1e-6)), 2)
    assert np.isclose(np.sum(np.isclose(nu, 0.0, atol=1e-6)), 2)


def test_regular_representation_decomp():
    # Regular repn character phi has:
    # phi(id) = |G|
    # phi(g) = 0 for g!= id
    # must contain every irrep chi_i with multiplicity d_i

    group = GroupGenerators.symmetric_group(3)
    order = group.order
    table, classes = compute_character_table(group)

    phi_reg = np.zeros(len(classes))
    phi_reg[0] = order

    multiplicities, _ = decompose_character(phi_reg, group)

    for i, chi in enumerate(table):
        dimension = int(round(chi[0].real))
        assert multiplicities[i] == dimension, (
            f"Irrep {i} should appear {dimension} times, got {multiplicities[i]}"
        )
