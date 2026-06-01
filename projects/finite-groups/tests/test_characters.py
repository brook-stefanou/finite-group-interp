import numpy as np

from finite_groups.generators import GroupGenerators
from finite_groups.representations.characters import (
    compute_character_table,
    decompose_character,
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
