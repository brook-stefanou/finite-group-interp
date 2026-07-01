import numpy as np
import pytest

from same_character_table_interp.groups.generators import GroupGenerators


def test_cyclic_factory():
    # Test the basic cyclic helper
    g = GroupGenerators.cyclic_group(3)
    expected_table = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    assert g.order == 3
    assert np.array_equal(g.cayley_table, expected_table)
    # Ensure it passes internal group axiom checks
    assert g._validate_group_axioms() is True


def test_symmetric_group_s3():
    # S3 should have order 3! = 6
    n = 3
    g = GroupGenerators.symmetric_group(n)

    assert g.order == 6
    # S3 is the smallest non-abelian group; verify it is non-abelian
    # a * b != b * a for some a, b
    is_abelian = np.array_equal(g.cayley_table, g.cayley_table.T)
    assert not is_abelian

    # Check axioms (associativity, identity, inverses)
    assert g._validate_group_axioms() is True


def test_dihedral_group_d4():
    # D4 (symmetries of a square) should have order 2*4 = 8
    n = 4
    g = GroupGenerators.dihedral_group(n)

    assert g.order == 8
    # The first element should be identity
    # r0 * r1 should be r1 (index 1)
    assert g.cayley_table[0, 1] == 1
    # s * s should be e (index 0). In our implementation s is at index n (4)
    assert g.cayley_table[4, 4] == 0

    assert g._validate_group_axioms() is True


def test_direct_product_z2_z2():
    # Z2 x Z2 is the Klein Four-group (order 4)
    z2 = GroupGenerators.cyclic_group(2)
    v4 = GroupGenerators.direct_product(z2, z2)

    assert v4.order == 4
    # Every non-identity element in V4 has order 2 (x * x = e)
    for i in range(v4.order):
        assert v4.cayley_table[i, i] == 0

    assert v4._validate_group_axioms() is True


def test_direct_product_mixed():
    # Test Z2 x S3 (order 2 * 6 = 12)
    z2 = GroupGenerators.cyclic_group(2)
    s3 = GroupGenerators.symmetric_group(3)
    g = GroupGenerators.direct_product(z2, s3)

    assert g.order == 12
    assert g._validate_group_axioms() is True


def test_symmetric_group_large():
    # Quick check for S4 order (4! = 24)
    g = GroupGenerators.symmetric_group(4)
    assert g.order == 24
    assert g._validate_group_axioms() is True


def test_semidirect_dihedral_is_nonabelian_order_six():
    # C_3 semidirect C_2 with the inversion action is D_3 (= S_3), the unique
    # non-abelian group of order 6.
    c3 = GroupGenerators.cyclic_group(3)
    c2 = GroupGenerators.cyclic_group(2)
    identity = c2.elements[0]

    def invert(h, n):
        return n if h == identity else c3.get_inverse(n)

    g = GroupGenerators.semidirect_product(c3, c2, invert)

    assert g.order == 6
    is_abelian = np.array_equal(g.cayley_table, g.cayley_table.T)
    assert not is_abelian
    assert g._validate_group_axioms() is True


def test_semidirect_trivial_action_is_direct_product():
    # A trivial action collapses to the direct product: C_2 x C_3 = C_6, abelian.
    c3 = GroupGenerators.cyclic_group(3)
    c2 = GroupGenerators.cyclic_group(2)

    g = GroupGenerators.semidirect_product(c3, c2, lambda h, n: n)

    assert g.order == 6
    assert np.array_equal(g.cayley_table, g.cayley_table.T)  # abelian
    assert g._validate_group_axioms() is True


def test_semidirect_builds_d4():
    # C_4 semidirect C_2 (inversion) is the dihedral group of the square, order 8.
    c4 = GroupGenerators.cyclic_group(4)
    c2 = GroupGenerators.cyclic_group(2)
    identity = c2.elements[0]

    def invert(h, n):
        return n if h == identity else c4.get_inverse(n)

    g = GroupGenerators.semidirect_product(c4, c2, invert)

    assert g.order == 8
    assert not np.array_equal(g.cayley_table, g.cayley_table.T)
    assert g._validate_group_axioms() is True


def test_semidirect_rejects_non_automorphism_action():
    # An action whose maps are not automorphisms of the normal subgroup is invalid;
    # a constant map is not even a bijection.
    c3 = GroupGenerators.cyclic_group(3)
    c2 = GroupGenerators.cyclic_group(2)
    constant = c3.elements[0]

    with pytest.raises(ValueError):
        GroupGenerators.semidirect_product(c3, c2, lambda h, n: constant)
