import numpy as np
import pytest

from same_character_table_interp.groups.generators import GroupGenerators
from same_character_table_interp.groups.presentations import build_group, from_presentation


def _element_orders(group) -> list[int]:
    """Sorted multiset of element orders — an isomorphism invariant."""
    identity = group.elements[group._check_identity()]
    orders = []
    for g in group.elements:
        power = g
        k = 1
        while power != identity:
            power = group.multiply(power, g)
            k += 1
        orders.append(k)
    return sorted(orders)


def _is_abelian(group) -> bool:
    return np.array_equal(group.cayley_table, group.cayley_table.T)


def test_presentation_c3_matches_cyclic_group():
    # <a | a^3> is the cyclic group C_3. There is only one group of order 3,
    # so matching order + abelian + element-order spectrum proves isomorphism
    # to our existing cyclic_group(3).
    g = from_presentation("a", ["aaa"])
    c3 = GroupGenerators.cyclic_group(3)

    assert g.order == c3.order == 3
    assert _is_abelian(g)
    assert _element_orders(g) == _element_orders(c3) == [1, 3, 3]


def test_presentation_c4_matches_cyclic_group():
    # <a | a^4> is C_4. Element orders [1, 2, 4, 4] distinguish it from the
    # other order-4 group (Klein four, [1, 2, 2, 2]).
    g = from_presentation("a", ["aaaa"])
    c4 = GroupGenerators.cyclic_group(4)

    assert g.order == c4.order == 4
    assert _is_abelian(g)
    assert _element_orders(g) == _element_orders(c4) == [1, 2, 4, 4]


def test_presentation_d3_matches_symmetric_group():
    # <a, b | a^3, b^2, (ab)^2> is the dihedral group D_3 = S_3. Order 6 and
    # non-abelian already forces S_3; element orders confirm it.
    g = from_presentation("ab", ["aaa", "bb", "abab"])
    s3 = GroupGenerators.symmetric_group(3)

    assert g.order == s3.order == 6
    assert not _is_abelian(g)
    assert _element_orders(g) == _element_orders(s3) == [1, 2, 2, 2, 3, 3]


def test_presentation_builds_quaternion_group():
    # Q_8 = <a, b | a^4, b^2 = a^2, b a b^-1 = a^-1>, written as relators:
    #   a^4 = e            -> "aaaa"
    #   b^2 a^-2 = e       -> "bbAA"
    #   b a b^-1 a = e     -> "baBa"
    # Q_8 has a unique element of order 2, so its order spectrum is
    # [1, 2, 4, 4, 4, 4, 4, 4] -- this is what separates it from D_4.
    g = from_presentation("ab", ["aaaa", "bbAA", "baBa"])

    assert g.order == 8
    assert not _is_abelian(g)
    assert _element_orders(g) == [1, 2, 4, 4, 4, 4, 4, 4]


def test_max_order_guard_rejects_infinite_group():
    # <a, b | a^2, b^2> is the infinite dihedral group; enumeration must abort
    # at the cap rather than loop forever.
    with pytest.raises(ValueError, match="max_order"):
        from_presentation("ab", ["aa", "bb"], max_order=50)


# === Named-group registry (CLI / config friendly specs) ===


def test_build_group_cyclic_spec_uses_labeled_constructor():
    # "C3" should dispatch to cyclic_group and keep its labels exactly.
    g = build_group("C3")
    c3 = GroupGenerators.cyclic_group(3)
    assert g.elements == c3.elements
    assert np.array_equal(g.cayley_table, c3.cayley_table)


def test_build_group_symmetric_spec_uses_labeled_constructor():
    g = build_group("S4")
    s4 = GroupGenerators.symmetric_group(4)
    assert g.elements == s4.elements
    assert np.array_equal(g.cayley_table, s4.cayley_table)


def test_build_group_dihedral_spec():
    g = build_group("D4")
    assert g.order == 8
    assert _element_orders(g) == [1, 2, 2, 2, 2, 2, 4, 4]


def test_build_group_quaternion_spec():
    g = build_group("Q8")
    assert g.order == 8
    assert _element_orders(g) == [1, 2, 4, 4, 4, 4, 4, 4]


def test_build_group_dicyclic_spec():
    # Dic_n has order 4n with a UNIQUE involution (a^n) -- the signature that
    # distinguishes it from the dihedral group of the same order.
    g = build_group("Dic3")
    assert g.order == 12
    assert _element_orders(g) == [1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 6, 6]
    assert _element_orders(g).count(2) == 1


def test_build_group_dicyclic_order_104():
    # Dic(104) = Dic26, the partner of Dih(104)=D52 in the primary pair.
    g = build_group("Dic26")
    assert g.order == 104
    assert _element_orders(g).count(2) == 1  # one involution (vs D52's 53)


def test_build_group_c13_semidirect_c8():
    # C13 ⋊ C8 (order 104): the order-matched, different-character-table contrast
    # to the Dih(104)/Dic(104) pair. C8 acts on C13 by mult-by-8 (order 4 mod 13).
    g = build_group("C13sdC8")
    assert g.order == 104
    assert not _is_abelian(g)  # nontrivial action -> non-abelian
    orders = set(_element_orders(g))
    # order-8 AND order-13 elements -- D52 and Dic26 have neither order 8, so this
    # is structurally a different group from both members of the pair.
    assert 8 in orders and 13 in orders


def test_build_group_klein_four_via_d2():
    # D2 = <a,b | a^2, b^2, (ab)^2> is the Klein four-group.
    g = build_group("D2")
    assert g.order == 4
    assert _is_abelian(g)
    assert _element_orders(g) == [1, 2, 2, 2]


def test_build_group_rejects_unknown_family():
    with pytest.raises(ValueError):
        build_group("Z9")


def test_build_group_rejects_malformed_spec():
    with pytest.raises(ValueError):
        build_group("C")


def test_build_group_rejects_undefined_quaternion_order():
    with pytest.raises(ValueError):
        build_group("Q7")
