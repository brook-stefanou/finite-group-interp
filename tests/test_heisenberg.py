"""The order-125 dim-5 discriminator: the Heisenberg group over F5.

Heisenberg/F5 is the exponent-5 extraspecial group of order 5^3 -- 25 linear
characters plus four irreducibles of degree 5. The degree-5 irreps are the point:
far richer matrix structure than the order-104 pair's degree-2 blocks, where the
matrix-vs-trace FVE gap washed out. This test pins the construction down BEFORE we
spend hours training, so we never train the wrong group.
"""

from finite_group_interp.groups.presentations import build_group
from finite_group_interp.representations.characters import compute_character_table


def _element_orders(group) -> list[int]:
    """Sorted multiset of element orders -- an isomorphism invariant."""
    identity = group.elements[group._check_identity()]
    orders = []
    for g in group.elements:
        power, k = g, 1
        while power != identity:
            power = group.multiply(power, g)
            k += 1
        orders.append(k)
    return sorted(orders)


def test_heis5_is_exponent5_extraspecial_order_125():
    g = build_group("Heis5")
    assert g.order == 125
    # Exponent 5: every non-identity element has order 5. This distinguishes the
    # Heisenberg/exp-5 group from its exp-25 extraspecial partner (C25 : C5), which
    # they share a character table with -- so degrees alone can't tell them apart.
    assert _element_orders(g) == [1] + [5] * 124
    # Extraspecial signature: 25 degree-1 chars + 4 degree-5 chars (sum of squares
    # 25*1 + 4*25 = 125). Column 0 of the (dimension-sorted) table is the degrees.
    table, _classes = compute_character_table(g)
    degrees = sorted(int(round(abs(d))) for d in table[:, 0])
    assert degrees == [1] * 25 + [5] * 4
