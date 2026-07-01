from collections import defaultdict

import pytest

from same_character_table_interp.groups.catalog import (
    CATALOG,
    all_groups,
    build_from_id,
    resolve_group,
)

# Known number of groups per order (the classification we must reproduce).
# Extended order-by-order as the catalog is populated.
EXPECTED_COUNTS = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 1,
    6: 2,
    7: 1,
    8: 5,
    9: 2,
    10: 2,
    11: 1,
    12: 5,
    13: 1,
    14: 2,
    15: 1,
    16: 14,
    17: 1,
    18: 5,
    19: 1,
}


def _element_orders(group) -> tuple[int, ...]:
    identity = group.elements[group._check_identity()]
    orders = []
    for g in group.elements:
        power, k = g, 1
        while power != identity:
            power = group.multiply(power, g)
            k += 1
        orders.append(k)
    return tuple(sorted(orders))


def _center_orders(group) -> tuple[int, ...]:
    els = group.elements
    center = [g for g in els if all(group.multiply(g, h) == group.multiply(h, g) for h in els)]
    identity = group.elements[group._check_identity()]
    orders = []
    for g in center:
        power, k = g, 1
        while power != identity:
            power = group.multiply(power, g)
            k += 1
        orders.append(k)
    return tuple(sorted(orders))


def _square_count(group) -> int:
    return len({group.multiply(g, g) for g in group.elements})


def test_every_entry_builds_to_its_declared_fingerprint():
    # Each catalog entry must build a group whose order and element-order
    # spectrum match its declared fingerprint. Catches transcription errors.
    # Where the spectrum is ambiguous (order-16 collisions), supplementary
    # invariants (center type, square-image size) pin the group down.
    for entry in CATALOG.values():
        g = entry.build()
        assert g.order == entry.gap_id[0], f"{entry.name}: wrong order"
        assert _element_orders(g) == entry.element_orders, f"{entry.name}: wrong spectrum"
        if entry.center_orders is not None:
            assert _center_orders(g) == entry.center_orders, f"{entry.name}: wrong center"
        if entry.square_count is not None:
            assert _square_count(g) == entry.square_count, f"{entry.name}: wrong square count"


def test_group_counts_match_known_classification():
    for order, expected in EXPECTED_COUNTS.items():
        present = [e for e in CATALOG.values() if e.gap_id[0] == order]
        assert len(present) == expected, f"order {order}: have {len(present)}, expected {expected}"


def test_gap_indices_are_contiguous_within_each_order():
    by_order = defaultdict(list)
    for order, index in CATALOG:
        by_order[order].append(index)
    for order, indices in by_order.items():
        assert sorted(indices) == list(range(1, len(indices) + 1)), f"order {order}: gappy indices"


def test_build_from_id_accepts_tuple_and_string():
    q8 = build_from_id((8, 4))
    assert q8.order == 8 and _element_orders(q8) == (1, 2, 4, 4, 4, 4, 4, 4)

    d4 = build_from_id("8,3")
    assert d4.order == 8 and _element_orders(d4) == (1, 2, 2, 2, 2, 2, 4, 4)


def test_build_from_id_rejects_unknown_id():
    with pytest.raises(KeyError):
        build_from_id((8, 99))


def test_all_groups_filters_by_order():
    groups = all_groups(max_order=8)
    assert all(e.gap_id[0] <= 8 for e in groups)
    assert len(groups) == sum(v for k, v in EXPECTED_COUNTS.items() if k <= 8)


def test_resolve_group_handles_alias_id_and_family():
    assert resolve_group("A4").order == 12  # catalog alias (build_group can't do A_n)
    assert resolve_group("Q8").order == 8  # catalog alias
    assert resolve_group("16,3").order == 16  # GAP id
    assert resolve_group("C8").order == 8  # parametric family fallback
    assert resolve_group("S3").order == 6  # alias and family agree


def test_catalog_is_complete_through_order_19():
    # Every order 1..19 present, and 49 groups in total.
    assert {gid[0] for gid in CATALOG} == set(range(1, 20))
    assert len(CATALOG) == 49
    assert len(CATALOG) == sum(EXPECTED_COUNTS.values())
