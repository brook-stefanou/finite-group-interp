"""A catalog of all finite groups up to order 19, keyed by GAP SmallGroup IDs.

Each entry pairs a GAP ``(order, index)`` identifier with a builder and a
fingerprint (order + element-order spectrum) sourced from the standard
classification. The fingerprints are asserted in the tests, so they double as
machine-checked documentation of which group each entry is.

Built for training sweeps over every group of a given order:

    for entry in all_groups(max_order=19):
        group = entry.build()
        ...
"""

from collections.abc import Callable
from dataclasses import dataclass

from same_character_table_interp.groups.generators import GroupGenerators
from same_character_table_interp.groups.group import FiniteGroup
from same_character_table_interp.groups.presentations import build_group, from_presentation


@dataclass(frozen=True)
class GroupEntry:
    gap_id: tuple[int, int]  # (order, index) in the GAP SmallGroups library
    name: str  # human-readable structure, e.g. "D4", "C2 x C2"
    aliases: tuple[str, ...]
    build: Callable[[], FiniteGroup]
    element_orders: tuple[int, ...]  # sorted element-order spectrum (fingerprint)
    # Supplementary fingerprints, only set where the element-order spectrum is
    # not enough to pin the group (the order-16 collisions).
    center_orders: tuple[int, ...] | None = None  # element-order spectrum of Z(G)
    square_count: int | None = None  # |{ g^2 : g in G }|

    @property
    def order(self) -> int:
        return self.gap_id[0]


# --- builder shorthands ------------------------------------------------------

_C = GroupGenerators.cyclic_group
_S = GroupGenerators.symmetric_group
_PROD = GroupGenerators.direct_product


def _dihedral(n: int) -> FiniteGroup:
    # <a, b | a^n, b^2, (ab)^2> -- dihedral group of order 2n.
    return from_presentation("ab", ["a" * n, "bb", "abab"])


def _dicyclic(n: int) -> FiniteGroup:
    # <a, b | a^2n, b^2 = a^n, b a b^-1 = a^-1> -- order 4n. Dic_2 = Q8.
    return from_presentation("ab", ["a" * (2 * n), "bb" + "A" * n, "baBa"])


def _elementary_abelian_2(k: int) -> FiniteGroup:
    group = _C(2)
    for _ in range(k - 1):
        group = _PROD(group, _C(2))
    return group


def _quaternion8() -> FiniteGroup:
    return from_presentation("ab", ["aaaa", "bbAA", "baBa"])


def _alternating4() -> FiniteGroup:
    # <a, b | a^2, b^3, (ab)^3> = A4.
    return from_presentation("ab", ["aa", "bbb", "ababab"])


def _generalized_dihedral_c3xc3() -> FiniteGroup:
    # (C3 x C3) : C2 with C2 inverting -- generalized dihedral of C3 x C3.
    base = _PROD(_C(3), _C(3))
    c2 = _C(2)
    identity = c2.elements[0]
    return GroupGenerators.semidirect_product(
        base, c2, lambda h, n: n if h == identity else base.get_inverse(n)
    )


# --- order-16 specific constructions (presentations from the standard list) ---


def _modular16() -> FiniteGroup:
    # M4(2) = <a, b | a^8, b^2, b a b^-1 = a^5>.
    return from_presentation("ab", ["aaaaaaaa", "bb", "baB" + "A" * 5])


def _semidihedral16() -> FiniteGroup:
    # SD16 = <a, b | a^8, b^2, b a b^-1 = a^3>.
    return from_presentation("ab", ["aaaaaaaa", "bb", "baB" + "A" * 3])


def _c4_semidirect_c4() -> FiniteGroup:
    # SmallGroup(16,4) = <a, b | a^4, b^4, b a b^-1 = a^-1>.
    return from_presentation("ab", ["aaaa", "bbbb", "baBa"])


def _k4_semidirect_c4() -> FiniteGroup:
    # SmallGroup(16,3) = (C2 x C2) : C4, the C4 acting by swapping the two
    # involutions (so a^2 acts trivially): <a,b,c | a^4, b^2, c^2, [b,c],
    # a b a^-1 = c, a c a^-1 = b>.
    return from_presentation("abc", ["aaaa", "bb", "cc", "bcBC", "abAC", "acAB"])


def _pauli_group() -> FiniteGroup:
    # SmallGroup(16,13), central product C4 o D4: <a,b,c | a^4, b^4, c^2,
    # c b c^-1 = b^-1, a^2 = b^2, [a,b], [a,c]> with a central of order 4.
    return from_presentation("abc", ["aaaa", "bbbb", "cc", "cbCb", "aaBB", "abAB", "acAC"])


# --- the catalog -------------------------------------------------------------

_ENTRIES: list[GroupEntry] = [
    GroupEntry((1, 1), "C1", ("C1",), lambda: _C(1), (1,)),
    GroupEntry((2, 1), "C2", ("C2",), lambda: _C(2), (1, 2)),
    GroupEntry((3, 1), "C3", ("C3",), lambda: _C(3), (1, 3, 3)),
    GroupEntry((4, 1), "C4", ("C4",), lambda: _C(4), (1, 2, 4, 4)),
    GroupEntry((4, 2), "C2 x C2", ("V4", "C2xC2"), lambda: _PROD(_C(2), _C(2)), (1, 2, 2, 2)),
    GroupEntry((5, 1), "C5", ("C5",), lambda: _C(5), (1, 5, 5, 5, 5)),
    GroupEntry((6, 1), "S3", ("S3", "D3"), lambda: _S(3), (1, 2, 2, 2, 3, 3)),
    GroupEntry((6, 2), "C6", ("C6",), lambda: _C(6), (1, 2, 3, 3, 6, 6)),
    GroupEntry((7, 1), "C7", ("C7",), lambda: _C(7), (1, 7, 7, 7, 7, 7, 7)),
    GroupEntry((8, 1), "C8", ("C8",), lambda: _C(8), (1, 2, 4, 4, 8, 8, 8, 8)),
    GroupEntry(
        (8, 2), "C4 x C2", ("C4xC2",), lambda: _PROD(_C(4), _C(2)), (1, 2, 2, 2, 4, 4, 4, 4)
    ),
    GroupEntry((8, 3), "D4", ("D4",), lambda: _dihedral(4), (1, 2, 2, 2, 2, 2, 4, 4)),
    GroupEntry((8, 4), "Q8", ("Q8",), _quaternion8, (1, 2, 4, 4, 4, 4, 4, 4)),
    GroupEntry(
        (8, 5),
        "C2 x C2 x C2",
        ("C2^3",),
        lambda: _elementary_abelian_2(3),
        (1, 2, 2, 2, 2, 2, 2, 2),
    ),
    GroupEntry((9, 1), "C9", ("C9",), lambda: _C(9), (1, 3, 3, 9, 9, 9, 9, 9, 9)),
    GroupEntry(
        (9, 2), "C3 x C3", ("C3xC3",), lambda: _PROD(_C(3), _C(3)), (1, 3, 3, 3, 3, 3, 3, 3, 3)
    ),
    GroupEntry((10, 1), "D5", ("D5",), lambda: _dihedral(5), (1, 2, 2, 2, 2, 2, 5, 5, 5, 5)),
    GroupEntry((10, 2), "C10", ("C10",), lambda: _C(10), (1, 2, 5, 5, 5, 5, 10, 10, 10, 10)),
    GroupEntry((11, 1), "C11", ("C11",), lambda: _C(11), (1,) + (11,) * 10),
    GroupEntry(
        (12, 1), "Dic3", ("Dic3", "Q12"), lambda: _dicyclic(3), (1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 6, 6)
    ),
    GroupEntry((12, 2), "C12", ("C12",), lambda: _C(12), (1, 2, 3, 3, 4, 4, 6, 6, 12, 12, 12, 12)),
    GroupEntry((12, 3), "A4", ("A4",), _alternating4, (1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)),
    GroupEntry((12, 4), "D6", ("D6",), lambda: _dihedral(6), (1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 6, 6)),
    GroupEntry(
        (12, 5),
        "C6 x C2",
        ("C6xC2",),
        lambda: _PROD(_C(6), _C(2)),
        (1, 2, 2, 2, 3, 3, 6, 6, 6, 6, 6, 6),
    ),
    GroupEntry((13, 1), "C13", ("C13",), lambda: _C(13), (1,) + (13,) * 12),
    GroupEntry(
        (14, 1), "D7", ("D7",), lambda: _dihedral(7), (1, 2, 2, 2, 2, 2, 2, 2, 7, 7, 7, 7, 7, 7)
    ),
    GroupEntry(
        (14, 2), "C14", ("C14",), lambda: _C(14), (1, 2, 7, 7, 7, 7, 7, 7, 14, 14, 14, 14, 14, 14)
    ),
    GroupEntry(
        (15, 1),
        "C15",
        ("C15",),
        lambda: _C(15),
        (1, 3, 3, 5, 5, 5, 5, 15, 15, 15, 15, 15, 15, 15, 15),
    ),
    GroupEntry((17, 1), "C17", ("C17",), lambda: _C(17), (1,) + (17,) * 16),
    GroupEntry(
        (18, 1),
        "D9",
        ("D9",),
        lambda: _dihedral(9),
        (1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 9, 9, 9, 9, 9, 9),
    ),
    GroupEntry(
        (18, 2),
        "C18",
        ("C18",),
        lambda: _C(18),
        (1, 2, 3, 3, 6, 6, 9, 9, 9, 9, 9, 9, 18, 18, 18, 18, 18, 18),
    ),
    GroupEntry(
        (18, 3),
        "C3 x S3",
        ("C3xS3",),
        lambda: _PROD(_C(3), _S(3)),
        (1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6),
    ),
    GroupEntry(
        (18, 4),
        "(C3 x C3) : C2",
        ("Dih(C3xC3)",),
        _generalized_dihedral_c3xc3,
        (1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3),
    ),
    GroupEntry(
        (18, 5),
        "C6 x C3",
        ("C6xC3",),
        lambda: _PROD(_C(6), _C(3)),
        (1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6),
    ),
    GroupEntry((19, 1), "C19", ("C19",), lambda: _C(19), (1,) + (19,) * 18),
    # --- order 16 (GAP indexing from the standard SmallGroups list) ---
    GroupEntry((16, 1), "C16", ("C16",), lambda: _C(16), (1, 2, 4, 4, 8, 8, 8, 8) + (16,) * 8),
    GroupEntry(
        (16, 2), "C4 x C4", ("C4xC4",), lambda: _PROD(_C(4), _C(4)), (1, 2, 2, 2) + (4,) * 12
    ),
    GroupEntry(
        (16, 3),
        "(C2 x C2) : C4",
        ("K4:C4",),
        _k4_semidirect_c4,
        (1,) + (2,) * 7 + (4,) * 8,
        center_orders=(1, 2, 2, 2),
    ),
    GroupEntry(
        (16, 4), "C4 : C4", ("C4:C4",), _c4_semidirect_c4, (1, 2, 2, 2) + (4,) * 12, square_count=3
    ),
    GroupEntry(
        (16, 5),
        "C8 x C2",
        ("C8xC2",),
        lambda: _PROD(_C(8), _C(2)),
        (1, 2, 2, 2, 4, 4, 4, 4) + (8,) * 8,
    ),
    GroupEntry((16, 6), "M4(2)", ("M16",), _modular16, (1, 2, 2, 2, 4, 4, 4, 4) + (8,) * 8),
    GroupEntry((16, 7), "D8", ("D8",), lambda: _dihedral(8), (1,) + (2,) * 9 + (4, 4) + (8,) * 4),
    GroupEntry(
        (16, 8), "SD16", ("SD16", "QD16"), _semidihedral16, (1,) + (2,) * 5 + (4,) * 6 + (8,) * 4
    ),
    GroupEntry((16, 9), "Q16", ("Q16",), lambda: _dicyclic(4), (1, 2) + (4,) * 10 + (8,) * 4),
    GroupEntry(
        (16, 10),
        "C4 x C2 x C2",
        ("C4xC2xC2",),
        lambda: _PROD(_PROD(_C(4), _C(2)), _C(2)),
        (1,) + (2,) * 7 + (4,) * 8,
    ),
    GroupEntry(
        (16, 11),
        "D4 x C2",
        ("D4xC2",),
        lambda: _PROD(_dihedral(4), _C(2)),
        (1,) + (2,) * 11 + (4,) * 4,
    ),
    GroupEntry(
        (16, 12),
        "Q8 x C2",
        ("Q8xC2",),
        lambda: _PROD(_quaternion8(), _C(2)),
        (1, 2, 2, 2) + (4,) * 12,
        square_count=2,
    ),
    GroupEntry(
        (16, 13),
        "C4 o D4 (Pauli)",
        ("Pauli",),
        _pauli_group,
        (1,) + (2,) * 7 + (4,) * 8,
        center_orders=(1, 2, 4, 4),
    ),
    GroupEntry((16, 14), "C2^4", ("C2^4",), lambda: _elementary_abelian_2(4), (1,) + (2,) * 15),
]

CATALOG: dict[tuple[int, int], GroupEntry] = {e.gap_id: e for e in _ENTRIES}


def all_groups(max_order: int = 19) -> list[GroupEntry]:
    """Catalog entries with order <= ``max_order``, sorted by GAP id.

    Returns entries (lazy builders), so call ``entry.build()`` to construct the
    group only when needed.
    """
    return [e for gid, e in sorted(CATALOG.items()) if gid[0] <= max_order]


def build_from_id(gap_id: tuple[int, int] | str) -> FiniteGroup:
    """Build a group from its GAP id, given as a ``(order, index)`` tuple or an
    ``"order,index"`` string."""
    if isinstance(gap_id, str):
        order_str, index_str = gap_id.split(",")
        key = (int(order_str), int(index_str))
    else:
        key = (gap_id[0], gap_id[1])

    if key not in CATALOG:
        raise KeyError(f"No group with GAP id {key} in the catalog")
    return CATALOG[key].build()


def resolve_group(spec: str) -> FiniteGroup:
    """Resolve a config/CLI group spec into a FiniteGroup.

    Accepts a GAP id (``"16,3"``), a catalog alias (``"A4"``, ``"Q8"``, ``"D8"``),
    or a family spec for the parametric families (``"C8"``, ``"S5"``, ``"D7"``).
    Catalog entries take priority; anything else falls back to ``build_group``.
    """
    spec = spec.strip()
    if "," in spec:
        return build_from_id(spec)
    for entry in CATALOG.values():
        if spec in entry.aliases:
            return entry.build()
    return build_group(spec)
