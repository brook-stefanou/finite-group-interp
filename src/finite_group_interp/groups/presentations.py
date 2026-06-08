"""Build finite groups from a presentation <generators | relations>.

Uses Todd-Coxeter coset enumeration over the trivial subgroup, so the
enumerated cosets are exactly the group elements. The general word problem is
undecidable, so a ``max_order`` cap guards against presentations of infinite
(or impractically large) groups: enumeration raises rather than looping forever.

Word convention: generators are lowercase letters; the matching uppercase letter
denotes the inverse. So ``"abA"`` means ``a * b * a^-1``. A relation string is a
word that equals the identity, e.g. ``"aaa"`` for ``a^3 = e``.
"""

import re
from collections import deque
from collections.abc import Callable

import numpy as np

from finite_group_interp.groups.group import Element, FiniteGroup


def _parse_symbols(generators: str) -> dict[str, int]:
    """Map each generator/inverse character to an integer symbol.

    Generator ``i`` gets forward symbol ``2i`` and inverse symbol ``2i + 1``.
    """
    symbols = {}
    for i, gen in enumerate(generators):
        if not gen.islower():
            raise ValueError(f"Generators must be lowercase letters, got {gen!r}")
        symbols[gen] = 2 * i
        symbols[gen.upper()] = 2 * i + 1
    return symbols


def _inv(symbol: int) -> int:
    # Forward/inverse symbols are paired (2i, 2i+1); flip the low bit.
    return symbol ^ 1


def from_presentation(generators: str, relations: list[str], max_order: int = 2048) -> FiniteGroup:
    """Construct the finite group presented by ``generators`` and ``relations``."""
    symbol_of = _parse_symbols(generators)
    n_symbols = 2 * len(generators)
    relators = [[symbol_of[ch] for ch in word] for word in relations]

    # Coset table: table[c] maps each symbol to the coset reached, or 0 if
    # undefined. Cosets are numbered from 1 (coset 1 = identity); 0 = undefined.
    table: list[list[int]] = [[0] * n_symbols, [0] * n_symbols]
    parent = [0, 1]  # union-find over cosets; parent[c] == c means c is live.

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def define(c: int, x: int) -> None:
        if len(table) - 1 >= max_order:
            raise ValueError(
                f"Coset enumeration exceeded max_order={max_order}; the group may "
                "be infinite or larger than expected."
            )
        d = len(table)
        table.append([0] * n_symbols)
        parent.append(d)
        table[c][x] = d
        table[d][_inv(x)] = c

    def union(a: int, b: int, queue: deque[int]) -> None:
        a, b = find(a), find(b)
        if a == b:
            return
        if a > b:
            a, b = b, a
        parent[b] = a
        queue.append(b)

    def coincidence(a: int, b: int) -> None:
        queue: deque[int] = deque()
        union(a, b, queue)
        while queue:
            dead = queue.popleft()
            for x in range(n_symbols):
                target = table[dead][x]
                if target == 0:
                    continue
                table[target][_inv(x)] = 0  # drop the back-edge into the dead coset
                live = find(dead)
                target_live = find(target)
                if table[live][x] != 0:
                    union(target_live, table[live][x], queue)
                elif table[target_live][_inv(x)] != 0:
                    union(live, table[target_live][_inv(x)], queue)
                else:
                    table[live][x] = target_live
                    table[target_live][_inv(x)] = live

    def scan_and_fill(start: int, word: list[int]) -> None:
        forward, i = start, 0
        backward, j = start, len(word) - 1
        while True:
            # Scan forward through defined entries.
            while i <= j and table[forward][word[i]] != 0:
                forward = table[forward][word[i]]
                i += 1
            if i > j:
                if forward != backward:
                    coincidence(forward, backward)
                return
            # Scan backward through defined entries.
            while j >= i and table[backward][_inv(word[j])] != 0:
                backward = table[backward][_inv(word[j])]
                j -= 1
            if j < i:
                coincidence(forward, backward)
                return
            if i == j:
                # Exactly one gap left: deduce the connecting edge.
                table[forward][word[i]] = backward
                table[backward][_inv(word[i])] = forward
                return
            # Gap wider than one: extend the forward frontier.
            define(forward, word[i])

    # Main HLT loop: scan every relator from every live coset, defining new
    # cosets as scans demand and merging via coincidences.
    alpha = 1
    while alpha < len(table):
        if find(alpha) == alpha:
            for relator in relators:
                scan_and_fill(alpha, relator)
                if find(alpha) != alpha:
                    break
        alpha += 1

    return _build_group(generators, table, find, n_symbols)


def _build_group(
    generators: str,
    table: list[list[int]],
    find: Callable[[int], int],
    n_symbols: int,
) -> FiniteGroup:
    # Collect live cosets and renumber them 0..m-1 (identity coset 1 -> index 0).
    live = [c for c in range(1, len(table)) if find(c) == c]
    index_of = {c: i for i, c in enumerate(live)}
    m = len(live)

    # Action of each symbol on the renumbered live cosets.
    action = np.full((m, n_symbols), -1, dtype=np.int64)
    for c in live:
        for x in range(n_symbols):
            target = table[c][x]
            if target == 0:
                raise ValueError(
                    "Presentation did not close to a complete table; it may need "
                    "more relations or define an infinite group."
                )
            action[index_of[c], x] = index_of[find(target)]

    # Shortest word (as symbol lists) reaching each element, via BFS from identity.
    identity = index_of[find(1)]
    word_of: list[list[int] | None] = [None] * m
    word_of[identity] = []
    bfs = deque([identity])
    while bfs:
        c = bfs.popleft()
        word_c = word_of[c]
        assert word_c is not None  # BFS only enqueues cosets whose word is set
        for x in range(n_symbols):
            nxt = action[c, x]
            if word_of[nxt] is None:
                word_of[nxt] = word_c + [x]
                bfs.append(nxt)

    # The action of a group on its cosets is transitive, so BFS must reach
    # every live coset; a gap here means the table above is not a group.
    words: list[list[int]] = []
    for i, w in enumerate(word_of):
        if w is None:
            raise ValueError(f"Coset {i} unreachable from identity; coset table is inconsistent")
        words.append(w)

    # Right-multiplication by element d = following d's word from coset c.
    cayley = np.zeros((m, m), dtype=np.int64)
    for c in range(m):
        for d in range(m):
            here = c
            for x in words[d]:
                here = action[here, x]
            cayley[c, d] = here

    elements: list[Element] = [_word_to_name(words[i], generators) for i in range(m)]
    return FiniteGroup(elements=elements, cayley_table=cayley)


def build_group(spec: str) -> FiniteGroup:
    """Build a group from a short spec like ``"C3"``, ``"S4"``, ``"D4"``, ``"Q8"``.

    Designed for config/CLI use. Each family dispatches to the most appropriate
    constructor: ``C``/``S`` keep their human-readable labels via the dedicated
    constructors, while ``D`` and ``Q`` are built from presentations. The Klein
    four-group is available as ``"D2"``.
    """
    # Local import avoids a module-level cycle (generators imports group only).
    from finite_group_interp.groups.generators import GroupGenerators

    match = re.fullmatch(r"([A-Za-z]+)(\d+)", spec.strip())
    if not match:
        raise ValueError(
            f"Unrecognised group spec {spec!r}; expected a family letter followed "
            "by an integer, e.g. 'C3', 'S4', 'D4', 'Q8'."
        )
    family, n = match.group(1).upper(), int(match.group(2))

    if family == "C":
        return GroupGenerators.cyclic_group(n)
    if family == "S":
        return GroupGenerators.symmetric_group(n)
    if family == "D":
        return from_presentation("ab", ["a" * n, "bb", "abab"])
    if family == "Q":
        if n != 8:
            raise ValueError("Only 'Q8' (the quaternion group) is defined in the Q family.")
        return from_presentation("ab", ["aaaa", "bbAA", "baBa"])
    if family == "DIC":
        # Dicyclic group of order 4n: <a, b | a^2n, b^2 = a^n, b a b^-1 = a^-1>.
        # Dic26 is Dic(104), the same-character-table partner of Dih(104) = D52.
        return from_presentation("ab", ["a" * (2 * n), "bb" + "A" * n, "baBa"])

    raise ValueError(f"Unknown group family {family!r} in spec {spec!r}.")


def _word_to_name(word: list[int], generators: str) -> str:
    if not word:
        return "e"
    chars: list[str] = []
    for symbol in word:
        gen = generators[symbol // 2]
        chars.append(gen if symbol % 2 == 0 else gen.upper())
    return "".join(chars)
