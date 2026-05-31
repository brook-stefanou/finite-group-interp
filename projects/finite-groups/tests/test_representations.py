import numpy as np
import pytest

from finite_groups.generators import GroupGenerators
from finite_groups.representations.core import Representation


def test_regular_representation_is_valid():
    # The regular representation is a homomorphism by construction and must
    # pass validation.
    group = GroupGenerators.cyclic_group(3)
    rep = Representation.regular(group)

    assert rep.degree == group.order
    # Character of the regular rep is |G| at the identity.
    chi = rep.character()
    assert np.isclose(chi[group.elements[0]], group.order)


def test_rejects_incomplete_mapping():
    # Every group element must have a matrix.
    group = GroupGenerators.cyclic_group(3)
    full_map = Representation.regular(group).map
    incomplete = {g: m for g, m in full_map.items() if g != group.elements[-1]}

    with pytest.raises(ValueError, match="missing matrices"):
        Representation(group=group, map=incomplete)


def test_rejects_inconsistent_degree():
    # All matrices must be square and of the same degree.
    group = GroupGenerators.cyclic_group(3)
    bad_map = dict(Representation.regular(group).map)
    bad_map[group.elements[1]] = np.eye(2, dtype=complex)

    with pytest.raises(ValueError, match=r"\d+x\d+"):
        Representation(group=group, map=bad_map)


def test_rejects_non_homomorphism():
    # The defining property: rho(g) @ rho(h) == rho(g * h).
    group = GroupGenerators.cyclic_group(3)
    bad_map = dict(Representation.regular(group).map)
    # Corrupt a non-identity element so the homomorphism law breaks.
    bad_map[group.elements[1]] = np.eye(group.order, dtype=complex)

    with pytest.raises(ValueError, match="homomorphism"):
        Representation(group=group, map=bad_map)
