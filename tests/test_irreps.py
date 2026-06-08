import numpy as np
import pytest

from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.representations.characters import compute_character_table
from finite_group_interp.representations.core import Representation
from finite_group_interp.representations.irreps import Irrep, _extract_one, extract_irreps


@pytest.mark.parametrize("name", ["C8", "C12", "S3"])
def test_extracted_irreps_are_valid_homomorphisms(name):
    group = resolve_group(name)
    irreps = extract_irreps(group)
    table, classes = compute_character_table(group)
    el_to_class = {el: i for i, cls in enumerate(classes) for el in cls}

    assert len(irreps) == table.shape[0]
    assert all(isinstance(irr, Irrep) for irr in irreps)
    for irr in irreps:
        d = irr.dimension
        for g in group.elements:
            for h in group.elements:
                gh = group.multiply(g, h)
                assert np.allclose(irr.matrices[g] @ irr.matrices[h], irr.matrices[gh], atol=1e-7)
            assert np.allclose(irr.matrices[g].conj().T @ irr.matrices[g], np.eye(d), atol=1e-7)
            chi = table[irr.table_index, el_to_class[g]]
            assert np.isclose(np.trace(irr.matrices[g]), chi, atol=1e-6)
        inner = sum(abs(irr.character[g]) ** 2 for g in group.elements) / group.order
        assert np.isclose(inner, 1.0, atol=1e-6)


def test_irrep_dimensions_obey_the_regular_rep_theorems():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    # sum of d_i^2 = |G|; number of irreps = number of conjugacy classes
    assert sum(irr.dimension**2 for irr in irreps) == group.order
    _, classes = compute_character_table(group)
    assert len(irreps) == len(classes)


def test_cyclic_irrep_characters_match_analytic_oracle():
    # Oracle: C_n's irreducible characters are exactly {e^{2 pi i k j / n}}.
    n = 8
    group = resolve_group("C8")
    irreps = extract_irreps(group)
    gen = group.elements[1]
    vals = sorted(
        (round(irr.character[gen].real, 6), round(irr.character[gen].imag, 6)) for irr in irreps
    )
    roots = sorted(
        (round(np.cos(2 * np.pi * k / n), 6), round(np.sin(2 * np.pi * k / n), 6)) for k in range(n)
    )
    assert vals == roots


def test_s3_has_the_known_two_dim_irrep():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    dims = sorted(irr.dimension for irr in irreps)
    assert dims == [1, 1, 2]  # trivial, sign, standard
    two = next(irr for irr in irreps if irr.dimension == 2)
    # standard irrep character: 2 on identity, 0 on transpositions, -1 on 3-cycles
    chars = sorted(round(two.character[g].real, 5) for g in group.elements)
    assert chars == [-1.0, -1.0, 0.0, 0.0, 0.0, 2.0]


def test_extract_one_raises_on_degenerate_subspace():
    group = resolve_group("S3")
    regular = Representation.regular(group).map
    n = group.order
    # A zero projector yields an empty isotypic component -> dim 0 != d_i^2 -> raises.
    with pytest.raises(ValueError, match="extraction failed"):
        _extract_one(group, np.zeros((n, n), dtype=complex), regular, d_i=2, table_index=99)
