import numpy as np

from same_character_table_interp.groups.generators import GroupGenerators
from same_character_table_interp.representations.characters import compute_character_table
from same_character_table_interp.representations.projectors import (
    isotypic_projectors,
    real_isotypic_blocks,
)


# === Individual per-irrep projectors (S_3, all-real characters) ===


def test_s3_projectors_complete():
    # The isotypic components partition the whole space: sum_i P_i == I.
    group = GroupGenerators.symmetric_group(3)
    projectors = isotypic_projectors(group)

    total = sum(projectors)
    assert np.allclose(total, np.eye(group.order), atol=1e-7)


def test_s3_projectors_idempotent():
    # A projector satisfies P @ P == P.
    group = GroupGenerators.symmetric_group(3)
    for P in isotypic_projectors(group):
        assert np.allclose(P @ P, P, atol=1e-7)


def test_s3_projectors_mutually_orthogonal():
    # Distinct isotypic components do not overlap: P_i @ P_j == 0 for i != j.
    group = GroupGenerators.symmetric_group(3)
    projectors = isotypic_projectors(group)

    for i, P_i in enumerate(projectors):
        for j, P_j in enumerate(projectors):
            if i != j:
                assert np.allclose(P_i @ P_j, 0.0, atol=1e-7)


def test_s3_projector_trace_equals_dim_squared():
    # Irrep i appears d_i times in the regular rep, so its isotypic component
    # has dimension d_i^2, i.e. trace(P_i) == d_i^2.
    group = GroupGenerators.symmetric_group(3)
    table, _ = compute_character_table(group)
    projectors = isotypic_projectors(group)

    for i, P in enumerate(projectors):
        d_i = int(round(table[i, 0].real))
        assert np.isclose(np.trace(P).real, d_i**2, atol=1e-7)


def test_s3_projectors_are_real():
    # S_3 has a real character table, so every projector is real.
    group = GroupGenerators.symmetric_group(3)
    for P in isotypic_projectors(group):
        assert np.allclose(P.imag, 0.0, atol=1e-7)


# === Complex characters require conjugate-pair grouping (C_4) ===


def test_c4_individual_projectors_complete():
    group = GroupGenerators.cyclic_group(4)
    projectors = isotypic_projectors(group)

    total = sum(projectors)
    assert np.allclose(total, np.eye(group.order), atol=1e-7)


def test_c4_has_a_complex_individual_projector():
    # C_4 has complex irreps (characters are powers of i), so at least one
    # individual projector must have a non-zero imaginary part.
    group = GroupGenerators.cyclic_group(4)
    projectors = isotypic_projectors(group)

    has_complex = any(not np.allclose(P.imag, 0.0, atol=1e-7) for P in projectors)
    assert has_complex


def test_c4_real_blocks_are_real():
    # Grouping each complex irrep with its conjugate cancels the imaginary
    # parts, leaving a real projector for every block.
    group = GroupGenerators.cyclic_group(4)
    for block in real_isotypic_blocks(group):
        assert np.allclose(block.projector.imag, 0.0, atol=1e-7)


def test_c4_real_blocks_complete():
    group = GroupGenerators.cyclic_group(4)
    blocks = real_isotypic_blocks(group)

    total = sum(block.projector for block in blocks)
    assert np.allclose(total, np.eye(group.order), atol=1e-7)


def test_c4_real_blocks_count_and_traces():
    # C_4: trivial (real), sign (real), and one conjugate pair {chi, conj chi}.
    # That is 3 real blocks with subspace dimensions 1, 1, 2.
    group = GroupGenerators.cyclic_group(4)
    blocks = real_isotypic_blocks(group)

    assert len(blocks) == 3
    traces = sorted(round(np.trace(b.projector).real) for b in blocks)
    assert traces == [1, 1, 2]


def test_c4_real_blocks_idempotent_and_orthogonal():
    group = GroupGenerators.cyclic_group(4)
    blocks = real_isotypic_blocks(group)

    for i, b_i in enumerate(blocks):
        assert np.allclose(b_i.projector @ b_i.projector, b_i.projector, atol=1e-7)
        for j, b_j in enumerate(blocks):
            if i != j:
                assert np.allclose(b_i.projector @ b_j.projector, 0.0, atol=1e-7)


def test_c4_real_blocks_are_symmetric():
    # Real isotypic projectors of the (orthogonal) regular rep are symmetric.
    group = GroupGenerators.cyclic_group(4)
    for block in real_isotypic_blocks(group):
        assert np.allclose(block.projector, block.projector.T, atol=1e-7)


def test_c4_conjugate_pair_is_grouped():
    # Exactly one block should bundle two irreps (the conjugate pair); the
    # others wrap a single self-conjugate (real) irrep.
    group = GroupGenerators.cyclic_group(4)
    blocks = real_isotypic_blocks(group)

    pair_sizes = sorted(len(b.irrep_indices) for b in blocks)
    assert pair_sizes == [1, 1, 2]
