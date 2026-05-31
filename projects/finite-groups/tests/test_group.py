import numpy as np
import pytest

from finite_groups.group import FiniteGroup


def test_no_identity():
    # An invalid group (no identity) must be rejected at construction time.
    with pytest.raises(ValueError, match="No identity found"):
        FiniteGroup(
            elements=[1, 2, 3],
            cayley_table=np.array([[1, 2, 2], [1, 2, 0], [2, 1, 2]]),
        )


def test_multiple_inverses():
    # Non-unique inverses must be rejected at construction time.
    with pytest.raises(ValueError, match="Right inverses are not unique"):
        FiniteGroup(elements=["e", "f"], cayley_table=np.array([[0, 1], [1, 1]]))


def test_associativity_failure():
    elements = ["e", "a", "b", "c"]
    table = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    # Modify to break associativity
    table[1, 2] = 1

    with pytest.raises(
        ValueError, match=r"Associativity fails for triplet: \(a, a, b\)"
    ):
        FiniteGroup(elements=elements, cayley_table=table)


def test_index_out_of_range():
    # A Cayley table entry that is not a valid element index must be rejected.
    with pytest.raises(ValueError, match=r"valid indices"):
        FiniteGroup(
            elements=[1, 2, 3],
            cayley_table=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 7]]),
        )


def test_correct_identity_inverses():
    # A valid group constructs successfully and passes axiom validation.
    g = FiniteGroup(
        elements=["e", 2, 3],
        cayley_table=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]),
    )
    assert g._validate_group_axioms() is True
