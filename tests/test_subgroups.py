from same_character_table_interp.groups.catalog import resolve_group


def _sizes(subs):
    return sorted(len(s) for s in subs)


def test_c8_subgroup_lattice():
    g = resolve_group("C8")
    subs = g.subgroups()
    assert _sizes(subs) == [1, 2, 4, 8]  # cyclic: one subgroup per divisor of 8


def test_s3_subgroup_lattice():
    g = resolve_group("S3")
    subs = g.subgroups()
    assert _sizes(subs) == [1, 2, 2, 2, 3, 6]  # {e}, three reflections C2, one C3, S3


def test_q8_subgroup_lattice():
    g = resolve_group("Q8")
    subs = g.subgroups()
    assert _sizes(subs) == [1, 2, 4, 4, 4, 8]  # {e}, unique order-2 center, three C4, Q8


def test_every_subgroup_contains_identity_and_is_closed():
    g = resolve_group("S3")
    identity = g.elements[g._check_identity()]
    for H in g.subgroups():
        assert identity in H
        Hset = set(H)
        for x in H:
            for y in H:
                assert g.multiply(x, y) in Hset


def _subgroup_of_size(g, size):
    return next(H for H in g.subgroups() if len(H) == size)


def test_left_cosets_partition_the_group():
    g = resolve_group("S3")
    H = _subgroup_of_size(g, 2)  # a reflection subgroup, index 3
    cosets = g.left_cosets(H)
    assert len(cosets) == 3  # |G|/|H|
    assert all(len(c) == 2 for c in cosets)
    flat = [e for c in cosets for e in c]
    assert sorted(flat, key=lambda e: g.el_to_inx(e)) == sorted(
        g.elements, key=lambda e: g.el_to_inx(e)
    )


def test_normality():
    g = resolve_group("S3")
    c3 = _subgroup_of_size(g, 3)
    c2 = _subgroup_of_size(g, 2)
    assert g.is_normal(c3) is True  # the rotation subgroup (index 2) is normal
    assert g.is_normal(c2) is False  # reflection subgroup is not normal in S3


def test_center():
    s3 = resolve_group("S3")
    assert s3.center() == [s3.elements[s3._check_identity()]]  # trivial center
    q8 = resolve_group("Q8")
    assert len(q8.center()) == 2  # {1, -1}
