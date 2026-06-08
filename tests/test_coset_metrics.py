import numpy as np
import torch

from finite_group_interp.analysis.coset_metrics import (
    CosetResult,
    _all_pairs_resid,
    _irrep_features,
    _random_subspace,
    _target_elements,
    ablate_coset_direction,
    coset_analysis,
    coset_labels,
    coset_probe_suite,
    coset_subspace,
    fit_linear_probe,
    random_partition_null,
)
from finite_group_interp.groups.catalog import resolve_group
from finite_group_interp.representations.irreps import extract_irreps
from finite_group_interp.training.config import ExperimentConfig, GrokkingConfig
from finite_group_interp.training.trainer import build_model


def test_coset_labels_match_left_coset_partition():
    g = resolve_group("S3")
    H = next(h for h in g.subgroups() if len(h) == 3)  # normal C3, 2 cosets
    y = coset_labels(g, H, target="a")
    assert y.shape == (g.order**2,)
    assert set(np.unique(y).tolist()) == {0, 1}
    cosets = g.left_cosets(H)
    coset_of = {e: ci for ci, c in enumerate(cosets) for e in c}
    for k in (0, 5, 17, 35):
        a_el = g.elements[k // g.order]
        assert y[k] == coset_of[a_el]


def test_coset_labels_target_ab_uses_the_product():
    g = resolve_group("S3")
    H = next(h for h in g.subgroups() if len(h) == 3)
    y = coset_labels(g, H, target="ab")
    cosets = g.left_cosets(H)
    coset_of = {e: ci for ci, c in enumerate(cosets) for e in c}
    for k in (0, 5, 17, 35):
        a = g.elements[k // g.order]
        b = g.elements[k % g.order]
        assert y[k] == coset_of[g.multiply(a, b)]


def test_probe_recovers_linearly_separable_labels():
    # 3 classes planted as mean offsets along distinct directions -> ~perfect.
    rng = np.random.default_rng(0)
    d, per = 16, 200
    centers = rng.normal(size=(3, d)) * 5.0
    X = np.concatenate([centers[c] + rng.normal(size=(per, d)) for c in range(3)])
    y = np.repeat([0, 1, 2], per)
    acc = fit_linear_probe(X, y, seed=0)
    assert acc > 0.95


def test_probe_on_noise_is_near_chance():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 16))
    y = rng.integers(0, 3, size=600)  # labels unrelated to X
    acc = fit_linear_probe(X, y, seed=0)
    assert acc < 0.55  # ~1/3 chance, generous upper bound


def test_probe_is_deterministic():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 16))
    y = rng.integers(0, 2, size=300)
    assert fit_linear_probe(X, y, seed=7) == fit_linear_probe(X, y, seed=7)


def test_random_partition_null_is_near_chance():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(size=(150, 8)), 5 + rng.normal(size=(150, 8))])
    y = np.repeat([0, 1], 150)
    mean, std = random_partition_null(X, y, draws=3, seed=0)
    assert mean < 0.65  # ~0.5 chance for balanced 2-way


def test_planted_coset_structure_beats_both_controls():
    # Coset(ab) of S3's C3 (2 cosets) planted as a mean offset along a random
    # direction. Control irrep set = the TRIVIAL irrep only (constant matrix
    # element -> provably can't decode any label -> a clean floor). We must NOT
    # use the 2-dim/sign irrep: C3 is normal so its coset = the sign, which IS
    # decodable from those irreps -- that case is the next test.
    g = resolve_group("S3")
    H = next(h for h in g.subgroups() if len(h) == 3)
    y = coset_labels(g, H, target="ab")
    rng = np.random.default_rng(0)
    d = 16
    direction = rng.normal(size=d)
    X = np.array([(lbl * 6.0) * direction for lbl in y]) + rng.normal(size=(len(y), d))
    irreps = extract_irreps(g)
    trivial = next(
        i
        for i, irr in enumerate(irreps)
        if all(abs(irr.character[e] - 1) < 1e-6 for e in g.elements)
    )
    pa, nm, ns, ir, k = coset_probe_suite(X, g, H, "ab", [trivial], seed=0)
    assert pa > 0.95
    assert pa - nm > 0.3  # beats random-partition null
    assert pa - ir > 0.3  # beats irrep-feature control (trivial irrep can't decode)
    assert k == 2


def test_pure_irrep_activations_do_not_beat_irrep_control():
    # If activations ARE the kept-irrep features, coset decodability is fully
    # explained by irreps: probe acc ≈ irrep-ref acc (excess ≈ 0).
    g = resolve_group("S3")
    H = next(h for h in g.subgroups() if len(h) == 3)
    irreps = extract_irreps(g)
    keep = [i for i, irr in enumerate(irreps) if irr.dimension == 2]
    X = _irrep_features(g, keep, _target_elements(g, "ab"))
    pa, nm, ns, ir, k = coset_probe_suite(X, g, H, "ab", keep, seed=0)
    assert abs(pa - ir) < 0.1


def test_coset_subspace_rank_is_at_most_k_minus_one():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 100)  # k=3
    X = np.array([[lbl, 0, 0] for lbl in y], dtype=float) + rng.normal(scale=0.01, size=(300, 3))
    B = coset_subspace(X, y)
    assert B.shape[0] <= 2  # rank <= k-1
    assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1e-6)  # orthonormal rows


def test_coset_ablation_is_nondestructive_and_well_formed():
    # Structural test on an UNTRAINED model: ablation is non-destructive and
    # returns finite, well-formed deltas. We do NOT assert "coset ablation hurts
    # more than random" here -- that is a scientific property that holds only for
    # a trained model (item 6); on an untrained model the coset subspace carries
    # no real signal, so that comparison is seed-noise. The ablation's actual
    # job (removing the coset signal) is calibrated on planted data below.
    g = resolve_group("S3")
    H = next(h for h in g.subgroups() if len(h) == 3)
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=0, use_wandb=False))
    torch.manual_seed(0)  # build_model draws weights from the global RNG; pin for determinism
    model = build_model(config, g)
    model.eval()
    resid, targets = _all_pairs_resid(model, g)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = ablate_coset_direction(model, g, H, "ab", resid, targets, seed=0)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])  # weights untouched (only W_U is read)
    assert all(np.isfinite(out[key]) for key in out)


def test_ablating_coset_subspace_destroys_decodability():
    # The ablation's real job: removing the class-mean (coset) subspace must
    # erase decodability, while removing a random subspace of the SAME rank must
    # not. Synthetic with large N for a stable probe (S3 has only 36 pairs, too
    # few test points); this tests the subspace-removal mechanics directly.
    rng = np.random.default_rng(0)
    d, per = 16, 250
    direction = rng.normal(size=d)
    y = np.repeat([0, 1], per)
    X = np.array([(lbl * 6.0) * direction for lbl in y]) + rng.normal(size=(2 * per, d))
    b = coset_subspace(X, y)
    x_abl = X - X @ (b.T @ b)  # remove the coset subspace
    # seed != 0: _random_subspace(seed=0) would replay default_rng(0)'s first
    # draw -- the very vector `direction` was drawn from -- and collide with the
    # signal. An independent seed gives a genuinely random direction.
    q = _random_subspace(b.shape[0], d, seed=12345)
    x_rand = X - X @ (q.T @ q)  # remove a random same-rank subspace
    assert fit_linear_probe(X, y, seed=0) > 0.95  # signal present
    assert fit_linear_probe(x_abl, y, seed=0) < 0.65  # coset removal -> ~chance
    assert fit_linear_probe(x_rand, y, seed=0) > 0.9  # random removal keeps signal


def test_coset_analysis_returns_a_result_per_proper_subgroup_and_target():
    g = resolve_group("S3")
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=0, use_wandb=False))
    model = build_model(config, g)
    model.eval()
    results = coset_analysis(model, g, keep_irreps=[], seed=0)
    proper = [h for h in g.subgroups() if 1 < len(h) < g.order]
    assert len(results) == len(proper) * 3  # {a, b, ab} per proper subgroup
    assert all(isinstance(r, CosetResult) for r in results)
    assert all(np.isnan(r.irrep_ref_acc) for r in results)  # keep_irreps=[] -> skipped


def test_coset_analysis_on_prime_group_is_empty():
    g = resolve_group("C7")  # prime: no proper subgroups; stands in for C113
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=0, use_wandb=False))
    model = build_model(config, g)
    model.eval()
    assert coset_analysis(model, g, keep_irreps=[], seed=0) == []
