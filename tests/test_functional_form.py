from pathlib import Path

import numpy as np
import pytest

from same_character_table_interp.analysis.figures import plot_functional_form_fve
from same_character_table_interp.analysis.functional_form import (
    center_over_c,
    fit_logit_tensor,
    functional_form_fit,
    logit_tensor,
    representation_product_features,
)
from same_character_table_interp.groups.catalog import resolve_group
from same_character_table_interp.representations.irreps import extract_irreps
from same_character_table_interp.representations.projectors import real_isotypic_blocks
from same_character_table_interp.training.config import ExperimentConfig, GrokkingConfig
from same_character_table_interp.training.trainer import build_model

C113_RUN = Path("runs/2026-06-04/2026-06-04_050749_grok-C113")


def test_center_over_c_zeroes_the_c_mean():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(6, 6, 6))
    Lc = center_over_c(L)
    assert np.allclose(Lc.mean(axis=2), 0.0, atol=1e-12)
    assert np.allclose(center_over_c(Lc), Lc, atol=1e-12)  # idempotent


def test_logit_tensor_shape_and_rawness():
    group = resolve_group("C8")
    config = GrokkingConfig(experiment=ExperimentConfig(name="t", seed=0, use_wandb=False))
    model = build_model(config, group)
    model.eval()
    L = logit_tensor(model, group)
    assert L.shape == (8, 8, 8)  # [a, b, c]
    # logit_tensor returns RAW logits (centering is the fit's job), so the
    # c-mean is generally nonzero for an untrained model.
    assert not np.allclose(L.mean(axis=2), 0.0)


def test_feature_shapes_for_a_two_dim_irrep():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    x_full, x_trace = representation_product_features(irreps, group, keep=[two])
    n = group.order
    assert x_full.shape == (n**3, 8)  # real+imag of d^2=4 entries -> 8 cols
    assert x_trace.shape == (n**3, 2)  # real+imag of the trace -> 2 cols


def test_nontrivial_irrep_features_are_mean_zero_over_c():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    x_full, _ = representation_product_features(irreps, group, keep=[two])
    n = group.order
    cols = x_full.reshape(n, n, n, -1)
    assert np.allclose(cols.mean(axis=2), 0.0, atol=1e-10)  # group-sum of nontrivial irrep = 0


def test_features_keep_out_of_range_raises():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    with pytest.raises(ValueError, match="out of range|irrep"):
        representation_product_features(irreps, group, keep=[99])


def _planted_logits_from_features(x, group, seed):
    """Build a logit tensor that IS a random linear combo of feature columns."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=(x.shape[1],))
    return (x @ beta).reshape(group.order, group.order, group.order)


def test_planted_full_form_recovers_fve_one():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    x_full, _ = representation_product_features(irreps, group, keep=[two])
    L = _planted_logits_from_features(x_full, group, seed=1)
    result = fit_logit_tensor(L, group, irreps, keep=[two])
    assert result.cumulative_full > 0.999
    assert result.per_irrep_full[two] > 0.999


def test_trace_fve_never_exceeds_full_and_gap_nonneg():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    x_full, _ = representation_product_features(irreps, group, keep=[two])
    L = _planted_logits_from_features(x_full, group, seed=2)
    result = fit_logit_tensor(L, group, irreps, keep=[two])
    assert result.cumulative_trace <= result.cumulative_full + 1e-9
    assert result.gap >= -1e-9


def test_one_dim_group_has_zero_gap_even_on_full_planted_data():
    # C8: all irreps 1-dim -> matrix element IS the trace -> gap must vanish.
    group = resolve_group("C8")
    irreps = extract_irreps(group)
    keep = [1, 2]
    x_full, _ = representation_product_features(irreps, group, keep=keep)
    L = _planted_logits_from_features(x_full, group, seed=3)
    result = fit_logit_tensor(L, group, irreps, keep=keep)
    assert abs(result.gap) < 1e-6


def test_character_only_logits_have_zero_gap_for_two_dim_group():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    _, x_trace = representation_product_features(irreps, group, keep=[two])
    L = _planted_logits_from_features(x_trace, group, seed=4)
    result = fit_logit_tensor(L, group, irreps, keep=[two])
    assert result.cumulative_trace > 0.999
    assert abs(result.gap) < 1e-3


def test_fit_keep_out_of_range_raises():
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    L = np.zeros((group.order,) * 3)
    with pytest.raises(ValueError, match="out of range|irrep"):
        fit_logit_tensor(L, group, irreps, keep=[99])


def test_functional_form_figure_renders(tmp_path):
    group = resolve_group("S3")
    irreps = extract_irreps(group)
    two = next(i for i, irr in enumerate(irreps) if irr.dimension == 2)
    x_full, _ = representation_product_features(irreps, group, keep=[two])
    L = _planted_logits_from_features(x_full, group, seed=5)
    result = fit_logit_tensor(L, group, irreps, keep=[two])
    out = tmp_path / "ff.png"
    plot_functional_form_fve(result, out, title="S3 functional form")
    assert out.is_file() and out.stat().st_size > 0


def test_block_keep_maps_to_irrep_rows_via_irrep_indices():
    # The orchestrator's `keep` indexes blocks; functional_form needs table rows.
    group = resolve_group("S3")
    blocks = real_isotypic_blocks(group)
    irreps = extract_irreps(group)
    keep_blocks = list(range(len(blocks)))
    rows = sorted({idx for i in keep_blocks for idx in blocks[i].irrep_indices})
    assert rows == list(range(len(irreps)))


@pytest.mark.skipif(not C113_RUN.exists(), reason="local C113 grokking run not present")
def test_c113_functional_form_calibration():
    from same_character_table_interp.analysis.loading import load_run
    from same_character_table_interp.analysis.irrep_metrics import (
        isotypic_energy,
        weight_as_functions,
    )
    from same_character_table_interp.representations.projectors import real_isotypic_blocks

    run = load_run(C113_RUN)
    group = run.checkpoint.group
    n = group.order
    blocks = real_isotypic_blocks(group)
    spectrum = isotypic_energy(weight_as_functions(run.checkpoint.model, "W_E", n), blocks)
    keep_blocks = [i for i, f in enumerate(spectrum.fractions) if f > 2 * spectrum.baseline[i]]
    keep_rows = sorted({idx for i in keep_blocks for idx in blocks[i].irrep_indices})

    irreps = extract_irreps(group)
    result = functional_form_fit(run.checkpoint.model, group, irreps, keep_rows)

    # The load-bearing calibration property: the gap (full-matrix minus
    # trace-only FVE) is EXACTLY zero, because C113's irreps are 1-dimensional
    # (the matrix element IS its own trace). This proves the instrument will not
    # fabricate sub-character structure when later pointed at genuine 2-dim
    # irreps -- a nonzero gap there will be real signal, not an artifact.
    assert abs(result.gap) < 1e-3

    # The homomorphism form rho(a)rho(b)rho(c)^-1 is the single dominant
    # structure in the logits, but only ~0.55 -- NOT the >0.9 a naive reading of
    # Nanda would predict. This is a real finding, not a shortfall: the model
    # computes the sum a+b (a separate diagnostic puts that at 98%), but its
    # readout comparison is only ~56% translation-invariant; the rest is the
    # compound-angle image term + harmonics, all at the same 3 frequencies.
    # See docs/functional-form-c113.md for the full decomposition. The band is
    # pinned tight because this is a fixed, committed checkpoint.
    assert 0.50 < result.cumulative_full < 0.60
