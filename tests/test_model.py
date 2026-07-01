import einops
import pytest
import torch

from same_character_table_interp.model import OneLayerTransformer

# Weight names the analysis layer relies on (the "contract").
CONTRACT = {"W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"}


def _model(use_mlp: bool = True, d_model: int = 8, n_heads: int = 2) -> OneLayerTransformer:
    return OneLayerTransformer(
        d_vocab_in=6,
        d_vocab_out=5,
        n_ctx=3,
        d_model=d_model,
        n_heads=n_heads,
        use_mlp=use_mlp,
        d_mlp=16,
        activation="relu",
    )


def test_parameter_shapes_match_contract():
    m = _model()
    d_head = 8 // 2
    assert m.W_E.shape == (6, 8)
    assert m.W_pos.shape == (3, 8)
    assert m.W_Q.shape == (2, 8, d_head)
    assert m.W_K.shape == (2, 8, d_head)
    assert m.W_V.shape == (2, 8, d_head)
    assert m.W_O.shape == (2, d_head, 8)
    assert m.W_in.shape == (8, 16)
    assert m.W_out.shape == (16, 8)
    assert m.W_U.shape == (8, 5)


def test_forward_output_shape():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))  # [batch, n_ctx], int64
    assert m(tokens).shape == (4, 3, 5)  # [batch, n_ctx, d_vocab_out]


def test_gradients_flow_to_every_parameter():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    m(tokens).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"


def test_no_mlp_omits_mlp_parameters_but_still_runs():
    m = _model(use_mlp=False)
    names = dict(m.named_parameters())
    assert "W_in" not in names and "W_out" not in names
    tokens = torch.randint(0, 6, (4, 3))
    assert m(tokens).shape == (4, 3, 5)


def test_state_dict_exposes_the_weight_contract():
    m = _model()
    assert CONTRACT <= set(m.state_dict().keys())


def test_init_uses_fan_in_scaling():
    # Nanda's grokking init: weights ~ N(0, 1/d_model); unembed ~ N(0, 1/d_vocab_out).
    # The scale is fixed to fan-in, not config-driven.
    torch.manual_seed(0)
    m = OneLayerTransformer(
        d_vocab_in=64,
        d_vocab_out=63,
        n_ctx=3,
        d_model=128,
        n_heads=4,
        use_mlp=True,
        d_mlp=512,
        activation="relu",
    )
    assert m.W_E.std().item() == pytest.approx(128**-0.5, rel=0.1)
    assert m.W_in.std().item() == pytest.approx(128**-0.5, rel=0.1)
    assert m.W_U.std().item() == pytest.approx(63**-0.5, rel=0.1)


def test_indivisible_d_model_raises():
    with pytest.raises(AssertionError):
        _model(d_model=8, n_heads=3)  # 8 % 3 != 0


def test_unknown_activation_raises():
    with pytest.raises(KeyError):
        OneLayerTransformer(
            d_vocab_in=6,
            d_vocab_out=5,
            n_ctx=3,
            d_model=8,
            n_heads=2,
            use_mlp=True,
            d_mlp=16,
            activation="tanh",
        )


def test_return_cache_logits_exactly_match_plain_forward():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert torch.equal(cache["logits"], m(tokens))


def test_cache_entry_shapes():
    m = _model()  # d_model=8, n_heads=2, d_mlp=16, d_vocab_out=5, n_ctx=3
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert cache["embed"].shape == (4, 3, 8)
    assert cache["attn_pattern"].shape == (4, 2, 3, 3)
    assert cache["attn_out"].shape == (4, 3, 8)
    assert cache["mlp_pre"].shape == (4, 3, 16)
    assert cache["mlp_post"].shape == (4, 3, 16)
    assert cache["resid_final"].shape == (4, 3, 8)
    assert cache["logits"].shape == (4, 3, 5)


def test_cache_attn_pattern_is_causal_probability():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    pattern = m(tokens, return_cache=True)["attn_pattern"]
    assert torch.allclose(pattern.sum(dim=-1), torch.ones(4, 2, 3), atol=1e-6)
    future = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    assert torch.all(pattern[..., future] == 0)


def test_cache_attn_pattern_produced_cached_attn_out():
    # The cached pattern and attn_out must be mutually consistent: recomputing
    # attention output from the cached pattern reproduces the cached attn_out.
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    resid = cache["embed"]
    v = einops.einsum(
        resid, m.W_V, "batch pos d_model, head d_model d_head -> batch head pos d_head"
    )
    z = einops.einsum(
        cache["attn_pattern"],
        v,
        "batch head query_pos key_pos, batch head key_pos d_head -> batch head query_pos d_head",
    )
    reconstructed = einops.einsum(
        z, m.W_O, "batch head pos d_head, head d_head d_model -> batch pos d_model"
    )
    assert torch.allclose(reconstructed, cache["attn_out"], atol=1e-6)


def test_cache_resid_final_reproduces_logits():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert torch.allclose(cache["resid_final"] @ m.W_U, cache["logits"], atol=1e-6)


def test_cache_residual_stream_decomposes():
    # resid_final = embed + attn_out + mlp_out, where mlp_out = mlp_post @ W_out
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    reconstructed = cache["embed"] + cache["attn_out"] + cache["mlp_post"] @ m.W_out
    assert torch.allclose(reconstructed, cache["resid_final"], atol=1e-6)


def test_cache_without_mlp_has_none_mlp_entries():
    m = _model(use_mlp=False)
    tokens = torch.randint(0, 6, (4, 3))
    cache = m(tokens, return_cache=True)
    assert cache["mlp_pre"] is None
    assert cache["mlp_post"] is None
    assert torch.allclose(cache["embed"] + cache["attn_out"], cache["resid_final"], atol=1e-6)
