import pytest
import torch

from core.models.one_layer_transformer import OneLayerTransformer

# Weight names the analysis layer relies on (the "contract").
CONTRACT = {"W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"}


def _model(use_mlp: bool = True, d_model: int = 8, n_heads: int = 2) -> OneLayerTransformer:
    return OneLayerTransformer(
        d_vocab_in=6, d_vocab_out=5, n_ctx=3, d_model=d_model, n_heads=n_heads,
        use_mlp=use_mlp, d_mlp=16, activation="relu", init_std=0.02,
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


def test_init_std_controls_weight_scale():
    big = OneLayerTransformer(d_vocab_in=6, d_vocab_out=5, n_ctx=3, d_model=8, n_heads=2,
                              use_mlp=True, d_mlp=16, activation="relu", init_std=1.0)
    small = _model()  # init_std=0.02
    assert big.W_E.std() > small.W_E.std()


def test_indivisible_d_model_raises():
    with pytest.raises(AssertionError):
        _model(d_model=8, n_heads=3)  # 8 % 3 != 0


def test_unknown_activation_raises():
    with pytest.raises(KeyError):
        OneLayerTransformer(d_vocab_in=6, d_vocab_out=5, n_ctx=3, d_model=8, n_heads=2,
                            use_mlp=True, d_mlp=16, activation="tanh", init_std=0.02)
