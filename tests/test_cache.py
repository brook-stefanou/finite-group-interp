import pytest
import torch

from same_character_table_interp.analysis.cache import ActivationCache, forward_with_cache
from same_character_table_interp.model import OneLayerTransformer


def _model() -> OneLayerTransformer:
    return OneLayerTransformer(
        d_vocab_in=6,
        d_vocab_out=5,
        n_ctx=3,
        d_model=8,
        n_heads=2,
        use_mlp=True,
        d_mlp=16,
        activation="relu",
    )


def test_wrapper_logits_match_plain_forward():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = forward_with_cache(m, tokens)
    with torch.no_grad():
        expected = m(tokens)
    assert torch.equal(cache["logits"], expected)


def test_wrapper_returns_detached_tensors():
    m = _model()
    tokens = torch.randint(0, 6, (4, 3))
    cache = forward_with_cache(m, tokens)
    for key, value in cache.items():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad, f"{key} requires grad"


def test_wrapper_restores_training_mode():
    m = _model()
    m.train()
    forward_with_cache(m, torch.randint(0, 6, (4, 3)))
    assert m.training  # left as found

    m.eval()
    forward_with_cache(m, torch.randint(0, 6, (4, 3)))
    assert not m.training


def test_activation_cache_is_reexported():
    from same_character_table_interp.model import ActivationCache as source

    assert ActivationCache is source


def test_wrapper_restores_training_mode_when_forward_raises():
    m = _model()
    m.train()
    bad_tokens = torch.full((4, 3), 99, dtype=torch.long)  # out-of-vocab -> indexing error
    with pytest.raises(IndexError):
        forward_with_cache(m, bad_tokens)
    assert m.training  # finally-block restored the mode despite the exception
