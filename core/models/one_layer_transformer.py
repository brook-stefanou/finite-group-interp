import einops
import torch
import torch.nn as nn


class OneLayerTransformer(nn.Module):
    """A 1-layer attention + MLP transformer for the group-multiplication task.

    Minimal and interpretability-friendly: no LayerNorm, no biases, learned
    positional embeddings, untied embed/unembed. Weight names follow the
    TransformerLens convention (W_E, W_Q, W_K, W_V, W_O, W_in, W_out, W_U) so
    the state_dict doubles as the analysis weight contract.
    """

    def __init__(
        self,
        d_vocab_in: int,
        d_vocab_out: int,
        n_ctx: int,
        d_model: int,
        n_heads: int,
        use_mlp: bool,
        d_mlp: int,
        activation: str,
        init_std: float,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"Model dimension ({d_model}) is not evenly divisible by number of heads "
            f"({n_heads}). This means shape won't match residual stream when adding back to it."
        )
        d_head = d_model // n_heads
        self.d_head = d_head
        self.use_mlp = use_mlp
        self.d_mlp = d_mlp
        self.d_vocab_in = d_vocab_in
        self.d_vocab_out = d_vocab_out
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_heads = n_heads
        self.init_std = init_std

        self.W_E = nn.Parameter(torch.randn(d_vocab_in, d_model) * init_std)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * init_std)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) * init_std)
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) * init_std)
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) * init_std)
        self.W_O = nn.Parameter(torch.randn(n_heads, d_head, d_model) * init_std)
        if use_mlp:
            self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * init_std)
            self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * init_std)
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab_out) * init_std)

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations[activation]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, n_ctx] int64 ids; returns logits [batch, n_ctx, d_vocab_out]
        resid = self.W_E[tokens] + self.W_pos
        q = einops.einsum(
            resid, self.W_Q, "batch pos d_model, head d_model d_head -> batch head pos d_head"
        )
        k = einops.einsum(
            resid, self.W_K, "batch pos d_model, head d_model d_head -> batch head pos d_head"
        )
        v = einops.einsum(
            resid, self.W_V, "batch pos d_model, head d_model d_head -> batch head pos d_head"
        )
        attn_scores = einops.einsum(
            q,
            k,
            "batch head query_pos d_head, batch head key_pos d_head -> batch head query_pos key_pos",
        )
        attn_scores = attn_scores / (self.d_head**0.5)
        pattern = torch.softmax(attn_scores, dim=-1)
        z = einops.einsum(
            pattern,
            v,
            "batch head query_pos key_pos, batch head key_pos d_head -> batch head query_pos d_head",
        )
        attn_out = einops.einsum(
            z, self.W_O, "batch head pos d_head, head d_head d_model -> batch pos d_model"
        )
        resid = resid + attn_out
        if self.use_mlp:
            resid = einops.einsum(
                resid, self.W_in, "batch pos d_model, d_model d_mlp -> batch pos d_mlp"
            )
            resid = self.activation(resid)
            resid = einops.einsum(
                resid, self.W_out, "batch pos d_mlp, d_mlp d_model -> batch pos d_model"
            )
        logits = einops.einsum(
            resid, self.W_U, "batch pos d_model, d_model vocab_out -> batch pos vocab_out"
        )
        return logits
