from typing import Literal, TypedDict, overload

import einops
import torch
import torch.nn as nn


class ActivationCache(TypedDict):
    """Every intermediate of one forward pass, keyed TransformerLens-style.

    ``mlp_pre``/``mlp_post`` are the d_mlp-sized activations before/after the
    nonlinearity (None when use_mlp=False). ``resid_mid`` and ``mlp_out`` are
    deliberately omitted as recoverable: resid_mid = embed + attn_out;
    mlp_out = resid_final - embed - attn_out.
    """

    embed: torch.Tensor  # W_E[tokens] + W_pos      [batch, pos, d_model]
    attn_pattern: torch.Tensor  # post-softmax      [batch, head, q_pos, k_pos]
    attn_out: torch.Tensor  # after W_O             [batch, pos, d_model]
    mlp_pre: torch.Tensor | None  # pre-activation  [batch, pos, d_mlp]
    mlp_post: torch.Tensor | None  # post-activation [batch, pos, d_mlp]
    resid_final: torch.Tensor  # final residual     [batch, pos, d_model]
    logits: torch.Tensor  #                         [batch, pos, d_vocab_out]


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

        # Initialization follows Nanda's grokking transformer: every weight is
        # scaled by 1/sqrt(d_model) (fan-in), except the unembedding, which uses
        # 1/sqrt(d_vocab_out). This places the model in the weight-norm regime
        # where grokking appears within the training budget, and is deliberately
        # fixed rather than config-driven.
        scale = d_model**-0.5
        self.W_E = nn.Parameter(torch.randn(d_vocab_in, d_model) * scale)
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) * scale)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, d_head) * scale)
        self.W_O = nn.Parameter(torch.randn(n_heads, d_head, d_model) * scale)
        if use_mlp:
            self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * scale)
            self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * scale)
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab_out) * d_vocab_out**-0.5)

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations[activation]

        # Causal mask: True above the diagonal (key_pos > query_pos) marks the
        # future positions each query is forbidden from attending to.
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(n_ctx, n_ctx, dtype=torch.bool), diagonal=1),
        )

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[False] = ...) -> torch.Tensor: ...

    @overload
    def forward(self, tokens: torch.Tensor, return_cache: Literal[True]) -> ActivationCache: ...

    def forward(
        self, tokens: torch.Tensor, return_cache: bool = False
    ) -> torch.Tensor | ActivationCache:
        # tokens: [batch, n_ctx] int64 ids; returns logits [batch, n_ctx, d_vocab_out],
        # or the full ActivationCache when return_cache=True.
        embed = self.W_E[tokens] + self.W_pos
        resid = embed
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
        attn_scores = attn_scores.masked_fill(self.causal_mask, float("-inf"))
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
        mlp_pre: torch.Tensor | None = None
        mlp_post: torch.Tensor | None = None
        if self.use_mlp:
            mlp_pre = einops.einsum(
                resid, self.W_in, "batch pos d_model, d_model d_mlp -> batch pos d_mlp"
            )
            mlp_post = self.activation(mlp_pre)
            mlp_out = einops.einsum(
                mlp_post, self.W_out, "batch pos d_mlp, d_mlp d_model -> batch pos d_model"
            )
            resid = resid + mlp_out
        logits = einops.einsum(
            resid, self.W_U, "batch pos d_model, d_model vocab_out -> batch pos vocab_out"
        )
        if return_cache:
            return ActivationCache(
                embed=embed,
                attn_pattern=pattern,
                attn_out=attn_out,
                mlp_pre=mlp_pre,
                mlp_post=mlp_post,
                resid_final=resid,
                logits=logits,
            )
        return logits
