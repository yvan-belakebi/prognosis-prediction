"""
multistain_model.py — UNICORN-style multi-stain fusion for biopsy-level prediction.

Architecture (mirrors the multi-stain prediction model of the UNICORN paper):

    per stain s, in parallel
        X_s (P, D_feat)                       patch features from run_trident_stain_feats.py
          -> LayerNorm + Linear -> d_model    stain-specific projection
          -> TransformerEncoder (2 layers)    stain-specific patch context
          -> AttentionPool                    one token t_s per stain
    fusion
        [t_1 ... t_S] + stain embedding       S tokens, absent stains masked out
          -> TransformerEncoder (2 layers)    aggregator across stains
          -> AttentionPool (stain mask)       biopsy vector z
          -> head                             risk (Cox) or regression target

Both heads are a single linear layer on ``z``; the task only changes the loss
(``--task`` in multistain_MIL.py), so ``out_dim`` stays 1 in both cases.

Masking.  The patch axis carries no mask — MultiStainBiopsyDataset hands every stain
exactly ``patches_per_stain`` rows (see its docstring) — so the patch transformer runs
the fast, memory-efficient attention path.  Only the stain axis is masked, with a
*key* padding mask: absent stains are removed from the keys but still exist as query
rows, which is what keeps the softmax well defined (a fully masked query row would
produce NaN, and the NaN would survive into the gradients even after the row is
discarded downstream).  Their outputs are junk and are dropped by the masked
attention pooling that follows.
"""

import torch
import torch.nn as nn

from torchmil.nn import AttentionPool


def _transformer(d_model: int, n_layers: int, n_heads: int, dropout: float) -> nn.Module:
    """Pre-norm transformer encoder stack (batch-first)."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)


class StainEncoder(nn.Module):
    """Patch-level encoder for one stain: projection -> transformer -> attention pool.

    Arguments:
        in_dim: Patch feature dimension of the foundation model (e.g. 1536 for UNI-v2).
        d_model: Working dimension of the transformer and of the emitted stain token.
        n_layers: Transformer layers (2 in the UNICORN configuration).
        n_heads: Attention heads.
        dropout: Dropout rate, applied in the projection and the transformer.
        pool_att_dim: Hidden dimension of the attention-pooling MLP.
        gated: Use gated attention pooling (Ilse et al.).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool_att_dim: int = 128,
        gated: bool = True,
    ) -> None:
        super().__init__()
        # LayerNorm first: foundation-model features are unnormalised and their
        # scale varies between backbones and between stains.
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = _transformer(d_model, n_layers, n_heads, dropout)
        self.pool = AttentionPool(in_dim=d_model, att_dim=pool_att_dim, gated=gated)

    def forward(self, X: torch.Tensor, return_att: bool = False):
        """
        Arguments:
            X: Patch features of shape `(batch_size, n_patches, in_dim)`.
            return_att: If True, also return the patch attention values (before
                normalisation) of shape `(batch_size, n_patches)`.

        Returns:
            t: Stain token of shape `(batch_size, d_model)`.
            att: Only when `return_att=True`.
        """
        H = self.encoder(self.proj(X))  # (batch_size, n_patches, d_model)
        return self.pool(H, return_att=return_att)


class MultiStainFusion(nn.Module):
    """Multi-stain biopsy model: per-stain encoders + a transformer aggregator.

    Arguments:
        in_dim: Patch feature dimension.
        n_stains: Size of the stain vocabulary (the stain axis of ``X``).
        d_model: Token dimension shared by the stain encoders and the aggregator.
        stain_layers: Transformer layers inside each stain encoder.
        agg_layers: Transformer layers in the aggregator.
        n_heads: Attention heads (both levels).
        dropout: Dropout rate.
        pool_att_dim: Hidden dimension of both attention-pooling MLPs.
        gated: Use gated attention pooling.
        out_dim: Head outputs — 1 for a Cox risk score or a regression target.
        share_stain_encoder: Use one encoder for every stain instead of one per stain.
            Per-stain encoders follow the paper; sharing trades stain specificity for
            sample efficiency, which helps when the rarer stains have few biopsies.
    """

    def __init__(
        self,
        in_dim: int,
        n_stains: int,
        d_model: int = 256,
        stain_layers: int = 2,
        agg_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool_att_dim: int = 128,
        gated: bool = True,
        out_dim: int = 1,
        share_stain_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.n_stains = n_stains
        self.out_dim = out_dim
        self.share_stain_encoder = share_stain_encoder

        def _make_encoder():
            return StainEncoder(
                in_dim=in_dim,
                d_model=d_model,
                n_layers=stain_layers,
                n_heads=n_heads,
                dropout=dropout,
                pool_att_dim=pool_att_dim,
                gated=gated,
            )

        n_encoders = 1 if share_stain_encoder else n_stains
        self.stain_encoders = nn.ModuleList([_make_encoder() for _ in range(n_encoders)])

        # Tells the aggregator which stain each token came from — necessary because
        # self-attention is permutation-equivariant and the stains present vary per
        # biopsy, so position alone carries no identity.
        self.stain_emb = nn.Parameter(torch.zeros(n_stains, d_model))
        nn.init.normal_(self.stain_emb, std=0.02)

        self.aggregator = _transformer(d_model, agg_layers, n_heads, dropout)
        self.stain_pool = AttentionPool(
            in_dim=d_model, att_dim=pool_att_dim, gated=gated
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, out_dim)
        )

    def _encoder(self, stain_idx: int) -> nn.Module:
        return self.stain_encoders[0 if self.share_stain_encoder else stain_idx]

    def forward(
        self,
        X: torch.Tensor,
        stain_mask: torch.Tensor,
        return_att: bool = False,
    ):
        """
        Arguments:
            X: Patch features of shape `(batch_size, n_stains, n_patches, in_dim)`.
            stain_mask: Bool tensor of shape `(batch_size, n_stains)`, True where the
                stain is available for that biopsy.
            return_att: If True, also return the stain and patch attention values.

        Returns:
            out: `(batch_size,)` when `out_dim == 1`, else `(batch_size, out_dim)`.
            stain_att: Only when `return_att=True` — `(batch_size, n_stains)` attention
                values over the stains (before normalisation), NaN for absent stains.
            patch_att: Only when `return_att=True` — `(batch_size, n_stains, n_patches)`
                patch attention values, NaN for absent stains.
        """
        batch_size, n_stains, n_patches, _ = X.shape
        if n_stains != self.n_stains:
            raise ValueError(
                f"X has {n_stains} stains but the model was built for {self.n_stains}."
            )
        stain_mask = stain_mask.bool()

        tokens = X.new_zeros(batch_size, n_stains, self.stain_emb.shape[-1])
        patch_att = (
            X.new_full((batch_size, n_stains, n_patches), float("nan"))
            if return_att
            else None
        )

        # One pass per stain, restricted to the biopsies where that stain exists, so
        # absent stains cost nothing and never feed zeros through an encoder.
        for s in range(n_stains):
            idx = stain_mask[:, s].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out = self._encoder(s)(X[idx, s], return_att=return_att)
            if return_att:
                t, att = out
                patch_att[idx, s] = att
            else:
                t = out
            tokens[idx, s] = t

        tokens = tokens + self.stain_emb.unsqueeze(0)
        tokens = self.aggregator(tokens, src_key_padding_mask=~stain_mask)

        z, s_att = self.stain_pool(tokens, stain_mask.float(), return_att=True)
        out = self.head(z)
        if self.out_dim == 1:
            out = out.squeeze(-1)

        if return_att:
            return out, s_att.masked_fill(~stain_mask, float("nan")), patch_att
        return out
