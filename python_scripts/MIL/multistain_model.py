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

Per-slide mode (``per_slide=True``, multistain_MIL.py --per_slide).  A biopsy often
has several slides of one stain (PAS: usually 2-3).  By default their patches are
pooled into one bag before the stain encoder.  In per-slide mode each slide instead
goes through the stain encoder on its own — one slide token per slide — and a
second, stain-specific attention pool weights the slide tokens (masked to the slides
present) into the stain token, so the model learns which slides to rely on.  With a
single slide that pool is the identity, and the model reduces exactly to the pooled
one.  Input becomes ``(batch, n_stains, max_slides, n_patches, D)`` plus a
``slide_mask``.

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
        per_slide: bool = False,
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
        # Per-slide mode: forward() emits one token per slide, and this pool
        # weights the slide tokens of one stain into the stain token.
        self.slide_pool = (
            AttentionPool(in_dim=d_model, att_dim=pool_att_dim, gated=gated)
            if per_slide
            else None
        )

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
        per_slide: bool = False,
    ) -> None:
        super().__init__()
        self.n_stains = n_stains
        self.out_dim = out_dim
        self.share_stain_encoder = share_stain_encoder
        self.per_slide = per_slide

        def _make_encoder():
            return StainEncoder(
                in_dim=in_dim,
                d_model=d_model,
                n_layers=stain_layers,
                n_heads=n_heads,
                dropout=dropout,
                pool_att_dim=pool_att_dim,
                gated=gated,
                per_slide=per_slide,
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
        slide_mask: torch.Tensor = None,
    ):
        """
        Arguments:
            X: Patch features of shape `(batch_size, n_stains, n_patches, in_dim)`,
                or `(batch_size, n_stains, max_slides, n_patches, in_dim)` for a
                per-slide model (one patch bag per slide).
            stain_mask: Bool tensor of shape `(batch_size, n_stains)`, True where the
                stain is available for that biopsy. Masking a stain here removes it
                even in per-slide mode (used by the stain ablation).
            return_att: If True, also return the stain and patch attention values.
            slide_mask: Per-slide model only — bool `(batch_size, n_stains,
                max_slides)`, True for the slide slots holding a slide.

        Returns:
            out: `(batch_size,)` when `out_dim == 1`, else `(batch_size, out_dim)`.
            stain_att: Only when `return_att=True` — `(batch_size, n_stains)` attention
                values over the stains (before normalisation), NaN for absent stains.
            patch_att: Only when `return_att=True` — `(batch_size, n_stains, n_patches)`
                patch attention values, NaN for absent stains. For a per-slide model,
                a tuple `(slide_att, patch_att)`: `(batch_size, n_stains, max_slides)`
                slide attention values and `(batch_size, n_stains, max_slides,
                n_patches)` patch attention values, NaN for absent slides.
        """
        if self.per_slide:
            if X.dim() != 5 or slide_mask is None:
                raise ValueError(
                    "This per-slide model needs X of shape (batch, stains, slides, "
                    "patches, dim) and a slide_mask."
                )
            tokens, stain_mask, patch_att = self._stain_tokens_per_slide(
                X, stain_mask.bool(), slide_mask.bool(), return_att
            )
        else:
            tokens, stain_mask, patch_att = self._stain_tokens_pooled(
                X, stain_mask.bool(), return_att
            )

        tokens = tokens + self.stain_emb.unsqueeze(0)
        tokens = self.aggregator(tokens, src_key_padding_mask=~stain_mask)

        z, s_att = self.stain_pool(tokens, stain_mask.float(), return_att=True)
        out = self.head(z)
        if self.out_dim == 1:
            out = out.squeeze(-1)

        if return_att:
            return out, s_att.masked_fill(~stain_mask, float("nan")), patch_att
        return out

    def _check_stains(self, n_stains):
        if n_stains != self.n_stains:
            raise ValueError(
                f"X has {n_stains} stains but the model was built for {self.n_stains}."
            )

    def _stain_tokens_pooled(self, X, stain_mask, return_att):
        """One token per stain from its pooled patch bag: (B, S, P, D) input."""
        batch_size, n_stains, n_patches, _ = X.shape
        self._check_stains(n_stains)
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
        return tokens, stain_mask, patch_att

    def _stain_tokens_per_slide(self, X, stain_mask, slide_mask, return_att):
        """One token per stain from its slides: (B, S, M, P, D) input.

        Every present slide goes through the stain's patch encoder on its own and
        becomes a slide token; the stain's slide pool then weights those tokens
        (masked to the slides present) into the stain token. All present slides
        of one stain, across the whole batch, run through the encoder as one
        batch.
        """
        batch_size, n_stains, max_slides, n_patches, _ = X.shape
        self._check_stains(n_stains)
        # A stain masked out in stain_mask (e.g. by the ablation) loses its slides;
        # a stain with no slide present is not present.
        slide_mask = slide_mask & stain_mask.unsqueeze(-1)
        stain_mask = slide_mask.any(-1)
        d_model = self.stain_emb.shape[-1]
        tokens = X.new_zeros(batch_size, n_stains, d_model)
        if return_att:
            slide_att = X.new_full((batch_size, n_stains, max_slides), float("nan"))
            patch_att = X.new_full(
                (batch_size, n_stains, max_slides, n_patches), float("nan")
            )

        for s in range(n_stains):
            b_idx = stain_mask[:, s].nonzero(as_tuple=True)[0]
            if b_idx.numel() == 0:
                continue
            m = slide_mask[b_idx, s]  # (b, M)
            pairs = m.nonzero(as_tuple=True)  # (row in b, slide slot)
            encoder = self._encoder(s)
            out = encoder(X[b_idx[pairs[0]], s, pairs[1]], return_att=return_att)
            t_slide, att = out if return_att else (out, None)
            slide_tokens = X.new_zeros(b_idx.numel(), max_slides, d_model)
            slide_tokens[pairs] = t_slide
            t, s_att = encoder.slide_pool(slide_tokens, m.float(), return_att=True)
            tokens[b_idx, s] = t
            if return_att:
                slide_att[b_idx, s] = s_att.masked_fill(~m, float("nan"))
                patch_att[b_idx[pairs[0]], s, pairs[1]] = att
        return tokens, stain_mask, ((slide_att, patch_att) if return_att else None)


# ---------------------------------------------------------------------------
# Early fusion
# ---------------------------------------------------------------------------
class _StainWeightedSum(nn.Module):
    """z = sum_s A_s * t_s — a named step so tools can hook the per-stain terms.

    analyze_stain_weights.py hooks the ``stain_pool`` module's input to read the
    per-stain tokens, exactly as for MultiStainFusion.
    """

    def forward(self, t: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return (A.unsqueeze(-1) * t).sum(1)


class EarlyFusionMIL(nn.Module):
    """Early fusion: one attention-pooled bag of the patches of every stain.

    Every patch of every stain goes through the same projection; a learned stain
    embedding is added so the model knows which stain a patch shows; then one
    gated attention pool runs over all patches of the biopsy at once, so patches
    of different stains compete directly for attention. Unlike MultiStainFusion
    there is no stain-specific encoder and no per-stain token stage. With
    ``n_layers > 0`` a transformer runs over the joint bag first, letting patches
    of different stains attend to each other (cost grows with the square of the
    total patch count, so keep the bag small).

    The stains come from different, unregistered sections, so there is no
    pixel-level correspondence to fuse: the patch level is as early as fusion can
    happen here.

    Takes and returns exactly what MultiStainFusion does — pooled ``X`` (B, S, P, D)
    or per-slide ``X`` (B, S, M, P, D) with ``slide_mask`` — so the same training,
    evaluation and analysis code runs on both. ``stain_att`` is log A_s, where
    A_s is the share of the attention that falls on stain s: the attention-pooled
    vector regroups exactly as z = sum_s A_s t_s, t_s being the attention-weighted
    mean of stain s's patches, which is the per-stain form analyze_stain_weights.py
    decomposes.

    Arguments:
        in_dim: Patch feature dimension.
        n_stains: Size of the stain vocabulary.
        d_model: Working dimension.
        n_layers: Transformer layers over the joint bag (0 = attention pooling only).
        n_heads: Attention heads of that transformer.
        dropout: Dropout rate.
        pool_att_dim: Hidden dimension of the attention-pooling MLP.
        gated: Use gated attention pooling.
        out_dim: Head outputs — 1 for a Cox risk score or a regression target.
    """

    def __init__(
        self,
        in_dim: int,
        n_stains: int,
        d_model: int = 256,
        n_layers: int = 0,
        n_heads: int = 8,
        dropout: float = 0.1,
        pool_att_dim: int = 128,
        gated: bool = True,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.n_stains = n_stains
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stain_emb = nn.Parameter(torch.zeros(n_stains, d_model))
        nn.init.normal_(self.stain_emb, std=0.02)
        self.encoder = (
            _transformer(d_model, n_layers, n_heads, dropout) if n_layers > 0 else None
        )
        self.patch_pool = AttentionPool(in_dim=d_model, att_dim=pool_att_dim, gated=gated)
        self.stain_pool = _StainWeightedSum()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, out_dim)
        )

    def forward(
        self,
        X: torch.Tensor,
        stain_mask: torch.Tensor,
        return_att: bool = False,
        slide_mask: torch.Tensor = None,
    ):
        """Same contract as MultiStainFusion.forward (see there)."""
        per_slide = X.dim() == 5
        if per_slide:
            if slide_mask is None:
                raise ValueError("Per-slide input (B, S, M, P, D) needs a slide_mask.")
            B, S, M, P, _ = X.shape
            slide_mask = slide_mask.bool() & stain_mask.bool().unsqueeze(-1)
            patch_mask = slide_mask.unsqueeze(-1).expand(B, S, M, P)
        else:
            B, S, P, _ = X.shape
            M = 1
            patch_mask = stain_mask.bool().unsqueeze(-1).expand(B, S, P)
        if S != self.n_stains:
            raise ValueError(f"X has {S} stains but the model was built for {self.n_stains}.")
        stain_mask = patch_mask.reshape(B, S, -1).any(-1)

        # One joint bag of S * M * P patches, each carrying its stain embedding.
        n = M * P
        H = self.proj(X.reshape(B, S * n, X.shape[-1]))
        H = H + self.stain_emb.repeat_interleave(n, dim=0).unsqueeze(0)
        mask = patch_mask.reshape(B, S * n)
        if self.encoder is not None:
            H = self.encoder(H, src_key_padding_mask=~mask)

        _, logits = self.patch_pool(H, mask.float(), return_att=True)  # (B, S*n)
        a = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=1)

        # Regroup z = sum_i a_i h_i by stain: A_s = attention mass of stain s,
        # t_s = attention-weighted mean of its patches.
        a_s = a.reshape(B, S, n)
        A = a_s.sum(-1)  # (B, S)
        t = (a_s.unsqueeze(-1) * H.reshape(B, S, n, -1)).sum(2)
        t = t / A.clamp_min(1e-12).unsqueeze(-1)
        z = self.stain_pool(t, A)

        out = self.head(z)
        if self.out_dim == 1:
            out = out.squeeze(-1)
        if not return_att:
            return out
        stain_att = torch.log(A.clamp_min(1e-30)).masked_fill(~stain_mask, float("nan"))
        patch_att = logits.masked_fill(~mask, float("nan"))
        if per_slide:
            patch_att = patch_att.reshape(B, S, M, P)
            # Slide "attention": the log attention mass of each slide.
            slide_att = torch.log(a.reshape(B, S, M, P).sum(-1).clamp_min(1e-30))
            return out, stain_att, (slide_att.masked_fill(~slide_mask, float("nan")), patch_att)
        return out, stain_att, patch_att.reshape(B, S, P)
