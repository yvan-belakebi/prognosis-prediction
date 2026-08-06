"""
late_fusion.py — Late-fusion wrapper that conditions a MIL survival model on a
per-bag diagnosis code.

The three supported backbones (ABMIL, DeepGraphSurv, PatchGCN) all share the
same tail: they pool the patches into a bag vector ``z`` and then apply a single
linear ``self.classifier`` to produce the risk scalar. This wrapper neutralises
that head (``classifier = Identity``) so ``backbone.forward(...)`` returns the
pooled bag vector ``z`` unchanged, embeds the diagnosis code, concatenates it to
``z``, and applies a fresh linear head:

    risk = Linear([ z ; Embedding(code) ])

Because the diagnosis is fused after pooling, the attention/GCN stack still sees
only image features — the code merely shifts/reshapes the final decision. This is
the "late fusion" (concat-at-head) variant.

The wrapper's ``forward`` takes the collated batch TensorDict directly, mirroring
MIL.py's ``_forward`` helper (same X / mask / adj handling) plus the diagnosis
code read from ``batch["Y"][:, 2]``.
"""

import torch
import torch.nn as nn

from torchmil.nn.utils import LazyLinear


class LateFusionSurv(nn.Module):
    """Condition a pooled-bag MIL backbone on a categorical diagnosis code.

    Arguments:
        backbone: An ABMIL / DeepGraphSurv / PatchGCN instance. Its ``.classifier``
            attribute is replaced in-place with ``nn.Identity`` so that
            ``backbone.forward`` returns the pooled bag vector ``z`` (and, when
            ``return_att=True``, the instance attention) instead of the risk.
        model_type: One of ``"abmil"``, ``"deepgraphsurv"``, ``"patchgcn"`` —
            selects the backbone call signature (abmil takes ``(X, mask)``, the
            graph models take ``(X, adj, mask)``).
        n_codes: Vocabulary size of the diagnosis code (including index 0, which
            is reserved for unknown/missing and kept as a neutral zero embedding
            via ``padding_idx=0``).
        diag_dim: Diagnosis embedding dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        model_type: str,
        n_codes: int,
        diag_dim: int = 16,
    ) -> None:
        super().__init__()
        if model_type not in ("abmil", "deepgraphsurv", "patchgcn"):
            raise ValueError(f"Unsupported model_type for late fusion: {model_type}")
        self.model_type = model_type
        self.backbone = backbone
        # Expose the pooled bag vector z: with an identity head the backbone's
        # trailing ``self.classifier(z).squeeze(...)`` returns z unchanged (the
        # squeeze is a no-op because z's last dim is the hidden size, not 1).
        self.backbone.classifier = nn.Identity()
        self.diag_emb = nn.Embedding(n_codes, diag_dim, padding_idx=0)
        # Lazy head: input dim = z_dim + diag_dim is resolved on first forward,
        # matching how the backbones' own LazyLinear heads are built.
        self.head = LazyLinear(in_features=None, out_features=1)

    def _pool(self, batch, return_att: bool):
        """Run the backbone and return (z, att-or-None)."""
        X, mask = batch["X"], batch.get("mask")
        if self.model_type == "abmil":
            out = self.backbone(X, mask, return_att=return_att)
        else:
            adj = batch["adj"]
            if adj.is_sparse:
                adj = adj.to_dense()
            adj = adj.float()
            out = self.backbone(X, adj, mask, return_att=return_att)
        if return_att:
            z, att = out
            return z, att
        return out, None

    def forward(self, batch, return_att: bool = False):
        """Forward pass on a collated batch TensorDict.

        Arguments:
            batch: Collated bag dict with keys ``X``, ``Y`` (``[:, 2]`` is the
                diagnosis code), ``mask``, and — for the graph backbones —
                ``adj``.
            return_att: If True, also return the backbone instance attention.

        Returns:
            risk: Risk logits of shape ``(batch_size,)``.
            att: Only when ``return_att=True`` — instance attention of shape
                ``(batch_size, bag_size)``.
        """
        z, att = self._pool(batch, return_att)
        code = batch["Y"][:, 2].long()
        d = self.diag_emb(code)  # (batch_size, diag_dim)
        risk = self.head(torch.cat([z, d], dim=-1)).squeeze(-1)  # (batch_size,)
        if return_att:
            return risk, att
        return risk
