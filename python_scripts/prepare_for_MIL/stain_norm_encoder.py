"""
stain_norm_encoder.py — Wrap a TRIDENT patch encoder with on-GPU stain normalisation.

TRIDENT's feature-extraction loop (trident/wsi_objects/WSI.py::extract_patch_features)
applies ``patch_encoder.eval_transforms`` per patch in the DataLoader worker, then runs
``patch_encoder(imgs)`` on the GPU batch inside ``torch.autocast``.  This wrapper exploits
that seam to reproduce the exact pipeline used by compute_feats_clam.py — stain
normalisation on the GPU batch, *between* ToTensor and the backbone's channel Normalize —
without editing any TRIDENT source.

How it works:
    * ``eval_transforms`` is the base encoder's transform with the trailing torchvision
      ``Normalize`` STRIPPED, so the DataLoader yields RGB [0,1] CxHxW tensors.
    * ``forward(x)`` runs:  stain_normalizer(x)  →  stripped Normalize  →  base_encoder(x).
      The stain step is forced to float32 with autocast disabled (the iterative QR /
      Vahadane solver is numerically fragile in bf16/fp16); the base encoder then runs in
      its native ``precision`` under the loop's outer autocast, exactly as stock TRIDENT.
    * ``enc_name`` / ``precision`` / ``embedding_dim`` are surfaced so the Processor's
      output-folder naming, autocast dtype and empty-slide handling keep working.

Because stain references are per-stain, build one wrapped encoder per stain group and run
TRIDENT's ``feat`` task once per group (see run_trident_stain_feats.py).  The ``method`` and
target buffers are read from the .pt references produced by fit_stain_reference.py, so the
per-patch source stain matrix is estimated the same way the target was.

This module intentionally does NOT import compute_feats_clam.py: that module imports CLAM
at top level, and the TRIDENT path should not depend on CLAM.  The two small helpers below
are therefore duplicated from it (kept byte-compatible on purpose).
"""

import os
import sys

import torch

# ---------------------------------------------------------------------------
# Resolve sibling torch-staintools package (same approach as compute_feats_clam.py)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_staintools_dir = os.path.abspath(
    os.path.join(_script_dir, "..", "torch-staintools-main")
)
if _staintools_dir not in sys.path:
    sys.path.insert(0, _staintools_dir)

from torch_staintools.constants import CONFIG  # noqa: E402
from torch_staintools.normalizer import NormalizerBuilder  # noqa: E402

# torch.compile for the stain-matrix estimation needs a working compiler toolchain that is
# often absent/misconfigured; keep it OFF (same default as compute_feats_clam.py).
CONFIG.ENABLE_COMPILE = False

# Solver settings — must match compute_feats_clam.py so TRIDENT-path features are
# numerically consistent with the CLAM-path features for the same references.
_CONCENTRATION_SOLVER = "qr"
_LUMINOSITY_THRESHOLD = 0.8
_SPARSE_DICT_STEPS = 30
_MAXITER = 30


# ---------------------------------------------------------------------------
# Helpers (duplicated from compute_feats_clam.py — keep in sync)
# ---------------------------------------------------------------------------


def load_stain_normalizer(ref_path: str, device: torch.device):
    """Build a stain normalizer from a .pt reference produced by fit_stain_reference.py.

    The algorithm is auto-matched to the 'method' recorded in the reference (defaulting to
    'vahadane' for older references), and the fitted target buffers are re-injected so it
    transforms without re-fitting.  Returns (nn.Module, method).
    """
    ref = torch.load(ref_path, weights_only=True)
    method = ref.get("method", "vahadane")
    kwargs = dict(
        concentration_solver=_CONCENTRATION_SOLVER,
        maxiter=_MAXITER,
        luminosity_threshold=_LUMINOSITY_THRESHOLD,
        device=device,
    )
    if method == "vahadane":
        kwargs["sparse_dict_steps"] = _SPARSE_DICT_STEPS  # Vahadane-only knob
    norm = NormalizerBuilder.build(method, **kwargs).to(device)
    norm.register_buffer("stain_matrix_target", ref["stain_matrix_target"].to(device))
    norm.register_buffer("maxC_target", ref["maxC_target"].to(device))
    return norm, method


def split_backbone_transform(base_transform):
    """Split a torchvision Compose at the trailing Normalize.

    Returns (pre_transform, backbone_normalize) where pre_transform turns a PIL patch into
    an RGB [0,1] CxHxW tensor and backbone_normalize is the mean/std Normalize.
    """
    from torchvision import transforms as T

    steps = list(base_transform.transforms)
    if not steps or not isinstance(steps[-1], T.Normalize):
        raise ValueError(
            "Expected the encoder transform to end with a torchvision Normalize; "
            f"got {type(steps[-1]).__name__ if steps else 'empty Compose'}."
        )
    return T.Compose(steps[:-1]), steps[-1]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class StainNormPatchEncoder(torch.nn.Module):
    """Wrap a TRIDENT patch encoder to apply GPU-batched stain normalisation.

    Args:
        base_encoder: An encoder from trident ``encoder_factory`` (exposes ``eval_transforms``,
            ``precision``, ``enc_name``, and a ``forward`` that maps a normalised batch to
            features).
        stain_normalizer: An nn.Module mapping an RGB [0,1] BxCxHxW batch to a normalised
            one (from ``load_stain_normalizer``).  If None, the wrapper is a passthrough
            (raw-patch features) but still TRIDENT-compatible.
        enc_name: Overrides the output-folder-determining name.  Defaults to the base
            encoder's ``enc_name`` so outputs land in ``features_{enc_name}/`` unchanged.
            Pass e.g. ``f"{base.enc_name}_stain"`` to keep stain-normalised features in a
            separate folder from any raw-patch run.
    """

    def __init__(self, base_encoder, stain_normalizer=None, enc_name: str = None):
        super().__init__()
        self.base_encoder = base_encoder
        self.stain_normalizer = stain_normalizer

        # Split the encoder transform; the DataLoader will now yield RGB [0,1] tensors and
        # the channel Normalize is re-applied inside forward() after stain normalisation.
        self.pre_transform, self.backbone_normalize = split_backbone_transform(
            base_encoder.eval_transforms
        )

        # --- Surface the attributes TRIDENT's extract loop / Processor read ---
        self.eval_transforms = self.pre_transform
        self.precision = getattr(base_encoder, "precision", torch.float32)
        self.enc_name = enc_name or getattr(base_encoder, "enc_name", None)
        self.embedding_dim = getattr(base_encoder, "embedding_dim", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: RGB [0,1] BxCxHxW (Normalize was stripped from eval_transforms).
        if self.stain_normalizer is not None:
            # The outer loop calls this under torch.autocast(precision); the stain solver
            # is fragile in low precision, so force float32 with autocast disabled.
            with torch.autocast(device_type=x.device.type, enabled=False):
                x = self.stain_normalizer(x.float())
        x = self.backbone_normalize(x)
        return self.base_encoder(x)


def build_stain_encoder(
    base_encoder,
    stain_ref_path: str = None,
    device: torch.device = None,
    enc_name: str = None,
) -> StainNormPatchEncoder:
    """Convenience builder: load a reference (if given) and wrap ``base_encoder``.

    Args:
        base_encoder: Output of trident ``encoder_factory(...)``.
        stain_ref_path: Path to a .pt reference from fit_stain_reference.py.  If None, the
            wrapper is a passthrough (no stain normalisation).
        device: Device the normalizer buffers/model live on.  Defaults to the base
            encoder's parameter device, falling back to CUDA-if-available.
        enc_name: See StainNormPatchEncoder.
    """
    if device is None:
        try:
            device = next(base_encoder.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = None
    if stain_ref_path is not None:
        normalizer, _method = load_stain_normalizer(stain_ref_path, device)

    return StainNormPatchEncoder(base_encoder, normalizer, enc_name=enc_name)
