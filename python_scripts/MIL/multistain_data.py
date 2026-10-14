"""
multistain_data.py — Biopsy-level, multi-stain feature bags for the UNICORN-style
fusion model (see multistain_model.py).

The single-stain scripts (MIL.py, regression_MIL.py) treat one *slide* as one bag.
Here the unit of prediction is the *biopsy*: every slide of the biopsy is grouped by
its stain, so one sample is a stack of per-stain patch bags.

Inputs (flat; produced by TRIDENT and define_labels.py / define_regression_labels.py):

    features_path/{slide_id}.h5     keys 'features' (N, D), 'coords' (N, 2)
    labels_path/{slide_id}.npy      [time, event]  or  scalar (regression)
    labels CSV                      file_name, biopsy_number, stain, split

The labels CSV (loaded with mil_utils.load_labels_table) gives every slide its
biopsy, its stain and its train/val split.  All slides of a biopsy carry the same
survival/regression label, so the biopsy label is read from the first slide found.

Fixed patch count per stain
---------------------------
``__getitem__`` returns a dense ``X`` of shape ``(n_stains, patches_per_stain, D)``.
Bags larger than ``patches_per_stain`` are subsampled; bags smaller than it are
*tiled* (patches repeated) rather than zero-padded.  This keeps every sequence the
same length, so the patch-level transformer needs no padding mask and can use the
memory-efficient attention kernels — and it is close to exact: attending over k
identical copies of a patch set gives the same softmax-weighted output as attending
over the set once (each weight is simply split across its copies).  Only the stain
axis is masked, via ``stain_mask``.

``stain_dropout`` optionally masks out available stains at random during training, so
the model sees stain combinations other than the one each biopsy happens to carry.

Returned sample (all tensors, so torch's default collate stacks them):

    X          (n_stains, patches_per_stain, D)  float32, zeros where the stain is absent
    stain_mask (n_stains,)                       bool, True where the stain is present
    Y          (n_label,)                        float32, [time, event] or [target]
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mil_utils import StainVocabulary, bag_group, canon_stain, discover_bags  # noqa: F401


# ---------------------------------------------------------------------------
# Indexing: features + labels + labels table -> one record per biopsy
# ---------------------------------------------------------------------------
def index_biopsies(
    features_path,
    labels_path,
    table,
    vocab,
    file_ext=".h5",
    authorized_slides=None,
    min_stains=1,
    verbose=True,
) -> list:
    """Group the slides under ``features_path`` into one record per biopsy.

    ``table`` is the labels table (mil_utils.load_labels_table): it gives each
    slide its biopsy and its stain. Returns a list of dicts::

        {"biopsy": str,                       # biopsy_number
         "label": np.ndarray,                 # from the first labelled slide
         "slides": {stain_idx: [h5 path, ...]},
         "bag_names": [bag name, ...]}

    Slides are dropped when they are not in the table, have no label file, have
    a stain outside ``vocab``, or are excluded by ``authorized_slides``.
    Biopsies left with fewer than ``min_stains`` stains are dropped.
    """
    bags = discover_bags(features_path, extensions=(file_ext,))
    labelled = set(discover_bags(labels_path, extensions=(".npy",)))

    dropped = {"unauthorized": 0, "no_label": 0, "not_in_csv": 0, "other_stain": 0}
    groups = {}
    for bag in bags:
        if authorized_slides is not None and bag not in authorized_slides:
            dropped["unauthorized"] += 1
            continue
        if bag not in labelled:
            dropped["no_label"] += 1
            continue
        if bag not in table.index:
            dropped["not_in_csv"] += 1
            continue
        stain_idx = vocab.index_of(table.at[bag, "stain"])
        if stain_idx is None:
            dropped["other_stain"] += 1
            continue

        biopsy = bag_group(bag, table)
        rec = groups.setdefault(
            biopsy, {"biopsy": biopsy, "label": None, "slides": {}, "bag_names": []}
        )
        rec["slides"].setdefault(stain_idx, []).append(
            os.path.join(features_path, f"{bag}{file_ext}")
        )
        rec["bag_names"].append(bag)
        if rec["label"] is None:
            rec["label"] = np.load(os.path.join(labels_path, f"{bag}.npy")).ravel()

    records = [r for r in groups.values() if len(r["slides"]) >= min_stains]

    if verbose:
        n_short = len(groups) - len(records)
        print(f"  [{features_path}] {len(records)} biopsies from {len(bags)} slides")
        print(
            f"    dropped slides — no label: {dropped['no_label']}, "
            f"not in labels CSV: {dropped['not_in_csv']}, "
            f"stain not in vocabulary: {dropped['other_stain']}, "
            f"not authorized: {dropped['unauthorized']}"
        )
        if n_short:
            print(f"    dropped {n_short} biopsies with < {min_stains} stain(s)")
        counts = np.zeros(len(vocab), dtype=int)
        for r in records:
            for s in r["slides"]:
                counts[s] += 1
        avail = ", ".join(
            f"{name}: {c} ({100 * c / max(len(records), 1):.0f}%)"
            for name, c in zip(vocab.names, counts)
        )
        print(f"    stain availability — {avail}")

    return records


def split_records(records, table):
    """Split records into (train, val) by the labels table's split column.

    A biopsy goes to validation when any of its slides is in the val split
    (define_labels.py splits by patient, so a biopsy's slides always agree).
    """
    train, val = [], []
    for rec in records:
        is_val = (table.loc[rec["bag_names"], "split"] == "val").any()
        (val if is_val else train).append(rec)
    return train, val


# ---------------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------------
def _choose_indices(n: int, n_patches: int, random: bool) -> np.ndarray:
    """Return ``n_patches`` sorted row indices into a bag of ``n`` patches.

    Subsamples when ``n > n_patches`` and tiles when ``n < n_patches``.  With
    ``random=False`` the selection is evenly spaced and deterministic (used for
    validation / inference so scores are reproducible); with ``random=True`` it is
    a fresh random draw each epoch, which doubles as patch-dropout augmentation.
    Indices are sorted because h5py fancy indexing requires increasing selections.
    """
    if not random:
        return np.round(np.linspace(0, n - 1, n_patches)).astype(np.int64)
    if n >= n_patches:
        idx = np.random.permutation(n)[:n_patches]
    else:
        reps = -(-n_patches // n)  # ceil
        idx = np.concatenate([np.random.permutation(n) for _ in range(reps)])[:n_patches]
    idx.sort()
    return idx


def _bag_shape(path) -> tuple:
    """(n_patches, feat_dim) of a feature file, without reading the features."""
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return tuple(f["features"].shape)
    return tuple(np.load(path, mmap_mode="r").shape)


def _read_rows(path, rows: np.ndarray) -> np.ndarray:
    """Read the given (sorted, unique) feature rows from an .h5 or .npy bag."""
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return np.asarray(f["features"][rows])
    return np.asarray(np.load(path, mmap_mode="r")[rows])


def _read_stain_bag(paths, n_patches: int, random: bool):
    """Load ``n_patches`` feature rows from one stain's slide file(s).

    Slides of the same stain within a biopsy are treated as one concatenated bag.
    Only the selected rows are read from disk (h5py fancy indexing / .npy memmap), so
    a 100k-patch slide costs the same as a small one.  Rows containing non-finite
    values are dropped (UNI-v2 bags hold a few failed patches, and a single NaN row
    turns the whole loss NaN); the selection is then refilled from the surviving rows.

    Returns a ``(n_patches, D)`` float32 tensor, or None when nothing usable is left.
    """
    counts = [_bag_shape(p)[0] for p in paths]
    total = int(sum(counts))
    if total == 0:
        return None

    idx = _choose_indices(total, n_patches, random)
    chunks = []
    offset = 0
    for p, c in zip(paths, counts):
        sel = idx[(idx >= offset) & (idx < offset + c)] - offset
        if sel.size:
            uniq, inv = np.unique(sel, return_inverse=True)
            chunks.append(_read_rows(p, uniq)[inv])
        offset += c

    x = np.concatenate(chunks).astype(np.float32, copy=False)
    keep = np.isfinite(x).all(axis=1)
    if not keep.all():
        x = x[keep]
        if x.shape[0] == 0:
            return None
        x = x[_choose_indices(x.shape[0], n_patches, random)]
    return torch.from_numpy(np.ascontiguousarray(x))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiStainBiopsyDataset(Dataset):
    """One sample per biopsy: a fixed-size patch bag for every available stain.

    Arguments:
        records: Output of ``index_biopsies`` (possibly concatenated across cohorts
            and split by ``split_records``).
        n_stains: Size of the stain vocabulary — the first axis of ``X``.
        patches_per_stain: Patches sampled per stain (see the module docstring).
        random_subsample: Draw a fresh random patch subset on every access. Set True
            for training, False for validation / inference.
        feat_dim: Feature dimension. Read from the first record when None.
        stain_dropout: Probability of masking out each *available* stain, drawn
            independently per stain on every access (see ``__getitem__``). Training-time
            augmentation only — leave at 0.0 for validation / inference.
    """

    def __init__(
        self,
        records,
        n_stains: int,
        patches_per_stain: int = 512,
        random_subsample: bool = True,
        feat_dim: int = None,
        stain_dropout: float = 0.0,
    ) -> None:
        self.records = list(records)
        self.n_stains = n_stains
        self.patches_per_stain = patches_per_stain
        self.random_subsample = random_subsample
        self.stain_dropout = float(stain_dropout)
        if not 0.0 <= self.stain_dropout < 1.0:
            raise ValueError(f"stain_dropout must be in [0, 1), got {stain_dropout}.")
        self.feat_dim = feat_dim if feat_dim is not None else self._peek_feat_dim()

    def _peek_feat_dim(self) -> int:
        for rec in self.records:
            for paths in rec["slides"].values():
                return int(_bag_shape(paths[0])[1])
        raise ValueError("Cannot infer feat_dim: no records with feature files.")

    def __len__(self) -> int:
        return len(self.records)

    def _keep_stains(self, rec) -> list:
        """Stain indices to load for this access, after stain dropout.

        Each available stain is dropped independently with probability
        ``stain_dropout``.  Biopsies vary in which stains they have, so without this
        the model never sees a stain combination other than the one each biopsy
        happens to carry, and the aggregator is free to lean on whichever stain is
        near-universal.  Dropping is decided *before* the features are read, so a
        dropped stain costs no disk I/O.

        At least one stain always survives — an empty mask would make the
        aggregator's attention softmax undefined — so single-stain biopsies are never
        emptied, and an all-dropped draw falls back to one randomly kept stain.
        """
        stain_ids = list(rec["slides"])
        if self.stain_dropout <= 0.0 or len(stain_ids) <= 1:
            return stain_ids
        keep = [s for s in stain_ids if np.random.rand() >= self.stain_dropout]
        return keep or [stain_ids[np.random.randint(len(stain_ids))]]

    def __getitem__(self, i):
        rec = self.records[i]
        X = torch.zeros(
            self.n_stains, self.patches_per_stain, self.feat_dim, dtype=torch.float32
        )
        stain_mask = torch.zeros(self.n_stains, dtype=torch.bool)

        for stain_idx in self._keep_stains(rec):
            paths = rec["slides"][stain_idx]
            bag = _read_stain_bag(paths, self.patches_per_stain, self.random_subsample)
            if bag is None:
                continue
            X[stain_idx] = bag
            stain_mask[stain_idx] = True

        if not bool(stain_mask.any()):
            # Every stain of this biopsy failed to load (all-NaN features). Keep the
            # sample valid — an all-False mask would make the aggregator's softmax
            # undefined — by presenting one zero-filled stain.
            print(f"[WARN] {rec['biopsy']}: no usable patches; using a zero bag.")
            stain_mask[next(iter(rec["slides"]), 0)] = True

        return {
            "X": X,
            "stain_mask": stain_mask,
            "Y": torch.as_tensor(rec["label"], dtype=torch.float32).view(-1),
        }
