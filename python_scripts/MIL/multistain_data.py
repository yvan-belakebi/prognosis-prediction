"""
multistain_data.py — Biopsy-level, multi-stain feature bags for the UNICORN-style
fusion model (see multistain_model.py).

The single-stain scripts (MIL.py, regression_MIL.py) treat one *slide* as one bag.
Here the unit of prediction is the *biopsy*: every slide of the biopsy is grouped by
its stain, so one sample is a stack of per-stain patch bags.

Input layout (produced by run_trident_stain_feats.py + reorganize_trident_feats.py):

    features_path/{biopsy_nr}/{slide_id}.h5     keys 'features' (N, D), 'coords' (N, 2)
    labels_path/{biopsy_nr}/{slide_id}.npy      [time, event]  or  scalar (regression)

The stain of each slide comes from the labels CSV written by define_labels.py
(columns: biopsy_number, file_name, stain).  All slides of a biopsy carry the same
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

Returned sample (all tensors, so torch's default collate stacks them):

    X          (n_stains, patches_per_stain, D)  float32, zeros where the stain is absent
    stain_mask (n_stains,)                       bool, True where the stain is present
    Y          (n_label,)                        float32, [time, event] or [target]
"""

import os
import re

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mil_utils import discover_bags, _bag_authorized


# ---------------------------------------------------------------------------
# Stain vocabulary
# ---------------------------------------------------------------------------
def canon_stain(name) -> str:
    """Canonical key for a stain name: lowercase, alphanumerics only.

    The registry stain column is free text with case and punctuation variants
    ('Congo' / 'CONGO', 'IgG - IgM - IgA'), so names are compared on this key.
    """
    if name is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


class StainVocabulary:
    """Ordered list of model stains, each optionally covering several source names.

    A stain spec is a ``'|'``-separated alias group whose first element is the
    display name, e.g. ``'HES|HE'`` maps both source stains onto one model stain.
    ``index_of`` returns the model stain index for a raw CSV stain name, or None
    when that stain is not part of the vocabulary.
    """

    def __init__(self, specs):
        self.names = []
        self._lookup = {}
        for i, spec in enumerate(specs):
            aliases = [a.strip() for a in str(spec).split("|") if a.strip()]
            if not aliases:
                raise ValueError(f"Empty stain spec: {spec!r}")
            self.names.append(aliases[0])
            for alias in aliases:
                key = canon_stain(alias)
                if key in self._lookup:
                    raise ValueError(
                        f"Stain alias {alias!r} is assigned to both "
                        f"{self.names[self._lookup[key]]!r} and {aliases[0]!r}."
                    )
                self._lookup[key] = i

    def __len__(self):
        return len(self.names)

    def index_of(self, raw_name):
        return self._lookup.get(canon_stain(raw_name))


def load_stain_lookup(stain_csv) -> dict:
    """Map bag name -> raw stain string from a define_labels.py CSV.

    Both the nested ('biopsy_number/file_name') and the flat ('file_name') key are
    registered so the lookup works whichever layout the features are stored in.
    """
    df = pd.read_csv(stain_csv, dtype=str)
    for col in ("file_name", "stain"):
        if col not in df.columns:
            raise ValueError(f"{stain_csv} must contain a '{col}' column.")
    lookup = {}
    has_biopsy = "biopsy_number" in df.columns
    for row in df.itertuples(index=False):
        stain = getattr(row, "stain")
        if stain is None or (isinstance(stain, float) and np.isnan(stain)):
            continue
        name = str(getattr(row, "file_name"))
        lookup[name] = stain
        if has_biopsy:
            biopsy = getattr(row, "biopsy_number")
            if biopsy and str(biopsy) not in ("nan", "None"):
                lookup[f"{biopsy}/{name}"] = stain
    return lookup


# ---------------------------------------------------------------------------
# Indexing: features + labels + stains -> one record per biopsy
# ---------------------------------------------------------------------------
def index_biopsies(
    features_path,
    labels_path,
    stain_csv,
    vocab,
    file_ext=".h5",
    authorized_slides=None,
    min_stains=1,
    verbose=True,
) -> list:
    """Group the slides under ``features_path`` into one record per biopsy.

    Returns a list of dicts::

        {"biopsy": str,                       # biopsy directory name (bag prefix)
         "source": features_path,
         "label": np.ndarray,                 # from the first labelled slide
         "slides": {stain_idx: [h5 path, ...]},
         "bag_names": [bag name, ...]}

    Slides are dropped when they have no label file, no stain in ``stain_csv``, a
    stain outside ``vocab``, or are excluded by ``authorized_slides``.  Biopsies
    left with fewer than ``min_stains`` stains are dropped.
    """
    lookup = load_stain_lookup(stain_csv)
    bags = discover_bags(features_path, extensions=(file_ext,))
    labelled = set(discover_bags(labels_path, extensions=(".npy",)))

    dropped = {"unauthorized": 0, "no_label": 0, "no_stain": 0, "other_stain": 0}
    groups = {}
    for bag in bags:
        if authorized_slides is not None and not _bag_authorized(bag, authorized_slides):
            dropped["unauthorized"] += 1
            continue
        if bag not in labelled:
            dropped["no_label"] += 1
            continue
        raw_stain = lookup.get(bag, lookup.get(bag.rsplit("/", 1)[-1]))
        if raw_stain is None:
            dropped["no_stain"] += 1
            continue
        stain_idx = vocab.index_of(raw_stain)
        if stain_idx is None:
            dropped["other_stain"] += 1
            continue

        biopsy = bag.rsplit("/", 1)[0] if "/" in bag else bag
        rec = groups.setdefault(
            biopsy,
            {
                "biopsy": biopsy,
                "source": features_path,
                "label": None,
                "slides": {},
                "bag_names": [],
            },
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
            f"no stain in CSV: {dropped['no_stain']}, "
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


def split_records(records, val_names):
    """Split records into (train, val) using a set of slide or biopsy names.

    A biopsy goes to validation when its biopsy id, or any of its slide bag names
    (full 'biopsy/slide' or bare slide stem), appears in ``val_names``.  Returns
    ``(records, [])`` when ``val_names`` is None.
    """
    if val_names is None:
        return records, []
    train, val = [], []
    for rec in records:
        keys = {rec["biopsy"]}
        for bag in rec["bag_names"]:
            keys.add(bag)
            keys.add(bag.rsplit("/", 1)[-1])
        (val if keys & val_names else train).append(rec)
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
    """

    def __init__(
        self,
        records,
        n_stains: int,
        patches_per_stain: int = 512,
        random_subsample: bool = True,
        feat_dim: int = None,
    ) -> None:
        self.records = list(records)
        self.n_stains = n_stains
        self.patches_per_stain = patches_per_stain
        self.random_subsample = random_subsample
        self.feat_dim = feat_dim if feat_dim is not None else self._peek_feat_dim()

    def _peek_feat_dim(self) -> int:
        for rec in self.records:
            for paths in rec["slides"].values():
                return int(_bag_shape(paths[0])[1])
        raise ValueError("Cannot infer feat_dim: no records with feature files.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        X = torch.zeros(
            self.n_stains, self.patches_per_stain, self.feat_dim, dtype=torch.float32
        )
        stain_mask = torch.zeros(self.n_stains, dtype=torch.bool)

        for stain_idx, paths in rec["slides"].items():
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
