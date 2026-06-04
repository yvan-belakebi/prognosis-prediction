"""
mil_utils.py — Shared utilities for MIL training and evaluation scripts.

Provides bag-discovery, stain filtering, val-split loading, patch subsampling,
and biopsy-level sampling — used by MIL.py, classification_MIL.py,
regression_MIL.py, and evaluate_survival.py.
"""

import os
import random

import pandas as pd
import torch
from torch.utils.data import ConcatDataset


def discover_bags(base_dir, extensions=(".npy", ".h5")):
    """Return relative bag paths (no extension) for flat or biopsy-nested layouts.

    Flat   : base_dir/slide.h5           → 'slide'
    Nested : base_dir/biopsy_nr/slide.h5 → 'biopsy_nr/slide'
    Always uses forward slashes so the result is portable across platforms.
    """
    bags = []
    for entry in os.scandir(base_dir):
        if entry.is_file():
            stem, ext = os.path.splitext(entry.name)
            if ext in extensions:
                bags.append(stem)
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file():
                    stem, ext = os.path.splitext(sub.name)
                    if ext in extensions:
                        bags.append(f"{entry.name}/{stem}")
    return sorted(bags)


def get_filtered_bag_names(features_path, stain_csv, stain_filter):
    """Return sorted bag names from features_path whose Stain matches stain_filter.

    Returns None when stain_csv is None or 'none', meaning no filtering is applied
    and ProcessedMILDataset will auto-discover all bags in features_path.
    """
    if stain_csv is None or stain_csv.lower() == "none":
        return None
    df = pd.read_csv(stain_csv)
    matching = set(df.loc[df["stain"] == stain_filter, "file_name"].astype(str))
    available = set(discover_bags(features_path))
    return sorted(matching & available)


def load_val_names(val_csv):
    """Load slide basenames from a CSV into a set.

    Accepts a CSV with a 'file_name' column (header row) or a headerless
    single-column file (one basename per row). Returns None when val_csv is None.
    """
    if val_csv is None:
        return None
    raw = pd.read_csv(val_csv, header=None, dtype=str)
    col = raw.iloc[:, 0].str.strip()
    if col.iloc[0].lower() == "file_name":
        col = col.iloc[1:]
    return set(col)


def _subsample_adj(adj, idx):
    """Subsample a 2D sparse or dense adjacency matrix to the given indices.

    For sparse COO tensors the nonzero entries are filtered without ever
    materialising the full dense matrix, so memory stays proportional to nnz
    rather than n_patches².
    """
    if not adj.is_sparse:
        return adj[idx][:, idx]
    adj = adj.coalesce()
    row, col = adj.indices()
    vals = adj.values()
    n_new = len(idx)
    old2new = torch.full((adj.size(0),), -1, dtype=torch.long)
    old2new[idx] = torch.arange(n_new)
    keep = (old2new[row] >= 0) & (old2new[col] >= 0)
    return torch.sparse_coo_tensor(
        torch.stack([old2new[row[keep]], old2new[col[keep]]]),
        vals[keep],
        (n_new, n_new),
    )


def make_collate_fn(base_collate, max_patches=None):
    """Wrap base_collate with random patch subsampling applied per bag.

    ProcessedMILDataset pre-computes adj in __getitem__, so both adj and
    the patch-level tensors (X, coords) must be subsampled here — before
    collate_fn stacks and pads them — to keep the batched adj matrix at
    (batch_size, max_patches, max_patches).
    """
    if max_patches is None:
        return base_collate

    def _collate_and_subsample(bags):
        subsampled = []
        for bag in bags:
            n = bag["X"].shape[0]
            if n > max_patches:
                idx = torch.randperm(n)[:max_patches]
                new_bag = {}
                for k, v in bag.items():
                    if k in ("X", "coords") and isinstance(v, torch.Tensor):
                        new_bag[k] = v[idx]
                    elif k == "adj" and isinstance(v, torch.Tensor):
                        new_bag[k] = _subsample_adj(v, idx)
                    else:
                        new_bag[k] = v
                bag = new_bag
            subsampled.append(bag)
        return base_collate(subsampled)

    return _collate_and_subsample


def get_bag_names(dataset):
    """Extract ordered bag names from a ProcessedMILDataset or ConcatDataset."""
    if isinstance(dataset, ConcatDataset):
        names = []
        for d in dataset.datasets:
            names.extend(get_bag_names(d))
        return names
    return list(dataset.bag_names)


class BiopsySampler(torch.utils.data.Sampler):
    """Sample exactly one slide per biopsy per epoch.

    Slides are grouped by the biopsy directory prefix (the part before '/'
    in the bag name). At each epoch, one slide is chosen at random from each
    biopsy group, eliminating repeated survival labels in the Cox risk set
    for biopsies with multiple slides.
    """

    def __init__(self, dataset):
        bag_names = get_bag_names(dataset)
        groups = {}
        for i, name in enumerate(bag_names):
            biopsy_id = name.rsplit("/", 1)[0] if "/" in name else name
            groups.setdefault(biopsy_id, []).append(i)
        self._groups = list(groups.values())

    def __iter__(self):
        indices = [random.choice(g) for g in self._groups]
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self._groups)
