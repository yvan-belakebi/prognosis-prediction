"""
mil_utils.py — Shared utilities for MIL training and evaluation scripts.

Provides bag-discovery, stain filtering, val-split loading, patch subsampling,
biopsy-level sampling, and dataset construction — used by MIL.py,
classification_MIL.py, regression_MIL.py, and evaluate_survival.py.
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

# ---------------------------------------------------------------------------
# Resolve local torchmil package (same logic as in the training scripts so
# that this module can be imported standalone without sys.path pre-seeding).
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
_torchmil_root = os.path.join(_project_root, "torchmil")
if (
    os.path.isdir(os.path.join(_torchmil_root, "torchmil"))
    and _torchmil_root not in sys.path
):
    sys.path.insert(0, _torchmil_root)

from torchmil.datasets import ProcessedMILDataset  # noqa: E402


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

    The stain CSV stores biopsy_number and file_name (slide stem) as separate
    columns. This builds both the flat ('file_name') and nested
    ('biopsy_number/file_name') candidate bag names and intersects them with the
    bags actually on disk, so it works whether or not the WSI layout is nested.
    """
    if stain_csv is None or stain_csv.lower() == "none":
        return None
    df = pd.read_csv(stain_csv)
    rows = df.loc[df["stain"] == stain_filter]
    matching = set(rows["file_name"].astype(str))
    if "biopsy_number" in rows.columns:
        matching |= set(
            rows["biopsy_number"].astype(str) + "/" + rows["file_name"].astype(str)
        )
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


def load_authorized_slides(authorized_csv):
    """Load authorized slide basenames from a CSV/text file into a set.

    Same format as load_val_names: a CSV with a 'file_name' column (header row)
    or a headerless single-column file (one basename per row). Returns None when
    authorized_csv is None, meaning no restriction is applied.

    Entries are normalised to the bare slide stem so the list matches biopsy-
    nested bags on disk: any directory/biopsy prefix is stripped and a trailing
    slide/feature extension is removed (e.g. 'biopsy_3/slide1.svs' and
    'slide1.h5' both become 'slide1'). Matching against nested bags is then
    handled by _bag_authorized.
    """
    names = load_val_names(authorized_csv)
    if names is None:
        return None
    slide_exts = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn", ".vms",
                  ".vmu", ".npy", ".h5"}
    normalised = set()
    for n in names:
        stem = os.path.basename(n.replace("\\", "/"))
        root, ext = os.path.splitext(stem)
        if ext.lower() in slide_exts:
            stem = root
        normalised.add(stem)
    return normalised


def _bag_authorized(bag_name, authorized_slides):
    """True when a bag (flat 'slide' or nested 'biopsy/slide') is authorized.

    Matches on either the full bag name or its file basename (the part after the
    last '/'), so the authorized list may hold either form.
    """
    return (
        bag_name in authorized_slides
        or bag_name.rsplit("/", 1)[-1] in authorized_slides
    )


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


def _select_rows(bag, idx):
    """Return a copy of bag keeping only patch rows `idx` (X, coords, adj)."""
    new_bag = {}
    for k, v in bag.items():
        if k in ("X", "coords") and isinstance(v, torch.Tensor):
            new_bag[k] = v[idx]
        elif k == "adj" and isinstance(v, torch.Tensor):
            new_bag[k] = _subsample_adj(v, idx)
        else:
            new_bag[k] = v
    return new_bag


def make_collate_fn(base_collate, max_patches=None):
    """Wrap base_collate with per-bag NaN-row removal and optional subsampling.

    Some feature bags contain a handful of NaN patch rows (failed extractions);
    a single NaN row turns the whole loss NaN, so rows with any non-finite
    feature are dropped here. Then, when max_patches is set, bags larger than
    that are randomly subsampled.

    ProcessedMILDataset pre-computes adj in __getitem__, so both adj and the
    patch-level tensors (X, coords) are filtered here — before collate_fn stacks
    and pads them — to keep them aligned.
    """

    def _collate(bags):
        out = []
        for bag in bags:
            x = bag["X"]
            keep = torch.isfinite(x).all(dim=1)
            if not bool(keep.all()) and bool(keep.any()):
                bag = _select_rows(bag, keep.nonzero(as_tuple=True)[0])
            if max_patches is not None and bag["X"].shape[0] > max_patches:
                idx = torch.randperm(bag["X"].shape[0])[:max_patches]
                bag = _select_rows(bag, idx)
            out.append(bag)
        return base_collate(out)

    return _collate


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


def _allocate_max_biopsies(train_names_per_ds, max_biopsies):
    """Distribute a biopsy budget equally across datasets, redistributing surplus.

    Each dataset gets an equal share of max_biopsies. If a dataset has fewer
    biopsies than its share, all of them are kept and the leftover budget is
    redistributed to the remaining datasets. All bags belonging to a selected
    biopsy are included (no bag-level subsampling within a biopsy).

    Returns a list of bag-name lists, one per input dataset.
    """
    biopsy_groups: list[dict] = []
    for names in train_names_per_ds:
        groups: dict = {}
        for n in names:
            biopsy_id = n.rsplit("/", 1)[0] if "/" in n else n
            groups.setdefault(biopsy_id, []).append(n)
        biopsy_groups.append(groups)

    n_avail = [len(g) for g in biopsy_groups]
    selected_counts: list = [None] * len(train_names_per_ds)
    budget = max_biopsies
    undecided = list(range(len(train_names_per_ds)))

    while undecided:
        target = budget // len(undecided)
        if target == 0:
            for i in undecided:
                selected_counts[i] = 0
            break
        newly_saturated = [i for i in undecided if n_avail[i] <= target]
        if not newly_saturated:
            # All remaining datasets exceed the target — assign equally and stop.
            for i in undecided:
                selected_counts[i] = target
            break
        for i in newly_saturated:
            selected_counts[i] = n_avail[i]
            budget -= n_avail[i]
        undecided = [i for i in undecided if i not in newly_saturated]

    result = []
    for groups, count in zip(biopsy_groups, selected_counts):
        all_biopsies = list(groups.keys())
        if count is None or count >= len(all_biopsies):
            chosen = all_biopsies
        elif count == 0:
            chosen = []
        else:
            chosen = random.sample(all_biopsies, count)
        bags: list = []
        for b in chosen:
            bags.extend(groups[b])
        result.append(bags)

    return result


def build_dataset(
    features_paths,
    labels_paths,
    coords_paths,
    bag_keys,
    dist_thr,
    val_names=None,
    stain_csvs=None,
    stain_filter=None,
    scan_labels_fn=None,
    max_biopsies=None,
    file_ext=".h5",
    authorized_slides=None,
):
    """Build train and (optionally) val datasets from lists of feature/label paths.

    Parameters
    ----------
    authorized_slides : set[str], optional
        When provided, only bags whose file basename (or full bag name) is in
        this set are included in the train and val datasets. Pass ``None`` to
        keep all discovered bags.
    scan_labels_fn : callable(labels_path, bag_names) -> np.ndarray, optional
        When provided, called on each train and val partition to collect scalar
        labels (e.g. class integers or regression targets) for use by samplers
        or metrics.  Pass ``None`` for survival training where labels live inside
        the .npy bags themselves and no separate array is needed.
    max_biopsies : int, optional
        Cap on the total number of training biopsies across all datasets.
        The budget is split equally; if one dataset has fewer biopsies than its
        share, all of them are kept and the surplus is redistributed to the
        others.  Has no effect on the validation set.

    Returns
    -------
    (train_ds, val_ds, train_labels, val_labels)
        val_ds is None when val_names is None.
        train_labels and val_labels are None when scan_labels_fn is None.
    """
    stain_csvs = stain_csvs if stain_csvs is not None else [None] * len(features_paths)
    train_datasets, val_datasets = [], []
    train_labels_parts, val_labels_parts = [], []

    # --- Pass 1: collect and filter bag names per dataset --------------------
    all_train_names: list = []
    all_val_names_here: list = []

    for fp, lp, cp, sc in zip(features_paths, labels_paths, coords_paths, stain_csvs):
        filtered = get_filtered_bag_names(fp, sc, stain_filter)
        available = filtered if filtered is not None else discover_bags(fp)

        if authorized_slides is not None:
            n_before = len(available)
            available = [
                n for n in available if _bag_authorized(n, authorized_slides)
            ]
            if len(available) < n_before:
                print(
                    f"  Excluded {n_before - len(available)} bags not in the "
                    f"authorized slides list from {fp}"
                )

        labelled = set(discover_bags(lp, extensions=(".npy",)))
        n_before = len(available)
        available = [n for n in available if n in labelled]
        if len(available) < n_before:
            print(f"  Skipped {n_before - len(available)} bags with no label file in {lp}")

        if val_names is not None:
            train_names = [n for n in available if n not in val_names]
            val_names_here = [n for n in available if n in val_names]
        else:
            train_names = available
            val_names_here = []

        all_train_names.append(train_names)
        all_val_names_here.append(val_names_here)

    # --- Optional biopsy-level subsampling of the training set ---------------
    if max_biopsies is not None:
        def _count_biopsies(names):
            return len({(n.rsplit("/", 1)[0] if "/" in n else n) for n in names})

        before = [_count_biopsies(names) for names in all_train_names]
        all_train_names = _allocate_max_biopsies(all_train_names, max_biopsies)
        for fp, nb, names in zip(features_paths, before, all_train_names):
            na = _count_biopsies(names)
            print(f"  [{fp}] max_biopsies: {na}/{nb} biopsies selected ({len(names)} bags)")
        print(
            f"  Total: {sum(_count_biopsies(n) for n in all_train_names)}/"
            f"{sum(before)} biopsies selected"
        )

    # --- Pass 2: build ProcessedMILDataset objects ---------------------------
    for fp, lp, cp, train_names, val_names_here in zip(
        features_paths, labels_paths, coords_paths, all_train_names, all_val_names_here
    ):
        train_datasets.append(
            ProcessedMILDataset(
                features_path=fp,
                labels_path=lp,
                coords_path=cp,
                bag_keys=bag_keys,
                dist_thr=dist_thr,
                bag_names=train_names,
                file_ext=file_ext,
                label_ext=".npy",
            )
        )
        if scan_labels_fn is not None:
            train_labels_parts.append(scan_labels_fn(lp, train_names))

        if val_names_here:
            val_datasets.append(
                ProcessedMILDataset(
                    features_path=fp,
                    labels_path=lp,
                    coords_path=cp,
                    bag_keys=bag_keys,
                    dist_thr=dist_thr,
                    bag_names=val_names_here,
                    file_ext=file_ext,
                    label_ext=".npy",
                )
            )
            if scan_labels_fn is not None:
                val_labels_parts.append(scan_labels_fn(lp, val_names_here))

    train_ds = (
        train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    )
    val_ds = (
        (val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets))
        if val_datasets
        else None
    )
    train_labels = np.concatenate(train_labels_parts) if train_labels_parts else None
    val_labels = np.concatenate(val_labels_parts) if val_labels_parts else None

    return train_ds, val_ds, train_labels, val_labels
