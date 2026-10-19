"""
mil_utils.py — Shared utilities for MIL training and evaluation scripts.

Provides bag discovery, the labels table (biopsy grouping, stain selection and
the train/val split), patch subsampling, biopsy-level sampling, and dataset
construction — used by MIL.py, regression_MIL.py, multistain_MIL.py and the
evaluation scripts.
"""

import csv
import os
import random
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch

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

from scipy.spatial import cKDTree  # noqa: E402
from torchmil.datasets import ProcessedMILDataset  # noqa: E402
from torchmil.utils import add_self_loops, normalize_adj  # noqa: E402


class BagDataset(ProcessedMILDataset):
    """ProcessedMILDataset without the unbounded in-memory bag cache.

    Upstream ``__getitem__`` stores every bag it has ever built in
    ``self.loaded_bags`` and never evicts. One pass over a 31k-bag
    training set therefore accumulates the entire feature archive in host RAM
    (tens of GB) and the process is killed by the OOM killer partway through
    the first epoch — a silent SIGKILL with no traceback.

    Rebuilding a bag from disk is cheap next to a training step, so the cache
    is kept only when the caller explicitly asked to preload everything with
    ``load_at_init=True``.
    """

    def __getitem__(self, index):
        bag = super().__getitem__(index)
        if not self.load_at_init:
            self.loaded_bags.pop(self.bag_names[index], None)
        return bag

    def _load_coords(self, name):
        """Patch coordinates in patch units, so ``dist_thr`` counts patches.

        TRIDENT stores level-0 pixel coordinates — neighbouring patches are
        ``patch_size_level0`` apart (448 for 224 px at 20x on a 40x scan) — and
        records that step as an attribute of the 'coords' dataset. Upstream
        builds the graph from the raw values, so with the default dist_thr of
        1.5 no two patches are ever neighbours and the graph models silently
        run on an empty graph. Dividing by the step puts neighbours 1 apart
        whatever the scan magnification: dist_thr 1.5 then links each patch to
        its 8 neighbours. Coordinates without the attribute (legacy .npy) are
        returned unchanged.
        """
        path = os.path.join(self.coords_path, name + self.file_ext)
        if not path.endswith(".h5"):
            return super()._load_coords(name)
        with h5py.File(path, "r") as f:
            coords = f["coords"][:]
            step = f["coords"].attrs.get("patch_size_level0")
        return coords / float(step) if step else coords

    def _build_adj(self, bag_dict):
        """Upstream's graph, built with one vectorised KD-tree query.

        torchmil's build_adj runs a Python loop with one KD-tree query per
        patch (~0.7 s for a 20k-patch slide), done again for every bag on
        every epoch — enough to leave the GPU idle. query_pairs returns the
        same edge set in one call (~40x faster). Feature-weighted edges
        (adj_with_dist) still go through upstream.
        """
        if self.adj_with_dist:
            return super()._build_adj(bag_dict)
        coords = bag_dict["coords"]
        n = coords.shape[0]
        pairs = cKDTree(coords).query_pairs(self.dist_thr, output_type="ndarray")
        edge_index = np.concatenate([pairs, pairs[:, ::-1]]).T.astype(np.int64)
        edge_weight = np.ones(edge_index.shape[1])
        norm_edge_weight = normalize_adj(edge_index, edge_weight, n_nodes=n)
        if n == 1:
            edge_index, norm_edge_weight = add_self_loops(edge_index, norm_edge_weight, n)
        return edge_index, edge_weight, norm_edge_weight


def discover_bags(base_dir, extensions=(".npy", ".h5")):
    """Return the sorted stems of the bag files directly under base_dir.

    Bags are stored flat (base_dir/<slide>.h5). Which biopsy a slide belongs to
    comes from the labels CSV (see load_labels_table), never from folders.
    """
    return sorted(
        os.path.splitext(e.name)[0]
        for e in os.scandir(base_dir)
        if e.is_file() and os.path.splitext(e.name)[1] in extensions
    )


def _bag_shape(file_path):
    """Return the shape of a bag's feature array, reading headers only."""
    if os.path.splitext(file_path)[1] == ".h5":
        with h5py.File(file_path, "r") as f:
            return f["features"].shape
    return np.load(file_path, mmap_mode="r").shape


def drop_empty_bags(features_path, bag_names, file_ext=".h5"):
    """Drop bags whose feature file holds no usable patches.

    TRIDENT writes an .h5 even when segmentation finds no tissue in a slide;
    those files carry a features array of shape (0,), which has no patch
    dimension and blows up in collate. Bags are filtered here, before the
    datasets are built, so that bag_names stays aligned with what the loaders
    yield — evaluate_survival.py pairs predictions with bag names positionally,
    so bags must never be dropped later, inside collate.

    Only file headers are read, so the scan is cheap even for tens of
    thousands of bags.

    Files that cannot be read at all (e.g. truncated by an interrupted feature
    extraction: "file signature not found") are skipped too, and listed, so one
    bad file does not stop a run; re-extract them to use those slides.
    """
    keep, unreadable = [], []
    for name in bag_names:
        try:
            shape = _bag_shape(os.path.join(features_path, name + file_ext))
        except OSError:
            unreadable.append(name)
            continue
        if len(shape) >= 2 and shape[0] > 0:
            keep.append(name)
    n_empty = len(bag_names) - len(keep) - len(unreadable)
    if n_empty:
        print(f"  Skipped {n_empty} bags with no patches in {features_path}")
    if unreadable:
        shown = ", ".join(f"{n}{file_ext}" for n in unreadable[:20])
        more = f" (+{len(unreadable) - 20} more)" if len(unreadable) > 20 else ""
        print(
            f"  Skipped {len(unreadable)} unreadable (corrupt or truncated) feature "
            f"files in {features_path}: {shown}{more}"
        )
    return keep


# ---------------------------------------------------------------------------
# Labels CSV: biopsy grouping, stains and the train/val split
# ---------------------------------------------------------------------------
def load_labels_table(labels_csv):
    """Load a define_labels.py / define_regression_labels.py CSV, indexed by slide.

    One row per slide (index: file_name, the bag stem) with at least the
    biopsy_number, stain and split columns. It is the single source for biopsy
    grouping (bag_group), stain selection (select_bags) and the train/val split.
    """
    df = pd.read_csv(
        labels_csv, dtype={"file_name": str, "biopsy_number": str, "patient": str}
    )
    missing = {"file_name", "biopsy_number", "stain", "split"} - set(df.columns)
    if missing:
        raise KeyError(f"{labels_csv} is missing column(s) {sorted(missing)}.")
    df["file_name"] = df["file_name"].str.strip()
    if df["file_name"].duplicated().any():
        raise ValueError(f"{labels_csv} lists some file_name more than once.")
    return df.set_index("file_name")


def bag_group(bag_name, table):
    """Biopsy id of a bag, from the labels table."""
    return table.at[bag_name, "biopsy_number"]


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


def load_authorized_slides(authorized_csv):
    """Load authorized slide stems from a CSV/text file into a set.

    Accepts a CSV with a 'file_name' column (header row) or a headerless
    single-column file (one name per row). Returns None when authorized_csv is
    None, meaning no restriction is applied. Entries are normalised to the bare
    slide stem: any directory prefix and a trailing slide/feature extension are
    removed (e.g. 'biopsy_3/slide1.svs' and 'slide1.h5' both become 'slide1').
    """
    if authorized_csv is None:
        return None
    col = pd.read_csv(authorized_csv, header=None, dtype=str).iloc[:, 0].str.strip()
    if col.iloc[0].lower() == "file_name":
        col = col.iloc[1:]
    slide_exts = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".scn", ".vms",
                  ".vmu", ".npy", ".h5", ".jpg", ".jpeg", ".png"}
    normalised = set()
    for n in col:
        stem = os.path.basename(n.replace("\\", "/"))
        root, ext = os.path.splitext(stem)
        normalised.add(root if ext.lower() in slide_exts else stem)
    return normalised


def select_bags(
    features_path, labels_path, table, stains=None, file_ext=".h5",
    authorized_slides=None,
):
    """Bags on disk that are usable for training/evaluation, in sorted order.

    Keeps a bag when it is listed in the labels table, its stain is one of
    ``stains`` (StainVocabulary specs; None keeps every stain), it is
    authorized, it has a label file, and its feature file holds patches. Each
    filter reports how many bags it dropped.
    """
    names = discover_bags(features_path, extensions=(file_ext,))

    def _keep(names, keep, why):
        kept = [n for n in names if keep(n)]
        if len(kept) < len(names):
            print(f"  Skipped {len(names) - len(kept)} bags {why}")
        return kept

    names = _keep(names, lambda n: n in table.index, "not in the labels CSV")
    if stains:
        vocab = StainVocabulary(stains)
        on_disk = names
        names = _keep(
            names,
            lambda n: vocab.index_of(table.at[n, "stain"]) is not None,
            f"with a stain outside {vocab.names}",
        )
        if not names:
            counts = table.loc[on_disk, "stain"].value_counts()
            available = ", ".join(f"{s!r} ({c})" for s, c in counts.items())
            raise ValueError(
                f"No bag has a stain matching {vocab.names}. Stain names of the bags "
                f"in {features_path} (slide counts): {available}"
            )
    if authorized_slides is not None:
        names = _keep(
            names, lambda n: n in authorized_slides, "not in the authorized slides list"
        )
    labelled = set(discover_bags(labels_path, extensions=(".npy",)))
    names = _keep(names, lambda n: n in labelled, f"with no label file in {labels_path}")
    return drop_empty_bags(features_path, names, file_ext)


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

    Returns a picklable object rather than a closure, so DataLoader workers can
    receive it under the 'spawn' start method (Windows, macOS) as well as fork.
    """
    return _Collate(base_collate, max_patches)


class _Collate:
    def __init__(self, base_collate, max_patches):
        self.base_collate = base_collate
        self.max_patches = max_patches

    def __call__(self, bags):
        out = []
        for bag in bags:
            x = bag["X"]
            keep = torch.isfinite(x).all(dim=1)
            if not bool(keep.all()) and bool(keep.any()):
                bag = _select_rows(bag, keep.nonzero(as_tuple=True)[0])
            if self.max_patches is not None and bag["X"].shape[0] > self.max_patches:
                idx = torch.randperm(bag["X"].shape[0])[: self.max_patches]
                bag = _select_rows(bag, idx)
            out.append(bag)
        return self.base_collate(out)


class BiopsySampler(torch.utils.data.Sampler):
    """Sample exactly one slide per biopsy per epoch.

    Slides are grouped by biopsy (bag_group, from the labels table). At each
    epoch, one slide is chosen at random from each biopsy group, eliminating
    repeated survival labels in the Cox risk set for biopsies with multiple
    slides.
    """

    def __init__(self, dataset, table):
        groups = {}
        for i, name in enumerate(dataset.bag_names):
            groups.setdefault(bag_group(name, table), []).append(i)
        self._groups = list(groups.values())

    def __iter__(self):
        indices = [random.choice(g) for g in self._groups]
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self._groups)


class EventStratifiedBatchSampler(torch.utils.data.Sampler):
    """Batches with an even share of events, spread over follow-up time.

    The Cox loss is computed within each batch: every event is ranked against
    the batch members still at risk at its time. With random batches the event
    count varies a lot from batch to batch (and can be zero, which gives no
    gradient at all), so some steps learn from a handful of comparisons and
    others from many. Here events and censored samples are dealt out
    separately, so every batch gets floor/ceil(n_events / n_batches) events.

    Within each group, samples are sorted by follow-up time and dealt one
    chunk of ``n_batches`` at a time, each chunk's members going to different
    batches in random order. Every batch thus spans the whole range of times,
    which gives each event more comparisons in its risk set. Batch membership
    is still re-randomised every epoch.

    A unit is a list of dataset indices sharing one label, of which each epoch
    draws one: build with ``from_bags`` (slides; with ``one_per_biopsy`` the unit
    is the biopsy, as in BiopsySampler) or ``from_labels`` (one index per unit,
    e.g. the biopsy records of multistain_MIL.py). Uses Python's ``random``,
    whose state --resume restores.
    """

    def __init__(self, units, event, time, batch_size):
        self._units = units
        self._event = np.asarray(event, dtype=int)
        self._time = np.asarray(time, dtype=float)
        self.n_batches = max(1, -(-len(self._units) // batch_size))  # ceil

    @classmethod
    def from_bags(cls, dataset, table, batch_size, one_per_biopsy=False):
        """Slides of a BagDataset; event and time from the labels table."""
        names = dataset.bag_names
        units = {}
        for i, name in enumerate(names):
            key = bag_group(name, table) if one_per_biopsy else name
            units.setdefault(key, []).append(i)
        units = list(units.values())
        first = [names[u[0]] for u in units]
        return cls(
            units,
            table.loc[first, "event"].to_numpy(dtype=int),
            table.loc[first, "time"].to_numpy(dtype=float),
            batch_size,
        )

    @classmethod
    def from_labels(cls, event, time, batch_size):
        """One dataset index per unit, with its event flag and follow-up time."""
        return cls([[i] for i in range(len(event))], event, time, batch_size)

    def events_per_batch(self):
        """(min, max) events per batch — fixed by construction."""
        n_events = int(self._event.sum())
        return n_events // self.n_batches, -(-n_events // self.n_batches)

    def __iter__(self):
        batches = [[] for _ in range(self.n_batches)]
        for flag in (1, 0):
            group = np.flatnonzero(self._event == flag)
            group = group[np.argsort(self._time[group], kind="stable")]
            for start in range(0, len(group), self.n_batches):
                chunk = group[start:start + self.n_batches]
                slots = random.sample(range(self.n_batches), len(chunk))
                for unit, slot in zip(chunk, slots):
                    batches[slot].append(random.choice(self._units[unit]))
        random.shuffle(batches)
        for batch in batches:
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def _sample_biopsies(bag_names, table, max_biopsies):
    """Keep every slide of at most ``max_biopsies`` randomly chosen biopsies."""
    groups = {}
    for n in bag_names:
        groups.setdefault(bag_group(n, table), []).append(n)
    chosen = random.sample(sorted(groups), min(max_biopsies, len(groups)))
    print(f"  max_biopsies: {len(chosen)}/{len(groups)} biopsies selected")
    return sorted(n for b in chosen for n in groups[b])


def build_dataset(
    features_path,
    labels_path,
    coords_path,
    bag_keys,
    dist_thr,
    table,
    stains=None,
    scan_labels_fn=None,
    max_biopsies=None,
    file_ext=".h5",
    authorized_slides=None,
):
    """Build the train and val datasets from one feature/label folder.

    Parameters
    ----------
    table : DataFrame
        From load_labels_table. Decides which bags exist (select_bags), the
        biopsy of each bag, and the train/val split (its ``split`` column).
    stains : list[str], optional
        StainVocabulary specs (e.g. ``["PAS", "HES|HE"]``) of the stains to keep;
        ``None`` keeps every stain.
    authorized_slides : set[str], optional
        When provided, only bags in this set are included.
    scan_labels_fn : callable(labels_path, bag_names) -> np.ndarray, optional
        When provided, called on the train and val partitions to collect scalar
        labels (e.g. regression targets) for use by samplers or metrics.  Pass
        ``None`` for survival training where labels live inside the .npy bags
        themselves and no separate array is needed.
    max_biopsies : int, optional
        Cap on the number of training biopsies (all slides of a selected biopsy
        are kept). Has no effect on the validation set.

    Returns
    -------
    (train_ds, val_ds, train_labels, val_labels)
        val_ds is None when no bag is in the val split.
        train_labels and val_labels are None when scan_labels_fn is None.
    """
    names = select_bags(
        features_path, labels_path, table, stains, file_ext, authorized_slides
    )
    split = table.loc[names, "split"].to_numpy()
    train_names = [n for n, s in zip(names, split) if s == "train"]
    val_names = [n for n, s in zip(names, split) if s == "val"]
    if max_biopsies is not None:
        train_names = _sample_biopsies(train_names, table, max_biopsies)

    def _make(bag_names):
        return BagDataset(
            features_path=features_path,
            labels_path=labels_path,
            coords_path=coords_path,
            bag_keys=bag_keys,
            dist_thr=dist_thr,
            bag_names=bag_names,
            file_ext=file_ext,
            label_ext=".npy",
        )

    train_ds = _make(train_names)
    val_ds = _make(val_names) if val_names else None
    train_labels = val_labels = None
    if scan_labels_fn is not None:
        train_labels = scan_labels_fn(labels_path, train_names)
        val_labels = scan_labels_fn(labels_path, val_names) if val_names else None
    return train_ds, val_ds, train_labels, val_labels


# ---------------------------------------------------------------------------
# Resumable checkpoints (MIL.py, multistain_MIL.py)
# ---------------------------------------------------------------------------
def save_last_checkpoint(path, model, optimizer, epoch, best_metric, best_epoch, args):
    """Write everything needed to resume after ``epoch``, atomically.

    The weights-only checkpoints (_epochN, _best, final) stay as they are for the
    evaluation scripts; this one adds the optimizer state (Adam's moment
    estimates), the epoch, the best val metric so far, the torch / Python /
    NumPy RNG states and the run's arguments. It is written to a temporary file
    and renamed, so a crash during the save leaves the previous checkpoint intact.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_last_checkpoint(path, model, optimizer, args, device, parser, arch_args):
    """Restore a save_last_checkpoint file; return (epoch, best_metric, best_epoch).

    ``arch_args`` names the arguments that define the weight shapes: a resume
    with any of them changed cannot load the weights, so it is refused with a
    clear message. Other changed arguments only print a warning; --lr and
    --weight_decay from the resume command line override the saved ones.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    saved = state["args"]
    changed = [k for k in arch_args if saved.get(k) != getattr(args, k)]
    if changed:
        parser.error(
            "--resume: the checkpoint was trained with a different architecture ("
            + ", ".join(f"{k}={saved.get(k)!r}, now {getattr(args, k)!r}" for k in changed)
            + "). Pass the same values to resume."
        )
    ignore = {"resume", "epochs", "num_workers", "mem_debug"}
    for k, v in sorted(vars(args).items()):
        if k not in ignore and k not in arch_args and k in saved and saved[k] != v:
            print(f"Warning: --{k} was {saved[k]!r} in the checkpoint, now {v!r}.")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    # The optimizer state carries the old learning rate and weight decay; the
    # resume command line wins, so either can be changed mid-run. Checkpoints
    # from before --weight_decay were saved by Adam, whose state carries
    # decoupled_weight_decay=False; it is forced back on so the decay stays
    # AdamW's (PyTorch versions without the key always decouple in AdamW).
    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["weight_decay"] = args.weight_decay
        if "decoupled_weight_decay" in group:
            group["decoupled_weight_decay"] = True
    # map_location moved every tensor to `device`, but the CPU generator only
    # accepts a CPU ByteTensor.
    torch.set_rng_state(state["torch_rng"].cpu())
    random.setstate(state["python_rng"])
    if "numpy_rng" in state:  # absent in MIL.py checkpoints from before 2026-10-19
        np.random.set_state(state["numpy_rng"])
    # "best_cindex" is the key MIL.py checkpoints used before it was shared.
    best = state.get("best_metric", state.get("best_cindex"))
    return state["epoch"], best, state["best_epoch"]


# ---------------------------------------------------------------------------
# GPU memory tracing
# ---------------------------------------------------------------------------
_MB = 1024 ** 2


def _host_mem_mb():
    """(this process's RSS, system MemAvailable) in MB; (-1, -1) off Linux.

    The GPU is not the only memory that ends a run: a host-RAM exhaustion shows
    up as a bare "Killed" (SIGKILL from the OOM killer) with no traceback, so
    the trace records both.
    """
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / _MB
        avail = -1
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024
                    break
        return round(rss), round(avail)
    except Exception:
        return -1, -1


class GpuMemLogger:
    """Per-step CUDA memory trace, for OOMs that only strike occasionally.

    Each row separates three things that an OOM message conflates:

    - ``alloc_mb`` / ``peak_mb`` — tensors this process holds, and its peak
      during the step. Rising at constant batch shape means we leak.
    - ``reserved_mb`` — what the caching allocator keeps from the driver. It
      only ever grows; a gap between reserved and peak is fragmentation.
    - ``other_mb`` — device memory that is neither free nor reserved by us:
      this process's CUDA context (a fixed few hundred MB) plus *every other
      process on the GPU*. Jumps here are somebody else's job arriving, which
      no amount of tuning on our side can prevent.

    ``bag_n`` (the padded patch count) is logged alongside, because activation
    memory scales with the largest slide in the batch, so a rare huge slide and
    external pressure both look like "it OOMs sometimes" from the outside.
    """

    HEADER = [
        "t", "phase", "step", "batch_b", "bag_n", "adj_nnz",
        "alloc_mb", "peak_mb", "reserved_mb", "free_mb", "total_mb", "other_mb",
        "rss_mb", "sys_avail_mb", "oom",
    ]

    def __init__(self, log_dir, filename="gpu_mem.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.phase = ""
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.HEADER)

    def set_phase(self, phase):
        self.phase = phase

    def step_start(self):
        torch.cuda.reset_peak_memory_stats()

    def log(self, step, batch, oom=False):
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        x = batch["X"]
        adj = batch.get("adj", None)
        row = [
            f"{time.time():.1f}", self.phase, step,
            x.shape[0], x.shape[1],
            adj._nnz() if adj is not None and adj.is_sparse else -1,
            round(torch.cuda.memory_allocated() / _MB),
            round(torch.cuda.max_memory_allocated() / _MB),
            round(reserved / _MB),
            round(free / _MB),
            round(total / _MB),
            round((total - free - reserved) / _MB),
            *_host_mem_mb(),
            int(oom),
        ]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        return dict(zip(self.HEADER, row))

    def report_oom(self, step, batch):
        """Log the failing step and print who else was on the GPU at that moment."""
        row = self.log(step, batch, oom=True)
        print("")
        print(
            f"[OOM] {self.phase} step {step}: batch {row['batch_b']}x{row['bag_n']} patches | "
            f"ours alloc {row['alloc_mb']} MB, peak {row['peak_mb']} MB, "
            f"reserved {row['reserved_mb']} MB | device free {row['free_mb']} MB | "
            f"context+other processes {row['other_mb']} MB | "
            f"host RSS {row['rss_mb']} MB, system available {row['sys_avail_mb']} MB",
            flush=True,
        )
        try:
            procs = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            print(f"[OOM] processes on the GPU (mine is pid {os.getpid()}):", flush=True)
            print(procs, flush=True)
        except Exception as e:  # nvidia-smi missing or wedged: not worth failing over
            print(f"[OOM] nvidia-smi unavailable: {e}", flush=True)
        print(f"[OOM] trace written to {self.path}", flush=True)


# ---------------------------------------------------------------------------
# Loss logging
# ---------------------------------------------------------------------------
class LossLogger:
    """CSV loss log + loss-curve plot, shared by the MIL training scripts.

    Writes a header on construction and one row per log() call.  This base
    handles the survival case (train/val loss, single plot panel); subclasses
    set COLUMNS and override log()/save_plot() to add metrics (regression adds
    val_mae / val_rmse and a second panel).
    """

    COLUMNS = ["epoch", "train_loss", "val_loss"]

    def __init__(self, log_dir: str, resume_epoch: int = None):
        """Start a new log, or — with ``resume_epoch`` — continue the existing one.

        On resume, rows after ``resume_epoch`` are dropped (epochs logged after
        the checkpoint was written, e.g. by a run that then crashed) and new
        rows are appended, so the log and plot cover the whole training.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "loss_log.csv")
        self.plot_path = os.path.join(log_dir, "loss_curves.png")
        if resume_epoch is not None and os.path.isfile(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if list(df.columns) == self.COLUMNS:
                df[df["epoch"] <= resume_epoch].to_csv(self.csv_path, index=False)
                return
            print(f"Warning: {self.csv_path} has other columns — starting a new log.")
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)

    def _append(self, row):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def log(self, epoch, train_loss, val_loss=None):
        self._append(
            [epoch, f"{train_loss:.6f}", "" if val_loss is None else f"{val_loss:.6f}"]
        )

    def save_plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping loss plot.")
            return
        df = pd.read_csv(self.csv_path)
        df["val_loss"] = pd.to_numeric(df["val_loss"], errors="coerce")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["epoch"], df["train_loss"], label="train loss")
        val = df[df["val_loss"].notna()]
        if not val.empty:
            ax.plot(val["epoch"], val["val_loss"], linestyle="--", label="val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)
        print(f"Loss plot saved to {self.plot_path}")
