"""
benchmark_loader.py — Where does a multistain_MIL.py training step spend its time?

Times three things on the same training setup as train_fusion_models.sh:
  A. the DataLoader alone (read h5 + build batch + send to the main process)
  B. the full step (A + copy to GPU + forward + backward + optimizer step)
  C. the GPU alone (forward + backward on one batch already on the GPU)

If A is close to B, the GPU is waiting for data. If C is much smaller than B, the
GPU could go faster with faster loading.

Usage (from the project root):
    source stain_config.sh
    python python_scripts/MIL/benchmark_loader.py \
        --features_path "$FEATURES" --labels_path "$LABELS" --labels_csv "$LABELS_CSV" \
        --stains "${STAINS[@]}" --fusion unicorn --batch_size 32 --num_workers 14
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import multistain_MIL as M  # noqa: E402  (also sets up the torchmil / evaluate paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--stains", nargs="+", required=True)
    p.add_argument("--fusion", default="unicorn", choices=["unicorn", "early"])
    p.add_argument("--patches_per_stain", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=14)
    p.add_argument("--n_batches", type=int, default=20, help="Batches timed per test.")
    p.add_argument("--file_ext", default=".h5", choices=[".h5", ".npy"])
    args = p.parse_args()
    device = torch.device("cuda")

    vocab = M.StainVocabulary(args.stains)
    table = M.load_labels_table(args.labels_csv)
    records = M.index_biopsies(args.features_path, args.labels_path, table, vocab,
                               file_ext=args.file_ext, verbose=False)
    train_records, _, _ = M.split_records(records, table)
    ds = M.MultiStainBiopsyDataset(train_records, n_stains=len(vocab),
                                   patches_per_stain=args.patches_per_stain,
                                   random_subsample=True, stain_dropout=0.2,
                                   dtype=torch.float16)
    sampler = M.EventStratifiedBatchSampler.from_labels(
        [r["label"][1] for r in train_records], [r["label"][0] for r in train_records],
        args.batch_size)
    n = min(args.n_batches, len(sampler))
    print(f"{len(train_records)} train biopsies, timing {n} batches of {args.batch_size}")

    # Size of one batch and how much of it is padding for absent stains.
    t = time.perf_counter()
    sample = ds[0]
    print(f"one __getitem__ in the main process: {time.perf_counter() - t:.3f}s")
    present = sum(len(r["slides"]) for r in train_records) / len(train_records)
    mb = sample["X"].numel() * sample["X"].element_size() * args.batch_size / 1e6
    print(f"batch X: {mb:.0f} MB, mean {present:.1f} of {len(vocab)} stains present "
          f"({100 * present / len(vocab):.0f}% of X is real data)")

    def loader():
        return M.DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                            worker_init_fn=M._seed_worker, pin_memory=True)

    # A. loader alone (worker start-up is timed separately).
    t0 = time.perf_counter()
    it = iter(loader())
    next(it)
    t1 = time.perf_counter()
    for _ in range(n - 1):
        next(it)
    t2 = time.perf_counter()
    del it
    print(f"A. worker start-up + first batch: {t1 - t0:.1f}s")
    print(f"A. loader alone:  {(t2 - t1) / (n - 1):.3f}s / batch")

    if args.fusion == "early":
        model = M.EarlyFusionMIL(in_dim=ds.feat_dim, n_stains=len(vocab), out_dim=1)
    else:
        model = M.MultiStainFusion(in_dim=ds.feat_dim, n_stains=len(vocab), out_dim=1)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(batch):
        Y = batch["Y"].to(device)
        loss = M.cox_ph_loss(M._predict(model, batch, device), Y[:, 0], Y[:, 1])
        loss.backward()
        opt.step()
        opt.zero_grad()

    # B. full step.
    it = iter(loader())
    step(next(it))  # warm-up
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n - 1):
        step(next(it))
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    del it
    print(f"B. full step:     {(t2 - t1) / (n - 1):.3f}s / batch")

    # C. GPU alone, on one batch kept on the GPU.
    gpu_batch = {k: v.to(device) for k, v in sample_batch(loader()).items()}
    step(gpu_batch)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(n - 1):
        step(gpu_batch)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"C. GPU alone:     {(t2 - t1) / (n - 1):.3f}s / batch")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


def sample_batch(dl):
    return next(iter(dl))


if __name__ == "__main__":
    main()
