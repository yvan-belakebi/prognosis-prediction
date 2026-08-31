"""
monitor_gpu.py — Sample GPU memory and the processes holding it, to CSV.

Training OOMs that come and go are usually not caused by the training process
alone: on a shared GPU another user's job (or a stale process from an earlier
crashed run) can take several GiB at any moment, and PyTorch only reports it as
"non-PyTorch memory". This samples both the device totals and the per-process
breakdown, so the culprit is visible after the fact.

Usage:
    # terminal 1 — start before the training run, Ctrl-C when done
    python python_scripts/monitor_gpu.py --out results/gpu_watch

    # terminal 2
    python python_scripts/MIL/MIL.py ...

    # afterwards
    python python_scripts/monitor_gpu.py --report results/gpu_watch
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict

DEVICE_FIELDS = "index,uuid,memory.total,memory.used,memory.free,utilization.gpu"
PROC_FIELDS = "gpu_uuid,pid,process_name,used_gpu_memory"


def _query(what, fields):
    """Run one nvidia-smi CSV query, returning a list of split rows."""
    out = subprocess.run(
        ["nvidia-smi", f"--query-{what}={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def _owner(pid):
    """Username owning a pid, or '?' — tells your processes from other people's."""
    try:
        import pwd

        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except Exception:
        return "?"


def sample(dev_writer, proc_writer, gpu_filter):
    ts = time.time()
    uuid_to_index = {}
    for index, uuid, total, used, free, util in _query("gpu", DEVICE_FIELDS):
        uuid_to_index[uuid] = index
        if gpu_filter is not None and int(index) != gpu_filter:
            continue
        dev_writer.writerow([f"{ts:.1f}", index, total, used, free, util])
    for uuid, pid, name, used in _query("compute-apps", PROC_FIELDS):
        index = uuid_to_index.get(uuid, "?")
        if gpu_filter is not None and index != str(gpu_filter):
            continue
        # "[Not Found]" appears for processes in another container/namespace.
        used = used if used.isdigit() else "0"
        proc_writer.writerow([f"{ts:.1f}", index, pid, _owner(pid), name, used])


def watch(out_dir, interval, gpu_filter):
    os.makedirs(out_dir, exist_ok=True)
    dev_path = os.path.join(out_dir, "gpu_device.csv")
    proc_path = os.path.join(out_dir, "gpu_procs.csv")
    with open(dev_path, "w", newline="") as df, open(proc_path, "w", newline="") as pf:
        dev_writer, proc_writer = csv.writer(df), csv.writer(pf)
        dev_writer.writerow(["t", "gpu", "total_mb", "used_mb", "free_mb", "util_pct"])
        proc_writer.writerow(["t", "gpu", "pid", "user", "name", "used_mb"])
        print(f"Sampling every {interval}s → {dev_path}, {proc_path}  (Ctrl-C to stop)")
        try:
            while True:
                sample(dev_writer, proc_writer, gpu_filter)
                df.flush()
                pf.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")


def report(out_dir):
    """Summarise a finished watch: free-memory low points and who was resident."""
    dev_path = os.path.join(out_dir, "gpu_device.csv")
    proc_path = os.path.join(out_dir, "gpu_procs.csv")
    if not os.path.exists(dev_path):
        sys.exit(f"No samples found in {out_dir} — run the watcher first.")

    with open(dev_path) as f:
        dev = [r for r in csv.DictReader(f)]
    if not dev:
        sys.exit("Sample file is empty.")
    t0 = float(dev[0]["t"])

    by_gpu = defaultdict(list)
    for r in dev:
        by_gpu[r["gpu"]].append(r)
    for gpu, rows in sorted(by_gpu.items()):
        free = [int(r["free_mb"]) for r in rows]
        low = rows[free.index(min(free))]
        span = float(rows[-1]["t"]) - t0
        print(
            f"GPU {gpu}: {len(rows)} samples over {span / 60:.1f} min | "
            f"total {rows[0]['total_mb']} MB | free min {min(free)} MB "
            f"(at t+{float(low['t']) - t0:.0f}s), max {max(free)} MB, "
            f"swing {max(free) - min(free)} MB"
        )

    if not os.path.exists(proc_path):
        return
    with open(proc_path) as f:
        procs = [r for r in csv.DictReader(f)]
    me = _owner(os.getpid())
    agg = defaultdict(lambda: {"user": "", "name": "", "peak": 0, "first": None, "last": None})
    for r in procs:
        a = agg[(r["gpu"], r["pid"])]
        a["user"], a["name"] = r["user"], r["name"]
        a["peak"] = max(a["peak"], int(r["used_mb"]))
        a["first"] = a["first"] if a["first"] is not None else float(r["t"])
        a["last"] = float(r["t"])
    print("\nProcesses seen on the GPU (peak MB, lifetime relative to first sample):")
    for (gpu, pid), a in sorted(agg.items(), key=lambda kv: -kv[1]["peak"]):
        mine = " <- this user" if a["user"] == me else ""
        print(
            f"  gpu{gpu} pid {pid:>8} {a['user']:>12} {a['peak']:>7} MB  "
            f"t+{a['first'] - t0:.0f}s..t+{a['last'] - t0:.0f}s  {a['name']}{mine}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/gpu_watch", help="Directory for the sample CSVs.")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between samples.")
    p.add_argument("--gpu", type=int, default=None, help="Only record this GPU index.")
    p.add_argument("--report", metavar="DIR", help="Summarise a finished watch instead of sampling.")
    args = p.parse_args()
    if args.report:
        report(args.report)
    else:
        watch(args.out, args.interval, args.gpu)


if __name__ == "__main__":
    main()
