"""nas_staging.py — Run GPU work on NAS-hosted WSIs via a local staging dir.

Reading whole-slide images straight off the mounted NAS makes TRIDENT's random
per-tile reads slow: the network latency, not the GPU, sets the pace. The fix
used by both the tiling and the feature-extraction pipelines is the same, so it
lives here once:

    copy a chunk of slides to local disk -> process it there -> delete it,
    while a background thread already stages the next chunk.

That overlaps the NAS transfer with the GPU work and bounds how much local disk
is in use (about `prefetch + 1` chunks at a time).

This module is mechanism, not policy. `stage_and_process` owns the copier
thread, the bounded queue and the cleanup of staged dirs, and it reports what
failed; deciding how to log those failures, and whether they should fail the
run, is left to the caller. It never calls sys.exit.

Callers describe their work as `StagedUnit`s and pass a `process` callback:

    def process(unit, staged_dir):
        stain, slides = unit.payload          # whatever the caller put there
        ...                                   # run the real work against staged_dir
        return ChunkResult(ok=True, failed_slides=[...])

    outcome = stage_and_process(units, wsi_root, staging_dir, prefetch, process)
"""

import os
import queue
import shutil
import subprocess
import threading
import time
from collections import namedtuple

# One chunk of work to stage and process.
#   seq       run-wide id; gives the chunk its own staging subdir, so the same
#             slide appearing in two chunks can never have one copy clobber the
#             other's mid-read.
#   label     human-readable name, used in progress and failure messages.
#   rel_paths paths of the files to copy, relative to the source root.
#   payload   whatever the caller needs back in its `process` callback.
StagedUnit = namedtuple("StagedUnit", "seq label rel_paths payload")

# What a `process` callback reports back.
#   ok            False if the chunk failed as a whole (e.g. the subprocess died
#                 even after retries) and its work should be retried on a re-run.
#   failed_slides identifiers of individual items that failed inside an otherwise
#                 successful chunk.
ChunkResult = namedtuple("ChunkResult", "ok failed_slides")

# What stage_and_process reports.
#   failed_chunks labels of chunks that failed as a whole, including copy errors.
#   failed_slides (label, slide identifier) pairs from chunks that otherwise ran.
StagingOutcome = namedtuple("StagingOutcome", "failed_chunks failed_slides")


def chunked(seq, size):
    """Yield successive size-length slices of seq."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def format_command(command):
    """Render a command list for printing, quoting the parts that hold spaces."""
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def child_env():
    """Environment for GPU subprocesses.

    We deliberately do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True:
    it uses CUDA virtual-memory APIs unsupported on the L40S vGPU this runs on,
    where it made even the first model-to-GPU allocation fail with a
    deterministic "CUDA error: out of memory". The environment is passed through
    untouched, so anyone who wants that allocator can export it themselves.
    """
    return os.environ.copy()


def copy_into_staging(rel_paths, source_root, dest_root):
    """Copy each file into dest_root, keeping its path relative to source_root.

    The relative layout is preserved rather than flattened so that callers whose
    slides live in a nested (e.g. biopsy-nested) tree keep working, and so two
    same-named files under different parents cannot collide.
    """
    for rel in rel_paths:
        dst = os.path.join(dest_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(source_root, rel), dst)


def gpu_free_gib(gpu):
    """Free memory (GiB) on GPU `gpu` via nvidia-smi, or None if it can't be read
    (nvidia-smi missing, a CPU run, etc.), in which case callers skip the gate."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def wait_for_gpu(gpu, min_free_gib, max_wait, poll=15):
    """Block until GPU `gpu` has `min_free_gib` free, or `max_wait` seconds pass.

    Guards against launching a chunk into a momentarily-full shared GPU, where it
    would OOM -- or, worse, be caught by a skip-errors flag and silently mark
    every slide failed. If nvidia-smi is unreadable the gate is a no-op; if the
    GPU never frees up within max_wait it proceeds anyway, leaving the caller's
    retry/skip paths to handle the fallout.

    Note this reads what nvidia-smi reports free, which on a vGPU does not always
    reflect what can actually be allocated; it is a guard, not a guarantee.
    """
    waited = 0
    while True:
        free = gpu_free_gib(gpu)
        if free is None or free >= min_free_gib:
            return
        if waited >= max_wait:
            print(f"    GPU {gpu}: only {free:.1f} GiB free after {waited}s, "
                  f"proceeding anyway")
            return
        print(f"    GPU {gpu}: {free:.1f} GiB free (< {min_free_gib}), "
              f"waiting {poll}s for it to clear")
        time.sleep(poll)
        waited += poll


def run_with_retries(command, env, retries, retry_wait):
    """Run command, retrying on a non-zero exit. Return True if it ever succeeds.

    Retrying is cheap because the tools driven here resume: a retried call skips
    the items it already finished, so it only redoes the one it died on.
    """
    for attempt in range(retries + 1):
        try:
            subprocess.run(command, check=True, env=env)
            return True
        except subprocess.CalledProcessError as error:
            if attempt < retries:
                print(f"    failed (attempt {attempt + 1}/{retries + 1}), "
                      f"retrying in {retry_wait}s: {error}")
                time.sleep(retry_wait)
            else:
                print(f"    failed after {retries + 1} attempts: {error}")
    return False


def stage_and_process(units, source_root, staging_dir, prefetch, process):
    """Stage each unit's files locally and hand the staged dir to `process`.

    A background thread copies the next chunk while the current one is being
    processed; the queue bound keeps at most `prefetch` chunks staged ahead,
    capping local disk use. Every staged dir is deleted once its chunk is done,
    including when `process` raises or the copy fails.

    `process(unit, staged_dir)` runs the caller's real work and returns a
    ChunkResult. Returning ok=False, or a copy failing, marks the chunk failed
    without stopping the run.

    Returns a StagingOutcome; the caller decides what to log and what exit code
    that deserves.
    """
    ready = queue.Queue(maxsize=max(1, prefetch))
    done = object()
    failed_chunks = []
    failed_slides = []

    def copier():
        for unit in units:
            dest = os.path.join(staging_dir, str(unit.seq))
            try:
                copy_into_staging(unit.rel_paths, source_root, dest)
            except Exception as error:  # transient NAS/read error: skip the chunk
                shutil.rmtree(dest, ignore_errors=True)
                ready.put((unit, error))
                continue
            ready.put((unit, None))
        ready.put(done)

    threading.Thread(target=copier, daemon=True).start()

    while True:
        item = ready.get()
        if item is done:
            break
        unit, error = item
        if error is not None:
            print(f"  {unit.label}: copy failed, skipping ({error})")
            failed_chunks.append(unit.label)
            continue
        dest = os.path.join(staging_dir, str(unit.seq))
        try:
            print(f"  {unit.label}: {len(unit.rel_paths)} slides staged -> {dest}")
            result = process(unit, dest)
            if not result.ok:
                failed_chunks.append(unit.label)
            failed_slides.extend(
                (unit.label, slide) for slide in (result.failed_slides or [])
            )
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    return StagingOutcome(failed_chunks, failed_slides)
