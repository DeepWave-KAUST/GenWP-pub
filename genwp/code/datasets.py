"""
Data loading pipeline for training the Generative Wave Propagator (GenWP).

This module provides an infinite-loop data generator that yields training
minibatches for the conditional diffusion model. Each sample consists of:

  - now_snap:    [1, H, W]           Target wavefield snapshot u^{n+1}
  - cond_prev:   [snap_steps+1, H, W] Conditioning input: the snap_steps most
                                       recent wavefield snapshots u^{n-4:n}
                                       stacked with the velocity model v
  - now_it_norm: scalar (float32)     Normalized wavefield time-step index n,
                                       used for the sinusoidal embedding in the
                                       U-Net (Section V)
  - now_it:      scalar (int64)       Integer snapshot index, used for looking up
                                       the causal weight ω(n) in the
                                       CausalWeightManager (Section III.3)
  - cond:        dict                 Additional model kwargs (empty by default;
                                       could contain class labels if class_cond=True)

Training data organization (Section III.I):
  - Training velocity models are drawn from an open-source dataset of 256×256
    velocity patches (BP1994, BP2004, Hess, Otway, Sigsbee, SEG/EAGE, Overthrust,
    SEAM Arid), resized to 128×128.
  - For each velocity model, 5 shot gathers are simulated with random surface
    source positions using a 10th-order staggered-grid FD solver.
  - Wavefield snapshots are saved every 0.01 s (= 10 × Δt_FD), yielding 101
    snapshots per simulation (0 to 1 s).
  - Each HDF5 shard file stores one shot gather: 'snaps' (101, 128, 128),
    'vp' (128, 128), and 'loc_x' (scalar source x-position).

At each training iteration, a random snapshot index n is sampled uniformly
from the valid range, and the corresponding (u^{n+1}, u^{n-4:n}, v) training
pair is assembled. Missing history frames (when n < snap_steps) are zero-padded,
reflecting the physical fact that the wavefield is identically zero prior to
source excitation (Section III.1).
"""

import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch
import os
import h5py
import threading
from collections import OrderedDict


def load_data(
    *, data_dir, snap_steps, batch_size, device, class_cond=False, deterministic=False
):
    """
    Create an infinite generator that yields training minibatches.

    Each iteration yields a tuple of five elements:
      (now_snap, cond_prev, now_it_norm, now_it_idx, cond_dict)

    corresponding to:
      - now_snap:    [B, 1, H, W]             Target snapshot u^{n+1}
      - cond_prev:   [B, snap_steps+1, H, W]  Conditioning: u^{n-4:n} ∥ v
      - now_it_norm: [B]                       Normalized time-step index n
                                                (for network embedding)
      - now_it_idx:  [B]                       Integer index n (for causal
                                                weight lookup in Eq. 11)
      - cond_dict:   dict                      Extra model kwargs (class labels, etc.)

    The generator loops over the dataset indefinitely, reshuffling after each
    epoch, so the training loop can call next(data) without worrying about
    dataset exhaustion.

    Args:
        data_dir:      Root directory containing HDF5 shard files.
        snap_steps:    Number of preceding wavefield snapshots used as
                       conditioning history (e.g., 5 for u^{n-4:n}).
        batch_size:    Number of samples per minibatch.
        device:        Compute device (used for pin_memory optimization).
        class_cond:    If True, include class labels in the output dict.
        deterministic: If True, disable shuffling for reproducible ordering.

    Yields:
        Tuples of (now_snap, cond_prev, now_it_norm, now_it_idx, cond_dict).
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = ShardedHDF5Dataset(
        data_dir, snap_steps,
        class_cond=class_cond,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=8,              # Parallel workers for data loading
        prefetch_factor=4,          # Each worker prefetches 4 batches
        persistent_workers=True,    # Reuse workers across epochs (avoid respawn overhead)
        worker_init_fn=worker_init_fn,  # Initialize per-worker HDF5 file handle cache
        pin_memory=True,            # Pin memory for faster CPU→GPU transfers
        drop_last=True,             # Drop incomplete final batch for consistent batch size
    )

    # Infinite loop: yield all batches from the DataLoader, then restart.
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively list all HDF5 (.h5) files under the given directory.

    Each HDF5 file corresponds to one shot gather simulation and contains:
      - 'snaps': [nt, ngrid, ngrid] Wavefield snapshots at Δt = 0.01 s intervals
      - 'vp':    [ngrid, ngrid]     Velocity model
      - 'loc_x': scalar             Source x-position on the surface

    Args:
        data_dir: Root directory to search.

    Returns:
        Sorted list of absolute paths to all .h5 files found.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["h5"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def normalizer_vel(x, dmin=1400, dmax=5000):
    """
    Normalize a velocity model to the range [-1, 1].

    The velocity values are linearly mapped from [dmin, dmax] (m/s) to [-1, 1],
    which provides a consistent scale for the neural network input. The default
    range [1400, 5000] m/s covers the typical velocity range encountered in the
    training dataset (Section III.I).

    Args:
        x:    Velocity model array (values in m/s).
        dmin: Minimum velocity for normalization (default: 1400 m/s).
        dmax: Maximum velocity for normalization (default: 5000 m/s).

    Returns:
        Normalized velocity model with values in [-1, 1].
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def denormalizer_vel(x, dmin=1400, dmax=5000):
    """
    Inverse of normalizer_vel: map from [-1, 1] back to [dmin, dmax] m/s.

    Args:
        x:    Normalized velocity model (values in [-1, 1]).
        dmin: Minimum velocity (default: 1400 m/s).
        dmax: Maximum velocity (default: 5000 m/s).

    Returns:
        Denormalized velocity model in m/s.
    """
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


# ========================================================================
# Per-worker LRU file handle cache for HDF5 files
# ========================================================================
# Each DataLoader worker maintains its own LRU cache of open HDF5 file
# handles. This avoids the overhead of repeatedly opening and closing files
# (which involves filesystem metadata lookups and internal HDF5 bookkeeping)
# while keeping the total number of open file descriptors bounded.
#
# With num_workers=8 and _MAX_OPEN_FILES=64, the maximum total open file
# descriptors across all workers is 8 × 64 = 512, which is well within
# typical OS limits.
# ========================================================================

_worker_file_cache = OrderedDict()  # LRU cache: path → h5py.File handle
_MAX_OPEN_FILES    = 64             # Max open files per worker


def _get_h5(path):
    """
    Get an open HDF5 file handle, using the per-worker LRU cache.

    If the file is already cached, it is moved to the end of the OrderedDict
    (most recently used). If the cache is full, the least recently used file
    is closed and evicted. If the file is not cached, it is opened in SWMR
    (Single Writer Multiple Reader) mode for safe concurrent access.

    Args:
        path: Absolute path to the HDF5 file.

    Returns:
        An open h5py.File handle, or None if the file could not be opened.
    """
    global _worker_file_cache

    # Cache hit: move to end (most recently used) and return.
    if path in _worker_file_cache:
        _worker_file_cache.move_to_end(path)
        return _worker_file_cache[path]

    # Cache full: evict the least recently used file (front of OrderedDict).
    if len(_worker_file_cache) >= _MAX_OPEN_FILES:
        oldest_path, oldest_f = _worker_file_cache.popitem(last=False)
        try:
            oldest_f.close()
        except Exception:
            pass

    # Open the new file in SWMR mode for safe multi-worker read access.
    try:
        f = h5py.File(path, 'r', swmr=True)
        _worker_file_cache[path] = f
        return f
    except Exception as e:
        print(f"[WARN] Failed to open {path}: {e}")
        return None


def worker_init_fn(worker_id):
    """
    DataLoader worker initialization function.

    Called once when each worker process starts. Initializes a fresh
    (empty) LRU file handle cache for the worker and seeds the numpy
    random number generator from the PyTorch seed for reproducibility.

    Args:
        worker_id: Integer ID of the worker (0 to num_workers-1).
    """
    global _worker_file_cache
    _worker_file_cache = OrderedDict()
    np.random.seed(torch.initial_seed() % 2**32)


class ShardedHDF5Dataset(Dataset):
    """
    PyTorch Dataset that loads training samples from sharded HDF5 files.

    Each HDF5 file stores one complete shot-gather simulation:
      - 'snaps': [nt, ngrid, ngrid]  Wavefield snapshots at Δt = 0.01 s intervals
                                      (101 snapshots for 0–1 s, Section III.I)
      - 'vp':    [ngrid, ngrid]      Velocity model
      - 'loc_x': scalar              Source x-position on the surface

    At each __getitem__ call, a random wavefield time-step index n is sampled
    uniformly from the valid range [it_gap, nt-1]. The method then assembles:

      - now_snap:   u^{n+1} = snaps[n]         — the target snapshot
      - prev_snaps: u^{n-4:n} = snaps[n-1], snaps[n-2], ..., snaps[n-snap_steps]
                    — the conditioning history (zero-padded for early snapshots
                    where fewer than snap_steps preceding frames exist, per
                    Section III.1)
      - vp:         velocity model (normalized to [-1, 1])

    The conditioning input cond_prev is formed by concatenating prev_snaps and
    vp along the channel dimension, yielding a (snap_steps + 1)-channel tensor
    that is passed to the U-Net's convolutional stem (Section V).

    Args:
        hdf5_dir:    Root directory containing HDF5 shard files.
        snap_steps:  Number of preceding snapshots in the conditioning history
                     (e.g., 5 for a 5-frame history u^{n-4:n}).
        nt:          Total number of wavefield snapshots per simulation
                     (default: 101, corresponding to 0–1 s at Δt = 0.01 s).
        it_gap:      Temporal stride between consecutive conditioning frames
                     (default: 1, meaning every saved snapshot is used).
        ngrid:       Spatial grid size (default: 128, matching the 128×128
                     training patches of Section III.I).
        class_cond:  If True, include class labels (not used for GenWP).
    """

    def __init__(self, hdf5_dir, snap_steps,
                 nt=101, it_gap=1, ngrid=128, class_cond=False):
        super().__init__()

        self.snap_steps = snap_steps    # Number of conditioning history frames
        self.nt         = nt            # Total snapshots per simulation (101)
        self.it_gap     = it_gap        # Temporal stride between frames
        self.ngrid      = ngrid         # Spatial grid size (128×128)

        # Recursively scan the directory for all HDF5 shard files.
        print(f"Listing H5 files from: {hdf5_dir}")
        self.index = _list_image_files_recursively(hdf5_dir)
        print(f"  Total H5 files  : {len(self.index):,}")
        print(f"  nt range        : [{it_gap}, {nt - it_gap}] (randomly sampled)")
        print(f"  Total samples   : {len(self.index):,}")

    def __len__(self):
        """Return the number of HDF5 shard files (= number of shot gathers)."""
        return len(self.index)

    def __getitem__(self, idx):
        """
        Load a single training sample from the idx-th HDF5 shard file.

        A random wavefield time-step index n is sampled uniformly from
        [it_gap, nt-1]. The method assembles the training tuple:

          (now_snap, cond_prev, now_it_norm, now_it, cond_dict)

        corresponding to:
          - now_snap:    [1, ngrid, ngrid]             u^{n+1} (target snapshot)
          - cond_prev:   [snap_steps+1, ngrid, ngrid]  u^{n-4:n} ∥ v (conditioning)
          - now_it_norm: float32                        n × 0.001 (normalized index
                                                        for network embedding)
          - now_it:      int64                          Integer n (for causal weight
                                                        lookup, Eq. 11)
          - cond_dict:   dict                           Empty dict (or class labels)

        The conditioning history is assembled by reading snapshots at indices
        [n-1, n-2, ..., n-snap_steps] (with zero-padding for indices < 0).
        All required snapshots are batch-read from the HDF5 file in a single
        I/O operation for efficiency.

        Args:
            idx: Integer index into the shard file list.

        Returns:
            Tuple of (now_snap, cond_prev, now_it_norm, now_it, cond_dict).
        """
        path   = self.index[idx]

        # Sample a random wavefield time-step index n from [it_gap, nt-1].
        # This random sampling, combined with the per-shard random access,
        # ensures that all (velocity model, source position, snapshot index)
        # combinations are visited across training iterations.
        now_it = random.randint(self.it_gap, self.nt - 1)

        # ---- Pre-allocate output arrays (zero-initialized) ----
        # Zero initialization handles the case where preceding frames are
        # unavailable (n < snap_steps), reflecting the physical fact that
        # the wavefield is identically zero prior to source excitation
        # (Section III.1).
        now_snap   = np.zeros((1,               self.ngrid, self.ngrid), dtype=np.float32)
        prev_snaps = np.zeros((self.snap_steps, self.ngrid, self.ngrid), dtype=np.float32)
        vp         = np.zeros((1,               self.ngrid, self.ngrid), dtype=np.float32)
        loc_x      = np.float32(0.0)

        # ---- Read data from the HDF5 shard file ----
        h5f = _get_h5(path)
        if h5f is not None:
            try:
                # Determine which preceding snapshot indices are valid (≥ 0).
                prev_its = [now_it - (i + 1)*self.it_gap
                            for i in range(self.snap_steps)
                            if now_it - (i + 1)*self.it_gap >= 0]

                # Collect all required time-step indices (target + history)
                # into a sorted list for a single batch HDF5 read, which is
                # much faster than reading individual time steps.
                all_its = sorted(set(prev_its + [now_it]))
                batch   = h5f['snaps'][all_its]     # Single I/O: [len(all_its), ngrid, ngrid]
                it2pos  = {t: pos for pos, t in enumerate(all_its)}

                # Extract the target snapshot u^{n+1} = snaps[now_it].
                now_snap[0] = batch[it2pos[now_it]]

                # Extract the conditioning history u^{n-4:n}.
                # Channel ordering: prev_snaps[0] = u^{n-1} (most recent),
                # prev_snaps[1] = u^{n-2}, ..., prev_snaps[snap_steps-1] = u^{n-snap_steps}.
                # Frames with index < 0 remain zero (pre-allocated).
                for i in range(self.snap_steps):
                    t = now_it - i - self.it_gap
                    if t >= 0:
                        prev_snaps[i] = batch[it2pos[t]]

                # Read the velocity model and normalize to [-1, 1].
                if 'vp' in h5f:
                    vp[0] = normalizer_vel(h5f['vp'][:].astype(np.float32))

                # Read the source x-position (used for conditioning or logging).
                if 'loc_x' in h5f:
                    loc_x = np.float32(h5f['loc_x'][()])

            except Exception as e:
                print(f"[WARN] {path} t={now_it}: {e}")
                # Evict the broken file handle from the cache.
                _worker_file_cache.pop(path, None)

        # ---- Normalize the wavefield time-step index ----
        # Multiplied by 0.001 to bring the index into a range suitable for
        # the sinusoidal positional embedding (TimeEmbedding in unet.py).
        now_it_norm = np.float32(now_it * 0.001)

        # ---- Assemble the spatial conditioning tensor ----
        # Concatenate the snap_steps history frames and the velocity model
        # along the channel dimension: [snap_steps + 1, ngrid, ngrid].
        # This forms the input to the U-Net's convolutional stem (Section V).
        cond_prev   = np.concatenate((prev_snaps, vp), axis=0)

        return now_snap, cond_prev, now_it_norm, np.int64(now_it), {}