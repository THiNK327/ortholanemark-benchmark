"""Shared training helpers for the faithful learned baselines.

Provides:
  - set_global_seed(seed): seeds torch/np/random + cudnn deterministic.
  - make_loader_generator(seed): a seeded torch.Generator for DataLoader
    shuffle, so batch order is deterministic and resume-safe.
  - atomic_save(obj, path): tmp-file + rename, won't corrupt on Ctrl-C.
  - atomic_json(obj, path): same for JSON.
  - rng_snapshot() / rng_restore(snap): full RNG state capture/restore
    so a `--resume` continues exactly where it left off.
  - load_checkpoint_for_resume(path): convenience loader returning the
    full state dict.

Used by the faithful baseline train.py entry points.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed all RNGs and request deterministic CUDA kernels where possible.
    Idempotent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader_generator(seed: int) -> torch.Generator:
    """Generator passed to DataLoader for reproducible shuffle order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def rng_snapshot() -> dict:
    """Return a dict capturing the full RNG state of torch/cuda/np/python.
    Re-applying via `rng_restore` reproduces subsequent random draws."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }


def rng_restore(snap: dict) -> None:
    random.setstate(snap['python'])
    np.random.set_state(snap['numpy'])
    torch.set_rng_state(snap['torch'])
    if snap.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap['cuda'])


def atomic_save(obj, path) -> None:
    """torch.save to a sibling tmpfile then os.replace → atomic on POSIX
    and Win32 (replace is atomic on Win since Vista if the destination is
    on the same filesystem). Prevents a Ctrl-C from leaving a half-
    written `last.pth`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # tempfile on same dir → guaranteed same filesystem → atomic replace.
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                     dir=str(path.parent))
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                                     dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def env_metadata() -> dict:
    """Capture environment info to log into ckpt for reproducibility."""
    return {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        'gpu_name': (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
    }


def should_skip_training(save_dir, target_epochs: int) -> bool:
    """Orchestrator-level: returns True if a complete run already exists.
    A run is complete iff `last.pth` exists AND its `epoch` >= target."""
    last = Path(save_dir) / 'last.pth'
    if not last.exists():
        return False
    try:
        ckpt = torch.load(str(last), map_location='cpu', weights_only=False)
    except Exception:
        return False
    return int(ckpt.get('epoch', 0)) >= int(target_epochs)


def load_resume_checkpoint(save_dir):
    """Return the `last.pth` dict, or None if missing."""
    last = Path(save_dir) / 'last.pth'
    if not last.exists():
        return None
    return torch.load(str(last), map_location='cpu', weights_only=False)
