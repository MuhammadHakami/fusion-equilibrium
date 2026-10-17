"""Per-process GPU memory cap from the environment (Sep 23, after two OOM reboots on the GB10).

GB10 memory is unified: a CUDA allocation takes system RAM that no cgroup cap sees, so a runaway batch can starve the
whole machine. `QW_GPU_GB=<GiB>` makes PyTorch's caching allocator raise OutOfMemoryError past that budget instead.
"""
from __future__ import annotations

import os


def cap_from_env() -> float:
    gb = float(os.environ.get("QW_GPU_GB", "0") or 0)
    if gb <= 0:
        return 0.0
    import torch
    if not torch.cuda.is_available():
        return 0.0
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(1.0, gb * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    return frac
