"""
Shared benchmarking helpers for NKI-vs-PyTorch microbenchmarks on Neuron.

Conventions:
- Compare a `nki_fn` to a `ref_fn` (both PyTorch nn.functional or callable wrappers).
- Run a warmup loop (kernel selection / cache prime) then time over `iters`.
- Bracket each timed iteration with `torch.neuron.synchronize()` to avoid
  measuring async dispatch overhead instead of compute.
- Compare numerics with absolute and relative tolerances appropriate for bf16.
- All benchmarks are off by default — they need `RUN_KERNEL_BENCH=1` to fire,
  so they don't run during regression CI or during regular pytest.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import torch


def _sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def should_run_bench() -> bool:
    return os.environ.get("RUN_KERNEL_BENCH") == "1"


@contextmanager
def timed():
    _sync()
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0
    _sync()


@dataclass
class BenchResult:
    label: str
    shape: tuple
    dtype: str
    ref_us: float
    kernel_us: float
    speedup: float
    max_abs_err: float
    rel_err_inf: float
    extras: dict = field(default_factory=dict)

    def line(self) -> str:
        return (
            f"[{self.label}] shape={self.shape} dtype={self.dtype} "
            f"ref={self.ref_us:.1f}us kernel={self.kernel_us:.1f}us "
            f"speedup={self.speedup:.2f}x maxabs={self.max_abs_err:.2e} "
            f"relinf={self.rel_err_inf:.2e}"
        )


def time_fn(fn: Callable, args: Iterable, iters: int = 20) -> float:
    """Mean per-call wall-clock in microseconds, with a 3-iter warmup."""
    args = tuple(args)
    for _ in range(3):
        fn(*args)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    _sync()
    elapsed = time.perf_counter() - t0
    return (elapsed / iters) * 1e6


def numerics(out_a: torch.Tensor, out_b: torch.Tensor) -> tuple[float, float]:
    a = out_a.detach().to(torch.float32)
    b = out_b.detach().to(torch.float32)
    abs_err = (a - b).abs()
    denom = b.abs().clamp(min=1e-6)
    rel_err = abs_err / denom
    return float(abs_err.max().item()), float(rel_err.max().item())


def run_compare(
    label: str,
    ref_fn: Callable,
    kernel_fn: Callable,
    args: Iterable,
    *,
    shape,
    dtype,
    iters: int = 20,
) -> BenchResult:
    """Sanity-check numerics, then time both implementations."""
    args = tuple(args)
    out_ref = ref_fn(*args)
    out_kernel = kernel_fn(*args)
    max_abs, rel_inf = numerics(out_ref, out_kernel)
    ref_us = time_fn(ref_fn, args, iters=iters)
    kernel_us = time_fn(kernel_fn, args, iters=iters)
    return BenchResult(
        label=label,
        shape=tuple(shape),
        dtype=str(dtype),
        ref_us=ref_us,
        kernel_us=kernel_us,
        speedup=ref_us / kernel_us if kernel_us else float("inf"),
        max_abs_err=max_abs,
        rel_err_inf=rel_inf,
    )
