"""Unit + (gated) on-device test for `WanRMSNormNKI`.

Compares `WanRMSNormNKI` to the original `WanRMSNorm` for the realistic
shapes used in TI2V-5B training:

    single device:  [1, 4400, 3072] bf16
    SP=8 per-rank:  [1,  550, 3072] bf16

Both shapes go through `_compare_one`, which:
  1. Runs both modules on the same input.
  2. Compares numerics with bf16-appropriate tolerances.
  3. (If RUN_KERNEL_BENCH=1) calls `run_compare` from `_bench.py` to time
     both implementations on device.

By default the test runs on CPU and exercises the PyTorch fallback path
(so the file imports on any host). Set `NEURON_KERNEL_TEST=1` to run on
the Neuron / XLA device — that path actually invokes the NKI kernel.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Allow running directly via `python tests/kernels/test_rmsnorm_nki.py`
import sys
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wan.modules.model import WanRMSNorm  # noqa: E402
from wan.kernels.wan_rms_norm_nki import WanRMSNormNKI  # noqa: E402

from tests.kernels._bench import (  # noqa: E402
    numerics,
    run_compare,
    should_run_bench,
)


# bf16 RMSNorm with fp32 internals: 5e-2 abs / 5e-2 rel matches the
# tolerance other Wan kernel tests (e.g. test_sdpa_ref) use for bf16.
_ABS_TOL = 5e-2
_REL_TOL = 5e-2


def _device():
    """Pick neuron device only when explicitly enabled."""
    if os.environ.get("NEURON_KERNEL_TEST") == "1":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            return torch.device("xla")
        except Exception:
            # Fall back to neuron device string if torch_xla is not present.
            return torch.device("neuron")
    return torch.device("cpu")


def _make_modules(dim: int, eps: float, device: torch.device):
    """Return a (reference, candidate) pair sharing weights."""
    torch.manual_seed(0)
    ref = WanRMSNorm(dim=dim, eps=eps)
    cand = WanRMSNormNKI(dim=dim, eps=eps)
    # Match weights so the comparison is meaningful (both classes default
    # to ones, but matching by copy_ guarantees parity if defaults change).
    with torch.no_grad():
        cand.weight.copy_(ref.weight)
    return ref.to(device), cand.to(device)


def _compare_one(label: str, B: int, L: int, H: int, eps: float):
    device = _device()
    if device.type == "cpu":
        # CPU path is the PyTorch fallback; running it still validates
        # the WanRMSNormNKI module's API + fallback math.
        pass

    ref, cand = _make_modules(H, eps, device)
    torch.manual_seed(123)
    x = torch.randn(B, L, H, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        out_ref = ref(x)
        out_cand = cand(x)

    # Move to CPU for the comparison so we don't generate stray XLA graphs.
    a = out_ref.detach().to("cpu")
    b = out_cand.detach().to("cpu")

    max_abs, rel_inf = numerics(a, b)
    print(
        f"[{label}] shape=({B},{L},{H}) eps={eps} "
        f"maxabs={max_abs:.3e} relinf={rel_inf:.3e}"
    )
    assert max_abs <= _ABS_TOL, (
        f"{label}: |ref - nki|_inf = {max_abs:.3e} exceeds {_ABS_TOL:.1e}"
    )
    assert rel_inf <= _REL_TOL, (
        f"{label}: rel_inf = {rel_inf:.3e} exceeds {_REL_TOL:.1e}"
    )

    if should_run_bench() and device.type != "cpu":
        result = run_compare(
            label=label,
            ref_fn=lambda t: ref(t),
            kernel_fn=lambda t: cand(t),
            args=(x,),
            shape=(B, L, H),
            dtype="bf16",
        )
        print(result.line())


def test_single_device_shape():
    """TI2V-5B single-device self-attn norm shape: [1, 4400, 3072]."""
    if _device().type == "cpu" and os.environ.get("NEURON_KERNEL_TEST") != "1":
        # CPU fallback path: still exercise to keep the file regression-safe.
        _compare_one("single", B=1, L=4400, H=3072, eps=1e-6)
        return
    _compare_one("single", B=1, L=4400, H=3072, eps=1e-6)


def test_sp_per_rank_shape():
    """SP=8 per-rank self-attn norm shape: [1, 550, 3072]."""
    if _device().type == "cpu" and os.environ.get("NEURON_KERNEL_TEST") != "1":
        _compare_one("sp8_rank", B=1, L=550, H=3072, eps=1e-6)
        return
    _compare_one("sp8_rank", B=1, L=550, H=3072, eps=1e-6)


def test_default_eps_matches_pytorch_class_default():
    """Sanity check at the class default eps=1e-5."""
    _compare_one("default_eps", B=1, L=550, H=3072, eps=1e-5)


def test_module_on_cpu_is_safe_to_import():
    """The NKI module must be importable + runnable on CPU.

    On CPU we hit the PyTorch fallback path. Verifies the file works in
    test envs without the Neuron toolchain.
    """
    if _device().type != "cpu":
        pytest.skip("Only meaningful when NEURON_KERNEL_TEST is unset.")
    ref = WanRMSNorm(dim=64, eps=1e-6)
    cand = WanRMSNormNKI(dim=64, eps=1e-6)
    with torch.no_grad():
        cand.weight.copy_(ref.weight)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        a = ref(x)
        b = cand(x)
    max_abs, _ = numerics(a, b)
    # CPU fallback should be effectively bit-identical (same fp32 path).
    assert max_abs <= 1e-5, f"CPU fallback diverged from WanRMSNorm: {max_abs:.3e}"


if __name__ == "__main__":
    test_module_on_cpu_is_safe_to_import()
    test_single_device_shape()
    test_sp_per_rank_shape()
    test_default_eps_matches_pytorch_class_default()
    print("All RMSNorm NKI tests passed.")
