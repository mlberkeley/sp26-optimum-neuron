"""CPU + (gated) on-device tests for `rope_apply_nki`.

Compares `wan.kernels.wan_rope_nki.rope_apply_nki` against the
reference real-valued masked-write `rope_apply` from
`wan.modules.model` (single-device path) and
`wan.distributed.sequence_parallel` (SP path, sp_size>1).

By default the tests run on CPU and exercise the PyTorch fallback (so
the file imports on any host). Set `NEURON_KERNEL_TEST=1` to dispatch
to the Neuron / XLA device — that path actually invokes the NKI
kernel.

Tolerances:
    bf16 path:  abs <= 5e-3
    fp32 path:  abs <= 5e-5

The fp32 path is the strict numerical oracle (mirroring tests/test_sp_rope.py
which sees max_abs_err ~5e-7 in float32). bf16 tolerance follows
test_rmsnorm_nki / test_sdpa_ref conventions in this repo.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

# Allow running directly via `python tests/kernels/test_rope_nki.py`.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from wan.kernels.rope_nki import (  # noqa: E402
    _rope_apply_torch_ref,
    build_angles,
    _build_cos_sin_il,
)
from wan.kernels.wan_rope_nki import rope_apply_nki, _rope_apply_ref  # noqa: E402


# -- helpers -----------------------------------------------------------
def _make_freqs(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Mirrors `wan.modules.model.rope_params`."""
    assert dim % 2 == 0
    idx = torch.arange(0, dim, 2)  # [dim/2]
    inv_freq = 1.0 / torch.pow(torch.tensor(theta), idx / dim)
    positions = torch.arange(max_seq_len)
    return torch.outer(positions.float(), inv_freq.float())  # [max_seq_len, dim/2]


def _make_masks(D: int) -> tuple[torch.Tensor, torch.Tensor]:
    even_mask = torch.zeros(1, 1, D, dtype=torch.float32)
    even_mask[..., 0::2] = 1.0
    odd_mask = 1.0 - even_mask
    return even_mask, odd_mask


def _device():
    if os.environ.get("NEURON_KERNEL_TEST") == "1":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            return torch.device("xla")
        except Exception:
            return torch.device("neuron")
    return torch.device("cpu")


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to(torch.float32).to("cpu")
    b = b.detach().to(torch.float32).to("cpu")
    return float((a - b).abs().max().item())


# -- core comparison driver --------------------------------------------
def _compare_one(
    label: str,
    B: int,
    F: int,
    H: int,
    W: int,
    N: int,
    D: int,
    sp_rank: int,
    sp_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    token_len = F * H * W
    full_L = math.ceil(token_len / sp_size) * sp_size
    s = full_L // sp_size  # per-rank length

    device = _device()

    # x is laid out as the per-rank shard.
    x = torch.randn(B, s, N, D, dtype=dtype)
    grid_sizes = torch.tensor([[F, H, W]] * B)
    freqs = _make_freqs(max_seq_len=max(F, H, W) + 8, dim=D)
    even_mask, odd_mask = _make_masks(D)

    # CPU reference (always run on cpu — ground truth).
    ref = _rope_apply_ref(
        x.cpu(), grid_sizes, freqs, even_mask, odd_mask, sp_rank=sp_rank, sp_size=sp_size
    )

    # Candidate: NKI wrapper. On CPU, this hits the PyTorch fallback and
    # should match ref to within fp32 rounding. On Neuron, it dispatches
    # to the kernel.
    x_dev = x.to(device=device)
    freqs_dev = freqs.to(device=device)
    even_mask_dev = even_mask.to(device=device)
    odd_mask_dev = odd_mask.to(device=device)

    out = rope_apply_nki(
        x_dev,
        grid_sizes,
        freqs_dev,
        even_mask_dev,
        odd_mask_dev,
        sp_rank=sp_rank,
        sp_size=sp_size,
    )
    out_cpu = out.detach().to("cpu")

    abs_err = _max_abs(ref, out_cpu)
    if dtype == torch.bfloat16:
        atol = 5e-3
    else:
        atol = 5e-5

    print(
        f"[{label}] B={B} F={F} H={H} W={W} N={N} D={D} "
        f"sp_rank={sp_rank}/{sp_size} dtype={dtype} dev={device.type} "
        f"max_abs_err={abs_err:.3e} (tol={atol:.0e})"
    )
    assert abs_err <= atol, f"{label}: max_abs_err {abs_err:.3e} > {atol:.0e}"


# -- pytest cases ------------------------------------------------------
def test_kernel_math_contract_matches_masked_write():
    """Critical CPU-only test: validates the kernel's mathematical
    formulation `out = x*cos_il + swap(x)*sin_il` is bit-equivalent to
    the reference masked-write form. Catches sign errors / channel-
    interleaving mistakes without needing on-device execution.

    This is the strongest CPU-side check we can do because it runs the
    `_rope_apply_torch_ref` (which mirrors the kernel's math exactly,
    only diff is host vs device) against `_rope_apply_ref` (the
    reference masked-write form from model.py / sequence_parallel.py).
    Mismatch here would mean the kernel is buggy by construction.
    """
    torch.manual_seed(42)
    B, F, H, W, N, D = 1, 5, 22, 40, 24, 128
    s = F * H * W

    x = torch.randn(B, s, N, D, dtype=torch.float32)
    grid_sizes = torch.tensor([[F, H, W]] * B)
    freqs = _make_freqs(max_seq_len=max(F, H, W) + 8, dim=D)
    even_mask, odd_mask = _make_masks(D)

    # Reference (masked-write form, what the production code does today).
    ref = _rope_apply_ref(x, grid_sizes, freqs, even_mask, odd_mask)

    # Kernel math contract: build cos_il/sin_il host-side, then apply
    # via the same pair-swap formulation the kernel uses.
    angles = build_angles(F, H, W, freqs)
    cos_il, sin_il = _build_cos_sin_il(angles, target_len=s)
    out = _rope_apply_torch_ref(x[0], cos_il, sin_il).unsqueeze(0)

    err = _max_abs(ref, out)
    print(f"[contract_check] masked-write vs kernel-math: max_abs_err = {err:.3e}")
    # Bit-exact: same operations, same dtype, same order.
    assert err < 1e-5, (
        f"Kernel mathematical contract diverges from masked-write reference: "
        f"{err:.3e}. Check cos_il / sin_il sign and interleave."
    )


def test_kernel_math_contract_sp_slice():
    """Same as above but for an SP rank slice. Verifies the slicing in
    `_build_cos_sin_il` matches `_rope_apply_ref`'s sp_rank semantics."""
    torch.manual_seed(7)
    B, F, H, W, N, D = 1, 5, 22, 40, 24, 128
    sp_size = 8
    sp_rank = 5
    token_len = F * H * W
    full_L = math.ceil(token_len / sp_size) * sp_size
    s = full_L // sp_size

    x = torch.randn(B, s, N, D, dtype=torch.float32)
    grid_sizes = torch.tensor([[F, H, W]] * B)
    freqs = _make_freqs(max_seq_len=max(F, H, W) + 8, dim=D)
    even_mask, odd_mask = _make_masks(D)

    ref = _rope_apply_ref(
        x, grid_sizes, freqs, even_mask, odd_mask,
        sp_rank=sp_rank, sp_size=sp_size,
    )

    # Replicate wrapper's host-side angle construction for this rank.
    angles = build_angles(F, H, W, freqs)
    if full_L > token_len:
        pad = torch.zeros(full_L - token_len, D // 2, dtype=torch.float32)
        angles_full = torch.cat([angles, pad], dim=0)
    else:
        angles_full = angles[:full_L]
    angles_rank = angles_full[sp_rank * s:(sp_rank + 1) * s]
    cos_il, sin_il = _build_cos_sin_il(angles_rank, target_len=s)

    out = _rope_apply_torch_ref(x[0], cos_il, sin_il).unsqueeze(0)
    err = _max_abs(ref, out)
    print(f"[contract_check_sp] sp_rank={sp_rank}/{sp_size}: max_abs_err = {err:.3e}")
    assert err < 1e-5, f"SP slice contract diverges: {err:.3e}"


def test_small_synthetic_fp32():
    """Tiny shape, fp32, single-device. Exercises the kernel logic with
    minimal volume and a strict numeric tolerance."""
    _compare_one(
        "tiny_f32", B=1, F=2, H=3, W=4, N=2, D=16,
        sp_rank=0, sp_size=1, dtype=torch.float32,
    )


def test_small_synthetic_bf16():
    """Tiny shape, bf16. Exercises dtype promotion path."""
    _compare_one(
        "tiny_bf16", B=1, F=2, H=3, W=4, N=2, D=16,
        sp_rank=0, sp_size=1, dtype=torch.bfloat16,
    )


def test_single_device_ti2v_5b_fp32():
    """TI2V-5B 17-frame chunk, single-device. fp32 to keep tolerance tight.

    Shape: [1, 4400, 24, 128].
    """
    _compare_one(
        "single_5B", B=1, F=5, H=22, W=40, N=24, D=128,
        sp_rank=0, sp_size=1, dtype=torch.float32,
    )


def test_single_device_ti2v_5b_bf16():
    """TI2V-5B 17-frame chunk, single-device, production dtype."""
    _compare_one(
        "single_5B_bf16", B=1, F=5, H=22, W=40, N=24, D=128,
        sp_rank=0, sp_size=1, dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_sp8_per_rank_fp32(rank):
    """SP=8 per-rank, fp32. token_len=4400 / 8 = 550 per rank. Verifies
    each rank's slice is correct."""
    _compare_one(
        f"sp8_r{rank}", B=1, F=5, H=22, W=40, N=24, D=128,
        sp_rank=rank, sp_size=8, dtype=torch.float32,
    )


def test_sp8_per_rank_bf16():
    """SP=8 per-rank, bf16."""
    _compare_one(
        "sp8_r0_bf16", B=1, F=5, H=22, W=40, N=24, D=128,
        sp_rank=0, sp_size=8, dtype=torch.bfloat16,
    )


def test_sp_padding_case():
    """token_len not divisible by sp_size — SP=7 forces tail padding.

    With F=3, H=4, W=5 → token_len=60; sp=7 → s=ceil(60/7)=9, full_L=63,
    so 3 padding tokens go into the last rank's slice. The reference
    handles this via zero-angle padding (identity rotation).
    """
    _compare_one(
        "pad_sp7_r6", B=1, F=3, H=4, W=5, N=2, D=16,
        sp_rank=6, sp_size=7, dtype=torch.float32,
    )


def test_batch_multiple_items():
    """B=2 (cond+uncond batched). Same grid for both, but exercises the
    batch loop in the wrapper."""
    _compare_one(
        "B2", B=2, F=2, H=3, W=4, N=2, D=16,
        sp_rank=0, sp_size=1, dtype=torch.float32,
    )


def test_cpu_fallback_module_safe_to_import():
    """On CPU we must hit the PyTorch fallback path. Verifies the file
    works in test envs without the Neuron toolchain. Strict numerical
    equivalence to the reference."""
    if _device().type != "cpu":
        pytest.skip("Only meaningful when NEURON_KERNEL_TEST is unset.")

    torch.manual_seed(0)
    B, F, H, W, N, D = 1, 2, 3, 4, 2, 16
    s = F * H * W
    x = torch.randn(B, s, N, D, dtype=torch.float32)
    grid_sizes = torch.tensor([[F, H, W]] * B)
    freqs = _make_freqs(max_seq_len=max(F, H, W) + 8, dim=D)
    even_mask, odd_mask = _make_masks(D)

    ref = _rope_apply_ref(x, grid_sizes, freqs, even_mask, odd_mask)
    out = rope_apply_nki(x, grid_sizes, freqs, even_mask, odd_mask)

    abs_err = _max_abs(ref, out)
    # Same fp32 path -> bit-equivalent.
    assert abs_err <= 1e-6, f"CPU fallback diverged: {abs_err:.3e}"


if __name__ == "__main__":
    test_kernel_math_contract_matches_masked_write()
    test_kernel_math_contract_sp_slice()
    test_cpu_fallback_module_safe_to_import()
    test_small_synthetic_fp32()
    test_small_synthetic_bf16()
    test_single_device_ti2v_5b_fp32()
    test_single_device_ti2v_5b_bf16()
    for r in (0, 3, 7):
        test_sp8_per_rank_fp32(r)
    test_sp8_per_rank_bf16()
    test_sp_padding_case()
    test_batch_multiple_items()
    print("All rope NKI tests passed.")
