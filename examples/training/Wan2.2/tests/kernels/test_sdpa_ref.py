"""
Reference numerics test for `torch.nn.functional.scaled_dot_product_attention`
on the Native Neuron device, against a from-scratch PyTorch reference.

Used as the correctness baseline before swapping in any NKI kernel: if a future
kernel's output deviates from `_sdpa_reference` beyond the bf16 tolerances
asserted here, we know the kernel itself is wrong (not the framework).

Runs on neuron device only when NEURON_KERNEL_TEST=1; otherwise CPU.
"""
from __future__ import annotations

import os

import torch


def _sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain softmax(QK^T/sqrt(d))*V, fp32 accumulation."""
    d = q.size(-1)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    attn = qf @ kf.transpose(-2, -1) / (d ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    out = attn @ vf
    return out.to(q.dtype)


def _sdpa_native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)


def _device():
    if os.environ.get("NEURON_KERNEL_TEST") == "1":
        return torch.device("neuron")
    return torch.device("cpu")


def test_sdpa_self_attention_shapes():
    """Shapes mirroring Wan TI2V-5B self-attn at 17-frame chunk: [1,24,4400,128]."""
    device = _device()
    torch.manual_seed(0)
    B, H, S, D = 1, 24, 4400, 128
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16, device=device)
    ref = _sdpa_reference(q, k, v)
    native = _sdpa_native(q, k, v)
    abs_err = (ref.float() - native.float()).abs().max().item()
    print(f"self_attn 17f abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"native SDPA diverged: {abs_err}"


def test_sdpa_cross_attention_shapes():
    """Cross-attn: Q [1,24,4400,128], K/V [1,24,512,128]."""
    device = _device()
    torch.manual_seed(0)
    B, H, Sq, Sk, D = 1, 24, 4400, 512, 128
    q = torch.randn(B, H, Sq, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, Sk, D, dtype=torch.bfloat16, device=device)
    ref = _sdpa_reference(q, k, v)
    native = _sdpa_native(q, k, v)
    abs_err = (ref.float() - native.float()).abs().max().item()
    print(f"cross_attn abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"native SDPA cross diverged: {abs_err}"


if __name__ == "__main__":
    test_sdpa_self_attention_shapes()
    test_sdpa_cross_attention_shapes()
    print("Reference SDPA tests pass.")
