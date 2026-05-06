"""
Numerics tests for ``wan.kernels.attention_nki`` and ``wan.kernels.wan_attention_nki``.

We compare the NKI fused-attention path against the same SDPA reference used
by ``test_sdpa_ref.py`` for both Wan2.2 self-attn and cross-attn shapes at the
TI2V-5B 17-frame configuration. CPU runs ``pytest.skip`` because the kernel
needs the Neuron toolchain; gate device runs behind ``NEURON_KERNEL_TEST=1``.

Tolerances:
    abs <= 5e-2 in bf16 — attention has accumulated softmax error so we use a
    looser bar than RMSNorm-style ports.
"""
from __future__ import annotations

import os

import pytest
import torch


# ---------- helpers ----------


def _sdpa_reference(q_blhd: torch.Tensor, k_blhd: torch.Tensor, v_blhd: torch.Tensor) -> torch.Tensor:
    """fp32 softmax(QK^T/sqrt(d)) V on inputs in [B,L,H,D] layout."""
    q = q_blhd.transpose(1, 2).float()  # [B,H,Lq,D]
    k = k_blhd.transpose(1, 2).float()
    v = v_blhd.transpose(1, 2).float()
    d = q.shape[-1]
    attn = q @ k.transpose(-2, -1) / (d ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    out = attn @ v                       # [B,H,Lq,D]
    return out.transpose(1, 2).contiguous().to(q_blhd.dtype)


def _device():
    if os.environ.get("NEURON_KERNEL_TEST") == "1":
        try:
            import torch_xla.core.xla_model as xm  # noqa: F401
            return torch.device("xla")
        except Exception:
            # PyTorch Native Neuron path (no torch_xla in this venv).
            return torch.device("neuron")
    return None  # signal to skip


def _maybe_skip(device):
    if device is None:
        pytest.skip("NEURON_KERNEL_TEST!=1 — NKI attention requires Neuron device")


# ---------- import sanity (CPU-safe) ----------


def test_import_clean():
    """`wan.kernels` and the attention modules must import on any host."""
    import importlib

    pkg = importlib.import_module("wan.kernels")
    # submodules must be reachable as `wan.kernels.X`
    assert hasattr(pkg, "attention_nki")  # submodule
    assert hasattr(pkg, "wan_attention_nki")  # submodule

    mod = importlib.import_module("wan.kernels.attention_nki")
    assert callable(mod.nki_attention)
    assert callable(mod.is_available)


def test_cpu_fallback_matches_sdpa():
    """`attention_nki` on CPU should fall through to plain SDPA and match it."""
    from wan.kernels.wan_attention_nki import attention_nki

    torch.manual_seed(0)
    B, L, H, D = 1, 64, 4, 16
    q = torch.randn(B, L, H, D, dtype=torch.float32)
    k = torch.randn(B, L, H, D, dtype=torch.float32)
    v = torch.randn(B, L, H, D, dtype=torch.float32)

    out = attention_nki(q.clone(), k.clone(), v.clone(), dtype=torch.float32)
    ref = _sdpa_reference(q.float(), k.float(), v.float())

    abs_err = (out.float() - ref.float()).abs().max().item()
    assert abs_err < 1e-4, f"CPU fallback diverged from SDPA: {abs_err}"


# ---------- on-device numerics ----------


def test_self_attention_self_17frame():
    """Self-attention shapes: q=k=v=[1,4400,24,128] bf16, non-causal."""
    device = _device()
    _maybe_skip(device)

    from wan.kernels.attention_nki import nki_attention

    torch.manual_seed(0)
    B, L, H, D = 1, 4400, 24, 128
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16)

    ref = _sdpa_reference(q, k, v)  # CPU

    out = nki_attention(q.to(device), k.to(device), v.to(device)).cpu()

    abs_err = (out.float() - ref.float()).abs().max().item()
    print(f"self_attn 17f abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"NKI self-attn diverged: {abs_err}"


def test_cross_attention_text():
    """Cross-attention shapes: q=[1,4400,24,128], k/v=[1,512,24,128] bf16."""
    device = _device()
    _maybe_skip(device)

    from wan.kernels.attention_nki import nki_attention

    torch.manual_seed(0)
    B, Lq, Lk, H, D = 1, 4400, 512, 24, 128
    q = torch.randn(B, Lq, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, Lk, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, Lk, H, D, dtype=torch.bfloat16)

    ref = _sdpa_reference(q, k, v)

    out = nki_attention(q.to(device), k.to(device), v.to(device)).cpu()

    abs_err = (out.float() - ref.float()).abs().max().item()
    print(f"cross_attn abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"NKI cross-attn diverged: {abs_err}"


def test_self_attention_sp_post_a2a():
    """Post-all-to-all SP shape: q=k=v=[1,4400,3,128] (24/8=3 heads/rank)."""
    device = _device()
    _maybe_skip(device)

    from wan.kernels.attention_nki import nki_attention

    torch.manual_seed(0)
    B, L, H, D = 1, 4400, 3, 128
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16)

    ref = _sdpa_reference(q, k, v)
    out = nki_attention(q.to(device), k.to(device), v.to(device)).cpu()

    abs_err = (out.float() - ref.float()).abs().max().item()
    print(f"self_attn sp post-a2a abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"NKI sp self-attn diverged: {abs_err}"


def test_wrapper_signature_parity():
    """`attention_nki` (wrapper) routes through NKI on device with same shape."""
    device = _device()
    _maybe_skip(device)

    from wan.kernels.wan_attention_nki import attention_nki as attn

    torch.manual_seed(0)
    B, L, H, D = 1, 4400, 24, 128
    q = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, dtype=torch.bfloat16)

    ref = _sdpa_reference(q, k, v)
    out = attn(
        q.to(device), k.to(device), v.to(device),
        causal=False, dropout_p=0.0, dtype=torch.bfloat16,
    ).cpu()

    abs_err = (out.float() - ref.float()).abs().max().item()
    print(f"wrapper abs_err={abs_err:.3e}")
    assert abs_err < 5e-2, f"attention_nki wrapper diverged: {abs_err}"


if __name__ == "__main__":
    test_import_clean()
    test_cpu_fallback_matches_sdpa()
    print("CPU-side import + fallback tests pass.")
    if os.environ.get("NEURON_KERNEL_TEST") == "1":
        test_self_attention_self_17frame()
        test_cross_attention_text()
        test_self_attention_sp_post_a2a()
        test_wrapper_signature_parity()
        print("On-device tests pass.")
