"""
Drop-in NKI replacement for the SDPA fallback in
``wan/modules/attention.py:attention``.

The signature mirrors the existing ``attention(q, k, v, ...)`` wrapper so
this can be swapped at the call site without changing argument plumbing in
``wan/modules/model.py`` or the SP forward in ``wan/distributed/``.

Routing rules:
    * On a Neuron device with the NKI kernel importable -> use ``nki_attention``.
    * Otherwise -> fall back to plain ``F.scaled_dot_product_attention``
      (CPU/CUDA dev workflows, unit tests, etc.).

This is a forward-only (inference) wrapper; the autograd backward path is
deliberately not implemented yet — see ``attention_nki.py`` docstring.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from .attention_nki import is_available as _nki_available
from .attention_nki import nki_attention

__all__ = ["attention_nki", "is_neuron_device"]


def is_neuron_device(t: torch.Tensor) -> bool:
    """True if ``t`` lives on the AWS Neuron / XLA device backend."""
    dev_type = t.device.type
    # PyTorch-XLA uses "xla"; some forks expose "neuron" directly.
    return dev_type in ("xla", "neuron", "privateuseone")


def attention_nki(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version=None,  # accepted for signature parity; ignored
) -> torch.Tensor:
    """Mirror of ``wan/modules/attention.py:attention`` that uses NKI on Neuron.

    Args:
        q, k, v: Tensors in Wan's native ``[B, L, H, D]`` layout.
        q_lens, k_lens: Padding mask vectors. NKI ``attention_cte`` does not
            currently consume these — when supplied, we warn and ignore them
            (mirroring the warning the existing SDPA fallback prints).
        dropout_p: Must be 0 for the NKI path (inference forward only).
        softmax_scale: ``Q@K^T`` multiplier; defaults to ``1/sqrt(D)``.
        q_scale: Optional pre-scale for ``q``.
        causal: Causal mask flag. Wan2.2 video DiT is always non-causal.
        window_size: Sliding window — NOT supported via NKI here yet.
        deterministic, dtype, fa_version: Accepted for signature parity.

    Returns:
        Tensor of shape ``[B, L_q, H, D]`` in ``q.dtype`` (NKI path) or in
        ``q.dtype`` after the SDPA fallback's transpose-back (CPU/CUDA path).
    """
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            "attention_nki: padding mask (q_lens/k_lens) is not propagated "
            "to NKI attention_cte; the kernel attends over the full L_q / "
            "L_kv. Pad inputs with zeros if you need masked behavior."
        )

    on_neuron = is_neuron_device(q) and is_neuron_device(k) and is_neuron_device(v)
    use_nki = on_neuron and _nki_available() and not causal and window_size == (-1, -1) \
        and dropout_p == 0.0

    if use_nki:
        # Apply q_scale and dtype cast (parity with the existing wrapper).
        if dtype is not None:
            q = q.to(dtype)
            k = k.to(dtype)
            v = v.to(dtype)
        if q_scale is not None:
            q = q * q_scale
        return nki_attention(q, k, v, scale=softmax_scale)

    # ---- Fallback: same code path as wan/modules/attention.py:attention ----
    if causal and window_size != (-1, -1):
        warnings.warn(
            "attention_nki fallback path: window_size is ignored by SDPA."
        )

    q = q.transpose(1, 2).to(dtype)  # [B, H, Lq, D]
    k = k.transpose(1, 2).to(dtype)  # [B, H, Lk, D]
    v = v.transpose(1, 2).to(dtype)  # [B, H, Lk, D]

    if q_scale is not None:
        q = q * q_scale

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=causal,
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return out.transpose(1, 2).contiguous()  # back to [B, Lq, H, D]
