"""
NKI fused attention for Wan2.2 (inference forward only).

This module wraps `nkilib.core.attention.attention_cte.attention_cte` —
the production-grade NKI Context-Encoding (prefill) attention kernel
documented at:
    https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/attention-cte.html

Public entrypoint: ``nki_attention(q, k, v, scale=None)``

Layout convention (matches the user-facing ``wan/modules/attention.py:attention``):
    * Inputs  q,k,v : ``[B, L, H, D]`` (batch, seq_len, num_heads, head_dim)
    * Output         : ``[B, L_q, H, D]``

The kernel itself takes a heads-folded-into-batch layout
``(batch_size, seqlen, d)`` so this entrypoint:
    1. permutes to ``[B, H, L, D]`` and flattens to ``[B*H, L, D]``,
    2. invokes ``attention_cte`` with ``causal_mask=False``,
    3. unflattens back to ``[B, L_q, H, D]``.

GQA / KV-fewer-heads layouts are supported transparently by ``attention_cte``
(``batch_size_kv`` may be smaller than ``batch_size``); for Wan2.2 the head
counts always match.

This is an inference-only forward. The training path lives elsewhere — wrap
this in a ``torch.autograd.Function`` (with ``cache_softmax=True``) when a
custom backward is added.
"""
from __future__ import annotations

import math

import torch

# ``attention_cte`` and the nki packages are only available on Neuron-flavored
# Python environments. Import lazily so this module can still be imported (and
# the CPU fallback path used) on hosts without the Neuron toolchain.
_ATTENTION_CTE_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - depends on host environment
    from nkilib.core.attention.attention_cte import attention_cte as _attention_cte
except Exception as e:  # noqa: BLE001 — surface any import failure later
    _attention_cte = None
    _ATTENTION_CTE_IMPORT_ERROR = e


__all__ = ["nki_attention", "is_available"]


def is_available() -> bool:
    """Whether the NKI ``attention_cte`` kernel can be imported in this env."""
    return _attention_cte is not None


def _require_kernel() -> None:
    if _attention_cte is None:
        raise RuntimeError(
            "nkilib.core.attention.attention_cte is not importable in this "
            "environment. Original error: "
            f"{_ATTENTION_CTE_IMPORT_ERROR!r}"
        )


def _fold_heads_into_batch(x: torch.Tensor) -> torch.Tensor:
    """[B, L, H, D] -> [B*H, L, D] (contiguous)."""
    if x.dim() != 4:
        raise ValueError(f"expected 4D [B,L,H,D] tensor, got shape {tuple(x.shape)}")
    b, l, h, d = x.shape
    # [B,L,H,D] -> [B,H,L,D] -> [B*H, L, D]
    return x.permute(0, 2, 1, 3).contiguous().reshape(b * h, l, d)


def _unfold_heads_from_batch(x: torch.Tensor, b: int, h: int) -> torch.Tensor:
    """[B*H, L, D] -> [B, L, H, D] (contiguous)."""
    if x.dim() != 3:
        raise ValueError(f"expected 3D [B*H,L,D] tensor, got shape {tuple(x.shape)}")
    bh, l, d = x.shape
    if bh != b * h:
        raise ValueError(f"batch*heads mismatch: got {bh}, expected {b*h}")
    return x.reshape(b, h, l, d).permute(0, 2, 1, 3).contiguous()


def nki_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Fused scaled-dot-product attention via NKI ``attention_cte``.

    Args:
        q: Query tensor of shape ``[B, L_q, H, D]``.
        k: Key   tensor of shape ``[B, L_kv, H_kv, D]``.
        v: Value tensor of shape ``[B, L_kv, H_kv, D]``. ``H`` must be divisible
           by ``H_kv`` (GQA); for plain MHA they're equal.
        scale: Multiplier for ``Q @ K^T`` before softmax. Defaults to
            ``1/sqrt(D)`` (the SDPA / "scaled" convention).

    Returns:
        Tensor of shape ``[B, L_q, H, D]`` matching ``q.dtype``.

    Notes:
        * Non-causal, no attention mask, no dropout — this matches the SDPA
          fallback in ``wan/modules/attention.py``.
        * Head dimension ``D`` must be ``<= 128`` (Wan uses 128 exactly).
    """
    _require_kernel()

    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"head_dim mismatch: q={q.shape[-1]} k={k.shape[-1]} v={v.shape[-1]}"
        )
    if q.shape[-1] > 128:
        raise ValueError(
            f"attention_cte requires head_dim <= 128 (got {q.shape[-1]})"
        )
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError(
            f"k/v shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(
            f"num_heads ({q.shape[2]}) must be divisible by num_kv_heads "
            f"({k.shape[2]}) for GQA"
        )

    B, L_q, H, D = q.shape
    _, L_kv, H_kv, _ = k.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Fold heads into batch. attention_cte handles GQA natively when
    # batch_size_kv < batch_size (here B*H_kv < B*H).
    q_flat = _fold_heads_into_batch(q)        # [B*H,    L_q,  D]
    k_flat = _fold_heads_into_batch(k)        # [B*H_kv, L_kv, D]
    v_flat = _fold_heads_into_batch(v)        # [B*H_kv, L_kv, D]

    out_flat = _attention_cte(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        scale=float(scale),
        causal_mask=False,
        # all other features (sliding window, prefix cache, sink, CP) disabled
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
    )                                         # [B*H, L_q, D]

    out = _unfold_heads_from_batch(out_flat, B, H)  # [B, L_q, H, D]
    return out.to(q.dtype)
