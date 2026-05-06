"""Drop-in NKI replacement for the Wan2.2 real-valued 3D RoPE.

Mirrors both:
    * `wan.modules.model.rope_apply` (single device path)
    * `wan.distributed.sequence_parallel.rope_apply` (sequence-parallel path)

Public entrypoint:
    rope_apply_nki(x, grid_sizes, freqs, even_mask, odd_mask,
                   sp_rank=0, sp_size=1)

Behavior contract
-----------------
Given:
    x:          [B, L, N, D]    L = s_per_rank for SP, full L for single-device
    grid_sizes: [B, 3]
    freqs:      [max_seq, D//2] fp32 angle table
    even_mask:  [1, 1, D]       fp32 — kept in the signature for API parity
    odd_mask:   [1, 1, D]       fp32 — kept in the signature for API parity

Returns:
    y: [B, L, N, D] tensor, dtype = x.dtype.

On Neuron tensors, the kernel in `rope_nki._rope_kernel` is invoked
once per batch item. On CPU / non-Neuron tensors, the reference
real-valued masked-write impl is executed verbatim (extracted below
from `wan.modules.model.rope_apply`); this lets the wrapper be unit-
tested off-device and used as a drop-in for both code paths.

`even_mask` / `odd_mask` are NOT used by the kernel path — that
masked-write step is replaced by the channel-aligned cos_il/sin_il
multiplication inside the kernel. They are accepted in the signature
purely so this wrapper can substitute for the reference impl without
caller-side changes.

SP semantics
------------
The reference SP impl (`wan/distributed/sequence_parallel.py`) builds
the full f*h*w angle table on host, pads to `s * sp_size` with zero
angles (identity rotation on the tail), then slices
`[sp_rank * s : (sp_rank + 1) * s]`. We follow the same pattern:
host-side, build the per-batch-item angle table, pad to `L * sp_size`,
slice this rank, build cos_il/sin_il for those s tokens, then call the
NKI kernel. For single-device callers, pass `sp_rank=0, sp_size=1` and
the slice covers the whole sequence.

Why a wrapper module separate from `rope_nki.py`?
We split the high-level orchestration (signature parity with the two
existing rope_apply functions, host-side angle construction, batch
loop, CPU fallback) from the raw kernel + cos_il/sin_il helper. This
mirrors the rmsnorm pair (`rmsnorm_nki.py` + `wan_rms_norm_nki.py`)
already in this directory and keeps the kernel module focused on the
on-device math.

Integration
-----------
After validation, the two reference functions can be replaced with::

    from wan.kernels.wan_rope_nki import rope_apply_nki as rope_apply

(in `wan/modules/model.py`, with `sp_size=1`) and::

    from wan.kernels.wan_rope_nki import rope_apply_nki

    def rope_apply(x, grid_sizes, freqs, even_mask, odd_mask):
        return rope_apply_nki(
            x, grid_sizes, freqs, even_mask, odd_mask,
            sp_rank=get_rank(), sp_size=get_world_size(),
        )

(in `wan/distributed/sequence_parallel.py`).

Per the task brief, this PR does not perform that integration step —
it only ships the kernel and a tested drop-in wrapper.
"""
from __future__ import annotations

import torch

from .rope_nki import (
    _is_neuron_tensor,
    _rope_kernel,
    build_angles,
    _build_cos_sin_il,
)


# ---- CPU reference (real-valued masked-write) -------------------------
def _rope_apply_ref(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    even_mask: torch.Tensor,
    odd_mask: torch.Tensor,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> torch.Tensor:
    """Reference real-valued masked-write rope_apply.

    Matches `wan.modules.model.rope_apply` when sp_size == 1, and
    `wan.distributed.sequence_parallel.rope_apply` when sp_size > 1.
    Used as the CPU fallback when x is not on a Neuron device.
    """
    B, s, N, D = x.shape
    c = D // 2
    full_s = s * sp_size

    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]

    x_compute = x.float()
    freqs_compute = freqs.float()
    freqs_f = freqs_compute[:, :split_sizes[0]]
    freqs_h = freqs_compute[:, split_sizes[0]:split_sizes[0] + split_sizes[1]]
    freqs_w = freqs_compute[:, split_sizes[0] + split_sizes[1]:]

    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        token_len = f * h * w

        angles_full = torch.cat(
            [
                freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(token_len, 1, c)

        if full_s > token_len:
            pad = angles_full.new_zeros(full_s - token_len, 1, c)
            angles_full = torch.cat([angles_full, pad], dim=0)
        elif full_s < token_len:
            angles_full = angles_full[:full_s]

        angles = angles_full[sp_rank * s:(sp_rank + 1) * s]  # [s, 1, c]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_head = x_compute[i].reshape(s, N, c, 2)
        x0 = x_head[:, :, :, 0]
        x1 = x_head[:, :, :, 1]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        y0e = y0.repeat_interleave(2, dim=-1)
        y1e = y1.repeat_interleave(2, dim=-1)
        x_rot = y0e * even_mask + y1e * odd_mask  # [s, N, D]

        out.append(x_rot)

    return torch.stack(out, dim=0).to(dtype=x.dtype)


# ---- Public NKI entrypoint --------------------------------------------
def rope_apply_nki(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    even_mask: torch.Tensor,
    odd_mask: torch.Tensor,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> torch.Tensor:
    """Drop-in for `rope_apply` in `wan/modules/model.py` (sp_size=1) and
    `wan/distributed/sequence_parallel.py` (sp_size>1, sp_rank>=0).

    Args:
        x:          [B, L, N, D]. L = s_per_rank for SP, full L otherwise.
        grid_sizes: [B, 3] integer (f, h, w) tuples per batch item.
        freqs:      [max_seq, D//2] fp32 angle table from `rope_params`.
        even_mask:  [1, 1, D] fp32 — kept for API parity (unused on the
                    Neuron device path).
        odd_mask:   [1, 1, D] fp32 — kept for API parity.
        sp_rank:    this rank's index, default 0 (single-device).
        sp_size:    sequence-parallel world size, default 1.

    Returns:
        y: [B, L, N, D] tensor on the same device as x, dtype = x.dtype.
    """
    if x.dim() != 4:
        raise ValueError(f"rope_apply_nki expects [B, L, N, D], got {tuple(x.shape)}")
    B, s, N, D = x.shape
    c = D // 2
    if D % 2 != 0:
        raise ValueError(f"head_dim D must be even, got {D}")
    if freqs.shape[1] != c:
        raise ValueError(
            f"freqs must have D//2={c} channels, got {tuple(freqs.shape)}"
        )

    # CPU / non-neuron fallback -- reference impl verbatim.
    if not _is_neuron_tensor(x):
        return _rope_apply_ref(
            x, grid_sizes, freqs, even_mask, odd_mask, sp_rank, sp_size
        )

    full_s = s * sp_size
    grid_list = grid_sizes.tolist()

    out_pieces = []
    for i, (f, h, w) in enumerate(grid_list):
        token_len = int(f) * int(h) * int(w)

        # Build angle table on host (CPU). freqs may live on the Neuron
        # device; pull it to CPU for the angle construction since it is
        # static across the 60 rope calls per forward and PyTorch's
        # complex/trig ops on XLA generate extra NEFFs. Callers can cache
        # if needed; for now we rebuild per-call to keep the wrapper
        # stateless. Cost: O(f*h*w*c) -- ~270k fp32 ops, negligible vs
        # the 60-call rope budget.
        freqs_cpu = freqs.detach().to(device="cpu", dtype=torch.float32)
        angles_cpu = build_angles(int(f), int(h), int(w), freqs_cpu)  # [token_len, c]

        # cos_il / sin_il for this rank's s tokens, padded with identity
        # rotation on the tail.
        # We slice on cpu, then move to x's device.
        if full_s >= token_len:
            # Pad to full_s, then slice this rank.
            pad = torch.zeros(full_s - token_len, c, dtype=torch.float32)
            angles_full_cpu = torch.cat([angles_cpu, pad], dim=0)
        else:
            angles_full_cpu = angles_cpu[:full_s]

        angles_rank_cpu = angles_full_cpu[sp_rank * s:(sp_rank + 1) * s]  # [s, c]
        cos_il_cpu, sin_il_cpu = _build_cos_sin_il(angles_rank_cpu, target_len=s)

        cos_il = cos_il_cpu.to(device=x.device)
        sin_il = sin_il_cpu.to(device=x.device)

        # Per-batch-item kernel call.
        x_i = x[i].reshape(s, N * D).contiguous()
        y_flat = _rope_kernel(x_i, cos_il, sin_il)  # [s, N*D]
        out_pieces.append(y_flat.reshape(s, N, D))

    # Stack along batch.
    return torch.stack(out_pieces, dim=0).to(dtype=x.dtype)
