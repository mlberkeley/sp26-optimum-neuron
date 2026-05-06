"""NKI 3D-RoPE kernel for Wan2.2 (real-valued, masked-write form).

Mathematical contract (matches `wan.modules.model.rope_apply` and
`wan.distributed.sequence_parallel.rope_apply` for a single batch item):

For every token position `t` and head `n`, treating the per-head vector
`x[t, n, :]` as `c = D//2` complex pairs `(x[t, n, 2k], x[t, n, 2k+1])`,
multiply by `cos[t, k] + i*sin[t, k]`:
    y0[t, n, k] = x[t, n, 2k]   * cos[t, k] - x[t, n, 2k+1] * sin[t, k]
    y1[t, n, k] = x[t, n, 2k]   * sin[t, k] + x[t, n, 2k+1] * cos[t, k]
    out[t, n, 2k]   = y0[t, n, k]
    out[t, n, 2k+1] = y1[t, n, k]

The reference implementation expresses the final interleaved write via
`repeat_interleave(2)` + masked add (`y0e * even_mask + y1e * odd_mask`).
This kernel implements the same numerics by working with channel-aligned
broadcast tables computed host-side, which removes the need for in-kernel
gather/scatter through complex strided views.

Strategy choice
---------------
We use **strategy (b)**: the host pre-builds two channel-aligned tables
per batch item:

    cos_il[token_len_padded, D]   each cos[t, k] copied to lanes 2k and 2k+1
    sin_il[token_len_padded, D]   each sin[t, k] *signed* per lane:
                                    lane 2k    -> -sin[t, k]
                                    lane 2k+1  -> +sin[t, k]

With these tables the rotation reduces to:

    out[t, n, d] = x[t, n, d] * cos_il[t, d]
                 + x_swap[t, n, d] * sin_il[t, d]

where `x_swap` is `x` with each adjacent (2k, 2k+1) pair swapped. The
swap is done **inside the kernel** using a single strided DMA from HBM
into an SBUF view shaped `(P, N, c, 2)` whose final-axis stride is
flipped — i.e. we reinterpret the contiguous `[P, N, c, 2]` tile into a
`[P, N*D]` flat where the inner pair indices have been transposed.

Doing the swap in-kernel keeps host preprocessing to two D-wide tables
(cheap to build and reuse across the 60 RoPE calls per forward), avoids
an extra HBM-sized buffer for `x_swapped`, and means the kernel reads x
exactly twice (once for the straight x*cos_il path, once via the swapped
view for the x_swap*sin_il path).

Why not strategy (a)? Building f/h/w angle tables and the cos/sin trig
internally would require either (i) `nl.cos`/`nl.sin` activation engine
ops over a `[token_len, c]` table allocated in SBUF (~1 MB fp32 for
`c=64` and `token_len=4256`, fits), plus a Cartesian-product expansion
along the free dim that is awkward to fuse cleanly with the data loop.
Building it on host is a one-time cost amortized across 30 attention
blocks * 2 (norm_q, norm_k) = 60 calls per forward, all with the same
grid_sizes; the host can cache the cos_il/sin_il tables.

Tile sizing
-----------
- Partition axis (P): tokens. P_MAX = 128 (`nl.tile_size.pmax`). For
  TI2V-5B 17-frame chunks single device L=4400 -> 35 tiles, last 80.
  SP=8 per-rank L=550 -> 5 tiles, last 38.
- Free axis: per-head D = 128. Per-token row = N*D = 24*128 = 3072 fp32
  -> 1.5 MiB SBUF working set per tile, well under 24 MiB.
- cos_il / sin_il loaded per tile as `[P, D]` fp32 (128 * 128 * 4B =
  64 KiB) and broadcast across the N-head axis using N strided
  `tensor_tensor` calls along contiguous N-slabs of the data tile.

Beta-2 / NKI 0.3.0 APIs used
----------------------------
- `@nki.jit`
- `nl.ndarray(..., buffer=nl.sbuf|nl.shared_hbm)`
- `nisa.dma_copy(dst=, src=)` for HBM<->SBUF traffic
- `nisa.tensor_copy` for dtype promotion (bf16 -> fp32) and PSUM->SBUF
  (not used here, no matmul)
- `nisa.tensor_tensor(op=nl.multiply|nl.add)` element-wise
- `nl.ds(offset, size)` for free-axis sub-ranges

This kernel does NOT use:
- `nl.load` / `nl.store` (Beta 1, removed)
- `nl.mgrid` / `nl.arange` (Beta 1, removed)
- mask= on ISA calls (Beta 1, removed)

Tail (padding) handling
-----------------------
The tail rows beyond `token_len` are passed through unchanged in the
real-valued reference. Host pre-builds cos_il=1.0, sin_il=0.0 for those
rows so the rotation reduces to the identity, matching the reference's
behavior in both `model.py` (where `x_compute[i, seq_len:]` is
re-concatenated) and `sequence_parallel.py` (where padded tokens get
identity rotation by construction).
"""
from __future__ import annotations

import torch

import nki
import nki.isa as nisa
import nki.language as nl


# ---- inline helpers (avoid runtime nkilib install dependency) ---------
def kernel_assert(cond, msg):
    if not cond:
        raise AssertionError(f"[NCC_INKI016] Kernel validation exception: {msg}")


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


# ---- the NKI kernel ---------------------------------------------------
@nki.jit
def _rope_kernel(x_hbm, cos_il_hbm, sin_il_hbm):
    """Real-valued 3D RoPE forward.

    Args:
        x_hbm:      [L, N*D] in source dtype (bf16 or fp32). The token
                    axis is partitioned across SBUF; (N, D) live entirely
                    in the free dimension. Contiguous.
        cos_il_hbm: [L, D]   fp32. Per-token, channel-interleaved cos:
                    cos_il[t, 2k] = cos_il[t, 2k+1] = cos(angle[t, k]).
        sin_il_hbm: [L, D]   fp32. Per-token, channel-interleaved *signed*
                    sin: sin_il[t, 2k] = -sin(angle[t, k]),
                    sin_il[t, 2k+1] = +sin(angle[t, k]).

    Returns:
        y_hbm: [L, N*D] bf16 (matches reference impl's final cast back to
        x.dtype via `torch.stack(out).to(dtype=x.dtype)`).
    """
    L, N_times_D = x_hbm.shape
    L2, D = cos_il_hbm.shape
    L3, D2 = sin_il_hbm.shape

    kernel_assert(L == L2 == L3, "rope: token axis mismatch x vs cos/sin")
    kernel_assert(D == D2, "rope: head_dim mismatch cos vs sin")
    kernel_assert(N_times_D % D == 0, "rope: N*D not divisible by D")

    N = N_times_D // D
    c2 = D  # full head dim; pair size = 2, so c = D//2

    TILE_L = nl.tile_size.pmax  # 128

    # Output bf16 to match reference final cast.
    y_hbm = nl.ndarray((L, N_times_D), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

    num_tiles = div_ceil(L, TILE_L)

    for tile_idx in nl.affine_range(num_tiles):
        t_start = tile_idx * TILE_L
        t_end = min(t_start + TILE_L, L)
        t_sz = t_end - t_start

        # ---- Load cos_il / sin_il as [P, D] fp32. -------------------
        cos_sb = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
        sin_sb = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cos_sb, src=cos_il_hbm[t_start:t_end, 0:D])
        nisa.dma_copy(dst=sin_sb, src=sin_il_hbm[t_start:t_end, 0:D])

        # ---- Load x tile in source dtype, promote to fp32. ----------
        x_in = nl.ndarray((t_sz, N_times_D), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_in, src=x_hbm[t_start:t_end, 0:N_times_D])

        x_f32 = nl.ndarray((t_sz, N_times_D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_f32, src=x_in)

        # ---- Build x_swap: x with each (2k, 2k+1) pair swapped. -----
        # x_swap is the same data, but with the inner pair lane flipped.
        # We materialize it by viewing x as [P, N, c, 2] and copying
        # lanes [..., 0] and [..., 1] into swapped slots.
        # x[..., 0]  -> x_swap[..., 1]
        # x[..., 1]  -> x_swap[..., 0]
        #
        # In flat [P, N*D] coordinates:
        #   x_swap[..., 2k]   = x[..., 2k+1]
        #   x_swap[..., 2k+1] = x[..., 2k]
        #
        # We do this with two strided tensor_copy calls per N-slab:
        # one copies even->odd, one copies odd->even, both with stride 2
        # along the free axis.
        x_swap = nl.ndarray((t_sz, N_times_D), dtype=nl.float32, buffer=nl.sbuf)

        # 4D views: contiguous reshape [P, N*D] <-> [P, N, c, 2] is just a
        # reinterpretation of the free axis since the data is contiguous.
        c = D // 2
        x_f32_4d = x_f32.reshape((t_sz, N, c, 2))
        x_swap_4d = x_swap.reshape((t_sz, N, c, 2))

        # Even <- Odd
        nisa.tensor_copy(
            dst=x_swap_4d[0:t_sz, 0:N, 0:c, 0:1],
            src=x_f32_4d[0:t_sz, 0:N, 0:c, 1:2],
        )
        # Odd <- Even
        nisa.tensor_copy(
            dst=x_swap_4d[0:t_sz, 0:N, 0:c, 1:2],
            src=x_f32_4d[0:t_sz, 0:N, 0:c, 0:1],
        )

        # ---- Apply rotation -----------------------------------------
        # out = x * cos_il + x_swap * sin_il (sin_il already signed)
        # Both cos_sb and sin_sb are [P, D]; for each of the N heads we
        # multiply the matching D-wide slab by the same [P, D] table.
        # We loop over N (compile-time-unrolled via nl.affine_range) to
        # express the broadcast across the head axis.
        out_f32 = nl.ndarray((t_sz, N_times_D), dtype=nl.float32, buffer=nl.sbuf)

        for n_idx in nl.affine_range(N):
            d_start = n_idx * D
            d_end = d_start + D

            # term0 = x * cos_il (per-head slab)
            term0 = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=term0,
                data1=x_f32[0:t_sz, d_start:d_end],
                data2=cos_sb,
                op=nl.multiply,
            )

            # term1 = x_swap * sin_il (signed)
            term1 = nl.ndarray((t_sz, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=term1,
                data1=x_swap[0:t_sz, d_start:d_end],
                data2=sin_sb,
                op=nl.multiply,
            )

            # out_slab = term0 + term1
            nisa.tensor_tensor(
                dst=out_f32[0:t_sz, d_start:d_end],
                data1=term0,
                data2=term1,
                op=nl.add,
            )

        # ---- Cast to output dtype and store -------------------------
        out_lo = nl.ndarray((t_sz, N_times_D), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_lo, src=out_f32)
        nisa.dma_copy(dst=y_hbm[t_start:t_end, 0:N_times_D], src=out_lo)

    return y_hbm


# ---- Python entrypoint ------------------------------------------------
def _is_neuron_tensor(t: torch.Tensor) -> bool:
    if t is None:
        return False
    return t.device.type in ("xla", "neuron")


def _build_cos_sin_il(
    angles: torch.Tensor, target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build channel-interleaved cos_il, sin_il from per-token angles.

    Args:
        angles: [token_len, c] fp32 — the per-token, per-channel angles.
        target_len: pad/trim length along token axis.

    Returns:
        (cos_il, sin_il), each [target_len, 2*c] fp32. Tail rows beyond
        token_len are identity rotation (cos=1, sin=0). Lane signing
        matches the kernel contract:
            cos_il[..., 2k]   = cos[..., k]
            cos_il[..., 2k+1] = cos[..., k]
            sin_il[..., 2k]   = -sin[..., k]
            sin_il[..., 2k+1] = +sin[..., k]
    """
    token_len, c = angles.shape
    D = 2 * c

    cos = torch.cos(angles)  # [token_len, c]
    sin = torch.sin(angles)  # [token_len, c]

    # Pad to target_len with zero angles -> cos=1, sin=0
    if target_len > token_len:
        cos_full = torch.ones(target_len, c, dtype=cos.dtype, device=cos.device)
        sin_full = torch.zeros(target_len, c, dtype=sin.dtype, device=sin.device)
        cos_full[:token_len] = cos
        sin_full[:token_len] = sin
    elif target_len == token_len:
        cos_full = cos
        sin_full = sin
    else:
        cos_full = cos[:target_len]
        sin_full = sin[:target_len]

    # Channel-interleave to [target_len, D]:
    cos_il = cos_full.unsqueeze(-1).expand(target_len, c, 2).reshape(target_len, D)

    # sin_il signed: even lane -sin, odd lane +sin
    sin_neg = -sin_full
    sin_il = (
        torch.stack([sin_neg, sin_full], dim=-1)  # [target_len, c, 2]
        .reshape(target_len, D)
        .contiguous()
    )

    return cos_il.contiguous(), sin_il.contiguous()


def build_angles(
    grid_f: int,
    grid_h: int,
    grid_w: int,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Build the [f*h*w, c] angle table from the (f, h, w) grid + freqs table.

    Mirrors the per-batch-item angle construction in `wan/modules/model.py`
    rope_apply. `freqs` is [max_seq, c] fp32, sliced into f/h/w sub-tables
    along the channel axis.
    """
    f, h, w = grid_f, grid_h, grid_w
    freqs = freqs.to(torch.float32)
    c = freqs.shape[1]
    split_sizes = [c - 2 * (c // 3), c // 3, c // 3]

    freqs_f = freqs[:, :split_sizes[0]]
    freqs_h = freqs[:, split_sizes[0]:split_sizes[0] + split_sizes[1]]
    freqs_w = freqs[:, split_sizes[0] + split_sizes[1]:]

    angles = torch.cat(
        [
            freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, c)

    return angles  # [token_len, c] fp32


def nki_rope_apply(
    x: torch.Tensor,
    cos_il: torch.Tensor,
    sin_il: torch.Tensor,
) -> torch.Tensor:
    """NKI entry point: rotate `x` by pre-built cos_il / sin_il tables.

    Args:
        x:      [L, N, D] tensor on Neuron device.
        cos_il: [L, D]   fp32 channel-interleaved cos.
        sin_il: [L, D]   fp32 channel-interleaved *signed* sin (see kernel).

    Returns:
        y: [L, N, D] same dtype as x (the kernel internally promotes to
        fp32 and casts back). The result matches the real-valued
        masked-write reference up to fp32 rounding.

    Notes:
        - This entrypoint expects a single batch item (L = token axis).
          Callers handle batching by either iterating over B or stacking
          batch items along L (only safe if the tables match).
        - On non-Neuron devices, a CPU PyTorch fallback that mirrors the
          reference math exactly is invoked.
    """
    if x.dim() != 3:
        raise ValueError(f"nki_rope_apply expects [L, N, D], got {tuple(x.shape)}")
    L, N, D = x.shape
    if cos_il.shape != (L, D):
        raise ValueError(
            f"cos_il must be [L={L}, D={D}], got {tuple(cos_il.shape)}"
        )
    if sin_il.shape != (L, D):
        raise ValueError(
            f"sin_il must be [L={L}, D={D}], got {tuple(sin_il.shape)}"
        )
    if D % 2 != 0:
        raise ValueError(f"head_dim D must be even, got {D}")

    # Non-neuron path: bypass kernel, do reference math.
    if not _is_neuron_tensor(x):
        return _rope_apply_torch_ref(x, cos_il, sin_il)

    x_flat = x.reshape(L, N * D).contiguous()
    cos_il_c = cos_il.to(torch.float32).contiguous()
    sin_il_c = sin_il.to(torch.float32).contiguous()

    y_flat = _rope_kernel(x_flat, cos_il_c, sin_il_c)
    return y_flat.reshape(L, N, D)


def _rope_apply_torch_ref(
    x: torch.Tensor, cos_il: torch.Tensor, sin_il: torch.Tensor
) -> torch.Tensor:
    """CPU reference matching the kernel contract: out = x*cos_il + swap(x)*sin_il."""
    L, N, D = x.shape
    in_dtype = x.dtype
    xf = x.to(torch.float32)
    cos_il = cos_il.to(torch.float32).reshape(L, 1, D)
    sin_il = sin_il.to(torch.float32).reshape(L, 1, D)
    # Pair-swap along the last dim
    xf_pairs = xf.reshape(L, N, D // 2, 2)
    x_swap = xf_pairs.flip(-1).reshape(L, N, D)
    out = xf * cos_il + x_swap * sin_il
    return out.to(in_dtype)
