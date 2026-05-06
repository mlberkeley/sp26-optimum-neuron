"""Spatial-tiled VAE decode.

Wraps `Wan2_2_VAE.model.decode` so that the latent input `z` is split
spatially into a 2x2 grid of tiles with halo overlap, each tile is decoded
independently, and the resulting pixel tiles are composited on CPU using a
linear ramp blend in the halo region. This keeps each per-iter decoder
intermediate within the per-LNC HBM budget on Trainium2 — the bottleneck
that blocks f81 on the standalone (untiled) VAE-Neuron path.

Shape contract
--------------
Input:
    vae_model: a `WanVAE_` instance (the inner module of `Wan2_2_VAE`)
    z        : [B, C=48, F_lat, H_lat, W_lat] latent. H_lat and W_lat
               must each be divisible by 2 (we split at the midpoint).
    scale    : [mean, 1/std] list-of-tensors as used by `vae_model.decode`.
    halo_lat : halo overlap *per side* of the tile boundary, in latent
               pixels. The default of 2 yields a 16-pixel-wide blend zone
               in the upsampled output (vae_stride=8).

Output:
    pixel tensor, same shape and device as `vae_model.decode(z, scale)`
    would produce, but constructed on CPU.

Why "on CPU"?
- Each tile's decode runs on Neuron, but its output is moved to CPU
  immediately and freed from device, mirroring the streaming-decode probe
  approach. The composited final frame buffer at f81 (~530 MB fp32) is
  small enough to live on CPU; allocating it on Neuron would fight the
  exact HBM pressure we're trying to avoid.

Correctness strategy
--------------------
- Halo overlap is symmetric: each interior tile has `halo_lat` extra
  latent pixels on every shared boundary. The blend weights ramp from
  0 at the *outer* edge of the halo to 1 at the *inner* edge (i.e.
  `halo_lat * vae_stride[1]` = 16 pixels of linear blend per boundary
  by default).
- The receptive field of the VAE decoder is small enough (3x3 spatial
  3D-conv stack with stride-2 transpose-convs) that artifacts from the
  boundary effect on a single tile are confined to within ~16 latent
  pixels of the cut. Halo of 2 (= 16 px blend zone) gives ~3x receptive-
  field coverage, which empirical practice from latent-diffusion
  tilers suggests is sufficient.
- A unit test (tests/vae/test_tiled_decode.py) compares this wrapper's
  output to the untiled `vae_model.decode` on f17 (the smaller shape that
  fits without tiling) and asserts agreement within bf16 tolerance.
"""
from __future__ import annotations

from typing import List

import torch


# vae_stride[1] == vae_stride[2] == 16 for Wan2.2 TI2V-5B (see
# `wan/configs/wan_ti2v_5B.py`). For the A14B family the value is 8 — if
# we ever wire the tiled wrapper up there too, this needs to be made
# config-driven or passed in. Verified empirically against the actual
# decode at f17 (22 lat -> 352 pix on TI2V-5B).
_SPATIAL_UPSAMPLE = 16


def _ramp(n: int) -> torch.Tensor:
    """Linear ramp from 0->1 over `n` samples, exclusive of endpoints.

    Used as the blend weight for a halo region of width `n` pixels:
    weight is 0 at the far edge of the halo (where the tile ends) and
    1 at the inner edge (where the next tile takes over). The endpoints
    are excluded because the actual tile boundary sits one step beyond
    the halo end, and we want a smooth ramp without zero or one as
    explicit values (avoids /0 in the normalization and lets the
    overlapping tile pick up the full weight at the interior).
    """
    if n <= 0:
        return torch.empty(0)
    # n+2 samples: drop the outermost 0 and 1, keep the n in between.
    return torch.linspace(0.0, 1.0, n + 2)[1:-1]


def _tile_weight_1d(extent_px: int, halo_px: int,
                    is_lo_edge: bool, is_hi_edge: bool) -> torch.Tensor:
    """Build the per-pixel blend weight for one tile along one spatial axis.

    Returns a [extent_px] float tensor that is:
    - 1.0 across the interior of the tile,
    - linearly ramping from 0->1 across the lower halo (if `is_lo_edge=False`,
      meaning this tile shares its lower boundary with another tile),
    - linearly ramping from 1->0 across the upper halo (mirror).

    Outer edges (is_lo_edge / is_hi_edge True) get weight 1 right up to
    the edge — there's no neighbor to blend against there.
    """
    w = torch.ones(extent_px, dtype=torch.float32)
    if not is_lo_edge:
        ramp = _ramp(halo_px)
        w[:halo_px] = ramp
    if not is_hi_edge:
        ramp = _ramp(halo_px).flip(0)
        w[-halo_px:] = ramp
    return w


def tiled_decode(vae_model, z: torch.Tensor, scale,
                 halo_lat: int = 2) -> torch.Tensor:
    """Decode `z` via 2x2 spatial tiling on the device of `z`.

    See module docstring for the contract.

    Implementation:
      1. Pre-compute the tile latent ranges (h_lo, h_hi, w_lo, w_hi).
      2. For each tile, decode on-device; move output tile to CPU; del tile.
      3. Composite on CPU with linear-ramp blending in the halo regions.
    """
    if z.dim() != 5:
        raise ValueError(f"tiled_decode expects [B,C,F,H,W], got {tuple(z.shape)}")
    B, C, F_lat, H_lat, W_lat = z.shape
    if H_lat % 2 != 0 or W_lat % 2 != 0:
        raise ValueError(
            f"tiled_decode requires even H_lat,W_lat for 2x2 split; "
            f"got H_lat={H_lat}, W_lat={W_lat}"
        )
    if halo_lat <= 0:
        raise ValueError(f"halo_lat must be positive, got {halo_lat}")
    if H_lat // 2 <= halo_lat or W_lat // 2 <= halo_lat:
        raise ValueError(
            f"halo_lat={halo_lat} too large for H_lat={H_lat}, W_lat={W_lat} "
            f"(half-extent must exceed halo)"
        )

    h_mid = H_lat // 2
    w_mid = W_lat // 2

    # Symmetric halo: each interior tile carries halo_lat extra latent
    # pixels into its neighbor's territory.
    tile_specs = [
        # (h_lo, h_hi, w_lo, w_hi, is_h_lo_edge, is_h_hi_edge,
        #  is_w_lo_edge, is_w_hi_edge)
        (0, h_mid + halo_lat, 0, w_mid + halo_lat,
         True, False, True, False),               # top-left
        (0, h_mid + halo_lat, w_mid - halo_lat, W_lat,
         True, False, False, True),               # top-right
        (h_mid - halo_lat, H_lat, 0, w_mid + halo_lat,
         False, True, True, False),               # bottom-left
        (h_mid - halo_lat, H_lat, w_mid - halo_lat, W_lat,
         False, True, False, True),               # bottom-right
    ]

    halo_px = halo_lat * _SPATIAL_UPSAMPLE
    H_px = H_lat * _SPATIAL_UPSAMPLE
    W_px = W_lat * _SPATIAL_UPSAMPLE

    # Decode each tile on the device, immediately stream output to CPU.
    cpu_tiles: List[tuple] = []  # list of (px_ranges, tile_cpu)
    for (h_lo, h_hi, w_lo, w_hi,
         h_lo_edge, h_hi_edge, w_lo_edge, w_hi_edge) in tile_specs:
        tile = z[:, :, :, h_lo:h_hi, w_lo:w_hi].contiguous()
        # vae_model.decode internally calls clear_cache() at the start, so
        # each tile runs with a fresh per-iter feat_cache. That's correct:
        # tiles are independent decodes; cache is intra-decode state.
        tile_out = vae_model.decode(tile, scale)
        tile_out_cpu = tile_out.to("cpu")
        del tile_out, tile
        cpu_tiles.append(
            ((h_lo * _SPATIAL_UPSAMPLE, h_hi * _SPATIAL_UPSAMPLE,
              w_lo * _SPATIAL_UPSAMPLE, w_hi * _SPATIAL_UPSAMPLE,
              h_lo_edge, h_hi_edge, w_lo_edge, w_hi_edge),
             tile_out_cpu)
        )

    # Composite on CPU. Output shape matches an untiled decode: same
    # leading dims, same temporal extent (taken from any tile, since they
    # all decode the same temporal length), same spatial = H_lat*8 by
    # W_lat*8.
    sample_tile = cpu_tiles[0][1]
    out_shape = list(sample_tile.shape)
    # Replace tile spatial dims with full-frame dims.
    out_shape[-2] = H_px
    out_shape[-1] = W_px
    out_dtype = sample_tile.dtype
    accum = torch.zeros(out_shape, dtype=torch.float32)
    weight_sum = torch.zeros(out_shape, dtype=torch.float32)

    for (h_lo_px, h_hi_px, w_lo_px, w_hi_px,
         h_lo_edge, h_hi_edge, w_lo_edge, w_hi_edge), tile_cpu in cpu_tiles:
        h_extent = h_hi_px - h_lo_px
        w_extent = w_hi_px - w_lo_px

        h_w = _tile_weight_1d(h_extent, halo_px, h_lo_edge, h_hi_edge)
        w_w = _tile_weight_1d(w_extent, halo_px, w_lo_edge, w_hi_edge)
        # Outer product to get a 2D weight, then broadcast over leading dims.
        weight_2d = h_w.unsqueeze(1) * w_w.unsqueeze(0)  # [h_extent, w_extent]
        # Build a shape that broadcasts: prepend 1s for all leading dims,
        # then [h_extent, w_extent]. tile_cpu is [..., h_extent, w_extent].
        bcast_shape = [1] * (tile_cpu.dim() - 2) + [h_extent, w_extent]
        weight_bcast = weight_2d.view(bcast_shape)

        tile_f32 = tile_cpu.to(torch.float32)
        accum[..., h_lo_px:h_hi_px, w_lo_px:w_hi_px] += tile_f32 * weight_bcast
        weight_sum[..., h_lo_px:h_hi_px, w_lo_px:w_hi_px] += weight_bcast

    # Normalize. weight_sum is strictly positive everywhere because every
    # output pixel is covered by at least one tile, and the blend weights
    # are >0 over the halo and ==1 over the interior. Clamp defensively.
    out = accum / weight_sum.clamp(min=1e-6)
    return out.to(out_dtype)
