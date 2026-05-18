# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""NKI fast path for Wan VAE cached causal Conv3d.

This first kernel is intentionally narrow: decoder-time 3x3x3, stride-1,
groups=1 convolutions with a two-frame temporal cache. The caller supplies
spatially padded input/cache tensors plus weights packed as [KT, KH, KW, C, OC],
so the kernel only needs to handle the causal temporal window and the
channel/spatial matmul work.
"""

from __future__ import annotations

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx import nki_op, wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _NKI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only off Neuron images.
    nki = None
    nisa = None
    nl = None
    nki_op = None
    get_logical_neuron_cores = None
    wrap_nki = None
    _NKI_IMPORT_ERROR = exc


CACHE_T = 2
CHANNEL_TILE = 128
OUT_CHANNEL_TILE = 128
WIDTH_TILE = 512
HEIGHT_BLOCK = 6
ENABLE_SPATIAL_PACK_ENTRY = False
SPATIAL_PACK_CHANNEL_TILE = 14
SPATIAL_PACK_MIN_CHANNELS = 32
SPATIAL_PACK_MAX_CHANNELS = 64
HEAD_IN_CHANNELS = 256
HEAD_OUT_CHANNELS = 12
HEAD_OUT_CHANNEL_PAD = 16
HEAD_WIDTH_TILE = 160
HEAD_HEIGHT_BLOCK = 3


def nki_available() -> bool:
    return _NKI_IMPORT_ERROR is None


def nki_import_error() -> Exception | None:
    return _NKI_IMPORT_ERROR


if nki_available():

    @nki.jit
    def _causal_conv3d_ks3s1_cached_kernel(x, cache_x, weight_packed, bias):
        batch, in_channels, time, height_padded, width_padded = x.shape
        _, _, _, _, out_channels = weight_packed.shape
        _, _, cache_time, _, _ = cache_x.shape
        height = height_padded - 2
        width = width_padded - 2

        output = nl.ndarray(
            (batch, out_channels, time, height, width),
            dtype=x.dtype,
            buffer=nl.shared_hbm,
        )

        channel_tiles = (in_channels + CHANNEL_TILE - 1) // CHANNEL_TILE
        out_channel_tiles = (out_channels + OUT_CHANNEL_TILE - 1) // OUT_CHANNEL_TILE
        width_tiles = (width + WIDTH_TILE - 1) // WIDTH_TILE
        height_block_size = min(HEIGHT_BLOCK, max(1, WIDTH_TILE // width))
        height_blocks = (height + height_block_size - 1) // height_block_size
        work_tiles = batch * time * out_channel_tiles
        program_count = nl.num_programs(0)
        program_idx = nl.program_id(0)

        for linear_idx in nl.affine_range(program_idx, work_tiles, program_count):
            oc_tile = linear_idx % out_channel_tiles
            work_idx = linear_idx // out_channel_tiles
            t_out = work_idx % time
            b = work_idx // time

            oc_start = oc_tile * OUT_CHANNEL_TILE
            oc_size = min(OUT_CHANNEL_TILE, out_channels - oc_start)

            bias_tile = nl.ndarray(
                (oc_size, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=bias_tile[:, 0],
                src=bias[oc_start : oc_start + oc_size],
            )

            if (
                ENABLE_SPATIAL_PACK_ENTRY
                and (
                    SPATIAL_PACK_MIN_CHANNELS
                    <= in_channels
                    <= SPATIAL_PACK_MAX_CHANNELS
                )
            ):
                packed_channel_tiles = (
                    in_channels + SPATIAL_PACK_CHANNEL_TILE - 1
                ) // SPATIAL_PACK_CHANNEL_TILE
                packed_weight_tiles = []
                for kt in nl.affine_range(3):
                    for c_tile in nl.affine_range(packed_channel_tiles):
                        c_start = c_tile * SPATIAL_PACK_CHANNEL_TILE
                        c_size = min(
                            SPATIAL_PACK_CHANNEL_TILE,
                            in_channels - c_start,
                        )
                        packed_size = c_size * 9
                        weight_tile = nl.ndarray(
                            (packed_size, oc_size),
                            dtype=output.dtype,
                            buffer=nl.sbuf,
                        )
                        for kh in nl.affine_range(3):
                            for kw in nl.affine_range(3):
                                packed_start = (kh * 3 + kw) * c_size
                                nisa.tensor_copy(
                                    dst=weight_tile[
                                        packed_start : packed_start + c_size,
                                        :,
                                    ],
                                    src=nl.load(
                                        weight_packed[
                                            kt,
                                            kh,
                                            kw,
                                            c_start : c_start + c_size,
                                            oc_start : oc_start + oc_size,
                                        ],
                                        dtype=output.dtype,
                                    ),
                                )
                        packed_weight_tiles.append(weight_tile)

                for w_tile in nl.affine_range(width_tiles):
                    w_start = w_tile * WIDTH_TILE
                    w_size = min(WIDTH_TILE, width - w_start)

                    for h_block in nl.affine_range(height_blocks):
                        h_start = h_block * height_block_size
                        h_size = min(height_block_size, height - h_start)
                        output_tile_size = h_size * w_size
                        acc = nl.zeros(
                            (oc_size, output_tile_size),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )

                        for kt in nl.affine_range(3):
                            src_t = t_out + kt - CACHE_T
                            use_current = src_t >= 0
                            use_cache = (src_t < 0) & (src_t >= -cache_time)

                            if use_current | use_cache:
                                for c_tile in nl.affine_range(packed_channel_tiles):
                                    c_start = c_tile * SPATIAL_PACK_CHANNEL_TILE
                                    c_size = min(
                                        SPATIAL_PACK_CHANNEL_TILE,
                                        in_channels - c_start,
                                    )
                                    packed_size = c_size * 9
                                    input_tile = nl.ndarray(
                                        (packed_size, output_tile_size),
                                        dtype=output.dtype,
                                        buffer=nl.sbuf,
                                    )
                                    for kh in nl.affine_range(3):
                                        h_in = h_start + kh
                                        for kw in nl.affine_range(3):
                                            packed_start = (kh * 3 + kw) * c_size
                                            if use_current:
                                                input_block = nl.load(
                                                    x[
                                                        b,
                                                        c_start : c_start + c_size,
                                                        src_t,
                                                        h_in : h_in + h_size,
                                                        w_start
                                                        + kw : w_start
                                                        + kw
                                                        + w_size,
                                                    ],
                                                    dtype=output.dtype,
                                                )
                                            else:
                                                cache_src_t = cache_time + src_t
                                                input_block = nl.load(
                                                    cache_x[
                                                        b,
                                                        c_start : c_start + c_size,
                                                        cache_src_t,
                                                        h_in : h_in + h_size,
                                                        w_start
                                                        + kw : w_start
                                                        + kw
                                                        + w_size,
                                                    ],
                                                    dtype=output.dtype,
                                                )
                                            nisa.tensor_copy(
                                                dst=input_tile[
                                                    packed_start : packed_start
                                                    + c_size,
                                                    :,
                                                ],
                                                src=input_block.reshape(
                                                    (c_size, output_tile_size)),
                                            )

                                    weight_idx = kt * packed_channel_tiles + c_tile
                                    nisa.nc_matmul(
                                        dst=acc[:, :],
                                        stationary=packed_weight_tiles[weight_idx],
                                        moving=input_tile,
                                        accumulate=True,
                                    )

                        out_tile = nl.ndarray(
                            (oc_size, output_tile_size),
                            dtype=output.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=out_tile,
                            data=acc,
                            op0=nl.add,
                            operand0=bias_tile,
                        )
                        nl.store(
                            output[
                                b,
                                oc_start : oc_start + oc_size,
                                t_out,
                                h_start : h_start + h_size,
                                w_start : w_start + w_size,
                            ],
                            value=out_tile.reshape((oc_size, h_size, w_size)),
                        )
            else:
                weight_tiles = []
                for kt in nl.affine_range(3):
                    for kh in nl.affine_range(3):
                        for kw in nl.affine_range(3):
                            for c_tile in nl.affine_range(channel_tiles):
                                c_start = c_tile * CHANNEL_TILE
                                c_size = min(CHANNEL_TILE, in_channels - c_start)
                                weight_tiles.append(
                                    nl.load(
                                        weight_packed[
                                            kt,
                                            kh,
                                            kw,
                                            c_start : c_start + c_size,
                                            oc_start : oc_start + oc_size,
                                        ],
                                        dtype=output.dtype,
                                    ))

                for w_tile in nl.affine_range(width_tiles):
                    w_start = w_tile * WIDTH_TILE
                    w_size = min(WIDTH_TILE, width - w_start)

                    for h_block in nl.affine_range(height_blocks):
                        h_start = h_block * height_block_size
                        h_size = min(height_block_size, height - h_start)
                        output_tile_size = h_size * w_size
                        acc = nl.zeros(
                            (oc_size, output_tile_size),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )

                        for kt in nl.affine_range(3):
                            src_t = t_out + kt - CACHE_T
                            use_current = src_t >= 0
                            use_cache = (src_t < 0) & (src_t >= -cache_time)

                            if use_current | use_cache:
                                for kh in nl.affine_range(3):
                                    h_in = h_start + kh
                                    for kw in nl.affine_range(3):
                                        for c_tile in nl.affine_range(channel_tiles):
                                            c_start = c_tile * CHANNEL_TILE
                                            c_size = min(
                                                CHANNEL_TILE,
                                                in_channels - c_start,
                                            )
                                            weight_idx = (
                                                ((kt * 3 + kh) * 3 + kw)
                                                * channel_tiles
                                                + c_tile
                                            )

                                            if use_current:
                                                input_block = nl.load(
                                                    x[
                                                        b,
                                                        c_start : c_start + c_size,
                                                        src_t,
                                                        h_in : h_in + h_size,
                                                        w_start
                                                        + kw : w_start
                                                        + kw
                                                        + w_size,
                                                    ],
                                                    dtype=output.dtype,
                                                )
                                            else:
                                                cache_src_t = cache_time + src_t
                                                input_block = nl.load(
                                                    cache_x[
                                                        b,
                                                        c_start : c_start + c_size,
                                                        cache_src_t,
                                                        h_in : h_in + h_size,
                                                        w_start
                                                        + kw : w_start
                                                        + kw
                                                        + w_size,
                                                    ],
                                                    dtype=output.dtype,
                                                )
                                            input_tile = input_block.reshape(
                                                (c_size, output_tile_size))

                                            nisa.nc_matmul(
                                                dst=acc[:, :],
                                                stationary=weight_tiles[weight_idx],
                                                moving=input_tile,
                                                accumulate=True,
                                            )

                        out_tile = nl.ndarray(
                            (oc_size, output_tile_size),
                            dtype=output.dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_scalar(
                            dst=out_tile,
                            data=acc,
                            op0=nl.add,
                            operand0=bias_tile,
                        )
                        nl.store(
                            output[
                                b,
                                oc_start : oc_start + oc_size,
                                t_out,
                                h_start : h_start + h_size,
                                w_start : w_start + w_size,
                            ],
                            value=out_tile.reshape((oc_size, h_size, w_size)),
                        )

        return output

    @nki_op("wan::causal_conv3d_ks3s1_cached", mutates_args={})
    def _causal_conv3d_ks3s1_cached_call(
        x: torch.Tensor,
        cache_x: torch.Tensor,
        weight_packed: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        grid = (min(2, max(1, int(get_logical_neuron_cores()))),)
        return wrap_nki(_causal_conv3d_ks3s1_cached_kernel)[grid](
            x,
            cache_x,
            weight_packed,
            bias,
        )

    @nki.jit
    def _causal_conv3d_head_256x12_cached_kernel(
        x,
        cache_x,
        weight_packed,
        bias,
    ):
        batch, in_channels, time, height_padded, width_padded = x.shape
        _, _, cache_time, _, _ = cache_x.shape
        height = height_padded - 2
        width = width_padded - 2

        output = nl.ndarray(
            (batch, HEAD_OUT_CHANNELS, time, height, width),
            dtype=x.dtype,
            buffer=nl.shared_hbm,
        )

        channel_tiles = in_channels // CHANNEL_TILE
        width_tiles = (width + HEAD_WIDTH_TILE - 1) // HEAD_WIDTH_TILE
        height_blocks = (height + HEAD_HEIGHT_BLOCK - 1) // HEAD_HEIGHT_BLOCK
        work_tiles = batch * time * height_blocks * width_tiles
        program_count = nl.num_programs(0)
        program_idx = nl.program_id(0)

        bias_tile = nl.ndarray(
            (HEAD_OUT_CHANNEL_PAD, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=bias_tile[:, 0], src=bias[:HEAD_OUT_CHANNEL_PAD])

        weight_tiles = []
        for kt in range(3):
            for kh in range(3):
                for kw in range(3):
                    for c_tile in nl.affine_range(channel_tiles):
                        c_start = c_tile * CHANNEL_TILE
                        weight_tiles.append(
                            nl.load(
                                weight_packed[
                                    kt,
                                    kh,
                                    kw,
                                    c_start : c_start + CHANNEL_TILE,
                                    :HEAD_OUT_CHANNEL_PAD,
                                ],
                                dtype=output.dtype,
                            ))

        for linear_idx in nl.affine_range(program_idx, work_tiles, program_count):
            w_tile = linear_idx % width_tiles
            work_idx = linear_idx // width_tiles
            h_block = work_idx % height_blocks
            work_idx = work_idx // height_blocks
            t_out = work_idx % time
            b = work_idx // time

            w_start = w_tile * HEAD_WIDTH_TILE
            w_size = min(HEAD_WIDTH_TILE, width - w_start)
            h_start = h_block * HEAD_HEIGHT_BLOCK
            h_size = min(HEAD_HEIGHT_BLOCK, height - h_start)
            output_tile_size = h_size * w_size

            acc = nl.zeros(
                (HEAD_OUT_CHANNEL_PAD, output_tile_size),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for kt in range(3):
                src_t = t_out + kt - CACHE_T
                use_current = src_t >= 0
                use_cache = (src_t < 0) & (src_t >= -cache_time)

                if use_current | use_cache:
                    for kh in range(3):
                        h_in = h_start + kh
                        for kw in range(3):
                            for c_tile in nl.affine_range(channel_tiles):
                                c_start = c_tile * CHANNEL_TILE
                                weight_idx = (
                                    ((kt * 3 + kh) * 3 + kw)
                                    * channel_tiles
                                    + c_tile
                                )

                                if use_current:
                                    input_block = nl.load(
                                        x[
                                            b,
                                            c_start : c_start + CHANNEL_TILE,
                                            src_t,
                                            h_in : h_in + h_size,
                                            w_start + kw : w_start + kw + w_size,
                                        ],
                                        dtype=output.dtype,
                                    )
                                else:
                                    cache_src_t = cache_time + src_t
                                    input_block = nl.load(
                                        cache_x[
                                            b,
                                            c_start : c_start + CHANNEL_TILE,
                                            cache_src_t,
                                            h_in : h_in + h_size,
                                            w_start + kw : w_start + kw + w_size,
                                        ],
                                        dtype=output.dtype,
                                    )

                                nisa.nc_matmul(
                                    dst=acc[:, :],
                                    stationary=weight_tiles[weight_idx],
                                    moving=input_block.reshape(
                                        (CHANNEL_TILE, output_tile_size)),
                                    accumulate=True,
                                )

            out_tile = nl.ndarray(
                (HEAD_OUT_CHANNEL_PAD, output_tile_size),
                dtype=output.dtype,
                buffer=nl.sbuf,
            )
            nisa.tensor_scalar(
                dst=out_tile,
                data=acc,
                op0=nl.add,
                operand0=bias_tile,
            )
            nl.store(
                output[
                    b,
                    :,
                    t_out,
                    h_start : h_start + h_size,
                    w_start : w_start + w_size,
                ],
                value=out_tile[:HEAD_OUT_CHANNELS, :].reshape(
                    (HEAD_OUT_CHANNELS, h_size, w_size)),
            )

        return output

    @nki_op("wan::causal_conv3d_head_256x12_cached", mutates_args={})
    def _causal_conv3d_head_256x12_cached_call(
        x: torch.Tensor,
        cache_x: torch.Tensor,
        weight_packed: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        grid = (min(2, max(1, int(get_logical_neuron_cores()))),)
        return wrap_nki(_causal_conv3d_head_256x12_cached_kernel)[grid](
            x,
            cache_x,
            weight_packed,
            bias,
        )
else:
    _causal_conv3d_ks3s1_cached_call = None
    _causal_conv3d_head_256x12_cached_call = None


def causal_conv3d_ks3s1_cached(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    weight_packed: torch.Tensor,
    bias: torch.Tensor,
):
    """Run the cached 3x3x3 stride-1 causal Conv3d NKI specialization.

    ``x`` and ``cache_x`` must already be padded in H/W by one pixel on each
    side, ``cache_x`` must contain exactly two temporal frames, and
    ``weight_packed`` must be laid out as [KT, KH, KW, C, OC].
    """

    if not nki_available():
        raise RuntimeError(f"NKI causal Conv3d is unavailable: {_NKI_IMPORT_ERROR!r}")

    return _causal_conv3d_ks3s1_cached_call(x, cache_x, weight_packed, bias)


def causal_conv3d_head_256x12_cached(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    weight_packed: torch.Tensor,
    bias: torch.Tensor,
):
    """Run the padded-output-channel head Conv3d NKI specialization.

    ``weight_packed`` and ``bias`` must already have output channels padded to
    ``HEAD_OUT_CHANNEL_PAD``. The returned tensor keeps the real 12 channels.
    """

    if not nki_available():
        raise RuntimeError(f"NKI causal Conv3d is unavailable: {_NKI_IMPORT_ERROR!r}")

    return _causal_conv3d_head_256x12_cached_call(
        x,
        cache_x,
        weight_packed,
        bias,
    )
