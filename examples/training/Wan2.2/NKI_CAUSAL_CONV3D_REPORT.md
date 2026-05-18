# Wan2.2 Causal Conv3d NKI Kernel Report

## Alternative Workstream 2: 3D Convolution Custom Kernel

### Git branch

`conv3d_adi_ghai_agi`

### Why It Is Slower

The custom NKI causal Conv3d path is slower because it does not match the execution strategy of the optimized Neuron Conv3d lowering. The full Wan profiler confirms this clearly: on the warmed full-pipeline run, VAE decode took `730.746 s` with NKI enabled versus `132.353 s` on the baseline path. The decoder causal conv region was the main regression: NKI spent `719.448 s` across only `90` NKI conv calls, while the baseline spent `123.528 s` across all `169` decoder Conv3d calls.

The main causes are:

- `torch_neuronx.wrap_nki` does not expose the SPMD grid needed for this workload. The grid probe showed `grid=(2,)` still reports `nl.num_programs(0) == 1`, and `grid=(4,)` / `grid=(8,)` are rejected. As a result, the NKI kernel cannot naturally shard the eight output-channel tiles for `1024 -> 1024` residual convs across programs the way a production compiler/library lowering can.
- The residual `1024 -> 1024 @ 44x80` convs are dominated by many input/output channel tile products. Even after hoisting weight loads and H-blocking, the NKI kernel still walks a long serial loop nest over temporal, spatial, input-channel, output-channel, and height/width tiles.
- The kernel emits many small Tensor Engine matmuls with a long accumulation chain. The accumulation probe showed `accumulate=True` is correct, so correctness is not the issue; the issue is scheduling overhead and poor effective throughput compared with the compiler's Conv3d implementation.
- Input-window reuse is still weak. Neighboring output rows share two of three input rows, but the current kernel does not stage and reuse those windows as effectively as a tuned convolution lowering.
- Shape-specific experiments were not enough. The padded head kernel for `256 -> 12` improved the isolated head microbenchmark from `6.675 ms` to `5.450 ms`, but it still lost to baseline at `3.446 ms`. The small-channel entry specialization for `48 -> 1024` added packing overhead and did not materially improve over the generic NKI path.
- Full-pipeline NKI compilation/cache generation is very expensive. The first NKI run generated multi-GB NKI artifacts and took `4257.607 s` total, with `2077.309 s` in VAE decode. Even after warming, the steady-state NKI decode remained much slower than baseline.
- The baseline is already highly optimized. The Neuron Conv3d path is not naive PyTorch; it is an optimized compiler/library path with better tiling, parallelism, and scheduling than this first custom implicit-GEMM NKI kernel.

### Conclusion

Do not use the current NKI causal Conv3d path for Wan2.2 inference. It is correct, but the warmed full profiler shows it makes generation about `4.2x` slower and VAE decode about `5.5x` slower than the baseline. The current implementation should remain opt-in and disabled by default.

The practical next step is not to keep optimizing this all-decoder-conv NKI path through `wrap_nki`. A future attempt would need either a lower-level launch path with real program-grid parallelism, or a much narrower shape-specific kernel that proves a win on one isolated layer family before being enabled in the full decoder.

## Scope

This pass targeted the cached decoder-time `3x3x3`, stride-1 causal Conv3d calls in Wan2.2 VAE decode:

- `ResidualBlock` residual convs
- `Decoder3d.conv1`
- decoder head conv

The implementation is opt-in through:

```bash
WAN_ENABLE_NKI_CAUSAL_CONV3D=1 python3 scripts/run.py
```

It is disabled by default because the current kernel is correct, but slower than the existing Neuron Conv3d path on the measured decoder shapes.

## Approach

The baseline `CausalConv3d` path concatenates the two-frame temporal cache with the current frame, pads causally in time and spatially in H/W, then calls PyTorch/Neuron `Conv3d`.

The NKI path specializes only the inference cached case:

- Requires `cache_x` with exactly two temporal frames.
- Requires `kernel_size=(3,3,3)`, `stride=(1,1,1)`, `dilation=(1,1,1)`, `groups=1`, bias present, and padding equivalent to `(1,1,1,1,2,0)`.
- Spatially pads `x` and `cache_x` in Python before calling NKI.
- Packs weights once per weight version from PyTorch layout `[OC, C, KT, KH, KW]` into NKI-friendly `[KT, KH, KW, C, OC]`.
- Computes each output row and output-channel tile using Tensor Engine matmuls over channel tiles and the causal temporal window.
- Uses the supported `torch_neuronx.wrap_nki` grid model with up to two logical NeuronCores, then strides over flattened work tiles.

Follow-up optimization pass:

- Changed the work unit from `(batch, t_out, h_out, oc_tile)` to `(batch, t_out, oc_tile)`.
- Hoisted bias loads out of the H loop.
- Hoisted generic weight tiles out of the H loop.
- Added H-blocking: for low-width shapes such as `W=80`, up to six output rows are flattened into the Tensor Engine moving dimension (`H_block * W <= 512`).
- Added and tested a small-input-channel specialization for `32 <= C <= 64` that packs the 3x3 spatial footprint into the contraction dimension in chunks of 14 channels. It did not materially improve `48 -> 1024`, so it is kept in the source but disabled by `ENABLE_SPATIAL_PACK_ENTRY = False`.
- Tried prepacking those spatial weights into HBM outside the NKI kernel, but that version failed NKI compilation for the real `48 -> 1024` entry shape with an SBUF memory-location error, so it was backed out.
- Added a dedicated `256 -> 12` decoder-head kernel that pads the matmul output-channel tile to 16 and stores only the true 12 channels.
- Added focused probe/benchmark scripts:
  - `scripts/nki_conv3d_probes.py`
  - `scripts/profile_causal_conv3d.py`

## Tested Shapes

These timings are second-run measurements from `scripts/profile_causal_conv3d.py`: the first call warms/compiles, and the second call is timed.

| Shape | Why Tested | Baseline Second Run | NKI Second Run | NKI / Baseline |
|---|---|---:|---:|---:|
| `8 -> 8 @ 8x8` | tiny sanity | `1.408 ms` | `1.424 ms` | `0.99x` |
| `48 -> 1024 @ 44x80` | `Decoder3d.conv1`-like | `1.406 ms` | `2.543 ms` | `0.55x` |
| `256 -> 12 @ 176x320` | decoder head-like, padded head kernel | `3.446 ms` | `5.450 ms` | `0.63x` |
| `1024 -> 1024 @ 44x80` | low-res residual dominant case | `3.023 ms` | `39.506 ms` | `0.08x` |

Correctness was bf16-close. The latest max absolute differences were `0.003906` for the head case and `0.007812` for entry/residual.

The padded head kernel improved the NKI head timing from the previous `6.675 ms` to `5.450 ms`, but it is still slower than the existing Neuron Conv3d path.

## Full Wan Profiler Runs

The full Wan profiling flow was also run using:

```bash
source ~/trn_workspace/native_venv/bin/activate
cd ~/sp26-optimum-neuron/examples/training/Wan2.2
WAN_ENABLE_NKI_CAUSAL_CONV3D=1 python3 scripts/run.py
WAN_ENABLE_NKI_CAUSAL_CONV3D=0 python3 scripts/run.py
```

Configuration for these runs:

- `DEVICE = "neuron"` in `scripts/config.py`
- `DEVICE = "neuron"` in `profiling/profiler.py`
- `SIZE = "1280*704"`
- `FRAME_NUM = 17`
- `SAMPLE_STEPS = 8`
- Profiling enabled through the in-repo `profiling` package.

Profile output directories:

- NKI warmup/compile run: `outputs/profiles/ti2v_run_20260518_063602`
- NKI warmed second run: `outputs/profiles/ti2v_run_20260518_074729`
- Baseline warmed run: `outputs/profiles/ti2v_run_20260518_080212`

### Full-Run Results

The NKI warmup run includes large NKI compile/cache generation and is not a fair steady-state timing. It is still useful because it showed the full NKI path generating multi-GB NKI artifacts.

| Run | Total Run | Build Pipeline | Generate | VAE Decode | Decode Causal Conv Total |
|---|---:|---:|---:|---:|---:|
| NKI warmup | `4257.607 s` | `1880.713 s` | `2355.394 s` | `2077.309 s` | `1797.017 s` |
| NKI warmed | `843.491 s` | `55.988 s` | `785.840 s` | `730.746 s` | `721.891 s` |
| Baseline warmed | `244.371 s` | `55.346 s` | `187.390 s` | `132.353 s` | `123.547 s` |

### Decode Conv Breakdown

| Run | Decode Conv Calls | NKI Conv Time | Torch Conv Time | Conv Avg |
|---|---:|---:|---:|---:|
| NKI warmup | `169` decode conv calls | `1420.441 s` across `90` calls | `376.547 s` across `79` calls | `10.633 s` |
| NKI warmed | `169` decode conv calls | `719.448 s` across `90` calls | `2.423 s` across `79` calls | `4.272 s` |
| Baseline warmed | `169` decode conv calls | `0.000 s` | `123.528 s` across `169` calls | `0.731 s` |

The warmed full-run result is worse than the microbenchmark suggested:

- Generate: NKI warmed `785.840 s` vs baseline `187.390 s` (`0.24x`, about `4.2x` slower).
- VAE decode: NKI warmed `730.746 s` vs baseline `132.353 s` (`0.18x`, about `5.5x` slower).
- Decode causal conv total: NKI warmed `721.891 s` vs baseline `123.547 s` (`0.17x`, about `5.8x` slower).

Conclusion from full profiling: do not use the current NKI path for Wan2.2 decode. The full profiler confirms that the all-decoder-conv NKI attempt is substantially slower than the baseline path even after warming compilation/cache state.

## Diagnostic Probes

### Accumulation

`scripts/nki_conv3d_probes.py` compares repeated `nisa.nc_matmul` calls with and without explicit `accumulate=True` using ones inputs. One matmul should produce `128.0`; `K` accumulated matmuls should produce `K * 128.0`.

| K | Explicit Accumulate Mean | Explicit / K | Default Mean |
|---:|---:|---:|---:|
| `1` | `128.0` | `128.0` | `128.0` |
| `8` | `1024.0` | `128.0` | `1024.0` |
| `27` | `3456.0` | `128.0` | `3456.0` |
| `64` | `8192.0` | `128.0` | `8192.0` |
| `216` | `27648.0` | `128.0` | `27648.0` |

Conclusion: `accumulate=True` is not being ignored. In this NKI version, repeated matmuls into the same PSUM destination also accumulate even without the explicit flag, so the earlier suspicion that the kernel was resetting PSUM every matmul is not supported by this probe.

The probe timings were noisy and launch/compiler dominated, so they are useful for correctness but not for modeling the real conv slope.

### Grid Behavior

The grid probe launched through `torch_neuronx.wrap_nki` reported:

| Launch Grid | Result |
|---|---|
| `(1,)` | `nl.num_programs(0) == 1` |
| `(2,)` | `nl.num_programs(0) == 1` |
| `(4,)` | error: NKI only supports LNC 1 or 2 |
| `(8,)` | error: NKI only supports LNC 1 or 2 |

Conclusion: in this eager `wrap_nki` path, the grid is effectively an LNC setting, not a general SPMD program grid. The kernel cannot create eight independent output-channel programs with this wrapper. This is worse than the earlier assumption that `grid=(2,)` would expose two programs through `program_id(0)`.

### Profiling / SBUF Report

Attempted `neuron-profile inspect` on the isolated residual benchmark:

- Baseline inspect started and produced an `ntrace.pb`, but did not complete in a reasonable time and had to be terminated.
- `neuron-profile view --ingest-only --disable-ui` could not run because `influx` is not installed in this environment.
- NKI inspect on the residual case failed with `signal: segmentation fault`.
- The generated NKI JSON allocator dump for the residual kernel did not show obvious internal DRAM spill entries. The visible memory-location summary was external DRAM for inputs/weights/output plus internal `SB` and `PSUM` allocations. That does not prove the hoist is optimal, but it also does not confirm an explicit spill.

Because profile view is blocked here, I do not have reliable TE busy %, PSUM stall, or DMA queue-depth counters yet.

## Current NKI Kernel

The live implementation is in:

```text
examples/training/Wan2.2/wan/modules/causal_conv3d_nki.py
```

Current tile constants:

- `CHANNEL_TILE = 128`
- `OUT_CHANNEL_TILE = 128`
- `WIDTH_TILE = 512`
- `HEIGHT_BLOCK = 6`
- `ENABLE_SPATIAL_PACK_ENTRY = False`
- `SPATIAL_PACK_CHANNEL_TILE = 14`
- `HEAD_OUT_CHANNEL_PAD = 16`
- `HEAD_WIDTH_TILE = 160`
- `HEAD_HEIGHT_BLOCK = 3`

The current source has three execution paths:

- Generic path: uses 128-channel tiles, hoists weights and bias above the H loop, and runs H-blocked matmuls where `H_block * W <= 512`.
- `C=256, OC=12`: uses the padded head specialization with `OC_PAD=16`, `W_TILE=160`, and `H_BLOCK=3`.
- Disabled experimental entry path: `32 <= C <= 64` spatially packs the 3x3 footprint into 14-channel chunks. It is left off because the measured entry case was effectively tied with the generic path.

The wrapper still launches with the supported logical-core grid:

```python
grid = (min(2, max(1, int(get_logical_neuron_cores()))),)
return wrap_nki(_causal_conv3d_ks3s1_cached_kernel)[grid](
    x,
    cache_x,
    weight_packed,
    bias,
)
```

## First-Pass NKI Kernel (Superseded)

The same kernel is used for all tested shapes. Shape-specific behavior comes from the runtime tensor dimensions and the fixed tile sizes:

- `CHANNEL_TILE = 128`
- `OUT_CHANNEL_TILE = 128`
- `WIDTH_TILE = 512`

```python
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
    work_tiles = batch * time * height * out_channel_tiles
    program_count = nl.num_programs(0)
    program_idx = nl.program_id(0)

    for linear_idx in nl.affine_range(program_idx, work_tiles, program_count):
        oc_tile = linear_idx % out_channel_tiles
        work_idx = linear_idx // out_channel_tiles
        h_out = work_idx % height
        work_idx = work_idx // height
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

        for w_tile in nl.affine_range(width_tiles):
            w_start = w_tile * WIDTH_TILE
            w_size = min(WIDTH_TILE, width - w_start)
            acc = nl.zeros(
                (oc_size, w_size),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for kt in nl.affine_range(3):
                src_t = t_out + kt - CACHE_T
                use_current = src_t >= 0
                use_cache = (src_t < 0) & (src_t >= -cache_time)

                if use_current | use_cache:
                    for kh in nl.affine_range(3):
                        h_in = h_out + kh
                        for kw in nl.affine_range(3):
                            for c_tile in nl.affine_range(channel_tiles):
                                c_start = c_tile * CHANNEL_TILE
                                c_size = min(CHANNEL_TILE, in_channels - c_start)

                                weight_tile = nl.load(
                                    weight_packed[
                                        kt,
                                        kh,
                                        kw,
                                        c_start : c_start + c_size,
                                        oc_start : oc_start + oc_size,
                                    ],
                                    dtype=output.dtype,
                                )

                                if use_current:
                                    input_tile = nl.load(
                                        x[
                                            b,
                                            c_start : c_start + c_size,
                                            src_t,
                                            h_in,
                                            w_start + kw : w_start + kw + w_size,
                                        ],
                                        dtype=output.dtype,
                                    )
                                else:
                                    cache_src_t = cache_time + src_t
                                    input_tile = nl.load(
                                        cache_x[
                                            b,
                                            c_start : c_start + c_size,
                                            cache_src_t,
                                            h_in,
                                            w_start + kw : w_start + kw + w_size,
                                        ],
                                        dtype=output.dtype,
                                    )

                                nisa.nc_matmul(
                                    dst=acc[:, :],
                                    stationary=weight_tile,
                                    moving=input_tile,
                                )

            out_tile = nl.ndarray(
                (oc_size, w_size),
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
                    h_out,
                    w_start : w_start + w_size,
                ],
                value=out_tile,
            )

    return output
```

The wrapper launches with the supported logical-core grid:

```python
grid = (min(2, max(1, int(get_logical_neuron_cores()))),)
return wrap_nki(_causal_conv3d_ks3s1_cached_kernel)[grid](
    x,
    cache_x,
    weight_packed,
    bias,
)
```

## Why It Is Not Fast Enough Yet

The current kernel is a straightforward implicit-GEMM implementation, but it does not match the efficiency of the existing Neuron Conv3d lowering.

Main issues:

- The eager `wrap_nki` path does not expose a useful SPMD grid for this custom op. `grid=(2,)` reports `nl.num_programs(0) == 1`, while `grid=(4,)` and above are rejected. Work is therefore mostly serialized inside one NKI program from the kernel's point of view.
- The residual-dominant `1024 -> 1024 @ 44x80` shape requires many output-channel and input-channel tiles. Weight loads are now hoisted out of the H loop, but runtime barely moved, which suggests the dominant cost is Tensor Engine work scheduling/throughput rather than only redundant HBM weight traffic.
- Full Wan profiling makes this more severe: the warmed NKI decode spent `719.448 s` in `90` NKI conv calls, while the warmed baseline spent `123.528 s` across all `169` decoder Torch conv calls.
- There is still little reuse across neighboring output rows. A faster conv kernel would stage/reuse input windows or use a more aggressive im2col/blocking scheme.
- H-blocking reduced the number of matmul calls for low-width shapes, but not enough to overcome the baseline Conv3d lowering. The NKI path is still far slower for `1024 -> 1024`, which points to poor effective TE utilization and/or dependency stalls around the long accumulation chain.
- The generic width tile is fixed at `512`, which works functionally, but the implementation does not tune all shapes well. The dedicated head kernel improves OC utilization by padding `12 -> 16`, but wrapper and scheduling overhead still dominate.
- The small-channel spatial-pack path for `48 -> 1024` reduced matmul count, but added SBUF packing overhead and did not materially improve over the simpler generic path.
- The baseline Neuron Conv3d path is already highly optimized. This first NKI kernel needs to beat a good compiler/library implementation, not naive PyTorch.

## Next Optimization Directions

The most promising next steps are:

- Specialize separate kernels per shape family:
  - high-channel residual: `1024 -> 1024`, `512 -> 512`, `256 -> 256`
  - entry conv: `48 -> 1024`
  - head conv: `256 -> 12`
- For high-channel residuals, block more output rows or width chunks per program to reuse weight tiles.
- Prepack weights persistently outside the measured path and explore layouts that reduce per-row reloads.
- Try a lower-level launch path than `torch_neuronx.wrap_nki` if possible, because the current wrapper does not expose the grid shape needed to shard output-channel tiles.
- For the head conv, try a vector/direct-convolution kernel or pack multiple independent spatial rows into a larger effective output-channel tile. Padding `OC=12` to `16` helped, but not enough.
- For entry conv, replace the 14-channel spatial-pack path with a simpler `48 -> 64/128` channel-pad strategy and measure it.
- Re-run profiling once `neuron-profile view` is usable in the environment, or export a NEFF/session pair that can be viewed outside this instance.
