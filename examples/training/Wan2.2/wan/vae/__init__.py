"""VAE acceleration helpers for Wan2.2 inference on Neuron.

Currently exports:
    tiled_decode: spatial 2x2 tiled wrapper around `Wan2_2_VAE.model.decode`,
        designed to keep f81 inputs within the per-LNC HBM budget on
        Trainium2.

The standalone VAE-on-Neuron probe at f81 fails on a single per-iter
intermediate of shape `[1, 512, 1, 704, 1280]` fp32 (~1.85 GB). Splitting
the latent spatially into 4 quadrants shrinks that intermediate to
`[1, 512, 1, ~370, ~660]` ≈ 470 MB per tile, which fits.
"""
from .tiled_decode import tiled_decode  # noqa: F401
