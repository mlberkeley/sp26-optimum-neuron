"""Cheap correctness test for `wan.vae.tiled_decode`.

The tiled decode wrapper splits the latent input into a 2x2 spatial grid
with halo overlap, decodes each tile independently on Neuron, and blends
them on CPU. The full standalone Neuron decode is known to fit at f17
(the smaller probe shape) but OOM at f81 — so f17 is the only shape
where we can compare tiled output against an on-device reference.

Strategy:
  1. Run untiled `vae.model.decode(z)` on Neuron at f17 -> ref output.
  2. Run `tiled_decode(vae.model, z)` on Neuron at f17 -> tiled output.
  3. Assert torch.allclose with bf16-grade tolerances.

If the tile mechanics, halo accounting, or blend weights are wrong, the
seam region of the tiled output will diverge from the reference. The
tolerance is set so true-bf16 noise passes but a real seam artifact
(typical 10-20% pixel jump at boundaries) doesn't.

This test is the gating step before running the V2 e2e.

Required env to run on hardware:
    NEURON_KERNEL_TEST=1            (matches the other on-device tests)
    NEURON_RT_VISIBLE_CORES=<idx>   (recommended: pick a free LNC, e.g. 8,
                                     so we don't trample the rolling
                                     generator's LNC 0..7)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# Allow CPU-only environments to import this test file (collection only).
_DEVICE_OPT_IN = os.environ.get("NEURON_KERNEL_TEST", "0") in ("1", "true", "yes", "on")


def _maybe_skip_no_neuron():
    if not _DEVICE_OPT_IN:
        pytest.skip(
            "set NEURON_KERNEL_TEST=1 to run tiled-VAE on-device test "
            "(off-device path doesn't exercise the bottleneck)."
        )
    try:
        import torch_neuronx  # noqa: F401
    except ImportError:
        pytest.skip("torch_neuronx not importable; can't reach the neuron device")


def _build_vae():
    """Load the production VAE checkpoint on CPU then move to Neuron."""
    from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
    ckpt_dir = _PKG_ROOT / "ckpts" / "Wan2.2-TI2V-5B"
    vae_path = ckpt_dir / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        pytest.skip(f"missing VAE checkpoint at {vae_path}")
    vae = Wan2_2_VAE(vae_pth=str(vae_path), device="cpu")
    # Migrate model + scale list-of-tensors to Neuron, mirroring the
    # working pattern from probe_vae_neuron.py.
    vae.model.to("neuron")
    if hasattr(vae, "scale") and isinstance(vae.scale, list):
        vae.scale = [t.to("neuron") if torch.is_tensor(t) else t
                     for t in vae.scale]
    return vae


# Tile-decode tolerances. The seam artifact shrinks with halo:
#   halo=2: max_abs=0.287
#   halo=4: max_abs=0.160
#   halo=6: max_abs=0.102
#   halo=8: max_abs=0.102
#   halo=10: max_abs=0.068
# Mean abs across the whole frame is <0.002 in all cases — i.e. ~99.99%
# of pixels match within bf16 noise; the only divergence is in the
# blend zone where each independent tile sees zero-padding at the cut
# instead of the actual neighboring latent values. The test runs at
# halo=10 (the production setting) with a tolerance of 0.10 — strict
# enough to catch any blend-mechanic bug (which would push max_abs >0.2)
# but loose enough to admit the residual seam noise from finite halo.
_ATOL = 0.10
_RTOL = 0.10
_PROD_HALO_LAT = 10


def test_tiled_decode_matches_reference_at_f17():
    """Tile mechanics check: the composited output of a 2x2 tile should
    agree with the full untiled decode at f17, where both fit on Neuron.
    """
    _maybe_skip_no_neuron()
    from wan.vae import tiled_decode  # noqa: E402

    vae = _build_vae()

    # f17 latent shape: [B=1, C=48, F_lat=5, H_lat=22, W_lat=40].
    # H_lat=22 is even; W_lat=40 is even; both pass the 2x2 split check.
    torch.manual_seed(42)
    z = torch.randn(1, 48, 5, 22, 40, dtype=torch.float32).to("neuron")

    # Reference: untiled decode on Neuron.
    ref = vae.model.decode(z, vae.scale)
    ref_cpu = ref.float().to("cpu")
    del ref

    # Tiled: same input, halo matches production.
    tiled = tiled_decode(vae.model, z, vae.scale, halo_lat=_PROD_HALO_LAT)
    # tiled is already CPU fp32 by construction.

    assert tiled.shape == ref_cpu.shape, (
        f"tiled shape={tuple(tiled.shape)} != ref shape={tuple(ref_cpu.shape)}"
    )
    max_abs = (tiled - ref_cpu).abs().max().item()
    rel_inf = ((tiled - ref_cpu).abs() / ref_cpu.abs().clamp(min=1e-3)).max().item()
    print(f"[tiled_decode f17] max_abs={max_abs:.4e} rel_inf={rel_inf:.4e}")
    assert torch.isfinite(tiled).all().item(), "tiled has non-finite values"
    assert torch.allclose(tiled, ref_cpu, atol=_ATOL, rtol=_RTOL), (
        f"tiled vs reference disagreement: max_abs={max_abs:.4e}"
    )


def test_tiled_decode_finite_at_f81():
    """f81 sanity check: the bottleneck shape should now decode without
    OOM via tiling, and the output should be finite. This does NOT compare
    to a reference (no on-device reference exists at f81 — that's the
    whole reason we're tiling).
    """
    _maybe_skip_no_neuron()
    from wan.vae import tiled_decode  # noqa: E402

    vae = _build_vae()

    # f81 latent shape: [B=1, C=48, F_lat=21, H_lat=88, W_lat=160].
    torch.manual_seed(42)
    z = torch.randn(1, 48, 21, 88, 160, dtype=torch.float32).to("neuron")

    tiled = tiled_decode(vae.model, z, vae.scale, halo_lat=_PROD_HALO_LAT)
    expected_shape = (1, 3, 81, 88 * 16, 160 * 16)
    assert tuple(tiled.shape) == expected_shape, (
        f"f81 tiled shape={tuple(tiled.shape)} != expected {expected_shape}"
    )
    assert torch.isfinite(tiled).all().item(), "f81 tiled has non-finite values"
    finite_max = tiled.abs().max().item()
    print(f"[tiled_decode f81] shape={tuple(tiled.shape)} max_abs={finite_max:.4f}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
