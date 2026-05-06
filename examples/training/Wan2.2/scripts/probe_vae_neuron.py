"""Probe whether the Wan2.2 VAE can run on the Neuron device.

Approach:
    1. Build the VAE module with the standard checkpoint on CPU.
    2. Move to torch.device('neuron'), then attempt a single decode call on a
       small synthetic latent shape that mirrors a 1-frame chunk.
    3. If the decode raises, capture the *first* RuntimeError / NotImplemented
       message — that pinpoints the unsupported op.
    4. Always emit a structured marker the chain can grep for:
         [PROBE:VAE_NEURON_OK]
         [PROBE:VAE_NEURON_BLOCKED op="<excerpt>"]
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_neuronx  # noqa: F401, E402  (registers the privateuse1 backend)


def _emit_blocked(reason: str) -> None:
    safe = reason.replace("\n", " | ")[:300]
    print(f'[PROBE:VAE_NEURON_BLOCKED] op="{safe}"')


def main() -> int:
    try:
        from wan.modules.vae2_2 import Wan2_2_VAE  # type: ignore
    except Exception as e:
        _emit_blocked(f"vae import failed: {type(e).__name__}: {e}")
        return 1

    ckpt_dir = ROOT / "ckpts" / "Wan2.2-TI2V-5B"
    vae_path = ckpt_dir / "Wan2.2_VAE.pth"
    if not vae_path.exists():
        _emit_blocked(f"vae checkpoint missing: {vae_path}")
        return 1

    print(f"[probe] loading VAE on CPU from {vae_path}")
    try:
        vae = Wan2_2_VAE(vae_pth=str(vae_path), device="cpu")
    except Exception as e:
        _emit_blocked(f"vae load on CPU failed: {type(e).__name__}: {e}")
        return 1

    # Minimal latent shape: [C=48 (TI2V-5B), F_lat=1, H_lat=8, W_lat=8].
    # The exact channel count is per-config; if mismatched, the call fails with
    # a comprehensible error and we still capture it.
    try:
        # Use a deliberately small spatial extent to keep the decode quick.
        z = torch.randn(48, 1, 8, 8, dtype=torch.float32)
    except Exception as e:
        _emit_blocked(f"latent build failed: {type(e).__name__}: {e}")
        return 1

    print("[probe] moving VAE -> neuron (model + scale tensors)")
    try:
        vae.model.to("neuron")
        # Wan2_2_VAE keeps self.scale = [mean, 1/std] tensors on the construction
        # device. The .decode() call uses these alongside the latent, so they
        # must move too — otherwise we get a device-mismatch before any real
        # VAE op is exercised.
        if hasattr(vae, "scale") and isinstance(vae.scale, list):
            vae.scale = [t.to("neuron") if torch.is_tensor(t) else t for t in vae.scale]
        # Bypass the wrapper's `with torch.amp.autocast('cpu', ...)` since CPU
        # autocast is meaningless on a Neuron tensor and may itself error.
    except Exception as e:
        _emit_blocked(f"vae move-to-neuron failed: {type(e).__name__}: {e}\n"
                      f"{traceback.format_exc(limit=4)}")
        return 1

    print("[probe] dispatching VAE.model.decode(z) on neuron")
    try:
        z_dev = z.to("neuron")
        # Call the inner model.decode directly to skip the CPU autocast guard
        # in the wrapper. Mirrors what wrapper.decode does, just without amp.
        out = vae.model.decode(z_dev.unsqueeze(0), vae.scale)
        out = out.float().clamp_(-1, 1).squeeze(0)
        out_cpu = out.to("cpu")
        print(f"[probe] decode succeeded; output shape={tuple(out_cpu.shape)}, "
              f"finite={torch.isfinite(out_cpu).all().item()}")
        print("[PROBE:VAE_NEURON_OK]")
        return 0
    except Exception as e:
        # Surface only the first ~6 traceback lines for the marker
        tb = traceback.format_exc(limit=6)
        _emit_blocked(f"{type(e).__name__}: {e}\n{tb}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
