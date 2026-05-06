"""Full-scale VAE-on-Neuron probe.

The minimal probe (probe_vae_neuron.py) showed VAE *can* run on Neuron with a
1-frame 8x8 latent. This script ramps to TI2V-5B production shapes to confirm
the full decode path doesn't OOM or hit unsupported ops at scale.

Production decode shape (per chunk_frame_num=81, 704x1280 video):
    latent: [48, 21, 88, 160]    # F_lat = (81-1)/4 + 1 = 21
    output: [3, 81, 704, 1280]

We probe at three sizes:
    1. tiny:   [48, 1, 8, 8]      — sanity (already passed)
    2. f17 :   [48, 5, 22, 40]    — 17-frame, 176x320 video (~17MB latent)
    3. f81 :   [48, 21, 88, 160]  — full production size (~280MB latent)

Stops at first failure, reports which size succeeded. Emits structured
markers for the chain monitor.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_neuronx  # noqa: F401, E402


SHAPES = [
    ("tiny", (48, 1, 8, 8)),
    ("f17",  (48, 5, 22, 40)),
    ("f81",  (48, 21, 88, 160)),
]


def main() -> int:
    from wan.modules.vae2_2 import Wan2_2_VAE
    ckpt_dir = ROOT / "ckpts" / "Wan2.2-TI2V-5B"
    vae_path = ckpt_dir / "Wan2.2_VAE.pth"

    print(f"[probe-full] loading VAE on CPU")
    vae = Wan2_2_VAE(vae_pth=str(vae_path), device="cpu")

    print("[probe-full] migrating model + scale to neuron")
    vae.model.to("neuron")
    if hasattr(vae, "scale") and isinstance(vae.scale, list):
        vae.scale = [t.to("neuron") if torch.is_tensor(t) else t for t in vae.scale]

    last_ok_size = None
    for label, shape in SHAPES:
        print(f"\n[probe-full] {label}: shape={shape}")
        try:
            torch.manual_seed(0)
            z = torch.randn(*shape, dtype=torch.float32).to("neuron")
            t0 = time.perf_counter()
            out = vae.model.decode(z.unsqueeze(0), vae.scale)
            out = out.float().clamp_(-1, 1).squeeze(0)
            out_cpu = out.to("cpu")
            elapsed = time.perf_counter() - t0
            print(f"[probe-full] {label} OK in {elapsed:.1f}s "
                  f"output={tuple(out_cpu.shape)} finite={torch.isfinite(out_cpu).all().item()}")
            last_ok_size = label
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            print(f"[probe-full] {label} FAILED: {type(e).__name__}: {e}")
            print(tb[:2000])
            print(f"[PROBE:VAE_NEURON_LIMIT] last_ok={last_ok_size or 'none'} failed_at={label}")
            return 1

    print(f"\n[PROBE:VAE_NEURON_FULL_OK] all 3 sizes decoded; largest={SHAPES[-1][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
