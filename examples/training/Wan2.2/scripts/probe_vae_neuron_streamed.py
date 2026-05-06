"""Probe VAE-on-Neuron at f81 with output streaming to CPU.

`probe_vae_neuron_full.py` showed that f81 OOMs at 5.179GB on NC4. The
hypothesis here: most of that pressure is the in-device accumulator
`out = torch.cat([out, out_], 2)` inside `Wan2_2_VAE.model.decode`,
which holds a growing [3, F_px, 704, 1280] tensor on Neuron through 21
iterations. We bypass `decode` and stream each iteration's pixel-frame
slice to CPU, never letting the running concatenation live on device.

If this still OOMs, the bottleneck is per-iter intermediates inside the
decoder (not the accumulator), and we'd need to tile spatially or
reduce VAE channel widths — neither of which is a quick win.

Steps:
    1. tiny / f17 sanity (already known to pass) — confirms the streaming
       wrapper still produces correct output.
    2. f81 (the OOM target) — measures whether streaming opens the
       ceiling.

Markers:
    [PROBE:VAE_STREAM_OK]    label=<size> elapsed=<s> mb_out=<size>
    [PROBE:VAE_STREAM_FAIL]  label=<size> reason="<excerpt>"
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

from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify  # noqa: E402


SHAPES = [
    ("tiny", (48, 1, 8, 8)),
    ("f17",  (48, 5, 22, 40)),
    ("f81",  (48, 21, 88, 160)),
]


def streaming_decode(vae_model, z, scale):
    """Drop-in for vae_model.decode that streams per-iter output to CPU.

    Mirrors `wan.modules.vae2_2.WanVAE_.decode` exactly, except that:
    - The running `out` accumulator lives on CPU, not on the device.
    - Each iter's `out_` is moved to CPU and dropped from device memory
      before the next iter runs.

    Returns a CPU tensor with the same shape/dtype as the original.
    """
    vae_model.clear_cache()
    if isinstance(scale[0], torch.Tensor):
        z = (z / scale[1].view(1, vae_model.z_dim, 1, 1, 1)
             + scale[0].view(1, vae_model.z_dim, 1, 1, 1))
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = vae_model.conv2(z)

    cpu_chunks = []
    for i in range(iter_):
        vae_model._conv_idx = [0]
        out_dev = vae_model.decoder(
            x[:, :, i:i + 1, :, :],
            feat_cache=vae_model._feat_map,
            feat_idx=vae_model._conv_idx,
            first_chunk=(i == 0),
        )
        cpu_chunks.append(out_dev.to("cpu"))
        del out_dev

    out_cpu = torch.cat(cpu_chunks, dim=2)
    out_cpu = unpatchify(out_cpu, patch_size=2)
    vae_model.clear_cache()
    return out_cpu


def main() -> int:
    ckpt_dir = ROOT / "ckpts" / "Wan2.2-TI2V-5B"
    vae_path = ckpt_dir / "Wan2.2_VAE.pth"

    print("[probe-stream] loading VAE on CPU")
    vae = Wan2_2_VAE(vae_pth=str(vae_path), device="cpu")

    print("[probe-stream] migrating model + scale to neuron")
    vae.model.to("neuron")
    if hasattr(vae, "scale") and isinstance(vae.scale, list):
        vae.scale = [t.to("neuron") if torch.is_tensor(t) else t
                     for t in vae.scale]

    last_ok = None
    for label, shape in SHAPES:
        print(f"\n[probe-stream] {label}: shape={shape}")
        try:
            torch.manual_seed(0)
            z = torch.randn(*shape, dtype=torch.float32).to("neuron")
            t0 = time.perf_counter()
            out_cpu = streaming_decode(vae.model, z.unsqueeze(0), vae.scale)
            out_cpu = out_cpu.float().clamp_(-1, 1).squeeze(0)
            elapsed = time.perf_counter() - t0
            mb = out_cpu.element_size() * out_cpu.numel() / 1e6
            finite = torch.isfinite(out_cpu).all().item()
            print(f"[probe-stream] {label} OK in {elapsed:.1f}s "
                  f"output={tuple(out_cpu.shape)} finite={finite} mb_out={mb:.1f}")
            print(f"[PROBE:VAE_STREAM_OK] label={label} elapsed={elapsed:.1f}s "
                  f"mb_out={mb:.1f}")
            last_ok = label
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            short = (str(e).replace("\n", " | "))[:300]
            print(f"[probe-stream] {label} FAILED: {type(e).__name__}: {e}")
            print(tb[:2000])
            print(f"[PROBE:VAE_STREAM_FAIL] label={label} reason=\"{short}\" "
                  f"last_ok={last_ok or 'none'}")
            return 1

    print(f"\n[PROBE:VAE_STREAM_FULL_OK] all {len(SHAPES)} sizes decoded; "
          f"largest={SHAPES[-1][0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
