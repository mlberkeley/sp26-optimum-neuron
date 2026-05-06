"""
One-shot converter for the two .pth checkpoints in `ckpts/Wan2.2-TI2V-5B/`
that currently dominate cold-cache load time:
- models_t5_umt5-xxl-enc-bf16.pth   (11 GB → ~5 min cold pickle deserialize)
- Wan2.2_VAE.pth                     (2.7 GB → ~7 min cold pickle deserialize)

After conversion, the sibling .safetensors files load via mmap with no
pickle parsing (the DiT shards already use this format and load in <25s).

This script DOES NOT modify the existing .pth files or the loaders that
read them — it only writes new .safetensors files alongside them. Switching
the loaders to read .safetensors is a follow-up edit (separate PR).

Usage:
    python tools/convert_pth_to_safetensors.py
        [--ckpt_dir ckpts/Wan2.2-TI2V-5B]
        [--files models_t5_umt5-xxl-enc-bf16.pth Wan2.2_VAE.pth]
        [--overwrite]

Implementation notes:
- T5 / VAE .pth files contain a state_dict (dict[str, Tensor]) at the top
  level (or wrapped in {'model': state_dict}). We unwrap if needed.
- safetensors requires contiguous tensors with no shared storage. The
  helper here calls `.contiguous().clone()` on each tensor before saving
  to make sure no two parameters alias the same storage region.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _coerce_state_dict(obj):
    """Unwrap common .pth wrappers ({'model': sd}, {'state_dict': sd}, sd)."""
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        # plain {name: tensor} dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise TypeError(
        f"Unsupported .pth top-level type: {type(obj).__name__}; "
        f"expected a state_dict (dict[str, Tensor]) or a wrapper."
    )


def convert(pth_path: Path, *, overwrite: bool) -> Path:
    out = pth_path.with_suffix(".safetensors")
    if out.exists() and not overwrite:
        print(f"[skip]   {out.name} already exists (use --overwrite to replace)")
        return out
    t0 = time.perf_counter()
    print(f"[load]   {pth_path.name} ({pth_path.stat().st_size / 1e9:.2f} GB)")
    obj = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = _coerce_state_dict(obj)
    print(f"[load]   done in {time.perf_counter() - t0:.1f}s, "
          f"{len(sd)} tensors")

    # detach, clone, contiguous → safetensors-friendly
    sd_clean = {k: v.detach().contiguous().clone() for k, v in sd.items()}

    from safetensors.torch import save_file  # lazy import
    t1 = time.perf_counter()
    save_file(sd_clean, out.as_posix())
    print(f"[write]  {out.name} ({out.stat().st_size / 1e9:.2f} GB) "
          f"in {time.perf_counter() - t1:.1f}s")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="ckpts/Wan2.2-TI2V-5B",
        help="Directory containing the .pth files.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"],
        help="Filenames inside --ckpt_dir to convert.")
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing .safetensors siblings.")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_dir():
        print(f"--ckpt_dir not found: {ckpt_dir}", file=sys.stderr)
        return 2
    rc = 0
    for fname in args.files:
        path = ckpt_dir / fname
        if not path.exists():
            print(f"[miss]   {fname}", file=sys.stderr)
            rc = 1
            continue
        try:
            convert(path, overwrite=args.overwrite)
        except Exception as e:  # noqa: BLE001
            print(f"[fail]   {fname}: {e!r}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
