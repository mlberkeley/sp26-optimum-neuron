# Same driver as scripts/run.py, but patches Neuron NKI attention onto ``wan.modules.attention.attention``
# **before** ``WanModel`` is imported (see lazy ``wan`` / ``wan.modules`` __init__.py).
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from wan.modules import attention_nki_routing as _wan_nkir

_wan_nkir.patch_wan_attention_dispatcher()

import config
from profiling import Profiler, set_active_profiler, clear_active_profiler
from ti2v_runner import run_once


def _print_run_routed_attention_banner() -> None:
    allowed = _wan_nkir.allowed_device_types_str()
    err = getattr(_wan_nkir, 'NKI_IMPORT_ERROR', None)
    err_line = f'\n  NKI_IMPORT_ERROR={err!r}' if err else ''
    print(
        '[run_routed] NKI shim on `wan.modules.attention.attention`.\n'
        f'  NKI_FLASH_FWD_AVAILABLE={_wan_nkir.NKI_FLASH_FWD_AVAILABLE}{err_line}\n'
        f'  WAN_NKI_DEVICE_TYPES → {allowed}\n'
        '  WAN_NKI_DEBUG=1: first flash_fwd log + exit line (`nki_calls`, `attempted`, `exceptions`).\n'
        '  Self-att: `k_lens=seq_lens`; cross-att: shape mismatch skips NKI (SDPA).',
        flush=True,
    )


def _print_ti2v_token_geometry_expected() -> None:
    """Token grid aligned with WanTI2V.generate (helps interpret flash_fwd pad L)."""
    import math

    from PIL import Image

    import torch.distributed as dist
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import best_output_size

    task = 'ti2v-5B'
    cfg = WAN_CONFIGS[task]
    img = Image.open(config.IMAGE_PATH).convert('RGB')
    iw, ih = img.width, img.height
    dh = cfg.patch_size[1] * cfg.vae_stride[1]
    dw = cfg.patch_size[2] * cfg.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, MAX_AREA_CONFIGS[config.SIZE])
    frame_num = config.FRAME_NUM if config.FRAME_NUM is not None else cfg.frame_num
    sp_size = 1
    if getattr(config, 'USE_SP', False) and dist.is_initialized():
        sp_size = max(1, int(dist.get_world_size()))
    raw_sl = (
        ((frame_num - 1) // cfg.vae_stride[0] + 1)
        * (oh // cfg.vae_stride[1])
        * (ow // cfg.vae_stride[2])
        // (cfg.patch_size[1] * cfg.patch_size[2])
    )
    seq_len = int(math.ceil(raw_sl / sp_size)) * sp_size
    pad2048 = ((seq_len + 2047) // 2048) * 2048
    lat_t = (frame_num - 1) // cfg.vae_stride[0] + 1
    lat_h, lat_w = oh // cfg.vae_stride[1], ow // cfg.vae_stride[2]
    print(
        '[run_routed] TI2V tokens: pixels '
        f'{ow}x{oh} latent_thw=({lat_t},{lat_h},{lat_w}) raw={raw_sl} seq_len={seq_len} '
        f'flash_fwd_pad_L={pad2048}.',
        flush=True,
    )


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    _print_run_routed_attention_banner()
    _print_ti2v_token_geometry_expected()

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = _timestamp()
    run_name = f"ti2v_run_{run_id}"

    profiler = Profiler(enabled=config.ENABLE_PROFILING)
    profiler.start_run(run_name=run_name)

    set_active_profiler(profiler)
    try:
        run_once()
    finally:
        clear_active_profiler()
        profiler.end_run()

    if config.ENABLE_PROFILING:
        run_dir = config.PROFILE_OUTPUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        collapsed_tree_path = run_dir / "collapsed_tree.txt"
        table_by_name_path = run_dir / "table_by_name.txt"
        table_by_path_path = run_dir / "table_by_path.txt"

        print(f"Profile run directory: {run_dir}")

        collapsed_tree_path.write_text(
            profiler.format_collapsed_tree(min_time_s=config.PROFILE_MIN_TIME_MS / 1000.0),
            encoding="utf-8",
        )
        print(f"Profile collapsed tree saved to: {collapsed_tree_path}")

        table_by_name_path.write_text(
            profiler.format_table(by="name", sort_by="exclusive_s", top_k=200),
            encoding="utf-8",
        )
        print(f"Profile table by name saved to: {table_by_name_path}")

        table_by_path_path.write_text(
            profiler.format_table(by="path", sort_by="exclusive_s", top_k=200),
            encoding="utf-8",
        )
        print(f"Profile table by path saved to: {table_by_path_path}")


if __name__ == "__main__":
    main()
