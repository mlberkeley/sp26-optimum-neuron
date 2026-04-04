from __future__ import annotations

from pathlib import Path
import sys
import json
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

import torch
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

import config
from wan_flops import estimate_flops


TASK = "ti2v-5B"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _maybe_cuda_sync() -> None:
    if _has_cuda():
        torch.cuda.synchronize()


def _maybe_nvtx_push(msg: str) -> None:
    if _has_cuda():
        torch.cuda.nvtx.range_push(msg)


def _maybe_nvtx_pop() -> None:
    if _has_cuda():
        torch.cuda.nvtx.range_pop()


def _output_path() -> Path:
    resolved = _resolved_cfg()
    return config.OUTPUT_DIR / f"ti2v_{config.SIZE}_{resolved['sample_steps']}steps_{_timestamp()}.mp4"


def _resolved_cfg():
    cfg = WAN_CONFIGS[TASK]

    sample_steps = config.SAMPLE_STEPS if config.SAMPLE_STEPS is not None else cfg.sample_steps
    sample_shift = config.SAMPLE_SHIFT if config.SAMPLE_SHIFT is not None else cfg.sample_shift
    sample_guide_scale = (
        config.SAMPLE_GUIDE_SCALE
        if config.SAMPLE_GUIDE_SCALE is not None
        else cfg.sample_guide_scale
    )
    frame_num = config.FRAME_NUM if config.FRAME_NUM is not None else cfg.frame_num

    return {
        "cfg": cfg,
        "sample_steps": sample_steps,
        "sample_shift": sample_shift,
        "sample_guide_scale": sample_guide_scale,
        "frame_num": frame_num,
    }


def _build_pipeline():
    resolved = _resolved_cfg()
    cfg = resolved["cfg"]
    return wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(config.CKPT_DIR),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=config.T5_CPU,
        convert_model_dtype=config.CONVERT_MODEL_DTYPE,
    )


def _generate(pipe, img):
    resolved = _resolved_cfg()

    _maybe_cuda_sync()
    _maybe_nvtx_push(config.NVTX_RANGE)

    t0 = time.perf_counter()
    video = pipe.generate(
        config.PROMPT,
        img=img,
        size=SIZE_CONFIGS[config.SIZE],
        max_area=MAX_AREA_CONFIGS[config.SIZE],
        frame_num=resolved["frame_num"],
        shift=resolved["sample_shift"],
        sample_solver=config.SAMPLE_SOLVER,
        sampling_steps=resolved["sample_steps"],
        guide_scale=resolved["sample_guide_scale"],
        seed=config.BASE_SEED,
        offload_model=config.OFFLOAD_MODEL,
    )
    _maybe_cuda_sync()
    t1 = time.perf_counter()

    _maybe_nvtx_pop()
    return video, (t1 - t0)


def _safe_div(x, y):
    if x is None or y is None or y == 0:
        return None
    return x / y


def _derived_metrics(
    *,
    wall_elapsed_s: float,
    ncu_summary: dict | None,
    peak_memory_allocated_bytes: int | None,
    peak_memory_reserved_bytes: int | None,
):
    resolved = _resolved_cfg()
    width, height = SIZE_CONFIGS[config.SIZE]

    total_flops = estimate_flops(
        batch_size=1,
        frames=resolved["frame_num"],
        height=height,
        width=width,
        sample_steps=resolved["sample_steps"],
    )
    total_pixels = width * height * resolved["frame_num"]

    kernel_elapsed_s = (
        ncu_summary.get("total_duration_s")
        if ncu_summary is not None else None
    )

    achieved_tflops_wall = _safe_div(total_flops, wall_elapsed_s)
    if achieved_tflops_wall is not None:
        achieved_tflops_wall /= 1e12

    achieved_tflops_kernel = _safe_div(total_flops, kernel_elapsed_s)
    if achieved_tflops_kernel is not None:
        achieved_tflops_kernel /= 1e12

    dram = ncu_summary.get("total_dram_bytes") if ncu_summary else None
    l2 = ncu_summary.get("total_l2_bytes") if ncu_summary else None
    l1 = ncu_summary.get("total_l1_bytes") if ncu_summary else None

    return {
        "estimated_total_flops": total_flops,
        "estimated_total_tflops": total_flops / 1e12,
        "wall_generation_elapsed_s": wall_elapsed_s,
        "kernel_elapsed_s": kernel_elapsed_s,
        "time_per_step_wall_s": _safe_div(wall_elapsed_s, resolved["sample_steps"]),
        "time_per_step_kernel_s": _safe_div(kernel_elapsed_s, resolved["sample_steps"]),
        "frames_per_s_wall": _safe_div(resolved["frame_num"], wall_elapsed_s),
        "frames_per_s_kernel": _safe_div(resolved["frame_num"], kernel_elapsed_s),
        "pixels_per_s_wall": _safe_div(total_pixels, wall_elapsed_s),
        "pixels_per_s_kernel": _safe_div(total_pixels, kernel_elapsed_s),
        "achieved_tflops_wall": achieved_tflops_wall,
        "achieved_tflops_kernel": achieved_tflops_kernel,
        "peak_memory_allocated_bytes": peak_memory_allocated_bytes,
        "peak_memory_allocated_gb": (
            peak_memory_allocated_bytes / (1024 ** 3)
            if peak_memory_allocated_bytes is not None else None
        ),
        "peak_memory_reserved_bytes": peak_memory_reserved_bytes,
        "peak_memory_reserved_gb": (
            peak_memory_reserved_bytes / (1024 ** 3)
            if peak_memory_reserved_bytes is not None else None
        ),
        "ai_dram": _safe_div(total_flops, dram),
        "ai_l2": _safe_div(total_flops, l2),
        "ai_l1": _safe_div(total_flops, l1),
        "mfu_wall": (
            achieved_tflops_wall / config.PEAK_TFLOPS
            if achieved_tflops_wall is not None and config.PEAK_TFLOPS not in (None, 0)
            else None
        ),
        "mfu_kernel": (
            achieved_tflops_kernel / config.PEAK_TFLOPS
            if achieved_tflops_kernel is not None and config.PEAK_TFLOPS not in (None, 0)
            else None
        ),
        "duration_weighted_sm_efficiency_pct": (
            ncu_summary.get("duration_weighted_sm_efficiency_pct") if ncu_summary else None
        ),
        "duration_weighted_tensor_active_pct": (
            ncu_summary.get("duration_weighted_tensor_active_pct") if ncu_summary else None
        ),
        "resolved_runtime_config": {
            "size": config.SIZE,
            "frame_num": resolved["frame_num"],
            "sample_steps": resolved["sample_steps"],
            "sample_shift": resolved["sample_shift"],
            "sample_guide_scale": resolved["sample_guide_scale"],
            "sample_solver": config.SAMPLE_SOLVER,
            "offload_model": config.OFFLOAD_MODEL,
            "t5_cpu": config.T5_CPU,
            "convert_model_dtype": config.CONVERT_MODEL_DTYPE,
            "cuda_available": _has_cuda(),
        },
    }


def run_once(*, ncu_summary: dict | None = None) -> dict:
    assert config.SIZE in SUPPORTED_SIZES[TASK], f"Unsupported size for {TASK}: {config.SIZE}"
    assert config.CKPT_DIR.exists(), f"Missing checkpoint dir: {config.CKPT_DIR}"
    assert config.IMAGE_PATH.exists(), f"Missing image: {config.IMAGE_PATH}"

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(config.IMAGE_PATH).convert("RGB")
    pipe = _build_pipeline()

    if config.WARMUP:
        warmup_video, _ = _generate(pipe, img)
        del warmup_video
        _maybe_cuda_sync()

    peak_memory_allocated_bytes = None
    peak_memory_reserved_bytes = None

    if _has_cuda():
        torch.cuda.reset_peak_memory_stats()

    video, wall_elapsed_s = _generate(pipe, img)

    if _has_cuda():
        peak_memory_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_memory_reserved_bytes = torch.cuda.max_memory_reserved()

    output_path = None
    if not config.SKIP_SAVE:
        output_path = _output_path()
        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=WAN_CONFIGS[TASK].sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

    payload = {
        "task": TASK,
        "output_path": str(output_path) if output_path is not None else None,
        "ncu_summary": ncu_summary,
        "derived_metrics": _derived_metrics(
            wall_elapsed_s=wall_elapsed_s,
            ncu_summary=ncu_summary,
            peak_memory_allocated_bytes=peak_memory_allocated_bytes,
            peak_memory_reserved_bytes=peak_memory_reserved_bytes,
        ),
    }

    summary_path = config.OUTPUT_DIR / f"ti2v_summary_{_timestamp()}.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    if output_path is not None:
        print(f"Video saved to: {output_path}")

    return payload