from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

import config
from profiling import region


TASK = "ti2v-5B"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolved_cfg() -> dict:
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


def _output_path() -> Path:
    resolved = _resolved_cfg()
    return config.OUTPUT_DIR / f"ti2v_{config.SIZE}_{resolved['sample_steps']}steps_{_timestamp()}.mp4"


def _build_pipeline():
    resolved = _resolved_cfg()
    cfg = resolved["cfg"]

    return wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(config.CKPT_DIR),
        device_id=config.DEVICE_ID,
        rank=config.RANK,
        t5_fsdp=config.T5_FSDP,
        dit_fsdp=config.DIT_FSDP,
        use_sp=config.USE_SP,
        t5_cpu=config.T5_CPU,
        convert_model_dtype=config.CONVERT_MODEL_DTYPE,
        device=config.DEVICE,
    )


def run_once():
    assert config.SIZE in SUPPORTED_SIZES[TASK], f"Unsupported size for {TASK}: {config.SIZE}"
    assert config.CKPT_DIR.exists(), f"Missing checkpoint dir: {config.CKPT_DIR}"
    assert config.IMAGE_PATH.exists(), f"Missing image: {config.IMAGE_PATH}"

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_cfg()

    with region("load_image"):
        img = Image.open(config.IMAGE_PATH).convert("RGB")

    with region("build_pipeline"):
        pipe = _build_pipeline()

    with region("generate"):
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

    output_path = None
    if not config.SKIP_SAVE:
        with region("save_video"):
            output_path = _output_path()
            save_video(
                tensor=video[None],
                save_file=str(output_path),
                fps=WAN_CONFIGS[TASK].sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            print(f"Video saved to: {output_path}")

    return {
        "task": TASK,
        "device": config.DEVICE,
        "output_path": str(output_path) if output_path is not None else None,
    }