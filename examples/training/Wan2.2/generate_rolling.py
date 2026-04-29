"""
Rolling inference script for Wan2.2-TI2V-5B.

Generates a long video by chaining TI2V chunks. Each chunk is conditioned on
the last latent frame of the previous chunk, extending the video indefinitely.

Total output frames = n_chunks * (chunk_frame_num - 1) + 1

Usage:
    python generate_rolling.py \
        --ckpt_dir /path/to/Wan2.2-TI2V-5B \
        --image examples/i2v_input.JPG \
        --prompt "A cat walking through a forest" \
        --n_chunks 3 \
        --chunk_frame_num 121 \
        --save_file output_rolling.mp4
"""
import argparse
import logging
import os
import random
import sys
from datetime import datetime

import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a long video using rolling TI2V inference.")
    parser.add_argument(
        "--ckpt_dir", type=str, required=True,
        help="Path to Wan2.2-TI2V-5B checkpoint directory.")
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the anchor image for the first chunk.")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt shared across all chunks.")
    parser.add_argument(
        "--n_chunks", type=int, default=2,
        help="Number of chunks to generate. Total frames = n_chunks*(chunk_frame_num-1)+1.")
    parser.add_argument(
        "--chunk_frame_num", type=int, default=121,
        help="Frames per chunk. Must satisfy 4n+1 (e.g. 81, 121).")
    parser.add_argument(
        "--max_area", type=int, default=704 * 1280,
        help="Maximum pixel area for resolution scaling.")
    parser.add_argument(
        "--sample_steps", type=int, default=40,
        help="Denoising steps per chunk.")
    parser.add_argument(
        "--sample_shift", type=float, default=5.0,
        help="Noise schedule shift parameter.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc",
        choices=["unipc", "dpm++"],
        help="Diffusion solver.")
    parser.add_argument(
        "--guide_scale", type=float, default=5.0,
        help="Classifier-free guidance scale.")
    parser.add_argument(
        "--n_prompt", type=str, default="",
        help="Negative prompt.")
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="Base random seed. Chunk k uses seed+k.")
    parser.add_argument(
        "--offload_model", type=str2bool, default=True,
        help="Offload DiT to CPU between denoising steps to save VRAM.")
    parser.add_argument(
        "--t5_cpu", action="store_true", default=False,
        help="Keep T5 encoder on CPU.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on: 'cuda', 'cpu', or 'neuron'.")
    parser.add_argument(
        "--device_id", type=int, default=0,
        help="GPU device index (only used when --device=cuda).")
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output .mp4 path. Auto-named if not specified.")
    return parser.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    cfg = WAN_CONFIGS["ti2v-5B"]

    if args.device == "cuda":
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = args.device

    logging.info("Loading WanTI2V pipeline...")
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        device=device,
    )

    img = Image.open(args.image).convert("RGB")
    logging.info(f"Anchor image: {args.image} ({img.width}x{img.height})")

    seed = args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)
    total_frames = args.n_chunks * (args.chunk_frame_num - 1) + 1
    logging.info(
        f"Generating {args.n_chunks} chunks × {args.chunk_frame_num} frames "
        f"= {total_frames} total frames")

    video = pipeline.generate_rolling(
        input_prompt=args.prompt,
        img=img,
        n_chunks=args.n_chunks,
        chunk_frame_num=args.chunk_frame_num,
        max_area=args.max_area,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.guide_scale,
        n_prompt=args.n_prompt,
        seed=seed,
        offload_model=args.offload_model,
    )

    if video is not None:
        if args.save_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = args.prompt.replace(" ", "_")[:40]
            args.save_file = f"rolling_{args.n_chunks}x{args.chunk_frame_num}_{slug}_{ts}.mp4"

        logging.info(f"Saving video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logging.info("Done.")


if __name__ == "__main__":
    main()
