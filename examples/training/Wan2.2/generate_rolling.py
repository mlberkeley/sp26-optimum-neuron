"""
Rolling inference script for Wan2.2-TI2V-5B.

Generates a long video by chaining TI2V chunks. Each chunk is conditioned on
the last latent frame of the previous chunk, extending the video indefinitely.

Total output frames = n_chunks * (chunk_frame_num - 1) + 1

Single-device usage:
    python generate_rolling.py \
        --ckpt_dir /path/to/Wan2.2-TI2V-5B \
        --image examples/i2v_input.JPG \
        --prompt "A cat walking through a forest" \
        --n_chunks 3 \
        --chunk_frame_num 121 \
        --save_file output_rolling.mp4

Distributed usage (sequence parallelism across N NeuronCores/GPUs):
    torchrun --nproc_per_node N generate_rolling.py \
        --ulysses_size N \
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
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.distributed.util import init_distributed_group
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
        "--offload_model", type=str2bool, default=None,
        help="Offload DiT to CPU after each forward. Defaults to False in distributed mode, True otherwise.")
    parser.add_argument(
        "--t5_cpu", action="store_true", default=False,
        help="Keep T5 encoder on CPU.")
    parser.add_argument(
        "--ulysses_size", type=int, default=1,
        help="Number of devices for sequence parallelism. Must equal --nproc_per_node when using torchrun.")
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on: 'cuda', 'cpu', or 'neuron'.")
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output .mp4 path. Auto-named if not specified.")
    return parser.parse_args()


def main():
    args = _parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        if args.device == "cuda":
            backend = "nccl"
        elif args.device == "neuron":
            import torch_xla.distributed.xla_backend  # registers "xla" backend
            backend = "xla"
        else:
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert args.ulysses_size == 1, "ulysses_size > 1 requires launching with torchrun --nproc_per_node."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            f"ulysses_size ({args.ulysses_size}) must equal world_size ({world_size})."
        init_distributed_group()

    cfg = WAN_CONFIGS["ti2v-5B"]

    if args.device == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = args.device

    # broadcast seed from rank 0 so all ranks use the same seed
    if dist.is_initialized():
        seed_list = [args.seed if rank == 0 else None]
        dist.broadcast_object_list(seed_list, src=0)
        args.seed = seed_list[0]

    seed = args.seed if args.seed >= 0 else random.randint(0, sys.maxsize)

    logging.info("Loading WanTI2V pipeline...")
    pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_cpu=args.t5_cpu,
        use_sp=(args.ulysses_size > 1),
        device=device,
        convert_model_dtype=True,
    )

    img = Image.open(args.image).convert("RGB")
    logging.info(f"Anchor image: {args.image} ({img.width}x{img.height})")

    total_frames = args.n_chunks * (args.chunk_frame_num - 1) + 1
    logging.info(
        f"Generating {args.n_chunks} chunks × {args.chunk_frame_num} frames "
        f"= {total_frames} total frames "
        f"(ulysses_size={args.ulysses_size})")

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

    if rank == 0 and video is not None:
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

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
