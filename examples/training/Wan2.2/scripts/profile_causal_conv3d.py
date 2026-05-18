from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

WAN_ROOT = Path(__file__).resolve().parents[1]
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from wan.modules.vae2_2 import CausalConv3d


SHAPES = {
    "tiny": (8, 8, 8, 8),
    "entry": (48, 1024, 44, 80),
    "head": (256, 12, 176, 320),
    "residual": (1024, 1024, 44, 80),
}


def _sync() -> None:
    torch.neuron.synchronize()


def _make_case(shape_name: str, mode: str):
    in_channels, out_channels, height, width = SHAPES[shape_name]
    device = torch.device("neuron")
    dtype = torch.bfloat16

    torch.manual_seed(0)
    conv = CausalConv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        use_nki_decoder_kernel=mode == "nki",
    )
    conv = conv.to(device=device, dtype=dtype).eval()
    x = torch.randn((1, in_channels, 1, height, width), dtype=dtype, device=device)
    cache_x = torch.randn((1, in_channels, 2, height, width), dtype=dtype, device=device)
    return conv, x, cache_x


def _time_forward(conv, x, cache_x, warmups: int, repeats: int):
    with torch.no_grad():
        for _ in range(warmups):
            conv(x, cache_x=cache_x)
            _sync()

        times = []
        out = None
        for _ in range(repeats):
            start = time.perf_counter()
            out = conv(x, cache_x=cache_x)
            _sync()
            times.append(time.perf_counter() - start)

    return out, times


def run_profile_case(args) -> None:
    conv, x, cache_x = _make_case(args.shape, args.mode)
    out, times = _time_forward(conv, x, cache_x, args.warmups, args.repeats)
    mean_ms = sum(times) / len(times) * 1000
    second_ms = times[0] * 1000
    print(
        f"shape={args.shape} mode={args.mode} "
        f"out_shape={tuple(out.shape)} second_ms={second_ms:.4f} mean_ms={mean_ms:.4f}"
    )


def compare_case(args) -> None:
    baseline_conv, x, cache_x = _make_case(args.shape, "baseline")
    nki_conv, _, _ = _make_case(args.shape, "nki")
    nki_conv.weight.data.copy_(baseline_conv.weight.data)
    nki_conv.bias.data.copy_(baseline_conv.bias.data)

    baseline_out, baseline_times = _time_forward(
        baseline_conv,
        x,
        cache_x,
        args.warmups,
        args.repeats,
    )
    nki_out, nki_times = _time_forward(nki_conv, x, cache_x, args.warmups, args.repeats)
    max_diff = (baseline_out.float() - nki_out.float()).abs().max().item()

    baseline_ms = baseline_times[0] * 1000
    nki_ms = nki_times[0] * 1000
    print(
        f"shape={args.shape} baseline_second_ms={baseline_ms:.4f} "
        f"nki_second_ms={nki_ms:.4f} nki_over_baseline={baseline_ms / nki_ms:.3f}x "
        f"max_diff={max_diff:.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), default="residual")
    parser.add_argument("--mode", choices=("baseline", "nki", "compare"), default="compare")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.mode == "compare":
        compare_case(parsed_args)
    else:
        run_profile_case(parsed_args)
