import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

from wan.modules.rope3d_nki import rope3d_forward_nki, build_dense_rope_tables
import wan.modules.rope3d_nki as rope3d_nki

print("USING:", rope3d_nki.__file__)
print("KERNEL:", rope3d_nki.rope3d_forward_nki)


def sync():
    torch.neuron.synchronize()


def ms(seconds):
    return seconds * 1000.0


def summarize(times):
    return {
        "avg": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def print_summary(name, times):
    s = summarize(times)
    print(
        f"{name}: "
        f"avg={ms(s['avg']):.3f} ms, "
        f"median={ms(s['median']):.3f} ms, "
        f"min={ms(s['min']):.3f} ms, "
        f"max={ms(s['max']):.3f} ms, "
        f"stdev={ms(s['stdev']):.3f} ms"
    )
    return s


def rope_params(max_seq_len, dim, theta=10000.0):
    assert dim % 2 == 0
    idx = torch.arange(0, dim, 2)
    inv_freq = 1.0 / torch.pow(torch.tensor(theta), idx / dim)
    positions = torch.arange(max_seq_len)
    return torch.outer(positions, inv_freq)


def rope_apply_reference(x, grid_sizes, freqs):
    B, L, N, D = x.shape

    c = D // 2
    c_f = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3

    x_compute = x.float()
    freqs_compute = freqs.float()

    freqs_f = freqs_compute[:, :c_f]
    freqs_h = freqs_compute[:, c_f:c_f + c_h]
    freqs_w = freqs_compute[:, c_f + c_h:c_f + c_h + c_w]

    out = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_head = x_compute[i, :seq_len].reshape(seq_len, N, c, 2)

        angles = torch.cat(
            [
                freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, c)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x0 = x_head[:, :, :, 0]
        x1 = x_head[:, :, :, 1]

        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos

        x_rot = torch.stack((y0, y1), dim=-1).flatten(2)
        out.append(torch.cat([x_rot, x_compute[i, seq_len:]], dim=0))

    return torch.stack(out, dim=0).to(dtype=x.dtype)


def time_reference_cpu(x, grid_sizes, freqs, iters):
    times = []
    y = None
    for _ in range(iters):
        t0 = time.perf_counter()
        y = rope_apply_reference(x, grid_sizes, freqs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return y, times


def time_reference_neuron(x, grid_sizes, freqs, iters):
    times = []
    y = None
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        y = rope_apply_reference(x, grid_sizes, freqs)
        sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return y, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--f", type=int, default=5)
    parser.add_argument("--h", type=int, default=34)
    parser.add_argument("--w", type=int, default=25)
    parser.add_argument("--N", type=int, default=24)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    assert args.D % 2 == 0

    device = torch.device("neuron")
    seq_len = args.f * args.h * args.w
    tokens = args.B * seq_len * args.N
    elements = tokens * args.D

    print("\n=== config ===")
    print(f"dtype={dtype}, B={args.B}, f={args.f}, h={args.h}, w={args.w}, seq_len={seq_len}, N={args.N}, D={args.D}")
    print(f"tokens*heads={tokens:,}, elements={elements:,}")

    torch.manual_seed(0)
    x = torch.randn(args.B, seq_len, args.N, args.D, dtype=dtype).contiguous()
    grid_sizes = torch.tensor([[args.f, args.h, args.w]] * args.B, dtype=torch.long)
    freqs = rope_params(max(args.f, args.h, args.w), args.D)

    print("\n=== CPU Reference ===")
    y_ref_cpu, ref_times_cpu = time_reference_cpu(x, grid_sizes, freqs, args.iters)
    ref_stats_cpu = print_summary("CPU Reference", ref_times_cpu)

    print("\n=== Neuron setup ===")
    sync()
    t0 = time.perf_counter()
    x = x.to(device)
    freqs = freqs.to(device)
    grid_sizes = grid_sizes.to(device)
    cos, sin = build_dense_rope_tables(freqs, args.D, args.f, args.h, args.w)
    cos = cos.contiguous().to(device)
    sin = sin.contiguous().to(device)
    sync()
    t1 = time.perf_counter()
    print(f"setup_time={ms(t1 - t0):.3f} ms")

    print("\n=== Neuron Reference ===")
    y_ref_neuron, ref_times_neuron = time_reference_neuron(x, grid_sizes, freqs, args.iters)
    ref_stats_neuron = print_summary("Neuron Reference", ref_times_neuron)

    def run_nki():
        return rope3d_forward_nki(
            x,
            cos,
            sin,
            args.f,
            args.h,
            args.w,
        )

    print("\n=== first NKI call ===")
    t0 = time.perf_counter()
    y = run_nki()
    sync()
    t1 = time.perf_counter()
    first_call = t1 - t0
    print(f"first_call_including_compile={ms(first_call):.3f} ms")

    print("\n=== correctness ===")
    max_abs_nki = (y.float() - y_ref_neuron.float()).abs().max().item()
    mean_abs_nki = (y.float() - y_ref_neuron.float()).abs().mean().item()
    print(f"max_abs_nki={max_abs_nki:.6e}")
    print(f"mean_abs_nki={mean_abs_nki:.6e}")

    if dtype == torch.bfloat16:
        atol = rtol = 2e-2
    elif dtype == torch.float16:
        atol = rtol = 2e-3
    else:
        atol = rtol = 1e-5

    torch.testing.assert_close(y, y_ref_neuron, atol=atol, rtol=rtol)
    print("correctness: PASS")

    sync()

    nki_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        y = run_nki()
        sync()
        t1 = time.perf_counter()
        nki_times.append(t1 - t0)

    nki_stats = print_summary("NKI", nki_times)

    avg_speedup_cpu = ref_stats_cpu["avg"] / nki_stats["avg"]
    median_speedup_cpu = ref_stats_cpu["median"] / nki_stats["median"]

    avg_speedup_neuron = ref_stats_neuron["avg"] / nki_stats["avg"]
    median_speedup_neuron = ref_stats_neuron["median"] / nki_stats["median"]
    tokens_per_s = tokens / nki_stats["avg"]
    elements_per_s = elements / nki_stats["avg"]

    print("\n=== summary ===")
    print(f"speedup_vs_cpu_reference_avg={avg_speedup_cpu:.3f}x")
    print(f"speedup_vs_cpu_reference_median={median_speedup_cpu:.3f}x")

    print(f"speedup_vs_neuron_reference_avg={avg_speedup_neuron:.3f}x")
    print(f"speedup_vs_neuron_reference_median={median_speedup_neuron:.3f}x")

    print(f"cold_to_warm_avg_ratio={first_call / nki_stats['avg']:.3f}x")
    print(f"NKI_tokens_heads_per_sec={tokens_per_s:,.0f}")
    print(f"NKI_elements_per_sec={elements_per_s:,.0f}")


if __name__ == "__main__":
    main()
