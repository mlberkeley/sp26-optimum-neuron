from __future__ import annotations

import time

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx import nki_op, wrap_nki


@nki.jit
def _accum_probe_kernel(a_hbm, b_hbm, k: int):
    out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.load(a_hbm)
    b = nl.load(b_hbm)
    acc = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)
    for _ in range(k):
        nisa.nc_matmul(
            dst=acc[:, :],
            stationary=a,
            moving=b,
            accumulate=True,
        )
    out_tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_tile, src=acc)
    nl.store(out, value=out_tile)
    return out


@nki.jit
def _reset_probe_kernel(a_hbm, b_hbm, k: int):
    out = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.load(a_hbm)
    b = nl.load(b_hbm)
    acc = nl.zeros((128, 512), dtype=nl.float32, buffer=nl.psum)
    for _ in range(k):
        nisa.nc_matmul(
            dst=acc[:, :],
            stationary=a,
            moving=b,
        )
    out_tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_tile, src=acc)
    nl.store(out, value=out_tile)
    return out


@nki.jit
def _grid_probe_kernel(dummy):
    out = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if nl.program_id(0) == 0:
        n = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(n, nl.num_programs(0))
        nl.store(out, value=n)
    return out


@nki_op("wan_probe::accum_probe", mutates_args={})
def accum_probe(a_hbm: torch.Tensor, b_hbm: torch.Tensor, k: int) -> torch.Tensor:
    return wrap_nki(_accum_probe_kernel)(a_hbm, b_hbm, k)


@nki_op("wan_probe::reset_probe", mutates_args={})
def reset_probe(a_hbm: torch.Tensor, b_hbm: torch.Tensor, k: int) -> torch.Tensor:
    return wrap_nki(_reset_probe_kernel)(a_hbm, b_hbm, k)


@nki_op("wan_probe::grid_probe", mutates_args={})
def grid_probe(dummy: torch.Tensor, grid_size: int) -> torch.Tensor:
    return wrap_nki(_grid_probe_kernel)[(grid_size,)](dummy)


def _time_call(fn, *args):
    fn(*args)
    torch.neuron.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.neuron.synchronize()
    return out, time.perf_counter() - start


def run_accumulation_probe():
    device = torch.device("neuron")
    a = torch.ones((128, 128), dtype=torch.bfloat16, device=device)
    b = torch.ones((128, 512), dtype=torch.bfloat16, device=device)

    print("Accumulation probe: ones input, matmul value should be 128 per pass")
    for k in [1, 8, 27, 64, 216]:
        accum_out, accum_s = _time_call(accum_probe, a, b, k)
        reset_out, reset_s = _time_call(reset_probe, a, b, k)
        accum_mean = accum_out.cpu().float().mean().item()
        reset_mean = reset_out.cpu().float().mean().item()
        print(
            f"K={k:3d} accum_mean={accum_mean:.1f} "
            f"accum/K={accum_mean / k:.1f} reset_mean={reset_mean:.1f} "
            f"accum_second_ms={accum_s * 1000:.4f} reset_second_ms={reset_s * 1000:.4f}"
        )


def run_grid_probe():
    device = torch.device("neuron")
    dummy = torch.empty((1,), dtype=torch.bfloat16, device=device)
    print("Grid probe:")
    for grid_size in [1, 2, 4, 8]:
        try:
            out, elapsed_s = _time_call(grid_probe, dummy, grid_size)
            print(
                f"grid=({grid_size},) -> num_programs={int(out.cpu()[0, 0])} "
                f"second_ms={elapsed_s * 1000:.4f}"
            )
        except Exception as exc:
            print(f"grid=({grid_size},) -> ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    run_accumulation_probe()
    run_grid_probe()
