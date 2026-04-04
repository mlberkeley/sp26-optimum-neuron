from __future__ import annotations

import csv
import shutil
import subprocess
import time
from pathlib import Path


NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "lts__t_bytes.sum",
    "l1tex__t_bytes.sum",
]


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def has_ncu() -> bool:
    return shutil.which("ncu") is not None


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def _parse_float(x: str):
    x = x.strip().replace(",", "")
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def run_with_ncu(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    raw_csv_path: Path,
    nvtx_range: str,
) -> dict:
    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--page", "raw",
        "--csv",
        "--nvtx",
        "--nvtx-include", nvtx_range,
        "--metrics", ",".join(NCU_METRICS),
        *cmd,
    ]

    start = time.perf_counter()
    result = subprocess.run(ncu_cmd, cwd=cwd, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - start

    log_path.write_text(
        "=== NCU CMD ===\n"
        + quote_cmd(ncu_cmd)
        + "\n\n=== STDOUT ===\n"
        + (result.stdout or "")
        + "\n=== STDERR ===\n"
        + (result.stderr or ""),
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ncu_cmd, result.stdout, result.stderr)

    raw_csv_path.write_text(result.stdout, encoding="utf-8")
    summary = summarize_ncu_csv(raw_csv_path)

    return {
        "elapsed_s": elapsed_s,
        "summary": summary,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def summarize_ncu_csv(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = None
        idx = None

        for row in reader:
            if not row or row[0].startswith("==PROF=="):
                continue
            if header is None and "Metric Name" in row and "Metric Value" in row:
                header = row
                idx = {name: i for i, name in enumerate(row)}
                continue
            if header is None or len(row) < len(header):
                continue
            rows.append({
                "kernel_name": row[idx["Kernel Name"]],
                "metric_name": row[idx["Metric Name"]],
                "metric_value": _parse_float(row[idx["Metric Value"]]),
            })

    by_kernel = {}
    for r in rows:
        by_kernel.setdefault(r["kernel_name"], {})
        by_kernel[r["kernel_name"]][r["metric_name"]] = r["metric_value"]

    total_duration_ns = 0.0
    total_dram_bytes = 0.0
    total_l2_bytes = 0.0
    total_l1_bytes = 0.0

    sm_weighted_num = 0.0
    tensor_weighted_num = 0.0
    weight_den = 0.0

    for m in by_kernel.values():
        dur = m.get("gpu__time_duration.sum")
        dram = m.get("dram__bytes.sum")
        l2 = m.get("lts__t_bytes.sum")
        l1 = m.get("l1tex__t_bytes.sum")
        sm = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        tensor = m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")

        if dur is not None:
            total_duration_ns += dur
            weight_den += dur
            if sm is not None:
                sm_weighted_num += sm * dur
            if tensor is not None:
                tensor_weighted_num += tensor * dur
        if dram is not None:
            total_dram_bytes += dram
        if l2 is not None:
            total_l2_bytes += l2
        if l1 is not None:
            total_l1_bytes += l1

    return {
        "total_duration_s": total_duration_ns / 1e9 if total_duration_ns > 0 else None,
        "total_dram_bytes": total_dram_bytes if total_dram_bytes > 0 else None,
        "total_l2_bytes": total_l2_bytes if total_l2_bytes > 0 else None,
        "total_l1_bytes": total_l1_bytes if total_l1_bytes > 0 else None,
        "duration_weighted_sm_efficiency_pct": (
            sm_weighted_num / weight_den if weight_den > 0 else None
        ),
        "duration_weighted_tensor_active_pct": (
            tensor_weighted_num / weight_den if weight_den > 0 else None
        ),
    }