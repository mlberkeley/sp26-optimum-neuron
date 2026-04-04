from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

import config
from cuda_metrics import has_cuda, has_ncu, run_with_ncu
from ti2v_runner import run_once


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if config.PROFILE_WITH_NCU:
        if not has_cuda():
            raise RuntimeError("CUDA is required for NCU profiling.")
        if not has_ncu():
            print("Warning: ncu not found on PATH; running without profiling.")
            run_once()
            return

        script_path = Path(__file__).resolve()
        log_path = config.OUTPUT_DIR / "ti2v_profile.log"
        raw_csv_path = config.OUTPUT_DIR / "ti2v_profile.csv"

        cmd = [sys.executable, str(script_path), "--child"]
        run_info = run_with_ncu(
            cmd,
            cwd=script_path.parent,
            log_path=log_path,
            raw_csv_path=raw_csv_path,
            nvtx_range=config.NVTX_RANGE,
        )
        run_once(ncu_summary=run_info["summary"])
        print(f"NCU raw CSV saved to: {raw_csv_path}")
        print(f"NCU log saved to: {log_path}")
    else:
        run_once()


if __name__ == "__main__":
    if "--child" in sys.argv:
        run_once()
    else:
        main()