# run.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

import config
from profiling import Profiler, set_active_profiler, clear_active_profiler
from ti2v_runner import run_once


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
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

        json_path = run_dir / "profile.json"
        tree_path = run_dir / "tree.txt"
        collapsed_tree_path = run_dir / "collapsed_tree.txt"
        table_by_name_path = run_dir / "table_by_name.txt"
        table_by_path_path = run_dir / "table_by_path.txt"

        print(f"Profile run directory: {run_dir}")

        # UNCOMMENT IF YOU WANT BIG JSON (can be used for AB testing)
        # profiler.write_json(json_path)
        # print(f"Profile JSON saved to: {json_path}")

        # UNCOMMENT IF YOU WANT BIG TREE (not super useful unless u want big details)
        '''
        tree_path.write_text(
            profiler.format_tree(min_time_s=config.PROFILE_MIN_TIME_MS / 1000.0),
            encoding="utf-8",
        )

        print(f"Profile tree saved to: {tree_path}")
        '''


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