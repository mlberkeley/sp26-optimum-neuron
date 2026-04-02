# run.py
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT.parent
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))

import config
from ti2v_runner import run_once


def main():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_once()


if __name__ == "__main__":
    main()