
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

from .profiler import Profiler

_ACTIVE_PROFILER: Optional[Profiler] = None


def set_active_profiler(profiler: Optional[Profiler]) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_active_profiler() -> Optional[Profiler]:
    return _ACTIVE_PROFILER


def clear_active_profiler() -> None:
    set_active_profiler(None)


def region(name: str):
    profiler = get_active_profiler()
    if profiler is None or not profiler.enabled:
        return nullcontext()
    return profiler.region(name)