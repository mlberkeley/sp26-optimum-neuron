
from __future__ import annotations

from functools import wraps
from typing import Optional

from .active import get_active_profiler


def trace(name: Optional[str] = None):
    def decorator(fn):
        trace_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            profiler = get_active_profiler()
            if profiler is None or not profiler.enabled:
                return fn(*args, **kwargs)

            with profiler.region(trace_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator