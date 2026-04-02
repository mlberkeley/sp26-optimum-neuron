
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional
import json
import time


@dataclass
class ProfileNode:
    name: str
    full_path: str
    parent: Optional["ProfileNode"] = None
    children: list["ProfileNode"] = field(default_factory=list)

    start_time: float = 0.0
    end_time: float = 0.0
    elapsed_s: float = 0.0
    child_time_s: float = 0.0

    @property
    def exclusive_s(self) -> float:
        return self.elapsed_s - self.child_time_s


@dataclass
class StatBucket:
    name: str
    count: int = 0
    total_s: float = 0.0
    exclusive_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0

    def update(self, elapsed_s: float, exclusive_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.exclusive_s += exclusive_s
        self.min_s = min(self.min_s, elapsed_s)
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_s": self.total_s,
            "exclusive_s": self.exclusive_s,
            "avg_s": self.avg_s,
            "min_s": 0.0 if self.count == 0 else self.min_s,
            "max_s": self.max_s,
        }


class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.stack: list[ProfileNode] = []

        self.run_name: Optional[str] = None
        self.root: Optional[ProfileNode] = None

        self.stats_by_name: dict[str, StatBucket] = {}
        self.stats_by_path: dict[str, StatBucket] = {}

    def start_run(self, run_name: str) -> None:
        if not self.enabled:
            return

        self.run_name = run_name
        self.root = ProfileNode(name=run_name, full_path=run_name)
        self.root.start_time = time.perf_counter()

        self.stack.clear()
        self.stack.append(self.root)

    def end_run(self) -> None:
        if not self.enabled or self.root is None:
            return

        self.root.end_time = time.perf_counter()
        self.root.elapsed_s = self.root.end_time - self.root.start_time
        self.stack.clear()

    @contextmanager
    def region(self, name: str):
        if not self.enabled:
            yield None
            return

        parent = self.stack[-1] if self.stack else None
        full_path = name if parent is None else f"{parent.full_path}.{name}"

        node = ProfileNode(
            name=name,
            full_path=full_path,
            parent=parent
        )

        if parent is not None:
            parent.children.append(node)

        node.start_time = time.perf_counter()
        self.stack.append(node)

        try:
            yield node
        finally:
            node.end_time = time.perf_counter()
            node.elapsed_s = node.end_time - node.start_time
            self.stack.pop()

            if parent is not None:
                parent.child_time_s += node.elapsed_s

            self._update_stats(node)

    def _update_stats(self, node: ProfileNode) -> None:
        if node.name not in self.stats_by_name:
            self.stats_by_name[node.name] = StatBucket(name=node.name)
        self.stats_by_name[node.name].update(node.elapsed_s, node.exclusive_s)

        if node.full_path not in self.stats_by_path:
            self.stats_by_path[node.full_path] = StatBucket(name=node.full_path)
        self.stats_by_path[node.full_path].update(node.elapsed_s, node.exclusive_s)

    def _node_to_dict(self, node: ProfileNode) -> dict[str, Any]:
        return {
            "name": node.name,
            "full_path": node.full_path,
            "elapsed_s": node.elapsed_s,
            "exclusive_s": node.exclusive_s,
            "children": [self._node_to_dict(child) for child in node.children],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "root": None if self.root is None else self._node_to_dict(self.root),
            "stats_by_name": {k: v.to_dict() for k, v in self.stats_by_name.items()},
            "stats_by_path": {k: v.to_dict() for k, v in self.stats_by_path.items()},
        }

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def format_tree(self, min_time_s: float = 0.0) -> str:
        if self.root is None:
            return ""

        lines: list[str] = []

        def visit(node: ProfileNode, depth: int) -> None:
            if depth > 0 and node.elapsed_s < min_time_s:
                return

            indent = "  " * depth
            lines.append(
                f"{indent}{node.name}: total={node.elapsed_s:.6f}s "
                f"exclusive={node.exclusive_s:.6f}s"
            )

            for child in sorted(node.children, key=lambda c: c.elapsed_s, reverse=True):
                visit(child, depth + 1)

        visit(self.root, 0)
        return "\n".join(lines)