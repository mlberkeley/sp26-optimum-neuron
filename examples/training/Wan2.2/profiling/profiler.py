
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


@dataclass
class CollapsedNode:
    name: str
    count: int = 0
    total_s: float = 0.0
    exclusive_s: float = 0.0
    children: dict[str, "CollapsedNode"] = field(default_factory=dict)

    @property
    def avg_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


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

    # outputs entire tree of calls (very big)
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
        out = "\n".join(lines) + "\n"
        return out

    def _collapse_subtree(self, node: ProfileNode) -> CollapsedNode:
        collapsed = CollapsedNode(
            name=node.name,
            count=1,
            total_s=node.elapsed_s,
            exclusive_s=node.exclusive_s,
        )

        # recursively collapse node's children
        for child in node.children:
            child_collapsed = self._collapse_subtree(child)

            # merge nodes with same name
            existing = collapsed.children.get(child_collapsed.name)
            if existing is None:
                collapsed.children[child_collapsed.name] = child_collapsed
            else:
                self._merge_collapsed_nodes(existing, child_collapsed)

        return collapsed

    def _merge_collapsed_nodes(self, dst: CollapsedNode, src: CollapsedNode) -> None:
        dst.count += src.count
        dst.total_s += src.total_s
        dst.exclusive_s += src.exclusive_s

        # need to recursively merge collapsed nodes since 
        # children of two nodes to be merged could have the same name, so they need to be merged
        for child_name, src_child in src.children.items():
            dst_child = dst.children.get(child_name)
            if dst_child is None:
                dst.children[child_name] = src_child
            else:
                self._merge_collapsed_nodes(dst_child, src_child)

    def format_collapsed_tree(self, min_time_s: float = 0.0) -> str:
        if self.root is None:
            return ""

        collapsed_root = self._collapse_subtree(self.root)
        lines: list[str] = []

        def visit(node: CollapsedNode, depth: int) -> None:
            if depth > 0 and node.total_s < min_time_s:
                return

            indent = "  " * depth
            lines.append(
                f"{indent}{node.name}: "
                f"count={node.count} "
                f"total={node.total_s:.6f}s "
                f"exclusive={node.exclusive_s:.6f}s "
                f"avg={node.avg_s:.6f}s"
            )

            for child in sorted(node.children.values(), key=lambda c: c.total_s, reverse=True):
                visit(child, depth + 1)

        visit(collapsed_root, 0)
        out = "\n".join(lines) + "\n"
        return out

    def format_table(
        self,
        by: str = "name",
        sort_by: str = "total_s",  # choose an attribute of StatBucket (total_s, exclusive_s, min_s, max_s)
        top_k: int = 200,
    ) -> str:
        if by == "name":
            buckets = self.stats_by_name
        elif by == "path":
            buckets = self.stats_by_path
        else:
            raise ValueError(f"Unsupported table grouping: {by}")

        rows = list(buckets.values())
        rows.sort(key=lambda bucket: getattr(bucket, sort_by), reverse=True)

        header = (
            f"{'name':100} "
            f"{'count':>8} "
            f"{'total_s':>12} "
            f"{'excl_s':>12} "
            f"{'avg_s':>12} "
            f"{'min_s':>12} "
            f"{'max_s':>12}"
        )
        sep = "-" * len(header)
        out = [header, sep]

        for bucket in rows[:top_k]:
            name = bucket.name
            if by == "path":
                parts = name.split('.')
                if len(parts) > 1:
                    name = "run." + ".".join(parts[1:])

            out.append(
                f"{name[:100]:100} "
                f"{bucket.count:8d} "
                f"{bucket.total_s:12.6f} "
                f"{bucket.exclusive_s:12.6f} "
                f"{bucket.avg_s:12.6f} "
                f"{(0.0 if bucket.count == 0 else bucket.min_s):12.6f} "
                f"{bucket.max_s:12.6f}"
            )

        p = "\n".join(out) + "\n"
        return p