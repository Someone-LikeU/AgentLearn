from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolCallTask:
    """单个工具调用任务。"""

    tool_call_id: str
    function_name: str
    raw_arguments: str
    function_args: dict[str, Any]


@dataclass
class ToolCallBatch:
    """一批可执行任务；parallel=True 表示并发执行。"""

    tasks: list[ToolCallTask]
    parallel: bool


class ToolScheduler:
    """V2 调度器：按工具运行画像做分段并发与串行隔离。"""

    # 这些作用域即便是只读也建议独占，避免底层资源争用导致抖动。
    _EXCLUSIVE_SCOPES = {"system"}

    def __init__(self, get_profile: Callable[[str], dict[str, Any]], max_workers: int = 4):
        self._get_profile = get_profile
        self._max_workers = max_workers

    def plan_batches(self, tasks: list[ToolCallTask]) -> list[ToolCallBatch]:
        batches: list[ToolCallBatch] = []
        current_parallel: list[ToolCallTask] = []
        current_scope: str | None = None

        for task in tasks:
            profile = self._get_profile(task.function_name)
            is_parallel_read = bool(profile.get("is_read_only")) and bool(profile.get("is_concurrency_safe"))
            scope = str(profile.get("side_effect_scope", "none"))
            # 作用域隔离：独占作用域或跨作用域任务不合并并发段。
            scope_conflict = (
                current_scope is not None
                and (scope != current_scope or scope in self._EXCLUSIVE_SCOPES)
            )

            if is_parallel_read and not scope_conflict:
                current_parallel.append(task)
                if current_scope is None:
                    current_scope = scope
                continue

            if current_parallel:
                batches.append(ToolCallBatch(tasks=current_parallel, parallel=True))
                current_parallel = []
                current_scope = None
            batches.append(ToolCallBatch(tasks=[task], parallel=False))

        if current_parallel:
            batches.append(ToolCallBatch(tasks=current_parallel, parallel=True))
        return batches

    def execute_batches(self, batches: list[ToolCallBatch], invoke: Callable[[ToolCallTask], Any]) -> list[tuple[ToolCallTask, Any]]:
        results: list[tuple[ToolCallTask, Any]] = []
        for batch in batches:
            if batch.parallel and len(batch.tasks) > 1:
                indexed_results: dict[int, tuple[ToolCallTask, Any]] = {}
                with ThreadPoolExecutor(max_workers=min(self._max_workers, len(batch.tasks))) as executor:
                    futures = {executor.submit(invoke, task): idx for idx, task in enumerate(batch.tasks)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        task = batch.tasks[idx]
                        indexed_results[idx] = (task, future.result())
                for idx in range(len(batch.tasks)):
                    results.append(indexed_results[idx])
            else:
                for task in batch.tasks:
                    results.append((task, invoke(task)))
        return results
