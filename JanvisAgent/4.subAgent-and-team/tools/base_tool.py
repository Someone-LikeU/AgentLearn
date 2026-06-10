from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """工具元信息定义。"""

    # 工具名称
    name: str
    # 工具描述
    description: str
    # 输入参数 Schema（JSON Schema）
    input_schema: dict[str, Any]
    # 输出结果 Schema（JSON Schema）
    output_schema: dict[str, Any]
    # 是否并发安全（并发执行时不会产生冲突）
    is_concurrency_safe: bool
    # 是否只读（不修改外部状态）
    is_read_only: bool
    # 是否有破坏风险（例如删除、覆盖、执行命令）
    is_destructive: bool
    # 副作用作用域（用于调度器判定是否可批量并发）
    side_effect_scope: str = "none"


class Tool(ABC):
    """工具抽象基类，所有具体工具都应继承该类。"""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """返回工具元信息。"""

    @abstractmethod
    def call(self, args: dict[str, Any]) -> Any:
        """执行工具逻辑。"""


class FunctionTool(Tool):
    """函数适配器：将现有函数调用包装为 Tool 实现。"""

    def __init__(self, spec: ToolSpec, handler: Callable[..., Any]):
        self._spec = spec
        self._handler = handler

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def call(self, args: dict[str, Any]) -> Any:
        return self._handler(**args)
