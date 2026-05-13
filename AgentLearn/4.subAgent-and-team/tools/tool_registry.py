from __future__ import annotations

from typing import Any

from tools.base_tool import Tool


class ToolRegistry:
    """工具注册中心，统一管理工具查找与导出。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """注册工具；同名会覆盖旧实现。"""
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """按名称获取工具实例。"""
        return self._tools.get(name)

    def as_openai_tools(self) -> list[dict[str, Any]]:
        """导出 OpenAI function calling 兼容格式。"""
        result = []
        for tool in self._tools.values():
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.spec.name,
                        "description": tool.spec.description,
                        "parameters": tool.spec.input_schema,
                    },
                }
            )
        return result

    def as_function_map(self):
        """导出函数映射，兼容现有执行流程。"""
        return {name: tool.call for name, tool in self._tools.items()}
