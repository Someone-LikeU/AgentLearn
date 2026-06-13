# encoding: utf-8
from __future__ import annotations

from pathlib import Path
from typing import Any

from external_mcp_manager import ExternalMCPManager
from mcp_client import create_tcp_local_mcp_manager


class _FailedMCPManager:
    def __init__(self, name: str, error: Exception):
        self.mode = name
        self._error = str(error)

    def start(self) -> None:
        raise RuntimeError(self._error)

    def close(self) -> None:
        return None


class MCPAggregator:
    """把本地 MCP 和外部 MCP 聚合成 Agent 可直接使用的 MCP client。"""

    def __init__(self, managers: list[Any] | None = None):
        self.mode = "aggregate"
        self.managers = managers or []
        self._started = False
        self._tools: list[dict[str, Any]] = []
        self._routes: dict[str, Any] = {}
        self._start_errors: dict[str, str] = {}

    def start(self) -> None:
        if self._started:
            return

        self._tools = []
        self._routes = {}
        self._start_errors = {}

        for manager in self.managers:
            manager_name = self._manager_name(manager)
            try:
                start = getattr(manager, "start", None)
                if callable(start):
                    start()
                for tool in manager.list_tools():
                    tool_name = str(tool.get("name", "")).strip()
                    if not tool_name:
                        continue
                    if tool_name in self._routes:
                        self._start_errors[tool_name] = f"duplicate tool name from {manager_name}"
                        continue
                    self._routes[tool_name] = manager
                    self._tools.append(tool)
            except Exception as error:
                self._start_errors[manager_name] = str(error)

        self._started = True

    def close(self) -> None:
        for manager in self.managers:
            close = getattr(manager, "close", None)
            if callable(close):
                close()
        self._started = False
        self._tools = []
        self._routes = {}

    def ping(self) -> dict[str, Any]:
        self.start()
        return {
            "message": "pong",
            "tool_count": len(self._tools),
            "counts": self.get_tool_counts(),
            "errors": dict(self._start_errors),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.start()
        manager = self._routes.get(name)
        if manager is None:
            raise ValueError(f"Unknown MCP tool '{name}'")
        return manager.call_tool(name, arguments)

    def get_tool_counts(self) -> dict[str, int]:
        self.start()
        counts: dict[str, int] = {"local": 0, "remote": 0, "total": len(self._tools)}
        for manager in self.managers:
            get_counts = getattr(manager, "get_tool_counts", None)
            if not callable(get_counts):
                continue
            for key, value in get_counts().items():
                counts[key] = counts.get(key, 0) + int(value or 0)
        counts["total"] = len(self._tools)
        return counts

    def get_tool_capabilities(self) -> dict[str, dict[str, Any]]:
        self.start()
        capabilities: dict[str, dict[str, Any]] = {}
        for manager in self.managers:
            get_capabilities = getattr(manager, "get_tool_capabilities", None)
            if callable(get_capabilities):
                capabilities.update(get_capabilities())
        return capabilities

    def get_start_errors(self) -> dict[str, str]:
        self.start()
        errors = dict(self._start_errors)
        for manager in self.managers:
            get_errors = getattr(manager, "get_start_errors", None)
            if callable(get_errors):
                for key, value in get_errors().items():
                    errors[f"{self._manager_name(manager)}:{key}"] = value
        return errors

    def _manager_name(self, manager: Any) -> str:
        return getattr(manager, "mode", manager.__class__.__name__)


def create_aggregate_mcp_client(
        *,
        project_root: str | Path | None = None,
        enable_local: bool = True,
        enable_external: bool = True,
        local_host: str = "127.0.0.1",
        local_port: int = 7777,
        local_server_script: str | None = None,
        external_config_path: str | Path | None = None,
) -> MCPAggregator:
    root = Path(project_root or Path(__file__).resolve().parent)
    managers: list[Any] = []

    if enable_local:
        try:
            local_manager, _ = create_tcp_local_mcp_manager(
                host=local_host,
                port=local_port,
                server_script=local_server_script,
            )
            managers.append(local_manager)
        except Exception as error:
            # 本地 MCP 启动失败时仍保留远程 MCP，避免一个来源失败拖垮全部工具。
            managers.append(_FailedMCPManager("local", error))

    if enable_external:
        config_path = Path(external_config_path or root / "tools" / "external_mcp_servers.json")
        managers.append(ExternalMCPManager(config_path))

    return MCPAggregator(managers)


create_aggregate_mcp_manager = create_aggregate_mcp_client
