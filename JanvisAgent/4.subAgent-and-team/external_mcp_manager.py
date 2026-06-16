# encoding: utf-8
import asyncio
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


# 支持在配置文件中用 ${ENV_NAME} 引用本机环境变量，避免把 API Key 写入仓库。
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# OpenAI function tool 名称只保留字母、数字、下划线和短横线。
_OPENAI_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]")
_TOOL_CALL_MAX_ATTEMPTS = 3
_TOOL_CALL_RETRY_DELAY_SECONDS = 0.8
_DEFAULT_EXTERNAL_TOOL_TIMEOUT_SECONDS = 30
_EXTERNAL_TOOL_TIMEOUT_SECONDS = {
    ("exa", "web_search_exa"): 30,
    ("exa", "web_fetch_exa"): 60,
    ("tavily", "tavily_search"): 30,
    ("tavily", "tavily_extract"): 60,
    ("tavily", "tavily_crawl"): 90,
    ("tavily", "tavily_map"): 60,
    ("tavily", "tavily_research"): 120,
}
_TAVILY_SEARCH_REST_URL = "https://api.tavily.com/search"
_NOISY_MCP_TRANSPORT_LOGGERS = ("mcp.client.streamable_http",)
_RETRYABLE_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "SSLEOFError",
    "TimeoutError",
    "EndOfStream",
}


def _suppress_noisy_mcp_transport_logs() -> None:
    for logger_name in _NOISY_MCP_TRANSPORT_LOGGERS:
        logger = logging.getLogger(logger_name)
        # MCP SDK 会在内部重试/清理时把 transport 异常栈直接打到控制台；调用方已把错误包装进工具结果。
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False


_suppress_noisy_mcp_transport_logs()


class ExternalMCPToolTimeoutError(TimeoutError):
    """外部 MCP 工具超过 Agent 侧硬超时。"""


@dataclass
class ExternalMCPServerConfig:
    """单个外部 MCP Server 的运行配置。"""
    name: str
    enabled: bool = True
    transport: str = "streamable_http"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    tool_prefix: str | None = None
    timeout_seconds: float = 30

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ExternalMCPServerConfig":
        # 配置文件字段统一在这里归一化，后续连接逻辑只处理强类型配置对象。
        return cls(
            name=name,
            enabled=bool(data.get("enabled", True)),
            transport=str(data.get("transport", "streamable_http")),
            url=data.get("url"),
            headers=dict(data.get("headers") or {}),
            command=data.get("command"),
            args=list(data.get("args") or []),
            env=dict(data.get("env") or {}),
            cwd=data.get("cwd"),
            tool_prefix=data.get("tool_prefix") or name,
            timeout_seconds=float(data.get("timeout_seconds", 30)),
        )


@dataclass
class _ToolRoute:
    """记录公开工具名到远端原始工具名的路由关系。"""

    server: ExternalMCPServerConfig
    raw_name: str


class ExternalMCPManager:
    """管理官方 MCP 协议的远程或第三方 MCP Server。"""

    def __init__(
            self,
            config_path: str | Path,
            *,
            use_cache: bool = True,
            refresh_tools: bool | None = None,
    ):
        self.mode = "external"
        self.config_path = Path(config_path)
        self.servers = self._load_server_configs()
        self.use_cache = use_cache
        self.refresh_tools = self._resolve_refresh_tools(refresh_tools)
        self.cache_path = self._default_cache_path()
        # _tools 是暴露给 Agent/ToolManager 的 OpenAI 兼容工具描述。
        self._tools: list[dict[str, Any]] = []
        # _tool_routes 保存公开工具名到真实 MCP Server + 原始工具名的映射。
        self._tool_routes: dict[str, _ToolRoute] = {}
        self._tool_counts_by_server: dict[str, int] = {}
        # 单个远程来源加载失败时记录错误，但不阻断其它来源。
        self._start_errors: dict[str, str] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self._tools = []
        self._tool_routes = {}
        self._tool_counts_by_server = {}
        self._start_errors = {}

        enabled_servers = [server for server in self.servers if server.enabled]
        if enabled_servers:
            max_workers = min(4, len(enabled_servers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._load_server_tools_sync, server): server
                    for server in enabled_servers
                }
                for future in as_completed(futures):
                    server, tools, error, should_cache = future.result()
                    if should_cache:
                        self._save_cached_server_tools(server, tools)
                    if error:
                        self._start_errors[server.name] = error
                    self._register_server_tools(server, tools)

        self._started = True

    def close(self) -> None:
        self._started = False
        self._tools = []
        self._tool_routes = {}
        self._tool_counts_by_server = {}

    def ping(self) -> dict[str, Any]:
        self.start()
        return {
            "message": "pong",
            "remote_tool_count": len(self._tools),
            "errors": dict(self._start_errors),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        self.start()
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.start()
        route = self._tool_routes.get(name)
        if route is None:
            raise ValueError(f"Unknown external MCP tool '{name}'")
        # Agent 传入的是公开工具名，真正调用远端时必须还原成该 MCP Server 的原始工具名。
        last_error: Exception | None = None
        started_at = time.perf_counter()
        timeout_seconds = self._tool_timeout_seconds(route.server, route.raw_name)
        for attempt in range(1, _TOOL_CALL_MAX_ATTEMPTS + 1):
            try:
                result = self._run_async(
                    self._call_tool_on_server_with_timeout(
                        route.server,
                        route.raw_name,
                        arguments,
                        timeout_seconds,
                    )
                )
                return self._attach_call_metadata(result, started_at, attempt, timeout_seconds)
            except Exception as error:
                last_error = error
                if isinstance(error, ExternalMCPToolTimeoutError):
                    break
                if attempt >= _TOOL_CALL_MAX_ATTEMPTS or not self._is_retryable_external_error(error):
                    break
                # 远程 MCP 的 streamable HTTP 连接偶发被提前关闭时，短暂等待后重建 session。
                time.sleep(_TOOL_CALL_RETRY_DELAY_SECONDS * attempt)
        if (
                last_error is not None
                and route.server.name == "tavily"
                and route.raw_name == "tavily_search"
                and (
                    isinstance(last_error, ExternalMCPToolTimeoutError)
                    or self._is_retryable_external_error(last_error)
                )
        ):
            try:
                result = self._call_tavily_search_rest(route.server, arguments, last_error)
                return self._attach_call_metadata(result, started_at, attempt, timeout_seconds)
            except Exception as fallback_error:
                return self._format_call_error(
                    route.server,
                    route.raw_name,
                    fallback_error,
                    attempts=attempt,
                    fallback_error=last_error,
                    started_at=started_at,
                    timeout_seconds=timeout_seconds,
                )
        return self._format_call_error(
            route.server,
            route.raw_name,
            last_error or RuntimeError("External MCP tool call failed"),
            attempts=attempt,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
        )

    def get_tool_counts(self) -> dict[str, int]:
        self.start()
        return {"local": 0, "remote": len(self._tools), **self._tool_counts_by_server}

    def get_tool_capabilities(self) -> dict[str, dict[str, Any]]:
        self.start()
        # 当前接入的 Exa/Tavily 搜索工具都按只读网络工具处理，供调度器安全并发。
        return {
            tool["name"]: {
                "is_concurrency_safe": True,
                "is_read_only": True,
                "is_destructive": False,
                "side_effect_scope": "network",
                "timeout_seconds": self._tool_timeout_seconds(
                    self._tool_routes[tool["name"]].server,
                    self._tool_routes[tool["name"]].raw_name,
                ),
            }
            for tool in self._tools
        }

    def get_start_errors(self) -> dict[str, str]:
        self.start()
        return dict(self._start_errors)

    def list_tools_by_server(self, server_name: str) -> list[dict[str, Any]]:
        self.start()
        return [
            tool
            for tool in self._tools
            if self._tool_routes.get(str(tool.get("name", "")), None)
            and self._tool_routes[str(tool.get("name", ""))].server.name == server_name
        ]

    def _load_server_tools_sync(
            self,
            server: ExternalMCPServerConfig,
    ) -> tuple[ExternalMCPServerConfig, list[dict[str, Any]], str | None, bool]:
        try:
            self._resolve_runtime_config(server)
        except Exception as error:
            return server, [], self._summarize_exception(error), False

        cached_tools = self._load_cached_server_tools(server) if self.use_cache else []
        if cached_tools and not self.refresh_tools:
            return server, cached_tools, None, False

        try:
            tools = self._run_async(self._list_tools_from_server(server))
            return server, tools, None, True
        except Exception as error:
            error_message = self._summarize_exception(error)
            if cached_tools:
                return server, cached_tools, f"{error_message}; using cached tool schema", False
            return server, [], error_message, False

    def _register_server_tools(self, server: ExternalMCPServerConfig, tools: list[dict[str, Any]]) -> None:
        exposed_count = 0
        for tool in tools:
            raw_name = str(tool.get("name", "")).strip()
            if not raw_name:
                continue
            # 给远程工具加来源前缀，避免 Exa/Tavily/本地 MCP 之间出现同名工具。
            public_name = self._public_tool_name(server, raw_name)
            if public_name in self._tool_routes:
                self._start_errors[f"{server.name}:{public_name}"] = "duplicate external MCP tool name"
                continue
            self._tool_routes[public_name] = _ToolRoute(server=server, raw_name=raw_name)
            self._tools.append(
                {
                    "name": public_name,
                    "description": self._format_description(server, tool),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            )
            exposed_count += 1
        self._tool_counts_by_server[server.name] = exposed_count

    def _load_server_configs(self) -> list[ExternalMCPServerConfig]:
        if not self.config_path.exists():
            return []
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        servers = config.get("servers", {}) if isinstance(config, dict) else {}
        result: list[ExternalMCPServerConfig] = []
        for name, data in servers.items():
            if isinstance(data, dict):
                result.append(ExternalMCPServerConfig.from_dict(name, data))
        return result

    def _resolve_refresh_tools(self, refresh_tools: bool | None) -> bool:
        if refresh_tools is not None:
            return bool(refresh_tools)
        return os.environ.get("EXTERNAL_MCP_REFRESH_TOOLS", "").strip().lower() in {"1", "true", "yes"}

    def _default_cache_path(self) -> Path:
        if self.config_path.parent.name == "tools":
            return self.config_path.parent.parent / "cache" / "external_mcp_tools_cache.json"
        return self.config_path.parent / "external_mcp_tools_cache.json"

    def _server_cache_key(self, server: ExternalMCPServerConfig) -> str:
        endpoint = server.url or server.command or ""
        return f"{server.name}|{server.transport}|{endpoint}"

    def _read_tool_cache(self) -> dict[str, Any]:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_tool_cache(self, cache: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_cached_server_tools(self, server: ExternalMCPServerConfig) -> list[dict[str, Any]]:
        cache = self._read_tool_cache()
        entry = cache.get(self._server_cache_key(server))
        if not isinstance(entry, dict):
            return []
        tools = entry.get("tools")
        if not isinstance(tools, list):
            return []
        return [tool for tool in tools if isinstance(tool, dict)]

    def _save_cached_server_tools(self, server: ExternalMCPServerConfig, tools: list[dict[str, Any]]) -> None:
        cache = self._read_tool_cache()
        cache[self._server_cache_key(server)] = {
            "server": server.name,
            "transport": server.transport,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tools": tools,
        }
        self._write_tool_cache(cache)

    @asynccontextmanager
    async def _open_session(
            self,
            server: ExternalMCPServerConfig,
            timeout_seconds: float | None = None,
    ) -> AsyncIterator[ClientSession]:
        # 每次连接前解析环境变量，保证用户运行前新设置的 API Key 能立即生效。
        runtime_server = self._resolve_runtime_config(server)
        request_timeout = timeout_seconds or runtime_server.timeout_seconds
        if runtime_server.transport in {"streamable_http", "http"}:
            if not runtime_server.url:
                raise ValueError(f"External MCP server '{server.name}' missing url")
            # 远程 MCP 走官方 streamable HTTP transport。
            http_client = create_mcp_http_client(
                headers=runtime_server.headers or None,
                timeout=httpx.Timeout(request_timeout, read=request_timeout),
            )
            async with http_client:
                async with streamable_http_client(
                    runtime_server.url,
                    http_client=http_client,
                    terminate_on_close=False,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
            return

        if runtime_server.transport == "stdio":
            if not runtime_server.command:
                raise ValueError(f"External MCP server '{server.name}' missing command")
            # 兼容后续接入本机 npm/python 启动的第三方 MCP Server。
            params = StdioServerParameters(
                command=runtime_server.command,
                args=runtime_server.args,
                env=runtime_server.env or None,
                cwd=runtime_server.cwd,
                encoding="utf-8",
                encoding_error_handler="replace",
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session
            return

        raise ValueError(f"Unsupported external MCP transport '{server.transport}' for '{server.name}'")

    async def _list_tools_from_server(self, server: ExternalMCPServerConfig) -> list[dict[str, Any]]:
        async with self._open_session(server) as session:
            result = await session.list_tools()
        # 官方 MCP Tool 使用 inputSchema，这里转成当前 ToolManager 期待的 parameters 字段。
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in result.tools
        ]

    async def _call_tool_on_server(
            self,
            server: ExternalMCPServerConfig,
            raw_name: str,
            arguments: dict[str, Any],
            timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        async with self._open_session(server, timeout_seconds=timeout_seconds) as session:
            result = await session.call_tool(raw_name, arguments or {})

        # MCP 返回内容可能包含 text/image/resource，统一转成可 JSON 序列化的数据。
        content = [self._dump_content_item(item) for item in result.content]
        payload: dict[str, Any] = {
            "ok": not bool(result.isError),
            "server": server.name,
            "tool": raw_name,
            "content": content,
        }
        if result.isError:
            payload["error_type"] = "mcp_tool_error"
            payload["message"] = self._content_error_message(content)
        if result.structuredContent is not None:
            payload["structured_content"] = result.structuredContent
        return payload

    def _content_error_message(self, content: list[dict[str, Any]]) -> str:
        for item in content:
            text = item.get("text") or item.get("value")
            if text:
                return str(text).splitlines()[0][:500]
        return "MCP tool returned an error result"

    async def _call_tool_on_server_with_timeout(
            self,
            server: ExternalMCPServerConfig,
            raw_name: str,
            arguments: dict[str, Any],
            timeout_seconds: float,
    ) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._call_tool_on_server(server, raw_name, arguments, timeout_seconds=timeout_seconds),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as error:
            raise ExternalMCPToolTimeoutError(
                f"External MCP tool '{raw_name}' timed out after {timeout_seconds:g}s"
            ) from error

    def _call_tavily_search_rest(
            self,
            server: ExternalMCPServerConfig,
            arguments: dict[str, Any],
            mcp_error: BaseException,
    ) -> dict[str, Any]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required for Tavily REST fallback")

        allowed_fields = {
            "query",
            "max_results",
            "search_depth",
            "topic",
            "time_range",
            "include_answer",
            "include_images",
            "include_image_descriptions",
            "include_raw_content",
            "include_domains",
            "exclude_domains",
            "country",
            "include_favicon",
            "start_date",
            "end_date",
            "exact_match",
        }
        payload = {
            key: value
            for key, value in (arguments or {}).items()
            if key in allowed_fields and value is not None
        }
        if not payload.get("query"):
            raise ValueError("Tavily search requires query")

        response = self._post_tavily_search_rest(server, api_key, payload)
        data = response.json()
        # Tavily MCP streamable HTTP 在部分网络下会偶发断流；REST fallback 只在 MCP 传输失败后兜底。
        return {
            "ok": True,
            "server": server.name,
            "tool": "tavily_search",
            "transport": "tavily_rest_fallback",
            "mcp_error": self._summarize_exception(mcp_error),
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(data, ensure_ascii=False),
                }
            ],
            "structured_content": data,
        }

    def _post_tavily_search_rest(
            self,
            server: ExternalMCPServerConfig,
            api_key: str,
            payload: dict[str, Any],
    ) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        last_error: BaseException | None = None
        # 先尊重用户环境代理；代理链路 TLS 断流时，再尝试不读取环境代理的直连路径。
        for trust_env in (True, False):
            for attempt in range(1, _TOOL_CALL_MAX_ATTEMPTS + 1):
                try:
                    with httpx.Client(timeout=server.timeout_seconds, trust_env=trust_env) as client:
                        response = client.post(_TAVILY_SEARCH_REST_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    return response
                except httpx.HTTPStatusError:
                    raise
                except httpx.TransportError as error:
                    last_error = error
                    if attempt >= _TOOL_CALL_MAX_ATTEMPTS:
                        break
                    time.sleep(_TOOL_CALL_RETRY_DELAY_SECONDS * attempt)
        raise last_error or RuntimeError("Tavily REST fallback failed")

    def _resolve_runtime_config(self, server: ExternalMCPServerConfig) -> ExternalMCPServerConfig:
        missing_env: set[str] = set()
        # URL、headers、env 都允许使用 ${ENV_NAME}，缺失时统一收集后报错。
        url = self._expand_env_value(server.url, missing_env) if server.url else None
        headers = {
            key: self._expand_env_value(value, missing_env)
            for key, value in server.headers.items()
        }
        env = {
            key: self._expand_env_value(value, missing_env)
            for key, value in server.env.items()
        }
        if missing_env:
            names = ", ".join(sorted(missing_env))
            raise RuntimeError(f"External MCP server '{server.name}' missing environment variables: {names}")
        # stdio Server 的 env 需要继承当前环境，再叠加配置中的额外变量。
        return ExternalMCPServerConfig(
            name=server.name,
            enabled=server.enabled,
            transport=server.transport,
            url=url,
            headers=headers,
            command=server.command,
            args=server.args,
            env={**os.environ, **env} if env else None,
            cwd=server.cwd,
            tool_prefix=server.tool_prefix,
            timeout_seconds=server.timeout_seconds,
        )

    def _expand_env_value(self, value: str, missing_env: set[str]) -> str:
        def _replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            env_value = os.environ.get(env_name)
            if env_value is None:
                missing_env.add(env_name)
                return ""
            return env_value

        return _ENV_PATTERN.sub(_replace, str(value))

    def _public_tool_name(self, server: ExternalMCPServerConfig, raw_name: str) -> str:
        prefix = server.tool_prefix or server.name
        clean_raw_name = _OPENAI_TOOL_NAME_PATTERN.sub("_", raw_name)
        raw_name_lower = clean_raw_name.lower()
        prefix_lower = prefix.lower()
        provider_already_in_name = (
            raw_name_lower.startswith(f"{prefix_lower}_")
            or raw_name_lower.startswith(f"{prefix_lower}-")
            or raw_name_lower.startswith(f"{prefix_lower}__")
            or raw_name_lower.endswith(f"_{prefix_lower}")
            or raw_name_lower.endswith(f"-{prefix_lower}")
        )
        public_name = clean_raw_name if provider_already_in_name else f"{prefix}__{clean_raw_name}"
        # 清洗后的公开名称会直接进入 OpenAI tools schema。
        return _OPENAI_TOOL_NAME_PATTERN.sub("_", public_name)

    def _format_description(self, server: ExternalMCPServerConfig, tool: dict[str, Any]) -> str:
        description = str(tool.get("description") or "").strip()
        return f"[{server.name}] {description}" if description else f"[{server.name}] external MCP tool"

    def _dump_content_item(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump(by_alias=True, exclude_none=True)
        return {"type": type(item).__name__, "value": str(item)}

    def _tool_timeout_seconds(self, server: ExternalMCPServerConfig, raw_name: str) -> float:
        timeout_seconds = _EXTERNAL_TOOL_TIMEOUT_SECONDS.get(
            (server.name, raw_name),
            max(float(server.timeout_seconds or 0), _DEFAULT_EXTERNAL_TOOL_TIMEOUT_SECONDS),
        )
        return max(1.0, float(timeout_seconds))

    def _attach_call_metadata(
            self,
            result: Any,
            started_at: float,
            attempts: int,
            timeout_seconds: float,
    ) -> Any:
        if isinstance(result, dict):
            result.setdefault("elapsed_seconds", round(time.perf_counter() - started_at, 3))
            result.setdefault("attempts", attempts)
            result.setdefault("timeout_seconds", timeout_seconds)
        return result

    def _leaf_exception(self, error: BaseException, seen: set[int] | None = None) -> BaseException:
        seen = seen or set()
        if id(error) in seen:
            return error
        seen.add(id(error))
        if isinstance(error, ExternalMCPToolTimeoutError):
            return error
        if isinstance(error, BaseExceptionGroup) and error.exceptions:
            return self._leaf_exception(error.exceptions[0], seen)
        cause = getattr(error, "__cause__", None)
        if cause is not None:
            return self._leaf_exception(cause, seen)
        context = getattr(error, "__context__", None)
        if context is not None and type(error).__name__ in _RETRYABLE_ERROR_NAMES:
            return self._leaf_exception(context, seen)
        return error

    def _is_retryable_external_error(self, error: BaseException) -> bool:
        # 只重试网络/传输层瞬断，避免把参数错误、认证错误这类确定性失败重复打到服务端。
        for current in self._walk_exception_tree(error):
            if type(current).__name__ in _RETRYABLE_ERROR_NAMES:
                return True
            if type(current).__name__ == "RuntimeError" and str(current).strip() == "no running event loop":
                return True
        return False

    def _walk_exception_tree(self, error: BaseException, seen: set[int] | None = None):
        seen = seen or set()
        if id(error) in seen:
            return
        seen.add(id(error))
        yield error
        if isinstance(error, BaseExceptionGroup):
            for child in error.exceptions:
                yield from self._walk_exception_tree(child, seen)
        for linked_error in (getattr(error, "__cause__", None), getattr(error, "__context__", None)):
            if linked_error is not None:
                yield from self._walk_exception_tree(linked_error, seen)

    def _summarize_exception(self, error: BaseException) -> str:
        leaf = self._leaf_exception(error)
        message = str(leaf).strip() or type(leaf).__name__
        if leaf is error:
            return f"{type(leaf).__name__}: {message}"
        return f"{type(error).__name__} -> {type(leaf).__name__}: {message}"

    def _format_call_error(
            self,
            server: ExternalMCPServerConfig,
            raw_name: str,
            error: BaseException,
            *,
            attempts: int = 1,
            fallback_error: BaseException | None = None,
            started_at: float | None = None,
            timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        leaf = self._leaf_exception(error)
        message = str(leaf).strip() or type(leaf).__name__
        is_timeout = isinstance(leaf, ExternalMCPToolTimeoutError)
        payload = {
            "ok": False,
            "server": server.name,
            "tool": raw_name,
            "error_type": "tool_timeout" if is_timeout else type(leaf).__name__,
            "retryable": bool(is_timeout or self._is_retryable_external_error(error)),
            "message": message,
            "detail": self._summarize_exception(error),
            "attempts": attempts,
        }
        if started_at is not None:
            payload["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        if timeout_seconds is not None:
            payload["timeout_seconds"] = timeout_seconds
        if fallback_error is not None:
            payload["mcp_error"] = self._summarize_exception(fallback_error)
        return payload

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # 当前 Agent 工具调用链是同步的；如果未来切到异步 Agent，需要把这里改成异步接口。
        raise RuntimeError("ExternalMCPManager cannot run inside an existing asyncio event loop")
