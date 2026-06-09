import atexit
import glob as glob_module
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Callable
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import ToolSpec, FunctionTool
from tools.tool_registry import ToolRegistry
from tools.tool_names import ToolNameConstant
from tools.web_tool import DDGS, WebTool


@dataclass
class ToolManagerConfig:
    """ToolManager 的静态配置。"""

    # 项目根目录，用于定位工具定义文件与相对路径资源。
    project_root: str
    # 当前模型客户端实例，web_search 总结阶段会复用该客户端。
    client: Any
    # 当前 Agent 使用的模型名称。
    model: str
    # 当前 Agent 使用的温度参数。
    temperature: float
    # 是否是主 Agent，用于决定是否暴露 SUB_AGENT 等能力。
    is_main_agent: bool = True
    # Spinner 创建函数，由 Agent 注入；普通测试场景可保持为空。
    spinner_factory: Callable[..., Any] | None = None


@dataclass
class AgentToolHandlers:
    """由 Agent 注入的工具处理回调。"""

    # 计划工具处理函数，对应 MAKE_PLAN。
    make_plan_handler: Callable[..., Any] | None = None
    # 读取技能详情处理函数，对应 LOAD_SKILL_DETAIL_BY_NAME。
    load_skill_detail_handler: Callable[..., Any] | None = None
    # 读取完整记忆处理函数，对应 LOAD_FULL_MEMORY_CONTEXT。
    load_full_memory_context_handler: Callable[..., Any] | None = None
    # 子 Agent 调用处理函数，对应 SUB_AGENT（仅主 Agent 注入）。
    sub_agent_handler: Callable[..., Any] | None = None


@dataclass
class BackgroundCommandInfo:
    service_id: str
    command: str
    working_dir: Path
    process: subprocess.Popen
    started_at: float
    output_chunks: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ToolManager:
    """统一管理本地工具、MCP工具以及工具调用分发。"""

    # Bash 命令黑名单：只放明确高风险、不可自动执行的模式。
    # 这里同时覆盖 Linux/macOS 常见危险命令和 Windows 批量删除命令。
    _BASH_DANGEROUS_PATTERNS: tuple[str, ...] = (
        r'\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*--no-preserve-root)',
        r'\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+)?/',
        r'\bmkfs\b',
        r'\bdd\s+.*of\s*=\s*/dev/',
        r'>\s*/dev/sd[a-z]',
        r'\bchmod\s+(-R\s+)?777\s+/',
        r':\(\)\s*\{',
        r'\bcurl\b.*\|\s*(ba)?sh',
        r'\bwget\b.*\|\s*(ba)?sh',
        r'\bshutdown\b',
        r'\breboot\b',
        r'\bdd\s+if\s*=',
        r'\bdel\s+/s\b',
        r'\brd\s+/s\b',
        r'\brmdir\s+/s\b',
        r'\bRemove-Item\b',
        r'\brm\s+-rf\b',
    )
    # 默认不弹出交互确认，避免子 Agent 批量执行任务时阻塞。
    # 如需人工确认，可在实例或类上改为 False。
    _BASH_AUTO_APPROVE = True
    # 目前默认不启用输出截断，保留给后续控制超长输出时使用。
    _BASH_MAX_OUTPUT_LENGTH = 5000
    # Bash 命令可能启动常驻服务，超过该时间就终止并把已收集输出返回给模型。
    _BASH_TIMEOUT_SECONDS = 60
    # 终止后只短暂等待读流线程收尾，避免工具再次卡在清理阶段。
    _BASH_TERMINATE_GRACE_SECONDS = 5
    # 后台服务输出只保留尾部，避免常驻服务长时间运行撑爆内存。
    _BACKGROUND_MAX_OUTPUT_LENGTH = 20000
    # 并发安全工具集合：允许并发执行，且不会产生状态冲突。
    _CONCURRENCY_SAFE_TOOLS = {
        ToolNameConstant.GET_TIME,
        ToolNameConstant.WEB_SEARCH,
        ToolNameConstant.GLOB,
        ToolNameConstant.GREP,
        ToolNameConstant.READ_FILE,
        ToolNameConstant.LIST_DIR,
        ToolNameConstant.CHECK_BACKGROUND_COMMAND,
        ToolNameConstant.READ_BACKGROUND_COMMAND_OUTPUT,
    }
    # 只读工具集合：不修改外部状态，可用于批处理并发判定。
    _READ_ONLY_TOOLS = {
        ToolNameConstant.GET_TIME,
        ToolNameConstant.WEB_SEARCH,
        ToolNameConstant.GLOB,
        ToolNameConstant.GREP,
        ToolNameConstant.READ_FILE,
        ToolNameConstant.LIST_DIR,
        ToolNameConstant.LOAD_SKILL_DETAIL_BY_NAME,
        ToolNameConstant.LOAD_FULL_MEMORY_CONTEXT,
        ToolNameConstant.CHECK_BACKGROUND_COMMAND,
        ToolNameConstant.READ_BACKGROUND_COMMAND_OUTPUT,
    }
    # 破坏性工具集合：存在写入、覆盖、执行高风险命令等副作用。
    _DESTRUCTIVE_TOOLS = {
        ToolNameConstant.WRITE_FILE,
        ToolNameConstant.EDIT,
        ToolNameConstant.EXECUTE_BASH,
        ToolNameConstant.START_BACKGROUND_COMMAND,
        ToolNameConstant.STOP_BACKGROUND_COMMAND,
    }
    # 工具副作用作用域：V2 调度按作用域做批次隔离。
    _TOOL_SIDE_EFFECT_SCOPE = {
        ToolNameConstant.READ_FILE: "filesystem",
        ToolNameConstant.WRITE_FILE: "filesystem",
        ToolNameConstant.EDIT: "filesystem",
        ToolNameConstant.GLOB: "filesystem",
        ToolNameConstant.GREP: "filesystem",
        ToolNameConstant.LIST_DIR: "filesystem",
        ToolNameConstant.WEB_SEARCH: "network",
        ToolNameConstant.GET_TIME: "runtime",
        ToolNameConstant.EXECUTE_BASH: "system",
        ToolNameConstant.START_BACKGROUND_COMMAND: "system",
        ToolNameConstant.CHECK_BACKGROUND_COMMAND: "system",
        ToolNameConstant.READ_BACKGROUND_COMMAND_OUTPUT: "system",
        ToolNameConstant.STOP_BACKGROUND_COMMAND: "system",
        ToolNameConstant.LOAD_SKILL_DETAIL_BY_NAME: "memory",
        ToolNameConstant.LOAD_FULL_MEMORY_CONTEXT: "memory",
        ToolNameConstant.MAKE_PLAN: "runtime",
        ToolNameConstant.SUB_AGENT: "agent",
    }
    # MCP 默认能力（保守策略）：未配置时按不可并发读处理。
    _MCP_DEFAULT_CAPABILITY = {
        "is_concurrency_safe": False,
        "is_read_only": False,
        "is_destructive": False,
        "side_effect_scope": "external",
    }

    def __init__(self, *, config: ToolManagerConfig, handlers: AgentToolHandlers, mcp_client=None):
        # 基础上下文依赖：目录、模型调用客户端、模型配置
        self.project_root = config.project_root
        self.client = config.client
        self.model = config.model
        self.temperature = config.temperature
        self.mcp_client = mcp_client
        self.is_main_agent = config.is_main_agent
        self.web_tool = WebTool(
            client=self.client,
            model=self.model,
            spinner_factory=config.spinner_factory,
        )

        # 通过依赖注入方式注册“由 Agent 提供”的工具处理函数，
        # 这样 ToolManager 不需要直接依赖 Agent 具体实现。
        self._make_plan_handler = handlers.make_plan_handler
        self._load_skill_detail_handler = handlers.load_skill_detail_handler
        self._load_full_memory_context_handler = handlers.load_full_memory_context_handler
        self._sub_agent_handler = handlers.sub_agent_handler

        # Bash hook 管道：before hook 可拦截；after hook 可做输出后处理。
        self._bash_before_hooks = [
            self._bash_check_dangerous_command,
            self._bash_ask_user_confirmation,
            self._bash_log_command,
        ]
        self._bash_after_hooks = [self._bash_log_result]
        # 本地工具注册中心：用统一抽象管理工具定义与执行逻辑。
        self._local_tool_registry = ToolRegistry()
        # MCP 工具能力配置：由配置文件覆盖默认能力，供调度判定使用。
        self._mcp_capabilities = self._load_mcp_capabilities()

        # 统一构建工具注册表，供 Agent 直接透传使用。
        self._local_tools_raw = self._load_local_tools()
        self._build_local_functions()
        self.local_tools = self._local_tool_registry.as_openai_tools()
        self.local_functions = self._local_tool_registry.as_function_map()
        self.mcp_tools = self._load_mcp_tools() if self.mcp_client else []
        self.available_functions = self._build_available_functions()
        self.all_tools = self.local_tools + self.mcp_tools
        self.last_web_search_results: dict[str, Any] | None = None
        self._background_commands: dict[str, BackgroundCommandInfo] = {}
        self._background_lock = threading.RLock()
        atexit.register(self.stop_all_background_commands)

    def _build_local_functions(self) -> None:
        # 本地基础工具：文件、bash、检索、时间、联网搜索。
        functions = {
            ToolNameConstant.EXECUTE_BASH: self.execute_bash,
            ToolNameConstant.START_BACKGROUND_COMMAND: self.start_background_command,
            ToolNameConstant.CHECK_BACKGROUND_COMMAND: self.check_background_command,
            ToolNameConstant.READ_BACKGROUND_COMMAND_OUTPUT: self.read_background_command_output,
            ToolNameConstant.STOP_BACKGROUND_COMMAND: self.stop_background_command,
            ToolNameConstant.READ_FILE: self.read_file,
            ToolNameConstant.WRITE_FILE: self.write_file,
            ToolNameConstant.EDIT: self.edit,
            ToolNameConstant.GLOB: self.glob,
            ToolNameConstant.GREP: self.grep,
            ToolNameConstant.GET_TIME: self.get_time,
            ToolNameConstant.WEB_SEARCH: self.web_search,
            ToolNameConstant.LIST_DIR: self.list_dir,
        }
        if self._make_plan_handler is not None:
            functions[ToolNameConstant.MAKE_PLAN] = self._make_plan_handler
        if self._load_skill_detail_handler is not None:
            functions[ToolNameConstant.LOAD_SKILL_DETAIL_BY_NAME] = self._load_skill_detail_handler
        if self._load_full_memory_context_handler is not None:
            functions[ToolNameConstant.LOAD_FULL_MEMORY_CONTEXT] = self._load_full_memory_context_handler
        if self.is_main_agent and self._sub_agent_handler is not None:
            functions[ToolNameConstant.SUB_AGENT] = self._sub_agent_handler

        # 基于 local_tools.json 的 schema 生成 ToolSpec，并绑定到具体实现。
        schema_map = {
            tool["function"]["name"]: tool["function"]
            for tool in self._local_tools_raw
            if tool.get("type") == "function" and tool.get("function")
        }
        for name, handler in functions.items():
            fn_meta = schema_map.get(name, {})
            spec = ToolSpec(
                name=name,
                description=fn_meta.get("description", ""),
                input_schema=fn_meta.get("parameters", {"type": "object", "properties": {}}),
                output_schema={"type": "string"},
                is_concurrency_safe=name in self._CONCURRENCY_SAFE_TOOLS,
                is_read_only=name in self._READ_ONLY_TOOLS,
                is_destructive=name in self._DESTRUCTIVE_TOOLS,
                side_effect_scope=self._TOOL_SIDE_EFFECT_SCOPE.get(name, "none"),
            )
            self._local_tool_registry.register(FunctionTool(spec, handler))

    def _build_available_functions(self) -> dict[str, Callable[..., Any]]:
        # 可调用函数 = 本地工具 + 动态包装后的 MCP 工具。
        available = dict(self.local_functions)
        for tool in self.mcp_tools:
            tool_name = tool["function"]["name"]
            available[tool_name] = self._make_mcp_executor(tool_name)
        return available

    def is_parallel_read_tool(self, tool_name: str) -> bool:
        """
        判断工具是否满足“只读 + 并发安全”条件。
        说明：V1 对未知工具（尤其 MCP 工具）采取保守策略，默认 False。
        """
        tool = self._local_tool_registry.get(tool_name)
        if tool is not None:
            return tool.spec.is_read_only and tool.spec.is_concurrency_safe
        capability = self._mcp_capabilities.get(tool_name, self._MCP_DEFAULT_CAPABILITY)
        return bool(capability.get("is_read_only")) and bool(capability.get("is_concurrency_safe"))

    def get_tool_runtime_profile(self, tool_name: str) -> dict[str, Any]:
        """
        获取工具运行画像（V2 调度使用）。
        包含：只读、并发安全、破坏性、作用域。
        """
        tool = self._local_tool_registry.get(tool_name)
        if tool is not None:
            return {
                "is_read_only": tool.spec.is_read_only,
                "is_concurrency_safe": tool.spec.is_concurrency_safe,
                "is_destructive": tool.spec.is_destructive,
                "side_effect_scope": tool.spec.side_effect_scope,
            }
        capability = self._mcp_capabilities.get(tool_name, self._MCP_DEFAULT_CAPABILITY)
        return {
            "is_read_only": bool(capability.get("is_read_only")),
            "is_concurrency_safe": bool(capability.get("is_concurrency_safe")),
            "is_destructive": bool(capability.get("is_destructive")),
            "side_effect_scope": str(capability.get("side_effect_scope", "external")),
        }

    def _load_mcp_capabilities(self) -> dict[str, dict[str, Any]]:
        """
        加载 MCP 工具能力配置（Step A）。
        规则：
        1) 先给所有 MCP 工具注入保守默认能力；
        2) 再用配置文件进行覆盖。
        """
        capabilities: dict[str, dict[str, Any]] = {}
        # 先填充默认能力，确保未知工具不会误判为可并发。
        if self.mcp_client:
            try:
                for tool in self.mcp_client.list_tools():
                    tool_name = tool.get("name")
                    if tool_name:
                        capabilities[tool_name] = dict(self._MCP_DEFAULT_CAPABILITY)
            except Exception:
                # 能力配置加载失败不影响主流程，保持默认空字典即可。
                return capabilities

        config_path = os.path.join(self.project_root, "tools", "mcp_tool_capabilities.json")
        if not os.path.exists(config_path):
            return capabilities
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                custom = json.load(f)
            if isinstance(custom, dict):
                for name, capability in custom.items():
                    if not isinstance(capability, dict):
                        continue
                    base = dict(self._MCP_DEFAULT_CAPABILITY)
                    base.update(capability)
                    capabilities[name] = base
        except Exception:
            return capabilities
        return capabilities

    def _load_local_tools(self) -> list[dict[str, Any]]:
        # 从 local_tools.json 读取 OpenAI function calling 需要的工具 schema。
        tools_path = os.path.join(self.project_root, "tools", "local_tools.json")
        with open(tools_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_mcp_tools(self) -> list[dict[str, Any]]:
        # 拉取远端 MCP 工具并转换为 OpenAI 兼容格式。
        mcp_tools = self.mcp_client.list_tools()
        tools_in_openai_format = []
        for tool in mcp_tools:
            tools_in_openai_format.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return tools_in_openai_format

    def _make_mcp_executor(self, tool_name: str):
        # 为每个 MCP 工具生成闭包执行器，统一参数入口为 kwargs。
        def _executor(**kwargs):
            return self.mcp_client.call_tool(tool_name, kwargs)

        return _executor

    def execute_bash(self, command, working_dir=None, cwd=None):
        # 对外入口保持简单，安全检查和日志由 hook 管道负责。
        return self._execute_bash_with_hooks(command, working_dir=working_dir, cwd=cwd)

    def start_background_command(self, command, service_id=None, startup_wait_seconds=2, working_dir=None, cwd=None):
        blocked = self._run_background_before_hooks(
            command,
            ToolNameConstant.START_BACKGROUND_COMMAND,
        )
        if blocked:
            return blocked
        try:
            resolved_working_dir = self._resolve_working_dir(working_dir=working_dir, cwd=cwd)
        except ValueError as error:
            return self._json_tool_result(
                {
                    "ok": False,
                    "status": "invalid_working_dir",
                    "error": str(error),
                }
            )
        normalized_service_id = self._normalize_background_service_id(service_id)
        with self._background_lock:
            existing = self._background_commands.get(normalized_service_id)
            if existing and self._background_status(existing) == "running":
                return self._json_tool_result(
                    {
                        "ok": False,
                        "service_id": normalized_service_id,
                        "status": "running",
                        "error": "service_id_already_running",
                    }
                )
        try:
            process = self._open_shell_process(command, working_dir=resolved_working_dir)
        except Exception as error:
            return self._json_tool_result(
                {
                    "ok": False,
                    "service_id": normalized_service_id,
                    "status": "failed_to_start",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        info = BackgroundCommandInfo(
            service_id=normalized_service_id,
            command=command,
            working_dir=resolved_working_dir,
            process=process,
            started_at=time.time(),
        )
        with self._background_lock:
            self._background_commands[normalized_service_id] = info
        self._start_background_output_readers(info)
        wait_seconds = self._clamp_float(startup_wait_seconds, 0, 10, default=2)
        if wait_seconds:
            time.sleep(wait_seconds)
        snapshot = self._background_snapshot(info, tail_chars=4000)
        snapshot["ok"] = True
        print(f"[Tool after] {ToolNameConstant.START_BACKGROUND_COMMAND}: {normalized_service_id} {snapshot['status']}")
        return self._json_tool_result(snapshot)

    def check_background_command(self, service_id):
        info = self._get_background_command(service_id)
        if info is None:
            return self._json_tool_result(
                {
                    "ok": False,
                    "service_id": service_id,
                    "status": "not_found",
                    "error": "Background command not found.",
                }
            )
        snapshot = self._background_snapshot(info, tail_chars=1000)
        snapshot["ok"] = True
        return self._json_tool_result(snapshot)

    def read_background_command_output(self, service_id, tail_chars=4000):
        info = self._get_background_command(service_id)
        if info is None:
            return self._json_tool_result(
                {
                    "ok": False,
                    "service_id": service_id,
                    "status": "not_found",
                    "error": "Background command not found.",
                }
            )
        snapshot = self._background_snapshot(
            info,
            tail_chars=self._clamp_int(tail_chars, 1, self._BACKGROUND_MAX_OUTPUT_LENGTH, default=4000),
        )
        snapshot["ok"] = True
        return self._json_tool_result(snapshot)

    def stop_background_command(self, service_id):
        info = self._get_background_command(service_id)
        if info is None:
            return self._json_tool_result(
                {
                    "ok": False,
                    "service_id": service_id,
                    "status": "not_found",
                    "error": "Background command not found.",
                }
            )
        if self._background_status(info) == "running":
            self._terminate_bash_process_tree(info.process)
            self._close_bash_streams(info.process)
            time.sleep(0.1)
        snapshot = self._background_snapshot(info, tail_chars=4000)
        snapshot["ok"] = True
        print(f"[Tool after] {ToolNameConstant.STOP_BACKGROUND_COMMAND}: {info.service_id} {snapshot['status']}")
        return self._json_tool_result(snapshot)

    def stop_all_background_commands(self) -> None:
        with self._background_lock:
            infos = list(self._background_commands.values())
        for info in infos:
            if self._background_status(info) != "running":
                continue
            self._terminate_bash_process_tree(info.process)
            self._close_bash_streams(info.process)

    def set_bash_auto_approve(self, enabled: bool) -> None:
        """
        设置 Bash 工具执行前是否自动确认。
        :param enabled: True 表示自动确认，False 表示执行前人工确认
        :return:
        """
        # Bash 安全策略状态统一由 ToolManager 持有，Agent 只负责展示和命令入口。
        self._BASH_AUTO_APPROVE = bool(enabled)

    def bash_approve_status_text(self) -> str:
        """
        返回 Bash 执行确认策略的展示文案。
        :return:
        """
        if self._BASH_AUTO_APPROVE:
            return "自动确认（无需手动确认）"
        return "手动确认（每次需确认）"

    def _resolve_working_dir(self, working_dir: Any = None, cwd: Any = None) -> Path:
        raw_dir = working_dir if working_dir not in (None, "") else cwd
        if raw_dir in (None, ""):
            default_dir = Path(self.project_root) / "runtime_output"
            default_dir.mkdir(parents=True, exist_ok=True)
            return default_dir
        path = Path(str(raw_dir)).expanduser()
        if not path.is_absolute():
            path = Path(self.project_root) / path
        path = path.resolve()
        if not path.exists():
            raise ValueError(f"Working directory does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Working directory is not a directory: {path}")
        return path

    def _open_shell_process(self, command: str, working_dir: Path | None = None) -> subprocess.Popen:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        resolved_working_dir = working_dir or self._resolve_working_dir()
        popen_kwargs = {
            "shell": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
            "start_new_session": os.name != "nt",
            "env": env,
            "cwd": str(resolved_working_dir),
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return subprocess.Popen(command, **popen_kwargs)

    def _decode_output_bytes(self, data: bytes) -> str:
        # Windows shell 输出可能是 GBK/GB18030，Python 子进程和现代工具多为 UTF-8。
        for enc in ("utf-8", "gb18030", "gbk"):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def _run_background_before_hooks(self, command: str, tool_name: str) -> str | None:
        args = {"command": command}
        for hook in (self._bash_check_dangerous_command, self._bash_ask_user_confirmation):
            blocked, message = hook(args)
            if blocked:
                return message or "Background command was blocked."
        print(f"[Tool before] {tool_name}: {command}")
        return None

    def _normalize_background_service_id(self, service_id: Any = None) -> str:
        raw = str(service_id or "").strip()
        if not raw:
            raw = f"svc_{uuid.uuid4().hex[:8]}"
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw).strip("._-")
        return normalized or f"svc_{uuid.uuid4().hex[:8]}"

    def _get_background_command(self, service_id: Any) -> BackgroundCommandInfo | None:
        normalized_service_id = self._normalize_background_service_id(service_id)
        with self._background_lock:
            return self._background_commands.get(normalized_service_id)

    def _start_background_output_readers(self, info: BackgroundCommandInfo) -> None:
        def _consume_stream(stream):
            if stream is None:
                return
            try:
                for raw_line in iter(stream.readline, b""):
                    self._append_background_output(info, self._decode_output_bytes(raw_line))
            except (OSError, ValueError):
                pass
            finally:
                stream.close()

        threading.Thread(target=_consume_stream, args=(info.process.stdout,), daemon=True).start()
        threading.Thread(target=_consume_stream, args=(info.process.stderr,), daemon=True).start()

    def _append_background_output(self, info: BackgroundCommandInfo, text: str) -> None:
        with info.lock:
            info.output_chunks.append(text)
            joined = "".join(info.output_chunks)
            if len(joined) > self._BACKGROUND_MAX_OUTPUT_LENGTH:
                info.output_chunks = [joined[-self._BACKGROUND_MAX_OUTPUT_LENGTH:]]

    def _background_status(self, info: BackgroundCommandInfo) -> str:
        return "running" if info.process.poll() is None else "exited"

    def _background_snapshot(self, info: BackgroundCommandInfo, tail_chars: int = 4000) -> dict[str, Any]:
        with info.lock:
            output = "".join(info.output_chunks)
        tail = output[-tail_chars:] if tail_chars else ""
        status = self._background_status(info)
        return {
            "service_id": info.service_id,
            "command": info.command,
            "working_dir": str(info.working_dir),
            "pid": info.process.pid,
            "status": status,
            "returncode": info.process.poll(),
            "elapsed_seconds": round(time.time() - info.started_at, 3),
            "output_tail": tail,
        }

    def _clamp_float(self, value: Any, minimum: float, maximum: float, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    def _clamp_int(self, value: Any, minimum: int, maximum: int, default: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    def _json_tool_result(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _run_bash_command(self, command, working_dir: Path):
        # 只负责执行与解码；是否可执行由 before hook 决定。
        # 这里改为流式读取子进程输出，避免长命令执行期间控制台长时间无反馈。
        process = self._open_shell_process(command, working_dir=working_dir)
        output_chunks: list[str] = []
        stream_lock = threading.Lock()

        def _consume_stream(stream, is_error: bool = False):
            # 按行读取子进程输出并实时打印，同时把内容收集到结果中返回给模型。
            if stream is None:
                return
            try:
                for raw_line in iter(stream.readline, b""):
                    line = self._decode_output_bytes(raw_line)
                    with stream_lock:
                        if is_error:
                            print(line, end="", file=sys.stderr, flush=True)
                        else:
                            print(line, end="", flush=True)
                        output_chunks.append(line)
            except (OSError, ValueError):
                # 主线程超时终止后会关闭管道，读流线程遇到关闭就直接退出。
                pass
            finally:
                stream.close()

        # stdout/stderr 并发消费，避免其中一侧缓冲区写满导致命令阻塞。
        stdout_thread = threading.Thread(target=_consume_stream, args=(process.stdout, False), daemon=True)
        stderr_thread = threading.Thread(target=_consume_stream, args=(process.stderr, True), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        timed_out = False
        try:
            process.wait(timeout=self._BASH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_bash_process_tree(process)
            self._close_bash_streams(process)

        stdout_thread.join(timeout=self._BASH_TERMINATE_GRACE_SECONDS)
        stderr_thread.join(timeout=self._BASH_TERMINATE_GRACE_SECONDS)
        if timed_out:
            timeout_message = (
                f"\n[Tool timeout] {ToolNameConstant.EXECUTE_BASH} timed out after "
                f"{self._BASH_TIMEOUT_SECONDS} seconds and was terminated. "
            )
            with stream_lock:
                print(timeout_message, end="", flush=True)
                output_chunks.append(timeout_message)
        return "".join(output_chunks)

    def _close_bash_streams(self, process: subprocess.Popen) -> None:
        # 超时终止后关闭管道，确保 readline 不会继续阻塞。
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def _terminate_bash_process_tree(self, process: subprocess.Popen) -> None:
        # 常驻服务通常由 shell 再启动子进程，必须终止整个进程树。
        if process.poll() is not None:
            return
        if os.name == "nt":
            ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
            if ctrl_break is not None:
                try:
                    process.send_signal(ctrl_break)
                    process.wait(timeout=self._BASH_TERMINATE_GRACE_SECONDS)
                    return
                except (OSError, subprocess.TimeoutExpired):
                    pass
            try:
                process.kill()
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=self._BASH_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()

    def _execute_bash_with_hooks(self, command, working_dir=None, cwd=None):
        args = {"command": command}
        # 任意 before hook 都可以拦截命令，避免危险命令进入 subprocess。
        for hook in self._bash_before_hooks:
            blocked, message = hook(args)
            if blocked:
                return message or "Bash command was blocked."
        try:
            resolved_working_dir = self._resolve_working_dir(working_dir=working_dir, cwd=cwd)
        except ValueError as error:
            return str(error)
        result = self._run_bash_command(command, resolved_working_dir)
        # after hook 用于日志、输出裁剪等后处理。
        for hook in self._bash_after_hooks:
            result = hook(result)
        return result

    def _bash_command_arg(self, args: dict[str, Any]) -> str:
        # hook 统一通过 args 传参，这里确保最终按字符串处理。
        command = args.get("command", "")
        return command if isinstance(command, str) else str(command)

    def _bash_check_dangerous_command(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 第一层防线：命中黑名单则直接拒绝执行。
        command = self._bash_command_arg(args)
        for pattern in self._BASH_DANGEROUS_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return True, "The command is dangerous, refused to execute it."
        return False, None

    def _bash_ask_user_confirmation(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 第二层防线：开启人工确认时，展示命令并等待用户输入。
        if self._BASH_AUTO_APPROVE:
            return False, None
        print("\nConfirm bash execution")
        print(f"command: {self._bash_command_arg(args)[:200]}")
        while True:
            answer = input("[Y] execute / [N] skip / [Q] quit Agent > ").strip().lower()
            if answer in ("y", "yes", ""):
                return False, None
            if answer in ("n", "no"):
                return True, "Bash command skipped by user."
            if answer in ("q", "quit"):
                # 不在工具层直接退出进程，改为抛出中断异常交给 Agent 统一收口，
                # 这样可以确保 run() 里的记忆落盘/清理逻辑先执行。
                raise KeyboardInterrupt("User requested to quit during bash confirmation.")

    def _bash_log_command(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 日志 hook 不改变执行结果，只记录即将运行的命令。
        print(f"[Tool before] {ToolNameConstant.EXECUTE_BASH}: {self._bash_command_arg(args)}")
        return False, None

    def _bash_log_result(self, result: Any) -> Any:
        # 只记录输出长度，避免将大段或敏感输出重复打印。
        text = result if isinstance(result, str) else str(result)
        print(f"[Tool after] {ToolNameConstant.EXECUTE_BASH}: {len(text)} chars")
        return result

    def _decode_subprocess_result(self, result: CompletedProcess[bytes] | CompletedProcess[Any]):
        # 兼容常见编码，避免中文终端输出乱码。
        if isinstance(result.stdout, str):
            stdout = result.stdout
        else:
            stdout = b"" if result.stdout is None else result.stdout
        if isinstance(result.stderr, str):
            stderr = result.stderr
        else:
            stderr = b"" if result.stderr is None else result.stderr
        if isinstance(stdout, str) and isinstance(stderr, str):
            return stdout, stderr
        return self._decode_output_bytes(stdout), self._decode_output_bytes(stderr)

    def read_file(self, path, offset=None, limit=None):
        # 返回带行号的片段，方便模型基于行定位修改内容。
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        numbered = [f"{i + 1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
        return "".join(numbered)

    def write_file(self, path, content):
        # 覆盖写入文件并返回简短执行结果。
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"

    def edit(self, path, old_string, new_string):
        # 为降低误改风险，要求 old_string 只出现一次。
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.count(old_string) != 1:
            return "Error: old_string must appear exactly once"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace(old_string, new_string))
        return f"Successfully edited {path}"

    def glob(self, pattern):
        # 按修改时间倒序返回，优先看到最近变更文件。
        files = glob_module.glob(pattern, recursive=True)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return "\n".join(files) if files else "No files found"

    def grep(self, pattern, path="."):
        # 使用 Python 实现递归搜索，避免 Windows 环境缺少 grep 命令导致工具失效。
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        target = Path(path)
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [p for p in target.rglob("*") if p.is_file()]
        else:
            return "No matches found"

        matches: list[str] = []
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{file_path}:{line_no}: {line.rstrip()}")
            except OSError:
                continue
        return "\n".join(matches) if matches else "No matches found"

    def get_time(self):
        # 统一系统时间格式，供规则注入和提示词拼接使用。
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _search_with_duckduckgo(self, query: str, max_results: int) -> list[dict[str, str]]:
        return self.web_tool._search_with_duckduckgo(query, max_results)

    def _search_with_bing(self, query: str, max_results: int) -> list[dict[str, str]]:
        return self.web_tool._search_with_bing(query, max_results)

    def _web_search_backend_order(self) -> list[tuple[str, Callable[[str, int], list[dict[str, str]]]]]:
        return self.web_tool._web_search_backend_order()

    def web_extract(self, search_results: list[dict[str, Any]] | str, max_pages: int = 5) -> list[dict[str, Any]]:
        return self.web_tool.web_extract(search_results, max_pages=max_pages)

    def _summarize_web_search_results(
            self,
            query: str,
            results: list[dict[str, str]],
            extracted_pages: list[dict[str, Any]] | None = None,
    ) -> str:
        return self.web_tool._summarize_web_search_results(query, results, extracted_pages or [])

    def web_search(self, query: str, max_results: int = 10) -> str:
        result = self.web_tool.web_search(query, max_results)
        self.last_web_search_results = self.web_tool.last_web_search_results
        return result

    def list_dir(self, path):
        # 统一目录列表输出格式，显式区分文件与目录。
        entries = sorted(os.listdir(path))
        result = []
        for entry in entries:
            full = os.path.join(path, entry)
            prefix = "[dir]" if os.path.isdir(full) else "[file]"
            result.append(f"{prefix} {entry}")
        return "\n".join(result) or "Empty directory"
