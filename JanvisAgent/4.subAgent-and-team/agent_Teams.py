# encoding : utf-8
import atexit
import hashlib
import httpx
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread, current_thread
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse
from openai import OpenAI, OpenAIError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from memory_manager import MemoryManager
from prompt_loader import load_prompt
from session_manager import SessionManager
from tools.tool_manager import ToolManager, ToolManagerConfig, AgentToolHandlers
from tools.tool_names import ToolNameConstant
from tools.tool_scheduler import ToolScheduler, ToolCallTask
from spinner import Spinner
from task_history_viewer import open_task_history_viewer
from token_tracker import TokenTracker


@dataclass
class ToolFailureGuardState:
    """单轮 Agent 执行中的工具失败熔断状态。"""

    active_tools: list[dict[str, Any]]
    disabled_tools: set[str]
    tools_without_make_plan: list[dict[str, Any]] | None = None
    last_failure_key: tuple[str, str] | None = None
    consecutive_failure_count: int = 0
    last_success_result_key: tuple[str, str, str] | None = None
    consecutive_success_duplicate_count: int = 0
    blocked_duplicate_call_keys: set[tuple[str, str]] = field(default_factory=set)
    executed_tool_count: int = 0
    completion_continue_count: int = 0

    def __post_init__(self):
        if self.tools_without_make_plan is None:
            self.refresh_tools_without_make_plan()

    def refresh_tools_without_make_plan(self):
        # active_tools 发生变化时同步刷新计划步骤可用工具，避免递归计划再次调用 MAKE_PLAN。
        self.tools_without_make_plan = [
            tool
            for tool in self.active_tools
            if tool.get("function", {}).get("name") != ToolNameConstant.MAKE_PLAN
        ]


@dataclass
class StreamToolCallState:
    """流式响应中单个 tool_call 的分片状态。"""

    id: str | None = None
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "".join(self.name_parts)

    @property
    def arguments(self) -> str:
        return "".join(self.argument_parts)


@dataclass
class StreamResponseState:
    """流式响应累积状态。"""

    content_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, StreamToolCallState] = field(default_factory=dict)
    usage: dict[str, int] | None = None

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


@dataclass
class PendingModelConfig:
    """待新增模型配置，测试成功后才写入配置文件。"""

    model_name: str
    base_url: str
    api_key: str
    base_url_env: str
    api_key_env: str
    max_model_context_token: int


@dataclass
class AgentRuntimeConfig:
    """控制 Agent 启动时加载哪些上下文和工具。"""

    base_prompt_file: str | None = None
    base_prompt_text: str | None = None
    prompt_variables: dict[str, Any] = field(default_factory=dict)
    extra_prompt_files: list[str] = field(default_factory=list)
    include_rules: bool = True
    include_skills: bool = True
    include_memory: bool = True
    enable_local_tools: bool = True
    enable_mcp_tools: bool = True
    allowed_tool_names: set[str] | None = None


class Agent:
    
    _DEFAULT_CONTEXT_WINDOW = 32768
    _COMPACT_TRIGGER_RATIO = 0.8
    _MIDDLE_COMPACT_RATIO = 0.3
    _KEEP_RECENT = 10
    _MAX_CONSECUTIVE_TOOL_FAILURES = 2
    _MAX_CONSECUTIVE_IDENTICAL_TOOL_RESULTS = 2
    _MAX_TASK_COMPLETION_CONTINUES = 2
    _MEMORY_WAIT_POLL_SECONDS = 5
    _TOOLS_WITH_OWN_PROGRESS_OUTPUT = {
        ToolNameConstant.WEB_SEARCH,
        ToolNameConstant.EXECUTE_BASH,
        ToolNameConstant.START_BACKGROUND_COMMAND,
        ToolNameConstant.CHECK_BACKGROUND_COMMAND,
        ToolNameConstant.READ_BACKGROUND_COMMAND_OUTPUT,
        ToolNameConstant.STOP_BACKGROUND_COMMAND,
    }

    def __init__(self, model="qwen3.5:9b",
                 temperature: float = 0.1,
                 base_url: str = None,
                 api_key: str = None,
                 mcp_client=None,
                 is_main_agent: bool = True,
                 role: str = "Main Agent",
                 name: str = None,
                 runtime_config: AgentRuntimeConfig | dict[str, Any] | None = None):
        """
        初始化Agent对象
        :param model: 使用模型
        :param temperature: 模型推理时温度
        :param base_url: 模型的url
        :param api_key: 模型api_key
        :param mcp_client: MCP客户端实例（由外部传入）
        :param is_main_agent: 是否是主Agent，默认True
        :param role: Agent角色，默认为主agent
        :param name: Agent名称，默认为主agent
        """
        # 角色和名字
        self.role = role
        self.name = name

        # 通信信道
        self.inbox = []

        # 通过运行配置裁剪 prompt、记忆和工具，桌游/模拟类 Agent 可避免加载工程规则与工具 schema。
        self.runtime_config = self._normalize_runtime_config(runtime_config)

        # base_url
        explicit_model_connection = base_url is not None or api_key is not None
        self._base_url = base_url if explicit_model_connection else None

        # api_key
        self._api_key = api_key if explicit_model_connection else None

        # OpenAI 请求客户端
        self._model_client_bypass_proxy = False
        self._missing_model_env_vars = []

        # 是否是主 Agent，False 表示由 Agent 创建的子 Agent，默认为 True。
        self._is_main_agent = is_main_agent

        # 最大迭代次数
        self.max_iterations = 100

        # 使用模型
        self.model = model
        # 显式传入连接信息时优先使用传参；否则再按模型配置解析环境变量引用。
        if explicit_model_connection:
            self._missing_model_env_vars = []
        else:
            model_config_applied = self._apply_model_config_by_name(self.model)
            if not model_config_applied and self._missing_model_env_vars:
                missing_names = ", ".join(self._missing_model_env_vars)
                raise ValueError(f"模型 {self.model} 配置存在，但缺少环境变量：{missing_names}")
            if not model_config_applied:
                self._base_url = os.environ.get("OPENAI_BASE_URL")
                self._api_key = os.environ.get("OPENAI_API_KEY")
                self.client = self._create_openai_client(self._base_url, self._api_key)
        if explicit_model_connection:
            self.client = self._create_openai_client(self._base_url, self._api_key)
        self._max_context_tokens = self._load_model_context_window()
        self._token_tracker = TokenTracker(
            max_context_tokens=self._max_context_tokens,
            default_context_window=self._DEFAULT_CONTEXT_WINDOW,
        )

        # LLM 温度参数
        self.temperature = temperature

        # 主 Agent 才保留长期记忆，由主 Agent 唤起的子 Agent 不保留记忆，用完即丢。
        self.memory_manager = (
            MemoryManager(
                project_root=os.path.dirname(__file__),
                client=self.client,
                model=self.model,
                temperature=self.temperature,
            )
            if self._is_main_agent and self.runtime_config.include_memory
            else None
        )

        # 是否处在计划模式
        self.plan_mode = False

        # 当前计划列表
        self.current_plan: list[str] = []

        # 规则和技能目录
        self.rules_dir = "agent/rules"
        self.skills_dir = "agent/skills"
        self.prompts_dir = "agent/prompts"

        # 各 SKILL.md 缓存，key 为技能名称，value 为 SKILL.md 完整内容。
        self._skills_cache = {}

        # MCP 客户端（由外部传入，不在 Agent 内部创建）。
        self.mcp_client = mcp_client if self.runtime_config.enable_mcp_tools else None
        self.console = Console()
        mcp_mode = getattr(self.mcp_client, "mode", "none") if self.mcp_client else "none"
        if self.mcp_client:
            startup_spinner = self._start_spinner(messages=[f"正在初始化 MCP 客户端（mode: {mcp_mode}）..."])
            try:
                self._prepare_mcp_client()
            finally:
                startup_spinner.stop()
        else:
            self._prepare_mcp_client()

        startup_spinner = self._start_spinner(messages=["正在加载本地工具和 MCP 工具..."])
        try:
            self._tool_manager = ToolManager(
                config=ToolManagerConfig(
                    project_root=os.path.dirname(__file__),
                    client=self.client,
                    model=self.model,
                    temperature=self.temperature,
                    is_main_agent=self._is_main_agent,
                    spinner_factory=(
                        lambda preset=Spinner.DEFAULT, **context: self._start_spinner(preset=preset, **context)
                        if self._should_show_tool_spinner()
                        else None
                    ),
                ),
                handlers=AgentToolHandlers(
                    make_plan_handler=self._make_plan,
                    load_skill_detail_handler=self._load_skill_detail_by_name,
                    load_full_memory_context_handler=self._load_full_memory_context,
                    sub_agent_handler=self._sub_agent if self._is_main_agent else None,
                ),
                mcp_client=self.mcp_client,
            )
        finally:
            startup_spinner.stop()
        self._local_tools = self._filter_tools_for_runtime(self._tool_manager.local_tools, is_mcp=False)
        self._mcp_tools = self._filter_tools_for_runtime(self._tool_manager.mcp_tools, is_mcp=True)
        self._all_tools = self._local_tools + self._mcp_tools
        enabled_tool_names = self._tool_names_from_tools(self._all_tools)
        self._available_functions = {
            name: function
            for name, function in self._tool_manager.available_functions.items()
            if name in enabled_tool_names
        }
        self._all_tools_without_make_plan = self._build_tools_without_make_plan(self._all_tools)
        mcp_tool_count = len(self._mcp_tools) if self.mcp_client else 0
        mcp_status = f"{mcp_tool_count} MCP tools" if self.mcp_client else "MCP disabled"
        self.console.print(f"[dim]工具加载完成：{len(self._local_tools)} local tools，{mcp_status}[/]")

        # 基础提示词，用于主 Agent。
        self._base_prompt_main_agent = self._resolve_base_prompt("base_main_agent.md")

        # 子 Agent 提示词。
        self._base_prompt_sub_agent = self._resolve_base_prompt("base_sub_agent.md")

        # 缓存系统提示词,后续记忆压缩的时候可能会用到
        self._cached_system_prompt = self._build_system_prompt()

        self._session_interrupted_recorded = False
        self._session_exit_hook_registered = False

        # 主 Agent 才记录完整会话；子 Agent 的过程保留在主任务 turn 中，避免产生零散会话文件。
        self.session_manager = (
            SessionManager(project_root=os.path.dirname(__file__))
            if self._is_main_agent
            else None
        )
        if self.session_manager is not None:
            # 会话文件懒创建，避免用户启动后直接退出时留下空 session。
            self._register_session_exit_hook()

        self._current_turn_id = None
        self._session_title_auto_started = False
        self._pending_session_title = None
        self._pending_session_title_source = None

        # 单次会话中的短期记忆，记录一次会话中的短期上下文
        self.messages = [
            # 初始化时添加系统提示词
            {"role": "system", "content": self._cached_system_prompt}
        ]
        # 当前任务的完整上下文，用于保存长期记忆
        self._current_task_full_context = None
        self._current_task_start_index = None

        # 当前会话的当前用户任务
        self._current_task = None

    def _ensure_token_tracker(self) -> TokenTracker:
        if not hasattr(self, "_token_tracker"):
            # 兼容测试中通过 Agent.__new__ 构造的轻量对象，首次访问时补齐 tracker。
            self._token_tracker = TokenTracker(
                max_context_tokens=getattr(self, "_max_context_tokens", self._DEFAULT_CONTEXT_WINDOW),
                default_context_window=self._DEFAULT_CONTEXT_WINDOW,
            )
        return self._token_tracker

    @property
    def _used_token(self) -> int:
        return self._ensure_token_tracker().used_token

    @_used_token.setter
    def _used_token(self, value: int) -> None:
        self._ensure_token_tracker().used_token = int(value or 0)

    @property
    def _total_token(self) -> int:
        return self._ensure_token_tracker().total_token

    @_total_token.setter
    def _total_token(self, value: int) -> None:
        self._ensure_token_tracker().total_token = int(value or 0)

    def _refresh_token_context_window(self) -> int:
        self._max_context_tokens = self._load_model_context_window()
        self._ensure_token_tracker().update_context_window(self._max_context_tokens)
        return self._max_context_tokens

    def _update_usage_from_response(self, response) -> dict[str, int] | None:
        # 保留旧私有方法入口，实际 token 解析和状态更新统一交给 TokenTracker。
        return self._ensure_token_tracker().update_from_response(response)

    def _update_and_record_response_usage(
            self,
            response,
            response_kind: str,
            message_id: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        tracker = self._ensure_token_tracker()
        usage = tracker.extract_usage(response)
        if not usage:
            return None
        if response_kind == "assistant_response":
            # 只有主会话 assistant_response 会更新当前上下文 token；独立检查请求只持久化，不覆盖状态。
            tracker.update_from_response(usage)
        return self._record_response_usage(usage, response_kind, message_id=message_id, metadata=metadata)

    def _record_response_usage(
            self,
            response_or_usage,
            response_kind: str,
            message_id: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        usage = self._ensure_token_tracker().extract_usage(response_or_usage)
        if not usage:
            return None
        session_manager = getattr(self, "session_manager", None)
        if session_manager is None or session_manager.current_session_id is None:
            return None
        try:
            event = session_manager.record_response_usage(
                turn_id=getattr(self, "_current_turn_id", None),
                usage=usage,
                response_kind=response_kind,
                model=getattr(self, "model", None),
                message_id=message_id,
                metadata=metadata,
            )
            self._restore_session_usage_summary()
            return event
        except Exception as error:
            self.console.print(f"[yellow]usage 记录写入失败：{error}[/]")
            return None

    def _restore_session_usage_summary(self, session_id: str | None = None) -> dict[str, Any]:
        tracker = self._ensure_token_tracker()
        session_manager = getattr(self, "session_manager", None)
        target_session_id = session_id or getattr(session_manager, "current_session_id", None)
        if session_manager is None or not target_session_id:
            tracker.clear_session_usage()
            return tracker.session_usage_summary()
        summary = session_manager.calculate_session_usage(target_session_id, include_deleted=False)
        tracker.set_session_usage_summary(summary)
        return summary

    def _message_text(self, message) -> str:
        return self._ensure_token_tracker().message_text(message)

    def _estimate_text_tokens(self, text) -> int:
        return self._ensure_token_tracker().estimate_text_tokens(text)

    def _estimate_messages_tokens(self, messages, tools=None) -> int:
        return self._ensure_token_tracker().estimate_messages_tokens(messages, tools)

    def _calculate_token_usage_ratio(self, used_tokens: int, context_window: int) -> float:
        return self._ensure_token_tracker().calculate_usage_ratio(used_tokens, context_window)

    def _render_token_usage_bar(self, usage_ratio: float, width: int = 30) -> str:
        return self._ensure_token_tracker().render_usage_bar(usage_ratio, width)

    def _mcp_client_ping_ok(self) -> bool:
        """
        检查 MCP 客户端是否可以 ping 通。
        :return:
        """
        ping = getattr(self.mcp_client, "ping", None)
        if not callable(ping):
            return False
        try:
            # 用 ping 做真实连通性检查，避免仅检查 socket/process 字段但连接已失效。
            ping()
            return True
        except Exception:
            return False

    def _prepare_mcp_client(self, max_attempts: int = 3):
        """
        初始化 ToolManager 前检查 MCP 客户端是否可用，不可用则退回本地工具。
        :param max_attempts: 最大启动尝试次数
        :return:
        """
        if not self.mcp_client:
            return

        if self._mcp_client_ping_ok():
            return

        start = getattr(self.mcp_client, "start", None)
        if not callable(start):
            # 传入对象无法启动且当前不可用，禁用 MCP，避免 ToolManager 初始化时报错。
            self.console.print("[yellow]传入的 MCP 客户端不可用：ping 不通且没有 start 方法，将只使用本地工具[/]")
            self.mcp_client = None
            return

        last_error = None
        for _ in range(max_attempts):
            try:
                start()
            except Exception as e:
                last_error = e
            if self._mcp_client_ping_ok():
                return

        self.console.print(
            f"[yellow]传入的 MCP 客户端不可用：尝试启动 {max_attempts} 次后仍 ping 不通，"
            f"将只使用本地工具。最后错误：{last_error}[/]"
        )
        self.mcp_client = None

    def _model_config_file_path(self) -> Path:
        """
        获取模型配置文件路径。
        :return:
        """
        # 统一模型配置路径，后续读取与写入共用。
        return Path(self._agent_file_path("agent/config/model_config.json"))

    def _read_model_config(self) -> dict:
        """
        读取模型配置文件。
        :return:
        """
        # 若文件不存在或内容异常，返回空配置结构，避免启动报错。
        config_path = self._model_config_file_path()
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"models": []}
        if not isinstance(config, dict):
            return {"models": []}
        if not isinstance(config.get("models"), list):
            config["models"] = []
        return config

    def _write_model_config(self, config: dict):
        """
        写入模型配置文件。
        :param config: 配置内容
        :return:
        """
        # 确保目录存在并按 UTF-8 美化输出 JSON。
        config_path = self._model_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_model_config_by_name(self, model_name: str) -> dict | None:
        """
        根据模型名获取模型配置项。
        :param model_name: 模型名
        :return:
        """
        # 从模型配置列表里查找目标模型。
        models = self._read_model_config().get("models", [])
        for model_info in models:
            if isinstance(model_info, dict) and model_info.get("name") == model_name:
                return model_info
        return None

    def _normalize_model_env_prefix(self, model_name: str) -> str:
        """
        根据模型名生成适合环境变量使用的前缀。
        :param model_name: 模型名
        :return:
        """
        # 环境变量名只保留字母、数字和下划线，避免模型名里的冒号、点号、横线影响系统环境变量。
        prefix = re.sub(r"[^0-9A-Za-z]+", "_", str(model_name or "")).strip("_").upper()
        return prefix or "MODEL"

    def _model_env_var_names(self, model_name: str) -> tuple[str, str]:
        """
        生成模型连接信息对应的环境变量名。
        :param model_name: 模型名
        :return: (base_url_env, api_key_env)
        """
        prefix = self._normalize_model_env_prefix(model_name)
        return f"{prefix}_BASE_URL", f"{prefix}_API_KEY"

    def _resolve_model_connection_config(self, model_info: dict) -> tuple[str | None, str | None, list[str]]:
        """
        从模型配置中解析真实 base_url/api_key。
        :param model_info: model_config.json 中的单个模型配置
        :return: (base_url, api_key, missing_env_vars)
        """
        missing_env_vars: list[str] = []

        base_url_env = model_info.get("base_url_env")
        if base_url_env:
            base_url = os.environ.get(str(base_url_env))
            if base_url is None:
                missing_env_vars.append(str(base_url_env))
        else:
            base_url = model_info.get("base_url")

        api_key_env = model_info.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(str(api_key_env))
            if api_key is None:
                missing_env_vars.append(str(api_key_env))
        else:
            api_key = model_info.get("api_key")

        return base_url, api_key, missing_env_vars

    def _set_persistent_environment_variable(self, name: str, value: str) -> tuple[bool, str | None]:
        """
        写入当前进程和当前用户环境变量。
        :param name: 环境变量名
        :param value: 环境变量值
        :return: (是否成功, 错误信息)
        """
        os.environ[name] = value
        if os.name != "nt":
            return True, "当前平台仅已写入本进程环境变量，请手动写入 shell profile 后再开启新终端。"

        try:
            env = os.environ.copy()
            env["AGENTLEARN_ENV_NAME"] = name
            env["AGENTLEARN_ENV_VALUE"] = value
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "[Environment]::SetEnvironmentVariable($env:AGENTLEARN_ENV_NAME,$env:AGENTLEARN_ENV_VALUE,'User')",
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except Exception as error:
            return False, str(error)
        return True, None

    def _persist_environment_variables_async(self, env_vars: dict[str, str]) -> None:
        """
        后台持久化环境变量，避免 model add 前台等待 PowerShell 写入。
        :param env_vars: 环境变量名和值
        :return:
        """
        # 当前进程先同步写入，保证本次运行立即可用；Windows 用户环境变量持久化交给后台线程。
        for name, value in env_vars.items():
            os.environ[name] = value

        def _persist_worker():
            for env_name, env_value in env_vars.items():
                ok, warning_or_error = self._set_persistent_environment_variable(env_name, env_value)
                if not ok:
                    self.console.print(f"[yellow]后台写入环境变量 {env_name} 失败：{warning_or_error}[/]")
                    continue
                if warning_or_error:
                    self.console.print(f"[yellow]{warning_or_error}[/]")
                else:
                    self.console.print(f"[green]后台环境变量 {env_name} 持久化完成。[/]")
            self.console.print("[dim]环境变量后台写入任务已结束；新开的终端才能自动继承这些变量。[/]")

        Thread(
            target=_persist_worker,
            daemon=True,
            name="model-env-persister",
        ).start()

    def _is_local_model_base_url(self, base_url: str | None) -> bool:
        """
        判断模型地址是否是本机服务。
        :param base_url: 模型服务地址
        :return:
        """
        parsed = urlparse(base_url or "")
        hostname = (parsed.hostname or "").lower()
        return hostname in {"localhost", "127.0.0.1", "::1"}

    def _create_openai_client(self, base_url: str | None, api_key: str | None) -> OpenAI:
        """
        根据模型地址创建 OpenAI 兼容客户端。
        :param base_url: 模型服务地址
        :param api_key: API Key
        :return:
        """
        self._model_client_bypass_proxy = self._is_local_model_base_url(base_url)
        return self._build_openai_client(base_url, api_key)

    def _build_openai_client(self, base_url: str | None, api_key: str | None) -> OpenAI:
        """
        创建不修改 Agent 运行状态的 OpenAI 兼容客户端。
        :param base_url: 模型服务地址
        :param api_key: API Key
        :return:
        """
        if self._is_local_model_base_url(base_url):
            # 本地 Ollama 等 loopback 服务不应走 Clash 全局代理，否则 localhost 可能被代理层错误处理。
            return OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=httpx.Client(trust_env=False),
            )
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def _sync_model_runtime_dependencies(self):
        """
        模型客户端切换后同步依赖该客户端的运行时组件。
        :return:
        """
        # ToolManager 和 MemoryManager 在初始化时持有 client/model，运行时切换模型后需要显式同步。
        if hasattr(self, "memory_manager") and self.memory_manager is not None:
            self.memory_manager.client = self.client
            self.memory_manager.model = self.model
            self.memory_manager.temperature = self.temperature
        if hasattr(self, "_tool_manager") and self._tool_manager is not None:
            self._tool_manager.client = self.client
            self._tool_manager.model = self.model
            self._tool_manager.temperature = self.temperature
            if hasattr(self._tool_manager, "web_tool"):
                self._tool_manager.web_tool.client = self.client
                self._tool_manager.web_tool.model = self.model

    def _apply_model_config_by_name(self, model_name: str) -> bool:
        """
        根据模型名应用模型配置（base_url/api_key）。
        :param model_name: 模型名
        :return: 是否成功应用模型连接配置
        """
        # 在不改变初始化接口的前提下，按模型配置自动补全连接信息。
        model_info = self._get_model_config_by_name(model_name)
        if not model_info:
            self._missing_model_env_vars = []
            return False
        base_url, api_key, missing_env_vars = self._resolve_model_connection_config(model_info)
        self._missing_model_env_vars = missing_env_vars
        if missing_env_vars:
            return False
        self._base_url = base_url if base_url is not None else self._base_url
        self._api_key = api_key if api_key is not None else self._api_key
        self.client = self._create_openai_client(self._base_url, self._api_key)
        self._sync_model_runtime_dependencies()
        return True

    def receive(self, sender, message):
        """
        通信，接收来自其他agent的信息
        :param sender: 发送者
        :param message: 消息
        :return: 无
        """
        self.inbox.append({"from": sender, "content": message})

    def _agent_file_path(self, relative_path):
        return os.path.join(os.path.dirname(__file__), relative_path)

    def _normalize_runtime_config(self, runtime_config):
        if runtime_config is None:
            return AgentRuntimeConfig()
        if isinstance(runtime_config, AgentRuntimeConfig):
            config = runtime_config
        elif isinstance(runtime_config, dict):
            config = AgentRuntimeConfig(**runtime_config)
        else:
            raise TypeError("runtime_config must be AgentRuntimeConfig, dict or None")

        # 统一把可迭代配置转成稳定的 list/set，后续过滤 prompt 和工具时不再分支判断。
        if config.extra_prompt_files is None:
            config.extra_prompt_files = []
        elif isinstance(config.extra_prompt_files, str):
            config.extra_prompt_files = [config.extra_prompt_files]
        else:
            config.extra_prompt_files = list(config.extra_prompt_files)

        if config.allowed_tool_names is not None:
            config.allowed_tool_names = set(config.allowed_tool_names)
        if config.prompt_variables is None:
            config.prompt_variables = {}
        else:
            config.prompt_variables = dict(config.prompt_variables)
        return config

    def _render_prompt_variables(self, content: str) -> str:
        # 先注入默认变量，再叠加领域运行时变量，支持桌游规则等动态上下文。
        variables = {
            "role": self.role or "",
            "name": self.name or self.role or "",
        }
        variables.update(self.runtime_config.prompt_variables)
        for key, value in variables.items():
            content = content.replace(f"<{key}>", str(value))
        return content

    def _load_prompt_file(self, prompt_file: str) -> str:
        prompt_path = Path(prompt_file)
        if prompt_path.is_absolute():
            candidates = [prompt_path]
        else:
            project_root = Path(os.path.dirname(__file__))
            candidates = [
                project_root / prompt_path,
                project_root / self.prompts_dir / prompt_path,
            ]

        for candidate in candidates:
            if candidate.exists():
                return self._render_prompt_variables(candidate.read_text(encoding="utf-8").strip())
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    def _resolve_base_prompt(self, default_prompt_name: str) -> str:
        if self.runtime_config.base_prompt_text:
            return self._render_prompt_variables(self.runtime_config.base_prompt_text.strip())
        if self.runtime_config.base_prompt_file:
            return self._load_prompt_file(self.runtime_config.base_prompt_file)
        return self._render_prompt_variables(load_prompt(default_prompt_name, role=self.role))

    def _extra_system_prompt_parts(self) -> list[str]:
        # 额外 prompt 按传入顺序拼接，桌游规则这类动态上下文放在基础人设之后。
        return [
            self._load_prompt_file(prompt_file)
            for prompt_file in self.runtime_config.extra_prompt_files
        ]

    def _load_model_context_window(self):
        config_path = Path(self._agent_file_path("agent/config/model_config.json"))
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return self._DEFAULT_CONTEXT_WINDOW
        # 从模型配置中按模型名读取上下文窗口配置。
        models = config.get("models", []) if isinstance(config, dict) else []
        for model_info in models:
            if isinstance(model_info, dict) and model_info.get("name") == self.model:
                return int(model_info.get("max_model_context_token") or self._DEFAULT_CONTEXT_WINDOW)
        return self._DEFAULT_CONTEXT_WINDOW

    def _schedule_memory_update(self, task, result):
        if not self._is_main_agent or self.memory_manager is None:
            return
        session_id = self.session_manager.current_session_id if self.session_manager is not None else None
        last_memory_checked_event_id = (
            self.session_manager.latest_event_id(session_id)
            if self.session_manager is not None and session_id
            else None
        )
        # 长期记忆摘要异步执行，避免拖慢当前任务完成。
        future = self.memory_manager.enqueue(
            task=task,
            result=result,
            context=self._current_task_full_context or [],
            session_id=session_id,
            turn_id=self._current_turn_id,
            last_memory_checked_event_id=last_memory_checked_event_id,
        )
        if future is None or self.session_manager is None or not self._current_turn_id:
            return

        self._attach_memory_future_session_callback(future, session_id=session_id, turn_id=self._current_turn_id)

    def _attach_memory_future_session_callback(self, future, session_id: str | None, turn_id: str | None):
        if future is None or self.session_manager is None or not turn_id:
            return

        def _record_memory_saved(done_future):
            try:
                index_record = done_future.result()
                if not index_record:
                    return
                memory_status = index_record.get("memory_status") or "unknown"
                # 每个 turn 都记录记忆处理 cursor，避免继续旧 session 后重复处理旧历史。
                self.session_manager.record_memory_checked(
                    turn_id=turn_id,
                    last_memory_checked_event_id=index_record.get("last_memory_checked_event_id"),
                    status=memory_status,
                    task_id=index_record.get("task_id"),
                    metadata={
                        "skip_reason": index_record.get("skip_reason"),
                        "memory_item_count": index_record.get("memory_item_count", 0),
                    },
                    session_id=session_id,
                )
                if memory_status == "captured":
                    # 只有长期记忆真正沉淀为 topic item 后，才把 task_id 关联到当前 turn。
                    self.session_manager.record_memory_saved(
                        turn_id=turn_id,
                        task_id=index_record.get("task_id"),
                        full_context_path=index_record.get("full_context_path"),
                        metadata={"memory_item_count": index_record.get("memory_item_count", 0)},
                        session_id=session_id,
                    )
            except Exception as error:
                try:
                    event = (
                        {
                            "event": "memory_save_error",
                            "turn_id": turn_id,
                            "error": str(error),
                        }
                    )
                    if session_id:
                        self.session_manager.append_event_to_session(session_id, event)
                    else:
                        self.session_manager.append_event(event)
                except Exception:
                    pass

        future.add_done_callback(_record_memory_saved)

    def _maybe_auto_title_session(self, task: str):
        """
        根据当前会话的第一个用户任务自动生成标题。
        :param task: 用户任务
        :return:
        """
        if not self._is_main_agent or self.session_manager is None:
            return
        if self._session_title_auto_started:
            return
        session_info = self.session_manager.get_current_session_info()
        if session_info.get("title_source") == "user":
            return

        normalized_task = " ".join(str(task or "").strip().split())
        if not normalized_task:
            return
        self._session_title_auto_started = True

        if len(normalized_task) <= 8:
            # 8 个字以内的任务本身已经足够短，直接作为会话标题。
            self._set_session_title(normalized_task, source="auto", silent=True)
            return

        # 长任务标题异步生成，不阻塞当前用户任务的执行。
        Thread(
            target=self._generate_session_title_async,
            args=(normalized_task,),
            daemon=True,
            name="session-title-generator",
        ).start()

    def _generate_session_title_async(self, task: str):
        """
        后台调用当前模型生成中文会话标题。
        :param task: 用户任务
        :return:
        """
        try:
            # 使用临时 messages 执行独立标题任务，避免污染主会话上下文窗口。
            title_messages = [
                {
                    "role": "system",
                    "content": (
                        "You generate concise Chinese conversation titles. "
                        "Return only the title, no punctuation, no explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the following user task into a Chinese title.\n"
                        "Requirements:\n"
                        "- 6 to 18 Chinese characters when possible\n"
                        "- No punctuation\n"
                        "- No explanation\n\n"
                        f"User task:\n{task}"
                    ),
                },
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=title_messages,
                temperature=0,
            )
            raw_title = response.choices[0].message.content
            title = self._normalize_generated_session_title(raw_title, task)
        except Exception:
            title = self._fallback_session_title(task)
        self._set_session_title(title, source="auto", silent=True)

    def _normalize_generated_session_title(self, raw_title: str, task: str) -> str:
        """
        清理模型生成的标题。
        :param raw_title: 模型原始输出
        :param task: 用户任务，用于兜底
        :return: 清理后的标题
        """
        title = str(raw_title or "").strip().splitlines()[0].strip()
        for prefix in ("标题：", "标题:", "Title:", "title:"):
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        title = title.strip("`\"'“”‘’《》<>。，、：:；;！!？?")
        if not title:
            return self._fallback_session_title(task)
        if len(title) > 18:
            title = title[:18]
        return title

    def _fallback_session_title(self, task: str) -> str:
        """
        本地生成兜底会话标题。
        :param task: 用户任务
        :return: 兜底标题
        """
        normalized_task = " ".join(str(task or "").strip().split())
        return normalized_task[:18] if normalized_task else "未命名会话"

    def _set_session_title(self, title: str, source: str = "user", silent: bool = False) -> bool:
        """
        设置当前会话标题。
        :param title: 标题
        :param source: 标题来源
        :param silent: 是否静默设置
        :return: 是否设置成功
        """
        if self.session_manager is None:
            if not silent:
                self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return False
        if self.session_manager.current_session_id is None:
            normalized_title = " ".join(str(title or "").strip().split())
            if not normalized_title:
                if not silent:
                    self.console.print("[yellow]会话标题不能为空。[/]")
                return False
            # 会话尚未落盘时只暂存标题，等第一条真实任务创建 session 时一起写入。
            self._pending_session_title = normalized_title
            self._pending_session_title_source = source if source in {"user", "auto", "default"} else "user"
            if not silent:
                self.console.print(f"[dim]已设置待创建会话标题：{normalized_title}[/]")
            return True
        try:
            event = self.session_manager.update_title(title, source=source)
        except ValueError:
            if not silent:
                self.console.print("[yellow]会话标题不能为空。[/]")
            return False
        except Exception as error:
            if not silent:
                self.console.print(f"[yellow]会话标题更新失败：{error}[/]")
            return False

        if event.get("event") == "session_title_update_skipped":
            return False
        if not silent:
            self.console.print(f"[dim]当前会话标题已更新为：{event.get('title')}[/]")
        return True

    def _load_memory_view(self):
        # 子 Agent 不继承长期记忆，只处理被委派的当前任务。
        if not self._is_main_agent or self.memory_manager is None:
            return ""
        return self.memory_manager.load_prompt_memory_view()

    def _load_full_memory_context(self, task_id):
        # 只通过 MemoryManager 按 task_id 读取完整上下文，避免模型直接拼接文件路径。
        if not self._is_main_agent or self.memory_manager is None:
            return {
                "error": "memory_unavailable",
                "message": "Long-term memory is only available to the main agent.",
            }
        return self.memory_manager.get_task_full_context(task_id)

    def _wait_for_memory_tasks(self):
        if not self._is_main_agent or self.memory_manager is None:
            return
        if self.memory_manager.has_pending():
            pending_count = self.memory_manager.pending_count()
            timeout = self.memory_manager.memory_request_timeout
            self.console.print(
                f"[dim]还有 {pending_count} 个记忆整理任务未完成，正在等待完成后退出"
                f"（后台模型请求最长约 {timeout:g} 秒）...[/]"
            )
        self.memory_manager.shutdown()

    def _wait_for_pending_memory_updates(self):
        """
        等待当前仍在运行的记忆整理任务，但不关闭 MemoryManager。
        :return:
        """
        if not self._is_main_agent or self.memory_manager is None:
            return
        if self.memory_manager.has_pending():
            pending_count = self.memory_manager.pending_count()
            timeout = self.memory_manager.memory_request_timeout
            self.console.print(
                f"[dim]还有 {pending_count} 个记忆整理任务未完成，正在等待完成"
                f"（后台模型请求最长约 {timeout:g} 秒/个）...[/]"
            )
        # 切换会话前只等待后台任务归档完成，后续新任务仍需要继续写长期记忆。
        while self.memory_manager.has_pending():
            if self.memory_manager.wait_for_pending(timeout=self._MEMORY_WAIT_POLL_SECONDS):
                break
            pending_count = self.memory_manager.pending_count()
            self.console.print(f"[dim]记忆整理仍在进行：剩余 {pending_count} 个任务...[/]")

    def _enqueue_session_end_memory_compaction(self):
        if not self._is_main_agent or self.memory_manager is None:
            return
        self.memory_manager.enqueue_compact_memory_index(reason="session_end")

    def _register_session_exit_hook(self):
        """
        注册进程退出兜底清理逻辑。
        :return:
        """
        if self._session_exit_hook_registered:
            return
        # 即使调用方没有走 Agent.run()，解释器正常退出时也尽量补写 session_end。
        atexit.register(self._cleanup_session_on_process_exit)
        self._session_exit_hook_registered = True

    def _cleanup_session_on_process_exit(self):
        """
        进程退出时兜底结束当前 session。
        :return:
        """
        if self.session_manager is None:
            return
        try:
            self._end_session()
        except Exception:
            pass

    def _record_session_interrupted(self, reason: str):
        """
        记录当前会话被中断。
        :param reason: 中断原因
        :return:
        """
        if (
            self.session_manager is None
            or self.session_manager.current_session_id is None
            or self._session_interrupted_recorded
        ):
            return
        try:
            # 中断事件单独落盘，便于区分正常退出和 Ctrl+C 退出。
            self.session_manager.record_session_interrupted(reason=reason, turn_id=self._current_turn_id)
            self._session_interrupted_recorded = True
        except Exception as error:
            self.console.print(f"[yellow]会话中断事件写入失败：{error}[/]")

    def _end_session(self):
        """
        结束当前 session。
        :return:
        """
        if self.session_manager is None:
            return
        try:
            self.session_manager.end_session()
        except Exception as error:
            self.console.print(f"[yellow]会话结束事件写入失败：{error}[/]")

    def _session_metadata(self) -> dict[str, Any]:
        # 会话元信息集中生成，避免 session new 和懒启动时字段不一致。
        return {
            "role": self.role,
            "name": self.name,
            "is_main_agent": self._is_main_agent,
        }

    def _ensure_session_started(self) -> str | None:
        """
        确保当前已有可写入的 session。
        :return: 当前 session_id
        """
        if self.session_manager is None:
            return None
        if self.session_manager.current_session_id:
            return self.session_manager.current_session_id

        title = self._pending_session_title or "未命名会话"
        title_source = self._pending_session_title_source or "default"
        session_id = self.session_manager.start_session(
            model=self.model,
            title=title,
            title_source=title_source,
            metadata=self._session_metadata(),
        )
        self._session_interrupted_recorded = False
        self._pending_session_title = None
        self._pending_session_title_source = None
        # session_start 后立刻写入 system message，保证历史会话可完整恢复。
        if self.messages:
            self._record_session_message(self.messages[0], turn_id=None)
        # 这里只在新建 session 的分支执行；已有会话会在前面的 current_session_id 分支直接返回。
        # 清空会话级 usage 是为了避免新会话继承上一个会话的上下文 token。
        self._ensure_token_tracker().clear_session_usage()
        return session_id

    def set_bash_auto_approve(self, enabled: bool):
        """
        设置 Bash 工具是否自动确认执行。
        :param enabled: True 表示自动确认，False 表示需要手动确认
        :return:
        """
        self._tool_manager.set_bash_auto_approve(enabled)

    def _bash_approve_status_text(self) -> str:
        """
        返回当前 Bash 执行确认策略的文本描述。
        :return:
        """
        return self._tool_manager.bash_approve_status_text()

    def _record_session_message(self, message, turn_id=None, metadata=None):
        """
        把短期上下文消息同步写入当前 session。
        :param message: OpenAI message
        :param turn_id: 当前用户任务 id
        :param metadata: 只写入 session 的辅助元信息
        :return:
        """
        if self.session_manager is None or self.session_manager.current_session_id is None:
            return None
        try:
            # 会话记录失败不应该打断 Agent 正常回答。
            return self.session_manager.append_message(message, turn_id=turn_id, metadata=metadata)
        except Exception as error:
            self.console.print(f"[yellow]会话记录写入失败：{error}[/]")
            return None

    def _append_message(self, message, capture_full_context=True, session_metadata=None):
        self.messages.append(message)
        if capture_full_context and self._current_task_full_context is not None:
            self._current_task_full_context.append(self._normalize_message_for_memory(message))
        return self._record_session_message(message, turn_id=self._current_turn_id, metadata=session_metadata)

    def _normalize_message_for_memory(self, message):
        if isinstance(message, dict):
            normalized = message
        elif hasattr(message, "model_dump"):
            normalized = message.model_dump()
        elif hasattr(message, "to_dict"):
            normalized = message.to_dict()
        else:
            normalized = {
                "role": getattr(message, "role", None),
                "content": getattr(message, "content", None),
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is not None:
                normalized["tool_calls"] = tool_calls
        return json.loads(json.dumps(normalized, ensure_ascii=False, default=str))

    def _make_plan(self, task):
        if self.plan_mode:
            return "Error: can't make plan within a plan"
        spinner = self._start_spinner()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("task_planning_system.md"),
                    },
                    {"role": "user", "content": load_prompt("task_planning_user.md", task=task)},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
        finally:
            spinner.stop()
        # 计划模型调用也属于当前任务的真实 API 消耗，需要写入 session usage。
        self._update_and_record_response_usage(response, "task_planning")
        try:
            plan_data = json.loads(response.choices[0].message.content)
            print("make plan response is: ", response)
            steps = plan_data.get("steps", [task]) if isinstance(plan_data, dict) else [task]
            self.current_plan = steps
            print(f"[Plan] {len(steps)} steps created")
            return steps
        except Exception:
            print(f"[Plan] Failed to parse steps, returning original task {task}, exception {e}")
            return [task]

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
        """
        解析调用工具的参数
        :param raw_arguments: 字符串形式的参数
        :return: 解析的json格式参数
        """
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError as error:
            return {"_argument_error": f"Invalid JSON arguments: {error}"}

    def _load_rules(self, memory_view=""):
        """
        加载所有规则md文档，字符串形式返回
        :return:
        """
        rules = []
        if not os.path.exists(self.rules_dir):
            return rules
        system_time = self._tool_manager.get_time()
        memory_root_path = self.memory_manager.memory_dir if self.memory_manager else self._agent_file_path("agent/memory")
        memory_view = memory_view.strip() or "No previous tasks recorded."
        for rule_file in Path(self.rules_dir).glob("*.md"):
            with open(rule_file, "r", encoding="utf-8") as f:
                content = (
                    f.read()
                    .replace("<system-time>", system_time)
                    .replace("<precise-memory>", memory_view)
                    .replace("<full-memory-path>", str(memory_root_path))
                    .replace("<memory-root-path>", str(memory_root_path))
                )
                rules.append(content)
        return "\n\n".join(rules) if rules else []

    def _load_skill_meta_infos(self):
        """
        获取skill元信息，name和description，顺便缓存一下完整SKILL.md的内容
        :return:
        """
        import yaml
        skills = []
        if not os.path.exists(self.skills_dir):
            return []
        # 先清空缓存
        self._skills_cache.clear()
        for item in os.listdir(self.skills_dir):
            skill_dir = os.path.join(self.skills_dir, item)
            if not os.path.isdir(skill_dir):
                continue
            skill_md_file = os.path.join(skill_dir, "SKILL.md")
            if not os.path.exists(skill_md_file):
                continue
            with open(skill_md_file, "r", encoding="utf-8") as f:
                content = f.read()
            # 解析 YAML 前置元数据。
            if content.startswith("---"):
                frontmatter_end = content.find("---", 3)
                if frontmatter_end != -1:
                    frontmatter = content[3:frontmatter_end].strip()
                    meta = yaml.safe_load(frontmatter)
                    if meta and "name" in meta:
                        skills.append({
                            "name": meta.get("name"),
                            "description": meta.get("description", ""),
                        })
                        # 缓存完整 SKILL 内容。
                        self._skills_cache[meta.get("name")] = content
        return skills

    def _load_skill_detail_by_name(self, name):
        """
        根据skill名称读取SKILL.md完整内容
        :param name:
        :return:
        """
        if not self._skills_cache:
            self._load_skill_meta_infos()
        return self._skills_cache.get(name, "")
    
    def _message_role(self, message):
        if isinstance(message, dict):
            return message.get("role", "unknown")
        return getattr(message, "role", "unknown")

    def _should_compact_messages(self, tools=None):
        return self._ensure_token_tracker().should_compact_messages(
            self.messages,
            tools,
            compact_trigger_ratio=self._COMPACT_TRIGGER_RATIO,
        )
    
    def _find_recent_start(self):
        start = max(1, len(self.messages) - self._KEEP_RECENT)
        # 不要从工具消息中间切开；工具消息必须紧跟触发它的 assistant tool_call。
        while start > 1 and self._message_role(self.messages[start]) == "tool":
            start -= 1
        return start

    def _format_messages_for_compaction(self, messages):
        text = ""
        for message in messages:
            role = self._message_role(message)
            content = self._ensure_token_tracker().message_text(message)
            if content:
                text += f"[{role}]: {content}\n"
        return text

    def _load_archived_task_reference_for_compaction(self):
        if self._is_main_agent and self.memory_manager is not None:
            return self.memory_manager.load_prompt_memory_view()
        return "No archived task references are available in this agent."

    def _select_messages_for_compaction_summary(self, old_messages, recent_start):
        if self._current_task_start_index is not None:
            live_start = max(1, self._current_task_start_index)
            live_messages = self.messages[live_start: recent_start]
            if live_messages:
                return live_messages

        middle_size = max(1, int(len(old_messages) * self._MIDDLE_COMPACT_RATIO))
        return old_messages[-middle_size:]

    def _compact_messages(self, tools=None):
        if not self._should_compact_messages(tools):
            return

        system_msg = self.messages[0]
        recent_start = self._find_recent_start()
        old_messages = self.messages[1: recent_start]
        recent_messages = self.messages[recent_start:]
        if not old_messages:
            return

        archived_task_reference = self._load_archived_task_reference_for_compaction()
        summary_messages = self._select_messages_for_compaction_summary(old_messages, recent_start)
        old_text = (
            "Archived task references for older completed tasks:\n"
            f"{archived_task_reference}\n\n"
            "Conversation messages to summarize:\n"
            f"{self._format_messages_for_compaction(summary_messages)}"
        )

        spinner = self._start_spinner()
        try:
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": load_prompt("conversation_compaction_system.md")},
                    {"role": "user", "content": load_prompt("conversation_compaction_user.md", old_text=old_text)}
                ]
            )
        finally:
            spinner.stop()
        # 压缩摘要调用不进入对话消息，但真实消耗需要计入当前 session。
        self._update_and_record_response_usage(summary_response, "compaction_summary")
        summary = summary_response.choices[0].message.content

        self.messages = [
            system_msg,
            {
                "role": "user",
                "content": (
                    "[Archived completed-task references]\n"
                    f"{archived_task_reference}\n\n"
                    "For details from an archived task, call LOAD_FULL_MEMORY_CONTEXT with the task_id."
                ),
            },
            {"role": "user", "content": load_prompt("conversation_compaction_summary_message.md", summary=summary)},
            *recent_messages
        ]

    def _tool_call_failure_key(self, task: ToolCallTask) -> tuple[str, str]:
        """生成工具失败去重 key，用于识别同一工具和同一参数的连续失败。"""
        try:
            normalized_args = json.dumps(task.function_args, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            normalized_args = task.raw_arguments
        return task.function_name, normalized_args

    def _tool_call_repeat_key(self, task: ToolCallTask) -> tuple[str, str]:
        """生成工具调用去重 key，用于识别相同工具和相同参数。"""
        return self._tool_call_failure_key(task)

    def _tool_result_fingerprint(self, function_response) -> str:
        """生成工具结果指纹，用于识别成功但没有新信息的重复调用。"""
        try:
            result_text = json.dumps(function_response, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            result_text = str(function_response)
        normalized_text = " ".join(str(result_text or "").split())
        return hashlib.sha256(normalized_text.encode("utf-8", errors="replace")).hexdigest()

    def _is_tool_call_failure(self, function_response) -> bool:
        """判断工具结果是否表示失败。"""
        if isinstance(function_response, dict):
            if function_response.get("ok") is False:
                return True
            return bool(function_response.get("error") or function_response.get("error_type"))
        text = str(function_response or "").strip()
        lowered = text.lower()
        return (
            lowered.startswith("error")
            or "执行失败" in text
            or "failed" in lowered
            or "exception" in lowered
        )

    def _filter_available_tools(self, tools: list[dict[str, Any]], failed_tool_name: str) -> list[dict[str, Any]]:
        """从本轮可用工具列表中移除连续失败的工具。"""
        return [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") != failed_tool_name
        ]

    @staticmethod
    def _tool_name_from_schema(tool: dict[str, Any]) -> str | None:
        if not isinstance(tool, dict):
            return None
        function_schema = tool.get("function", {})
        return function_schema.get("name") or tool.get("name")

    @classmethod
    def _tool_names_from_tools(cls, tools: list[dict[str, Any]]) -> set[str]:
        return {
            tool_name
            for tool_name in (cls._tool_name_from_schema(tool) for tool in tools)
            if tool_name
        }

    def _filter_tools_for_runtime(self, tools: list[dict[str, Any]], *, is_mcp: bool) -> list[dict[str, Any]]:
        if is_mcp and not self.runtime_config.enable_mcp_tools:
            return []
        if not is_mcp and not self.runtime_config.enable_local_tools:
            return []

        allowed_tool_names = self.runtime_config.allowed_tool_names
        if allowed_tool_names is None:
            return list(tools)

        # 白名单只作用于最终暴露给模型的工具 schema，不影响 ToolManager 内部能力初始化。
        return [
            tool
            for tool in tools
            if self._tool_name_from_schema(tool) in allowed_tool_names
        ]

    @staticmethod
    def _build_tools_without_make_plan(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """生成去除 MAKE_PLAN 后的工具列表，供计划步骤递归调用复用。"""
        return [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") != ToolNameConstant.MAKE_PLAN
        ]

    def _format_tool_failure_response(self, task: ToolCallTask, error_type: str, message: str, retryable: bool = False) -> dict[str, Any]:
        """生成结构化工具错误，帮助模型区分工具系统失败和正常业务结果。"""
        return {
            "ok": False,
            "tool": task.function_name,
            "arguments": task.function_args,
            "error_type": error_type,
            "retryable": retryable,
            "message": message,
        }

    def _request_next_model_message(self, active_tools):
        # 模型调用前先做上下文压缩，避免把压缩逻辑散在主循环里。
        self._compact_messages(active_tools)
        spinner = self._start_spinner()
        try:
            response_stream = self._create_stream_completion(active_tools, include_usage=True)
            return self._deal_stream_response(response_stream, spinner=spinner)
        finally:
            spinner.stop()

    def _create_stream_completion(self, active_tools, include_usage: bool = True):
        request_args = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if active_tools:
            request_args["tools"] = active_tools
        if include_usage:
            request_args["stream_options"] = {"include_usage": True}
        try:
            return self.client.chat.completions.create(**request_args)
        except Exception as error:
            if include_usage and self._is_stream_options_unsupported_error(error):
                return self._create_stream_completion(active_tools, include_usage=False)
            raise

    @staticmethod
    def _is_stream_options_unsupported_error(error: Exception) -> bool:
        text = str(error or "").lower()
        return "stream_options" in text and any(
            marker in text
            for marker in ("unsupported", "not support", "unrecognized", "unknown", "extra", "invalid")
        )

    def _append_assistant_response(self, message):
        tool_calls = getattr(message, "tool_calls", None)
        event = self._append_message({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (tool_calls or [])
            ] if tool_calls else None,
        })
        message_id = event.get("message_id") if event else None
        self._record_response_usage(message, "assistant_response", message_id=message_id)

    def _append_tool_result(self, task: ToolCallTask, function_response):
        self._append_message(
            {
                "role": "tool",
                "tool_call_id": task.tool_call_id,
                "content": json.dumps(function_response, ensure_ascii=False),
            }
        )

    def _check_repeated_success_tool_call(
            self,
            task: ToolCallTask,
            function_response,
            guard_state: ToolFailureGuardState,
    ):
        call_key = self._tool_call_repeat_key(task)
        result_key = (*call_key, self._tool_result_fingerprint(function_response))
        if result_key == guard_state.last_success_result_key:
            guard_state.consecutive_success_duplicate_count += 1
        else:
            previous_call_key = guard_state.last_success_result_key[:2] if guard_state.last_success_result_key else None
            if previous_call_key is not None and call_key != previous_call_key:
                # 模型已经尝试了不同动作，允许后续重新验证之前被拦截的精确调用。
                guard_state.blocked_duplicate_call_keys.clear()
            guard_state.last_success_result_key = result_key
            guard_state.consecutive_success_duplicate_count = 1

        if guard_state.consecutive_success_duplicate_count < self._MAX_CONSECUTIVE_IDENTICAL_TOOL_RESULTS:
            return None

        guard_state.blocked_duplicate_call_keys.add(call_key)
        self._append_message(
            {
                "role": "user",
                "content": (
                    f"Tool {task.function_name} returned the same result for the same arguments "
                    f"{guard_state.consecutive_success_duplicate_count} consecutive times. "
                    "Do not repeat this exact tool call. Use the existing result to continue, "
                    "try a meaningfully different action, or explain why no further progress is possible."
                ),
            },
            session_metadata={"is_task_entry": False, "tool_repeat_guard": True},
        )
        return None

    def _reset_success_repeat_guard(self, guard_state: ToolFailureGuardState) -> None:
        guard_state.last_success_result_key = None
        guard_state.consecutive_success_duplicate_count = 0

    def _append_tool_result_and_check_guard(
            self,
            task: ToolCallTask,
            function_response,
            guard_state: ToolFailureGuardState,
    ):
        self._append_tool_result(task, function_response)
        guard_state.executed_tool_count += 1
        if not self._is_tool_call_failure(function_response):
            guard_state.last_failure_key = None
            guard_state.consecutive_failure_count = 0
            return self._check_repeated_success_tool_call(task, function_response, guard_state)

        self._reset_success_repeat_guard(guard_state)
        failure_key = self._tool_call_failure_key(task)
        if failure_key == guard_state.last_failure_key:
            guard_state.consecutive_failure_count += 1
        else:
            guard_state.last_failure_key = failure_key
            guard_state.consecutive_failure_count = 1

        if guard_state.consecutive_failure_count < self._MAX_CONSECUTIVE_TOOL_FAILURES:
            return None

        guard_state.active_tools = self._filter_available_tools(guard_state.active_tools, task.function_name)
        guard_state.disabled_tools.add(task.function_name)
        guard_state.refresh_tools_without_make_plan()
        user_reason = (
            f"工具 {task.function_name} 使用相同参数连续失败 {guard_state.consecutive_failure_count} 次。"
            f"最后一次结果：{function_response}"
        )
        model_reason = (
            f"Tool {task.function_name} failed {guard_state.consecutive_failure_count} consecutive times "
            f"with the same arguments. Last result: {function_response}"
        )
        if not guard_state.active_tools:
            return f"{user_reason}\n没有其他可用工具，已结束当前任务。"

        # 提醒模型换用其他相关工具；如果没有合适工具，应直接给用户说明失败原因。
        self._append_message(
            {
                "role": "user",
                "content": (
                    f"{model_reason}\n"
                    f"Do not call {task.function_name} again. Use another relevant tool if available. "
                    "If no suitable tool is available, stop tool use and explain why the task cannot be completed."
                ),
            },
            session_metadata={"is_task_entry": False, "tool_failure_guard": True},
        )
        return None

    def _build_tool_call_task(self, tool_call):
        function_payload = getattr(tool_call, "function", None)
        if function_payload is None:
            return None
        function_name = str(getattr(function_payload, "name", ""))
        raw_arguments = str(getattr(function_payload, "arguments", ""))
        return ToolCallTask(
            tool_call_id=tool_call.id,
            function_name=function_name,
            raw_arguments=raw_arguments,
            function_args=self._parse_tool_arguments(raw_arguments),
        )

    def _append_disabled_tool_call_response(self, task: ToolCallTask):
        function_response = self._format_tool_failure_response(
            task,
            "disabled_repeated_failure",
            f"Tool {task.function_name} is disabled for this turn after repeated failures.",
            retryable=False,
        )
        self._append_tool_result(task, function_response)
        return f"工具 {task.function_name} 已因连续失败被禁用，但模型再次请求该工具，已结束当前任务。"

    def _append_repeated_tool_call_response(self, task: ToolCallTask):
        function_response = self._format_tool_failure_response(
            task,
            "repeated_identical_call",
            (
                f"Tool {task.function_name} already returned the same result for the same arguments. "
                "This exact repeated call was blocked to prevent a tool loop."
            ),
            retryable=False,
        )
        self._append_tool_result(task, function_response)
        return f"工具 {task.function_name} 使用相同参数连续返回相同结果，模型仍重复请求，已结束当前任务。"

    def _execute_pending_tool_tasks(
            self,
            scheduler: ToolScheduler,
            pending_tasks: list[ToolCallTask],
            guard_state: ToolFailureGuardState,
    ):
        # 保留调度器的批次规划结果顺序，确保 tool message 和原 tool_call 一一对应。
        for task, function_response in scheduler.execute_batches(
                scheduler.plan_batches(pending_tasks), self._invoke_tool_task
        ):
            stop_reason = self._append_tool_result_and_check_guard(task, function_response, guard_state)
            if stop_reason:
                return stop_reason
        return None

    def _run_plan_steps(self, steps: list[str], tools_without_make_plan: list[dict[str, Any]]):
        results = []
        step_cnt = 0
        for step in steps:
            print(f"[Step {step_cnt + 1}]: {step}")
            self._append_message(
                {"role": "user", "content": step},
                session_metadata={"is_task_entry": False},
            )
            result = self._run_agent_step(tools_without_make_plan, task_goal=step)
            print(f"[Step {step_cnt + 1}] result:{result}, all messages: {self.messages}")
            step_cnt += 1
            results.append(result)
        return "\n".join(results)

    def _execute_make_plan_task(self, task: ToolCallTask, guard_state: ToolFailureGuardState):
        function_impl = self._available_functions.get(task.function_name)
        if function_impl is None:
            return self._format_tool_failure_response(
                task,
                "unknown_tool",
                f"Unknown tool '{task.function_name}'",
                retryable=False,
            )
        if "_argument_error" in task.function_args:
            return self._format_tool_failure_response(
                task,
                "argument_error",
                task.function_args["_argument_error"],
                retryable=False,
            )

        # MAKE_PLAN 会递归调用 Agent，递归调用时移除 MAKE_PLAN，避免计划中再次计划。
        self.plan_mode = True
        steps = function_impl(**task.function_args)
        if not isinstance(steps, list):
            function_response = steps
        else:
            function_response = self._run_plan_steps(steps, guard_state.tools_without_make_plan)
        self.plan_mode = False
        self.current_plan = []
        return function_response

    def _handle_tool_calls(self, tool_calls, guard_state: ToolFailureGuardState):
        # 第二版调度策略：依据工具画像（只读/并发安全/作用域）做分段并发。
        scheduler = ToolScheduler(get_profile=self._tool_manager.get_tool_runtime_profile)
        pending_tasks: list[ToolCallTask] = []

        for tool_call in tool_calls:
            current_task = self._build_tool_call_task(tool_call)
            if current_task is None:
                continue

            if current_task.function_name in guard_state.disabled_tools:
                return self._append_disabled_tool_call_response(current_task)

            if self._tool_call_repeat_key(current_task) in guard_state.blocked_duplicate_call_keys:
                return self._append_repeated_tool_call_response(current_task)

            if current_task.function_name == ToolNameConstant.MAKE_PLAN:
                stop_reason = self._execute_pending_tool_tasks(scheduler, pending_tasks, guard_state)
                if stop_reason:
                    return stop_reason
                pending_tasks = []

                function_response = self._execute_make_plan_task(current_task, guard_state)
                stop_reason = self._append_tool_result_and_check_guard(
                    current_task, function_response, guard_state
                )
                if stop_reason:
                    return stop_reason
                continue

            pending_tasks.append(current_task)

        return self._execute_pending_tool_tasks(scheduler, pending_tasks, guard_state)

    def _is_trivial_task(self, task_goal: str) -> bool:
        text = " ".join(str(task_goal or "").strip().lower().split())
        trivial_tasks = {
            "hi", "hello", "hey", "你好", "您好", "谢谢", "thanks", "thank you", "ok", "好的", "嗯",
            "good bye", "see you", "再见", "yep", "good", "that's it", "alright", "yes, it is", "yes"
        }
        return text in trivial_tasks or len(text) <= 2

    def _is_execution_task(self, task_goal: str) -> bool:
        text = str(task_goal or "").lower()
        keywords = (
            "写", "创建", "生成", "修改", "修复", "实现", "重构", "整理", "搜索", "检查",
            "删除", "运行", "测试", "分析", "阅读", "查找", "保存", "更新", "提交",
            "write", "create", "generate", "modify", "fix", "implement", "refactor",
            "search", "check", "delete", "run", "test", "analyze", "read", "find",
            "save", "update", "commit",
        )
        return any(keyword in text for keyword in keywords)

    def _looks_like_terminal_response(self, content: str | None) -> bool:
        text = str(content or "").strip().lower()
        if not text:
            return False
        terminal_markers = (
            "需要你", "请提供", "请补充", "无法继续", "不能继续", "没有足够信息",
            "已完成", "已经完成", "已保存", "已写入", "任务完成",
            "need you", "please provide", "need more information", "cannot continue",
            "unable to continue", "completed", "done", "saved",
        )
        return any(marker in text for marker in terminal_markers)

    def _should_check_task_complete(
            self,
            task_goal: str | None,
            message,
            guard_state: ToolFailureGuardState,
    ) -> bool:
        if not task_goal:
            return False
        if guard_state.completion_continue_count >= self._MAX_TASK_COMPLETION_CONTINUES:
            return False
        if self._is_trivial_task(task_goal):
            return False
        if self._looks_like_terminal_response(getattr(message, "content", None)):
            return False
        # 计划步骤、工具链任务和执行型任务才需要额外判断，普通问答直接结束。
        return self.plan_mode or guard_state.executed_tool_count > 0 or self._is_execution_task(task_goal)

    def _completion_check_recent_messages(self, limit: int = 5) -> list[dict[str, Any]]:
        recent_messages = []
        for message in self.messages[-limit:]:
            recent_messages.append(self._normalize_message_for_memory(message))
        return recent_messages

    def _parse_completion_status(self, answer: str | None) -> str:
        text = str(answer or "").strip().upper()
        normalized = text
        for char in "{}[]():,;\"'":
            normalized = normalized.replace(char, " ")
        for status in ("DONE", "CONTINUE", "NEED_USER", "BLOCKED"):
            if text.startswith(status) or status in normalized.split():
                return status
        return "DONE"

    def _check_task_complete(self, task_goal: str, last_reply: str | None) -> str:
        """
        轻量检查当前执行目标是否完成，返回 DONE/CONTINUE/NEED_USER/BLOCKED。
        """
        recent_messages = json.dumps(
            self._completion_check_recent_messages(),
            ensure_ascii=False,
            default=str,
        )
        check_prompt = load_prompt(
            "task_completion_check_user.md",
            task_goal=task_goal,
            recent_messages=recent_messages,
            last_reply=last_reply or "",
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": load_prompt("task_completion_check_system.md"),
                    },
                    {"role": "user", "content": check_prompt},
                ],
                temperature=0,
            )
        except Exception:
            return "DONE"
        self._update_and_record_response_usage(response, "task_completion_check")
        answer = response.choices[0].message.content
        return self._parse_completion_status(answer)

    def _append_task_completion_continue_prompt(self, task_goal: str) -> None:
        self._append_message(
            {
                "role": "user",
                "content": load_prompt("task_completion_continue_user.md", task_goal=task_goal),
            },
            session_metadata={"is_task_entry": False, "task_completion_guard": True},
        )

    def _append_empty_tool_followup_prompt(self, task_goal: str | None) -> None:
        self._append_message(
            {
                "role": "user",
                "content": (
                    "The previous model response after tool execution was empty. "
                    "Use the tool result above to answer the user's task directly. "
                    "Do not call another tool unless it is necessary.\n"
                    f"User task: {task_goal or ''}"
                ),
            },
            session_metadata={"is_task_entry": False, "empty_tool_followup_guard": True},
        )

    def _run_agent_step(self, tools, task_goal: str | None = None):
        active_tools = list(tools)
        tools_without_make_plan = (
            list(self._all_tools_without_make_plan)
            if tools is self._all_tools
            else self._build_tools_without_make_plan(active_tools)
        )
        guard_state = ToolFailureGuardState(
            active_tools=active_tools,
            disabled_tools=set(),
            tools_without_make_plan=tools_without_make_plan,
        )

        for i in range(self.max_iterations):
            message = self._request_next_model_message(guard_state.active_tools)
            print()
            self._append_assistant_response(message)

            if not message.tool_calls:
                if guard_state.executed_tool_count > 0 and not str(getattr(message, "content", "") or "").strip():
                    if guard_state.completion_continue_count < self._MAX_TASK_COMPLETION_CONTINUES:
                        # 工具结果之后只收到 usage 空响应时，补一个内部继续提示，避免交互模式静默结束。
                        guard_state.completion_continue_count += 1
                        self._append_empty_tool_followup_prompt(task_goal)
                        continue
                    empty_result = "模型在工具结果后连续返回空响应，已停止重试。"
                    self.console.print(f"[yellow]{empty_result}[/]")
                    return empty_result
                # 没有工具调用，可能是任务结束了，先判断然后再返回结果
                if not self._should_check_task_complete(task_goal, message, guard_state):
                    return message.content

                status = self._check_task_complete(task_goal, message.content)
                # 任务没有完成，需要继续
                if status == "CONTINUE":
                    guard_state.completion_continue_count += 1
                    self._append_task_completion_continue_prompt(task_goal)
                    continue
                return message.content
            stop_reason = self._handle_tool_calls(message.tool_calls, guard_state)
            if stop_reason:
                return stop_reason
        return "Max iterations reached"

    def _should_show_tool_spinner(self) -> bool:
        # 并发工具在工作线程中执行，避免多个 Live 动画同时刷新控制台。
        return current_thread().name == "MainThread"

    def _should_wrap_tool_with_spinner(self, tool_name: str) -> bool:
        # 有些工具内部会分阶段显示进度或实时打印输出，外层不再叠加 Rich Live 动画。
        return self._should_show_tool_spinner() and tool_name not in self._TOOLS_WITH_OWN_PROGRESS_OUTPUT

    def _print_network_search_results(self):
        search_results = getattr(self._tool_manager, "last_web_search_results", None)
        if not search_results:
            return
        self.console.print("[bold cyan]网络搜索结果：[/]")
        self.console.print(json.dumps(search_results, ensure_ascii=False, indent=2))

    def _invoke_tool_task(self, task: ToolCallTask):
        """调度器执行入口：单工具调用的统一异常处理。"""
        function_impl = self._available_functions.get(task.function_name)
        if function_impl is None:
            return self._format_tool_failure_response(
                task,
                "unknown_tool",
                f"Unknown tool '{task.function_name}'",
                retryable=False,
            )
        if "_argument_error" in task.function_args:
            return self._format_tool_failure_response(
                task,
                "argument_error",
                task.function_args["_argument_error"],
                retryable=False,
            )
        try:
            print(f"[Tool call] tool name: {task.function_name}, tool arguments: {task.raw_arguments}")
            spinner = self._start_spinner(preset=Spinner.TOOL, tool_name=task.function_name) \
                if self._should_wrap_tool_with_spinner(task.function_name) else None
            try:
                result = function_impl(**task.function_args)
            finally:
                if spinner is not None:
                    spinner.stop()
            return result
        except Exception as error:
            return self._format_tool_failure_response(
                task,
                "tool_exception",
                f"Error when calling '{task.function_name}': {error}",
                retryable=False,
            )

    def _build_system_prompt(self):
        from prompt_builder import build_system_prompt
        # 置空当前提示词。
        self._cached_system_prompt = None
        memory = self._load_memory_view() if self.runtime_config.include_memory else ""
        rules = self._load_rules(memory) if self.runtime_config.include_rules else []
        if self.runtime_config.include_skills:
            skills = self._load_skill_meta_infos()
        else:
            self._skills_cache.clear()
            skills = []
        base_prompt = [
            self._base_prompt_main_agent if self._is_main_agent else self._base_prompt_sub_agent,
            *self._extra_system_prompt_parts(),
        ]
        self._cached_system_prompt = build_system_prompt(base_prompt, rules, skills, memory)
        return self._cached_system_prompt

    def _deal_stream_response(self, stream_response, spinner: Spinner | None = None):
        """
        处理模型流式响应，返回兼容 OpenAI message 结构的对象。
        :param stream_response: 模型流式响应迭代器
        :param spinner: 等待动画
        :return: 转换后的 message 对象
        """
        state = StreamResponseState()
        try:
            for chunk in self._iter_stream_chunks(stream_response, spinner):
                self._consume_stream_chunk(chunk, state)
        except Exception as error:
            self._record_stream_error(state.content, error)
            raise
        return self._build_stream_message(state)

    def _iter_stream_chunks(self, stream_response, spinner: Spinner | None = None):
        first_chunk_arrived = False
        for chunk in stream_response:
            if not first_chunk_arrived and spinner is not None:
                # 首个响应到达后关闭等待动画，切换到真实流式输出。
                spinner.stop()
                first_chunk_arrived = True
            yield chunk

    def _consume_stream_chunk(self, chunk, state: StreamResponseState) -> None:
        # 流式响应中若包含 usage 字段，则实时更新 token 统计。
        usage = self._update_usage_from_response(chunk)
        if usage:
            state.usage = usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            # include_usage 的流式尾包可能只包含 usage 且 choices 为空，这类 chunk 不再解析正文或工具调用。
            return

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return

        content = getattr(delta, "content", None)
        if content:
            self._append_stream_content(content, state)

        delta_tool_calls = getattr(delta, "tool_calls", None)
        if delta_tool_calls:
            self._append_tool_call_delta(delta_tool_calls, state)

    def _append_stream_content(self, content: str, state: StreamResponseState) -> None:
        print(content, end="", flush=True)
        state.content_parts.append(content)

    def _append_tool_call_delta(self, delta_tool_calls, state: StreamResponseState) -> None:
        for tc in delta_tool_calls:
            idx = tc.index
            if idx not in state.tool_calls:
                state.tool_calls[idx] = StreamToolCallState(id=getattr(tc, "id", None))
            tool_call_state = state.tool_calls[idx]

            # 若后续数据片段才补齐 tool_call_id，这里做增量回填，避免首包无 id 导致丢失。
            tool_call_id = getattr(tc, "id", None)
            if tool_call_id:
                tool_call_state.id = tool_call_id

            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", None)
            args = getattr(func, "arguments", None)
            if name:
                # 使用追加方式拼接，兼容极端情况下 name 被分片返回。
                tool_call_state.name_parts.append(name)
            if args:
                tool_call_state.argument_parts.append(args)

    def _record_stream_error(self, partial_content: str, error: Exception) -> None:
        session_manager = getattr(self, "session_manager", None)
        current_turn_id = getattr(self, "_current_turn_id", None)
        if session_manager is None or not current_turn_id:
            return
        try:
            # 正常流式片段不落盘，只有异常时保存已收到内容辅助排查。
            session_manager.record_model_stream_error(
                turn_id=current_turn_id,
                partial_content=partial_content,
                error=str(error),
            )
        except Exception:
            pass

    def _build_stream_message(self, state: StreamResponseState):
        ordered_tool_calls = []
        for idx in sorted(state.tool_calls.keys()):
            item = state.tool_calls[idx]
            ordered_tool_calls.append(
                SimpleNamespace(
                    id=item.id,
                    function=SimpleNamespace(
                        name=item.name,
                        arguments=item.arguments,
                    ),
                )
            )
        content = state.content
        return SimpleNamespace(
            content=content or "",
            tool_calls=ordered_tool_calls if ordered_tool_calls else None,
            usage=state.usage,
        )

    def _start_spinner(
            self,
            messages: list[str] | None = None,
            *,
            preset: str = Spinner.DEFAULT,
            **context: Any,
    ) -> Spinner:
        """
        创建并启动等待动画。
        :return:
        """
        # 统一由该方法管理等待动画的创建，尽量减少对业务逻辑的侵入。
        spinner = Spinner(self.console, messages=messages, preset=preset, **context)
        spinner.start()
        return spinner

    def _sub_agent(self, role, task):
        """
        调用一个子agent，处理一个专门的子任务
        这里sub_agent的话，怎么编排？怎么输入输出？结果怎么处理
        :param role: 角色
        :param task: 任务
        :return: 任务结果
        """
        if not self._is_main_agent:
            return "Error: can't create sub-agent within a sub-agent"

        sub_agent = Agent(model=self.model,
                          temperature=self.temperature,
                          base_url=self._base_url,
                          api_key=self._api_key,
                          role=role,
                          is_main_agent=False,
                          runtime_config=self.runtime_config
                          )
        final_result = sub_agent.chat(task)
        return final_result

    def _clear_memory(self):
        """
        清空记忆文件
        :return:
        """
        if not self._is_main_agent or self.memory_manager is None:
            return
        self.memory_manager.clear()

    def _rollback_messages_to(self, start_index: int):
        """
        回滚短期上下文到指定位置。
        :param start_index: 保留到的消息下标
        :return:
        """
        # 模型请求失败时不保留本轮未完成上下文，避免切换模型后被旧任务污染。
        if start_index < 0:
            return
        if len(self.messages) > start_index:
            self.messages = self.messages[:start_index]

    def _print_model_call_error(self, error: Exception):
        """
        使用 Rich 展示模型调用失败信息。
        :param error: 模型调用异常
        :return:
        """
        status_code = getattr(error, "status_code", None)
        status_text = f"\n[yellow]状态码：[/] {status_code}" if status_code else ""
        self.console.print(
            Panel(
                (
                    "[bold red]模型调用失败，当前任务已中止。[/]\n\n"
                    f"[yellow]当前模型：[/] {self.model}\n"
                    f"[yellow]接口地址：[/] {self._base_url or '-'}"
                    f"{status_text}\n"
                    f"[yellow]错误类型：[/] {type(error).__name__}\n"
                    f"[yellow]错误信息：[/] {error}\n\n"
                    "[cyan]请检查模型配置、模型服务状态或网络代理；也可以使用 "
                    "[bold]model <名称或编号>[/] 切换模型后重试。[/]"
                ),
                title="[bold red]Model Error[/]",
                border_style="red",
            )
        )

    def chat(self, task):
        """
        Agent单次任务运行入口
        :param task: 用户任务
        :return: 执行任务结果
        """
        self._ensure_session_started()
        rollback_message_index = len(self.messages)
        # 初始化当前任务的完整上下文记录
        self._current_turn_id = (
            self.session_manager.create_turn_id()
            if self.session_manager is not None
            else None
        )
        self._current_task_full_context = []
        try:
            # 如果收件箱有新消息，先注入 self.messages。
            if self.inbox:
                mail = "\n".join(f"[from {m['from']}]: {m['content']}" for m in self.inbox)
                self._append_message(
                    {"role": "user", "content": load_prompt("inbox_digest_user.md", mail=mail)},
                    session_metadata={"is_task_entry": False},
                )
                # 让 Agent 先消化这些消息
                spinner = self._start_spinner()
                try:
                    resp = self.client.chat.completions.create(model=self.model, messages=self.messages)
                finally:
                    spinner.stop()
                event = self._append_message(resp.choices[0].message)
                message_id = event.get("message_id") if event else None
                self._update_and_record_response_usage(resp, "inbox_digest", message_id=message_id)
                self.inbox.clear()

            # 再拼接本次任务并执行
            self._maybe_auto_title_session(task)
            self._append_message(
                {"role": "user", "content": task},
                session_metadata={"is_task_entry": True},
            )
            self._current_task = task
            try:
                final_result = self._run_agent_step(self._all_tools, task_goal=task)
            finally:
                self._current_task = None
            # print(f"final result: {final_result}")
            self._schedule_memory_update(task, final_result)
            if self.session_manager is not None:
                # 每个用户任务结束后只更新一次索引活跃时间，避免消息级索引膨胀。
                self.session_manager.touch_session(reason="task_completed")
            return final_result
        except KeyboardInterrupt:
            self._record_session_interrupted("keyboard_interrupt")
            raise
        except OpenAIError:
            self._rollback_messages_to(rollback_message_index)
            raise
        finally:
            self._current_task_full_context = None
            self._current_turn_id = None

    def run(self):
        """
        Agent loop实现，对话入口
        :return:
        """
        # 统一通过 help 文案展示可用命令，避免 run() 内硬编码过长提示。
        self._print_help(show_welcome=True)
        # 集中展示运行时状态，后续配置变更后可复用该方法刷新显示。
        self._print_runtime_status()

        confirm_choice = ("y", "yes", "是", "确认", "对", "")

        while True:
            try:
                # 在输入框上方展示当前模型，便于用户随时确认当前会话模型。
                self._print_input_header()
                user_input = self.console.input("[bold cyan]You >[/] ")
                # 将命令分支提取到独立方法中，降低 run() 循环复杂度。
                handled, should_exit = self._handle_user_command(user_input, confirm_choice)
                if should_exit:
                    break
                if handled:
                    continue
                if not user_input.strip():
                    continue

                # 上面分支都没中，就是用户任务了
                task_started_at = time.perf_counter()
                self.chat(user_input)
                elapsed_seconds = time.perf_counter() - task_started_at
                self.console.print(f"[green]已完成 {self._format_elapsed_time(elapsed_seconds)}[/]")
                self.console.print()
            except OpenAIError as error:
                self._print_model_call_error(error)
                self.console.print()
            except KeyboardInterrupt:
                self._record_session_interrupted("keyboard_interrupt")
                self._wait_for_pending_memory_updates()
                self._end_session()
                self._enqueue_session_end_memory_compaction()
                try:
                    self._wait_for_memory_tasks()
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]已保存当前 session，记忆整理等待被再次中断。[/]")
                self.console.print("\n[bold red] See you next time! [/]")
                break

    def _print_help(self, show_welcome: bool = False):
        """
        打印交互帮助信息。
        :param show_welcome: 是否展示欢迎语
        :return:
        """
        # 将帮助提示抽取为独立方法，便于复用和后续维护。
        if show_welcome:
            self.console.print(Panel(
                "[bold green]JanVis[/] — At you service, sir! What can I do for you today?\n\n"
                "You can ask me to do some task or type help/h",
                border_style="green", padding=(1, 2),
            ))
        commands = [
            ("clear", "清空当前终端窗口显示内容"),
            ("help / h", "显示帮助信息"),
            ("exit / q / quit", "退出当前 Agent 会话"),
            ("clear current_session", "清空当前会话,可选删除本会话任务记忆"),
            ("clear sessions", "选择并删除历史会话及对应记忆"),
            ("clear all_history", "删除全部历史会话及对应记忆"),
            ("bash approve on", "开启 Bash 命令人工确认"),
            ("bash approve off", "关闭 Bash 命令人工确认"),
            ("model_list/models", "查看可用模型配置"),
            ("model", "查看模型相关命令"),
            ("model add", "新增模型配置并测试"),
            ("model <name|编号>", "切换到指定模型"),
            ("tools", "列出当前可用工具"),
            ("compact", "压缩当前会话上下文"),
            ("memory compact", "整理长期记忆索引"),
            ("memory summarize sessions", "整理所有历史会话提炼长期记忆"),
            ("status", "查看当前运行状态"),
            ("history", "查看当前会话用户任务历史"),
            ("sessions", "列出最近会话"),
            ("continue", "继续上一次会话"),
            ("session current", "查看当前会话信息"),
            ("session new", "创建一个新会话"),
            ("session load <序号|session_id>", "加载并继续一个历史会话"),
            ("session title", "查看当前会话标题"),
            ("session title <标题>", "自定义当前会话标题"),
        ]
        command_table = Table.grid(padding=(0, 2))
        command_table.add_column(justify="left", no_wrap=True)
        command_table.add_column(justify="left")
        # 命令名加粗突出，说明文字使用浅色，便于快速扫描。
        for command, description in commands:
            command_table.add_row(f"[bold cyan]{command}[/]", f"[dim]{description}[/]")

        self.console.print("[bold]可用命令[/]")
        self.console.print(command_table)

    def _print_available_tools(self):
        """
        打印当前 Agent 可用工具列表。
        :return:
        """
        tool_groups = [
            ("本地工具", self._local_tools),
            ("MCP 工具", self._mcp_tools),
        ]
        has_tools = False
        for group_name, tools in tool_groups:
            if not tools:
                continue
            has_tools = True
            tool_table = Table.grid(padding=(0, 2))
            tool_table.add_column(justify="left", no_wrap=True)
            tool_table.add_column(justify="left")
            # 工具结构统一按 OpenAI tool 格式读取，兼容少量直接含 name/description 的旧格式。
            for tool in tools:
                tool_schema = tool if isinstance(tool, dict) else {}
                function_schema = tool_schema.get("function", {})
                name = function_schema.get("name") or tool_schema.get("name") or "UNKNOWN_TOOL"
                description = function_schema.get("description") or tool_schema.get("description") or "暂无描述"
                tool_table.add_row(f"[bold cyan]{name}[/]", f"[dim]{description}[/]")

            self.console.print(f"[bold]{group_name}[/]")
            self.console.print(tool_table)

        if not has_tools:
            self.console.print("[yellow]当前没有可用工具。[/]")

    def _print_runtime_status(self):
        """
        打印当前运行时状态信息。
        :return:
        """
        # 抽取状态打印逻辑，便于启动时和配置变更后统一刷新展示。
        self.console.print(f"[dim]当前工作目录：{os.getcwd()}[/]")
        self.console.print(f"[dim]使用模型：{self.model}[/]")
        proxy_mode = "本地直连（忽略系统代理）" if self._model_client_bypass_proxy else "默认网络环境"
        self.console.print(f"[dim]模型连接方式：{proxy_mode}[/]")
        if self._missing_model_env_vars:
            missing_names = ", ".join(self._missing_model_env_vars)
            self.console.print(f"[yellow]模型环境变量缺失：{missing_names}[/]")
        self.console.print(f"[dim]Bash 命令确认策略：{self._bash_approve_status_text()}[/]")

    def _print_input_header(self):
        """
        在输入提示前展示当前模型信息。
        :return:
        """
        # 每轮输入前展示当前模型，模型切换后可自动反映最新状态。
        self.console.print(f"[dim]当前模型：{self.model}[/]")

    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """
        格式化任务耗时。
        :param elapsed_seconds: 耗时秒数
        :return: 形如 1h2m30s、2m30s、30s 的文本
        """
        total_seconds = max(0, int(round(elapsed_seconds)))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes}m{seconds}s"
        if minutes:
            return f"{minutes}m{seconds}s"
        return f"{seconds}s"

    def _normalize_command(self, user_input: str) -> str:
        """
        规范化用户命令输入，支持 `/` 前缀命令。
        :param user_input: 原始用户输入
        :return:
        """
        # 统一处理前后空白和可选的 "/" 前缀（如 /clear session）。
        cmd = (user_input or "").strip().lower()
        if cmd.startswith("/"):
            cmd = cmd[1:].strip()
        return cmd

    def _input_masked_secret(self, prompt: str, mask: str = "*") -> str:
        """
        读取敏感输入，同时用固定掩码字符反馈输入长度。
        :param prompt: 输入提示
        :param mask: 掩码字符
        :return: 用户输入的原始内容
        """
        try:
            self.console.print(prompt, end="")
            if os.name == "nt":
                return self._read_masked_secret_windows(mask)
            return self._read_masked_secret_posix(mask)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            # 回退到 Rich 的隐藏输入，避免特殊终端下自定义按键读取失败。
            self.console.print()
            return self.console.input(prompt, password=True)

    def _read_masked_secret_windows(self, mask: str) -> str:
        """
        Windows 终端下逐字符读取敏感输入。
        :param mask: 掩码字符
        :return: 用户输入的原始内容
        """
        import msvcrt

        chars: list[str] = []
        while True:
            char = msvcrt.getwch()
            if char in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                return "".join(chars)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x1a":
                raise EOFError
            if char in ("\b", "\x7f"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if char in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            chars.append(char)
            sys.stdout.write(mask)
            sys.stdout.flush()

    def _read_masked_secret_posix(self, mask: str) -> str:
        """
        POSIX 终端下逐字符读取敏感输入。
        :param mask: 掩码字符
        :return: 用户输入的原始内容
        """
        import termios
        import tty

        file_descriptor = sys.stdin.fileno()
        old_settings = termios.tcgetattr(file_descriptor)
        chars: list[str] = []
        try:
            tty.setcbreak(file_descriptor)
            while True:
                char = sys.stdin.read(1)
                if char in ("\r", "\n"):
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return "".join(chars)
                if char == "\x03":
                    raise KeyboardInterrupt
                if char == "\x04":
                    raise EOFError
                if char in ("\b", "\x7f"):
                    if chars:
                        chars.pop()
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                    continue
                chars.append(char)
                sys.stdout.write(mask)
                sys.stdout.flush()
        finally:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)

    def _normalize_session_command_alias(self, raw_cmd: str) -> str:
        """
        纠正常见 session 命令误输入。
        :param raw_cmd: 已去掉可选 / 前缀的原始命令
        :return: 纠正后的命令
        """
        # 只处理明确的 session 命令别名，不做全局模糊匹配，避免普通问题被误判为命令。
        stripped = (raw_cmd or "").strip()
        if not stripped:
            return stripped
        parts = stripped.split(maxsplit=2)
        lowered = [part.lower() for part in parts]

        session_aliases = {"session", "sesssion", "sesstion"}
        sessions_aliases = {"sessions", "sesssions", "sesstions"}
        current_aliases = {"current", "currrent", "curent"}
        title_aliases = {"title", "tittle"}
        load_aliases = {"load", "lod", "laod"}
        new_aliases = {"new", "neu"}

        if len(parts) == 1 and lowered[0] in sessions_aliases:
            return "sessions"
        if lowered[0] not in session_aliases:
            return stripped
        if len(parts) == 1:
            return "session"

        sub_command = lowered[1]
        rest = parts[2] if len(parts) >= 3 else ""
        if sub_command == "s":
            return "sessions"
        if sub_command in current_aliases:
            return "session current"
        if sub_command in new_aliases:
            return "session new"
        if sub_command in load_aliases:
            return f"session load {rest}".strip()
        if sub_command in title_aliases:
            return f"session title {rest}".strip()
        return stripped

    def _handle_user_command(self, user_input: str, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        """
        处理命令型输入。
        :param user_input: 用户输入
        :param confirm_choice: 确认命令可接受输入
        :return: (是否已处理, 是否需要退出循环)
        """
        # 集中处理所有命令分支，主循环仅保留调度逻辑。
        cmd = self._normalize_command(user_input)
        raw_cmd = (user_input or "").strip()
        if raw_cmd.startswith("/"):
            raw_cmd = raw_cmd[1:].strip()
        raw_cmd = self._normalize_session_command_alias(raw_cmd)
        cmd = self._normalize_command(raw_cmd)
        # 使用命令映射表分发处理函数，便于后续扩展命令而不是堆叠 if/else。
        command_handlers = self._command_handler_map()
        handler = command_handlers.get(cmd)
        if handler:
            return handler(confirm_choice)
        raw_parts = raw_cmd.split(maxsplit=2)
        if len(raw_parts) >= 3 and raw_parts[0].lower() == "session" and raw_parts[1].lower() == "title":
            return self._handle_cmd_session_title_set(confirm_choice, raw_parts[2].strip())
        if len(raw_parts) >= 3 and raw_parts[0].lower() == "session" and raw_parts[1].lower() == "load":
            return self._handle_cmd_session_load(confirm_choice, raw_parts[2].strip())
        # 支持带参数命令，例如 model gpt-4o-mini。
        if cmd.startswith("model "):
            return self._handle_cmd_model_switch(confirm_choice, raw_cmd)
        return False, False

    def _command_handler_map(self):
        """
        命令到处理函数的映射表。
        :return:
        """
        # 集中管理命令和处理方法的映射关系，提升可维护性。
        return {
            "exit": self._handle_cmd_exit,
            "q": self._handle_cmd_exit,
            "quit": self._handle_cmd_exit,
            "help": self._handle_cmd_help,
            "h": self._handle_cmd_help,
            "clear current_session": self._handle_cmd_clear_current_session,
            "clear session": self._handle_cmd_clear_current_session,
            "clear sessions": self._handle_cmd_clear_sessions,
            "clear all_history": self._handle_cmd_clear_all_history,
            "bash approve on": self._handle_cmd_bash_approve_on,
            "bash approve off": self._handle_cmd_bash_approve_off,
            "model_list": self._handle_cmd_model_list,
            "models": self._handle_cmd_model_list,
            "model": self._handle_cmd_model_show_current,
            "model add": self._handle_cmd_model_add,
            "tools": self._handle_cmd_tools,
            "clear": self._handle_cmd_clear_screen,
            "campact": self._handle_cmd_compact_history,
            "compact": self._handle_cmd_compact_history,
            "memory compact": self._handle_cmd_memory_compact,
            "memory summarize sessions": self._handle_cmd_memory_summarize_sessions,
            "status": self._handle_cmd_status,
            "history": self._handle_cmd_history,
            "sessions": self._handle_cmd_sessions,
            "continue": self._handle_cmd_continue_session,
            "session": self._handle_cmd_session_help,
            "session current": self._handle_cmd_session_current,
            "session new": self._handle_cmd_session_new,
            "session load": self._handle_cmd_session_load_prompt,
            "session title": self._handle_cmd_session_title_show,
        }

    def _handle_cmd_exit(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 退出命令处理。
        self._wait_for_pending_memory_updates()
        self._end_session()
        self._enqueue_session_end_memory_compaction()
        self._wait_for_memory_tasks()
        self.console.print("\n[bold red] See you next time! [/]")
        return True, True

    def _handle_cmd_help(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 帮助命令处理。
        self._print_help(show_welcome=False)
        return True, False

    def _handle_cmd_tools(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 展示当前已加载的本地工具和 MCP 工具。
        self._print_available_tools()
        return True, False

    def _handle_cmd_session_help(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 用户只输入 session 时展示会话相关命令，避免被当作普通问题发送给模型。
        session_commands = [
            ("sessions", "列出最近会话"),
            ("continue", "继续上一次会话"),
            ("session current", "查看当前会话信息"),
            ("session new", "创建一个新会话"),
            ("session load", "从最近会话列表中选择并加载"),
            ("session load <序号|session_id>", "加载并继续一个历史会话"),
            ("session title", "查看当前会话标题"),
            ("session title <标题>", "自定义当前会话标题"),
        ]
        command_table = Table.grid(padding=(0, 2))
        command_table.add_column(justify="left", no_wrap=True)
        command_table.add_column(justify="left")
        for command, description in session_commands:
            command_table.add_row(f"[bold cyan]{command}[/]", f"[dim]{description}[/]")
        self.console.print("[bold]session 相关命令[/]")
        self.console.print(command_table)
        return True, False

    def _handle_cmd_clear_screen(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 清空终端窗口显示内容，不影响 Agent 的会话上下文和长期记忆。
        self.console.clear()
        return True, False

    def _handle_cmd_clear_current_session(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 当前会话采用追加 session_cleared 事件的逻辑清空，不硬改 session 文件。
        session_manager = self.session_manager
        session_id = getattr(session_manager, "current_session_id", None)
        if not self._confirm_irreversible_delete(
                "清空当前会话",
                "当前会话上下文会被清空",
        ):
            return True, False

        delete_memory = self._confirm_default_yes("[bold cyan]是否同时删除当前会话所有任务对应记忆？(Y/n，默认删除)[/] ")
        self._wait_for_pending_memory_updates()
        memory_result = None
        memory_task_ids = []
        if session_id and delete_memory:
            active_turn_ids = [
                task.get("turn_id")
                for task in session_manager.list_tasks(session_id)
                if task.get("turn_id")
            ]
            memory_result = self._delete_memories_for_session_turns(session_id, active_turn_ids)
            memory_task_ids = memory_result.get("deleted_task_ids") or memory_result.get("task_ids", [])

        if session_manager is not None and session_id:
            session_manager.record_session_cleared(delete_memory=delete_memory, memory_task_ids=memory_task_ids)
            self._reset_messages_for_session(record_system_message=True)
            self._restore_session_usage_summary(session_id)
        else:
            self._reset_messages_for_session(record_system_message=False)

        if memory_result:
            self.console.print(
                f"[dim]当前会话已清空；已删除记忆 {memory_result.get('deleted_task_count', 0)} 条，"
                f"完整上下文文件 {memory_result.get('deleted_full_context_count', 0)} 个。[/]"
            )
        else:
            self.console.print("[dim]当前会话已清空；未删除长期记忆。[/]")
        return True, False

    def _handle_cmd_clear_sessions(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 选择性硬删除历史 session 文件和对应长期记忆；当前会话不允许从该入口删除。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        sessions = self.session_manager.list_sessions(limit=100, include_empty=True)
        if not sessions:
            self.console.print("[yellow]暂无可删除的历史会话。[/]")
            return True, False

        self._print_sessions_table(sessions)
        selector = self.console.input("[bold cyan]请输入要删除的会话编号或 session_id（空格分隔，或 all）：[/] ").strip()
        if self._normalize_command(selector) == "all":
            return self._handle_cmd_clear_all_history(_confirm_choice)

        target_session_ids = self._resolve_session_delete_targets(selector, sessions)
        current_session_id = self.session_manager.current_session_id
        if current_session_id in target_session_ids:
            self.console.print("[yellow]已跳过当前会话；请使用 clear current_session 清空当前会话。[/]")
            target_session_ids = [session_id for session_id in target_session_ids if session_id != current_session_id]
        if not target_session_ids:
            self.console.print("[yellow]没有可删除的目标会话。[/]")
            return True, False

        details = "\n".join(f"- {session_id}" for session_id in target_session_ids[:20])
        if len(target_session_ids) > 20:
            details += f"\n- ... 还有 {len(target_session_ids) - 20} 个"
        if not self._confirm_irreversible_delete(
                "硬删除指定会话",
                f"将删除 {len(target_session_ids)} 个 session 文件及其对应长期记忆：\n{details}",
        ):
            return True, False

        result = self._delete_sessions_and_memories(target_session_ids)
        self._print_delete_sessions_result(result)
        return True, False

    def _handle_cmd_clear_all_history(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 删除所有历史会话；若当前处于会话中，则保留当前会话和它对应的记忆。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        sessions = self.session_manager.list_sessions(limit=10000, include_empty=True)
        current_session_id = self.session_manager.current_session_id
        target_session_ids = [
            item.get("session_id")
            for item in sessions
            if item.get("session_id") and item.get("session_id") != current_session_id
        ]
        delete_all_memories = current_session_id is None
        if delete_all_memories:
            target_session_ids = [item.get("session_id") for item in sessions if item.get("session_id")]
        if not target_session_ids and not delete_all_memories:
            self.console.print("[yellow]没有当前会话之外的历史会话可删除。[/]")
            return True, False

        kept_text = f"\n当前会话 {current_session_id} 会被保留。" if current_session_id else "\n当前未处于会话中，所有记忆都会被删除。"
        if not self._confirm_irreversible_delete(
                "硬删除全部历史",
                f"将删除 {len(target_session_ids)} 个 session 文件及其对应长期记忆。{kept_text}",
        ):
            return True, False

        result = self._delete_sessions_and_memories(target_session_ids, delete_all_memories=delete_all_memories)
        if current_session_id:
            # 删除其他会话记忆后刷新当前 system prompt，避免继续携带已删除记忆视图。
            self._reload_messages_after_turn_delete(current_session_id)
        else:
            self._reset_messages_for_session(record_system_message=False)
        self._print_delete_sessions_result(result)
        return True, False

    def _confirm_irreversible_delete(self, title: str, details: str) -> bool:
        self.console.print(
            Panel(
                f"[bold red]{details}[/]\n\n[red]该删除操作为高危操作，执行后不可逆。[/]",
                title=f"[bold red]{title}[/]",
                border_style="red",
            )
        )
        confirm_input = self.console.input("[bold red]输入 yes/y 确认执行删除：[/] ")
        return self._normalize_command(confirm_input) in {"yes", "y", "是", "确认", "对"}

    def _confirm_default_yes(self, prompt: str) -> bool:
        value = self._normalize_command(self.console.input(prompt))
        return value not in {"n", "no", "否", "不", "取消"}

    def _resolve_session_delete_targets(self, selector: str, sessions: list[dict[str, Any]]) -> list[str]:
        session_by_index = {
            str(index): item.get("session_id")
            for index, item in enumerate(sessions, start=1)
            if item.get("session_id")
        }
        known_session_ids = {item.get("session_id") for item in sessions if item.get("session_id")}
        target_session_ids = []
        for token in re.split(r"[\s,]+", selector or ""):
            if not token:
                continue
            session_id = session_by_index.get(token) if token.isdigit() else token
            if not session_id:
                self.console.print(f"[yellow]会话编号无效：{token}[/]")
                continue
            if session_id not in known_session_ids and not self.session_manager.load_session(session_id):
                self.console.print(f"[yellow]未找到会话：{session_id}[/]")
                continue
            if session_id not in target_session_ids:
                target_session_ids.append(session_id)
        return target_session_ids

    def _delete_memories_for_sessions(self, session_ids: list[str]) -> dict[str, Any]:
        # 硬删除历史 session 时保留旧 memory_saved 引用作为兼容输入，同时让 MemoryManager 按 task_index 反查新结构记录。
        refs = []
        if self.session_manager is not None:
            for session_id in session_ids:
                refs.extend(self.session_manager.session_memory_refs(session_id))
        task_ids = sorted({ref.get("task_id") for ref in refs if ref.get("task_id")})
        full_context_paths = sorted({ref.get("full_context_path") for ref in refs if ref.get("full_context_path")})
        if self.memory_manager is None:
            return {
                "task_ids": task_ids,
                "deleted_task_count": 0,
                "deleted_full_context_count": 0,
                "missing_full_context_count": 0,
            }
        result = self.memory_manager.delete_memories(
            task_ids=task_ids,
            session_ids=session_ids,
            full_context_paths=full_context_paths,
        )
        result["task_ids"] = result.get("deleted_task_ids") or task_ids
        return result

    def _delete_memories_for_session_turns(self, session_id: str, turn_ids: list[str]) -> dict[str, Any]:
        # 当前会话逻辑清空只处理最近一次 session_cleared 之后的可见任务，避免误删早先已保留的记忆。
        normalized_turn_ids = [turn_id for turn_id in turn_ids if turn_id]
        if self.memory_manager is None or not normalized_turn_ids:
            return {
                "task_ids": [],
                "deleted_task_ids": [],
                "deleted_task_count": 0,
                "deleted_full_context_count": 0,
                "missing_full_context_count": 0,
            }
        result = self.memory_manager.delete_memories(
            session_ids=[session_id],
            turn_ids=normalized_turn_ids,
        )
        result["task_ids"] = result.get("deleted_task_ids") or []
        return result

    def _delete_sessions_and_memories(
            self,
            session_ids: list[str],
            delete_all_memories: bool = False,
    ) -> dict[str, Any]:
        self._wait_for_pending_memory_updates()
        if self.memory_manager is not None:
            memory_result = (
                self.memory_manager.delete_all_memories()
                if delete_all_memories
                else self._delete_memories_for_sessions(session_ids)
            )
        else:
            memory_result = {"deleted_task_count": 0, "deleted_full_context_count": 0}

        deleted_sessions = []
        failed_sessions = []
        if self.session_manager is not None:
            for session_id in session_ids:
                result = self.session_manager.delete_session_file(session_id)
                if result.get("deleted"):
                    deleted_sessions.append(result)
                else:
                    failed_sessions.append(result)
        return {
            "memory": memory_result,
            "deleted_sessions": deleted_sessions,
            "failed_sessions": failed_sessions,
        }

    def _print_delete_sessions_result(self, result: dict[str, Any]) -> None:
        memory_result = result.get("memory") or {}
        deleted_sessions = result.get("deleted_sessions") or []
        failed_sessions = result.get("failed_sessions") or []
        self.console.print(
            "[dim]"
            f"已删除 session {len(deleted_sessions)} 个；"
            f"已删除记忆 {memory_result.get('deleted_task_count', 0)} 条；"
            f"已删除完整上下文文件 {memory_result.get('deleted_full_context_count', 0)} 个。"
            "[/]"
        )
        if failed_sessions:
            failed_text = ", ".join(item.get("session_id") or "-" for item in failed_sessions)
            self.console.print(f"[yellow]以下 session 删除失败或不存在：{failed_text}[/]")

    def _handle_cmd_bash_approve_on(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 开启 Bash 自动确认。
        self._update_bash_approve_status(True)
        return True, False

    def _handle_cmd_bash_approve_off(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 关闭 Bash 自动确认。
        self._update_bash_approve_status(False)
        return True, False

    def _update_bash_approve_status(self, enabled: bool):
        """
        更新 Bash 自动确认配置并刷新状态展示。
        :param enabled: 是否开启自动确认
        :return:
        """
        # 复用 on/off 两个命令的公共逻辑，避免重复代码。
        old_status = self._bash_approve_status_text()
        self.set_bash_auto_approve(enabled)
        new_status = self._bash_approve_status_text()
        if old_status != new_status:
            # 当确认策略发生变化后，自动刷新状态展示，用户可立即看到最新配置。
            self.console.print(f"[dim]已更新 Bash 命令确认策略：{new_status}[/]")
            self._print_runtime_status()
        else:
            # 若用户重复设置同一状态，明确告知并保持当前展示一致。
            self.console.print(f"[dim]Bash 命令确认策略未变化，当前为：{new_status}[/]")

    def _load_available_models(self) -> list[str]:
        """
        读取当前已配置的可用模型列表。
        :return:
        """
        # 从 model_config.json 读取模型列表，作为可切换模型来源。
        models = self._read_model_config().get("models", [])
        return [model_info.get("name") for model_info in models if isinstance(model_info, dict) and model_info.get("name")]

    def _handle_cmd_model_list(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 列出当前已配置模型；若暂无配置则先保留为空实现提示。
        available_models = self._load_available_models()
        if not available_models:
            self.console.print("[yellow]当前暂无可用模型配置，model_list/models 暂无可展示内容。[/]")
            return True, False
        self.console.print("[dim]已配置模型列表：[/]")
        for index, model_name in enumerate(available_models, start=1):
            marker = " [bold green]（当前）[/]" if model_name == self.model else ""
            self.console.print(f"[bold green]{index}[/]. [bold cyan]{model_name}[/]{marker}")
        self.console.print(
            "[dim]切换方式：[/][bold cyan]model[/] "
            "[bold yellow]<模型名|编号>[/][dim]，例如 [/][bold cyan]model[/] [bold green]1[/]"
        )
        return True, False

    def _handle_cmd_model_show_current(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 只输入 model 时展示模型相关命令，避免把 model 当成普通问题发给模型。
        self.console.print(f"[dim]当前使用模型：[/][bold cyan]{self.model}[/]")
        self.console.print("[bold cyan]model_list/models[/][dim]：查看可用模型配置[/]")
        self.console.print(
            "[bold cyan]model[/] [bold yellow]<模型名|编号>[/]"
            "[dim]：切换模型，例如 [/][bold cyan]model[/] [bold green]1[/]"
            "[dim] 或 [/][bold cyan]model[/] [bold cyan]gpt-4o-mini[/]"
        )
        self.console.print("[bold cyan]model add[/][dim]：新增模型配置[/]")
        self.console.print("[bold cyan]status[/][dim]：查看当前模型和 token 状态[/]")
        return True, False

    def _collect_model_add_config(self) -> PendingModelConfig | None:
        """
        收集 model add 需要的输入并做本地校验。
        :return: 待写入的模型配置；校验失败时返回 None
        """
        model_name = self.console.input("[bold cyan]请输入模型名：[/] ").strip()
        if not model_name:
            self.console.print("[yellow]模型名不能为空，已取消新增。[/]")
            return None
        if self._get_model_config_by_name(model_name):
            self.console.print(f"[yellow]模型 {model_name} 已存在，已取消新增。[/]")
            return None

        # 环境变量名由模型名生成，配置文件只保存变量名引用。
        base_url_env, api_key_env = self._model_env_var_names(model_name)
        self.console.print(
            "[dim]将写入当前用户环境变量：[/]"
            f"[bold cyan]{base_url_env}[/][dim] 和 [/][bold cyan]{api_key_env}[/]"
        )
        base_url = self.console.input("[bold cyan]请输入 base_url：[/] ").strip()
        api_key = self._input_masked_secret("[bold cyan]请输入 api_key：[/] ").strip()
        context_token_input = self.console.input("[bold cyan]请输入上下文窗口大小：[/] ").strip()

        if not base_url:
            self.console.print("[yellow]base_url 不能为空，已取消新增。[/]")
            return None
        try:
            max_model_context_token = int(context_token_input)
            if max_model_context_token <= 0:
                raise ValueError
        except ValueError:
            self.console.print("[yellow]上下文窗口大小必须是正整数，已取消新增。[/]")
            return None

        return PendingModelConfig(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            max_model_context_token=max_model_context_token,
        )

    def _test_pending_model_config(self, pending_config: PendingModelConfig) -> bool:
        """
        使用临时客户端测试待新增模型。
        :param pending_config: 待新增模型配置
        :return: 测试是否通过
        """
        self.console.print(f"[dim]正在测试新模型：[/][bold cyan]{pending_config.model_name}[/]")
        temp_client = None
        try:
            temp_client = self._build_openai_client(pending_config.base_url, pending_config.api_key)
            test_response = temp_client.chat.completions.create(
                model=pending_config.model_name,
                messages=[{"role": "user", "content": "Say 'Hello', don't do anything else"}],
            )
        except Exception as error:
            self.console.print(
                f"[yellow]模型信息错误，请检查模型名、base_url、api_key 和服务状态。"
                f"错误类型：{type(error).__name__}；错误信息：{error}[/]"
            )
            return False
        finally:
            close_temp_client = getattr(temp_client, "close", None)
            if callable(close_temp_client):
                close_temp_client()

        test_content = getattr(test_response.choices[0].message, "content", "")
        self.console.print(f"[green]模型测试成功。响应：[/][dim]{test_content}[/]")
        return True

    def _write_pending_model_config(self, pending_config: PendingModelConfig) -> None:
        """
        写入环境变量和模型配置文件。
        :param pending_config: 待新增模型配置
        :return:
        """
        # 先让当前进程立即可用，持久化环境变量交给后台线程处理。
        self._persist_environment_variables_async(
            {
                pending_config.base_url_env: pending_config.base_url,
                pending_config.api_key_env: pending_config.api_key,
            }
        )
        self.console.print(
            "[dim]环境变量已写入当前进程，本次运行可立即使用；"
            "持久化到 Windows 用户环境变量将在后台继续执行，请勿立即退出当前程序。[/]"
        )

        config = self._read_model_config()
        config.setdefault("models", [])
        config["models"].append(
            {
                "name": pending_config.model_name,
                "base_url_env": pending_config.base_url_env,
                "api_key_env": pending_config.api_key_env,
                "max_model_context_token": pending_config.max_model_context_token,
            }
        )
        self._write_model_config(config)
        self.console.print("[green]模型配置已写入 agent/config/model_config.json（仅保存环境变量名）[/]")

    def _switch_to_pending_model_if_requested(
            self,
            pending_config: PendingModelConfig,
            confirm_choice: tuple[str, ...],
    ) -> None:
        """
        根据用户确认决定是否立即切换到新增模型。
        :param pending_config: 已写入的新增模型配置
        :param confirm_choice: 确认输入集合
        :return:
        """
        switch_input = self.console.input("[bold cyan]是否立即切换到新模型？(yes/y)[/] ").strip()
        if self._normalize_command(switch_input) in confirm_choice:
            self.model = pending_config.model_name
            if not self._apply_model_config_by_name(self.model):
                missing_names = ", ".join(self._missing_model_env_vars)
                self.console.print(f"[yellow]新模型环境变量缺失：{missing_names}[/]")
                return
            self._refresh_token_context_window()
            self.console.print(
                f"[green]模型已切换为：{self.model}（上下文窗口：{self._max_context_tokens}）[/]"
            )
            self._print_runtime_status()

    def _handle_cmd_model_add(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增模型前先真实发送一条测试消息，避免把不可用配置写入模型列表。
        pending_config = self._collect_model_add_config()
        if pending_config is None:
            return True, False
        if not self._test_pending_model_config(pending_config):
            return True, False
        self._write_pending_model_config(pending_config)
        self._switch_to_pending_model_if_requested(pending_config, confirm_choice)
        return True, False

    def _resolve_model_selector(self, selector: str, available_models: list[str]) -> str | None:
        normalized_selector = (selector or "").strip()
        if not normalized_selector:
            return None
        if normalized_selector.isdigit():
            selected_index = int(normalized_selector)
            if selected_index < 1 or selected_index > len(available_models):
                self.console.print(f"[yellow]模型编号超出范围：{normalized_selector}[/]")
                return None
            # 编号和 model_list/models 展示顺序保持一致。
            return available_models[selected_index - 1]
        return normalized_selector

    def _handle_cmd_model_switch(self, _confirm_choice: tuple[str, ...], cmd: str) -> tuple[bool, bool]:
        # 处理 model <model_name|编号> 命令，校验配置后执行切换。
        target_selector = cmd.split(" ", 1)[1].strip()
        self.console.print(f"[dim]当前使用模型：{self.model}[/]")
        if not target_selector:
            self.console.print("[yellow]请输入目标模型，例如：model 1 或 model gpt-4o-mini[/]")
            return True, False
        available_models = self._load_available_models()
        target_model = self._resolve_model_selector(target_selector, available_models)
        if target_model is None:
            return True, False
        if target_model not in available_models:
            self.console.print(f"[yellow]模型 {target_model} 未配置，请先使用 model add 新增并测试。[/]")
            return True, False

        self.model = target_model
        if not self._apply_model_config_by_name(self.model):
            missing_names = ", ".join(self._missing_model_env_vars)
            self.console.print(f"[yellow]模型 {target_model} 的环境变量缺失：{missing_names}[/]")
            return True, False
        self._refresh_token_context_window()
        self.console.print(f"[green]模型已切换为：{self.model}（上下文窗口：{self._max_context_tokens}）[/]")
        self._print_runtime_status()
        return True, False

    def _handle_cmd_compact_history(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 主动触发当前会话历史压缩。
        self._compact_messages(self._all_tools)
        self.console.print("[dim]已执行会话压缩检查（达到阈值时会执行压缩）。[/]")
        return True, False

    def _handle_cmd_memory_compact(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        if self.memory_manager is None:
            self.console.print("[yellow]当前没有可整理的长期记忆。[/]")
            return True, False
        result = self.memory_manager.compact_memory_index(force=True, reason="manual")
        if result.get("status") == "compacted":
            self.console.print(f"[dim]MEMORY.md 已整理：{result.get('memory_index_path')}[/]")
        else:
            self.console.print("[dim]当前没有值得整理的新长期记忆。[/]")
        return True, False

    def _handle_cmd_memory_summarize_sessions(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        if self.session_manager is None or self.memory_manager is None:
            self.console.print("[yellow]当前没有可整理的历史 session 或长期记忆管理器。[/]")
            return True, False
        self._wait_for_pending_memory_updates()
        sessions = self.session_manager.list_sessions(limit=10000, include_empty=False)
        candidates = self._collect_unchecked_session_tasks(sessions)
        if not candidates:
            self.console.print("[dim]没有发现需要纳入新记忆系统的历史任务。[/]")
            return True, False

        self.console.print(f"[dim]发现 {len(candidates)} 个未经过新记忆系统检查的历史任务。[/]")
        if not self._confirm_default_yes("[bold cyan]是否开始整理这些历史任务？(Y/n，默认开始)[/] "):
            return True, False

        scheduled = 0
        for item in candidates:
            future = self.memory_manager.enqueue(
                task=item["task"],
                result=item["result"],
                context=[
                    {"role": "user", "content": item["task"]},
                    {"role": "assistant", "content": item["result"]},
                ],
                session_id=item["session_id"],
                turn_id=item["turn_id"],
                last_memory_checked_event_id=item["last_event_id"],
            )
            if future is not None:
                scheduled += 1
                self._attach_memory_future_session_callback(
                    future,
                    session_id=item["session_id"],
                    turn_id=item["turn_id"],
                )
        self._wait_for_pending_memory_updates()
        self.console.print(f"[dim]历史任务整理完成：已检查 {scheduled} 个任务。[/]")
        return True, False

    def _collect_unchecked_session_tasks(self, sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = []
        for session in sessions:
            session_id = session.get("session_id")
            if not session_id:
                continue
            events = self.session_manager.load_session(session_id)
            checked_turn_ids = {
                event.get("turn_id")
                for event in events
                if event.get("event") == "memory_checked" and event.get("turn_id")
            }
            tasks = self.session_manager.list_tasks(session_id)
            last_event_id = self.session_manager.latest_event_id(session_id)
            for task in tasks:
                turn_id = task.get("turn_id")
                if not turn_id or turn_id in checked_turn_ids:
                    continue
                candidates.append(
                    {
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "task": task.get("content") or "",
                        "result": task.get("final_output") or "",
                        "last_event_id": last_event_id,
                    }
                )
        return candidates

    def _handle_cmd_status(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 展示当前模型状态，优先使用 session 中最新 assistant_response 的真实 usage。
        tracker = self._ensure_token_tracker()
        session_usage = self._restore_session_usage_summary()
        if session_usage.get("has_real_usage"):
            used_tokens = int(session_usage.get("total_tokens") or 0)
        else:
            # status 只展示模型 API 返回的真实 usage；还没有模型调用时消耗为 0。
            used_tokens = 0
        # status 命令展示 token 使用柱状图，直观反馈使用率。
        usage_ratio = tracker.calculate_usage_ratio(used_tokens, self._max_context_tokens)
        token_bar = tracker.render_usage_bar(usage_ratio)
        self.console.print(f"[dim]模型名：{self.model}[/]")
        self.console.print(f"[dim]已使用上下文 Token：{used_tokens}[/]")
        self.console.print(f"[dim]上下文窗口：{self._max_context_tokens}[/]")
        self.console.print(f"[dim]上下文使用率：{usage_ratio * 100:.2f}%[/]")
        if session_usage.get("is_stale"):
            self.console.print("[yellow]Token 用量已因删除中间任务失效，下一个问题后会用真实 usage 更新。[/]")
        # 按评审意见直接打印柱状图，不额外增加提示前缀。
        self.console.print(token_bar)
        return True, False

    def _handle_cmd_history(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # history 只展示真实用户任务，不展示 assistant/tool 等底层消息。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        if self.session_manager.current_session_id is None:
            self.console.print("[yellow]当前还没有已保存的用户任务。[/]")
            return True, False

        while True:
            tasks = self.session_manager.list_tasks()
            if not tasks:
                self.console.print("[yellow]当前会话没有可展示的用户任务。[/]")
                return True, False

            result = open_task_history_viewer(tasks)
            if result.action == "delete" and result.task:
                # 删除会改变 session 事件和短期上下文，完成后重新拉取任务列表继续展示 history。
                self._delete_history_task(result.task)
                continue
            return True, False

    def _delete_history_task(self, task: dict[str, Any]) -> bool:
        """
        删除 history 中选中的用户任务。
        :param task: history 任务项
        :return: 是否删除成功
        """
        if self.session_manager is None:
            return False
        turn_id = task.get("turn_id")
        if not turn_id:
            self.console.print("[yellow]无法删除：任务缺少 turn_id。[/]")
            return False

        session_id = self.session_manager.current_session_id
        self._wait_for_pending_memory_updates()
        tasks = self.session_manager.list_tasks()
        selected_index = next((index for index, item in enumerate(tasks) if item.get("turn_id") == turn_id), None)
        if selected_index is None:
            self.console.print("[yellow]无法删除：任务已经不存在或当前会话已被清空。[/]")
            return False

        following_tasks = tasks[selected_index + 1:]
        delete_following = False
        if following_tasks:
            self.console.print(
                f"[yellow]任务 {task.get('index')} 之后还有 {len(following_tasks)} 个任务，"
                "这些任务可能依赖被删除任务的上下文，推荐一并删除。[/]"
            )
            delete_following = self._confirm_default_yes(
                "[bold cyan]是否同时删除该任务之后的所有任务？(Y/n，默认删除)[/] "
            )
        tasks_to_delete = tasks[selected_index:] if delete_following else [tasks[selected_index]]
        delete_memory = self._confirm_default_yes(
            "[bold cyan]是否同时硬删除这些任务对应的长期记忆？(Y/n，默认删除)[/] "
        )
        memory_task_ids = self._history_task_memory_ids(tasks_to_delete)
        turn_ids_to_delete = [
            item.get("turn_id")
            for item in tasks_to_delete
            if item.get("turn_id")
        ]
        try:
            memory_result = None
            deleted_memory_task_ids = memory_task_ids
            deleted_task_ids_by_turn = {}
            if delete_memory and self.memory_manager is not None:
                # 新记忆系统里 skipped 任务没有 memory_saved 事件，因此必须按 turn_id 反查 task_index。
                memory_result = self.memory_manager.delete_memories(
                    task_ids=memory_task_ids,
                    session_ids=[session_id] if session_id else None,
                    turn_ids=turn_ids_to_delete,
                )
                deleted_memory_task_ids = memory_result.get("deleted_task_ids") or memory_task_ids
                deleted_task_ids_by_turn = memory_result.get("deleted_task_ids_by_turn") or {}
            for item in tasks_to_delete:
                item_turn_id = item.get("turn_id")
                if not item_turn_id:
                    continue
                item_task_ids = deleted_task_ids_by_turn.get(item_turn_id) or [
                    task_id
                    for task_id in (item.get("memory_task_ids") or [])
                    if task_id in deleted_memory_task_ids
                ]
                self.session_manager.mark_turn_deleted(
                    turn_id=item_turn_id,
                    task_ids=item_task_ids,
                )
            if session_id:
                self._reload_messages_after_turn_delete(session_id)
        except Exception as error:
            self.console.print(f"[yellow]删除任务失败：{error}[/]")
            return False

        deleted_indexes = ", ".join(str(item.get("index")) for item in tasks_to_delete)
        self.console.print(f"[dim]已删除任务 {deleted_indexes}。[/]")
        if memory_result is not None:
            self.console.print(
                f"[dim]已硬删除长期记忆 {memory_result.get('deleted_task_count', 0)} 条，"
                f"完整上下文文件 {memory_result.get('deleted_full_context_count', 0)} 个。[/]"
            )
        elif delete_memory:
            self.console.print("[dim]未找到需要硬删除的长期记忆。[/]")
        if following_tasks and not delete_following:
            self.console.print("[yellow]已保留后续任务；当前 token 用量记录已失效，下一个问题后会用真实 usage 更新。[/]")
        return True

    @staticmethod
    def _history_task_memory_ids(tasks: list[dict[str, Any]]) -> list[str]:
        """
        提取 history 任务关联的长期记忆 task_id。
        :param tasks: 待删除的 history 任务
        :return: 去重后的 task_id 列表
        """
        memory_task_ids = []
        for task in tasks:
            for task_id in task.get("memory_task_ids") or []:
                if task_id and task_id not in memory_task_ids:
                    memory_task_ids.append(task_id)
        return memory_task_ids

    def _reload_messages_after_turn_delete(self, session_id: str) -> None:
        """
        删除 turn 后重建当前短期上下文。
        :param session_id: 当前会话 id
        :return:
        """
        if self.session_manager is None:
            return
        messages = self.session_manager.rebuild_messages(session_id)
        # 删除长期记忆后刷新 system prompt，避免被软删除任务继续出现在当前上下文中。
        current_system_prompt = self._build_system_prompt()
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": current_system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": current_system_prompt})
        self.messages = messages
        self._ensure_token_tracker().set_estimated_usage(self.messages, self._all_tools)
        self._restore_session_usage_summary(session_id)

    def _reset_messages_for_session(self, record_system_message: bool):
        """
        重置当前短期上下文为新的 system prompt。
        :param record_system_message: 是否把 system message 写入当前 session
        :return:
        """
        self._cached_system_prompt = self._build_system_prompt()
        self.messages = [{"role": "system", "content": self._cached_system_prompt}]
        self._current_task_full_context = None
        self._current_task_start_index = None
        self._current_turn_id = None
        self._ensure_token_tracker().reset()
        if record_system_message:
            # 新会话需要把 system prompt 作为可恢复上下文的第一条消息。
            self._record_session_message(self.messages[0], turn_id=None)

    def _load_messages_from_session(self, session_id: str) -> list[dict[str, Any]]:
        """
        从历史 session 重建短期上下文。
        :param session_id: 会话 id
        :return: OpenAI messages
        """
        if self.session_manager is None:
            return []
        messages = self.session_manager.rebuild_messages(session_id)
        system_prompt = self._build_system_prompt()
        if not messages or messages[0].get("role") != "system":
            # 老会话缺少 system message 时，只在内存中补齐，不反向改写历史文件。
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # 长期记忆可能已被清理，恢复会话时用最新 system prompt 替换历史快照。
            messages[0] = {"role": "system", "content": system_prompt}
        return messages

    def _format_session_time(self, value: Any) -> str:
        # 会话列表只做轻量展示，缺失时间用短横线占位。
        return str(value or "-").replace("T", " ")

    def _format_session_label(self, session: dict[str, Any]) -> str:
        """
        格式化会话名称，便于提示中展示。
        :param session: 会话索引记录
        :return: 标题和 session_id
        """
        title = session.get("title") or "未命名会话"
        session_id = session.get("session_id") or "-"
        return f"{title}（{session_id}）"

    def _print_sessions_table(self, sessions: list[dict[str, Any]]) -> None:
        """
        打印会话列表。
        :param sessions: 会话索引列表
        :return:
        """
        if not sessions:
            return

        table = Table(title="最近会话", show_lines=False)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("标题", overflow="fold")
        table.add_column("更新时间", no_wrap=True)
        table.add_column("状态", no_wrap=True)
        table.add_column("session_id", no_wrap=True)
        for index, item in enumerate(sessions, start=1):
            marker = " *" if item.get("session_id") == self.session_manager.current_session_id else ""
            table.add_row(
                str(index),
                f"{item.get('title') or '未命名会话'}{marker}",
                self._format_session_time(item.get("updated_at") or item.get("created_at")),
                item.get("status") or "active",
                item.get("session_id") or "",
            )
        self.console.print(table)

    def _print_continue_skipped_empty_sessions(self, target_session_id: str | None) -> None:
        """
        打印 continue 命令因空会话过滤而跳过的最近会话。
        :param target_session_id: 最终要加载的 session_id
        :return:
        """
        if self.session_manager is None or not target_session_id:
            return
        recent_sessions = self.session_manager.list_sessions(limit=100, include_empty=True)
        skipped_sessions = []
        for session in recent_sessions:
            session_id = session.get("session_id")
            if session_id == target_session_id:
                break
            skipped_sessions.append(session)
        else:
            return
        if not skipped_sessions:
            return

        visible_skipped = skipped_sessions[:3]
        skipped_text = "；".join(self._format_session_label(session) for session in visible_skipped)
        if len(skipped_sessions) > len(visible_skipped):
            skipped_text += f"；还有 {len(skipped_sessions) - len(visible_skipped)} 个"
        self.console.print(f"[dim]continue 已跳过最近空会话：{skipped_text}[/]")

    def _handle_cmd_sessions(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 列出最近会话，序号可直接用于 session load。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        sessions = self.session_manager.list_sessions(limit=20)
        if not sessions:
            self.console.print("[yellow]暂无历史会话。[/]")
            return True, False

        self._print_sessions_table(sessions)
        return True, False

    def _handle_cmd_session_current(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 展示当前会话完整定位信息，便于客户端化前人工排查。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        if self.session_manager.current_session_id is None:
            pending_title = self._pending_session_title or "未命名会话"
            self.console.print("[dim]当前还没有已保存会话，提问新内容以创建新会话。[/]")
            self.console.print(f"[dim]待创建标题：{pending_title}[/]")
            return True, False
        info = self.session_manager.get_current_session_info()
        self.console.print(f"[dim]session_id：{info.get('session_id')}[/]")
        self.console.print(f"[dim]标题：{info.get('title') or '未命名会话'}（来源：{info.get('title_source') or 'default'}）[/]")
        self.console.print(f"[dim]模型：{info.get('model') or self.model}[/]")
        self.console.print(f"[dim]状态：{info.get('status') or 'active'}[/]")
        self.console.print(f"[dim]创建时间：{self._format_session_time(info.get('created_at'))}[/]")
        self.console.print(f"[dim]路径：{info.get('path') or '-'}[/]")
        return True, False

    def _handle_cmd_session_new(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 创建新会话前先等待旧任务归档，避免 memory_saved 被写入新 session。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        self._wait_for_pending_memory_updates()
        self._end_session()
        self._enqueue_session_end_memory_compaction()
        self.session_manager.detach_current_session()
        self._session_title_auto_started = False
        self._session_interrupted_recorded = False
        self._pending_session_title = None
        self._pending_session_title_source = None
        self._reset_messages_for_session(record_system_message=False)
        self.console.print("[dim]已准备新会话，第一条用户任务后会自动保存。[/]")
        return True, False

    def _resolve_session_load_selector(self, selector: str) -> str | None:
        """
        把用户输入的序号或 session_id 解析为 session_id。
        :param selector: 序号或 session_id
        :return: session_id
        """
        if self.session_manager is None:
            return None
        normalized_selector = str(selector or "").strip()
        if not normalized_selector:
            return None
        if normalized_selector.isdigit():
            sessions = self.session_manager.list_sessions(limit=20)
            selected_index = int(normalized_selector)
            if selected_index < 1 or selected_index > len(sessions):
                self.console.print(f"[yellow]会话序号超出范围：{normalized_selector}[/]")
                return None
            # 序号与 sessions 命令展示顺序保持一致。
            return sessions[selected_index - 1].get("session_id")
        return normalized_selector

    def _load_session_by_id(self, session_id: str) -> bool:
        """
        加载指定历史会话。
        :param session_id: 会话 id
        :return: 是否加载成功
        """
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return False
        target_session_id = (session_id or "").strip()
        if not target_session_id:
            self.console.print("[yellow]请输入 session_id，例如：session load session_20260521_120000_abcdef[/]")
            return False

        current_session_id = self.session_manager.get_current_session_info().get("session_id")
        try:
            if not self.session_manager.load_session(target_session_id):
                self.console.print(f"[yellow]未找到会话：{target_session_id}[/]")
                return False
            self._wait_for_pending_memory_updates()
            if current_session_id and current_session_id != target_session_id:
                self._end_session()
            info = self.session_manager.switch_session(target_session_id)
            if info.get("resume_event_id"):
                self.session_manager.record_memory_checked(
                    turn_id=None,
                    last_memory_checked_event_id=info.get("resume_event_id"),
                    status="cursor_only",
                    metadata={"reason": "session_resumed"},
                )
            self.messages = self._load_messages_from_session(info["session_id"])
            self._current_task_full_context = None
            self._current_task_start_index = None
            self._current_turn_id = None
            self._ensure_token_tracker().set_estimated_usage(self.messages, self._all_tools)
            self._restore_session_usage_summary(info["session_id"])
            # 继续历史会话时不再把下一条用户消息当作“第一个任务”自动改标题。
            self._session_title_auto_started = True
            self._session_interrupted_recorded = False
            self._pending_session_title = None
            self._pending_session_title_source = None
        except Exception as error:
            self.console.print(f"[yellow]会话加载失败：{error}[/]")
            return False

        self.console.print(f"[dim]已加载会话：{info.get('title') or '未命名会话'}（{info.get('session_id')}）[/]")
        return True

    def _handle_cmd_session_load(self, _confirm_choice: tuple[str, ...], session_id: str) -> tuple[bool, bool]:
        # 加载历史会话会替换当前短期上下文，但加载动作本身不写入任何 session。
        target_session_id = self._resolve_session_load_selector(session_id)
        if target_session_id:
            self._load_session_by_id(target_session_id)
        return True, False

    def _handle_cmd_session_load_prompt(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 不带参数时展示列表，让用户输入序号或 session_id。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        sessions = self.session_manager.list_sessions(limit=20)
        if not sessions:
            self.console.print("[yellow]暂无历史会话。[/]")
            return True, False
        self._print_sessions_table(sessions)
        selector = self.console.input("[bold cyan]请输入要加载的会话序号或 session_id：[/] ").strip()
        target_session_id = self._resolve_session_load_selector(selector)
        if target_session_id:
            self._load_session_by_id(target_session_id)
        return True, False

    def _handle_cmd_continue_session(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 继续上一次会话。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        sessions = self.session_manager.list_sessions(limit=1, include_empty=False)
        if not sessions:
            self.console.print("[yellow]没有可继续的历史会话。[/]")
            return True, False
        target_session_id = sessions[0].get("session_id")
        self._print_continue_skipped_empty_sessions(target_session_id)
        self._load_session_by_id(target_session_id)
        return True, False

    def _handle_cmd_session_title_show(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 展示当前会话标题，便于用户确认客户端列表中会看到的名称。
        if self.session_manager is None:
            self.console.print("[yellow]当前没有可管理的主会话。[/]")
            return True, False
        if self.session_manager.current_session_id is None:
            title = self._pending_session_title or "未命名会话"
            source = self._pending_session_title_source or "default"
            self.console.print(f"[dim]待创建会话标题：{title}（来源：{source}）[/]")
            return True, False
        session_info = self.session_manager.get_current_session_info()
        title = session_info.get("title") or "未命名会话"
        source = session_info.get("title_source") or "default"
        self.console.print(f"[dim]当前会话标题：{title}（来源：{source}）[/]")
        return True, False

    def _handle_cmd_session_title_set(self, _confirm_choice: tuple[str, ...], title: str) -> tuple[bool, bool]:
        # 用户手动设置的标题优先级最高，后续异步自动标题不会覆盖它。
        self._set_session_title(title, source="user", silent=False)
        return True, False

"""
Team 类管理多个Agent，
"""

from multi_agent import (
    Team,
    TeamOrchestrator,
    build_table_game_runtime_config,
    create_table_game_orchestrator,
    plan_team,
    run_team,
)

if __name__ == "__main__":
    from agent_run import AgentRunner
    runner = AgentRunner(
        model="minimax-m2.7:cloud",
        mcp_mode="subprocess",
    )
    runner.run()
