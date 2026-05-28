# encoding : utf-8
# @Time    : 2026/4/19
import atexit
import httpx
import json
import os
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


@dataclass
class ToolFailureGuardState:
    """单轮 Agent 执行中的工具失败熔断状态。"""

    active_tools: list[dict[str, Any]]
    disabled_tools: set[str]
    tools_without_make_plan: list[dict[str, Any]] | None = None
    last_failure_key: tuple[str, str] | None = None
    consecutive_failure_count: int = 0

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

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


class Agent:
    """支持本地工具 + MCP工具的Agent。"""

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
    # 目前没有默认加入 after hook，保留给后续控制超长输出时使用。
    _BASH_MAX_OUTPUT_LENGTH = 5000
    _DEFAULT_CONTEXT_WINDOW = 32768
    _COMPACT_TRIGGER_RATIO = 0.8
    _MIDDLE_COMPACT_RATIO = 0.3
    _KEEP_RECENT = 10
    _MAX_CONSECUTIVE_TOOL_FAILURES = 2

    def __init__(self, model="qwen3.5:9b",
                 temperature: float = 0.1,
                 base_url: str = None,
                 api_key: str = None,
                 mcp_client=None,
                 is_main_agent: bool = True,
                 role: str = "Main Agent",
                 name: str = None):
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

        # base_url
        self._base_url = os.environ.get("OPENAI_BASE_URL") if base_url is None else base_url

        # api_key
        self._api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key

        # OpenAI 请求客户端
        self._model_client_bypass_proxy = False
        self.client = self._create_openai_client(self._base_url, self._api_key)

        # 是否是主 Agent，False 表示由 Agent 创建的子 Agent，默认为 True。
        self._is_main_agent = is_main_agent

        # 最大迭代次数
        self.max_iterations = 100

        # 使用模型
        self.model = model
        # 根据传入模型从 model_config.json 读取模型配置（base_url/api_key/上下文窗口等）。
        self._apply_model_config_by_name(self.model)
        self._max_context_tokens = self._load_model_context_window()
        # 记录最近一次模型调用的已使用 token 和总 token（来自 OpenAI 标准 usage 字段）。
        self._used_token = 0
        self._total_token = 0

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
            if self._is_main_agent
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
        self.mcp_client = mcp_client
        self._prepare_mcp_client()
        self.console = Console()
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
        self._local_tools = self._tool_manager.local_tools
        self._local_functions = self._tool_manager.local_functions
        self._mcp_tools = self._tool_manager.mcp_tools
        self._available_functions = self._tool_manager.available_functions
        self._all_tools = self._tool_manager.all_tools
        self._all_tools_without_make_plan = self._build_tools_without_make_plan(self._all_tools)
        print(f"{len(self._local_tools)} local tools loaded")
        if self.mcp_client:
            print(f"{len(self._mcp_tools)} MCP tools loaded")
        else:
            print("No MCP client provided, MCP tools not loaded")

        # 基础提示词，用于主 Agent。
        self._base_prompt_main_agent = load_prompt("base_main_agent.md")

        # 子 Agent 提示词。
        self._base_prompt_sub_agent = load_prompt("base_sub_agent.md", role=role)

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
            print("传入的 MCP 客户端不可用：ping 不通且没有 start 方法，将只使用本地工具")
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

        print(f"传入的 MCP 客户端不可用：尝试启动 {max_attempts} 次后仍 ping 不通，将只使用本地工具。最后错误：{last_error}")
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

    def _apply_model_config_by_name(self, model_name: str):
        """
        根据模型名应用模型配置（base_url/api_key）。
        :param model_name: 模型名
        :return:
        """
        # 在不改变初始化接口的前提下，按模型配置自动补全连接信息。
        model_info = self._get_model_config_by_name(model_name)
        if not model_info:
            return
        self._base_url = model_info.get("base_url") or self._base_url
        self._api_key = model_info.get("api_key") or self._api_key
        self.client = self._create_openai_client(self._base_url, self._api_key)
        self._sync_model_runtime_dependencies()

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
        # 长期记忆摘要异步执行，避免拖慢当前任务完成。
        future = self.memory_manager.enqueue(
            task=task,
            result=result,
            context=self._current_task_full_context or [],
            session_id=self.session_manager.current_session_id if self.session_manager is not None else None,
            turn_id=self._current_turn_id,
        )
        if future is None or self.session_manager is None or not self._current_turn_id:
            return

        turn_id = self._current_turn_id

        def _record_memory_saved(done_future):
            try:
                index_record = done_future.result()
                if not index_record:
                    return
                # 只有长期记忆真正落盘后，才把 task_id 关联到当前 turn。
                self.session_manager.record_memory_saved(
                    turn_id=turn_id,
                    task_id=index_record.get("task_id"),
                    full_context_path=index_record.get("full_context_path"),
                )
            except Exception as error:
                try:
                    self.session_manager.append_event(
                        {
                            "event": "memory_save_error",
                            "turn_id": turn_id,
                            "error": str(error),
                        }
                    )
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
            self.console.print("[dim]还有记忆整理任务未完成，正在等待完成后退出...[/]")
        self.memory_manager.shutdown()

    def _wait_for_pending_memory_updates(self):
        """
        等待当前仍在运行的记忆整理任务，但不关闭 MemoryManager。
        :return:
        """
        if not self._is_main_agent or self.memory_manager is None:
            return
        if self.memory_manager.has_pending():
            self.console.print("[dim]还有记忆整理任务未完成，正在等待完成后切换会话...[/]")
        # 切换会话前只等待后台任务归档完成，后续新任务仍需要继续写长期记忆。
        self.memory_manager.wait_for_pending()

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
        return session_id

    def set_bash_auto_approve(self, enabled: bool):
        """
        设置当前 Agent 的 Bash 命令是否自动确认执行。
        :param enabled: True 表示自动确认，False 表示需要手动确认
        :return:
        """
        # 支持运行时切换 Bash 自动确认配置，便于用户动态控制安全策略。
        self._BASH_AUTO_APPROVE = bool(enabled)

    def _bash_approve_status_text(self) -> str:
        """
        返回当前 Bash 执行确认策略的文本描述。
        :return:
        """
        # 统一管理状态文案，避免 run() 中重复拼接字符串。
        if self._BASH_AUTO_APPROVE:
            return "自动确认（无需手动确认）"
        return "手动确认（每次需确认）"

    def _record_session_message(self, message, turn_id=None, metadata=None):
        """
        把短期上下文消息同步写入当前 session。
        :param message: OpenAI message
        :param turn_id: 当前用户任务 id
        :param metadata: 只写入 session 的辅助元信息
        :return:
        """
        if self.session_manager is None or self.session_manager.current_session_id is None:
            return
        try:
            # 会话记录失败不应该打断 Agent 正常回答。
            self.session_manager.append_message(message, turn_id=turn_id, metadata=metadata)
        except Exception as error:
            self.console.print(f"[yellow]会话记录写入失败：{error}[/]")

    def _append_message(self, message, capture_full_context=True, session_metadata=None):
        self.messages.append(message)
        if capture_full_context and self._current_task_full_context is not None:
            self._current_task_full_context.append(self._normalize_message_for_memory(message))
        self._record_session_message(message, turn_id=self._current_turn_id, metadata=session_metadata)

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
        # 每次模型调用后更新 token 使用统计。
        self._update_usage_from_response(response)
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

    def _message_text(self, message):
        if isinstance(message, dict):
            parts = [str(message.get("role", "")), str(message.get("content", ""))]
            if message.get("tool_calls"):
                parts.append(json.dumps(message.get("tool_calls"), ensure_ascii=False, default=str))
            if message.get("tool_call_id"):
                parts.append(str(message.get("tool_call_id")))
            return "\n".join(part for part in parts if part)

        parts = [str(getattr(message, "role", "")), str(getattr(message, "content", ""))]
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            parts.append(json.dumps(tool_calls, ensure_ascii=False, default=str))
        return "\n".join(part for part in parts if part)

    def _estimate_text_tokens(self, text):
        text = text or ""
        ascii_count = sum(1 for char in text if ord(char) < 128)
        non_ascii_count = len(text) - ascii_count
        return max(1, ascii_count // 4 + non_ascii_count * 2)

    def _estimate_messages_tokens(self, messages, tools=None):
        total = 0
        for message in messages:
            total += 4 + self._estimate_text_tokens(self._message_text(message))
        if tools:
            total += self._estimate_text_tokens(json.dumps(tools, ensure_ascii=False, default=str))
        return total

    def _should_compact_messages(self, tools=None):
        used_tokens = self._estimate_messages_tokens(self.messages, tools)
        return used_tokens >= int(self._max_context_tokens * self._COMPACT_TRIGGER_RATIO)
    
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
            content = self._message_text(message)
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
        # 每次模型调用后更新 token 使用统计。
        self._update_usage_from_response(summary_response)
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
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=active_tools,
                temperature=self.temperature,
                stream=True,
            )
            return self._deal_stream_response(response_stream, spinner=spinner)
        finally:
            spinner.stop()

    def _append_assistant_response(self, message):
        tool_calls = getattr(message, "tool_calls", None)
        self._append_message({
            "role": "assistant",
            "content": message.content,
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

    def _append_tool_result(self, task: ToolCallTask, function_response):
        self._append_message(
            {
                "role": "tool",
                "tool_call_id": task.tool_call_id,
                "content": json.dumps(function_response, ensure_ascii=False),
            }
        )

    def _append_tool_result_and_check_guard(
            self,
            task: ToolCallTask,
            function_response,
            guard_state: ToolFailureGuardState,
    ):
        self._append_tool_result(task, function_response)
        if not self._is_tool_call_failure(function_response):
            guard_state.last_failure_key = None
            guard_state.consecutive_failure_count = 0
            return None

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
            result = self._run_agent_step(tools_without_make_plan)
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

    def _run_agent_step(self, tools):
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
                return message.content

            print(f"[Iter {i}]: message is: {message}")
            stop_reason = self._handle_tool_calls(message.tool_calls, guard_state)
            if stop_reason:
                return stop_reason
        return "Max iterations reached"

    def _should_show_tool_spinner(self) -> bool:
        # 并发工具在工作线程中执行，避免多个 Live 动画同时刷新控制台。
        return current_thread().name == "MainThread"

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
                if self._should_show_tool_spinner() and task.function_name != ToolNameConstant.WEB_SEARCH else None
            try:
                result = function_impl(**task.function_args)
            finally:
                if spinner is not None:
                    spinner.stop()
            # if task.function_name == ToolNameConstant.WEB_SEARCH:
            #     self._print_network_search_results()
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
        memory = self._load_memory_view()
        rules = self._load_rules(memory)
        skills = self._load_skill_meta_infos()
        base_prompt = [
            self._base_prompt_main_agent if self._is_main_agent else self._base_prompt_sub_agent,
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
        self._update_usage_from_response(chunk)
        choice = chunk.choices[0]
        delta = choice.delta

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
            content=content if content else None,
            tool_calls=ordered_tool_calls if ordered_tool_calls else None,
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

    def _update_usage_from_response(self, response):
        """
        从 OpenAI 响应对象中提取 usage 并更新 token 统计。
        :param response: 非流式 response 或流式 chunk
        :return:
        """
        # 统一处理标准 OpenAI usage 字段，避免各处重复解析逻辑。
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        # 优先使用 total_tokens；若为空则回退为 prompt+completion。
        if isinstance(total_tokens, int):
            self._used_token = total_tokens
            self._total_token = total_tokens
            return

        prompt = prompt_tokens if isinstance(prompt_tokens, int) else 0
        completion = completion_tokens if isinstance(completion_tokens, int) else 0
        estimated_total = prompt + completion
        if estimated_total > 0:
            self._used_token = estimated_total
            self._total_token = estimated_total

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
                          is_main_agent=False
                          )
        final_result, _ = sub_agent.chat(task)
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
                # 每次模型调用后更新 token 使用统计。
                self._update_usage_from_response(resp)
                self._append_message(resp.choices[0].message)
                self.inbox.clear()

            # 再拼接本次任务并执行
            self._maybe_auto_title_session(task)
            self._append_message(
                {"role": "user", "content": task},
                session_metadata={"is_task_entry": True},
            )
            final_result = self._run_agent_step(self._all_tools)
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
                self.chat(user_input)
                self.console.print()
            except OpenAIError as error:
                self._print_model_call_error(error)
                self.console.print()
            except KeyboardInterrupt:
                self._record_session_interrupted("keyboard_interrupt")
                self._end_session()
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
            ("clear session", "清空当前短期会话上下文"),
            ("clear history", "清空当前会话上下文和长期记忆"),
            ("clear memory", "只清空长期记忆"),
            ("bash approve on", "开启 Bash 命令人工确认"),
            ("bash approve off", "关闭 Bash 命令人工确认"),
            ("model_list/models", "查看可用模型配置"),
            ("model", "查看模型相关命令"),
            ("model add", "新增模型配置并测试"),
            ("model <name|编号>", "切换到指定模型"),
            ("tools", "列出当前可用工具"),
            ("compact", "压缩当前会话上下文"),
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
        self.console.print(f"[dim]Bash 命令确认策略：{self._bash_approve_status_text()}[/]")

    def _print_input_header(self):
        """
        在输入提示前展示当前模型信息。
        :return:
        """
        # 每轮输入前展示当前模型，模型切换后可自动反映最新状态。
        self.console.print(f"[dim]当前模型：{self.model}[/]")

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
            "clear session": self._handle_cmd_clear_session,
            "clear history": self._handle_cmd_clear_history,
            "clear memory": self._handle_cmd_clear_memory,
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
        self._wait_for_memory_tasks()
        self._end_session()
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

    def _handle_cmd_clear_session(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 仅清空当前会话历史，不影响长期记忆。
        confirm_input = self.console.input("[bold cyan]是否确认清除当前会话历史？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._build_system_prompt()
            self.console.print("[dim]当前对话历史已清空[/]")
        return True, False

    def _handle_cmd_clear_history(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 清空当前会话历史并清空历史记忆。
        confirm_input = self.console.input("[bold cyan]是否确认清除当前会话历史和全部历史记忆？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._build_system_prompt()
            self._clear_memory()
            self.console.print("[dim]当前对话历史与历史记忆已清空；full_context 文件未批量删除[/]")
        return True, False

    def _handle_cmd_clear_memory(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 兼容旧命令 clear memory，行为与 clear history 一致。
        confirm_input = self.console.input("[bold cyan]是否确认清除历史记忆？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._clear_memory()
            self.console.print("[dim]记忆索引和汇总已清空；full_context 文件未批量删除[/]")
        return True, False

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

    def _handle_cmd_model_add(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增模型前先真实发送一条测试消息，避免把不可用配置写入模型列表。
        model_name = self.console.input("[bold cyan]请输入模型名：[/] ").strip()
        base_url = self.console.input("[bold cyan]请输入 base_url：[/] ").strip()
        api_key = self.console.input("[bold cyan]请输入 api_key：[/] ").strip()
        context_token_input = self.console.input("[bold cyan]请输入上下文窗口大小：[/] ").strip()

        if not model_name or not base_url:
            self.console.print("[yellow]模型名和 base_url 不能为空，已取消新增。[/]")
            return True, False
        if self._get_model_config_by_name(model_name):
            self.console.print(f"[yellow]模型 {model_name} 已存在，已取消新增。[/]")
            return True, False
        try:
            max_model_context_token = int(context_token_input)
            if max_model_context_token <= 0:
                raise ValueError
        except ValueError:
            self.console.print("[yellow]上下文窗口大小必须是正整数，已取消新增。[/]")
            return True, False

        self.console.print(f"[dim]正在测试新模型：[/][bold cyan]{model_name}[/]")
        temp_client = None
        try:
            temp_client = self._build_openai_client(base_url, api_key)
            test_response = temp_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say this is a test"}],
            )
        except Exception as error:
            self.console.print(
                f"[yellow]模型信息错误，请检查模型名、base_url、api_key 和服务状态。"
                f"错误类型：{type(error).__name__}；错误信息：{error}[/]"
            )
            return True, False
        finally:
            close_temp_client = getattr(temp_client, "close", None)
            if callable(close_temp_client):
                close_temp_client()

        test_content = getattr(test_response.choices[0].message, "content", "")
        self.console.print(f"[green]模型测试成功。响应：[/][dim]{test_content}[/]")

        config = self._read_model_config()
        config.setdefault("models", [])
        config["models"].append(
            {
                "name": model_name,
                "base_url": base_url,
                "api_key": api_key,
                "max_model_context_token": max_model_context_token,
            }
        )
        self._write_model_config(config)
        self.console.print("[green]模型配置已写入 agent/config/model_config.json[/]")

        switch_input = self.console.input("[bold cyan]是否立即切换到新模型？(yes/y)[/] ").strip()
        if self._normalize_command(switch_input) in confirm_choice:
            self.model = model_name
            self._apply_model_config_by_name(self.model)
            self._max_context_tokens = self._load_model_context_window()
            self.console.print(
                f"[green]模型已切换为：{self.model}（上下文窗口：{self._max_context_tokens}）[/]"
            )
            self._print_runtime_status()
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
        self._apply_model_config_by_name(self.model)
        self._max_context_tokens = self._load_model_context_window()
        self.console.print(f"[green]模型已切换为：{self.model}（上下文窗口：{self._max_context_tokens}）[/]")
        self._print_runtime_status()
        return True, False

    def _handle_cmd_compact_history(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 主动触发当前会话历史压缩。
        self._compact_messages(self._all_tools)
        self.console.print("[dim]已执行会话压缩检查（达到阈值时会执行压缩）。[/]")
        return True, False

    def _handle_cmd_status(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 展示当前模型状态（模型名、已用 token、token 总量）。
        used_tokens = self._used_token or self._estimate_messages_tokens(self.messages, self._all_tools)
        total_tokens = self._total_token or self._max_context_tokens
        # status 命令展示 token 使用柱状图，直观反馈使用率。
        usage_ratio = self._calculate_token_usage_ratio(used_tokens, self._max_context_tokens)
        token_bar = self._render_token_usage_bar(usage_ratio)
        self.console.print(f"[dim]模型名：{self.model}[/]")
        self.console.print(f"[dim]已使用 Token：{used_tokens}[/]")
        self.console.print(f"[dim]Token 总量：{total_tokens}[/]")
        self.console.print(f"[dim]上下文窗口：{self._max_context_tokens}[/]")
        self.console.print(f"[dim]上下文使用率：{usage_ratio * 100:.2f}%[/]")
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
        memory_task_ids = [task_id for task_id in (task.get("memory_task_ids") or []) if task_id]
        try:
            # 删除采用追加式软删除事件，不物理删除 session 文件或长期记忆上下文文件。
            self.session_manager.mark_turn_deleted(turn_id=turn_id, task_ids=memory_task_ids)
            if self.memory_manager is not None:
                self.memory_manager.record_deleted_task(
                    session_id=session_id,
                    turn_id=turn_id,
                    task_ids=memory_task_ids,
                )
            if session_id:
                self._reload_messages_after_turn_delete(session_id)
        except Exception as error:
            self.console.print(f"[yellow]删除任务失败：{error}[/]")
            return False

        self.console.print(f"[dim]已删除任务 {task.get('index')}（turn_id: {turn_id}）[/]")
        return True

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
        self._used_token = self._estimate_messages_tokens(self.messages, self._all_tools)
        self._total_token = self._used_token

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
        self._used_token = 0
        self._total_token = 0
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
        if not messages or messages[0].get("role") != "system":
            # 老会话缺少 system message 时，只在内存中补齐，不反向改写历史文件。
            system_prompt = self._cached_system_prompt or self._build_system_prompt()
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _format_session_time(self, value: Any) -> str:
        # 会话列表只做轻量展示，缺失时间用短横线占位。
        return str(value or "-").replace("T", " ")

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
            self.messages = self._load_messages_from_session(info["session_id"])
            self._current_task_full_context = None
            self._current_task_start_index = None
            self._current_turn_id = None
            self._used_token = self._estimate_messages_tokens(self.messages, self._all_tools)
            self._total_token = self._used_token
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
        self._load_session_by_id(sessions[0].get("session_id"))
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

    def _calculate_token_usage_ratio(self, used_tokens: int, context_window: int) -> float:
        """
        计算 token 使用率。
        :param used_tokens: 已使用 token
        :param context_window: 上下文窗口 token 上限
        :return:
        """
        # 统一处理边界情况，避免除零并将结果限制在 0~1。
        if context_window <= 0:
            return 0.0
        ratio = max(0.0, used_tokens / context_window)
        return min(ratio, 1.0)

    def _render_token_usage_bar(self, usage_ratio: float, width: int = 30) -> str:
        """
        渲染横向 token 使用柱状图。
        :param usage_ratio: 使用率（0~1）
        :param width: 柱状图宽度
        :return:
        """
        # 使用 Unicode 方块字符绘制柱状图，并根据阈值设置颜色。
        ratio = min(max(usage_ratio, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        if ratio >= 0.9:
            color = "red"
        elif ratio >= 0.7:
            color = "yellow"
        else:
            color = "green"
        return f"[{color}]{bar}[/{color}] {ratio * 100:.2f}%"

"""
Team 类管理多个Agent，
"""

class Team:
    def __init__(self, agent_factory=None):
        self.agents = {}  # 名称到 Agent 的映射。
        self.agent_factory = agent_factory or (lambda name, role: Agent(role=role, name=name))

    def hire(self, name, role):
        """招募：创建一个持久 Agent"""
        agent = self.agent_factory(name, role)
        self.agents[name] = agent
        return agent

    def send(self, from_name, to_name, message):
        """Agent 之间的通信通道"""
        if to_name not in self.agents:
            return f"Error: {to_name} not found"
        self.agents[to_name].receive(from_name, message)
        print(f"  [communication] from {from_name} to {to_name}: {message[:60]}...")

    def broadcast(self, from_name, message):
        """广播：给团队所有其他人发消息"""
        for name, agent in self.agents.items():
            if name != from_name:
                agent.receive(from_name, message)
        print(f"  [broadcast] from {from_name} to all teammates: {message[:60]}...")

    def disband(self):
        """解散：所有 Agent 生命周期结束"""
        names = list(self.agents.keys())
        self.agents.clear()
        print(f"  [dismiss] The team are dismissed ({', '.join(names)})")

"""
团队编排
"""
class TeamOrchestrator:
    def __init__(self, model="qwen3.5:9b", temperature: float = 0.1,
                 base_url: str = None, api_key: str = None,
                 client=None, mcp_client=None):
        self.model = model
        self.temperature = temperature
        self._base_url = os.environ.get("OPENAI_BASE_URL") if base_url is None else base_url
        self._api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
        self.client = client or OpenAI(base_url=self._base_url, api_key=self._api_key)
        self.mcp_client = mcp_client

    def _create_agent(self, name, role):
        return Agent(
            model=self.model,
            temperature=self.temperature,
            base_url=self._base_url,
            api_key=self._api_key,
            mcp_client=self.mcp_client,
            role=role,
            name=name,
        )

    def plan_team(self, task):
        """让 LLM 根据任务规划团队成员"""
        print(f"\n[PM] 分析任务，组建团队...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": load_prompt("team_planning_system.md")},
                {"role": "user", "content": load_prompt("team_planning_user.md", task=task)}
            ],
            response_format={"type": "json_object"}
        )
        try:
            return json.loads(response.choices[0].message.content).get("team", [])
        except Exception:
            return [{"name": "dev", "role": "developer", "task": task}]

    def run(self, task):
        """
        完整的团队协作流程，展示三个核心能力:

        1. 持久记忆 —— 同一个 Agent 被多次 chat()，记得之前做过什么
        2. 身份生命周期 —— hire() 创建 → 多次交互 → disband() 解散
        3. 通信通道 —— Agent 之间通过 send()/broadcast() 传递信息
        """
        team = Team(agent_factory=self._create_agent)

        members = self.plan_team(task)
        print(f"\n[团队] {len(members)} 人")
        for i, member in enumerate(members, 1):
            print(f"  {i}. {member['name']} - {member['role']} -> {member['task']}")

        print(f"\n{'='*60}")
        print("  阶段 1: 招募团队")
        print(f"{'='*60}")
        for member in members:
            team.hire(member["name"], member["role"])

        print(f"\n{'='*60}")
        print("  阶段 2: 协作开发")
        print(f"{'='*60}")

        results = {}
        for i, member in enumerate(members):
            print(f"\n{'-'*60}")
            print(f"  [{i+1}/{len(members)}] {member['name']} 开始工作")
            print(f"{'-'*60}")

            agent = team.agents[member["name"]]
            result = agent.chat(member["task"])
            results[member["name"]] = result
            team.broadcast(member["name"], f"我完成了任务。摘要: {result[:200]}")

        if members:
            last = members[-1]
            reviewer = team.agents[last["name"]]

            print(f"\n{'='*60}")
            print(f"  阶段 3: {last['name']} 做最终审查")
            print(f"{'='*60}")

            review = reviewer.chat("请根据你收到的所有团队成果，做一个最终的总结和审查。如果有问题请指出。")
            results["final_review"] = review

        print(f"\n{'='*60}")
        print("  阶段 4: 解散团队")
        print(f"{'='*60}")
        team.disband()

        print(f"\n{'='*60}")
        print("  最终成果")
        print(f"{'='*60}\n")
        for name, result in results.items():
            print(f"[{name}]")
            print(f"  {result[:300]}\n")

        return results


def plan_team(task):
    return TeamOrchestrator().plan_team(task)


def run_team(task):
    return TeamOrchestrator().run(task)
"""
TODO 压缩方案优化方向：
 压缩前先写回永久记忆，（问GPT，先实现永久记忆用RAG （用chroma 向量数据库）， 还是先实现这里的写回永久记忆
 触发条件：	消息条数超过固定阈值  ->	基于 token 数精确计算，考虑模型的实际窗口大小
压缩方式：	一次性把所有旧消息压缩成一段摘要 ->	分层压缩：最近的保留原文，稍远的做摘要，更远的只保留关键事实
保留策略：	固定保留最近 N 条   ->	智能选择：保留包含文件路径、错误信息等关键消息
摘要质量：	通用摘要 prompt	-> 针对 coding 场景优化的摘要 prompt，确保保留文件路径、代码片段、决策原因
 
"""

# TODO 优化：每次启动的会话中所有短期记忆在退出时都写到磁盘的长期记忆里
# TODO 优化：短期记忆的加载方式以及系统提示
# TODO 优化思考：当前通过 MemoryManager 在一次 chat(task) 期间保存“本任务上下文快照”，并异步写入 agent/memory/full_context。

if __name__ == "__main__":
    from agent_run import AgentRunner
    runner = AgentRunner(
        model="minimax-m2.7:cloud",
        mcp_mode="subprocess",
    )
    runner.run()
