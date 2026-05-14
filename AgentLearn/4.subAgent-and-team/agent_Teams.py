# encoding : utf-8
# @Time    : 2026/4/19
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from openai import OpenAI
from mcp_client import MCPClient
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from memory_manager import MemoryManager
from prompt_loader import load_prompt
from tools.tool_manager import ToolManager, ToolManagerConfig, AgentToolHandlers
from tools.tool_names import ToolNameConstant
from tools.tool_scheduler import ToolScheduler, ToolCallTask


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

        # openAI 请求客户端
        self.client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        # 是否是主agent, False表示由Agent创建的子Agent,默认为True
        self._is_main_agent = is_main_agent

        # 最大迭代次数
        self.max_iterations = 100

        # 使用模型
        self.model = model
        # 新增：根据传入模型从 model_config.json 读取模型配置（base_url/api_key/上下文窗口等）。
        self._apply_model_config_by_name(self.model)
        self._max_context_tokens = self._load_model_context_window()
        # 新增：记录最近一次模型调用的已使用 token 和总 token（来自 OpenAI 标准 usage 字段）。
        self._used_token = 0
        self._total_token = 0

        # llm温度参数
        self.temperature = temperature

        # 主Agent才保留长期记忆，由主Agent唤起的子Agent不给保留记忆，用完就扔
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

        # 是否处在plan模式
        self.plan_mode = False

        # 当前plan列表
        self.current_plan: list[str] = []

        # 规则和技能目录
        self.rules_dir = "agent/rules"
        self.skills_dir = "agent/skills"
        self.prompts_dir = "agent/prompts"

        # 各SKILL.md缓存,key 为SKILL的name，value为SKILL.md完整内容
        self._skills_cache = {}

        # MCP客户端（由外部传入，不在Agent内部创建）
        self.mcp_client = mcp_client
        self._tool_manager = ToolManager(
            config=ToolManagerConfig(
                project_root=os.path.dirname(__file__),
                client=self.client,
                model=self.model,
                temperature=self.temperature,
                is_main_agent=self._is_main_agent,
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
        print(f"{len(self._local_tools)} local tools loaded")
        if self.mcp_client:
            print(f"{len(self._mcp_tools)} MCP tools loaded")
        else:
            print("No MCP client provided, MCP tools not loaded")

        # 基础提示词，用于主agent
        self._base_prompt_main_agent = load_prompt("base_main_agent.md")

        # 子agent提示词
        self._base_prompt_sub_agent = load_prompt("base_sub_agent.md", role=role)

        # 缓存系统提示词,后续记忆压缩的时候可能会用到
        self._cached_system_prompt = self._build_system_prompt()

        # 单次会话中的短期记忆，记录一次会话中的短期上下文
        self.messages = [
            # 初始化时添加系统提示词
            {"role": "system", "content": self._cached_system_prompt}
        ]
        # 当前任务的完整上下文，用于保存长期记忆
        self._current_task_full_context = None
        self._current_task_start_index = None

        self.console = Console()

    def _model_config_file_path(self) -> Path:
        """
        获取模型配置文件路径。
        :return:
        """
        # 新增：统一模型配置路径，后续读取与写入共用。
        return Path(self._agent_file_path("agent/config/model_config.json"))

    def _read_model_config(self) -> dict:
        """
        读取模型配置文件。
        :return:
        """
        # 新增：若文件不存在或内容异常，返回空配置结构，避免启动报错。
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
        # 新增：确保目录存在并按 UTF-8 美化输出 JSON。
        config_path = self._model_config_file_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_model_config_by_name(self, model_name: str) -> dict | None:
        """
        根据模型名获取模型配置项。
        :param model_name: 模型名
        :return:
        """
        # 新增：从模型配置列表里查找目标模型。
        models = self._read_model_config().get("models", [])
        for model_info in models:
            if isinstance(model_info, dict) and model_info.get("name") == model_name:
                return model_info
        return None

    def _apply_model_config_by_name(self, model_name: str):
        """
        根据模型名应用模型配置（base_url/api_key）。
        :param model_name: 模型名
        :return:
        """
        # 新增：在不改变初始化接口的前提下，按模型配置自动补全连接信息。
        model_info = self._get_model_config_by_name(model_name)
        if not model_info:
            return
        self._base_url = model_info.get("base_url") or self._base_url
        self._api_key = model_info.get("api_key") or self._api_key
        self.client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

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
        # 新增：从模型配置中按模型名读取上下文窗口配置。
        models = config.get("models", []) if isinstance(config, dict) else []
        for model_info in models:
            if isinstance(model_info, dict) and model_info.get("name") == self.model:
                return int(model_info.get("max_model_context_token") or self._DEFAULT_CONTEXT_WINDOW)
        return self._DEFAULT_CONTEXT_WINDOW

    def _schedule_memory_update(self, task, result):
        if not self._is_main_agent or self.memory_manager is None:
            return
        # Memory summarization is intentionally asynchronous so task completion stays responsive.
        self.memory_manager.enqueue(
            task=task,
            result=result,
            context=self._current_task_full_context or [],
        )

    def _load_memory_view(self):
        # Sub agents do not inherit long-term memory; they only work on the delegated task.
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

    def set_bash_auto_approve(self, enabled: bool):
        """
        设置当前 Agent 的 Bash 命令是否自动确认执行。
        :param enabled: True 表示自动确认，False 表示需要手动确认
        :return:
        """
        # 新增：支持运行时切换 Bash 自动确认配置，便于用户动态控制安全策略。
        self._BASH_AUTO_APPROVE = bool(enabled)

    def _bash_approve_status_text(self) -> str:
        """
        返回当前 Bash 执行确认策略的文本描述。
        :return:
        """
        # 新增：统一管理状态文案，避免 run() 中重复拼接字符串。
        if self._BASH_AUTO_APPROVE:
            return "自动确认（无需手动确认）"
        return "手动确认（每次需确认）"

    def _append_message(self, message, capture_full_context=True):
        self.messages.append(message)
        if capture_full_context and self._current_task_full_context is not None:
            self._current_task_full_context.append(self._normalize_message_for_memory(message))

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
        # 新增：每次模型调用后更新 token 使用统计。
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
            # 解析 YAML frontmatter
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
                        # 缓存完整SKILL内容
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
        # 不要从 tool 消息中间切开；tool 必须紧跟触发它的 assistant tool_call。
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

        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": load_prompt("conversation_compaction_system.md")},
                {"role": "user", "content": load_prompt("conversation_compaction_user.md", old_text=old_text)}
            ]
        )
        # 新增：每次模型调用后更新 token 使用统计。
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
            {
                "role": "assistant",
                "content": "Understood. I will load archived task context by task_id when needed.",
            },
            {"role": "user", "content": load_prompt("conversation_compaction_summary_message.md", summary=summary)},
            {"role": "assistant", "content": load_prompt("conversation_compaction_ack.md")},
            *recent_messages
        ]

    def _run_agent_step(self, tools):
        for i in range(self.max_iterations):
            # 先压缩历史对话
            self._compact_messages(tools)
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tools,
                temperature=self.temperature,
                stream=True,
            )
            # message = response_stream.choices[0].message
            # messages.append(message)
            message = self._deal_stream_response(response_stream)
            print()
            # 按照OpenAI的格式对message进行复原然后加回当前的短期历史记忆
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
                    for tc in (message.tool_calls or [])
                ] if message.tool_calls else None,
            })

            # 无工具调用，结束
            if not message.tool_calls:
                return message.content
            print(f"[Iter {i}]: message is: {message}")
            # V2 调度策略：依据工具画像（只读/并发安全/作用域）做分段并发。
            scheduler = ToolScheduler(get_profile=self._tool_manager.get_tool_runtime_profile)
            pending_tasks: list[ToolCallTask] = []

            for tool_call in message.tool_calls:
                function_payload = getattr(tool_call, "function", None)
                if function_payload is None:
                    continue
                function_name = str(getattr(function_payload, "name", ""))
                raw_arguments = str(getattr(function_payload, "arguments", ""))
                function_args = self._parse_tool_arguments(raw_arguments)

                # MAKE_PLAN 需要维持原有特殊流程，先刷新 pending 队列再串行处理。
                if function_name == ToolNameConstant.MAKE_PLAN:
                    for task, function_response in scheduler.execute_batches(
                        scheduler.plan_batches(pending_tasks), self._invoke_tool_task
                    ):
                        self._append_message(
                            {"role": "tool", "tool_call_id": task.tool_call_id,
                             "content": json.dumps(function_response, ensure_ascii=False)}
                        )
                    pending_tasks = []

                    function_impl = self._available_functions.get(function_name)
                    if function_impl is None:
                        function_response = f"Error: Unknown tool '{function_name}'"
                    elif "_argument_error" in function_args:
                        function_response = f"Error: {function_args['_argument_error']}"
                    else:
                        # 如果模型选择了先做计划，这个分支就对计划模式特殊处理
                        self.plan_mode = True
                        steps = function_impl(**function_args)
                        if not isinstance(steps, list):
                            function_response = steps
                        else:
                            results = []
                            step_cnt = 0
                            for step in steps:
                                print(f"[Step {step_cnt + 1}]: {step}")
                                self._append_message({"role": "user", "content": step})
                                result = self._run_agent_step(
                                    [t for t in tools if t["function"]["name"] != ToolNameConstant.MAKE_PLAN]
                                )
                                print(f"[Step {step_cnt + 1}] result:{result}, all messages: {self.messages}")
                                step_cnt += 1
                                results.append(result)
                            function_response = "\n".join(results)
                        self.plan_mode = False
                        self.current_plan = []
                    self._append_message(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(function_response, ensure_ascii=False)}
                    )
                    continue

                pending_tasks.append(
                    ToolCallTask(
                        tool_call_id=tool_call.id,
                        function_name=function_name,
                        raw_arguments=raw_arguments,
                        function_args=function_args,
                    )
                )

            # 收尾执行剩余任务（可能包含并发批次）。
            for task, function_response in scheduler.execute_batches(
                scheduler.plan_batches(pending_tasks), self._invoke_tool_task
            ):
                self._append_message(
                    {"role": "tool", "tool_call_id": task.tool_call_id, "content": json.dumps(function_response, ensure_ascii=False)}
                )
        return "Max iterations reached"

    def _invoke_tool_task(self, task: ToolCallTask):
        """调度器执行入口：单工具调用的统一异常处理。"""
        function_impl = self._available_functions.get(task.function_name)
        if function_impl is None:
            return f"Error: Unknown tool '{task.function_name}'"
        if "_argument_error" in task.function_args:
            return f"Error: {task.function_args['_argument_error']}"
        try:
            print(f"[Tool call] tool name: {task.function_name}, tool arguments: {task.raw_arguments}")
            return function_impl(**task.function_args)
        except Exception as error:
            return f"Error when calling '{task.function_name}': {error}"

    def _build_system_prompt(self):
        from prompt_builder import build_system_prompt
        # 置空当前prompt
        self._cached_system_prompt = None
        memory = self._load_memory_view()
        rules = self._load_rules(memory)
        skills = self._load_skill_meta_infos()
        base_prompt = [
            self._base_prompt_main_agent if self._is_main_agent else self._base_prompt_sub_agent,
        ]
        self._cached_system_prompt = build_system_prompt(base_prompt, rules, skills, memory)
        return self._cached_system_prompt

    def _deal_stream_response(self, stream_response):
        """
        为了用户体验，需要做流式响应，这里需要处理模型的流式响应，
        所以需要 1）流式打印响应内容，2）累积工具调用的chunks,因为同一个工具调用的参数可能在两个chunk里分两次返回
        :param stream_response:
        :return: 转换后的一个message对象
        """
        # 完整reply
        full_reply = ""
        # 累积工具调用
        tool_calls = {}
        for chunk in stream_response:
            # 新增：流式响应中若包含 usage 字段，则实时更新 token 统计。
            self._update_usage_from_response(chunk)
            # 防御，有时delta可能不存在
            choice = chunk.choices[0]
            delta = choice.delta
            # 1) 流式打印响应内容
            content = getattr(delta, "content", None)
            if content:
                print(content, end="", flush=True)
                full_reply += content

            # 2) 累积还原工具调用
            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": getattr(tc, "id", None),
                            "name": "",
                            "arguments": "",
                        }
                    func = getattr(tc, "function", None)
                    if func is not None:
                        name = getattr(func, "name", None)
                        args = getattr(func, "arguments", None)
                        if name:
                            tool_calls[idx]["name"] = name
                        if args:
                            tool_calls[idx]["arguments"] += args

        # 转换兼容的结构
        ordered_tool_calls = []
        for idx in sorted(tool_calls.keys()):
            item = tool_calls[idx]
            ordered_tool_calls.append(
                SimpleNamespace(
                    id=item["id"],
                    function=SimpleNamespace(
                        name=item["name"],
                        arguments=item["arguments"],
                    ),
                )
            )
        message = SimpleNamespace(
            content=full_reply if full_reply else None,
            tool_calls=ordered_tool_calls if ordered_tool_calls else None,
        )
        return message

    def _update_usage_from_response(self, response):
        """
        从 OpenAI 响应对象中提取 usage 并更新 token 统计。
        :param response: 非流式 response 或流式 chunk
        :return:
        """
        # 新增：统一处理标准 OpenAI usage 字段，避免各处重复解析逻辑。
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        # 新增：优先使用 total_tokens；若为空则回退为 prompt+completion。
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

    def chat(self, task):
        """
        Agent单次任务运行入口
        :param task: 用户任务
        :return: 执行任务结果
        """
        # 初始化当前任务的完整上下文记录
        self._current_task_full_context = []
        try:
            # 如果 inbox 有新消息，先注入self.messages
            if self.inbox:
                mail = "\n".join(f"[from {m['from']}]: {m['content']}" for m in self.inbox)
                self._append_message({"role": "user", "content": load_prompt("inbox_digest_user.md", mail=mail)})
                # 让 Agent 先消化这些消息
                resp = self.client.chat.completions.create(model=self.model, messages=self.messages)
                # 新增：每次模型调用后更新 token 使用统计。
                self._update_usage_from_response(resp)
                self._append_message(resp.choices[0].message)
                self.inbox.clear()

            # 再拼接本次任务并执行
            self._append_message({"role": "user", "content": task})
            final_result = self._run_agent_step(self._all_tools)
            # print(f"final result: {final_result}")
            self._schedule_memory_update(task, final_result)
            return final_result
        finally:
            self._current_task_full_context = None

    def run(self):
        """
        Agent loop实现，对话入口
        :return:
        """
        # 新增：统一通过 help 文案展示可用命令，避免 run() 内硬编码过长提示。
        self._print_help(show_welcome=True)
        # 新增：集中展示运行时状态，后续配置变更后可复用该方法刷新显示。
        self._print_runtime_status()

        confirm_choice = ("y", "yes", "是", "确认", "对", "")

        while True:
            try:
                # 新增：在输入框上方展示当前模型，便于用户随时确认当前会话模型。
                self._print_input_header()
                user_input = self.console.input("[bold cyan]You >[/] ")
                # 新增：将命令分支提取到独立方法中，降低 run() 循环复杂度。
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
            except KeyboardInterrupt:
                self._wait_for_memory_tasks()
                self.console.print("\n[bold red] See you next time! [/]")
                break

    def _print_help(self, show_welcome: bool = False):
        """
        打印交互帮助信息。
        :param show_welcome: 是否展示欢迎语
        :return:
        """
        # 新增：将帮助提示抽取为独立方法，便于复用和后续维护。
        if show_welcome:
            self.console.print(Panel(
                "[bold green]JanVis[/] — At you service, sir! What can I do for you today?\n\n"
                "You can ask me to do some task or type help/h",
                border_style="green", padding=(1, 2),
            ))
        self.console.print(
            "[dim]可用命令：help/h | exit/q/quit | clear session | clear history | bash approve on/off | "
            "model_list | model | model <name> | campact/compact | status[/]"
        )

    def _print_runtime_status(self):
        """
        打印当前运行时状态信息。
        :return:
        """
        # 新增：抽取状态打印逻辑，便于启动时和配置变更后统一刷新展示。
        self.console.print(f"[dim]当前工作目录：{os.getcwd()}[/]")
        self.console.print(f"[dim]使用模型：{self.model}[/]")
        self.console.print(f"[dim]Bash 命令确认策略：{self._bash_approve_status_text()}[/]")

    def _print_input_header(self):
        """
        在输入提示前展示当前模型信息。
        :return:
        """
        # 新增：每轮输入前展示当前模型，模型切换后可自动反映最新状态。
        self.console.print(f"[dim]当前模型：{self.model}[/]")

    def _normalize_command(self, user_input: str) -> str:
        """
        规范化用户命令输入，支持 `/` 前缀命令。
        :param user_input: 原始用户输入
        :return:
        """
        # 新增：统一处理前后空白和可选的 "/" 前缀（如 /clear session）。
        cmd = (user_input or "").strip().lower()
        if cmd.startswith("/"):
            cmd = cmd[1:].strip()
        return cmd

    def _handle_user_command(self, user_input: str, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        """
        处理命令型输入。
        :param user_input: 用户输入
        :param confirm_choice: 确认命令可接受输入
        :return: (是否已处理, 是否需要退出循环)
        """
        # 新增：集中处理所有命令分支，主循环仅保留调度逻辑。
        cmd = self._normalize_command(user_input)
        # 新增：使用命令映射表分发处理函数，便于后续扩展命令而不是堆叠 if/else。
        command_handlers = self._command_handler_map()
        handler = command_handlers.get(cmd)
        if handler:
            return handler(confirm_choice)
        # 新增：支持带参数命令，例如 model gpt-4o-mini。
        if cmd.startswith("model "):
            return self._handle_cmd_model_switch(confirm_choice, cmd)
        return False, False

    def _command_handler_map(self):
        """
        命令到处理函数的映射表。
        :return:
        """
        # 新增：集中管理命令和处理方法的映射关系，提升可维护性。
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
            "model": self._handle_cmd_model_show_current,
            "campact": self._handle_cmd_compact_history,
            "compact": self._handle_cmd_compact_history,
            "status": self._handle_cmd_status,
        }

    def _handle_cmd_exit(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：退出命令处理。
        self._wait_for_memory_tasks()
        self.console.print("\n[bold red] See you next time! [/]")
        return True, True

    def _handle_cmd_help(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：帮助命令处理。
        self._print_help(show_welcome=False)
        return True, False

    def _handle_cmd_clear_session(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：仅清空当前会话历史，不影响长期记忆。
        confirm_input = self.console.input("[bold cyan]是否确认清除当前会话历史？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._build_system_prompt()
            self.console.print("[dim]当前对话历史已清空[/]")
        return True, False

    def _handle_cmd_clear_history(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：清空当前会话历史并清空历史记忆。
        confirm_input = self.console.input("[bold cyan]是否确认清除当前会话历史和全部历史记忆？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._build_system_prompt()
            self._clear_memory()
            self.console.print("[dim]当前对话历史与历史记忆已清空；full_context 文件未批量删除[/]")
        return True, False

    def _handle_cmd_clear_memory(self, confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：兼容旧命令 clear memory，行为与 clear history 一致。
        confirm_input = self.console.input("[bold cyan]是否确认清除历史记忆？(yes/y)[/] ")
        if self._normalize_command(confirm_input) in confirm_choice:
            self._clear_memory()
            self.console.print("[dim]记忆索引和汇总已清空；full_context 文件未批量删除[/]")
        return True, False

    def _handle_cmd_bash_approve_on(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：开启 Bash 自动确认。
        self._update_bash_approve_status(True)
        return True, False

    def _handle_cmd_bash_approve_off(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：关闭 Bash 自动确认。
        self._update_bash_approve_status(False)
        return True, False

    def _update_bash_approve_status(self, enabled: bool):
        """
        更新 Bash 自动确认配置并刷新状态展示。
        :param enabled: 是否开启自动确认
        :return:
        """
        # 新增：复用 on/off 两个命令的公共逻辑，避免重复代码。
        old_status = self._bash_approve_status_text()
        self.set_bash_auto_approve(enabled)
        new_status = self._bash_approve_status_text()
        if old_status != new_status:
            # 新增：当确认策略发生变化后，自动刷新状态展示，用户可立即看到最新配置。
            self.console.print(f"[dim]已更新 Bash 命令确认策略：{new_status}[/]")
            self._print_runtime_status()
        else:
            # 新增：若用户重复设置同一状态，明确告知并保持当前展示一致。
            self.console.print(f"[dim]Bash 命令确认策略未变化，当前为：{new_status}[/]")

    def _load_available_models(self) -> list[str]:
        """
        读取当前已配置的可用模型列表。
        :return:
        """
        # 新增：从 model_config.json 读取模型列表，作为可切换模型来源。
        models = self._read_model_config().get("models", [])
        return [model_info.get("name") for model_info in models if isinstance(model_info, dict) and model_info.get("name")]

    def _handle_cmd_model_list(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：列出当前已配置模型；若暂无配置则先保留为空实现提示。
        available_models = self._load_available_models()
        if not available_models:
            self.console.print("[yellow]当前暂无可用模型配置，model_list 暂无可展示内容。[/]")
            return True, False
        self.console.print("[dim]已配置模型列表：[/]")
        for index, model_name in enumerate(available_models, start=1):
            marker = "（当前）" if model_name == self.model else ""
            self.console.print(f"[dim]{index}. {model_name} {marker}[/]")
        return True, False

    def _handle_cmd_model_show_current(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：展示当前模型并提示如何切换。
        self.console.print(f"[dim]当前使用模型：{self.model}[/]")
        self.console.print("[dim]切换方式：model <模型名>，例如 model gpt-4o-mini[/]")
        return True, False

    def _handle_cmd_model_switch(self, _confirm_choice: tuple[str, ...], cmd: str) -> tuple[bool, bool]:
        # 新增：处理 model <model_name> 命令，校验配置后执行切换。
        target_model = cmd.split(" ", 1)[1].strip()
        self.console.print(f"[dim]当前使用模型：{self.model}[/]")
        if not target_model:
            self.console.print("[yellow]请输入目标模型，例如：model gpt-4o-mini[/]")
            return True, False
        available_models = self._load_available_models()
        if target_model not in available_models:
            # 新增：目标模型不在配置列表时，引导用户录入4个配置字段并写入 model_config.json。
            self.console.print(f"[yellow]模型 {target_model} 未配置，开始新增模型配置。[/]")
            base_url = self.console.input("[bold cyan]请输入 base_url：[/] ").strip()
            api_key = self.console.input("[bold cyan]请输入 api_key：[/] ").strip()
            context_token_input = self.console.input("[bold cyan]请输入 max_model_context_token：[/] ").strip()
            try:
                max_model_context_token = int(context_token_input)
            except ValueError:
                self.console.print("[red]max_model_context_token 必须是整数，已取消新增。[/]")
                return True, False

            config = self._read_model_config()
            config.setdefault("models", [])
            config["models"].append(
                {
                    "name": target_model,
                    "base_url": base_url,
                    "api_key": api_key,
                    "max_model_context_token": max_model_context_token,
                }
            )
            self._write_model_config(config)
            self.console.print(f"[green]模型 {target_model} 配置已写入 model_config.json[/]")

        self.model = target_model
        self._apply_model_config_by_name(self.model)
        self._max_context_tokens = self._load_model_context_window()
        self.console.print(f"[green]模型已切换为：{self.model}（上下文窗口：{self._max_context_tokens}）[/]")
        self._print_runtime_status()
        return True, False

    def _handle_cmd_compact_history(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：主动触发当前会话历史压缩。
        self._compact_messages(self._all_tools)
        self.console.print("[dim]已执行会话压缩检查（达到阈值时会执行压缩）。[/]")
        return True, False

    def _handle_cmd_status(self, _confirm_choice: tuple[str, ...]) -> tuple[bool, bool]:
        # 新增：展示当前模型状态（模型名、已用 token、token 总量）。
        used_tokens = self._used_token or self._estimate_messages_tokens(self.messages, self._all_tools)
        total_tokens = self._total_token or self._max_context_tokens
        # 新增：status 命令展示 token 使用柱状图，直观反馈使用率。
        usage_ratio = self._calculate_token_usage_ratio(used_tokens, self._max_context_tokens)
        token_bar = self._render_token_usage_bar(usage_ratio)
        self.console.print(f"[dim]模型名：{self.model}[/]")
        self.console.print(f"[dim]已使用 Token：{used_tokens}[/]")
        self.console.print(f"[dim]Token 总量：{total_tokens}[/]")
        self.console.print(f"[dim]上下文窗口：{self._max_context_tokens}[/]")
        self.console.print(f"[dim]上下文使用率：{usage_ratio * 100:.2f}%[/]")
        # 新增：按评审意见直接打印柱状图，不额外增加提示前缀。
        self.console.print(token_bar)
        return True, False

    def _calculate_token_usage_ratio(self, used_tokens: int, context_window: int) -> float:
        """
        计算 token 使用率。
        :param used_tokens: 已使用 token
        :param context_window: 上下文窗口 token 上限
        :return:
        """
        # 新增：统一处理边界情况，避免除零并将结果限制在 0~1。
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
        # 新增：使用 Unicode 方块字符绘制柱状图，并根据阈值设置颜色。
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
        self.agents = {}  # name → Agent
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
