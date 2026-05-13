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
        self._max_context_tokens = self._load_model_context_window()

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
        config_path = Path(self._agent_file_path("agent/config/model_context_windows.json"))
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return self._DEFAULT_CONTEXT_WINDOW
        return int(config.get(self.model) or config.get("default") or self._DEFAULT_CONTEXT_WINDOW)

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
        self.console.print(Panel(
            "[bold green]JanVis[/] — At you service, sir! What can I do for you today?\n\n"
            "  [blue]命令[/]: exit/q/quit 退出 | clear 清空当前会话历史 | clear memory",
            border_style="green", padding=(1, 2),
        ))
        self.console.print(f"[dim]当前工作目录：{os.getcwd()}[/]")
        self.console.print(f"[dim]使用模型：{self.model}[/]")

        confirm_choice = ("y", "yes", "是", "确认", "对", "")

        while True:
            try:
                user_input = self.console.input("[bold cyan]You >[/] ")
                cmd = user_input.strip().lower()
                if cmd in ("exit", "q", "quit"):
                    self._wait_for_memory_tasks()
                    self.console.print("\n[bold red] See you next time! [/]")
                    break
                elif cmd == "clear":
                    user_input = self.console.input("[bold cyan]是否确认清除当前会话历史？(yes/y)[/] ")
                    cmd = user_input.strip().lower()
                    if cmd in confirm_choice:
                        self._build_system_prompt()
                        self.console.print("[dim]当前对话历史已清空[/]")
                    continue
                elif cmd == "clear memory":
                    user_input = self.console.input("[bold cyan]是否确认清除历史记忆？(yes/y)[/] ")
                    cmd = user_input.strip().lower()
                    if cmd in confirm_choice:
                        self._clear_memory()
                        self.console.print("[dim]记忆索引和汇总已清空；full_context 文件未批量删除[/]")
                    continue
                elif not cmd:
                    continue

                # 上面分支都没中，就是用户任务了
                self.chat(user_input)
                self.console.print()
            except KeyboardInterrupt:
                self._wait_for_memory_tasks()
                self.console.print("\n[bold red] See you next time! [/]")
                break

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
