# encoding : utf-8
# @Time    : 2026/4/19
import glob as glob_module
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
from typing import Any, Final
from openai import OpenAI
from mcp_client import MCPClient
from duckduckgo_search import DDGS
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class ToolNameConstant:
    """
    工具名称常量类
    """
    READ_FILE: Final = "READ_FILE"
    WRITE_FILE: Final = "WRITE_FILE"
    EDIT: Final = "EDIT"
    GLOB: Final = "GLOB"
    GREP: Final = "GREP"
    EXECUTE_BASH: Final = "EXECUTE_BASH"
    MAKE_PLAN: Final = "MAKE_PLAN"
    LOAD_SKILL_DETAIL_BY_NAME: Final = "LOAD_SKILL_DETAIL_BY_NAME",
    GET_TIME: Final = "GET_TIME",
    WEB_SEARCH: Final = "WEB_SEARCH",
    LIST_DIR: Final = "LIST_DIR",
    SUB_AGENT: Final = "SUB_AGENT",


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
    _COMPACT_THRESHOLD = 20
    _KEEP_RECENT = 6

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

        # 记忆文件：precise 保存任务/结果摘要，full 保存单次任务完整上下文。
        self.precise_memory_file = "agent/precise_memory.md"
        self.full_memory_file = "agent/full_memory.md"

        # 最大迭代次数
        self.max_iterations = 100

        # 使用模型
        self.model = model

        # llm温度参数
        self.temperature = temperature

        # 是否处在plan模式
        self.plan_mode = False

        # 当前plan列表
        self.current_plan: list[str] = []

        # 规则和技能目录
        self.rules_dir = "agent/rules"
        self.skills_dir = "agent/skills"

        # 各SKILL.md缓存,key 为SKILL的name，value为SKILL.md完整内容
        self._skills_cache = {}

        # Bash hook 管道只服务 _execute_bash，避免引入全局工具执行器。
        # before hook 返回 (blocked, message)，blocked=True 时直接拦截命令。
        self._bash_before_hooks = [
            self._bash_check_dangerous_command,
            self._bash_ask_user_confirmation,
            self._bash_log_command,
        ]
        # after hook 接收上一个 hook 的返回值，必须返回处理后的 result。
        self._bash_after_hooks = [
            self._bash_log_result,
        ]

        # 加载本地工具
        self.local_tools = self._load_local_tools()
        print(f"{len(self.local_tools)} local tools loaded")
        self.local_functions = {
            ToolNameConstant.EXECUTE_BASH: self._execute_bash,
            ToolNameConstant.READ_FILE: self._read_file,
            ToolNameConstant.WRITE_FILE: self._write_file,
            ToolNameConstant.EDIT: self._edit,
            ToolNameConstant.GLOB: self._glob,
            ToolNameConstant.GREP: self._grep,
            ToolNameConstant.MAKE_PLAN: self._make_plan,
            ToolNameConstant.LOAD_SKILL_DETAIL_BY_NAME: self._load_skill_detail_by_name,
            ToolNameConstant.GET_TIME: self._get_time,
            ToolNameConstant.WEB_SEARCH: self._web_search,
            ToolNameConstant.LIST_DIR: self._list_dir
        }
        # 主agent才加subagent工具
        if self._is_main_agent:
            self.local_functions[ToolNameConstant.SUB_AGENT] = self._sub_agent

        # MCP客户端（由外部传入，不在Agent内部创建）
        self.mcp_client = mcp_client
        # 加载MCP工具
        if self.mcp_client:
            self.mcp_tools = self._load_mcp_tools()
            print(f"{len(self.mcp_tools)} MCP tools loaded")
        else:
            self.mcp_tools = []
            print("No MCP client provided, MCP tools not loaded")

        self.available_functions: dict[str, Any] = {}
        self.available_functions.update(self.local_functions)
        # 动态更新可用的工具列表
        for tool in self.mcp_tools:
            tool_name = tool["function"]["name"]
            self.available_functions[tool_name] = self._make_mcp_executor(tool_name)

        self.all_tools = self.local_tools + self.mcp_tools

        # 基础提示词，用于主agent
        self._base_prompt_main_agent = "You are an interactive agent that helps users with daily tasks or software engineering tasks. Use the instructions below and the tools available to you to assist the user."

        # 子agent提示词
        self._base_prompt_sub_agent = f"You are a {role} that helps users with a specific task, focus on the task. Use the instructions below and the tools available to you to assist the user."

        # 缓存系统提示词,后续记忆压缩的时候可能会用到
        self._cached_system_prompt = self._build_system_prompt()

        # 单次会话中的短期记忆，记录一次会话中的短期上下文
        self.messages = [
            # 初始化时添加系统提示词
            {"role": "system", "content": self._cached_system_prompt}
        ]
        # 当前任务的完整上下文，用于保存长期记忆
        self._current_task_full_context = None

        self.console = Console()

    def receive(self, sender, message):
        """
        通信，接收来自其他agent的信息
        :param sender: 发送者
        :param message: 消息
        :return: 无
        """
        self.inbox.append({"from": sender, "content": message})

    def _make_mcp_executor(self, tool_name: str):
        """为MCP工具生成执行器，就是调用mcp客户端的call_tool方法"""

        def _executor(**kwargs):
            return self.mcp_client.call_tool(tool_name, kwargs)

        return _executor

    def _load_local_tools(self) -> list[dict[str, Any]]:
        """
        加载本地工具列表
        :return:
        """
        print("loading local tools...")
        tools_path = os.path.join(os.path.dirname(__file__), "local_tools.json")
        with open(tools_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_mcp_tools(self) -> list[dict[str, Any]]:
        """
        加载远端可用mcp工具列表
        :return:
        """
        mcp_tools = self.mcp_client.list_tools()
        tools_in_openai_format = []
        # 按OpenAI的工具格式添加
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

    def _execute_bash(self, command):
        # 保持对外签名不变，把安全检查和日志交给类内 hook 管道处理。
        return self._execute_bash_with_hooks(command)

    def _run_bash_command(self, command):
        # 只负责实际执行和输出解码；是否允许执行由 before hook 决定。
        result = subprocess.run(command, shell=True, capture_output=True)
        stdout, stderr = self._decode_subprocess_result(result)
        return stdout + stderr

    def _execute_bash_with_hooks(self, command):
        args = {"command": command}
        # 任意 before hook 都可以拦截命令，避免危险命令进入 subprocess。
        for hook in self._bash_before_hooks:
            blocked, message = hook(args)
            if blocked:
                return message or "Bash command was blocked."

        result = self._run_bash_command(command)

        for hook in self._bash_after_hooks:
            result = hook(result)
        return result

    def _bash_command_arg(self, args: dict[str, Any]) -> str:
        # hook 统一通过 args 传递参数，这里保证 command 最终按字符串处理。
        command = args.get("command", "")
        return command if isinstance(command, str) else str(command)

    def _bash_check_dangerous_command(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 第一层防线：命中黑名单时直接拒绝执行，不再进入确认环节。
        command = self._bash_command_arg(args)
        for pattern in self._BASH_DANGEROUS_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                return True, "The command is dangerous, refused to execute it."
        return False, None

    def _bash_ask_user_confirmation(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 第二层防线：需要人工确认时，把命令展示给用户再继续。
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
                sys.exit(0)

    def _bash_log_command(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        # 日志 hook 不改变执行结果，只记录即将运行的命令。
        print(f"[Tool before] {ToolNameConstant.EXECUTE_BASH}: {self._bash_command_arg(args)}")
        return False, None

    def _bash_log_result(self, result: Any) -> Any:
        # 只记录输出长度，避免把可能很长或敏感的命令输出重复打印到控制台。
        text = result if isinstance(result, str) else str(result)
        print(f"[Tool after] {ToolNameConstant.EXECUTE_BASH}: {len(text)} chars")
        return result

    def _bash_truncate_output(self, result: Any) -> Any:
        # 可选 after hook：需要限制上下文长度时，可加入 self._bash_after_hooks。
        text = result if isinstance(result, str) else str(result)
        if len(text) <= self._BASH_MAX_OUTPUT_LENGTH:
            return result
        half = self._BASH_MAX_OUTPUT_LENGTH // 2
        return (
            text[:half]
            + f"\n\n... [output truncated, original {len(text)} chars, kept first/last {half} chars] ...\n\n"
            + text[-half:]
        )

    def _decode_subprocess_result(self, result: CompletedProcess[bytes] | CompletedProcess[Any]):
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

        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                decoded_out = stdout.decode(enc)
                decoded_err = stderr.decode(enc)
                return decoded_out, decoded_err
            except UnicodeDecodeError:
                continue
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

    def _read_file(self, path, offset=None, limit=None):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        start = offset if offset else 0
        end = start + limit if limit else len(lines)
        numbered = [f"{i + 1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
        return "".join(numbered)

    def _write_file(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"

    def _edit(self, path, old_string, new_string):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.count(old_string) != 1:
            return "Error: old_string must appear exactly once"
        new_content = content.replace(old_string, new_string)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully edited {path}"

    def _glob(self, pattern):
        files = glob_module.glob(pattern, recursive=True)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return "\n".join(files) if files else "No files found"

    def _grep(self, pattern, path="."):
        result = subprocess.run(f"grep -r '{pattern}' {path}", shell=True, capture_output=True)
        stdout, _ = self._decode_subprocess_result(result)
        return stdout if stdout else "No matches found"

    def _get_time(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _web_search(self, query: str, max_results: int = 10) -> str:
        """
        使用duckduckgo_search 搜索网络结果，然后再叫模型进行总结
        :param query: 搜索内容
        :param max_results: 最多搜索条数，默认10
        :return: 经过模型总结后的结果
        """
        max_results = int(max_results) if max_results else 10
        max_results = max(1, min(max_results, 10))

        try:
            # 1) 搜索网络
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"未搜索到与“{query}”相关的结果。"

            # 2) 整理搜索结果，给模型做总结
            search_text_lines = []
            for i, item in enumerate(results, 1):
                title = item.get("title", "").strip()
                body = item.get("body", "").strip()
                href = item.get("href", "").strip()

                search_text_lines.append(
                    f"{i}. 标题: {title}\n"
                    f"   摘要: {body}\n"
                    f"   链接: {href}\n"
                )
            search_text = "\n".join(search_text_lines)

            summarization_prompt = f"""
You are a professional research analyst. Please provide a summary based on the following user search content and web search results.
# Requirements:
1.Source Fidelity: Summarize based only on the provided search results; do not fabricate facts or include external information.
2.Structure: Provide a concise conclusion first, followed by a bulleted list of key points.
3.Conflict Resolution: If there are contradictions or conflicts between different sources, clearly point them out.
4.Sufficiency Check: If the provided information is insufficient to answer the query, explicitly state that information is lacking.
5.Language Consistency: If the user's query is in Chinese, respond in Chinese. If the query is in English, respond in English.
6.Tone: The output must be polished and suitable for direct presentation to the end user.
"""
            user_content = f"""
# User search content
{query}
# Search Results
{search_text}
"""
            # 3) 调用模型总结
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": summarization_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )

            summary = summary_response.choices[0].message.content
            return summary if summary else "未能生成总结。"

        except Exception as e:
            return f"WEB_SEARCH 执行失败: {e}"

    def _list_dir(self, path):
        """
        列出目录path下的所有内容，忽略.git等
        :param path: 路径
        :return: 该路径下的所有内容
        """
        entries = sorted(os.listdir(path))
        result = []
        for entry in entries:
            full = os.path.join(path, entry)
            prefix = "[dir]" if os.path.isdir(full) else "[file]"
            result.append(f"{prefix} {entry}")
        return "\n".join(result) or "Empty directory"

    def _agent_file_path(self, relative_path):
        return os.path.join(os.path.dirname(__file__), relative_path)

    def _save_precise_memory(self, task, result):
        if not self._is_main_agent:
            # 如果不是主agent，即由主agent临时创建的子agent，就不保存记忆
            return ""
        timestamp = self._get_time()
        entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
        try:
            with open(self._agent_file_path(self.precise_memory_file), "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"Error in saving precise memory: {task}, exception: {e}")

    def _save_full_memory(self, task, result):
        if not self._is_main_agent:
            return ""

        timestamp = self._get_time()
        context = self._current_task_full_context or []
        entry = (
            f"\n## {timestamp}\n"
            f"**Task:** {task}\n"
            f"**Result:** {result}\n\n"
            "### Full Context\n"
            "```json\n"
            f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        try:
            with open(self._agent_file_path(self.full_memory_file), "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            print(f"Error in saving full memory: {task}, exception: {e}")

    def _load_precise_memory(self):
        # 如果是子agent，就不给前面的记忆
        if not self._is_main_agent:
            return ""
        memory_path = self._agent_file_path(self.precise_memory_file)
        try:
            if not os.path.exists(memory_path):
                print("The agent is initializing for the first time, creating the memory file")
                with open(memory_path, "w", encoding="utf-8") as f:
                    f.write("")
                return ""
            with open(memory_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                return "\n".join(lines[-50:]) if len(lines) > 50 else content
        except Exception as e:
            print(f"Error in loading memory, exception: {e}")

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
                    "content": "You are a task planning assistant. Break down the task into simple steps as JSON object with key 'steps'.",
                },
                {"role": "user", "content": f"Task: {task}"},
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

    def _load_rules(self):
        """
        加载所有规则md文档，字符串形式返回
        :return:
        """
        rules = []
        if not os.path.exists(self.rules_dir):
            return rules
        system_time = self._get_time()
        for rule_file in Path(self.rules_dir).glob("*.md"):
            with open(rule_file, "r", encoding="utf-8") as f:
                content = f.read().replace("<system-time>", system_time)
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

    def _compact_messages(self):
        if len(self.messages) <= self._COMPACT_THRESHOLD:
            return

        system_msg = self.messages[0]
        old_messages = self.messages[1:-self._KEEP_RECENT]
        recent_messages = self.messages[-self._KEEP_RECENT:]

        old_text = ""
        for msg in old_messages:
            role = msg.get("role", "unknown") if isinstance(msg, dict) else getattr(msg, "role", "unknown")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if content:
                old_text += f"[{role}]: {content}\n"

        # TODO 做摘要的提示词应该还需要进一步扩充，以及需要修改加载方式，以及下面重新构造self.messages的提示词内容
        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize the following conversation history. Keep all important facts, file paths, command results, and decisions. Be concise but don't lose critical details."},
                {"role": "user", "content": old_text}
            ]
        )
        summary = summary_response.choices[0].message.content

        self.messages = [
            system_msg,
            {"role": "user", "content": f"[Previous conversation summary]: {summary}"},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. Let me continue."},
            *recent_messages
        ]

    def _run_agent_step(self, tools):
        for i in range(self.max_iterations):
            # 先压缩历史对话
            self._compact_messages()
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

            for tool_call in message.tool_calls:
                function_payload = getattr(tool_call, "function", None)
                if function_payload is None:
                    continue
                function_name = str(getattr(function_payload, "name", ""))
                raw_arguments = str(getattr(function_payload, "arguments", ""))
                function_args = self._parse_tool_arguments(raw_arguments)
                function_impl = self.available_functions.get(function_name)

                if function_impl is None:
                    function_response = f"Error: Unknown tool '{function_name}'"
                elif "_argument_error" in function_args:
                    function_response = f"Error: {function_args['_argument_error']}"
                elif function_name == ToolNameConstant.MAKE_PLAN:
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
                else:
                    # 到这个分支就是正常调用工具
                    try:
                        print(f"[Tool call] tool name: {function_name}, tool arguments: {raw_arguments}")
                        function_response = function_impl(**function_args)
                    except Exception as error:
                        function_response = f"Error when calling '{function_name}': {error}"
                # 加入本次会话的短期记忆
                self._append_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_response, ensure_ascii=False)
                    }
                )
        return "Max iterations reached"

    def _build_system_prompt(self):
        from prompt_builder import build_system_prompt
        # 置空当前prompt
        self._cached_system_prompt = None
        memory = self._load_precise_memory()
        rules = self._load_rules()
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
        if not self._is_main_agent:
            return
        for memory_file in (self.precise_memory_file, self.full_memory_file):
            memory_path = self._agent_file_path(memory_file)
            try:
                if not os.path.exists(memory_path):
                    continue
                with open(memory_path, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception as e:
                print(f"Error in clearing memory {memory_file}, exception: {e}")

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
                self._append_message({"role": "user", "content": f"You received message from teammate:\n{mail}"})
                # 让 Agent 先消化这些消息
                resp = self.client.chat.completions.create(model=self.model, messages=self.messages)
                self._append_message(resp.choices[0].message)
                self.inbox.clear()

            # 再拼接本次任务并执行
            self._append_message({"role": "user", "content": task})
            final_result = self._run_agent_step(self.all_tools)
            # print(f"final result: {final_result}")
            self._save_precise_memory(task, final_result)
            self._save_full_memory(task, final_result)
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
                        self.console.print("[dim]所有历史记忆已清空[/]")
                    continue
                elif not cmd:
                    continue

                # 上面分支都没中，就是用户任务了
                self.chat(user_input)
                self.console.print()
            except KeyboardInterrupt:
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
                {"role": "system", "content": """You are a project manager. Given a task, plan a team of 2-4 members.
Return JSON: {"team": [{"name": "alice", "role": "...", "task": "..."}]}
Rules: use lowercase english names, last member should be a reviewer, keep tasks concise."""},
                {"role": "user", "content": task}
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
# TODO 优化思考：如何保存一个任务的详细上下文且不影响当前完整工作上下文。这里有三种实现：按 self.messages 下标切片最简单，但一旦中途触发压缩会丢细节；给消息加任务边界标记会污染模型上下文；更稳的是在一次 chat(task) 期间额外维护一份“本任务上下文快照”，每次向 self.messages 追加本任务相关消息时同步记录，最后写入 full_memory.md

if __name__ == "__main__":
    from agent_run import AgentRunner
    runner = AgentRunner(
        model="minimax-m2.7:cloud",
        mcp_mode="subprocess",
    )
    runner.run()
