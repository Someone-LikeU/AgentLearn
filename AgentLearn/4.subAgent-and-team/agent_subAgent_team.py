# encoding : utf-8
# @Time    : 2026/4/19
import glob as glob_module
import json
import os
import subprocess
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


class Agent:
	"""支持本地工具 + MCP工具的Agent。"""

	def __init__(self, model="qwen3.5:9b",
				 temperature: float=0.1,
				 base_url: str=None,
				 api_key: str=None,
				 mcp_server_script: str=None,
				 is_main_agent: bool = True,
				 role: str="Main Agent"):
		"""
		初始化Agent对象
		:param model: 使用模型
		:param temperature: 模型推理时温度
		:param base_url: 模型的url
		:param api_key: 模型api_key
		:param mcp_server_script: mcp服务器脚本名
		:param is_main_agent: 是否是主Agent，默认True
		:param role: Agent角色，默认为主agent
		"""
		# base_url
		self._base_url = os.environ.get("OPENAI_BASE_URL") if base_url is None else base_url

		# api_key
		self._api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key

		# openAI 请求客户端
		self.client = OpenAI(
			base_url=self._base_url,
			api_key=self._api_key,
		)

		# 记忆文件
		self.memory_file = "agent/memory.md"

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
			ToolNameConstant.LIST_DIR: self._list_dir,
		}
		# TODO 这里客户端后续要剥离出来，不在这里初始化，在一个编排类里面初始化
		self.mcp_client = MCPClient(server_script=mcp_server_script)
		self.mcp_client.start()
		# 加载MCP工具
		self.mcp_tools = self._load_mcp_tools()
		print(f"{len(self.mcp_tools)} MCP tools loaded")

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
		self._cached_system_prompt = None

		# 是否是主agent, False表示由Agent创建的子Agent,默认为True
		self._is_main_agent = is_main_agent

		self.console = Console()

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
		# 安全检查：拦截危险命令
		dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/sd"]
		if any(d in command for d in dangerous):
			return "The command is dangerous, refused to execute it.", "The command is dangerous, refused to execute it."
		result = subprocess.run(command, shell=True, capture_output=True)
		stdout, stderr = self._decode_subprocess_result(result)
		return stdout + stderr

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
		
	def _save_memory(self, task, result):
		if not self._is_main_agent:
			# 如果不是主agent，即由主agent临时创建的子agent，就不保存记忆
			return ""
		timestamp = self._get_time()
		entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
		try:
			with open(self.memory_file, "a", encoding="utf-8") as f:
				f.write(entry)
		except Exception as e:
			print(f"Error in saving memory: {task}, exception: {e}")

	def _load_memory(self):
		# 如果是子agent，就不给前面的记忆
		if not self._is_main_agent:
			return ""
		memory_path = os.path.join(os.path.dirname(__file__), self.memory_file)
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
		for rule_file in Path(self.rules_dir).glob("*.md"):
			with open(rule_file, "r", encoding="utf-8") as f:
				rules.append(f.read())
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

	def _run_agent_step(self, messages, tools):
		for i in range(self.max_iterations):
			response_stream = self.client.chat.completions.create(
				model=self.model,
				messages=messages,
				tools=tools,
				temperature=self.temperature,
				stream=True,
			)
			# message = response_stream.choices[0].message
			# messages.append(message)
			message = self._deal_stream_response(response_stream)
			print()
			# 按照OpenAI的格式对message进行复原然后加回历史对话
			messages.append({
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
				return message.content, messages
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
					self.plan_mode = True
					steps = function_impl(**function_args)
					if not isinstance(steps, list):
						function_response = steps
					else:
						results = []
						step_cnt = 0
						for step in steps:
							print(f"[Step {step_cnt + 1}]: {step}")
							messages.append({"role": "user", "content": step})
							result, messages = self._run_agent_step(
								messages, [t for t in tools if t["function"]["name"] != ToolNameConstant.MAKE_PLAN]
							)
							print(f"[Step {step_cnt + 1}] result:{result}, messages: {messages}")
							step_cnt += 1
							results.append(result)
						function_response = "\n".join(results)
					self.plan_mode = False
					self.current_plan = []
				else:
					try:
						print(f"[Tool call] tool name: {function_name}, tool arguments: {raw_arguments}")
						function_response = function_impl(**function_args)
					except Exception as error:
						function_response = f"Error when calling '{function_name}': {error}"

				messages.append(
					{
						"role": "tool",
						"tool_call_id": tool_call.id,
						"content": json.dumps(function_response, ensure_ascii=False)
					}
				)
		return "Max iterations reached", messages

	def _build_system_prompt(self):
		from prompt_builder import build_system_prompt
		# 置空当前prompt
		self._cached_system_prompt = None
		memory = self._load_memory()
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
		TODO 抄一下那个执行bash命令的黑名单还有列出目录文件的工具逻辑,以及GPT给我的另外两个工具
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
					idx = tc.idx
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

	def _close(self):
		# 关闭mcp客户端，TODO 后续修改为在agent loop前后进行打开和关闭，不要让外界感知
		self.mcp_client.close()

	def _sub_agent(self, role, task):
		"""
		调用一个子agent，处理一个专门的子任务
		这里sub_agent的话，怎么编排？怎么输入输出？结果怎么处理
		:param role:
		:param task:
		:return:
		"""
		if not self._is_main_agent:
			# TODO 这里的返回格式可能需要再改
			return "Error: can't create sub-agent within a sub-agent"
		# TODO 这里如果只是新建一个Agent，还存在如下问题：
		#  	1、memory文件共享的问题，没有做到记忆隔离,
		#  	2、还有system prompt要区分出子agent和主agent的部分
		#   3、本地工具列表里现在还没加主agent有的sub_agent方法，加上后新建子agent对象时怎么少给这个本地方法
		#   4、子agent的运行结果怎么返回，主agent和子agent的提示词有哪些需要区分的，怎么用文件区分出来
		#   5、子agent直接执行agent_run时，会有一步save_memory，会和主agent矛盾，写同一个memory文件，这个要怎么处理
					# save_memory的时候先判断是否是主agent，否就直接返回
		# TODO 如果把run的逻辑改为一个编排类来做，这里的逻辑可能还得改
		# TODO 建一个记录过程中怎么设计架构的文档
		# TODO 可能需要将现有逻辑拆解出一个编排类来解决以上关于子agent的问题

		# TODO 28日最新，先改一版Agent loop的出来
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
		pass

	def chat(self, task):
		"""
		Agent运行入口
		:param task: 用户任务
		:return: 执行任务结果
		"""
		system_prompt = self._build_system_prompt()
		# 拼接完整上下文
		messages = [
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": task}
				]
		final_result, _ = self._run_agent_step(messages, self.all_tools)
		print(f"final result: {final_result}")
		self._save_memory(task, final_result)

		self._close()
		return final_result

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
				elif not cmd:
					continue
				
				# 上面分支都没中，就是用户任务了
				self.chat(user_input)
				self.console.print()
			except KeyboardInterrupt:
				self.console.print("\n[bold red] See you next time! [/]")
				break
		

if __name__ == "__main__":
	my_agent = Agent(model="minimax-m2.7:cloud")
	task = "找到当前目录下所有文件中的TODO内容并整理到TODO.md文件中，如果TODO.md文件已存在，就先删除它"
	my_agent.run()

	# TODO 新建一个编排类，由这个编排类来控制Agent的运行，需要剥离Agent的mcp_server属性
	# TODO 任务完成得不好，考虑设计一个评价器，调整温度重新生成
	# TODO 实现后台定时任务，agent自主行动，类似车机上车后自动打开空调等
	# TODO 记忆系统修改，维护两个记忆md文档，一个放未压缩的，一个放压缩的，运行时写两个文件，load记忆时优先load压缩的，再结合RAG做运行时检索旧记忆
	# TODO sub agent的记忆怎么处理
	# TODO agent的角色问题，比如问你是谁，系统提示词里加角色扮演以及工具或者skills等东西无法回答用户问题时回答“对不起。。。。。”
	# TODO 模型配置，比如api_key, model, base_url等，用一个json，再提供工具给用户来修改模型等参数
	# fc-90a9530d614f483f8a26d7f427be688d firecrawl秘钥

	"""
	TODO 子agent，两种实现方式：
	1、隐式定义：一个类似读写文件、执行bash命令等的工具，在工具中另起一个Agent对象，通过提示词区分角色，该agent临时启用，生命周期为本次对话，任务完成就丢失
		需要扩展Agent对象的属性，新增is_main_agent、agent_name、agent id等属性，
	
	2、显式定义：针对特定任务预先定义一个独立的子agent，独立的设定，可用工具列表和主agent存在部分交集，可能完全继承所有工具，也可能有主agent没有的工具，
		记忆可以存到磁盘里做永久，下次类似的任务唤起该子agent时能load上一次的工作记忆
	
	二者区别：
	对比项	 |    隐式    |      显式
	创建方式  | agent运行时动态创建 | 预先根据特定任务定义特定agent，代码/配置文件定义
	生命周期  | 临时创建，任务完成即销毁 | 可复用，类似于程序可永久存活在磁盘
	有无状态  |  有状态   |	无状态
	可控性	| 灵活，可控性弱 | 可控性强，输出可预测
	适用场景	| 不确定性任务，由Agent自主决策 | 流程可固化的企业工作流，可配合Skill一起实现
	"""
