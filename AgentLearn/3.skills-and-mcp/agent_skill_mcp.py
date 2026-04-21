# encoding : utf-8
# @Time    : 2026/4/13 21:12
import datetime
import json
import os
import subprocess
import sys
import glob as glob_module
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from openai import OpenAI


# Agent 类定义
# TODO 这个记忆实现方式很简单，直接写到一个md文件里，更优的做法为放到向量数据库中
class Agent:
	def __init__(self, model="qwen3.5:9b", temperature=0.0, base_url=None, api_key=None):
		"""
		初始化agent对象
		:param model: 使用什么模型，默认 qwen3.5:9b
		:param temperature: 温度，指定模型输出时的随机性，默认0
		:param base_url: 模型API的地址，默认None，None时从环境变量获取
		:param api_key: 模型API_KEY，默认None，None时从环境变量获取
		"""
		# 创建openai请求客户端
		# 前提，使用本地模型的话ollama要是正常运行的状态
		self.client = OpenAI(
			base_url=os.environ.get('OPENAI_BASE_URL') if base_url is None else base_url,
			api_key=os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
		)
		# 可用的工具列表
		self.base_tools = self._load_tools()
		print(f"[Tool] loaded {len(self.base_tools)} base tools")
		self.mcp_tools = self._load_mcp_tools()
		print(f"[Tool] loaded {len(self.mcp_tools)} mcp tools")
		self.all_tools = self.base_tools + self.mcp_tools

		# Agent可以调用的工具/方法有哪些，
		# TODO 有mcp server获取到之后怎么动态的添加到这个列表里
		self.available_functions = {
			"execute_bash": self._execute_bash,
			"read_file": self._read_file,
			"write_file": self._write_file,
			"edit": self._edit,
			"glob": self._glob,
			"grep": self._grep,
			"plan": self._make_plan
		}

		# 记忆文件，记录Agent执行过哪些任务，方便后续追溯
		self.memory_file = "agent_memory.md"

		# 最大迭代次数，防止某个任务死循环
		self.MAX_ITERATIONS = 100

		# 使用的模型
		self.MODEL = model
		print("Agent model: ", self.MODEL)

		# 温度
		self.temperature = temperature

		# 是否规划模式
		self.plan_mode = False

		# 当前任务的plan列表
		self.current_plan = []

		# 记忆文件， TODO 后续修改成从向量数据库中找
		self.MEMORY_FILE = "./agent/memory.md"

		# 规则文件
		self.RULES_DIR = "./agent/rules"

		# skills目录
		self.SKILLS_DIR = "./agent/skills"

		# MCP服务器
		self.MCP_SERVER = None

	def _load_tools(self):
		"""
		加载工具列表,本地和调用mcp服务得到其他工具
		TODO 实现本地MCP服务器
		:return: 工具列表json
		"""
		print("loading tools from local and remote mcp server")
		tools = []
		try:
			tools_path = os.path.join(os.path.dirname(__file__), "local_tools.json")
			with open(tools_path, "r", encoding="utf-8") as f:
				tools += json.load(f)
				print(f"{len(tools)} local tools loaded")

			# TODO 从mcp服务器获取可用的工具列表
			tools += self._load_mcp_from_server()
			return tools
		except FileNotFoundError:
			raise FileNotFoundError(f": {tools_path}")
		except json.JSONDecodeError as e:
			raise ValueError(f"JSON parse failed: {e}")

	def _load_mcp_from_server(self):
		"""
		从MCP服务器获取可用工具列表
		:return:
		"""
		mcp_tools = []

		print("loading mcp tools from local and remote mcp server")

		print(f"{len(mcp_tools)} mcp tools loaded")
		return mcp_tools

	def _execute_bash(self, command):
		"""
		执行bash命令
		:param command: 命令行
		:return:  无
		"""
		try:
			result = subprocess.run(command, shell=True, capture_output=True)
			# 优先尝试 utf-8，失败则回退到 gbk（常见于中文Windows）
			stdout, stderr = self._decode_subprocess_result(result)
			return stdout + stderr
		except Exception as e:
			print(f"Exception when executing command '{command}': {str(e)}")

	def _decode_subprocess_result(self, result: CompletedProcess[bytes] | CompletedProcess[Any]):
		"""
		decode subprocess执行后的结果
		:param result:
		:return:
		"""
		for enc in ('utf-8', 'gbk', 'gb18030'):
			try:
				stdout = result.stdout.decode(enc)
				stderr = result.stderr.decode(enc)
				print(f"decode with encoding {enc} success")
				break
			except UnicodeDecodeError:
				continue
		else:
			# 所有编码都失败，使用 replace 强制解码
			print("all encoding failed, try to use replace")
			stdout = result.stdout.decode('utf-8', errors='replace')
			stderr = result.stderr.decode('utf-8', errors='replace')
		return stdout, stderr

	def _read_file(self, path, offset=None, limit=None):
		"""
		读文件
		:param path: 文件路径
		:return: 文件内容
		"""
		try:
			with open(path, 'r', encoding='utf-8') as f:
				lines = f.readlines()
			start = offset if offset else 0
			end = start + limit if limit else len(lines)
			numbered = [f"{i + 1:4d} {line}" for i, line in enumerate(lines[start:end], start)]
			return ''.join(numbered)
		except Exception as e:
			return f"Error in reading file: {str(e)}"

	def _write_file(self, path, content):
		"""
		往文件里面写内容
		:param path: 文件路径
		:param content: 内容
		:return:
		"""
		try:
			with open(path, 'w', encoding='utf-8') as f:
				f.write(content)
			return f"Successfully wrote to {path}"
		except Exception as e:
			return f"Error in writing file: {str(e)}"

	def _edit(self, path, old_string, new_string):
		"""
		编辑文件
		:param path:
		:param old_string:
		:param new_string:
		:return:
		"""
		try:
			with open(path, 'r', encoding='utf-8') as f:
				content = f.read()
			if content.count(old_string) != 1:
				return f"Error: old_string must appear exactly once"
			new_content = content.replace(old_string, new_string)
			with open(path, 'w') as f:
				f.write(new_content)
			return f"Successfully edited {path}"
		except Exception as e:
			return f"Error: {str(e)}"

	def _glob(self, pattern):
		try:
			files = glob_module.glob(pattern, recursive=True)
			files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
			return '\n'.join(files) if files else "No files found"
		except Exception as e:
			return f"Exception when glob: {str(e)}"

	def _grep(self, pattern, path="."):
		try:
			result = subprocess.run(f"grep -r '{pattern}' {path}", shell=True, capture_output=True, text=True,
									timeout=30)
			stdout, _ = self._decode_subprocess_result(result)
			return stdout if stdout else "No matches found"
		except Exception as e:
			return f"Exception when grep: {str(e)}"

	def _save_memory(self, task, result):
		"""
		保存长期记忆，往磁盘里写
		# TODO 最好改成往向量数据库里写
		:param task: 任务
		:param result: 结果
		:return:
		"""
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
		try:
			with open(self.memory_file, 'a', encoding='utf-8') as f:
				f.write(entry)
		except Exception as e:
			print(f"Error in saving memory: {task}, exception: {e}")

	def _load_memory(self):
		if not os.path.exists(self.memory_file):
			print("There is no memory file")
			return ""

		try:
			with open(self.memory_file, 'r', encoding='utf-8') as f:
				content = f.read()
				lines = content.split("\n")
				# TODO 取记忆的逻辑需要改成从向量数据库里获取和当前任务相关的top n条记忆
				return '\n'.join(lines[-50:]) if len(lines) > 50 else content
		except Exception as e:
			print(f"Error in loading memory, exception: {e}")
			return ""

	def _make_plan(self, task):
		"""
		将任务拆分成子步骤，调模型
		:param task: 用户任务描述
		:return:
		"""
		if self.plan_mode:
			return "Error: can't make plan within a plan"
		print("[Planning] Breaking down task {}".format(task))
		response = self.client.chat.completions.create(
			model=self.MODEL,
			messages=[
				{
					"role": "system",
					"content": "You are a task planning assistant. Break down the task into simple, executable steps. Return as JSON array of strings. For example, you should return like {\"steps\" : [ \"step 1: dosomething\",\"step 2: dosomething\"]}"
				},
				{"role": "user", "content": f"Task: {task}"}
			],
			response_format={"type": "json_object"},
			temperature=self.temperature
		)
		try:
			plan_data = json.loads(response.choices[0].message.content)
			if isinstance(plan_data, dict):
				steps = plan_data.get("steps", [task])
			elif isinstance(plan_data, list):
				steps = plan_data
			else:
				steps = [task]
			self.current_plan = steps
			print(f"[Plan] {len(steps)} steps created")
			for i, step in enumerate(steps, 1):
				print(f"{i}: {step}")
			return steps
		except Exception as e:
			# 异常情况直接返回这个任务,至少保证流程能运行下去
			print(f"[Plan] Failed to parse steps, returning original task {task}, exception {e}")
			return [task]

	def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
		"""
		解析工具调用的参数
		:param raw_arguments: 参数字符串
		:return: 解析成字典
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
		加载规则
		:return:
		"""
		rules = []
		if not os.path.exists(self.RULES_DIR):
			print("There is no rule dir")
			return rules
		try:
			for rule_file in Path(self.RULES_DIR).glob("*.md"):
				with open(rule_file, 'r', encoding='utf-8') as f:
					rules.append(f.read())

			return "\n\n".join(rules) if rules else []
		except Exception as e:
			print(f"Error in loading rules: {e}")
			return []

	def _load_skill_meta_infos(self):
		"""
		加载skill元信息，有需要时再去加载完整的那个skill
		:return:
		"""
		skills = []
		if not os.path.exists(self.SKILLS_DIR):
			print("There is no skill dir")
			return []
		try:
			# TODO 这里要修改成从md格式的文件中加载skill
			for skill_file in Path(self.SKILLS_DIR).glob("*.json"):
				with open(skill_file, 'r') as f:
					skills.append(json.load(f))
			return skills
		except Exception as e:
			print(f"Error in loading skills: {e}")
			return []

	def _load_skill_by_name(self, skill_name):
		"""
		根据skill名称加载完整的skill
		:param skill_name: skill名称
		:return:
		"""
		pass

		# TODO 实现根据名称加载完整skill

	def _load_mcp_tools(self):
		"""
		从mcp server加载工具列表
		:return: 工具列表
		"""
		if not self.MCP_SERVER:
			print("There is no mcp_server address")
			return []
		# TODO 实现调MCP server地址获得mcp工具列表，还要实现一个mcp client来执行具体的工具逻辑
		mcp_tools = []

		return mcp_tools

	def _run_agent_step(self, messages, tools):
		"""
		拆分步骤执行任务
		:param tools: 	工具列表
		:param messages:  消息
		:return: 执行结果
		"""
		for _ in range(self.MAX_ITERATIONS):
			response = self.client.chat.completions.create(
				model=self.MODEL,
				messages=messages,
				tools=self.tools,
				temperature=self.temperature
			)
			message = response.choices[0].message
			messages.append(message)
			# 如果某一个iter时不是工具调用，说明该任务结束了，返回消息内容，中间行动，消息列表
			if not message.tool_calls:
				return message.content, messages
			# 遍历工具列表
			for tool_call in message.tool_calls:
				function_payload = getattr(tool_call, "function", None)
				if function_payload is None:
					continue
				function_name = str(getattr(function_payload, "name", ""))
				raw_arguments = str(getattr(function_payload, "arguments", ""))
				function_args = self._parse_tool_arguments(raw_arguments)
				print(f"[Tool call] {function_name}(params: {function_args})")
				function_impl = self.available_functions.get(function_name)
				if function_impl is not None:
					function_response = function_impl(**function_args)
				elif "_argument_error" in function_args:
					function_response = f"Error: {function_args['_argument_error']}"
				elif function_name == "make_plan" and function_impl is not None:
					# 如果是plan模式
					self.plan_mode = True
					function_response = function_impl(**function_args)
					messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})
					if self.current_plan:
						results = []
						plan_size = len(self.current_plan)
						for i, step in enumerate(self.current_plan, 1):
							print(f"\n[Step {i}/{plan_size}]: {step}")
							messages.append({"role": "user", "content": step})
							result, messages = self._run_agent_step(messages, [t for t in tools if
																			   t["function"]["name"] != "make_plan"])
							results.append(result)
							print(f"\n{result}")
						self.plan_mode = False
						self.current_plan = []
						return "\n".join(results), messages
				else:
					function_response = f"Error: Unknown tool '{function_name}'"
				messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})

		# 如果超过最大迭代次数，返回
		return "Max iterations reached", messages

	def agent_run(self, task):
		"""
		启动agent运行，入口
		:param task 用户输入的任务
		:return:
		"""
		print("[Init] Initializing agent..")
		# 先加载存在磁盘的长期记忆
		memory = self._load_memory()
		rules = self._load_rules()
		skills = self._load_skill_meta_infos()
		mcp_tools = self._load_mcp_tools()
		print(f"[MCP] Got {len(mcp_tools)} mcp tools.")
		# 构造系统prompt，base prompt + rules + memory + skills + tools
		# TODO 提示词加怎么让它用skill
		system_prompt = ["You are an interactive agent that helps users with daily tasks or software engineering tasks. Use the instructions below and the tools available to you to assist the user."]
		if rules:
			system_prompt.append(f"\n# Rules\n{rules}")
			print(f"[Rules] {len(rules.split('# '))} rule files loaded.")
		if skills:
			system_prompt.append(f"\n# Skills\n{skills}\n" + "\n".join([f"- {skill['name']}: {skill.get('description', '')}" for skill in skills]))
			print(f"[Skills] {len(skills)} skill files loaded.")
		if self.all_tools:
			print(f"[Tools] {len(self.all_tools)} tools loaded.")
		if memory:
			system_prompt.append(f"\n# Previous context\n{memory}")
		messages = [{"role": "system", "content": "\n".join(system_prompt)}, {"role": "user", "content": task}]
		final_result, messages = self._run_agent_step(messages, self.all_tools)
		print(f"\nFinal result: {final_result}")
		self._save_memory(task, final_result)
		return final_result


if __name__ == '__main__':
	myAgent = Agent(model="minimax-m2.7:cloud")
	# use_plan = "--plan" in sys.argv
	# if len(sys.argv) < 2:
	# 	print("Usage: python agent_memory.py [--plan] 'your task here'")
	# 	print("  --plan: Enable task planning and decomposition")
	# 	sys.exit(1)
	# task = " ".join(sys.argv[1:])
	task = "找到当前目录下所有文件中的TODO内容并整理到TODO.md文件中，如果TODO.md文件已存在，就先删除它"
	myAgent.agent_run(task, use_plan=True)

# TODO 4.14 凌晨2点遗留问题：1.模型响应要设置超时时间控制，2.写memory.md文件后pycharm打开是乱码，3.模型本身能力不足，生成的工具调用或者命令行都不是很对
