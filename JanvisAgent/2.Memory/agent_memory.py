# encoding : utf-8
# @Time    : 2026/4/13 21:12
import datetime
import json
import os
import subprocess
import sys
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
		self.tools = self._load_tools()

		# Agent可以调用的工具/方法有哪些，
		self.available_functions = {
			"execute_bash": self._execute_bash,
			"read_file": self._read_file,
			"write_file": self._write_file
		}

		# 记忆文件，记录Agent执行过哪些任务，方便后续追溯
		self.memory_file = "agent_memory.md"

		# 最大迭代次数，防止某个任务死循环
		self.MAX_ITERATIONS = 10

		# 使用的模型
		self.MODEL = os.environ.get("OPENAI_MODEL", model)

		# 温度
		self.temperature = temperature

	def _load_tools(self):
		"""
		加载工具列表
		:return: 工具列表json
		"""
		print("loading tools")
		try:
			tools_path = os.path.join(os.path.dirname(__file__), "tools.json")
			with open(tools_path, "r", encoding="utf-8") as f:
				tools = json.load(f)
				print(f"{len(tools)} tools loaded")
				return tools
		except FileNotFoundError:
			raise FileNotFoundError(f": {tools_path}")
		except json.JSONDecodeError as e:
			raise ValueError(f"JSON parse failed: {e}")

	def _execute_bash(self, command):
		"""
		执行bash命令
		:param command: 命令行
		:return:  无
		"""
		# result = subprocess.run(command, shell = True, capture_output = True, text = True, encoding='utf-8')
		result = subprocess.run(command, shell=True, capture_output=True)
		# 优先尝试 utf-8，失败则回退到 gbk（常见于中文Windows）
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
			stdout = result.stdout.decode('utf-8', errors='replace')
			stderr = result.stderr.decode('utf-8', errors='replace')
		return stdout + stderr

	def _read_file(self, path):
		"""
		读文件
		:param path: 文件路径
		:return: 文件内容
		"""
		# TODO 还需要加上异常处理
		with open(path, 'r', encoding='utf-8') as f:
			return f.read()

	def _write_file(self, path, content):
		"""
		往文件里面写内容
		:param path: 文件路径
		:param content: 内容
		:return:
		"""
		with open(path, 'w', encoding='utf-8') as f:
			f.write(content)
		return f"write to {path}"

	def _save_memory(self, task, result):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
		try:
			with open(self.memory_file, 'a') as f:
				f.write(entry)
		except Exception as e:
			print(f"Error in saving memory: {task}, exception: {e}")

	def _load_memory(self):
		if not os.path.exists(self.memory_file):
			print("There is no memory file")

		try:
			with open(self.memory_file, 'r') as f:
				content = f.read()
			lines = content.split("\n")
			# TODO 这里取后50行有待修改
			return '\n'.join(lines[-50:]) if len(lines) > 50 else content
		except Exception as e:
			print(f"Error in loading memory, exception: {e}")

	def _make_plan(self, task):
		"""
		将任务拆分成子步骤，调模型
		:param task: 用户任务描述
		:return:
		"""
		print("[Planning] Breaking down task...")
		response = self.client.chat.completions.create(
			model=self.MODEL,
			messages=[
				{"role": "system",
				 "content": "You are a task planning assistant. Break down the task into simple, executable steps. Return as JSON array of strings."},
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

	def _run_agent_step(self, task, messages):
		"""
		拆分步骤执行任务
		:param task: 	子任务
		:param messages:  消息
		:return: 消息内容，执行动作列表，消息列表
		"""
		messages.append({"role": "user", "content": task})
		# 记录迭代过程中执行的动作
		actions = []
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
				return message.content, actions, messages

			for tool_call in message.tool_calls:
				function_payload = getattr(tool_call, "function", None)
				if function_payload is None:
					continue
				function_name = str(getattr(function_payload, "name", ""))
				raw_arguments = str(getattr(function_payload, "arguments", ""))
				function_args = self._parse_tool_arguments(raw_arguments)
				print(f"[Tool] {function_name}({function_args})")
				func_impl = self.available_functions.get(function_name)
				if func_impl is None:
					function_response = f"Error: Unknown tool '{function_name}'"
				elif "_argument_error" in function_args:
					function_response = f"Error: {function_args['_argument_error']}"
				else:
					function_response = func_impl(**function_args)
					actions.append({"tool": function_name, "args": function_args})
				messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": function_response})

		# 如果超过最大迭代次数，返回
		return "Max iterations reached", actions, messages

	def agent_run(self, task, use_plan=True):
		"""
		启动agent运行，入口
		:param task 用户输入的任务
		:param use_plan:  启动规划模式，默认True，# TODO 这个参数要用户给，这里比较挫，后面会优化
		:return:
		"""
		# 先加载存在磁盘的长期记忆
		memory = self._load_memory()
		system_prompt = "You are a helpful assistant that can interact with the system. Be concise. You can't execute dangerous command directly such as 'rm -rf *', ':{:|:&};:' and so on"
		if memory:
			system_prompt += f"\n\nPrevious content:\n{memory}"
		messages = [{"role": "system", "content": system_prompt}]
		if use_plan:
			steps = self._make_plan(task)
		else:
			steps = [task]
		all_results = []
		total_steps = len(steps)
		for i, step in enumerate(steps, 1):
			print(f"\n[Step {i}/{total_steps}]: {step}")
			result, actions, messages = self._run_agent_step(step, messages)
			all_results.append(result)
		print(f"\n{result}")

		final_result = "\n".join(all_results)
		self._save_memory(task, final_result)
		return final_result


# if __name__ == '__main__':
# 	myAgent = Agent(model='deepseek-r1:14b')
# 	use_plan = "--plan" in sys.argv
# 	if len(sys.argv) < 2:
# 		print("Usage: python agent_memory.py [--plan] 'your task here'")
# 		print("  --plan: Enable task planning and decomposition")
# 		sys.exit(1)
# 	task = " ".join(sys.argv[1:])
# 	myAgent.agent_run(task, use_plan=use_plan)

# TODO 4.14 凌晨2点遗留问题：1.模型响应要设置超时时间控制，2.写memory.md文件后pycharm打开是乱码，3.模型本身能力不足，生成的工具调用或者命令行都不是很对
