# encoding : utf-8
# @Author  : Zjj
# @Time    : 2026/4/1 21:12
# @File    : hello.py
# @Contact : peterporkerzjj@163.com
import datetime
import json
import os
import subprocess
import sys

from openai import OpenAI

# 创建openai请求客户端
client = OpenAI(
	base_url=os.environ.get('OPENAI_BASE_URL'),
	api_key=os.environ.get("OPENAI_API_KEY")
)
# 前提，ollama要是正常运行的状态

# 这个列表是给大模型看的
tools = [
	{
		"type": "function",
		"function": {
			"name": "execute_bash",
			"description": "Execute bash command",
			"params": {
				"type": "object",
				"properties": {"command": {"type": "string"}},
				"required": ["command"]
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "read_file",
			"description": "Read a file",
			"parameters": {
				"type": "object",
				"properties": {"path": {"type": "string"}},
				"required": ["path"]
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "write_file",
			"description": "Write content to a file",
			"parameters": {
				"type": "object",
				"properties": {
					"path": {"type": "string"},
					"content": {"type": "string"},
				},
			},
			"required": ["path", "content"]
		}
	}
]


def execute_bash(command):
	"""
	执行bash命令
	:param command: 命令
	:return:  无
	"""
	result = subprocess.run(command, shell=True, capture_output=True, text=True)
	return result.stdout + result.stderr


def read_file(path):
	"""
	读文件
	:param path: 文件路径
	:return: 文件内容
	"""
	with open(path, 'r') as f:
		return f.read()


def write_file(path, content):
	"""
	往文件里面写内容
	:param path: 文件路径
	:param content: 内容
	:return:
	"""
	with open(path, 'w') as f:
		f.write(content)
	return f"write to {path}"


# 定义可以调用的工具/方法有哪些，这个dict是给程序看的
functions = {
	"execute_bash": execute_bash,
	"read_file": read_file,
	"write_file": write_file
}

MEMORY_FILE = "agent_memory.md"


def save_memory(task, result):
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
	try:
		with open(MEMORY_FILE, 'a') as f:
			f.write(entry)
	except Exception as e:
		print(f"Error in saving memory: {task}, exception: {e}")


def load_memory():
	if not os.path.exists(MEMORY_FILE):
		print("There is no memory file")

	try:
		with open(MEMORY_FILE, 'r') as f:
			content = f.read()
		lines = content.split("\n")
		# TODO 这里取后50行有待修改
		return '\n'.join(lines[-50:]) if len(lines) > 50 else content
	except Exception as e:
		print(f"Error in loading memory, exception: {e}")


def agent_run(user_message, max_iteration=10):
	memory = load_memory()
	system_prompt = "You are a helpful assistant that can interact with the system. Be concise."
	# TODO 这里加载记忆的方式比较粗暴
	if memory:
		system_prompt += f"\n\nPrevious content:\n{memory}"

	messages = [
		# system prompt
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_message}
	]
	for i in range(max_iteration):
		response = client.chat.completions.create(
			model='qwen3.5:9b',
			messages=messages,
			tools=tools,
			# 还有温度可以设置
		)
		message = response.choices[0].message
		messages.append(message)
		print(f"Step{i + 1}/{max_iteration} messages: {message}")
		if not message.tool_calls:
			print(f"Step{i + 1}/{max_iteration} No tool calls")
			return message.content
		for tool_call in message.tool_calls:
			name = tool_call.function.name
			args = json.loads(tool_call.function.arguments)
			print(f"Step{i + 1}/{max_iteration}[Tool] {name}: {args}")
			if name not in functions:
				result = f"Error: Unknown tool '{name}'"
			else:
				result = functions[name](**args)
			messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

	return "Max iterations reached"


if __name__ == '__main__':
	# response = client.chat.completions.create(
	#     model = 'qwen3.5:9b',
	#     messages = [
	#         {"role": "system", "content": "你是一个智能助手。"},
	#         {"role": "user", "content": "你是谁？"}
	#     ]
	# )
	# message = response.choices[0].message
	# print(type(message))
	# print(message)
	task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "你好"
	print(agent_run(task))