# encoding : utf-8
# @Time    : 2026/4/13 21:12
import datetime
import glob as glob_module
import json
import os
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from openai import OpenAI

from mcp_client import MCPClient


class Agent:
	"""支持本地工具 + MCP工具的Agent。"""

	def __init__(self, model="qwen3.5:9b", temperature=0.0, base_url=None, api_key=None, mcp_server_script=None):
		self.client = OpenAI(
			base_url=os.environ.get("OPENAI_BASE_URL") if base_url is None else base_url,
			api_key=os.environ.get("OPENAI_API_KEY") if api_key is None else api_key,
		)

		self.memory_file = "agent_memory.md"
		self.MAX_ITERATIONS = 100
		self.MODEL = model
		self.temperature = temperature
		self.plan_mode = False
		self.current_plan: list[str] = []
		self.RULES_DIR = "./agent/rules"
		self.SKILLS_DIR = "./agent/skills"

		self.local_tools = self._load_local_tools()
		self.local_functions = {
			"execute_bash": self._execute_bash,
			"read_file": self._read_file,
			"write_file": self._write_file,
			"edit": self._edit,
			"glob": self._glob,
			"grep": self._grep,
			"make_plan": self._make_plan,
		}

		self.mcp_client = MCPClient(server_script=mcp_server_script)
		self.mcp_client.start()
		self.mcp_tools = self._load_mcp_tools()

		self.available_functions: dict[str, Any] = {}
		self.available_functions.update(self.local_functions)
		for tool in self.mcp_tools:
			tool_name = tool["function"]["name"]
			self.available_functions[tool_name] = self._make_mcp_executor(tool_name)

		self.all_tools = self.local_tools + self.mcp_tools

	def _make_mcp_executor(self, tool_name: str):
		def _executor(**kwargs):
			return self.mcp_client.call_tool(tool_name, kwargs)

		return _executor

	def _load_local_tools(self) -> list[dict[str, Any]]:
		tools_path = os.path.join(os.path.dirname(__file__), "local_tools.json")
		with open(tools_path, "r", encoding="utf-8") as f:
			return json.load(f)

	def _load_mcp_tools(self) -> list[dict[str, Any]]:
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

	def _execute_bash(self, command):
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

	def _save_memory(self, task, result):
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		entry = f"\n## {timestamp}\n**Task:** {task}\n**Result:** {result}\n"
		with open(self.memory_file, "a", encoding="utf-8") as f:
			f.write(entry)

	def _load_memory(self):
		if not os.path.exists(self.memory_file):
			return ""
		with open(self.memory_file, "r", encoding="utf-8") as f:
			content = f.read()
			lines = content.split("\n")
			return "\n".join(lines[-50:]) if len(lines) > 50 else content

	def _make_plan(self, task):
		if self.plan_mode:
			return "Error: can't make plan within a plan"
		response = self.client.chat.completions.create(
			model=self.MODEL,
			messages=[
				{
					"role": "system",
					"content": "You are a task planning assistant. Break down the task into simple steps as JSON object with key steps.",
				},
				{"role": "user", "content": f"Task: {task}"},
			],
			response_format={"type": "json_object"},
			temperature=self.temperature,
		)
		try:
			plan_data = json.loads(response.choices[0].message.content)
			steps = plan_data.get("steps", [task]) if isinstance(plan_data, dict) else [task]
			self.current_plan = steps
			return steps
		except Exception:
			return [task]

	def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
		if not raw_arguments:
			return {}
		try:
			parsed = json.loads(raw_arguments)
			return parsed if isinstance(parsed, dict) else {}
		except json.JSONDecodeError as error:
			return {"_argument_error": f"Invalid JSON arguments: {error}"}

	def _load_rules(self):
		rules = []
		if not os.path.exists(self.RULES_DIR):
			return rules
		for rule_file in Path(self.RULES_DIR).glob("*.md"):
			with open(rule_file, "r", encoding="utf-8") as f:
				rules.append(f.read())
		return "\n\n".join(rules) if rules else []

	def _load_skill_meta_infos(self):
		skills = []
		if not os.path.exists(self.SKILLS_DIR):
			return []
		for skill_file in Path(self.SKILLS_DIR).glob("*.json"):
			with open(skill_file, "r", encoding="utf-8") as f:
				skills.append(json.load(f))
		return skills

	def _run_agent_step(self, messages, tools):
		for _ in range(self.MAX_ITERATIONS):
			response = self.client.chat.completions.create(
				model=self.MODEL,
				messages=messages,
				tools=tools,
				temperature=self.temperature,
			)
			message = response.choices[0].message
			messages.append(message)
			if not message.tool_calls:
				return message.content, messages

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
				elif function_name == "make_plan":
					self.plan_mode = True
					steps = function_impl(**function_args)
					if not isinstance(steps, list):
						function_response = steps
					else:
						results = []
						for step in steps:
							messages.append({"role": "user", "content": step})
							result, messages = self._run_agent_step(
								messages, [t for t in tools if t["function"]["name"] != "make_plan"]
							)
							results.append(result)
						function_response = "\n".join(results)
					self.plan_mode = False
					self.current_plan = []
				else:
					try:
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

	def agent_run(self, task):
		memory = self._load_memory()
		rules = self._load_rules()
		skills = self._load_skill_meta_infos()
		system_prompt = [
			"You are an interactive agent that helps users with daily tasks or software engineering tasks."
		]
		if rules:
			system_prompt.append(f"\n# Rules\n{rules}")
		if skills:
			system_prompt.append(
				f"\n# Skills\n{skills}\n" + "\n".join([f"- {skill['name']}: {skill.get('description', '')}" for skill in skills])
			)
		if memory:
			system_prompt.append(f"\n# Previous context\n{memory}")
		messages = [{"role": "system", "content": "\n".join(system_prompt)}, {"role": "user", "content": task}]
		final_result, _ = self._run_agent_step(messages, self.all_tools)
		self._save_memory(task, final_result)
		return final_result

	def close(self):
		self.mcp_client.close()


if __name__ == "__main__":
	my_agent = Agent(model="minimax-m2.7:cloud")
	task = "找到当前目录下所有文件中的TODO内容并整理到TODO.md文件中，如果TODO.md文件已存在，就先删除它"
	try:
		my_agent.agent_run(task)
	finally:
		my_agent.close()
