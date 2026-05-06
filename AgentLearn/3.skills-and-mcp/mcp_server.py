# encoding: utf-8
# @Time    : 2026/04/24 00:00
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from mcp_tools import MCPToolsRegistry


class MCPServer:
	"""MCP服务端：只关注协议处理和工具调度。"""

	def __init__(self):
		self.registry = MCPToolsRegistry()

	def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
		request_id = request.get("id")
		method = request.get("method")
		params = request.get("params", {})
		try:
			if method == "ping":
				result = {"message": "pong"}
			elif method == "list_tools":
				result = self.registry.list_tools()
			elif method == "call_tool":
				tool_name = params.get("name")
				arguments = params.get("arguments", {})
				result = self.registry.call_tool(tool_name, arguments)
			else:
				raise ValueError(f"Unknown method '{method}'")
			return {"id": request_id, "result": result}
		except Exception as error:
			return {
				"id": request_id,
				"error": {
					"message": str(error),
					"traceback": traceback.format_exc(limit=3),
				},
			}

	def serve_stdio(self):
		for raw_line in sys.stdin:
			line = raw_line.strip()
			if not line:
				continue
			try:
				request = json.loads(line)
			except json.JSONDecodeError as error:
				response = {"id": None, "error": {"message": f"Invalid JSON input: {error}"}}
			else:
				response = self.handle_request(request)
			sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
			sys.stdout.flush()


class MCPServerProcess:
	"""MCP服务进程管理器：由外部管理类创建，不放在client中，降低耦合。"""

	def __init__(self, server_script: str | None = None):
		default_script = Path(__file__).resolve()
		self.server_script = str(default_script if server_script is None else Path(server_script).resolve())
		self.process: subprocess.Popen[str] | None = None

	def start(self):
		if self.process is not None and self.process.poll() is None:
			return self.process
		self.process = subprocess.Popen(
			[sys.executable, "-u", self.server_script, "--mode", "server"],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
		)
		return self.process

	def stop(self):
		if self.process is None:
			return
		if self.process.poll() is None:
			self.process.terminate()
			self.process.wait(timeout=3)
		self.process = None


def main():
	parser = argparse.ArgumentParser(description="MCP server")
	parser.add_argument("--mode", choices=["server"], default="server")
	args = parser.parse_args()
	if args.mode == "server":
		MCPServer().serve_stdio()


if __name__ == "__main__":
	main()
