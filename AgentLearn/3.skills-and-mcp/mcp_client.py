# encoding: utf-8
# @Time    : 2026/04/22 00:00
import json
import os
import subprocess
import sys
import uuid
from typing import Any


class MCPClient:
	def __init__(self, server_script: str | None = None):
		base_dir = os.path.dirname(os.path.abspath(__file__))
		self.server_script = server_script or os.path.join(base_dir, "mcp_server.py")
		self.process: subprocess.Popen[str] | None = None

	def start(self):
		if self.process is not None and self.process.poll() is None:
			return
		self.process = subprocess.Popen(
			[sys.executable, "-u", self.server_script],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
		)
		self.ping()

	def close(self):
		if self.process is None:
			return
		if self.process.poll() is None:
			self.process.terminate()
			self.process.wait(timeout=3)
		self.process = None

	def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
		if self.process is None or self.process.poll() is not None:
			raise RuntimeError("MCP server process is not running")
		if self.process.stdin is None or self.process.stdout is None:
			raise RuntimeError("MCP server stdio is unavailable")
		payload = {
			"id": str(uuid.uuid4()),
			"method": method,
			"params": params or {},
		}
		self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
		self.process.stdin.flush()
		line = self.process.stdout.readline()
		if not line:
			stderr = self.process.stderr.read() if self.process.stderr else ""
			raise RuntimeError(f"No response from MCP server. stderr: {stderr}")
		response = json.loads(line)
		if response.get("error"):
			raise RuntimeError(response["error"].get("message", "Unknown MCP error"))
		return response.get("result")

	def ping(self):
		return self._request("ping")

	def list_tools(self) -> list[dict[str, Any]]:
		result = self._request("list_tools")
		return result if isinstance(result, list) else []

	def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
		return self._request("call_tool", {"name": name, "arguments": arguments})

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass
