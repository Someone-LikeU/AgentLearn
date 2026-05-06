# encoding: utf-8
# @Time    : 2026/04/24 00:00
import json
import subprocess
import uuid
from typing import Any, TextIO


class MCPClient:
	"""MCP客户端：仅负责通过已有stdio通道发送/接收请求，不负责启动服务端。"""

	def __init__(self, reader: TextIO, writer: TextIO, error_reader: TextIO | None = None):
		self.reader = reader
		self.writer = writer
		self.error_reader = error_reader

	@classmethod
	def from_process(cls, process: subprocess.Popen[str]):
		"""通过外部创建好的server进程构建client。"""
		if process.stdin is None or process.stdout is None:
			raise RuntimeError("MCP server stdio is unavailable")
		return cls(reader=process.stdout, writer=process.stdin, error_reader=process.stderr)

	def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
		payload = {
			"id": str(uuid.uuid4()),
			"method": method,
			"params": params or {},
		}
		self.writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
		self.writer.flush()
		line = self.reader.readline()
		if not line:
			stderr = self.error_reader.read() if self.error_reader else ""
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
