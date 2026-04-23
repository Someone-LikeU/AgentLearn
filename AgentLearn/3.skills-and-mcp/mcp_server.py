# encoding: utf-8
# @Time    : 2026/04/22
import json
import sys
import traceback
from typing import Any

from mcp_tools import MCPToolsRegistry


class MCPServer:
	"""
	MCP 本地服务端实现
	"""
	def __init__(self):
		self.registry = MCPToolsRegistry()

	def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
		"""
		请求分发
		:param request: JSON-RPC格式的请求体
		:return:
		"""
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
		# 持续监听stdin的请求
		for raw_line in sys.stdin:
			line = raw_line.strip()
			if not line:
				continue
			try:
				request = json.loads(line)
			except json.JSONDecodeError as error:
				response = {
					"id": None,
					"error": {"message": f"Invalid JSON input: {error}"},
				}
			else:
				response = self.handle_request(request)

			# 将响应写入stdout
			sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
			sys.stdout.flush()


if __name__ == "__main__":
	MCPServer().serve_stdio()
