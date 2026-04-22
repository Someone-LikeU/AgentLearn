# encoding: utf-8
# @Time    : 2026/04/22 00:00
import json
import sys
import traceback
from typing import Any

from mcp_tools import MCPToolsRegistry


class MCPServer:
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
				response = {
					"id": None,
					"error": {"message": f"Invalid JSON input: {error}"},
				}
			else:
				response = self.handle_request(request)
			sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
			sys.stdout.flush()


if __name__ == "__main__":
	MCPServer().serve_stdio()
