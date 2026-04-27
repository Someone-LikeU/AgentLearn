# encoding: utf-8
# @Time    : 2026/04/22
import json
import socket
import sys
import threading
import traceback
from typing import Any

from mcp_tools import MCPToolsRegistry


class MCPServer:
	"""
	MCP 本地服务端实现，支持 STDIO 和 TCP 两种通信方式
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
		"""
		通过 STDIO 方式提供服务（原有方式）
		"""
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

	def serve_tcp(self, host: str = "127.0.0.1", port: int = 8765):
		"""
		通过 TCP Socket 方式提供服务
		:param host: 监听地址
		:param port: 监听端口
		"""
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		server_socket.bind((host, port))
		server_socket.listen(5)
		print(f"MCP Server listening on {host}:{port}", file=sys.stderr)
		sys.stderr.flush()

		while True:
			try:
				client_socket, address = server_socket.accept()
				threading.Thread(
					target=self._handle_client,
					args=(client_socket,),
					daemon=True
				).start()
			except Exception as e:
				print(f"Error accepting connection: {e}", file=sys.stderr)
				break

	def _handle_client(self, client_socket: socket.socket):
		"""
		处理客户端连接
		"""
		try:
			buffer = ""
			while True:
				data = client_socket.recv(4096)
				if not data:
					break
				buffer += data.decode("utf-8")
				
				while "\n" in buffer:
					line, buffer = buffer.split("\n", 1)
					line = line.strip()
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
					
					client_socket.sendall(
						(json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
					)
		except Exception as e:
			print(f"Client error: {e}", file=sys.stderr)
		finally:
			client_socket.close()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="MCP Server")
	parser.add_argument("--mode", choices=["stdio", "tcp"], default="stdio",
						help="Communication mode")
	parser.add_argument("--host", default="127.0.0.1", help="TCP host")
	parser.add_argument("--port", type=int, default=7777, help="TCP port")
	args = parser.parse_args()

	server = MCPServer()
	if args.mode == "tcp":
		server.serve_tcp(args.host, args.port)
	else:
		server.serve_stdio()
