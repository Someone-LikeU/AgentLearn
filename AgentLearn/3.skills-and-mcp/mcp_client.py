# encoding: utf-8
# @Time    : 2026/04/22
import json
import socket
import subprocess
import sys
import uuid
from typing import Any


class MCPClient:
	"""
	MCP 客户端实现，支持两种通信方式：
	1. 子进程模式：服务器作为客户端的子进程运行（通过 STDIO 通信）
	2. TCP 模式：服务器独立运行，客户端通过 TCP 连接通信
	"""
	def __init__(self, server_script: str | None = None, 
				 mode: str = "subprocess", host: str = "127.0.0.1", port: int = 8765):
		"""
		初始化 MCP 客户端
		:param server_script: 服务器脚本路径（仅子进程模式需要）
		:param mode: 通信模式，"subprocess" 或 "tcp"
		:param host: TCP 模式下的服务器地址
		:param port: TCP 模式下的服务器端口
		"""
		base_dir = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
		self.server_script = server_script or __import__("os").path.join(base_dir, "mcp_server.py")
		self.mode = mode
		self.host = host
		self.port = port
		
		self.process: subprocess.Popen[str] | None = None
		self.socket: socket.socket | None = None

	def start(self):
		"""
		启动客户端和服务器（子进程模式）或仅连接服务器（TCP 模式）
		"""
		if self.mode == "subprocess":
			self._start_subprocess()
		elif self.mode == "tcp":
			self._connect_tcp()
		else:
			raise ValueError(f"Unknown mode: {self.mode}")

	def _start_subprocess(self):
		"""以子进程方式启动服务器"""
		if self.process is not None and self.process.poll() is None:
			return
		self.process = subprocess.Popen(
			[sys.executable, "-u", self.server_script],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			bufsize=1,
			encoding="utf-8",
		)
		self.ping()

	def _connect_tcp(self):
		"""连接到 TCP 服务器"""
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.connect((self.host, self.port))
		self.ping()

	def close(self):
		"""关闭客户端和服务器连接"""
		if self.mode == "subprocess":
			if self.process and self.process.poll() is None:
				self.process.terminate()
				self.process.wait(timeout=3)
			self.process = None
		elif self.mode == "tcp":
			if self.socket:
				self.socket.close()
				self.socket = None

	def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
		"""
		处理请求
		:param method: 方法名
		:param params: 参数
		:return:
		"""
		payload = {
			"id": str(uuid.uuid4()),
			"method": method,
			"params": params or {},
		}
		payload_str = json.dumps(payload, ensure_ascii=False) + "\n"

		if self.mode == "subprocess":
			return self._request_subprocess(payload_str)
		elif self.mode == "tcp":
			return self._request_tcp(payload_str)
		else:
			raise ValueError(f"Unknown mode: {self.mode}")

	def _request_subprocess(self, payload_str: str) -> Any:
		"""通过子进程 STDIO 发送请求"""
		if self.process is None or self.process.poll() is not None:
			raise RuntimeError("MCP server process is not running")
		if self.process.stdin is None or self.process.stdout is None:
			raise RuntimeError("MCP server stdio is unavailable")
		
		self.process.stdin.write(payload_str)
		self.process.stdin.flush()
		line = self.process.stdout.readline()
		if not line:
			stderr = self.process.stderr.read() if self.process.stderr else ""
			raise RuntimeError(f"No response from MCP server. stderr: {stderr}")
		response = json.loads(line)
		if response.get("error"):
			raise RuntimeError(response["error"].get("message", "Unknown MCP error"))
		return response.get("result")

	def _request_tcp(self, payload_str: str) -> Any:
		"""通过 TCP Socket 发送请求"""
		if self.socket is None:
			raise RuntimeError("MCP server is not connected")
		
		self.socket.sendall(payload_str.encode("utf-8"))
		response_data = b""
		while b"\n" not in response_data:
			chunk = self.socket.recv(4096)
			if not chunk:
				break
			response_data += chunk
		
		response = json.loads(response_data.decode("utf-8"))
		if response.get("error"):
			raise RuntimeError(response["error"].get("message", "Unknown MCP error"))
		return response.get("result")

	def ping(self):
		"""
		测试和server连接性
		"""
		return self._request("ping")

	def list_tools(self) -> list[dict[str, Any]]:
		"""
		查询工具列表
		"""
		result = self._request("list_tools")
		return result if isinstance(result, list) else []

	def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
		"""
		调用工具
		:param name:
		:param arguments:
		"""
		return self._request("call_tool", {"name": name, "arguments": arguments})

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="MCP Client")
	parser.add_argument("--mode", choices=["subprocess", "tcp"], default="tcp",
						help="Communication mode")
	parser.add_argument("--host", default="127.0.0.1", help="TCP server host")
	parser.add_argument("--port", type=int, default=7777, help="TCP server port")
	args = parser.parse_args()

	client = MCPClient(mode=args.mode, host=args.host, port=args.port)
	client.start()
	tools = client.list_tools()
	print("mcp tool list:", tools)
	print("client ping:", client.ping())

	print("\n=== 测试查询天气 ===")
	weather_result = client.call_tool("query_weather", {"city": "上海", "days": 5})
	print("上海天气:", weather_result)

	print("\n=== 测试查询机票 ===")
	flight_result = client.call_tool("query_flight_tickets", {
		"from_city": "北京",
		"to_city": "上海",
		"direct": False,
	})
	print("北京到上海机票:", flight_result)

	client.close()
