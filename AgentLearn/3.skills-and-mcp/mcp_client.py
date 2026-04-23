# encoding: utf-8
# @Time    : 2026/04/22
import json
import os
import subprocess
import sys
import uuid
from typing import Any


class MCPClient:
	"""
	MCP 客户端实现，使用进程间通信的方式和服务端交互，服务端封装以客户端的子进程方式运行
	"""
	def __init__(self, server_script: str | None = None):
		base_dir = os.path.dirname(os.path.abspath(__file__))
		# 默认MCP服务器脚本路径
		self.server_script = server_script or os.path.join(base_dir, "mcp_server.py")
		self.process: subprocess.Popen[str] | None = None

	def start(self):
		"""
		启动客户端和服务器
		:return:
		"""
		if self.process is not None and self.process.poll() is None:
			return
		self.process = subprocess.Popen(
			# "-u" 确保不缓冲输出，实时输出
			[sys.executable, "-u", self.server_script],
			# 用一个管道建立和子进程的标准io通信
			stdin=subprocess.PIPE, # 向子进程的标准输入写入
			stdout=subprocess.PIPE,	# 从子进程的标准输出读取
			stderr=subprocess.PIPE,	# 从子进程的标准错误读取
			text=True,	# 以文本模式进行io
			bufsize=1,	# 行缓冲大小
			encoding="utf-8",
		)
		# 调用约定的ping方法检测连接状态
		self.ping()

	def close(self):
		# 关闭客户端和服务器连接以及子进程
		if self.process is None:
			return
		if self.process.poll() is None:
			self.process.terminate()
			self.process.wait(timeout=3)
		self.process = None

	def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
		"""
		处理请求
		:param method: 方法名
		:param params: 参数
		:return:
		"""
		if self.process is None or self.process.poll() is not None:
			raise RuntimeError("MCP server process is not running")
		if self.process.stdin is None or self.process.stdout is None:
			raise RuntimeError("MCP server stdio is unavailable")
		payload = {
			"id": str(uuid.uuid4()),
			"method": method,
			"params": params or {},
		}
		# 向子进程的stdin写入请求数据
		self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
		self.process.stdin.flush()
		# 从子进程的stdout读取响应数据
		line = self.process.stdout.readline()
		if not line:
			stderr = self.process.stderr.read() if self.process.stderr else ""
			raise RuntimeError(f"No response from MCP server. stderr: {stderr}")
		response = json.loads(line)
		if response.get("error"):
			raise RuntimeError(response["error"].get("message", "Unknown MCP error"))
		return response.get("result")

	def ping(self):
		"""
		测试和server连接性
		:return:
		"""
		return self._request("ping")

	def list_tools(self) -> list[dict[str, Any]]:
		"""
		查询工具列表
		:return:
		"""
		result = self._request("list_tools")
		return result if isinstance(result, list) else []

	def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
		"""
		调用工具
		:param name:
		:param arguments:
		:return:
		"""
		return self._request("call_tool", {"name": name, "arguments": arguments})

	def __del__(self):
		# 兜底，对象回收时保证客户端被关闭
		try:
			self.close()
		except Exception:
			pass


if __name__ == '__main__':
	client = MCPClient()
	client.start()
	tools = client.list_tools()
	print("mcp tool list: ", tools)
	print("client ping: ", client.ping())

	# 测试查询天气
	print("\n=== 测试查询天气 ===")
	weather_result = client.call_tool("query_weather", {"city": "上海", "days": 5})
	print("上海天气:", weather_result)

	# 测试查询机票
	print("\n=== 测试查询机票 ===")
	flight_result = client.call_tool("query_flight_tickets", {
		"from_city": "北京",
		"to_city": "上海",
		"direct": False,
	})
	print("北京到上海机票:", flight_result)

	client.close()
