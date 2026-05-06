# encoding: utf-8
# @Time    : 2026/04/24 00:00
"""
统一管理 MCP server、MCP client 与 Agent 生命周期。

设计目标：
1. client 不再负责启动 server，降低耦合；
2. server 可独立运行（`python mcp_server.py --mode server`）；
3. 需要一体化运行时，由本类集中创建与释放资源。
"""
import os

from agent_skill_mcp import Agent
from mcp_client import MCPClient
from mcp_server import MCPServerProcess


class AgentRuntimeManager:
	"""集中编排 server/client/agent 的运行时管理器。"""

	def __init__(self, model: str = "qwen3.5:9b", temperature: float = 0.0, base_url: str | None = None, api_key: str | None = None):
		self.model = model
		self.temperature = temperature
		self.base_url = base_url
		self.api_key = api_key

		self.server_process = MCPServerProcess()
		self.mcp_client: MCPClient | None = None
		self.agent: Agent | None = None

	def start(self):
		"""启动server，创建client，并注入agent。"""
		process = self.server_process.start()
		self.mcp_client = MCPClient.from_process(process)
		# 启动后先做一次探活，确保agent初始化时可以拿到工具列表。
		self.mcp_client.ping()
		self.agent = Agent(
			model=self.model,
			temperature=self.temperature,
			base_url=self.base_url,
			api_key=self.api_key,
			mcp_client=self.mcp_client,
		)
		return self.agent

	def stop(self):
		"""停止运行时资源。"""
		self.server_process.stop()
		self.mcp_client = None
		self.agent = None


if __name__ == "__main__":
	manager = AgentRuntimeManager(
		model=os.environ.get("OPENAI_MODEL", "qwen3.5:9b"),
		base_url=os.environ.get("OPENAI_BASE_URL"),
		api_key=os.environ.get("OPENAI_API_KEY"),
	)
	try:
		agent = manager.start()
		print(agent.agent_run("查询北京未来3天天气"))
	finally:
		manager.stop()
