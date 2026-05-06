# encoding: utf-8
# @Time    : 2026/04/24 00:00
import os

from agent_runtime_manager import AgentRuntimeManager


if __name__ == '__main__':
	manager = AgentRuntimeManager(
		model=os.environ.get("OPENAI_MODEL", "LongCat-Flash-Lite"),
		base_url=os.environ.get("OPENAI_BASE_URL"),
		api_key=os.environ.get("OPENAI_API_KEY"),
	)
	try:
		agent = manager.start()
		task = "找到当前目录下所有TODO并整理到TODO.md文件中，如果TODO.md文件已存在，就先删除它"
		print(agent.agent_run(task))
	finally:
		manager.stop()
