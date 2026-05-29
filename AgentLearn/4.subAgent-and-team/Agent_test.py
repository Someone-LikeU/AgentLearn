from agent_Teams import Agent
import os

from mcp_client import create_tcp_mcp_client

if __name__ == '__main__':
	"""
	美团龙猫LongCat-Flash-Chat
	LONGCAT_API_KEY
	LONGCAT_BASE_URL
	
	智普  glm-4.5-air
	GLM_4_5_AIR_API_KEY
	GLM_4_5_AIR_BASE_URL
	"""
	
	API_KEY = os.environ.get("GLM_4_5_AIR_API_KEY", "NO")
	BASE_URL = os.environ.get("GLM_4_5_AIR_BASE_URL", "NO")
	MODEL = "glm-4.5-air"
	mcp_client = None
	server_process = None
	try:
		mcp_client, server_process = create_tcp_mcp_client()
		myAgent = Agent(
			model=MODEL,
			base_url=BASE_URL,
			api_key=API_KEY,
			mcp_client=mcp_client,
		)
		# use_plan = "--plan" in sys.argv
		# if len(sys.argv) < 2:
		# 	print("Usage: python agent_memory.py [--plan] 'your task here'")
		# 	print("  --plan: Enable task planning and decomposition")
		# 	sys.exit(1)
		# task = " ".join(sys.argv[1:])
		task = "找到当前目录的上一级目录下的所有子目录的python文件中的TODO项并整理到上一级目录的'开发过程所有todo.md'文件中，要包含TODO所在代码文件的位置、TODO内容，并给每个TODO项预留一行由我来记录该TODO的结果，如果该文件已存在，就先删除它"
		# myAgent.chat(task)
		myAgent.run()
	finally:
		if mcp_client:
			mcp_client.close()
		if server_process and server_process.poll() is None:
			# 只关闭本入口自动启动的 MCP server，不影响用户手动启动的外部服务。
			server_process.terminate()
			server_process.wait(timeout=3)
