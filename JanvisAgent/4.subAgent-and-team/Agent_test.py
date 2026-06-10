import os
import sys


def configure_stdio_encoding():
	# 统一当前进程和子 Python 进程的文本编码，避免 UTF-8 中文被 Windows 代码页误解码。
	os.environ.setdefault("PYTHONUTF8", "1")
	os.environ.setdefault("PYTHONIOENCODING", "utf-8")
	for stream in (sys.stdin, sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio_encoding()

from agent_Teams import Agent

from mcp_client import create_tcp_mcp_client

if __name__ == '__main__':
	"""
	美团龙猫LongCat-2.0-Preview
	LONGCAT_2_0_PREVIEW_API_KEY
	LONGCAT_2_0_PREVIEW_BASE_URL
	
	智普  glm-4.5-air
	GLM_4_5_AIR_API_KEY
	GLM_4_5_AIR_BASE_URL
	"""
	
	API_KEY = os.environ.get("LONGCAT_2_0_PREVIEW_API_KEY", "NO")
	BASE_URL = os.environ.get("LONGCAT_2_0_PREVIEW_BASE_URL", "NO")
	MODEL = "LongCat-2.0-Preview"
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
		myAgent.run()
	finally:
		if mcp_client:
			mcp_client.close()
		if server_process and server_process.poll() is None:
			# 只关闭本入口自动启动的 MCP server，不影响用户手动启动的外部服务。
			server_process.terminate()
			server_process.wait(timeout=3)
