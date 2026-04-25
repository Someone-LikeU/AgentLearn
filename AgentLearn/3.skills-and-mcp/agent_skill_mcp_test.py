from agent_skill_mcp import Agent
import sys


if __name__ == '__main__':
	# 美团龙猫模型
	API_KEY = "ak_2Nu3Zp7IO0fa5M01Aa3xq6F66uh0k"
	BASE_URL = "https://api.longcat.chat/openai"
	MODEL = "LongCat-Flash-Lite"
	myAgent = Agent(
		model=MODEL,
		base_url=BASE_URL,
		api_key=API_KEY
	)
	# use_plan = "--plan" in sys.argv
	# if len(sys.argv) < 2:
	# 	print("Usage: python agent_memory.py [--plan] 'your task here'")
	# 	print("  --plan: Enable task planning and decomposition")
	# 	sys.exit(1)
	# task = " ".join(sys.argv[1:])
	task = "找到当前目录下所有TODO并整理到TODO.md文件中，如果TODO.md文件已存在，就先删除它"
	myAgent.agent_run(task)
