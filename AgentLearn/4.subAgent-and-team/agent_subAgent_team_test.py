from agent_subAgent_team import Agent
import sys


if __name__ == '__main__':
	# 美团龙猫模型
	API_KEY = "ak_2Nu3Zp7IO0fa5M01Aa3xq6F66uh0k"
	BASE_URL = "https://api.longcat.chat/openai"
	MODEL = "LongCat-Flash-Chat"
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
	task = "找到当前目录下的所有TODO并整理到上一级目录的'开发过程问题记录.md'文件中，要包含TODO所在代码文件的位置、TODO内容，并给每个TODO项预留一行由我来记录该TODO的结果，如果该文件已存在，就在后面追加"
	myAgent.agent_run(task)
