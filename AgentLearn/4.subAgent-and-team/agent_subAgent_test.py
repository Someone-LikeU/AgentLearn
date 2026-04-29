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
	task = "找到当前目录的上一级目录下的所有子目录的python文件中的TODO项并整理到上一级目录的'开发过程所有todo.md'文件中，要包含TODO所在代码文件的位置、TODO内容，并给每个TODO项预留一行由我来记录该TODO的结果，如果该文件已存在，就先删除它"
	# myAgent.chat(task)
	myAgent.run()
	
	
	# TODO 4.28晚调试记录:天气和机票工具不可用，应该是进程没就绪的问题。2.网络搜索工具不可用，看响应的工具调用找到了这个工具，但是接着就直接失败了，或许还要优化一下提示词。
	# TODO 此外，要用mcp 客户端的tcp模式的话，还要加一段代码先启动tcp server
