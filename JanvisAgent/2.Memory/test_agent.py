from agent_memory import Agent
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
	use_plan = "--plan" in sys.argv
	if len(sys.argv) < 2:
		print("Usage: python agent_memory.py [--plan] 'your task here'")
		print("  --plan: Enable task planning and decomposition")
		sys.exit(1)
	task = " ".join(sys.argv[1:])
	myAgent.agent_run(task, use_plan=use_plan)


你好！很高兴见到你 😊  
有什么我可以帮忙的吗？或者你想聊点什么？
response:
 ChatCompletion(id='7911bad70c5c46759813eec1bbd46aff', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='你好！很高兴见到你 😊  \n有什么我可以帮忙的吗？或者你想聊点什么？', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None), matched_stop=2)], created=1776275361, model='longcat-flash-chatai-api', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=19, prompt_tokens=13, total_tokens=32, completion_tokens_details=None, prompt_tokens_details=None, cache_write_tokens=0, cache_read_tokens=0, input_tokens=0, output_tokens=0, cached_tokens=0))
