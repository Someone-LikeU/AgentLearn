# encoding : utf-8
# @Time    : 2026/4/16 00:11
# @File    : LongCatTest.py
from openai import OpenAI, AsyncOpenAI
import asyncio

client = OpenAI(
	api_key = "ak_2UR4m62oO3wH6k27l50Bv3zU54045",
	base_url = "https://api.longcat.chat/openai"
)

async_client = AsyncOpenAI(
	api_key = "ak_2UR4m62oO3wH6k27l50Bv3zU54045",
	base_url = "https://api.longcat.chat/openai"
)

async def async_call():
	stream = await async_client.chat.completions.create(
		model = "LongCat-Flash-Chat",
		messages = [
			{"role": "user", "content": "介绍一下python和Java的区别"}
		],
		stream = True
	)
	
	async for chunk in stream:
		content = chunk.choices[0].delta.content
		if content:
			print(content, end = "", flush = True)

if __name__ == '__main__':
	response = client.chat.completions.create(
	    model="LongCat-Flash-Chat",
	    messages=[
	        {"role": "user", "content": "你好!"}
	    ],
	    max_tokens=10000
	)

	# print(response.choices[0].message.content)
	print("response:\n")
	print(response)
	
	# 流式处理响应，同步方式
	stream = client.chat.completions.create(
		model = "LongCat-Flash-Chat",
		messages = [
			{"role": "user", "content": "输出一篇1000字的记叙文，主题是小美和小帅的恋爱故事"}
		],
		stream = True
	)
	all_content = ""
	for chunk in stream:
		content = chunk.choices[0].delta.content
		print('chunk object:', chunk)
		if content:
			all_content += content
			print(content, end = "", flush = True)
	
	print("all content: ", all_content)

	# 异步方式
	# TODO 将agent的调用方式改为流式，学习一下主流agent的调用处理方式
	print("异步调用方式：")
	asyncio.run(async_call())