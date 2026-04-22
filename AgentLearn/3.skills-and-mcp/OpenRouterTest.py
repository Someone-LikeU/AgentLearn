from openai import OpenAI
from openrouter import OpenRouter
import os

openrouter_KEY = "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"
# openAI格式
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key=openrouter_KEY
)

def openAI_format():
	completion = client.chat.completions.create(
		# extra_headers={
		# 	"HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
		# 	"X-OpenRouter-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
		# },
		# extra_body={},
		model="inclusionai/ling-2.6-flash:free",
		messages=[
			{
				"role": "user",
				"content": "hello!"
			}
		]
	)
	print(completion.choices[0].message.content)

def openRouter_format():
	with OpenRouter(
			api_key=openrouter_KEY
	) as client:
		response = client.chat.send(
			model="inclusionai/ling-2.6-flash:free",
			messages=[
				{
					"role": "user",
					"content": "What is the meaning of life?"
				}
			]
		)

		print(response.choices[0].message.content)


if __name__ == '__main__':
	openRouter_format()


