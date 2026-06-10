# encoding : utf-8
# @Time    : 2026/6/9 16:27
from openai import OpenAI

client = OpenAI(
    api_key="ak_2tE45i3sY4D53Y36AS8SU1DS3mW44",
    base_url="https://api.longcat.chat/openai"
)

if __name__ == '__main__':
	response = client.chat.completions.create(
	    model="LongCat-2.0-Preview",
	    messages=[
	        {"role": "user", "content": "Hello!"}
	    ],
	    max_tokens=1000
	)
	
	print(response.choices[0].message.content)
