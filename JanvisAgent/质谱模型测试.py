# encoding : utf-8
# @File    : 质谱模型测试.py


from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("ZHIPU_API_KEY", "NO"),
    base_url=os.environ.get("ZHIPU_OPENAI_URL", "NO")
)

if __name__ == '__main__':
	completion = client.chat.completions.create(
	    model="glm-4.5-air",
	    messages=[
	        {"role": "user", "content": "Say 'hello', don't do anything else."}
	    ],
	    top_p=0.7,
	    temperature=0.1
	)
	
	print(completion)