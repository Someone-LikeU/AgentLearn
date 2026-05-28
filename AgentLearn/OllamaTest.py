# encoding : utf-8
# @File    : OllamaTest.py
import httpx
from openai import OpenAI

# token自由
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    # required but ignored
    api_key='ollama',
    http_client=httpx.Client(trust_env=False),
)

if __name__ == '__main__':
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': 'who are you?',
            }
        ],
        model='qwen3.5:9b',
    )
    print(chat_completion.choices[0].message.content)