# encoding : utf-8
# @File    : OllamaTest.py
import httpx
from openai import OpenAI
import os

# token自由，运行时记得终端里先运行ollama serve
client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
    # required but ignored
    api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    http_client=httpx.Client(trust_env=False),
)

def chat(task: str, use_model: str):
    chat_completion = client.chat.completions.create(
        messages = [
            {
                'role'   : 'user',
                'content': task,
            }
        ],
        model = use_model,
    )
    print(chat_completion.choices[0].message.content)


if __name__ == '__main__':
    try:
        # chat_completion = client.chat.completions.create(
        #     messages=[
        #         {
        #             'role': 'user',
        #             'content': '你是谁？',
        #         }
        #     ],
        #     model=,
        # )
        # print(chat_completion.choices[0].message.content)
        #
        chat("你是谁？", 'qwen3.5:9b')
        # chat("你是谁？", 'deepseek-r1:14b')
        # chat("你是谁？", 'gemma4:e4b')
        
    except Exception as error:
        print(error)
    finally:
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()