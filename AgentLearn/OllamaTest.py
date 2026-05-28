# encoding : utf-8
# @File    : OllamaTest.py
import httpx
from openai import OpenAI

# token自由，运行时记得终端里先运行ollama serve
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    # required but ignored
    api_key='ollama',
    http_client=httpx.Client(trust_env=False),
)

if __name__ == '__main__':
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    'role': 'user',
                    'content': 'Say "this is a test", do not do anything else.',
                }
            ],
            model='deepseek-r1:14b',
        )
        print(chat_completion)
    except Exception as error:
        print(error)
    finally:
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()