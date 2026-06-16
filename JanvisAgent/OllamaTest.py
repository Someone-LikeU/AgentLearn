# encoding : utf-8
# @File    : OllamaTest.py
import base64
import mimetypes
import httpx
import os
from openai import OpenAI
from time import perf_counter

# token自由，运行时记得终端里先运行ollama serve
client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
    # required but ignored
    api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    http_client=httpx.Client(trust_env=False),
)


def get_ollama_native_base_url():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url.rstrip("/")


def unload_ollama_model(use_model: str):
    # 调用 Ollama 原生接口，让本次使用的模型立即从内存和显存中卸载。
    with httpx.Client(trust_env=False, timeout=60) as unload_client:
        response = unload_client.post(
            f"{get_ollama_native_base_url()}/api/chat",
            json={
                "model": use_model,
                "messages": [],
                "keep_alive": 0,
            },
        )
        response.raise_for_status()


def chat(task: str, use_model: str, temperature=1.0, top_p=0.95):
    chat_completion = client.chat.completions.create(
        messages = [
            {
                'role'   : 'user',
                'content': task,
            }
        ],
        model = use_model,
        temperature = temperature,
        top_p = top_p,
    )
    print(chat_completion.choices[0].message.content)


def vision_chat(task: str, image_url: str, use_model: str, temperature=1.0, top_p=0.95):
    mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"
    with open(image_url, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    start = perf_counter()
    try:
        chat_completion = client.chat.completions.create(
            model = use_model,
            messages = [
                {
                    "role"   : "user",
                    "content": [
                        {
                            "type": "text",
                            "text": task,
                        },
                        {
                            "type"     : "image_url",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                        },
                    ],
                }
            ],
            temperature = temperature,
            top_p = top_p,
        )
        cost = perf_counter() - start
        print(chat_completion.choices[0].message.content)
        print(f"time cost {cost}")
    finally:
        try:
            unload_ollama_model(use_model)
        except Exception as error:
            print(f"unload model failed: {error}")


if __name__ == '__main__':
    try:
        # chat("你是谁？", 'qwen3.5:9b')
        # chat("你是谁？", 'deepseek-r1:14b')
        image_path = r"D:\OllamaTestInput\testImage3.jpg"
        vision_task = "这张图片描述了什么？"
        vision_chat(vision_task, image_path, 'gemma4:e4b')
        
    except Exception as error:
        print(error)
    finally:
        close_client = getattr(client, "close", None)
        if callable(close_client):
            close_client()
