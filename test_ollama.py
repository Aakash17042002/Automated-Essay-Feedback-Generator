from ollama import Client
import os

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"
    }
)

response = client.chat(
    model="gpt-oss:20b",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ]
)

print(response["message"]["content"])