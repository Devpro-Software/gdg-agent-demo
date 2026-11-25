"""Basic chat loop. No tools."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"


def chat_loop():
    """Chat loop"""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    messages = [{
        "role": "system",
        "content": "Your name is Jimmy. You are a helpful assistant. Only output text, no special characters or parentheses.",
    }]

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        if len(resp.choices) == 0:
            print(f"No response from model. Error: {resp.error}")
            continue

        content = resp.choices[0].message.content or ""
        print("Bot:", content)
        messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    chat_loop()
